"""Binance REST client for historical klines (candles).

Scope is deliberately narrow: this fetches public historical candles and
nothing else. No authentication, no trading, no websockets.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from types import TracebackType
from typing import Any

import httpx
import pandas as pd

from backtool.config import BINANCE_MAX_KLINES_PER_REQUEST, Settings
from backtool.core.time import UTC, ensure_utc, from_utc_ms, utc_ms
from backtool.core.types import Interval
from backtool.data.candles import normalise_candles

logger = logging.getLogger(__name__)

# Positions of the fields we use within Binance's kline array response.
_OPEN_TIME = 0
_OPEN = 1
_HIGH = 2
_LOW = 3
_CLOSE = 4
_VOLUME = 5

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_FRAME_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


class BinanceError(RuntimeError):
    """The exchange could not be reached, or returned an unusable response."""


class BinanceGeoBlockedError(BinanceError):
    """The endpoint is unavailable from this jurisdiction (HTTP 451)."""


class BinanceClient:
    """Fetches historical candles from Binance's public REST API.

    Usable as a context manager so the HTTP connection pool closes
    deterministically.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or Settings.from_env()
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=self._settings.binance_base_url,
            timeout=self._settings.request_timeout_s,
            headers={"User-Agent": "backtool/0.1"},
        )

    def __enter__(self) -> BinanceClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        *,
        now: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch every closed candle whose open_time falls in ``[start, end)``.

        Paginates transparently: Binance caps a response at 1000 candles, so a
        one-day 5m range is a single request while a multi-year backfill is
        many.

        **Unclosed candles are dropped.** When ``end`` reaches into the present,
        Binance returns the in-progress candle, whose high, low, close and
        volume are still changing. Caching it would persist a price that is
        about to be wrong, and every metric derived from it would drift between
        runs with no visible cause.

        Args:
            symbol: Exchange symbol, e.g. ``"BTCUSDT"``.
            interval: Candle interval.
            start: Inclusive lower bound on ``open_time``. Timezone-aware.
            end: Exclusive upper bound on ``open_time``. Timezone-aware.
            now: Override for "the current instant", for deterministic tests.

        Returns:
            A normalised candle frame. Empty if the range predates the symbol's
            first trade, which is a legitimate outcome rather than an error.

        Raises:
            ValueError: if the range is empty or the bounds are naive.
            BinanceGeoBlockedError: the endpoint is blocked in this region.
            BinanceError: the request failed after exhausting retries, or the
                response was not in the expected shape.
        """
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if start_utc >= end_utc:
            raise ValueError(f"start {start_utc} must be before end {end_utc}")

        current = ensure_utc(now) if now is not None else dt.datetime.now(tz=UTC)
        rows: list[list[Any]] = []
        cursor_ms = utc_ms(start_utc)
        end_ms = utc_ms(end_utc)

        while cursor_ms < end_ms:
            batch = self._request_klines(symbol, interval, cursor_ms, end_ms)
            if not batch:
                break

            rows.extend(batch)
            next_cursor = int(batch[-1][_OPEN_TIME]) + interval.milliseconds
            if next_cursor <= cursor_ms:  # defensive: never loop forever
                break
            cursor_ms = next_cursor

            if len(batch) < BINANCE_MAX_KLINES_PER_REQUEST:
                break

        logger.info(
            "Fetched %d raw kline(s) for %s %s over %s .. %s",
            len(rows),
            symbol,
            interval.value,
            start_utc,
            end_utc,
        )
        return self._to_frame(rows, interval, end_utc, current)

    def _request_klines(
        self, symbol: str, interval: Interval, start_ms: int, end_ms: int
    ) -> list[list[Any]]:
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval.value,
            "startTime": start_ms,
            "endTime": end_ms - 1,  # Binance treats endTime as inclusive
            "limit": BINANCE_MAX_KLINES_PER_REQUEST,
        }

        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries):
            try:
                response = self._http.get("/api/v3/klines", params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("Request failed (%s), attempt %d", exc, attempt + 1)
                self._sleep_before_retry(attempt, None)
                continue

            if response.status_code == 451:
                raise BinanceGeoBlockedError(
                    f"{self._settings.binance_base_url} returned HTTP 451 "
                    "(unavailable for legal reasons) from this location. Set "
                    "BINANCE_BASE_URL to a reachable mirror such as "
                    "https://data-api.binance.vision"
                )
            if response.status_code == 418:
                raise BinanceError(
                    "Binance returned HTTP 418: this IP is temporarily banned for "
                    "exceeding rate limits. Wait before retrying."
                )
            if response.status_code in _RETRYABLE_STATUS:
                last_error = BinanceError(f"HTTP {response.status_code}")
                logger.warning(
                    "Binance returned HTTP %s (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    self._settings.max_retries,
                )
                self._sleep_before_retry(attempt, response.headers.get("Retry-After"))
                continue
            if response.status_code != 200:
                raise BinanceError(
                    f"Binance returned HTTP {response.status_code}: {response.text[:200]}"
                )

            payload = response.json()
            if not isinstance(payload, list):
                raise BinanceError(f"Expected a list of klines, got {type(payload).__name__}")
            return payload

        raise BinanceError(
            f"Failed to fetch klines for {symbol} {interval.value} after "
            f"{self._settings.max_retries} attempts: {last_error}"
        )

    def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        if retry_after is not None:
            try:
                time.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass
        time.sleep(min(2.0**attempt, 30.0))

    @staticmethod
    def _to_frame(
        rows: list[list[Any]],
        interval: Interval,
        end: dt.datetime,
        now: dt.datetime,
    ) -> pd.DataFrame:
        if not rows:
            empty = pd.DataFrame({column: pd.Series(dtype="object") for column in _FRAME_COLUMNS})
            empty["open_time"] = pd.Series(dtype="datetime64[ns, UTC]")
            return normalise_candles(empty, interval)

        frame = pd.DataFrame(
            {
                "open_time": [from_utc_ms(int(row[_OPEN_TIME])) for row in rows],
                "open": [row[_OPEN] for row in rows],
                "high": [row[_HIGH] for row in rows],
                "low": [row[_LOW] for row in rows],
                "close": [row[_CLOSE] for row in rows],
                "volume": [row[_VOLUME] for row in rows],
            }
        )
        # Pagination can overlap at range boundaries; normalise_candles rejects
        # duplicates, so drop them here rather than surfacing an exchange quirk
        # as a data-integrity failure.
        frame = frame.drop_duplicates(subset="open_time")

        closes_at = frame["open_time"] + interval.duration
        keep = (closes_at <= pd.Timestamp(now)) & (frame["open_time"] < pd.Timestamp(end))
        dropped = int((~keep).sum())
        if dropped:
            logger.debug("Dropped %d unclosed or out-of-range candle(s)", dropped)

        return normalise_candles(frame.loc[keep], interval)
