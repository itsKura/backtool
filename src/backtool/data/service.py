"""The cache-aware market data layer.

This is the only thing the research engine talks to when it needs prices. It
answers "give me candles for this range" by consulting local storage first,
fetching only what is genuinely missing, persisting the result, and returning
the union.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Protocol

import pandas as pd

from backtool.config import Settings
from backtool.core.time import UTC, ensure_utc, floor_to_interval
from backtool.core.types import Interval
from backtool.data.binance.client import BinanceClient
from backtool.data.candles import empty_candles
from backtool.data.storage.base import CandleRepository
from backtool.data.storage.sqlite import SQLiteCandleRepository

logger = logging.getLogger(__name__)

#: Candles fetched before the requested start so that anchoring exactly at the
#: start resolves. The anchor rule takes the last candle closed *at or before*
#: an instant, so a frame that begins at the instant itself has nothing to
#: anchor to. One candle is the minimum; a few gives slack around market gaps.
DEFAULT_WARMUP_CANDLES = 3


class KlineSource(Protocol):
    """Where candles come from, as the service needs them.

    Declared here rather than in the Binance package on purpose: the consumer
    owns the interface, so the service depends on a capability rather than on
    one exchange. Adding Coinbase or a bulk-file loader means implementing this,
    not editing the service.
    """

    def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        *,
        now: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Return closed candles with ``open_time`` in ``[start, end)``."""
        ...

    def close(self) -> None:
        """Release any underlying connections."""
        ...


class MarketDataService:
    """Serves candles from local storage, downloading only what is missing."""

    def __init__(
        self,
        client: KlineSource,
        repository: CandleRepository,
    ) -> None:
        self._client = client
        self._repository = repository

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MarketDataService:
        """Build the standard production wiring: Binance + SQLite on disk."""
        resolved = settings or Settings.from_env()
        return cls(
            client=BinanceClient(resolved),
            repository=SQLiteCandleRepository(resolved.candle_db),
        )

    def get_candles(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        *,
        warmup_candles: int = DEFAULT_WARMUP_CANDLES,
        now: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Return candles covering the request, fetching any uncached portion.

        The returned frame starts ``warmup_candles`` intervals *before*
        ``start``. That is deliberate: anchoring at exactly ``start`` needs a
        candle that already closed by then, so a frame beginning at ``start``
        would fail to price its own first boundary.

        Args:
            symbol: Exchange symbol, e.g. ``"BTCUSDT"``.
            interval: Candle interval.
            start: Inclusive lower bound of the range of interest.
            end: Exclusive upper bound.
            warmup_candles: Extra candles prepended before ``start``.
            now: Override for the current instant, for deterministic tests.

        Returns:
            A normalised candle frame spanning
            ``[start - warmup_candles * interval, end)``. May contain gaps
            where the market genuinely has none, and may be empty if the range
            predates the symbol's listing.
        """
        current = ensure_utc(now) if now is not None else dt.datetime.now(tz=UTC)
        fetch_start = ensure_utc(start) - warmup_candles * interval.duration
        fetch_end = ensure_utc(end)

        if fetch_start >= fetch_end:
            raise ValueError(f"start {start} must be before end {end}")

        self._ensure_cached(symbol, interval, fetch_start, fetch_end, current)
        return self._repository.read(symbol, interval, fetch_start, fetch_end)

    def _ensure_cached(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        now: dt.datetime,
    ) -> None:
        """Download and persist whatever part of ``[start, end)`` is uncached."""
        # Never claim coverage past the most recent closed candle. The in-flight
        # candle is excluded from what we store, so recording coverage up to
        # `now` would permanently skip it once it does close.
        horizon = floor_to_interval(now, interval.duration)
        gaps = self._repository.coverage(symbol, interval).missing(start, min(end, horizon))

        if not gaps:
            logger.info(
                "Cache hit: %s %s %s .. %s already fetched",
                symbol,
                interval.value,
                start,
                end,
            )
            return

        total = sum((gap.duration for gap in gaps), dt.timedelta())
        logger.info(
            "Cache miss: fetching %d range(s) totalling %s for %s %s",
            len(gaps),
            total,
            symbol,
            interval.value,
        )

        for gap in gaps:
            candles = self._client.fetch_klines(symbol, interval, gap.start, gap.end, now=now)
            if candles.empty:
                logger.info("No candles returned for %s (likely before listing)", gap)
            # Candles and coverage commit together, so a crash cannot leave the
            # range marked fetched with nothing behind it.
            self._repository.store(symbol, interval, candles, gap.start, gap.end)

    def cached_range(
        self, symbol: str, interval: Interval
    ) -> tuple[dt.datetime, dt.datetime] | None:
        """Overall span of fetched coverage, or ``None`` if nothing is cached."""
        ranges = self._repository.coverage(symbol, interval).ranges
        if not ranges:
            return None
        return ranges[0].start, ranges[-1].end

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MarketDataService:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def empty_result(interval: Interval) -> pd.DataFrame:
    """Convenience re-export for callers building a no-data response."""
    return empty_candles(interval)
