"""Tests for the Binance klines client, against a mocked transport.

No network is touched here. A separate, marked integration test exercises the
real endpoint.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pytest

from backtool.config import Settings
from backtool.core.time import UTC, utc_ms
from backtool.core.types import Interval
from backtool.data.binance.client import (
    BinanceClient,
    BinanceError,
    BinanceGeoBlockedError,
)

BASE = dt.datetime(2024, 1, 31, 18, 0, tzinfo=UTC)


def kline(open_time: dt.datetime, close: float, interval: Interval = Interval.M5) -> list[Any]:
    """One kline in Binance's array shape, with the fields we read populated."""
    open_ms = utc_ms(open_time)
    price = f"{close:.8f}"
    return [
        open_ms,
        price,
        f"{close * 1.01:.8f}",
        f"{close * 0.99:.8f}",
        price,
        "12.34500000",
        open_ms + interval.milliseconds - 1,
        "500000.0",
        250,
        "6.0",
        "250000.0",
        "0",
    ]


def client_with(
    handler: object, *, max_retries: int = 4
) -> BinanceClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    settings = Settings(max_retries=max_retries, request_timeout_s=5.0)
    return BinanceClient(
        settings,
        http_client=httpx.Client(transport=transport, base_url="https://mock.invalid"),
    )


def static_response(payload: Any, status: int = 200) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


class TestFetching:
    def test_returns_normalised_candles(self) -> None:
        rows = [kline(BASE + i * Interval.M5.duration, 100.0 + i) for i in range(12)]
        with client_with(static_response(rows)) as client:
            frame = client.fetch_klines(
                "BTCUSDT",
                Interval.M5,
                BASE,
                BASE + dt.timedelta(hours=1),
                now=BASE + dt.timedelta(days=1),
            )

        assert len(frame) == 12
        assert list(frame.columns) == [
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert frame["close"].iloc[0] == 100.0
        assert str(frame["open_time"].dtype) == "datetime64[ns, UTC]"

    def test_empty_response_yields_a_typed_empty_frame(self) -> None:
        """Requesting a range before the symbol was listed is legitimate."""
        with client_with(static_response([])) as client:
            frame = client.fetch_klines(
                "BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1)
            )

        assert frame.empty
        assert "close_time" in frame.columns

    def test_sends_the_expected_query_parameters(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(request.url.params))
            return httpx.Response(200, json=[])

        with client_with(handler) as client:
            client.fetch_klines("BTCUSDT", Interval.M15, BASE, BASE + dt.timedelta(hours=1))

        assert seen["symbol"] == "BTCUSDT"
        assert seen["interval"] == "15m"
        assert seen["startTime"] == str(utc_ms(BASE))
        # endTime is exclusive for us, inclusive for Binance.
        assert seen["endTime"] == str(utc_ms(BASE + dt.timedelta(hours=1)) - 1)

    def test_rejects_inverted_range(self) -> None:
        with client_with(static_response([])) as client, pytest.raises(ValueError, match="before"):
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE - dt.timedelta(hours=1))

    def test_rejects_naive_bounds(self) -> None:
        with (
            client_with(static_response([])) as client,
            pytest.raises(ValueError, match="Naive datetime"),
        ):
            client.fetch_klines("BTCUSDT", Interval.M5, dt.datetime(2024, 1, 31), BASE)


class TestUnclosedCandles:
    def test_drops_the_in_progress_candle(self) -> None:
        """Binance returns the current, still-forming candle. Its high, low,
        close and volume are all still changing, so caching it would persist a
        price that is about to be wrong."""
        rows = [kline(BASE + i * Interval.M5.duration, 100.0 + i) for i in range(4)]
        # "Now" falls inside the 4th candle [18:15, 18:20), so it has not closed.
        now = BASE + dt.timedelta(minutes=17)

        with client_with(static_response(rows)) as client:
            frame = client.fetch_klines(
                "BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1), now=now
            )

        assert len(frame) == 3
        assert frame["open_time"].iloc[-1] == dt.datetime(2024, 1, 31, 18, 10, tzinfo=UTC)

    def test_keeps_a_candle_that_closed_exactly_now(self) -> None:
        rows = [kline(BASE + i * Interval.M5.duration, 100.0 + i) for i in range(4)]
        now = BASE + dt.timedelta(minutes=20)  # the 4th candle closes precisely here

        with client_with(static_response(rows)) as client:
            frame = client.fetch_klines(
                "BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1), now=now
            )

        assert len(frame) == 4

    def test_drops_candles_beyond_the_requested_end(self) -> None:
        rows = [kline(BASE + i * Interval.M5.duration, 100.0 + i) for i in range(12)]
        with client_with(static_response(rows)) as client:
            frame = client.fetch_klines(
                "BTCUSDT",
                Interval.M5,
                BASE,
                BASE + dt.timedelta(minutes=30),
                now=BASE + dt.timedelta(days=1),
            )

        assert len(frame) == 6


class TestPagination:
    def test_follows_multiple_pages(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            start_ms = int(request.url.params["startTime"])
            calls.append(start_ms)
            first_open = dt.datetime.fromtimestamp(start_ms / 1000, tz=UTC)
            # Full page on the first call, short page on the second.
            count = 1000 if len(calls) == 1 else 5
            rows = [
                kline(first_open + i * Interval.M5.duration, 100.0 + i) for i in range(count)
            ]
            return httpx.Response(200, json=rows)

        with client_with(handler) as client:
            frame = client.fetch_klines(
                "BTCUSDT",
                Interval.M5,
                BASE,
                BASE + dt.timedelta(days=5),
                now=BASE + dt.timedelta(days=30),
            )

        assert len(calls) == 2
        assert calls[1] > calls[0], "cursor did not advance"
        assert len(frame) == 1005

    def test_stops_on_a_short_page(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            start_ms = int(request.url.params["startTime"])
            first_open = dt.datetime.fromtimestamp(start_ms / 1000, tz=UTC)
            return httpx.Response(
                200,
                json=[kline(first_open + i * Interval.M5.duration, 100.0) for i in range(3)],
            )

        with client_with(handler) as client:
            client.fetch_klines(
                "BTCUSDT",
                Interval.M5,
                BASE,
                BASE + dt.timedelta(days=5),
                now=BASE + dt.timedelta(days=30),
            )

        assert len(calls) == 1

    def test_deduplicates_overlapping_pages(self) -> None:
        """Pagination boundaries commonly repeat a candle; normalise_candles
        rejects duplicates, so the client must absorb the quirk."""
        rows = [kline(BASE + i * Interval.M5.duration, 100.0 + i) for i in range(6)]
        overlapping = rows + rows[-2:]

        with client_with(static_response(overlapping)) as client:
            frame = client.fetch_klines(
                "BTCUSDT",
                Interval.M5,
                BASE,
                BASE + dt.timedelta(hours=1),
                now=BASE + dt.timedelta(days=1),
            )

        assert len(frame) == 6


class TestErrorHandling:
    def test_geo_block_raises_a_specific_error(self) -> None:
        with (
            client_with(static_response({"msg": "blocked"}, status=451)) as client,
            pytest.raises(BinanceGeoBlockedError, match="data-api.binance.vision"),
        ):
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1))

    def test_ip_ban_raises_immediately(self) -> None:
        with (
            client_with(static_response({"msg": "banned"}, status=418)) as client,
            pytest.raises(BinanceError, match="418"),
        ):
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1))

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("backtool.data.binance.client.time.sleep", lambda _: None)
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(429, json={"msg": "slow down"})
            return httpx.Response(
                200, json=[kline(BASE + i * Interval.M5.duration, 100.0) for i in range(3)]
            )

        with client_with(handler) as client:
            frame = client.fetch_klines(
                "BTCUSDT",
                Interval.M5,
                BASE,
                BASE + dt.timedelta(hours=1),
                now=BASE + dt.timedelta(days=1),
            )

        assert len(attempts) == 3
        assert len(frame) == 3

    def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("backtool.data.binance.client.time.sleep", lambda _: None)
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(503, json={"msg": "unavailable"})

        with (
            client_with(handler, max_retries=3) as client,
            pytest.raises(BinanceError, match="after 3 attempts"),
        ):
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1))

        assert len(attempts) == 3

    def test_retries_on_transport_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("backtool.data.binance.client.time.sleep", lambda _: None)
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 2:
                raise httpx.ConnectTimeout("boom")
            return httpx.Response(200, json=[])

        with client_with(handler) as client:
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1))

        assert len(attempts) == 2

    def test_unexpected_status_raises(self) -> None:
        with (
            client_with(static_response({"msg": "nope"}, status=400)) as client,
            pytest.raises(BinanceError, match="HTTP 400"),
        ):
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1))

    def test_non_list_payload_raises(self) -> None:
        with (
            client_with(static_response({"unexpected": "shape"})) as client,
            pytest.raises(BinanceError, match="Expected a list"),
        ):
            client.fetch_klines("BTCUSDT", Interval.M5, BASE, BASE + dt.timedelta(hours=1))
