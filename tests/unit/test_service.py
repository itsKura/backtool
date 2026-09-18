"""Tests for the cache-aware market data service.

Uses a recording fake in place of the HTTP client, so these assert the caching
*policy* -- what gets requested and what does not -- rather than HTTP details.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import build_candles

from backtool.core.time import UTC
from backtool.core.types import Interval
from backtool.data.candles import empty_candles
from backtool.data.service import MarketDataService
from backtool.data.storage.sqlite import SQLiteCandleRepository

SYMBOL = "BTCUSDT"
DAY = dt.datetime(2024, 1, 31, tzinfo=UTC)
FAR_FUTURE = dt.datetime(2024, 3, 1, tzinfo=UTC)


class RecordingSource:
    """A KlineSource that synthesises candles and records every request."""

    def __init__(self, *, empty: bool = False) -> None:
        self.calls: list[tuple[dt.datetime, dt.datetime]] = []
        self._empty = empty

    def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        *,
        now: dt.datetime | None = None,
    ) -> pd.DataFrame:
        self.calls.append((start, end))
        if self._empty:
            return empty_candles(interval)

        count = int((end - start) / interval.duration)
        if count <= 0:
            return empty_candles(interval)
        return build_candles(start, [100.0] * count, interval)

    def close(self) -> None:
        pass

    @property
    def requested_span(self) -> dt.timedelta:
        return sum((end - start for start, end in self.calls), dt.timedelta())


@pytest.fixture
def repo(tmp_path: Path) -> SQLiteCandleRepository:
    return SQLiteCandleRepository(tmp_path / "candles.db")


class TestCaching:
    def test_cold_request_fetches_once(
        self, repo: SQLiteCandleRepository
    ) -> None:
        source = RecordingSource()
        service = MarketDataService(source, repo)

        frame = service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=2), now=FAR_FUTURE
        )

        assert len(source.calls) == 1
        assert not frame.empty

    def test_second_identical_request_hits_the_cache(
        self, repo: SQLiteCandleRepository
    ) -> None:
        source = RecordingSource()
        service = MarketDataService(source, repo)
        args = (SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=2))

        first = service.get_candles(*args, now=FAR_FUTURE)
        second = service.get_candles(*args, now=FAR_FUTURE)

        assert len(source.calls) == 1, "re-fetched data it already had"
        pd.testing.assert_frame_equal(first, second)

    def test_only_the_uncached_extension_is_fetched(
        self, repo: SQLiteCandleRepository
    ) -> None:
        """The core promise of the cache: widening a range downloads only the
        new part, not the whole thing again."""
        source = RecordingSource()
        service = MarketDataService(source, repo)

        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=2), now=FAR_FUTURE
        )
        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=4), now=FAR_FUTURE
        )

        assert len(source.calls) == 2
        second_start, second_end = source.calls[1]
        assert second_end - second_start == dt.timedelta(hours=2)
        assert second_start == DAY + dt.timedelta(hours=2)

    def test_interior_gap_is_fetched_in_pieces(
        self, repo: SQLiteCandleRepository
    ) -> None:
        source = RecordingSource()
        service = MarketDataService(source, repo)

        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=1), now=FAR_FUTURE
        )
        service.get_candles(
            SYMBOL,
            Interval.M5,
            DAY + dt.timedelta(hours=4),
            DAY + dt.timedelta(hours=5),
            now=FAR_FUTURE,
        )
        source.calls.clear()

        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=5), now=FAR_FUTURE
        )

        # Only the hole between the two cached spans should be requested.
        assert len(source.calls) == 1
        assert source.calls[0][0] >= DAY + dt.timedelta(minutes=55)

    def test_empty_response_is_not_re_requested(
        self, repo: SQLiteCandleRepository
    ) -> None:
        """A range before the symbol existed returns nothing. Without recording
        coverage for it, every future run would ask again forever."""
        source = RecordingSource(empty=True)
        service = MarketDataService(source, repo)
        args = (SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=2))

        service.get_candles(*args, now=FAR_FUTURE)
        service.get_candles(*args, now=FAR_FUTURE)

        assert len(source.calls) == 1

    def test_different_intervals_cache_independently(
        self, repo: SQLiteCandleRepository
    ) -> None:
        source = RecordingSource()
        service = MarketDataService(source, repo)

        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=2), now=FAR_FUTURE
        )
        service.get_candles(
            SYMBOL, Interval.H1, DAY, DAY + dt.timedelta(hours=2), now=FAR_FUTURE
        )

        assert len(source.calls) == 2


class TestWarmup:
    def test_fetch_starts_before_the_requested_start(
        self, repo: SQLiteCandleRepository
    ) -> None:
        """Anchoring at exactly `start` needs a candle that already closed by
        then, so the fetch must reach back past it."""
        source = RecordingSource()
        service = MarketDataService(source, repo)

        service.get_candles(
            SYMBOL,
            Interval.M5,
            DAY,
            DAY + dt.timedelta(hours=1),
            warmup_candles=3,
            now=FAR_FUTURE,
        )

        requested_start, _ = source.calls[0]
        assert requested_start == DAY - 3 * Interval.M5.duration

    def test_returned_frame_can_anchor_its_own_start(
        self, repo: SQLiteCandleRepository
    ) -> None:
        from backtool.research.anchoring import anchor_at

        source = RecordingSource()
        service = MarketDataService(source, repo)

        frame = service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=1), now=FAR_FUTURE
        )

        anchor = anchor_at(frame, DAY)
        assert anchor is not None, "warmup candles did not make the start anchorable"
        assert anchor.lag == dt.timedelta(0)

    def test_zero_warmup_cannot_anchor_its_start(
        self, repo: SQLiteCandleRepository
    ) -> None:
        from backtool.research.anchoring import anchor_at

        source = RecordingSource()
        service = MarketDataService(source, repo)

        frame = service.get_candles(
            SYMBOL,
            Interval.M5,
            DAY,
            DAY + dt.timedelta(hours=1),
            warmup_candles=0,
            now=FAR_FUTURE,
        )

        assert anchor_at(frame, DAY) is None


class TestHorizon:
    def test_coverage_never_extends_past_the_last_closed_candle(
        self, repo: SQLiteCandleRepository
    ) -> None:
        """The in-flight candle is not stored. If coverage claimed to include
        it, it would be skipped forever once it finally closed."""
        source = RecordingSource()
        service = MarketDataService(source, repo)
        now = DAY + dt.timedelta(hours=2, minutes=3)  # mid-candle

        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=3), now=now
        )

        coverage = repo.coverage(SYMBOL, Interval.M5)
        assert coverage.ranges[-1].end == DAY + dt.timedelta(hours=2)

    def test_the_unclosed_candle_is_fetched_on_a_later_run(
        self, repo: SQLiteCandleRepository
    ) -> None:
        source = RecordingSource()
        service = MarketDataService(source, repo)

        service.get_candles(
            SYMBOL,
            Interval.M5,
            DAY,
            DAY + dt.timedelta(hours=3),
            now=DAY + dt.timedelta(hours=2, minutes=3),
        )
        source.calls.clear()

        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=3), now=FAR_FUTURE
        )

        assert len(source.calls) == 1
        assert source.calls[0][0] == DAY + dt.timedelta(hours=2)


class TestValidation:
    def test_rejects_inverted_range(self, repo: SQLiteCandleRepository) -> None:
        service = MarketDataService(RecordingSource(), repo)
        with pytest.raises(ValueError, match="before"):
            service.get_candles(SYMBOL, Interval.M5, DAY, DAY - dt.timedelta(hours=1))

    def test_rejects_naive_bounds(self, repo: SQLiteCandleRepository) -> None:
        service = MarketDataService(RecordingSource(), repo)
        with pytest.raises(ValueError, match="Naive datetime"):
            service.get_candles(SYMBOL, Interval.M5, dt.datetime(2024, 1, 31), DAY)

    def test_cached_range_is_none_when_empty(self, repo: SQLiteCandleRepository) -> None:
        service = MarketDataService(RecordingSource(), repo)
        assert service.cached_range(SYMBOL, Interval.M5) is None

    def test_cached_range_reports_the_span(self, repo: SQLiteCandleRepository) -> None:
        service = MarketDataService(RecordingSource(), repo)
        service.get_candles(
            SYMBOL, Interval.M5, DAY, DAY + dt.timedelta(hours=2), now=FAR_FUTURE
        )

        cached = service.cached_range(SYMBOL, Interval.M5)
        assert cached is not None
        assert cached[1] == DAY + dt.timedelta(hours=2)
