"""Tests for the SQLite candle repository."""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from tests.conftest import CANDLE_START, build_candles

from backtool.core.types import Interval
from backtool.data.storage.coverage import CoverageIndex, TimeRange
from backtool.data.storage.sqlite import SQLiteCandleRepository

SYMBOL = "BTCUSDT"


@pytest.fixture
def repo(tmp_path: Path) -> SQLiteCandleRepository:
    return SQLiteCandleRepository(tmp_path / "candles.db")


def span(hours: int = 3) -> tuple[dt.datetime, dt.datetime]:
    return CANDLE_START, CANDLE_START + dt.timedelta(hours=hours)


class TestRoundTrip:
    def test_stores_and_reads_back_identical_prices(
        self, repo: SQLiteCandleRepository
    ) -> None:
        candles = build_candles(CANDLE_START, [100.0 + i for i in range(36)])
        start, end = span()
        repo.store(SYMBOL, Interval.M5, candles, start, end)

        result = repo.read(SYMBOL, Interval.M5, start, end)

        assert len(result) == 36
        pd.testing.assert_frame_equal(result, candles)

    def test_read_preserves_the_pinned_timestamp_dtype(
        self, repo: SQLiteCandleRepository
    ) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        start, end = span(1)
        repo.store(SYMBOL, Interval.M5, candles, start, end)

        result = repo.read(SYMBOL, Interval.M5, start, end)
        assert str(result["open_time"].dtype) == "datetime64[ns, UTC]"

    def test_read_is_bounded_half_open(self, repo: SQLiteCandleRepository) -> None:
        """[start, end): a candle opening exactly at `end` is excluded."""
        candles = build_candles(CANDLE_START, [100.0] * 12)
        repo.store(SYMBOL, Interval.M5, candles, *span(1))

        result = repo.read(
            SYMBOL,
            Interval.M5,
            CANDLE_START,
            CANDLE_START + dt.timedelta(minutes=30),
        )
        assert len(result) == 6
        assert result["open_time"].iloc[0] == pd.Timestamp(CANDLE_START)

    def test_empty_read_returns_a_typed_empty_frame(
        self, repo: SQLiteCandleRepository
    ) -> None:
        result = repo.read(SYMBOL, Interval.M5, *span())
        assert result.empty
        assert "close_time" in result.columns

    def test_symbol_is_case_insensitive(self, repo: SQLiteCandleRepository) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        repo.store("btcusdt", Interval.M5, candles, *span(1))
        assert len(repo.read("BTCUSDT", Interval.M5, *span(1))) == 12


class TestIsolation:
    def test_intervals_do_not_collide(self, repo: SQLiteCandleRepository) -> None:
        """5m and 1h candles share open_time values; they must not overwrite."""
        five = build_candles(CANDLE_START, [100.0] * 36, Interval.M5)
        hourly = build_candles(CANDLE_START, [500.0] * 3, Interval.H1)
        repo.store(SYMBOL, Interval.M5, five, *span())
        repo.store(SYMBOL, Interval.H1, hourly, *span())

        assert len(repo.read(SYMBOL, Interval.M5, *span())) == 36
        assert len(repo.read(SYMBOL, Interval.H1, *span())) == 3
        assert repo.read(SYMBOL, Interval.H1, *span())["close"].iloc[0] == 500.0

    def test_symbols_do_not_collide(self, repo: SQLiteCandleRepository) -> None:
        repo.store(SYMBOL, Interval.M5, build_candles(CANDLE_START, [100.0] * 12), *span(1))
        repo.store("ETHUSDT", Interval.M5, build_candles(CANDLE_START, [50.0] * 12), *span(1))

        assert repo.read(SYMBOL, Interval.M5, *span(1))["close"].iloc[0] == 100.0
        assert repo.read("ETHUSDT", Interval.M5, *span(1))["close"].iloc[0] == 50.0
        assert repo.symbols() == [("BTCUSDT", "5m"), ("ETHUSDT", "5m")]


class TestIdempotency:
    def test_storing_twice_leaves_one_copy(self, repo: SQLiteCandleRepository) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        repo.store(SYMBOL, Interval.M5, candles, *span(1))
        repo.store(SYMBOL, Interval.M5, candles, *span(1))

        assert repo.candle_count(SYMBOL, Interval.M5) == 12

    def test_re_storing_corrects_a_stale_candle(self, repo: SQLiteCandleRepository) -> None:
        """A re-fetch must win, so an exchange correction can be applied."""
        repo.store(SYMBOL, Interval.M5, build_candles(CANDLE_START, [100.0] * 12), *span(1))
        repo.store(SYMBOL, Interval.M5, build_candles(CANDLE_START, [222.0] * 12), *span(1))

        result = repo.read(SYMBOL, Interval.M5, *span(1))
        assert len(result) == 12
        assert (result["close"] == 222.0).all()

    def test_overlapping_stores_merge_without_duplicates(
        self, repo: SQLiteCandleRepository
    ) -> None:
        first = build_candles(CANDLE_START, [100.0] * 24)
        second = build_candles(CANDLE_START + dt.timedelta(minutes=60), [100.0] * 24)
        repo.store(SYMBOL, Interval.M5, first, CANDLE_START, CANDLE_START + dt.timedelta(hours=2))
        repo.store(
            SYMBOL,
            Interval.M5,
            second,
            CANDLE_START + dt.timedelta(hours=1),
            CANDLE_START + dt.timedelta(hours=3),
        )

        assert repo.candle_count(SYMBOL, Interval.M5) == 36


class TestCoverage:
    def test_coverage_is_recorded_on_store(self, repo: SQLiteCandleRepository) -> None:
        start, end = span()
        repo.store(SYMBOL, Interval.M5, build_candles(CANDLE_START, [100.0] * 36), start, end)

        assert repo.coverage(SYMBOL, Interval.M5) == CoverageIndex((TimeRange(start, end),))

    def test_coverage_is_recorded_even_with_no_candles(
        self, repo: SQLiteCandleRepository
    ) -> None:
        """An empty response means the market had nothing there -- a pre-listing
        period or an outage. Without this, every run would re-request it."""
        from backtool.data.candles import empty_candles

        start, end = span()
        repo.store(SYMBOL, Interval.M5, empty_candles(Interval.M5), start, end)

        assert repo.coverage(SYMBOL, Interval.M5).covers(start, end)
        assert repo.candle_count(SYMBOL, Interval.M5) == 0

    def test_coverage_merges_across_stores(self, repo: SQLiteCandleRepository) -> None:
        from backtool.data.candles import empty_candles

        empty = empty_candles(Interval.M5)
        base = CANDLE_START
        repo.store(SYMBOL, Interval.M5, empty, base, base + dt.timedelta(hours=1))
        repo.store(
            SYMBOL,
            Interval.M5,
            empty,
            base + dt.timedelta(hours=1),
            base + dt.timedelta(hours=2),
        )

        coverage = repo.coverage(SYMBOL, Interval.M5)
        assert len(coverage) == 1
        assert coverage.covers(base, base + dt.timedelta(hours=2))

    def test_coverage_is_per_symbol_and_interval(
        self, repo: SQLiteCandleRepository
    ) -> None:
        start, end = span()
        repo.store(SYMBOL, Interval.M5, build_candles(CANDLE_START, [100.0] * 36), start, end)

        assert repo.coverage(SYMBOL, Interval.H1) == CoverageIndex()
        assert repo.coverage("ETHUSDT", Interval.M5) == CoverageIndex()

    def test_unknown_symbol_has_empty_coverage(self, repo: SQLiteCandleRepository) -> None:
        assert repo.coverage("DOGEUSDT", Interval.M5) == CoverageIndex()


class TestAtomicity:
    def test_failed_write_leaves_no_coverage_behind(
        self, repo: SQLiteCandleRepository
    ) -> None:
        """The dangerous failure is coverage recorded without candles: the range
        would be permanently skipped and research would run on missing data.
        Candles and coverage therefore commit in one transaction.
        """
        candles = build_candles(CANDLE_START, [100.0] * 12)
        start, end = span(1)

        # Fail between the candle insert and the coverage insert, inside the
        # open transaction.
        with (
            patch.object(
                CoverageIndex,
                "with_range",
                side_effect=sqlite3.OperationalError("simulated disk failure"),
            ),
            pytest.raises(sqlite3.OperationalError),
        ):
            repo.store(SYMBOL, Interval.M5, candles, start, end)

        assert repo.coverage(SYMBOL, Interval.M5) == CoverageIndex()
        assert repo.candle_count(SYMBOL, Interval.M5) == 0, (
            "candles were committed without their coverage record"
        )

    def test_successful_write_commits_both(self, repo: SQLiteCandleRepository) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        start, end = span(1)
        repo.store(SYMBOL, Interval.M5, candles, start, end)

        assert repo.candle_count(SYMBOL, Interval.M5) == 12
        assert repo.coverage(SYMBOL, Interval.M5).covers(start, end)


class TestPersistenceAcrossInstances:
    def test_data_survives_reopening_the_database(self, tmp_path: Path) -> None:
        path = tmp_path / "candles.db"
        start, end = span(1)

        first = SQLiteCandleRepository(path)
        first.store(SYMBOL, Interval.M5, build_candles(CANDLE_START, [100.0] * 12), start, end)

        second = SQLiteCandleRepository(path)
        assert len(second.read(SYMBOL, Interval.M5, start, end)) == 12
        assert second.coverage(SYMBOL, Interval.M5).covers(start, end)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        repo = SQLiteCandleRepository(tmp_path / "nested" / "deeper" / "candles.db")
        assert repo.path.exists()
