"""Tests for candle normalisation and data-integrity checks.

These checks are the system's immune response to bad market data. Corrupt
candles do not crash anything downstream -- they quietly produce plausible,
wrong statistics -- so they must be rejected at the boundary.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from tests.conftest import CANDLE_START, build_candles

from backtool.core.time import UTC
from backtool.core.types import Interval
from backtool.data.candles import (
    CANDLE_COLUMNS,
    CANDLE_TIME_DTYPE,
    CandleDataError,
    expected_candle_count,
    normalise_candles,
)


def raw_frame(**overrides: object) -> pd.DataFrame:
    """A minimal valid three-candle raw frame, with optional column overrides."""
    data: dict[str, object] = {
        "open_time": [
            dt.datetime(2024, 1, 31, 18, 0, tzinfo=UTC),
            dt.datetime(2024, 1, 31, 18, 5, tzinfo=UTC),
            dt.datetime(2024, 1, 31, 18, 10, tzinfo=UTC),
        ],
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 102.5],
        "volume": [10.0, 11.0, 12.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestNormalisation:
    def test_derives_close_time_as_open_time_plus_interval(self) -> None:
        """Consecutive candles share a boundary: close_time[i] == open_time[i+1].
        This is what makes a 19:00:00 event anchor with zero lag."""
        result = normalise_candles(raw_frame(), Interval.M5)

        assert result["close_time"].iloc[0] == pd.Timestamp("2024-01-31 18:05", tz="UTC")
        assert result["close_time"].iloc[0] == result["open_time"].iloc[1]

    def test_output_has_the_documented_column_set_and_order(self) -> None:
        result = normalise_candles(raw_frame(), Interval.M5)
        assert tuple(result.columns) == CANDLE_COLUMNS

    def test_pins_the_timestamp_resolution(self) -> None:
        """Regression guard.

        pandas infers resolution from its input -- pandas 3.0 yields
        ``datetime64[us]`` from Python datetimes where 2.x yielded
        ``datetime64[ns]``. Code that converts timestamps to integers is only
        correct if it knows the unit, so normalisation pins it. Letting this
        drift previously made every grid check fail and every anchor silently
        resolve to the last candle in the frame.
        """
        result = normalise_candles(raw_frame(), Interval.M5)
        assert str(result["open_time"].dtype) == CANDLE_TIME_DTYPE
        assert str(result["close_time"].dtype) == CANDLE_TIME_DTYPE

    def test_pins_resolution_regardless_of_input_resolution(self) -> None:
        frame = raw_frame()
        frame["open_time"] = frame["open_time"].astype("datetime64[s, UTC]")
        result = normalise_candles(frame, Interval.M5)
        assert str(result["open_time"].dtype) == CANDLE_TIME_DTYPE

    def test_sorts_unordered_input(self) -> None:
        shuffled = raw_frame().iloc[[2, 0, 1]].reset_index(drop=True)
        result = normalise_candles(shuffled, Interval.M5)
        assert result["open_time"].is_monotonic_increasing
        assert list(result.index) == [0, 1, 2]

    def test_converts_non_utc_timestamps_to_utc(self) -> None:
        from backtool.core.time import NEW_YORK

        frame = raw_frame(
            open_time=[
                dt.datetime(2024, 1, 31, 13, 0, tzinfo=NEW_YORK),
                dt.datetime(2024, 1, 31, 13, 5, tzinfo=NEW_YORK),
                dt.datetime(2024, 1, 31, 13, 10, tzinfo=NEW_YORK),
            ]
        )
        result = normalise_candles(frame, Interval.M5)
        assert result["open_time"].iloc[0] == pd.Timestamp("2024-01-31 18:00", tz="UTC")

    def test_ignores_extra_columns(self) -> None:
        frame = raw_frame()
        frame["number_of_trades"] = [5, 6, 7]
        result = normalise_candles(frame, Interval.M5)
        assert "number_of_trades" not in result.columns

    def test_does_not_mutate_the_input_frame(self) -> None:
        frame = raw_frame()
        before = frame.copy()
        normalise_candles(frame, Interval.M5)
        pd.testing.assert_frame_equal(frame, before)


class TestStructuralRejections:
    def test_rejects_missing_columns(self) -> None:
        frame = raw_frame().drop(columns=["volume"])
        with pytest.raises(CandleDataError, match="missing required column"):
            normalise_candles(frame, Interval.M5)

    def test_rejects_naive_timestamps(self) -> None:
        frame = raw_frame(
            open_time=[
                dt.datetime(2024, 1, 31, 18, 0),
                dt.datetime(2024, 1, 31, 18, 5),
                dt.datetime(2024, 1, 31, 18, 10),
            ]
        )
        with pytest.raises(CandleDataError, match="timezone-naive"):
            normalise_candles(frame, Interval.M5)

    def test_rejects_duplicate_open_times(self) -> None:
        """Exchange pagination commonly overlaps at range boundaries; a
        duplicated candle would be double-counted in every path metric."""
        frame = raw_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with pytest.raises(CandleDataError, match="Duplicate open_time"):
            normalise_candles(frame, Interval.M5)

    def test_rejects_timestamps_off_the_interval_grid(self) -> None:
        """Catches the case where a 5m frame is passed as 15m, or where data
        was resampled with a shifted origin."""
        with pytest.raises(CandleDataError, match="not aligned to the 15m grid"):
            normalise_candles(raw_frame(), Interval.M15)

    def test_accepts_a_correctly_aligned_hourly_frame(self) -> None:
        frame = raw_frame(
            open_time=[
                dt.datetime(2024, 1, 31, 18, 0, tzinfo=UTC),
                dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC),
                dt.datetime(2024, 1, 31, 20, 0, tzinfo=UTC),
            ]
        )
        result = normalise_candles(frame, Interval.H1)
        assert len(result) == 3


class TestPriceIntegrityRejections:
    def test_rejects_missing_prices(self) -> None:
        with pytest.raises(CandleDataError, match="missing value"):
            normalise_candles(raw_frame(close=[101.0, None, 102.5]), Interval.M5)

    def test_rejects_high_below_low(self) -> None:
        with pytest.raises(CandleDataError, match="high < low"):
            normalise_candles(raw_frame(high=[98.0, 102.0, 103.0]), Interval.M5)

    def test_rejects_high_below_the_candle_body(self) -> None:
        """high must be >= both open and close; otherwise the columns are
        mislabelled or the source is corrupt."""
        with pytest.raises(CandleDataError, match="high is below open or close"):
            normalise_candles(raw_frame(high=[100.5, 102.0, 103.0]), Interval.M5)

    def test_rejects_low_above_the_candle_body(self) -> None:
        with pytest.raises(CandleDataError, match="low is above open or close"):
            normalise_candles(raw_frame(low=[100.5, 100.0, 101.0]), Interval.M5)

    def test_rejects_non_positive_prices(self) -> None:
        with pytest.raises(CandleDataError, match="non-positive price"):
            normalise_candles(raw_frame(low=[0.0, 100.0, 101.0]), Interval.M5)

    def test_accepts_a_flat_doji(self) -> None:
        """open == high == low == close is unusual but legal on thin data."""
        frame = raw_frame(
            open=[100.0, 100.0, 100.0],
            high=[100.0, 100.0, 100.0],
            low=[100.0, 100.0, 100.0],
            close=[100.0, 100.0, 100.0],
        )
        assert len(normalise_candles(frame, Interval.M5)) == 3


class TestExpectedCandleCount:
    @pytest.mark.parametrize(
        ("hours", "interval", "expected"),
        [
            (1, Interval.M5, 12),
            (24, Interval.M5, 288),
            (1, Interval.M15, 4),
            (24, Interval.M15, 96),
            (1, Interval.H1, 1),
            (24, Interval.H1, 24),
        ],
    )
    def test_counts_for_common_windows(
        self, hours: int, interval: Interval, expected: int
    ) -> None:
        start = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert expected_candle_count(start, start + dt.timedelta(hours=hours), interval) == expected

    def test_zero_for_empty_or_backwards_span(self) -> None:
        start = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert expected_candle_count(start, start, Interval.M5) == 0
        assert expected_candle_count(start, start - dt.timedelta(hours=1), Interval.M5) == 0

    def test_partial_interval_rounds_down(self) -> None:
        start = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert expected_candle_count(start, start + dt.timedelta(minutes=7), Interval.M5) == 1


class TestBuildCandlesHelper:
    def test_helper_produces_a_frame_that_passes_validation(self) -> None:
        candles = build_candles(CANDLE_START, [100.0, 101.0, 102.0])
        assert tuple(candles.columns) == CANDLE_COLUMNS
        assert len(candles) == 3
