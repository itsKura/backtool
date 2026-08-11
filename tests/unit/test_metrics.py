"""Tests for per-window metrics, against hand-computed expected values.

Every number asserted here can be verified with a calculator from the synthetic
prices in the test itself. That is the point: if a metric drifts, the test fails
for an arithmetic reason.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest
from tests.conftest import CANDLE_START, EVENT_TIME, build_candles

from backtool.core.time import UTC
from backtool.core.types import Interval
from backtool.research.metrics import Direction, WindowStatus, compute_window_metrics
from backtool.research.windows import WindowSpec

POST_EVENT_1H = WindowSpec(name="post_event_1h", start="0h", end="1h")
PRE_EVENT_1H = WindowSpec(name="pre_event_1h", start="-1h", end="0h")


class TestReturns:
    def test_simple_upward_move(self) -> None:
        """36 flat candles at 100.0, except the one closing at 20:00 which is
        110.0. Anchors: 19:00 -> 100.0 (index 11), 20:00 -> 110.0 (index 23).
        Return is therefore exactly +10%."""
        closes = [100.0] * 36
        closes[23] = 110.0
        candles = build_candles(CANDLE_START, closes)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.status is WindowStatus.OK
        assert metrics.start_price == 100.0
        assert metrics.end_price == 110.0
        assert metrics.return_pct == pytest.approx(10.0)
        assert metrics.abs_move_pct == pytest.approx(10.0)
        assert metrics.direction is Direction.UP

    def test_downward_move_reports_negative_return_and_positive_abs_move(self) -> None:
        closes = [100.0] * 36
        closes[23] = 95.0
        candles = build_candles(CANDLE_START, closes)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.return_pct == pytest.approx(-5.0)
        assert metrics.abs_move_pct == pytest.approx(5.0)
        assert metrics.direction is Direction.DOWN

    def test_unchanged_price_is_flat(self, flat_candles: pd.DataFrame) -> None:
        metrics = compute_window_metrics(
            flat_candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )
        assert metrics.return_pct == 0.0
        assert metrics.direction is Direction.FLAT

    def test_pre_and_post_windows_share_the_event_anchor(self) -> None:
        """The pre-event window ends *at* the event and the post-event window
        starts there, so both must resolve to the same candle. If they ever
        disagree, one of them is straddling the announcement.

        Candles start at 17:00 so that history exists before the pre-event
        window's 18:00 start -- see
        ``test_window_starting_at_the_first_candle_has_no_anchor``.
        """
        start = dt.datetime(2024, 1, 31, 17, 0, tzinfo=UTC)
        closes = [float(100 + index) for index in range(48)]
        candles = build_candles(start, closes)

        pre = compute_window_metrics(candles, PRE_EVENT_1H.resolve(EVENT_TIME), Interval.M5)
        post = compute_window_metrics(candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5)

        # close_time = 17:05 + i*5m, so 18:00 -> i=11 (111.0), 19:00 -> i=23 (123.0).
        assert pre.start_price == 111.0
        assert pre.end_price == post.start_price == 123.0
        assert pre.return_pct == pytest.approx((123.0 / 111.0 - 1) * 100)


class TestExcursion:
    def test_path_metrics_are_independent_of_the_end_return(self) -> None:
        """Price ends exactly where it started, but traded to 105 and down to
        92 along the way. A return-only view would call this a non-event."""
        closes = [100.0] * 36
        highs = [100.0] * 36
        lows = [100.0] * 36
        highs[15] = 105.0
        lows[18] = 92.0
        candles = build_candles(CANDLE_START, closes, highs=highs, lows=lows)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.return_pct == 0.0
        assert metrics.direction is Direction.FLAT
        assert metrics.high == 105.0
        assert metrics.low == 92.0
        assert metrics.mfe_pct == pytest.approx(5.0)
        assert metrics.mae_pct == pytest.approx(-8.0)
        assert metrics.range_pct == pytest.approx(13.0)

    def test_mfe_is_negative_when_price_never_traded_above_entry(self) -> None:
        """Deliberately not floored at zero: "the best it ever got was -2%" is
        information worth keeping."""
        closes = [100.0] * 36
        for index in range(12, 24):
            closes[index] = 98.0
        candles = build_candles(CANDLE_START, closes)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.mfe_pct == pytest.approx(-2.0)
        assert metrics.mae_pct == pytest.approx(-2.0)

    def test_excursion_ignores_candles_outside_the_window(self) -> None:
        """A spike before the window must not contaminate the window's high."""
        closes = [100.0] * 36
        highs = [100.0] * 36
        highs[5] = 500.0  # 18:25-18:30, well before the 19:00 window start
        candles = build_candles(CANDLE_START, closes, highs=highs)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.high == 100.0


class TestRealizedVolatility:
    def test_flat_series_has_zero_volatility(self, flat_candles: pd.DataFrame) -> None:
        metrics = compute_window_metrics(
            flat_candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )
        assert metrics.realized_vol_pct == pytest.approx(0.0)

    def test_single_jump_equals_the_log_return_of_that_jump(self) -> None:
        """Eleven zero returns then one jump of +10%: the root sum of squares
        collapses to |ln(1.1)| = 0.0953102, i.e. 9.53102%."""
        closes = [100.0] * 36
        closes[23] = 110.0
        candles = build_candles(CANDLE_START, closes)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.realized_vol_pct == pytest.approx(math.log(1.1) * 100)


class TestCoverage:
    def test_complete_window_reports_full_coverage(self, flat_candles: pd.DataFrame) -> None:
        """A 1-hour window at 5m expects exactly 12 candles."""
        metrics = compute_window_metrics(
            flat_candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )
        assert metrics.expected_candles == 12
        assert metrics.candle_count == 12
        assert metrics.coverage_pct == pytest.approx(100.0)
        assert metrics.start_anchor_lag_s == 0.0
        assert metrics.end_anchor_lag_s == 0.0

    def test_missing_candles_lower_coverage_without_failing(self) -> None:
        """Metrics are still produced -- the caller decides whether 75% of the
        data is enough. Silently dropping the event would hide the problem."""
        candles = build_candles(CANDLE_START, [100.0] * 36)
        gapped = candles[
            ~(
                (candles["close_time"] > pd.Timestamp("2024-01-31 19:15", tz="UTC"))
                & (candles["close_time"] <= pd.Timestamp("2024-01-31 19:30", tz="UTC"))
            )
        ].reset_index(drop=True)

        metrics = compute_window_metrics(
            gapped, POST_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.status is WindowStatus.OK
        assert metrics.expected_candles == 12
        assert metrics.candle_count == 9
        assert metrics.coverage_pct == pytest.approx(75.0)

    def test_expected_count_scales_with_interval(self) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 6, Interval.H1)
        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(EVENT_TIME), Interval.H1
        )
        assert metrics.expected_candles == 1
        assert metrics.candle_count == 1


class TestInsufficientData:
    def test_window_entirely_before_history_is_reported_not_raised(self) -> None:
        """BTCUSDT began trading 2017-08-17; earlier FOMC events land here. One
        unusable event must not abort a twenty-event study."""
        candles = build_candles(CANDLE_START, [100.0] * 36)
        early_event = dt.datetime(2024, 1, 30, 12, 0, tzinfo=UTC)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(early_event), Interval.M5
        )

        assert metrics.status is WindowStatus.INSUFFICIENT_DATA
        assert metrics.return_pct is None
        assert metrics.is_usable is False
        assert "No candle closed at or before" in (metrics.reason or "")

    def test_window_starting_at_the_first_candle_has_no_anchor(self) -> None:
        """A window needs at least one candle closed *strictly before* its
        start, because the start price is the last close at or before it.

        Candles here begin at 18:00, so the first close is 18:05 and a window
        starting at 18:00 cannot be priced. This is a real constraint on the
        fetch layer: it must request history that begins before the earliest
        window, not merely at it.
        """
        candles = build_candles(CANDLE_START, [100.0] * 36)

        metrics = compute_window_metrics(
            candles, PRE_EVENT_1H.resolve(EVENT_TIME), Interval.M5
        )

        assert metrics.status is WindowStatus.INSUFFICIENT_DATA
        assert "No candle closed at or before" in (metrics.reason or "")

    def test_window_shorter_than_one_candle_is_insufficient(self) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 36)
        tiny = WindowSpec(name="tiny", start="0m", end="3m")

        metrics = compute_window_metrics(candles, tiny.resolve(EVENT_TIME), Interval.M5)

        assert metrics.status is WindowStatus.INSUFFICIENT_DATA
        assert "No candles closed inside the window" in (metrics.reason or "")
        # Anchors still resolved, so the caller can see prices existed.
        assert metrics.start_price == 100.0

    def test_unusable_window_still_reports_its_identity(self) -> None:
        """Needed so aggregation can count attempted-but-failed events."""
        candles = build_candles(CANDLE_START, [100.0] * 36)
        early_event = dt.datetime(2024, 1, 30, 12, 0, tzinfo=UTC)

        metrics = compute_window_metrics(
            candles, POST_EVENT_1H.resolve(early_event), Interval.M5
        )

        assert metrics.name == "post_event_1h"
        assert metrics.start_utc == early_event
        assert metrics.expected_candles == 12
