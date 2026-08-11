"""Tests for no-look-ahead price anchoring (ADR 002).

The first test in this file is the most important one in the project: it pins
the rule that separates a real event study from a fictional one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from tests.conftest import CANDLE_START, build_candles

from backtool.core.time import UTC
from backtool.core.types import Interval
from backtool.research.anchoring import anchor_at


class TestLookAheadPrevention:
    def test_never_uses_a_candle_that_closes_after_the_target(self) -> None:
        """The core guarantee of ADR 002.

        With 5m candles and a target of 19:03, the candle *containing* the
        target is [19:00, 19:05) -- it closes at 19:05, two minutes after the
        event. Using its close would import post-event information into a
        pre-event measurement.

        Candle 11 closes at 19:00 and is priced 100. Candle 12 closes at 19:05
        and is priced 999. Anchoring at 19:03 must return 100.
        """
        closes = [100.0] * 36
        closes[12] = 999.0
        candles = build_candles(CANDLE_START, closes)

        anchor = anchor_at(candles, dt.datetime(2024, 1, 31, 19, 3, tzinfo=UTC))

        assert anchor is not None
        assert anchor.price == 100.0, "leaked the post-target candle's close"
        assert anchor.time == dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert anchor.lag == dt.timedelta(minutes=3)

    def test_aligned_target_has_zero_lag(self) -> None:
        """A candle closing exactly at the target is fully observable then, so
        it is used with no staleness. This is why an FOMC release at 19:00:00
        anchors perfectly against 5m candles."""
        candles = build_candles(CANDLE_START, [float(100 + i) for i in range(36)])

        anchor = anchor_at(candles, dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC))

        assert anchor is not None
        assert anchor.lag == dt.timedelta(0)
        assert anchor.lag_seconds == 0.0
        # open_time 18:55 is index 11, priced 100 + 11.
        assert anchor.price == 111.0

    def test_lag_grows_when_candles_are_missing(self) -> None:
        """A gap in the feed shows up as a large anchor lag rather than as a
        silently substituted price."""
        candles = build_candles(CANDLE_START, [100.0] * 36)
        # Drop everything that closes in (18:30, 19:00] to simulate an outage.
        gapped = candles[
            ~(
                (candles["close_time"] > pd.Timestamp("2024-01-31 18:30", tz="UTC"))
                & (candles["close_time"] <= pd.Timestamp("2024-01-31 19:00", tz="UTC"))
            )
        ].reset_index(drop=True)

        anchor = anchor_at(gapped, dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC))

        assert anchor is not None
        assert anchor.time == dt.datetime(2024, 1, 31, 18, 30, tzinfo=UTC)
        assert anchor.lag == dt.timedelta(minutes=30)


class TestBoundaries:
    def test_returns_none_when_target_precedes_all_history(self) -> None:
        """A legitimate outcome, not an error: BTCUSDT simply did not exist
        before 2017-08-17."""
        candles = build_candles(CANDLE_START, [100.0] * 12)
        assert anchor_at(candles, dt.datetime(2024, 1, 31, 17, 0, tzinfo=UTC)) is None

    def test_target_exactly_at_first_close_is_resolvable(self) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        anchor = anchor_at(candles, dt.datetime(2024, 1, 31, 18, 5, tzinfo=UTC))
        assert anchor is not None
        assert anchor.lag == dt.timedelta(0)

    def test_target_one_second_before_first_close_is_not(self) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        assert anchor_at(candles, dt.datetime(2024, 1, 31, 18, 4, 59, tzinfo=UTC)) is None

    def test_target_after_all_history_uses_the_last_candle(self) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        anchor = anchor_at(candles, dt.datetime(2024, 2, 1, 12, 0, tzinfo=UTC))
        assert anchor is not None
        assert anchor.time == dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert anchor.lag > dt.timedelta(hours=16)

    def test_empty_frame_returns_none(self) -> None:
        empty = build_candles(CANDLE_START, [100.0]).iloc[0:0]
        assert anchor_at(empty, dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)) is None

    def test_rejects_naive_target(self) -> None:
        candles = build_candles(CANDLE_START, [100.0] * 12)
        with pytest.raises(ValueError, match="Naive datetime"):
            anchor_at(candles, dt.datetime(2024, 1, 31, 19, 0))

    def test_accepts_non_utc_target(self) -> None:
        from backtool.core.time import NEW_YORK

        candles = build_candles(CANDLE_START, [float(100 + i) for i in range(36)])
        anchor = anchor_at(candles, dt.datetime(2024, 1, 31, 14, 0, tzinfo=NEW_YORK))
        assert anchor is not None
        assert anchor.price == 111.0  # identical to the 19:00 UTC result

    def test_works_across_intervals(self) -> None:
        candles = build_candles(CANDLE_START, [100.0, 200.0, 300.0], Interval.H1)
        anchor = anchor_at(candles, dt.datetime(2024, 1, 31, 20, 30, tzinfo=UTC))
        assert anchor is not None
        assert anchor.price == 200.0  # candle [19:00, 20:00) closed at 20:00
        assert anchor.lag == dt.timedelta(minutes=30)
