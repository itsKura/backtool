"""Tests for the UTC standard and DST-aware local/UTC conversion.

These are the highest-value tests in the project. A one-hour error here shifts
every event window and produces results that look entirely plausible.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from backtool.core.time import (
    NEW_YORK,
    UTC,
    AmbiguousLocalTimeError,
    NonExistentLocalTimeError,
    ensure_utc,
    from_utc_ms,
    local_to_utc,
    utc_ms,
    utc_to_local,
)
from backtool.core.types import Interval


class TestEnsureUtc:
    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="Naive datetime"):
            ensure_utc(dt.datetime(2024, 1, 31, 19, 0))

    def test_passes_through_utc(self) -> None:
        value = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert ensure_utc(value) == value

    def test_converts_other_zone_to_utc(self) -> None:
        local = dt.datetime(2024, 1, 31, 14, 0, tzinfo=NEW_YORK)
        assert ensure_utc(local) == dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)


class TestLocalToUtc:
    """2:00 PM ET is 19:00 UTC under EST (UTC-5) and 18:00 UTC under EDT (UTC-4).

    Hardcoding either offset corrupts roughly half of all FOMC events.
    """

    @pytest.mark.parametrize(
        ("local_date", "expected_utc_hour", "label"),
        [
            (dt.date(2024, 1, 31), 19, "deep winter, EST"),
            (dt.date(2024, 7, 31), 18, "deep summer, EDT"),
            (dt.date(2024, 3, 20), 18, "10 days after spring-forward, EDT"),
            (dt.date(2024, 11, 7), 19, "4 days after fall-back, EST"),
            (dt.date(2023, 11, 1), 18, "4 days before fall-back, still EDT"),
            (dt.date(2023, 3, 22), 18, "10 days after spring-forward, EDT"),
        ],
    )
    def test_fomc_release_time_across_dst(
        self, local_date: dt.date, expected_utc_hour: int, label: str
    ) -> None:
        result = local_to_utc(local_date, dt.time(14, 0), NEW_YORK)
        assert result == dt.datetime(
            local_date.year, local_date.month, local_date.day, expected_utc_hour, 0, tzinfo=UTC
        ), label

    def test_rejects_nonexistent_time_in_spring_forward_gap(self) -> None:
        # 2023-03-12: US clocks jump 02:00 -> 03:00, so 02:30 never happened.
        with pytest.raises(NonExistentLocalTimeError, match="does not exist"):
            local_to_utc(dt.date(2023, 3, 12), dt.time(2, 30), NEW_YORK)

    def test_rejects_ambiguous_time_in_fall_back_overlap(self) -> None:
        # 2023-11-05: US clocks fall 02:00 -> 01:00, so 01:30 happened twice.
        with pytest.raises(AmbiguousLocalTimeError, match="ambiguous"):
            local_to_utc(dt.date(2023, 11, 5), dt.time(1, 30), NEW_YORK)

    def test_accepts_times_adjacent_to_the_dst_gap(self) -> None:
        """The boundary check must not reject valid neighbouring times."""
        before = local_to_utc(dt.date(2023, 3, 12), dt.time(1, 59), NEW_YORK)
        after = local_to_utc(dt.date(2023, 3, 12), dt.time(3, 0), NEW_YORK)
        assert before == dt.datetime(2023, 3, 12, 6, 59, tzinfo=UTC)
        assert after == dt.datetime(2023, 3, 12, 7, 0, tzinfo=UTC)

    def test_utc_zone_has_no_dst_edge_cases(self) -> None:
        result = local_to_utc(dt.date(2024, 3, 10), dt.time(2, 30), ZoneInfo("UTC"))
        assert result == dt.datetime(2024, 3, 10, 2, 30, tzinfo=UTC)


class TestUtcToLocal:
    def test_renders_utc_instant_as_new_york_wall_clock(self) -> None:
        instant = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        local = utc_to_local(instant, NEW_YORK)
        assert (local.hour, local.minute) == (14, 0)

    def test_round_trips_through_local_and_back(self) -> None:
        original = dt.datetime(2024, 7, 31, 18, 0, tzinfo=UTC)
        assert ensure_utc(utc_to_local(original, NEW_YORK)) == original


class TestEpochMilliseconds:
    def test_known_epoch_value(self) -> None:
        # 2024-01-31 19:00:00 UTC, verified independently.
        assert utc_ms(dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)) == 1706727600000

    def test_round_trip(self) -> None:
        original = dt.datetime(2021, 12, 15, 19, 0, tzinfo=UTC)
        assert from_utc_ms(utc_ms(original)) == original

    def test_rejects_naive_input(self) -> None:
        with pytest.raises(ValueError, match="Naive datetime"):
            utc_ms(dt.datetime(2024, 1, 31, 19, 0))


class TestInterval:
    @pytest.mark.parametrize(
        ("interval", "seconds"),
        [(Interval.M5, 300), (Interval.M15, 900), (Interval.H1, 3600)],
    )
    def test_duration(self, interval: Interval, seconds: int) -> None:
        assert interval.duration.total_seconds() == seconds
        assert interval.milliseconds == seconds * 1000

    def test_serialises_as_exchange_native_string(self) -> None:
        assert Interval.M5 == "5m"
        assert f"{Interval.H1.value}" == "1h"
