"""Tests for the CPI calendar and the generalised event loader.

CPI is the proof that the architecture generalises: a different release time
(08:30 ET, not 14:00), a different cadence (monthly, not eight a year), and a
data file is the entire change. These tests pin both the data and that claim.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.core.time import UTC
from backtool.core.types import EventType
from backtool.events import CALENDAR_FILES, MarketEvent, load_events, select_last_n


@pytest.fixture(scope="module")
def cpi() -> list[MarketEvent]:
    return load_events(EventType.CPI)


class TestCalendarRegistry:
    def test_every_event_type_has_a_calendar(self) -> None:
        """Adding an EventType without a calendar would fail only at run time."""
        assert set(CALENDAR_FILES) == set(EventType)

    def test_loads_each_type_independently(self) -> None:
        for kind in EventType:
            events = load_events(kind)
            assert events, f"{kind.value} calendar is empty"
            assert {event.event_type for event in events} == {kind}

    def test_loading_all_types_merges_and_sorts(self) -> None:
        merged = load_events()
        timestamps = [event.timestamp_utc for event in merged]
        assert timestamps == sorted(timestamps)
        assert {event.event_type for event in merged} == set(EventType)
        assert len(merged) == sum(len(load_events(kind)) for kind in EventType)

    def test_event_ids_are_unique_across_types(self) -> None:
        ids = [event.event_id for event in load_events()]
        assert len(ids) == len(set(ids))

    def test_path_requires_an_explicit_type(self) -> None:
        with pytest.raises(ValueError, match="requires an explicit event_type"):
            load_events(path="anything.csv")  # type: ignore[arg-type]


class TestCpiTiming:
    """CPI exercises the time layer at a different hour than FOMC.

    08:30 ET is 13:30 UTC under EST and 12:30 UTC under EDT. The same DST
    machinery that serves a 14:00 release has to serve this one with no
    special-casing.
    """

    @pytest.mark.parametrize(
        ("event_id", "expected_utc"),
        [
            ("CPI-2026-02-13", dt.datetime(2026, 2, 13, 13, 30, tzinfo=UTC)),
            ("CPI-2026-06-10", dt.datetime(2026, 6, 10, 12, 30, tzinfo=UTC)),
            ("CPI-2025-12-18", dt.datetime(2025, 12, 18, 13, 30, tzinfo=UTC)),
            ("CPI-2025-07-15", dt.datetime(2025, 7, 15, 12, 30, tzinfo=UTC)),
        ],
    )
    def test_release_time_across_dst(
        self, cpi: list[MarketEvent], event_id: str, expected_utc: dt.datetime
    ) -> None:
        event = next(e for e in cpi if e.event_id == event_id)
        assert event.timestamp_utc == expected_utc

    def test_every_release_is_at_0830_eastern(self, cpi: list[MarketEvent]) -> None:
        for event in cpi:
            assert event.local_time == dt.time(8, 30), event.event_id
            assert event.timezone == "America/New_York", event.event_id

    def test_cpi_and_fomc_differ_by_five_and_a_half_hours(self) -> None:
        """Both are 'the announcement', at different clock times. A system that
        assumed one release hour would silently mis-window the other."""
        fomc = next(
            e for e in load_events(EventType.FOMC) if e.event_id == "FOMC-2026-06-17"
        )
        cpi_event = next(
            e for e in load_events(EventType.CPI) if e.event_id == "CPI-2026-06-10"
        )
        assert fomc.timestamp_utc.hour == 18
        assert cpi_event.timestamp_utc.hour == 12


class TestCpiData:
    def test_ids_match_their_publication_date(self, cpi: list[MarketEvent]) -> None:
        """The event is the publication instant, not the month it describes."""
        for event in cpi:
            assert event.event_id == f"CPI-{event.local_date.isoformat()}"

    def test_sorted_and_unique(self, cpi: list[MarketEvent]) -> None:
        timestamps = [event.timestamp_utc for event in cpi]
        assert timestamps == sorted(timestamps)
        assert len({event.event_id for event in cpi}) == len(cpi)

    def test_releases_fall_on_weekdays(self, cpi: list[MarketEvent]) -> None:
        for event in cpi:
            assert event.local_date.weekday() < 5, (
                f"{event.event_id} falls on {event.local_date.strftime('%A')}"
            )

    def test_no_october_2025_release_exists(self, cpi: list[MarketEvent]) -> None:
        """The October 2025 reference month was never published -- the data
        could not be collected during the appropriations lapse. A calendar
        generated from a "monthly on the ~12th" rule would invent one, and the
        study would silently measure a non-event.
        """
        dates = {event.local_date for event in cpi}
        assert dt.date(2025, 11, 12) not in dates
        assert dt.date(2025, 11, 13) not in dates
        # The gap is visible: Sept ref published late Oct, then nothing until Dec.
        assert dt.date(2025, 10, 24) in dates
        assert dt.date(2025, 12, 18) in dates

    def test_shutdown_delayed_releases_are_marked_unscheduled(
        self, cpi: list[MarketEvent]
    ) -> None:
        delayed = {event.event_id for event in cpi if not event.is_scheduled}
        assert delayed == {"CPI-2025-10-24", "CPI-2025-12-18", "CPI-2026-02-13"}

    def test_delayed_releases_explain_themselves(self, cpi: list[MarketEvent]) -> None:
        for event in cpi:
            if not event.is_scheduled:
                assert "appropriations lapse" in event.notes, event.event_id

    def test_roughly_monthly_cadence_with_the_two_shutdown_gaps(
        self, cpi: list[MarketEvent]
    ) -> None:
        """CPI is monthly, so consecutive releases sit 20-40 days apart -- with
        exactly two exceptions, both caused by the 2025 appropriations lapse:

        * **43 days**, 11 Sep to 24 Oct 2025: the September reference month was
          published late.
        * **55 days**, 24 Oct to 18 Dec 2025: the October reference month was
          never published at all.

        A calendar generated from a "monthly, around the 12th" rule would show
        neither gap, would invent a November release that does not exist, and
        would misdate the two that were delayed.
        """
        gaps = {
            (a.local_date, b.local_date): (b.local_date - a.local_date).days
            for a, b in zip(cpi, cpi[1:], strict=False)
        }

        long_gaps = {pair: days for pair, days in gaps.items() if days > 40}
        assert long_gaps == {
            (dt.date(2025, 9, 11), dt.date(2025, 10, 24)): 43,
            (dt.date(2025, 10, 24), dt.date(2025, 12, 18)): 55,
        }

        for pair, days in gaps.items():
            if pair not in long_gaps:
                assert 20 <= days <= 40, f"unexpected gap of {days} days at {pair}"


class TestSelection:
    def test_filters_by_type_from_a_merged_calendar(self) -> None:
        merged = load_events()
        cutoff = dt.datetime(2026, 9, 18, tzinfo=UTC)
        selected = select_last_n(merged, 10, before=cutoff, event_type=EventType.CPI)

        assert len(selected) == 10
        assert {event.event_type for event in selected} == {EventType.CPI}

    def test_future_releases_are_excluded(self) -> None:
        cutoff = dt.datetime(2026, 9, 18, tzinfo=UTC)
        selected = select_last_n(
            load_events(EventType.CPI), 999, before=cutoff, event_type=EventType.CPI
        )
        assert all(event.timestamp_utc < cutoff for event in selected)
        assert "CPI-2026-12-10" not in {event.event_id for event in selected}
