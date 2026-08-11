"""Tests for the event model and the FOMC calendar loader.

Includes structural invariants over the shipped calendar data. These catch
transcription errors (a date typed into the wrong month, a duplicated row) but
they cannot catch a date that is simply the wrong Wednesday -- that requires
verification against the Federal Reserve's published calendar.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from backtool.core.time import UTC
from backtool.core.types import EventType
from backtool.events import EventDataError, MarketEvent, load_fomc_events, select_last_n


def make_event(
    event_id: str = "FOMC-TEST",
    local_date: dt.date = dt.date(2024, 1, 31),
    local_time: dt.time = dt.time(14, 0),
    timezone: str = "America/New_York",
    is_scheduled: bool = True,
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        event_type=EventType.FOMC,
        local_date=local_date,
        local_time=local_time,
        timezone=timezone,
        is_scheduled=is_scheduled,
    )


class TestMarketEvent:
    def test_derives_utc_from_local_time_in_winter(self) -> None:
        event = make_event(local_date=dt.date(2024, 1, 31))
        assert event.timestamp_utc == dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)

    def test_derives_utc_from_local_time_in_summer(self) -> None:
        event = make_event(local_date=dt.date(2024, 7, 31))
        assert event.timestamp_utc == dt.datetime(2024, 7, 31, 18, 0, tzinfo=UTC)

    def test_rejects_unknown_timezone(self) -> None:
        with pytest.raises(ValidationError, match="Unknown IANA timezone"):
            make_event(timezone="America/Atlantis")

    def test_is_immutable(self) -> None:
        event = make_event()
        with pytest.raises(ValidationError):
            event.local_time = dt.time(15, 0)  # type: ignore[misc]

    def test_parses_string_fields_from_csv_row(self) -> None:
        """The loader hands raw CSV strings straight to the model."""
        event = MarketEvent.model_validate(
            {
                "event_id": "FOMC-2024-01-31",
                "event_type": "FOMC",
                "local_date": "2024-01-31",
                "local_time": "14:00",
                "timezone": "America/New_York",
                "is_scheduled": "true",
                "notes": "",
            }
        )
        assert event.is_scheduled is True
        assert event.local_time == dt.time(14, 0)
        assert event.timestamp_utc == dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def calendar() -> list[MarketEvent]:
    """The real shipped FOMC calendar, loaded once for the whole module."""
    return load_fomc_events()


class TestLoadFomcEvents:
    @pytest.fixture
    def events(self, calendar: list[MarketEvent]) -> list[MarketEvent]:
        return calendar

    def test_loads_a_plausible_number_of_events(self, events: list[MarketEvent]) -> None:
        assert len(events) > 50

    def test_sorted_ascending_by_announcement_instant(self, events: list[MarketEvent]) -> None:
        timestamps = [event.timestamp_utc for event in events]
        assert timestamps == sorted(timestamps)

    def test_event_ids_are_unique(self, events: list[MarketEvent]) -> None:
        ids = [event.event_id for event in events]
        assert len(ids) == len(set(ids))

    def test_event_id_matches_local_date(self, events: list[MarketEvent]) -> None:
        for event in events:
            assert event.event_id == f"FOMC-{event.local_date.isoformat()}"

    def test_scheduled_meetings_land_on_tuesday_wednesday_or_thursday(
        self, events: list[MarketEvent]
    ) -> None:
        """Scheduled FOMC statements are released on the second day of a
        two-day meeting, which lands midweek. An unscheduled emergency meeting
        can land anywhere, including a Sunday, so it is exempt."""
        for event in events:
            if event.is_scheduled:
                assert event.local_date.weekday() in {1, 2, 3}, (
                    f"{event.event_id} falls on "
                    f"{event.local_date.strftime('%A')}, which is unexpected "
                    "for a scheduled meeting"
                )

    def test_scheduled_meetings_are_released_at_14_00_eastern(
        self, events: list[MarketEvent]
    ) -> None:
        """Statement releases moved to 2:00 PM ET in 2013; every event in this
        calendar postdates that change."""
        for event in events:
            if event.is_scheduled:
                assert event.local_time == dt.time(14, 0), event.event_id
                assert event.timezone == "America/New_York", event.event_id

    #: The FOMC holds eight scheduled meetings a year. 2020 is the one
    #: exception in this calendar: the scheduled 17-18 March meeting was
    #: cancelled, its business folded into the unscheduled 15 March Sunday
    #: emergency action, leaving seven scheduled meetings that year.
    EXPECTED_SCHEDULED_PER_YEAR = {2020: 7}

    def test_expected_scheduled_meetings_per_completed_year(
        self, events: list[MarketEvent]
    ) -> None:
        by_year: dict[int, int] = {}
        for event in events:
            if event.is_scheduled:
                by_year[event.local_date.year] = by_year.get(event.local_date.year, 0) + 1

        # Skip the first and last year in the file, which may be partial.
        for year in sorted(by_year)[1:-1]:
            expected = self.EXPECTED_SCHEDULED_PER_YEAR.get(year, 8)
            assert by_year[year] == expected, (
                f"{year} has {by_year[year]} scheduled meetings, expected {expected}"
            )

    def test_march_2020_emergency_meetings_are_marked_unscheduled(
        self, events: list[MarketEvent]
    ) -> None:
        """Guards the exception above: if these rows were ever "corrected" to
        scheduled, the per-year count test would start passing for the wrong
        reason."""
        unscheduled = {event.event_id for event in events if not event.is_scheduled}
        assert unscheduled == {"FOMC-2020-03-03", "FOMC-2020-03-15"}

    @pytest.mark.parametrize(
        ("event_id", "expected_utc"),
        [
            # "For release at 10:00 a.m. EST" -> EST is UTC-5 -> 15:00 UTC.
            ("FOMC-2020-03-03", dt.datetime(2020, 3, 3, 15, 0, tzinfo=UTC)),
            # "For release at 5:00 p.m. EDT" -> EDT is UTC-4 -> 21:00 UTC.
            # US DST began 2020-03-08, so 15 March was already on EDT.
            ("FOMC-2020-03-15", dt.datetime(2020, 3, 15, 21, 0, tzinfo=UTC)),
        ],
    )
    def test_emergency_announcement_times_match_the_press_releases(
        self, events: list[MarketEvent], event_id: str, expected_utc: dt.datetime
    ) -> None:
        """Verified against federalreserve.gov press releases; see
        docs/data-sources.md. These are the only two events in the calendar not
        released at 14:00 ET."""
        event = next(event for event in events if event.event_id == event_id)
        assert event.timestamp_utc == expected_utc

    def test_every_event_resolves_to_an_unambiguous_utc_instant(
        self, events: list[MarketEvent]
    ) -> None:
        for event in events:
            assert event.timestamp_utc.tzinfo is not None

    def test_raises_on_malformed_row(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text(
            "event_id,event_type,local_date,local_time,timezone,is_scheduled,notes\n"
            "FOMC-X,FOMC,not-a-date,14:00,America/New_York,true,\n",
            encoding="utf-8",
        )
        with pytest.raises(EventDataError, match="line 2"):
            load_fomc_events(path)

    def test_raises_on_duplicate_event_id(self, tmp_path) -> None:
        path = tmp_path / "dupes.csv"
        row = "FOMC-2024-01-31,FOMC,2024-01-31,14:00,America/New_York,true,\n"
        path.write_text(
            "event_id,event_type,local_date,local_time,timezone,is_scheduled,notes\n"
            + row
            + row,
            encoding="utf-8",
        )
        with pytest.raises(EventDataError, match="Duplicate event id"):
            load_fomc_events(path)


class TestSelectLastN:
    @pytest.fixture
    def events(self) -> list[MarketEvent]:
        return [
            make_event("FOMC-2024-01-31", dt.date(2024, 1, 31)),
            make_event("FOMC-2024-03-20", dt.date(2024, 3, 20)),
            make_event("FOMC-2024-05-01", dt.date(2024, 5, 1)),
            make_event("FOMC-2024-06-12", dt.date(2024, 6, 12)),
        ]

    def test_returns_the_n_most_recent_in_ascending_order(
        self, events: list[MarketEvent]
    ) -> None:
        selected = select_last_n(events, 2, before=dt.datetime(2025, 1, 1, tzinfo=UTC))
        assert [event.event_id for event in selected] == ["FOMC-2024-05-01", "FOMC-2024-06-12"]

    def test_excludes_events_at_or_after_the_cutoff(self, events: list[MarketEvent]) -> None:
        """A future-dated calendar entry must never enter an analysis."""
        selected = select_last_n(events, 10, before=dt.datetime(2024, 4, 1, tzinfo=UTC))
        assert [event.event_id for event in selected] == [
            "FOMC-2024-01-31",
            "FOMC-2024-03-20",
        ]

    def test_cutoff_boundary_is_exclusive(self, events: list[MarketEvent]) -> None:
        """An event exactly at the cutoff has not finished being observable."""
        exact = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert select_last_n(events, 10, before=exact) == []

    def test_returns_fewer_than_requested_without_raising(
        self, events: list[MarketEvent]
    ) -> None:
        selected = select_last_n(events, 99, before=dt.datetime(2025, 1, 1, tzinfo=UTC))
        assert len(selected) == 4

    def test_accepts_non_utc_cutoff(self, events: list[MarketEvent]) -> None:
        from backtool.core.time import NEW_YORK

        selected = select_last_n(
            events, 10, before=dt.datetime(2024, 3, 31, 20, 0, tzinfo=NEW_YORK)
        )
        assert len(selected) == 2

    def test_rejects_naive_cutoff(self, events: list[MarketEvent]) -> None:
        with pytest.raises(ValueError, match="Naive datetime"):
            select_last_n(events, 1, before=dt.datetime(2025, 1, 1))

    def test_rejects_non_positive_n(self, events: list[MarketEvent]) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            select_last_n(events, 0)

    def test_defaults_cutoff_to_now(self, events: list[MarketEvent]) -> None:
        assert len(select_last_n(events, 10)) == 4
