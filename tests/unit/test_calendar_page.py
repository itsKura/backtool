"""Tests for the calendar and event pages.

These cover the navigation layer: what gets listed, what a card says, and the
one correctness rule the event page adds -- that a page about a release must not
include that release in the history it presents as prior context.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from backtool.core.types import EventType
from backtool.events import load_events
from backtool.events.meta import EVENT_META, Importance, meta_for
from backtool.web.app import DEFAULT_HISTORY_EVENTS, SYMBOLS, app
from backtool.web.calendar_page import PAST_LIMIT, format_delta, render_calendar

NOW = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def events() -> list:  # type: ignore[type-arg]
    return load_events()


class TestEventMetadata:
    def test_every_event_type_has_metadata(self) -> None:
        """A type without metadata renders an event page with no explanation
        on it, which would only be noticed by looking."""
        assert set(EVENT_META) == set(EventType)

    def test_metadata_is_populated(self) -> None:
        for kind in EventType:
            meta = meta_for(kind)
            assert meta.label and meta.source and meta.release_time
            assert len(meta.what_it_is) > 80
            assert len(meta.why_markets_care) > 80
            assert meta.what_to_watch

    def test_importance_is_a_known_level(self) -> None:
        for kind in EventType:
            assert meta_for(kind).importance in set(Importance)


class TestCalendarRendering:
    def test_splits_upcoming_from_past(self, events: list) -> None:  # type: ignore[type-arg]
        html = render_calendar(events, now=NOW)
        assert "<h2>Upcoming</h2>" in html
        assert "<h2>Recent</h2>" in html
        assert html.index("Upcoming") < html.index("Recent")

    def test_caps_the_past_list(self, events: list) -> None:  # type: ignore[type-arg]
        """A calendar is for triage, not an archive."""
        html = render_calendar(events, now=NOW)
        past_section = html[html.index("<h2>Recent</h2>") :]
        assert past_section.count("event-card") <= PAST_LIMIT

    def test_cards_link_to_their_event(self, events: list) -> None:  # type: ignore[type-arg]
        html = render_calendar(events, now=NOW)
        assert 'href="/event/FOMC-' in html or 'href="/event/CPI-' in html

    def test_rescheduled_events_are_flagged(self, events: list) -> None:  # type: ignore[type-arg]
        """An irregular release is exactly the one not to assume behaved
        normally, so it is marked on the card itself."""
        html = render_calendar(events, now=dt.datetime(2026, 3, 1, tzinfo=dt.UTC))
        assert "rescheduled" in html

    def test_empty_calendar_does_not_crash(self) -> None:
        html = render_calendar([], now=NOW)
        assert "No scheduled events ahead" in html
        assert "No past events" in html


class TestFormatDelta:
    @pytest.mark.parametrize(
        ("delta", "upcoming", "expected"),
        [
            (dt.timedelta(days=22), True, "in 3 weeks"),
            (dt.timedelta(days=7), True, "in 7 days"),
            (dt.timedelta(days=1), True, "in 1 day"),
            (dt.timedelta(hours=5), True, "in 5 hours"),
            (dt.timedelta(minutes=30), True, "in 30 minutes"),
            (dt.timedelta(days=-14), False, "2 weeks ago"),
            (dt.timedelta(days=-3), False, "3 days ago"),
        ],
    )
    def test_reads_as_a_human_phrase(
        self, delta: dt.timedelta, upcoming: bool, expected: str
    ) -> None:
        assert format_delta(delta, upcoming) == expected

    def test_never_says_zero_minutes(self) -> None:
        assert format_delta(dt.timedelta(seconds=10), True) == "in 1 minute"


class TestEventRoutes:
    def test_serves_a_known_event(self, client: TestClient) -> None:
        response = client.get("/event/FOMC-2026-06-17")
        assert response.status_code == 200
        assert "Why markets care" in response.text
        assert "What to watch" in response.text

    def test_unknown_event_is_a_404(self, client: TestClient) -> None:
        assert client.get("/event/FOMC-1999-01-01").status_code == 404

    def test_offers_both_symbols(self, client: TestClient) -> None:
        response = client.get("/event/FOMC-2026-06-17")
        for symbol in SYMBOLS:
            assert symbol in response.text

    def test_an_unknown_symbol_falls_back_rather_than_erroring(
        self, client: TestClient
    ) -> None:
        response = client.get("/event/FOMC-2026-06-17?symbol=DOGEUSDT")
        assert response.status_code == 200
        assert "symbol=BTCUSDT" in response.text

    def test_future_event_says_it_has_not_happened(self, client: TestClient) -> None:
        """Without this, the history below reads as a forecast of the event."""
        upcoming = next(
            event
            for event in load_events()
            if event.timestamp_utc > dt.datetime.now(tz=dt.UTC)
        )
        response = client.get(f"/event/{upcoming.event_id}")
        assert "has not happened yet" in response.text
        assert "record, not a forecast" in response.text

    def test_forecast_gap_is_stated_not_hidden(self, client: TestClient) -> None:
        """Substituting something else and labelling it 'forecast' is the exact
        silent redefinition this project exists to avoid."""
        response = client.get("/event/FOMC-2026-06-17")
        assert "no free licence-clean source" in response.text

    def test_history_loads_asynchronously(self, client: TestClient) -> None:
        """A cold cache means fetching candles for twenty events; the
        explanation should not wait on that."""
        response = client.get("/event/FOMC-2026-06-17")
        assert 'id="history"' in response.text
        assert "/api/event/FOMC-2026-06-17/history" in response.text


class TestEventHistory:
    def test_excludes_the_event_itself(self) -> None:
        """A page about one release must not include that release in the
        history it presents as prior context -- that is the event's own outcome
        leaking into its own background."""
        from unittest.mock import patch

        captured: dict = {}

        def capture(spec, service, calendar, *, now=None):  # type: ignore[no-untyped-def]
            captured["as_of"] = spec.as_of
            captured["symbol"] = spec.symbol
            captured["count"] = spec.event_count
            return object()

        event = next(e for e in load_events() if e.event_id == "FOMC-2025-06-18")

        with patch("backtool.web.app.run_study", side_effect=capture), patch(
            "backtool.web.app.render_body", return_value="<h2>ok</h2>"
        ):
            client = TestClient(app)
            response = client.get("/api/event/FOMC-2025-06-18/history")

        assert response.status_code == 200
        assert captured["as_of"] == event.timestamp_utc
        assert captured["count"] == DEFAULT_HISTORY_EVENTS

    def test_honours_the_symbol(self) -> None:
        from unittest.mock import patch

        captured: dict = {}

        def capture(spec, service, calendar, *, now=None):  # type: ignore[no-untyped-def]
            captured["symbol"] = spec.symbol
            return object()

        with patch("backtool.web.app.run_study", side_effect=capture), patch(
            "backtool.web.app.render_body", return_value=""
        ):
            TestClient(app).get(
                "/api/event/FOMC-2025-06-18/history?symbol=ETHUSDT"
            )

        assert captured["symbol"] == "ETHUSDT"

    def test_unknown_event_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/event/NOPE/history").status_code == 404
