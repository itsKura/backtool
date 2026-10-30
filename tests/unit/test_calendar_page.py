"""Tests for the calendar and event pages.

These cover the navigation layer: what gets listed, what a card says, and the
one correctness rule the event page adds -- that a page about a release must not
include that release in the history it presents as prior context.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest
from fastapi.testclient import TestClient

from backtool.core.types import EventType
from backtool.events import load_events
from backtool.events.meta import EVENT_META, Importance, meta_for
from backtool.research.windows import window_title
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

    def test_forecast_gap_is_shown_as_absent_not_as_a_substitute(
        self, client: TestClient
    ) -> None:
        """Substituting something else and labelling it 'forecast' is the exact
        silent redefinition this project exists to avoid. The row is rendered
        with the value missing rather than omitted, so a reader can see what is
        tracked and that there is nothing in it yet."""
        html = client.get("/event/FOMC-2026-06-17").text
        assert "Expected" in html
        assert 'class="expect-value">&mdash;' in html

    def test_a_cpi_event_shows_the_figures_it_has(
        self, client: TestClient
    ) -> None:
        """The row printed three dashes over figures sitting in the repository.
        Nobody noticed while the preparation page was a click away from every
        card; taking that button off past releases left the placeholder as the
        only thing a reader sees."""
        from backtool.events.releases import load_releases

        values = load_releases()["CPI-2026-09-11"]
        assert values.actual_yoy is not None
        html = client.get("/event/CPI-2026-09-11").text
        assert f"{values.actual_yoy:.1f}%" in html
        assert f"{values.previous_yoy:.1f}%" in html  # type: ignore[union-attr]

    def test_a_cpi_event_names_the_absence_it_has(
        self, client: TestClient
    ) -> None:
        """A figure we do not hold says which kind of absence it is, rather
        than leaving a reader to guess between "nobody published it" and "we
        have not got to it"."""
        html = client.get("/event/CPI-2026-10-14").text
        assert "no verified forecast recorded for this release" in html

    def test_an_untracked_event_type_borrows_no_wording_from_cpi(
        self, client: TestClient
    ) -> None:
        """The values file covers one event type. Giving an FOMC decision a
        row reading "Consensus -- not yet collected" would promise a dataset
        nobody is building."""
        html = client.get("/event/FOMC-2026-06-17").text
        assert "not yet collected for this release" not in html
        assert "Expected" in html

    def test_an_actual_that_cannot_exist_yet_gets_no_cell(
        self, client: TestClient
    ) -> None:
        """Before the release the row is two cells, not three.

        The third was a box that said "not published until the release" every
        single time anyone opened the page, which is the header's own state
        badge and countdown said a third time -- and a box that is always
        empty reads as broken rather than as scheduled.
        """
        html = client.get("/event/CPI-2026-11-10").text
        labels = re.findall(r'expect-label">([^<]*)', html)
        assert labels == ["Previous", "Consensus"]
        assert "not published until the release" not in html

    def test_the_actual_appears_once_the_release_has_happened(
        self, client: TestClient
    ) -> None:
        """Withheld until it exists, not dropped from the product. The whole
        dataset is built around this figure."""
        html = client.get("/event/CPI-2026-09-11").text
        assert re.findall(r'expect-label">([^<]*)', html) == [
            "Previous",
            "Consensus",
            "Actual",
        ]

    def test_an_untracked_event_type_follows_the_same_rule(
        self, client: TestClient
    ) -> None:
        """An FOMC row has no dataset behind any of its cells, which is a
        different absence from this one -- but "the figure published at the
        release instant" cannot exist before the release either way, and a
        reader looking at the two pages should not have to work out why one
        holds a box open and the other does not."""
        upcoming = client.get("/event/FOMC-2026-12-09").text
        assert re.findall(r'expect-label">([^<]*)', upcoming) == [
            "Previous",
            "Expected",
        ]
        past = client.get("/event/FOMC-2026-10-28").text
        assert re.findall(r'expect-label">([^<]*)', past) == [
            "Previous",
            "Expected",
            "Actual",
        ]

    def test_the_model_keeps_the_distinction_the_page_stops_drawing(
        self,
    ) -> None:
        """The reason an actual is absent is still carried, and still
        different either side of the release. Only the cell went away."""
        from backtool.events.releases import ValueStatus, release_figures

        before = {f.label: f for f in release_figures(None, released=False)}
        assert before["Actual"].status is ValueStatus.AWAITING_RELEASE
        after = {f.label: f for f in release_figures(None, released=True)}
        assert after["Actual"].status is not ValueStatus.AWAITING_RELEASE

    def test_no_development_terminology_reaches_the_page(
        self, client: TestClient
    ) -> None:
        """'Not yet wired up' and 'planned' describe the state of the codebase,
        which is not something a reader of a market page has any use for."""
        html = client.get("/event/FOMC-2026-06-17").text
        for phrase in ("not wired up", "Not yet wired up", "are planned", "TODO"):
            assert phrase not in html

    def test_history_loads_asynchronously(self, client: TestClient) -> None:
        """A cold cache means fetching candles for twenty events; the
        explanation should not wait on that."""
        response = client.get("/event/FOMC-2026-06-17")
        assert 'id="slot-summary"' in response.text
        assert "/api/event/FOMC-2026-06-17/history" in response.text

    def test_history_fills_three_separate_slots(self, client: TestClient) -> None:
        """Other sections sit between the statistics, the evidence table and the
        methodology, so the fragment cannot be injected as one block."""
        html = client.get("/event/FOMC-2026-06-17").text
        for slot in ("slot-summary", "slot-evidence", "slot-methodology"):
            assert f'id="{slot}"' in html


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
            "backtool.web.app.render_history", return_value="<h2>ok</h2>"
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
            "backtool.web.app.render_history", return_value=""
        ):
            TestClient(app).get(
                "/api/event/FOMC-2025-06-18/history?symbol=ETHUSDT"
            )

        assert captured["symbol"] == "ETHUSDT"

    def test_unknown_event_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/event/NOPE/history").status_code == 404


class TestEventPageOrder:
    """Price first, explainer last.

    Someone who clicked a CPI release came to see what price did, not to have
    CPI explained to them before they can reach the chart.
    """

    def test_price_chart_comes_before_the_explainer(self, client: TestClient) -> None:
        html = client.get("/event/FOMC-2026-09-16").text
        assert html.index("Price around the release") < html.index("About this event")

    def test_history_comes_before_the_explainer(self, client: TestClient) -> None:
        html = client.get("/event/FOMC-2026-09-16").text
        assert html.index('id="slot-summary"') < html.index("About this event")

    def test_findings_come_before_the_evidence_table(
        self, client: TestClient
    ) -> None:
        """What the numbers show is primary; the rows they were computed from
        are there to be checked, not to be read first."""
        html = client.get("/event/FOMC-2026-09-16").text
        assert html.index("Key findings") < html.index('id="slot-evidence"')

    def test_methodology_is_last(self, client: TestClient) -> None:
        html = client.get("/event/FOMC-2026-09-16").text
        assert html.index("About this event") < html.index('id="slot-methodology"')

    def test_explainer_is_still_present(self, client: TestClient) -> None:
        html = client.get("/event/FOMC-2026-09-16").text
        assert "What it is" in html
        assert "Why markets care" in html


class TestPriceChart:
    def test_past_event_loads_the_chart_library(self, client: TestClient) -> None:
        html = client.get("/event/FOMC-2026-09-16").text
        assert "lightweight-charts" in html
        assert 'id="price-chart"' in html

    def test_future_event_has_no_chart_and_says_why(self, client: TestClient) -> None:
        """An empty frame the reader has to interpret is worse than a sentence."""
        upcoming = next(
            event
            for event in load_events()
            if event.timestamp_utc > dt.datetime.now(tz=dt.UTC)
        )
        html = client.get(f"/event/{upcoming.event_id}").text
        assert "no candles around it to chart" in html
        assert "lightweight-charts" not in html

    def test_bands_come_from_the_measured_windows(self) -> None:
        """The shaded regions and the statistics must describe the same spans,
        so both are derived from DEFAULT_WINDOWS rather than restated."""
        from backtool.research.windows import DEFAULT_WINDOWS
        from backtool.web.event_page import chart_window_bands

        event_time = dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC)
        bands = chart_window_bands(event_time)

        assert len(bands) == len(DEFAULT_WINDOWS)
        for band, window in zip(bands, DEFAULT_WINDOWS, strict=True):
            assert band["start"] == int(
                (event_time + window.start_offset).timestamp()
            )
            assert band["end"] == int((event_time + window.end_offset).timestamp())

    def test_band_labels_are_never_window_names(self) -> None:
        from backtool.web.event_page import chart_window_bands

        bands = chart_window_bands(dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC))
        for band in bands:
            assert "_" not in str(band["title"])
            assert "_" not in str(band["label"])

    def test_bands_are_split_into_before_and_after(self) -> None:
        from backtool.web.event_page import chart_window_bands

        event_time = dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC)
        stamp = int(event_time.timestamp())
        for band in chart_window_bands(event_time):
            if band["phase"] == "before":
                assert band["end"] <= stamp
            else:
                assert band["start"] >= stamp

    def test_a_band_is_clipped_to_the_charted_span(self) -> None:
        """A window reaching past the last candle would otherwise shade empty
        canvas, which reads as data that is not there."""
        from backtool.web.event_page import CHART_SPAN, chart_window_bands

        event_time = dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC)
        low = int((event_time - CHART_SPAN).timestamp())
        high = int((event_time + CHART_SPAN).timestamp())
        for band in chart_window_bands(event_time):
            assert low <= int(str(band["start"])) <= high
            assert low <= int(str(band["end"])) <= high

    def test_candles_endpoint_ships_the_bands(self, client: TestClient) -> None:
        payload = client.get(
            "/api/event/FOMC-2026-09-16/candles?symbol=BTCUSDT"
        ).json()
        assert payload["windows"]
        first = payload["windows"][0]
        assert set(first) == {"label", "title", "start", "end", "phase", "high", "low"}

    def test_band_price_range_comes_from_the_research_layer(
        self, client: TestClient
    ) -> None:
        """A window's high is a statistic with a specific rule about which
        candles fall inside the window. Taking a max in the browser instead
        would put a second, untested definition of it in the codebase."""
        payload = client.get(
            "/api/event/FOMC-2026-09-16/candles?symbol=BTCUSDT"
        ).json()

        for band in payload["windows"]:
            assert band["high"] is not None, band["title"]
            assert band["low"] <= band["high"]

        # The 24h window contains the 1h window, so it cannot be the narrower
        # of the two in either direction.
        by_title = {b["title"]: b for b in payload["windows"]}
        wide, narrow = by_title["24 hours after"], by_title["1 hour after"]
        assert wide["high"] >= narrow["high"]
        assert wide["low"] <= narrow["low"]

    def test_band_prices_equal_what_the_metrics_layer_reports(self) -> None:
        """Not merely close to it: the same function, so there is exactly one
        definition of a window's high in the codebase."""
        from tests.conftest import CANDLE_START, EVENT_TIME, build_candles

        from backtool.core.types import Interval
        from backtool.research.metrics import compute_window_metrics
        from backtool.research.windows import DEFAULT_WINDOWS
        from backtool.web.event_page import chart_window_bands

        # A visible spike, so a band that silently used the wrong slice of
        # candles would report a different high rather than the same flat one.
        closes = [100.0] * 36
        closes[30] = 140.0
        frame = build_candles(CANDLE_START, closes)

        bands = chart_window_bands(EVENT_TIME, frame, Interval.M5)
        priced = {band["title"]: band for band in bands}

        for window in DEFAULT_WINDOWS:
            expected = compute_window_metrics(
                frame, window.resolve(EVENT_TIME), Interval.M5
            )
            band = priced[window_title(window)]
            if expected.is_usable:
                assert band["high"] == expected.high
                assert band["low"] == expected.low
            else:
                assert band["high"] is None

    def test_a_band_without_candles_falls_back_to_full_height(self) -> None:
        """No frame means no price range, and a box drawn to an invented one
        would be worse than a full-height band."""
        from backtool.web.event_page import chart_window_bands

        for band in chart_window_bands(dt.datetime(2026, 9, 16, 18, tzinfo=dt.UTC)):
            assert band["high"] is None
            assert band["low"] is None

    def test_chart_library_version_is_pinned(self) -> None:
        """v5 replaced addCandlestickSeries with addSeries(CandlestickSeries,
        ...); an unpinned range would break the page on the next major."""
        from backtool.web.event_page import LIGHTWEIGHT_CHARTS_CDN

        assert "lightweight-charts@5" in LIGHTWEIGHT_CHARTS_CDN

    def test_saved_report_stays_free_of_external_scripts(self) -> None:
        """The chart is a web-page feature. A saved report must still open
        offline, so the charting CDN must not leak into it."""
        import datetime as dtime

        from backtool.reporting.html import render_report
        from backtool.research.results import StudyResult
        from backtool.research.spec import ResearchSpec

        spec = ResearchSpec(
            event_count=1, as_of=dtime.datetime(2026, 1, 1, tzinfo=dtime.UTC)
        )
        study = StudyResult(
            spec=spec,
            generated_at=dtime.datetime(2026, 1, 1, tzinfo=dtime.UTC),
            events=(),
        )
        html = render_report(study)
        assert "<script" not in html
        assert "cdn.jsdelivr.net" not in html


class TestCandlesEndpoint:
    def test_returns_candles_centred_on_the_event(self, client: TestClient) -> None:
        payload = client.get(
            "/api/event/FOMC-2026-09-16/candles?symbol=BTCUSDT"
        ).json()

        assert payload["symbol"] == "BTCUSDT"
        assert payload["candles"], "no candles returned"

        first = payload["candles"][0]["time"]
        last = payload["candles"][-1]["time"]
        event = payload["event_time"]
        assert first < event < last
        # Roughly symmetric either side, within one candle.
        assert abs((event - first) - (last - event)) < 3600

    def test_candle_shape_matches_the_chart_library(
        self, client: TestClient
    ) -> None:
        candle = client.get(
            "/api/event/FOMC-2026-09-16/candles?symbol=BTCUSDT"
        ).json()["candles"][0]
        assert set(candle) == {"time", "open", "high", "low", "close"}
        assert isinstance(candle["time"], int)
        assert candle["high"] >= candle["low"]

    def test_future_event_returns_empty_with_a_reason(
        self, client: TestClient
    ) -> None:
        upcoming = next(
            event
            for event in load_events()
            if event.timestamp_utc > dt.datetime.now(tz=dt.UTC)
        )
        payload = client.get(f"/api/event/{upcoming.event_id}/candles").json()
        assert payload["candles"] == []
        assert "not happened yet" in payload["message"]

    def test_unknown_event_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/event/NOPE/candles").status_code == 404


class TestAiSummary:
    """The AI summary is on-demand, and never pretends to be a measurement."""

    def test_button_is_offered_when_configured(self) -> None:
        from unittest.mock import patch

        with patch("backtool.web.app._ai_available", return_value=True):
            html = TestClient(app).get("/event/FOMC-2026-09-16").text
        assert 'id="explain-btn"' in html
        assert "Summarise these results" in html

    def test_it_is_not_automatic(self) -> None:
        """Each summary is a paid call taking tens of seconds. A reader who came
        for the chart should not be billed for prose they did not ask for."""
        from unittest.mock import patch

        with patch("backtool.web.app._ai_available", return_value=True):
            html = TestClient(app).get("/event/FOMC-2026-09-16").text
        # The container exists but starts empty; nothing fires on load.
        assert 'id="explain-out"></div>' in html
        assert "addEventListener('click'" in html

    def test_absent_key_explains_rather_than_hiding(self) -> None:
        from unittest.mock import patch

        with patch("backtool.web.app._ai_available", return_value=False):
            html = TestClient(app).get("/event/FOMC-2026-09-16").text
        assert "ANTHROPIC_API_KEY" in html
        assert 'id="explain-btn"' not in html

    def test_summary_is_labelled_as_written_not_computed(self) -> None:
        """A reader must be able to tell at a glance which parts of the page
        were measured and which were written."""
        from unittest.mock import patch

        with patch("backtool.web.app._ai_available", return_value=True):
            html = TestClient(app).get("/event/FOMC-2026-09-16").text
        assert "computes nothing" in html

    def test_endpoint_refuses_without_a_key(self) -> None:
        from unittest.mock import patch

        with patch("backtool.web.app._ai_available", return_value=False):
            response = TestClient(app).get("/api/event/FOMC-2026-09-16/explain")
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.text

    def test_endpoint_renders_the_findings_card(self) -> None:
        """The headline leads, the model's own words are used verbatim, and the
        attribution travels with them."""
        from unittest.mock import patch

        from backtool.ai.models import Interpretation

        fake = Interpretation(
            headline="A headline with 1.234% in it.",
            findings=["A finding."],
            caveats=["A caveat."],
            suggested_followups=["A follow-up."],
        )
        with patch("backtool.web.app._ai_available", return_value=True), patch(
            "backtool.web.app.run_study", return_value=object()
        ), patch("backtool.ai.interpret_study", return_value=fake):
            response = TestClient(app).get("/api/event/FOMC-2026-09-16/explain")

        assert response.status_code == 200
        assert "A headline with 1.234% in it." in response.text
        assert 'class="finding-headline"' in response.text
        assert "It computed none of them." in response.text
        assert "Explore further" in response.text

    def test_failure_is_reported_not_swallowed(self) -> None:
        from unittest.mock import patch

        from backtool.ai import AIError

        with patch("backtool.web.app._ai_available", return_value=True), patch(
            "backtool.web.app.run_study", return_value=object()
        ), patch("backtool.ai.interpret_study", side_effect=AIError("model down")):
            response = TestClient(app).get("/api/event/FOMC-2026-09-16/explain")

        assert response.status_code == 502
        assert "model down" in response.text
