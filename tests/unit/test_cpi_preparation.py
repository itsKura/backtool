"""The CPI preparation page: what a trader is shown before a release.

Product behaviour, not plumbing. What these protect is the set of claims the
page makes implicitly by putting numbers next to each other -- that a cohort
was selected by something knowable beforehand, that a figure which cannot be
computed is absent rather than zero, that the asset tab changes what was
measured rather than only what is labelled, and that none of it costs money.

The look-ahead tests are the ones worth having. A cohort chosen by its own
outcome produces a spectacular statistic, renders identically to an honest one,
and is wrong in a way no reader could detect from the page.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backtool.core.types import EventType, Interval
from backtool.events import load_events
from backtool.events.releases import (
    SCENARIO_RULES,
    ReleaseValues,
    classify_cpi,
    load_releases,
)
from backtool.research.aggregate import aggregate_window
from backtool.research.cohort import split_study
from backtool.research.metrics import Direction, WindowMetrics, WindowStatus
from backtool.research.preparation import (
    CONTEXT_ENTRY_WINDOW,
    ENTERED_DOWN,
    ENTERED_UP,
    ENTRY_WINDOW,
    MarketContext,
    baseline_cohort,
    build_market_context,
    cohort_for_entry,
    entry_cohorts,
    intersection_cohorts,
)
from backtool.research.results import EventResult, StudyResult
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.app import app
from backtool.web.prepare_page import (
    HEADLINE_WINDOW,
    SCENARIO_LABELS,
    WINDOW_LABELS,
    Preparation,
    _reaction_for,
    build_preparation,
    render_preparation,
    what_to_watch_lines,
)

#: A release that has long since happened, with candles either side.
PAST_EVENT = "CPI-2026-09-11"
#: The next one on the calendar, which is what the Prepare button opens.
NEXT_EVENT = "CPI-2026-11-10"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def no_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to reach a model an immediate, loud failure.

    Every AI path in this project goes through one client, so refusing to build
    one is a chokepoint rather than a list of functions to keep in step with.
    A test that merely *happened* not to call a model would keep passing after
    somebody added a call.
    """
    import backtool.ai.client as client_module

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the preparation page must not call a model")

    monkeypatch.setattr(client_module.AIClient, "__init__", refuse)


def plain(html: str) -> str:
    """Page text with the tags taken out, for asserting on what is read."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&mdash;", "-")
    return re.sub(r"\s+", " ", text)


def window(name: str, value: float | None, *, ok: bool = True) -> WindowMetrics:
    """One window's metrics, hand-built so a test can state its own data.

    Whole rather than partial: the model refuses to be OK while missing the
    fields that make it usable, which is the invariant doing its job. The
    prices are derived from the return so the row is internally consistent --
    a window whose end price contradicted its own return would be a fixture
    that could never occur, and a test built on one proves nothing.
    """
    if value is None or not ok:
        return WindowMetrics(
            name=name,
            start_utc=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            end_utc=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
            status=WindowStatus.INSUFFICIENT_DATA,
            reason="no candles in this window",
        )

    start = 100.0
    end = start * (1 + value / 100)
    return WindowMetrics(
        name=name,
        start_utc=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        end_utc=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        status=WindowStatus.OK,
        start_price=start,
        end_price=end,
        return_pct=value,
        abs_move_pct=abs(value),
        direction=(
            Direction.UP if value > 0 else Direction.DOWN if value < 0 else Direction.FLAT
        ),
        high=max(start, end),
        low=min(start, end),
        mfe_pct=max(value, 0.0),
        mae_pct=min(value, 0.0),
        range_pct=abs(value),
    )


def synthetic_study(rows: list[tuple[str, float, float]]) -> StudyResult:
    """A study from ``(event_id, entry_return, outcome_return)`` triples.

    Entry and outcome are set independently on purpose: every look-ahead test
    below works by making them disagree, so a rule that selected on the outcome
    would pick a visibly different set from one that selected on the entry.
    """
    events = tuple(
        EventResult(
            event_id=event_id,
            event_time_utc=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            local_date=dt.date(2026, 1, 1),
            is_scheduled=True,
            windows={
                ENTRY_WINDOW.name: window(ENTRY_WINDOW.name, entry),
                HEADLINE_WINDOW: window(HEADLINE_WINDOW, outcome),
            },
        )
        for event_id, entry, outcome in rows
    )
    return StudyResult(
        spec=ResearchSpec(
            symbol="BTCUSDT",
            event_type=EventType.CPI,
            # What was asked for, not what came back: a study that found
            # nothing still asked for something, and the spec rejects zero.
            event_count=max(len(rows), 1),
            interval=Interval.M5,
            windows=(*REACTION_WINDOWS, ENTRY_WINDOW),
        ),
        generated_at=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        events=events,
    )


# --- A, B: the page opens, and opens for nothing ---------------------------


class TestThePageOpensFromTheCalendar:
    def test_the_calendar_offers_prepare_on_a_cpi_release(
        self, client: TestClient
    ) -> None:
        """The entry point. A trader should not have to know a URL, and should
        not have to start in the chatbot."""
        html = client.get("/").text
        assert "/prepare" in html
        assert ">Prepare</a>" in html

    def test_it_is_offered_only_before_the_release(
        self, client: TestClient
    ) -> None:
        """Because that is what the word means. A release that has printed is
        read about rather than prepared for, and every card already leads to
        the event page where that reading lives.
        """
        html = client.get("/").text
        now = dt.datetime.now(tz=dt.UTC)
        upcoming = {
            item.event_id
            for item in load_events(EventType.CPI)
            if item.timestamp_utc > now
        }
        offered = set(re.findall(r'href="/event/([^"]+)/prepare"', html))
        assert offered, "the next release should still be offered"
        assert offered <= upcoming, f"offered for a release that has happened: {offered - upcoming}"

    def test_a_past_release_is_still_reachable_by_its_own_url(
        self, client: TestClient
    ) -> None:
        """The button is gone, not the page. A link kept from last month goes
        on working, and what it shows afterwards was built deliberately."""
        assert client.get(f"/event/{PAST_EVENT}/prepare").status_code == 200

    def test_an_upcoming_release_renders(self, client: TestClient) -> None:
        response = client.get(f"/event/{NEXT_EVENT}/prepare")
        assert response.status_code == 200
        text = plain(response.text)
        assert "Event preparation" in text
        assert "Upcoming" in text

    def test_a_past_release_renders(self, client: TestClient) -> None:
        response = client.get(f"/event/{PAST_EVENT}/prepare")
        assert response.status_code == 200
        assert "Released" in plain(response.text)

    def test_the_prepare_link_is_only_on_releases_that_have_one(
        self, client: TestClient
    ) -> None:
        """The page reads a CPI values file and a CPI scenario rule. Offering
        it for an event type with neither would promise a page that the route
        correctly refuses to render."""
        html = client.get("/").text
        for match in re.findall(r'href="/event/([^"]+)/prepare"', html):
            assert match.startswith("CPI-")

    def test_a_non_cpi_event_is_refused_rather_than_half_rendered(
        self, client: TestClient
    ) -> None:
        fomc = next(
            event
            for event in load_events(EventType.FOMC)
            if event.curated_type is EventType.FOMC
        )
        assert client.get(f"/event/{fomc.event_id}/prepare").status_code == 404

    def test_rendering_costs_nothing(self, client: TestClient, no_ai: None) -> None:
        """B and P together. The whole page is a dataset, a candle cache and a
        comparison; a model involved in rendering it would be charging a reader
        to be told what is already in the repository."""
        assert client.get(f"/event/{NEXT_EVENT}/prepare").status_code == 200
        assert client.get(f"/event/{PAST_EVENT}/prepare").status_code == 200


# --- C, D: the scenario rule is the published comparison -------------------


class TestAScenarioIsActualAgainstConsensus:
    """Not a score, a threshold or a model: a comparison a reader can redo."""

    def cohort(self, actual: float, consensus: float) -> str | None:
        values = ReleaseValues(
            event_id="X",
            actual_yoy=actual,
            consensus_yoy=consensus,
            consensus_verified=True,
            first_release_verified=True,
        )
        classified = classify_cpi(values)
        return classified[0] if classified else None

    def test_cooler_is_actual_below_consensus(self) -> None:
        assert self.cohort(2.9, 3.0) == "cooler"

    def test_hotter_is_actual_above_consensus(self) -> None:
        assert self.cohort(3.1, 3.0) == "hotter"

    def test_equal_is_in_line(self) -> None:
        assert self.cohort(3.0, 3.0) == "in_line"

    def test_the_page_sorts_real_releases_the_same_way(self) -> None:
        """The rule above, exercised through the page's own split rather than
        restated. A cohort whose members contradict their label would be the
        most convincing wrong page this product could produce."""
        prep = build_preparation(
            _event(PAST_EVENT), symbol="BTCUSDT", now=_now()
        )
        values = load_releases()
        for cohort in prep.scenarios:
            for event_id in cohort.event_ids:
                release = values[event_id]
                assert release.actual_yoy is not None
                assert release.consensus_yoy is not None
                gap = release.surprise_pp
                assert gap is not None
                if cohort.key == "cooler":
                    assert gap < 0, event_id
                elif cohort.key == "hotter":
                    assert gap > 0, event_id
                else:
                    assert gap == 0, event_id


# --- E, F: the numbers are the engine's, and the baseline is real ----------


class TestTheNumbersComeFromTheEngine:
    def test_a_cohort_statistic_is_the_aggregate_of_its_members(self) -> None:
        """E. Nothing on the page recomputes a median; the cohort carries what
        aggregate_window produced, and this re-derives one from the same study
        to show the page is not holding a second opinion."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        assert prep.study is not None
        cohort = next(c for c in prep.scenarios if c.sample_size)
        chosen = {event_id for event_id in cohort.event_ids}

        subset = prep.study.model_copy(
            update={
                "events": tuple(
                    event for event in prep.study.events if event.event_id in chosen
                )
            }
        )
        expected = aggregate_window(subset, HEADLINE_WINDOW)
        assert cohort.windows[HEADLINE_WINDOW] == expected

    def test_the_baseline_is_every_measured_release(self) -> None:
        """F. The comparison a scenario is read against has to be the whole
        set, or the difference quoted beside it is against nothing in
        particular."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        assert prep.study is not None
        assert prep.baseline is not None
        assert prep.baseline.sample_size == len(prep.study.events)
        assert prep.baseline.windows[HEADLINE_WINDOW] == aggregate_window(
            prep.study, HEADLINE_WINDOW
        )

    def test_the_baseline_is_not_one_of_the_scenarios(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        assert prep.baseline is not None
        sizes = {cohort.sample_size for cohort in prep.scenarios}
        assert prep.baseline.sample_size >= max(sizes)

    def test_the_difference_shown_is_the_two_medians_subtracted(self) -> None:
        """The page's only arithmetic, checked against the numbers it claims to
        be subtracting."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        assert prep.baseline is not None
        cohort = next(c for c in prep.scenarios if c.windows[HEADLINE_WINDOW].sample_size)
        gap = (
            cohort.windows[HEADLINE_WINDOW].median_return_pct
            - prep.baseline.windows[HEADLINE_WINDOW].median_return_pct
        )
        assert f"{gap:+.1f} pp vs all" in render_preparation(prep, ("BTCUSDT",))


# --- G: the asset tab changes the measurement ------------------------------


class TestTheAssetSelectionIsReal:
    def test_the_two_assets_do_not_share_numbers(self) -> None:
        """G. The failure this rules out is a page that relabels itself while
        showing the figures it already had."""
        btc = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        eth = build_preparation(_event(PAST_EVENT), symbol="ETHUSDT", now=_now())

        assert btc.study is not None and eth.study is not None
        assert btc.study.spec.symbol == "BTCUSDT"
        assert eth.study.spec.symbol == "ETHUSDT"
        assert btc.context.price != eth.context.price

        btc_median = btc.baseline.windows[HEADLINE_WINDOW].median_return_pct  # type: ignore[union-attr]
        eth_median = eth.baseline.windows[HEADLINE_WINDOW].median_return_pct  # type: ignore[union-attr]
        assert btc_median != eth_median

    def test_the_page_names_the_asset_it_measured(self, client: TestClient) -> None:
        html = client.get(f"/event/{PAST_EVENT}/prepare?symbol=ETHUSDT").text
        assert "ETHUSDT" in html
        assert 'href="/event/' + PAST_EVENT + '/prepare?symbol=BTCUSDT"' in html

    def test_an_unknown_symbol_falls_back_rather_than_failing(
        self, client: TestClient
    ) -> None:
        response = client.get(f"/event/{PAST_EVENT}/prepare?symbol=DOGEUSDT")
        assert response.status_code == 200
        assert "BTCUSDT" in response.text


# --- H, I: the cohort is selected by what was knowable beforehand ----------


class TestTheSimilarCohortCannotSeeTheFuture:
    """The rule whose violation would be invisible.

    Every study below sets the entry return and the outcome independently, so a
    selection made on the wrong one picks a visibly different set. A rule that
    peeked would look exactly like this one on the page.
    """

    def study(self) -> StudyResult:
        # Entry down, outcome up -- and entry up, outcome down. Any rule that
        # selected on the outcome would return the opposite membership.
        return synthetic_study(
            [
                ("down-then-up-1", -3.0, +5.0),
                ("down-then-up-2", -1.0, +4.0),
                ("up-then-down-1", +2.0, -6.0),
                ("up-then-down-2", +4.0, -2.0),
            ]
        )

    def test_membership_follows_the_entry_not_the_outcome(self) -> None:
        cohorts = {cohort.key: cohort for cohort in entry_cohorts(self.study())}
        assert set(cohorts["entered_down"].event_ids) == {
            "down-then-up-1",
            "down-then-up-2",
        }
        assert set(cohorts["entered_up"].event_ids) == {
            "up-then-down-1",
            "up-then-down-2",
        }

    def test_the_selecting_window_ends_at_the_release(self) -> None:
        """H, structurally. The condition is a window that closes when the
        figure prints, so there is no instant inside it the market had not
        already seen."""
        assert ENTRY_WINDOW.end == "0h"
        assert ENTRY_WINDOW.start.startswith("-")

    def test_today_is_matched_to_the_cohort_it_resembles(self) -> None:
        context = _context(-2.0)
        cohort = cohort_for_entry(self.study(), context)
        assert cohort is not None
        assert cohort.key == "entered_down"

    def test_an_unmeasurable_today_matches_nothing(self) -> None:
        """Showing the larger group instead would answer a question nobody
        asked, with a number that looks like an answer to theirs."""
        assert cohort_for_entry(self.study(), _context(None)) is None

    def test_a_release_with_no_measurable_entry_is_declined(self) -> None:
        """An unmeasurable day before a release is not a flat one, and a
        cohort that quietly absorbed it would be reporting on a condition that
        was never checked."""
        study = synthetic_study([("a", -1.0, 1.0)])
        broken = study.model_copy(
            update={
                "events": (
                    study.events[0].model_copy(
                        update={
                            "windows": {
                                ENTRY_WINDOW.name: window(
                                    ENTRY_WINDOW.name, None, ok=False
                                ),
                                HEADLINE_WINDOW: window(HEADLINE_WINDOW, 1.0),
                            }
                        }
                    ),
                )
            }
        )
        assert entry_cohorts(broken) == []

    def test_the_page_and_the_cohort_agree_on_today(self) -> None:
        """A panel describing today as down beside a cohort selected as up
        would be two correct halves making one wrong claim."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        if prep.context.entry_direction is None or prep.entry_match is None:
            pytest.skip("no measurable run-up in the cache right now")
        expected = (
            "entered_down" if prep.context.entry_direction == "down" else "entered_up"
        )
        assert prep.entry_match.key == expected


class TestWhichTwentyFourHoursCount:
    """Before the print and after it, "entering" means different windows.

    Before, the run-up to now is the best estimate of a window that has not
    closed. After, the release has its own measured 24 hours, and going on
    quoting the run-up to now would describe a day that mostly happened *after*
    the event while calling it the day before -- and then select the comparable
    cohort with it.
    """

    def test_before_the_print_it_is_the_run_up_to_now(self) -> None:
        prep = build_preparation(_event(NEXT_EVENT), symbol="BTCUSDT", now=_now())
        assert not prep.released
        assert prep.entry_return_pct == prep.context.entry_return_pct

    def test_after_the_print_it_is_the_release_own_window(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        assert prep.released
        assert prep.study is not None
        own = next(
            item for item in prep.study.events if item.event_id == PAST_EVENT
        )
        assert prep.entry_return_pct == own.windows[ENTRY_WINDOW.name].return_pct

    def test_the_tense_follows_the_release(self) -> None:
        """A page that says a release "is entering" nine hours after it printed
        is describing something that already finished."""
        upcoming = plain(
            render_preparation(
                build_preparation(_event(NEXT_EVENT), symbol="BTCUSDT", now=_now()),
                ("BTCUSDT",),
            )
        )
        past = plain(
            render_preparation(
                build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now()),
                ("BTCUSDT",),
            )
        )
        assert "is entering this release" in upcoming
        assert "is entering this release" not in past
        assert "went into this release" in past

    def test_a_release_is_not_compared_against_itself(self) -> None:
        """A comparison group containing the thing it explains is not a
        comparison. It would also pull its own outcome into the median it is
        being read against."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        assert prep.entry_match is not None
        assert PAST_EVENT not in prep.entry_match.event_ids

    def test_an_upcoming_release_has_nothing_to_exclude(self) -> None:
        prep = build_preparation(_event(NEXT_EVENT), symbol="BTCUSDT", now=_now())
        if prep.entry_match is None:
            pytest.skip("no measurable run-up in the cache right now")
        assert NEXT_EVENT not in prep.entry_match.event_ids
        assert prep.entry_match.sample_size > 0

    def test_the_panel_is_labelled_for_what_it_shows(self) -> None:
        """The run-up panel still shows the present after the print, which is
        worth seeing -- but calling it "market going in" would be a caption
        contradicting the numbers under it."""
        past = plain(
            render_preparation(
                build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now()),
                ("BTCUSDT",),
            )
        )
        assert "Market now" in past
        assert "Market going in" not in past


# --- J, K, L: what the page says when it cannot say a number ---------------


class TestSampleSizeIsNeverHidden:
    def test_every_scenario_card_carries_its_count(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        html = render_preparation(prep, ("BTCUSDT",))
        for cohort in prep.scenarios:
            assert f"n={cohort.sample_size}" in html

    def test_a_small_sample_is_flagged_rather_than_merely_counted(self) -> None:
        """A median over three releases and one over thirty render identically
        unless something says so."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        small = [c for c in prep.scenarios if c.quality.needs_warning]
        if not small:
            pytest.skip("no small cohort in the dataset as it stands")
        html = render_preparation(prep, ("BTCUSDT",))
        assert "sample</span>" in html
        for cohort in small:
            assert cohort.caveat is not None

    def test_the_similar_cohort_leads_with_its_count(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        if prep.entry_match is None:
            pytest.skip("no measurable run-up in the cache right now")
        text = plain(render_preparation(prep, ("BTCUSDT",)))
        assert f"n = {prep.entry_match.sample_size}" in text


class TestAnUnfinishedWindowStaysUnfinished:
    def half_finished(self) -> Preparation:
        """A release four hours old: the short windows closed, the day open.

        Built rather than simulated. The protection lives in the data, not in
        the clock -- a release from last September has a complete 24-hour
        window however recent the caller claims the present is -- so moving
        ``now`` would not produce this state and a test that tried would be
        asserting against the wrong mechanism.
        """
        event = _event(PAST_EVENT)
        now = event.timestamp_utc + dt.timedelta(hours=4)
        closed = {
            "post_event_5m": window("post_event_5m", -1.2),
            "post_event_15m": window("post_event_15m", -1.2),
            "post_event_1h": window("post_event_1h", -1.4),
            "post_event_4h": window("post_event_4h", -2.5),
        }
        still_open = WindowMetrics(
            name="post_event_24h",
            start_utc=event.timestamp_utc,
            end_utc=event.timestamp_utc + dt.timedelta(hours=24),
            status=WindowStatus.INSUFFICIENT_DATA,
            reason="the window end is beyond the last candle available",
        )
        study = StudyResult(
            spec=ResearchSpec(
                symbol="BTCUSDT",
                event_type=EventType.CPI,
                event_count=1,
                interval=Interval.M5,
                windows=REACTION_WINDOWS,
            ),
            generated_at=now,
            events=(
                EventResult(
                    event_id=event.event_id,
                    event_time_utc=event.timestamp_utc,
                    local_date=event.timestamp_utc.date(),
                    is_scheduled=True,
                    windows={**closed, "post_event_24h": still_open},
                ),
            ),
        )
        return Preparation(
            event=event,
            symbol="BTCUSDT",
            now=now,
            released=True,
            values=load_releases().get(event.event_id),
            study=study,
            scenarios=[],
            baseline=None,
            context=_context(-2.0),
            entry_match=None,
            all_values=load_releases(),
            reaction=_reaction_for(study, event, now),
        )

    def test_a_horizon_that_has_not_closed_carries_no_figure(self) -> None:
        """K. The single most misleading number this page could print is a
        24-hour return on a release four hours old. It must not be zero, which
        is what "price whatever we have" produces and what a reader would read
        as a release that did nothing."""
        prep = self.half_finished()
        assert prep.reaction is not None
        rows = prep.reaction["rows"]
        assert isinstance(rows, list)
        by_label = {str(row["label"]): row for row in rows}

        assert by_label["4h"]["return_pct"] == -2.5
        assert by_label["24h"]["return_pct"] is None
        assert "closes" in str(by_label["24h"]["reason"])

    def test_the_unfinished_horizon_reads_as_waiting_not_broken(self) -> None:
        """In the HTML a reader actually sees. A dash with "closes in 20 hours"
        beside it is a window still running; "unavailable" reads as a product
        that is not working."""
        text = plain(render_preparation(self.half_finished(), ("BTCUSDT",)))
        assert "What actually happened" in text
        assert "closes in" in text
        assert "0.0%" not in text

    def test_the_closed_horizons_are_still_reported(self) -> None:
        """Half-finished is not nothing. Withholding the hour because the day
        is open would be the opposite failure."""
        text = plain(render_preparation(self.half_finished(), ("BTCUSDT",)))
        assert "-1.4%" in text
        assert "-2.5%" in text

    def test_an_open_window_says_when_it_closes(self) -> None:
        """Rather than "unavailable", which reads as broken. The window knows
        its own end instant, so this is a comparison and not a guess."""
        just_now = dt.datetime.now(tz=dt.UTC)
        today = next(
            (
                event
                for event in load_events(EventType.CPI)
                if just_now - dt.timedelta(hours=20)
                < event.timestamp_utc
                <= just_now
            ),
            None,
        )
        if today is None:
            pytest.skip("no release inside the last 20 hours")
        prep = build_preparation(today, symbol="BTCUSDT", now=just_now)
        assert prep.reaction is not None
        rows = prep.reaction["rows"]
        assert isinstance(rows, list)
        day = next(row for row in rows if row["label"] == "24h")
        assert day["return_pct"] is None
        assert "closes" in str(day["reason"])

    def test_a_released_event_shows_the_horizons_that_closed(self) -> None:
        just_now = dt.datetime.now(tz=dt.UTC)
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=just_now)
        assert prep.reaction is not None
        rows = prep.reaction["rows"]
        assert isinstance(rows, list)
        assert all(row["return_pct"] is not None for row in rows)


class TestMissingValuesDoNotBreakThePage:
    def test_a_release_with_no_consensus_still_renders(
        self, client: TestClient
    ) -> None:
        """L. The dataset declines to record a consensus it could not source,
        and that is a stronger statement than a number would be."""
        response = client.get(f"/event/{NEXT_EVENT}/prepare")
        assert response.status_code == 200
        text = plain(response.text)
        assert "Consensus" in text
        # Said as the specific thing it is. "Unavailable" would cover a
        # forecast nobody has published and one we simply have not
        # transcribed, and only the second is ours to fix.
        assert "no verified forecast recorded for this release" in text
        # And the two figures around it are no longer blank, which is the
        # whole point of wiring BLS in.
        assert "not published until the release" in text

    def test_consensus_is_never_invented(self, client: TestClient) -> None:
        """The upcoming release has no published consensus in our data. A
        number here would be the one genuinely dishonest thing the page could
        do, and it would look exactly like the honest ones."""
        values = load_releases().get(NEXT_EVENT)
        assert values is None or values.consensus_yoy is None
        text = plain(client.get(f"/event/{NEXT_EVENT}/prepare").text)
        assert "Consensus - " in text or "Consensus -" in text

    def test_an_unmeasurable_run_up_does_not_take_the_page_down(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        blank = MarketContext(
            symbol="BTCUSDT",
            as_of=_now(),
            interval=Interval.M5,
            price=None,
            windows={},
        )
        stripped = Preparation(
            event=prep.event,
            symbol=prep.symbol,
            now=prep.now,
            released=prep.released,
            values=prep.values,
            study=prep.study,
            scenarios=prep.scenarios,
            baseline=prep.baseline,
            context=blank,
            entry_match=None,
            all_values=prep.all_values,
            reaction=prep.reaction,
        )
        html = render_preparation(stripped, ("BTCUSDT",))
        assert "cannot be measured" in plain(html)
        assert "Scenario playbook" in plain(html)


# --- M: the sample is inspectable ------------------------------------------


class TestTheSampleCanBeInspected:
    def test_the_releases_behind_a_cohort_are_listed(self) -> None:
        """M. "These are the eleven events behind this statistic" is the whole
        of why a reader should believe the statistic."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        html = render_preparation(prep, ("BTCUSDT",))
        assert "The releases behind these numbers" in html
        cohort = next(c for c in prep.scenarios if c.sample_size)
        study_by_id = {e.event_id: e for e in prep.study.events}  # type: ignore[union-attr]
        for event_id in cohort.event_ids:
            date = study_by_id[event_id].local_date
            assert f"{date:%d %b %Y}" in html

    def test_each_row_carries_the_figures_that_sorted_it(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = plain(render_preparation(prep, ("BTCUSDT",)))
        for header in ("Previous", "Consensus", "Actual", "Surprise"):
            assert header in text

    def test_the_methodology_states_what_was_measured(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = plain(render_preparation(prep, ("BTCUSDT",)))
        for claim in (
            "Binance candles",
            "America/New_York",
            "Releases examined",
            "year-over-year",
        ):
            assert claim in text


# --- the prose is a template over computed facts ---------------------------


class TestWhatToWatchIsTemplatedNotWritten:
    def test_it_quotes_the_numbers_on_the_page(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        lines = what_to_watch_lines(prep)
        assert lines
        joined = " ".join(lines)
        assert "median" in joined

    def test_it_says_nothing_it_cannot_support(self) -> None:
        """A cohort with no releases produces no sentence about one, rather
        than a sentence hedged into meaning nothing."""
        empty = Preparation(
            event=_event(PAST_EVENT),
            symbol="BTCUSDT",
            now=_now(),
            released=False,
            values=None,
            study=None,
            scenarios=[],
            baseline=None,
            context=MarketContext(
                symbol="BTCUSDT",
                as_of=_now(),
                interval=Interval.M5,
                price=None,
                windows={},
            ),
            entry_match=None,
            all_values={},
        )
        assert what_to_watch_lines(empty) == []

    def test_it_never_tells_the_reader_what_to_do(self) -> None:
        """The page provides historical evidence. A verb like "buy" would turn
        it into personalised advice nobody is licensed to give."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = plain(render_preparation(prep, ("BTCUSDT",))).lower()
        for forbidden in (" buy ", " sell ", " long ", " short ", "you should"):
            assert forbidden not in text

    def test_it_speaks_historically_rather_than_causally(self) -> None:
        """Every claim is scoped to the sample it came from. "In this sample"
        does that work at least as well as "historically", and does it in the
        clause where the number actually appears."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = plain(render_preparation(prep, ("BTCUSDT",))).lower()
        assert "in this sample" in text
        assert "historical frequencies from the sample" in text
        for causal in ("causes", "will push", "leads to", "because cpi"):
            assert causal not in text


class TestTheReaderKnowsWhichFigureThisIs:
    """"US Consumer Price Index" does not say whether 3.8% is headline or core,
    annual or monthly. Those surprise in opposite directions on the same
    morning, so a reader who assumes the wrong one is reading a different
    release's statistics off this page with nothing to warn them.
    """

    def page(self) -> str:
        return render_preparation(
            build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now()),
            ("BTCUSDT",),
        )

    def test_the_measure_is_under_the_title(self) -> None:
        from backtool.events.releases import CPI_METRIC_SHORT

        assert CPI_METRIC_SHORT == "Headline CPI, year-over-year"
        assert f'<p class="prep-measure">{CPI_METRIC_SHORT}</p>' in self.page()

    def test_the_methodology_states_it_in_full(self) -> None:
        from backtool.events.releases import CPI_METRIC

        assert "year-over-year" in CPI_METRIC
        assert CPI_METRIC in plain(self.page())

    def test_nothing_claims_core_or_month_over_month(self) -> None:
        """The dataset holds neither. A page that implied otherwise would be
        describing a series it has never measured."""
        text = plain(self.page()).lower()
        assert "core cpi" in text, "it should say so, to rule it out"
        assert "not covered" in text
        # And never as a thing being reported.
        assert "core cpi yoy:" not in text
        assert "month-over-month reading" not in text

    def test_the_dataset_really_only_holds_headline_yearly(self) -> None:
        """The claim above is only honest while this holds. If a core or
        monthly column is ever added, this fails and the wording has to be
        revisited rather than quietly becoming wrong."""
        import csv
        from importlib import resources

        with resources.files("backtool.events.data").joinpath(
            "releases.csv"
        ).open(encoding="utf-8") as handle:
            columns = set(next(csv.reader(handle)))
        assert {"previous_yoy", "consensus_yoy", "actual_yoy"} <= columns
        assert not any("core" in name or "_mom" in name for name in columns)


class TestTheCurrentSetupComesFirst:
    """Somebody opening this twenty minutes before a print wants the current
    situation, not a report that opens with history. The evidence is context
    for today; today is the subject.
    """

    def body(self) -> str:
        return render_preparation(
            build_preparation(_event(NEXT_EVENT), symbol="BTCUSDT", now=_now()),
            ("BTCUSDT",),
        )

    def test_the_market_now_sits_above_the_playbook(self) -> None:
        html = self.body()
        assert html.index('class="prep-context"') < html.index(
            'class="prep-scenarios"'
        )

    def test_and_above_the_chart(self) -> None:
        html = self.body()
        assert html.index('class="prep-context"') < html.index('"prep-chart')

    def test_the_header_is_still_first_and_method_still_last(self) -> None:
        html = self.body()
        order = [
            html.index('class="prep-header"'),
            html.index('class="prep-watch'),
            html.index('class="prep-context"'),
            html.index('class="prep-scenarios"'),
            html.index('"prep-chart'),
            html.index('class="prep-occurrences"'),
            html.index('class="prep-method"'),
            html.index('class="prep-followup'),
        ]
        assert order == sorted(order)


class TestTheSummaryTableMatchesTheCohorts:
    def prep(self) -> Preparation:
        return build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())

    def test_every_scenario_has_a_row(self) -> None:
        prep = self.prep()
        html = render_preparation(prep, ("BTCUSDT",))
        table = html.split("prep-summary")[1].split("</table>")[0]
        for cohort in prep.scenarios:
            assert SCENARIO_LABELS[cohort.key] in table

    def test_the_figures_are_the_aggregates(self) -> None:
        """D and I. The table is a view of the cohorts, not a second
        calculation of them."""
        prep = self.prep()
        table = (
            render_preparation(prep, ("BTCUSDT",))
            .split("prep-summary")[1]
            .split("</table>")[0]
        )
        for cohort in prep.scenarios:
            window = cohort.windows[HEADLINE_WINDOW]
            assert f">{cohort.sample_size}</td>" in table
            assert f"{window.median_return_pct:+.1f}%" in table
            assert f"{window.up_count} / {window.sample_size}" in table

    def test_the_counts_match_the_release_lists(self) -> None:
        """E. A count is only honest if it is the length of the list the next
        section shows."""
        prep = self.prep()
        for cohort in prep.scenarios:
            assert cohort.sample_size == len(cohort.event_ids)

    def test_the_baseline_is_in_the_table_and_marked_apart(self) -> None:
        prep = self.prep()
        html = render_preparation(prep, ("BTCUSDT",))
        assert "prep-row-baseline" in html
        assert "All releases" in html

    def test_the_rule_is_stated_without_inventing_a_threshold(self) -> None:
        """A number here would be a consensus for a release that has none."""
        text = plain(render_preparation(self.prep(), ("BTCUSDT",)))
        assert "actual above consensus" in text
        assert "actual below consensus" in text
        assert "actual equals consensus" in text


class TestTheChartIsBarsNotAPath:
    """Five medians from one anchor. A line between them draws a journey
    nobody measured, and invites a reader to see momentum in the gaps.
    """

    def svg(self) -> str:
        html = render_preparation(
            build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now()),
            ("BTCUSDT",),
        )
        return html.split('class="prep-chart')[1].split("</section>")[0]

    def test_it_draws_bars(self) -> None:
        assert "<rect" in self.svg()

    def test_no_path_semantics_survive(self) -> None:
        """G. Not just "there is no polyline" -- no line joining the horizons
        at all, by any element."""
        svg = self.svg()
        assert "polyline" not in svg
        assert "<path" not in svg
        assert "prep-dot" not in svg

    def test_it_says_so_in_words_as_well(self) -> None:
        text = plain(self.svg())
        assert "not a path any release followed" in text

    def test_each_bar_carries_its_own_horizon_value(self) -> None:
        """F. The bars are the independent medians, not a cumulative series."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        svg = self.svg()
        cohort = next(c for c in prep.scenarios if c.sample_size)
        for name, label in WINDOW_LABELS.items():
            median = cohort.windows[name].median_return_pct
            assert f"at {label}: {median:+.1f}% median" in svg

    def test_every_horizon_is_on_the_axis(self) -> None:
        svg = self.svg()
        for label in WINDOW_LABELS.values():
            assert f">{label}</text>" in svg


class TestTheMissingConsensusIsExplained:
    """J and K. The biggest gap on the page, and the one most tempting to fill
    with a number from somewhere.
    """

    def test_no_figure_is_invented_for_the_next_release(self) -> None:
        """The row now exists -- `refresh-cpi` fills its previous from the
        published series -- so the thing to pin is no longer "there is no row".
        It is that the one number nobody publishes is still absent."""
        from backtool.events.releases import load_releases

        values = load_releases().get(NEXT_EVENT)
        assert values is not None
        assert values.consensus_yoy is None
        assert values.actual_yoy is None
        assert values.previous_yoy is not None
        text = plain(
            render_preparation(
                build_preparation(_event(NEXT_EVENT), symbol="BTCUSDT", now=_now()),
                ("BTCUSDT",),
            )
        )
        assert "Consensus - " in text or "Consensus -" in text

    def test_the_three_kinds_of_absence_are_distinguished(self) -> None:
        """A dash means "no number here", which is the same appearance for a
        forecast nobody has published, one we have not transcribed, and an
        actual that cannot exist yet."""
        from backtool.events.releases import ValueStatus, release_figures

        upcoming = {f.label: f for f in release_figures(None, released=False)}
        assert upcoming["Consensus"].status is ValueStatus.NOT_COLLECTED
        assert upcoming["Actual"].status is ValueStatus.AWAITING_RELEASE
        assert "not published until the release" in upcoming["Actual"].note

    def test_a_status_never_claims_a_number_we_do_not_have(self) -> None:
        """PUBLISHED means "we have it". A figure carrying that status beside
        no value is the kind of thing a later caller trusts and a reader never
        sees, so it is the kind worth pinning."""
        from backtool.events.releases import ValueStatus, release_figures

        for released in (True, False):
            for figure in release_figures(None, released=released):
                assert figure.status is not ValueStatus.PUBLISHED, figure.label
                assert not figure.is_known

    def test_an_actual_that_does_not_exist_is_not_called_a_gap(self) -> None:
        """It is not missing from our data. It has not happened."""
        text = plain(
            render_preparation(
                build_preparation(_event(NEXT_EVENT), symbol="BTCUSDT", now=_now()),
                ("BTCUSDT",),
            )
        )
        assert "not published until the release" in text

    def test_a_disputed_consensus_is_shown_as_disputed(self) -> None:
        """Rather than resolved by taking whichever source happens to be
        stored first."""
        from backtool.events.releases import (
            ReleaseValues,
            ValueStatus,
            release_figures,
        )

        disputed = ReleaseValues(
            event_id="X",
            consensus_yoy=2.5,
            actual_yoy=2.6,
            consensus_verified=True,
            consensus_conflict=True,
        )
        figures = {f.label: f for f in release_figures(disputed, released=True)}
        assert figures["Consensus"].status is ValueStatus.DISPUTED
        assert "disagree" in figures["Consensus"].note


class TestTheSummaryCoversTheWholeSample:
    """It compared cooler against hotter and stopped, which in this dataset
    skipped the strongest of the three groups -- a true sentence leaving a
    false impression, which is the worse kind.
    """

    def lines(self) -> str:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        return " ".join(what_to_watch_lines(prep))

    def test_every_measurable_scenario_is_named(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = self.lines().lower()
        for cohort in prep.scenarios:
            if cohort.windows[HEADLINE_WINDOW].sample_size:
                assert SCENARIO_LABELS[cohort.key].lower() in text

    def test_they_are_ranked_by_outcome(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = self.lines()
        ranked = sorted(
            (c for c in prep.scenarios if c.windows[HEADLINE_WINDOW].sample_size),
            key=lambda c: c.windows[HEADLINE_WINDOW].median_return_pct,
            reverse=True,
        )
        seen = [text.index(SCENARIO_LABELS[c.key].lower()) for c in ranked]
        assert seen == sorted(seen)

    def test_the_counts_travel_with_the_medians(self) -> None:
        """So a three-release median and a thirty-release one do not read
        identically in the sentence people quote."""
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = self.lines()
        for cohort in prep.scenarios:
            if cohort.windows[HEADLINE_WINDOW].sample_size:
                assert f"n={cohort.sample_size}" in text

    def test_the_smallest_group_is_called_out_when_it_is_too_small(self) -> None:
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        smallest = min(prep.scenarios, key=lambda c: c.sample_size)
        if not smallest.quality.needs_warning:
            pytest.skip("no group small enough in the dataset as it stands")
        assert "too few to describe a tendency" in self.lines()


class TestBothConditionsAtOnce:
    """Scenario cohorts narrowed to setups resembling today's.

    The playbook answers "if it lands cooler"; the setup panel answers "the
    asset is going in falling". People ask them together, and the answers are
    not the same question twice -- over this sample cooler CPI is +0.5%, and
    cooler CPI the asset went into falling is +2.0%.
    """

    def labelled(self, rows: list[tuple[str, str, float, float]]) -> StudyResult:
        """A study from ``(event_id, scenario, entry, outcome)``.

        The scenario travels in the id so a test can state membership without
        reaching for the real values file, and entry and outcome are set
        independently so a rule that read the wrong one picks a visibly
        different set.
        """
        return synthetic_study([(f"{key}:{eid}", entry, out) for eid, key, entry, out in rows])

    def rule(self) -> object:
        def scenario(event: EventResult) -> tuple[str, str, str] | None:
            key = event.event_id.split(":")[0]
            if key == "none":
                return None
            return (key, key.replace("_", " ").title(), "")

        return scenario

    def study(self) -> StudyResult:
        return self.labelled(
            [
                ("a", "cooler", -2.0, +5.0),
                ("b", "cooler", -1.0, -4.0),
                ("c", "cooler", +3.0, +1.0),
                ("d", "hotter", -2.5, +2.0),
                ("e", "in_line", +1.0, -1.0),
                ("f", "none", -1.5, +9.0),
            ]
        )

    def test_a_negative_setup_selects_the_negative_releases(self) -> None:
        """A. Both conditions, and only the releases meeting both."""
        found = intersection_cohorts(
            self.study(), scenario=self.rule(), direction=ENTERED_DOWN
        )
        assert set(found["cooler"].event_ids) == {"cooler:a", "cooler:b"}
        assert set(found["hotter"].event_ids) == {"hotter:d"}

    def test_a_positive_setup_selects_a_different_set(self) -> None:
        """B. Otherwise the condition is decoration."""
        found = intersection_cohorts(
            self.study(), scenario=self.rule(), direction=ENTERED_UP
        )
        assert set(found["cooler"].event_ids) == {"cooler:c"}
        assert "hotter" not in found
        assert set(found["in_line"].event_ids) == {"in_line:e"}

    def test_each_group_holds_only_its_own_scenario(self) -> None:
        """C. A cohort whose members contradict its label would be the most
        convincing wrong table this page could produce."""
        for direction in (ENTERED_DOWN, ENTERED_UP):
            for key, cohort in intersection_cohorts(
                self.study(), scenario=self.rule(), direction=direction
            ).items():
                for event_id in cohort.event_ids:
                    assert event_id.startswith(f"{key}:"), (key, event_id)

    def test_a_release_the_rule_declines_is_in_no_group(self) -> None:
        """No verified consensus is not "in line". It is unknown, and the two
        must not collapse."""
        found = intersection_cohorts(
            self.study(), scenario=self.rule(), direction=ENTERED_DOWN
        )
        assert all(
            "none:" not in event_id
            for cohort in found.values()
            for event_id in cohort.event_ids
        )

    def test_membership_does_not_move_when_the_outcomes_do(self) -> None:
        """D, as a property rather than an example.

        Every post-event return is negated and the membership re-derived. If
        any outcome took part in selecting a cohort, flipping the sign of all
        of them would move somebody -- so this fails for any rule that peeks,
        not merely for the one way of peeking a hand-written case would check.

        The scenario itself is *not* look-ahead: actual against consensus is
        the condition being asked about, and the thing measured is the price
        return, which is a different quantity. Selecting on the price and then
        reporting the price is the circularity that matters.
        """
        original = self.study()
        flipped = original.model_copy(
            update={
                "events": tuple(
                    event.model_copy(
                        update={
                            "windows": {
                                **event.windows,
                                HEADLINE_WINDOW: window(
                                    HEADLINE_WINDOW,
                                    -event.windows[HEADLINE_WINDOW].return_pct,  # type: ignore[operator]
                                ),
                            }
                        }
                    )
                    for event in original.events
                )
            }
        )
        for direction in (ENTERED_DOWN, ENTERED_UP):
            before = {
                key: set(cohort.event_ids)
                for key, cohort in intersection_cohorts(
                    original, scenario=self.rule(), direction=direction
                ).items()
            }
            after = {
                key: set(cohort.event_ids)
                for key, cohort in intersection_cohorts(
                    flipped, scenario=self.rule(), direction=direction
                ).items()
            }
            assert before == after, direction

    def test_the_selecting_window_still_ends_at_the_release(self) -> None:
        assert ENTRY_WINDOW.end == "0h"
        assert ENTRY_WINDOW.start.startswith("-")

    def test_an_unmeasurable_entry_is_declined(self) -> None:
        study = self.labelled([("a", "cooler", -1.0, 1.0)])
        broken = study.model_copy(
            update={
                "events": (
                    study.events[0].model_copy(
                        update={
                            "windows": {
                                ENTRY_WINDOW.name: window(
                                    ENTRY_WINDOW.name, None, ok=False
                                ),
                                HEADLINE_WINDOW: window(HEADLINE_WINDOW, 1.0),
                            }
                        }
                    ),
                )
            }
        )
        assert (
            intersection_cohorts(
                broken, scenario=self.rule(), direction=ENTERED_DOWN
            )
            == {}
        )


class TestTheIntersectionOnThePage:
    def prep(self, symbol: str = "BTCUSDT", event: str = NEXT_EVENT) -> Preparation:
        return build_preparation(_event(event), symbol=symbol, now=_now())

    def test_it_sits_between_the_setup_and_the_playbook(self) -> None:
        html = render_preparation(self.prep(), ("BTCUSDT",))
        assert (
            html.index('class="prep-context"')
            < html.index('class="prep-similar-scenarios"')
            < html.index('class="prep-scenarios"')
        )

    def test_every_scenario_gets_a_row(self) -> None:
        prep = self.prep()
        table = render_preparation(prep, ("BTCUSDT",)).split(
            "prep-similar-scenarios"
        )[1].split("</table>")[0]
        for cohort in prep.scenarios:
            assert SCENARIO_LABELS[cohort.key] in table

    def test_the_figures_are_the_cohort_aggregates(self) -> None:
        """Not recomputed for display. The table is a view of what the research
        layer produced."""
        prep = self.prep()
        table = render_preparation(prep, ("BTCUSDT",)).split(
            "prep-similar-scenarios"
        )[1].split("</table>")[0]
        for key, cohort in prep.intersections.items():
            assert key in SCENARIO_LABELS
            metrics = cohort.windows[HEADLINE_WINDOW]
            assert f"{metrics.median_return_pct:+.1f}%" in table
            assert f"{metrics.up_count} / {metrics.sample_size}" in table

    def test_an_empty_group_says_so_rather_than_showing_zero(self) -> None:
        """F. It is information: no release has ever printed that way with the
        asset going in like this. A 0.0% would be a statistic nobody computed,
        and a missing row would read as an oversight.

        Built rather than waited for. Whether an intersection is empty depends
        on which way the market happens to be moving today, so a test that took
        what it was given would quietly cover nothing on half the days it ran
        -- and this is the branch most likely to be wrong.
        """
        prep = self.prep()
        emptied = replace(prep, intersections={})
        table = (
            render_preparation(emptied, ("BTCUSDT",))
            .split("prep-similar-scenarios")[1]
            .split("</table>")[0]
        )
        assert table.count("No release matched both conditions in this sample.") == len(
            prep.scenarios
        )
        assert "0.0%" not in table
        for cohort in prep.scenarios:
            assert SCENARIO_LABELS[cohort.key] in table

    def test_a_group_that_exists_is_not_reported_as_empty(self) -> None:
        """So the test above is evidence of a branch rather than of a constant."""
        prep = self.prep()
        if not prep.intersections:
            pytest.skip("no measurable run-up right now")
        table = (
            render_preparation(prep, ("BTCUSDT",))
            .split("prep-similar-scenarios")[1]
            .split("</table>")[0]
        )
        assert table.count(
            "No release matched both conditions in this sample."
        ) == len(prep.scenarios) - len(prep.intersections)

    def test_a_thin_group_carries_its_warning(self) -> None:
        """G. These are the smallest groups on the page and the likeliest to be
        quoted as if they were tendencies."""
        prep = self.prep()
        thin = [c for c in prep.intersections.values() if c.quality.needs_warning]
        if not thin:
            pytest.skip("no small intersection in the dataset as it stands")
        html = render_preparation(prep, ("BTCUSDT",))
        assert "cohort-quality" in html

    def test_it_is_compared_against_the_wider_scenario(self) -> None:
        """The number that makes the section worth having: the narrowing either
        moved the median or it did not."""
        prep = self.prep()
        table = (
            render_preparation(prep, ("BTCUSDT",))
            .split("prep-similar-scenarios")[1]
            .split("</table>")[0]
        )
        for key, narrowed in prep.intersections.items():
            wider = next(c for c in prep.scenarios if c.key == key)
            gap = (
                narrowed.windows[HEADLINE_WINDOW].median_return_pct
                - wider.windows[HEADLINE_WINDOW].median_return_pct
            )
            expected = "no change" if abs(gap) < 0.05 else f"{gap:+.1f} pp"
            assert expected in table

    def test_a_difference_too_small_to_print_is_not_given_a_direction(
        self,
    ) -> None:
        """"-0.0 pp" is what -0.03 rounds to, and it reads as a direction the
        figure does not support. Below the resolution being printed, the honest
        statement is that the narrowing did not move the median."""
        from backtool.web.prepare_page import _difference

        assert _difference(-0.03) == "no change"
        assert _difference(0.0) == "no change"
        assert _difference(0.04) == "no change"
        assert _difference(-1.4) == "-1.4 pp"
        assert _difference(1.4) == "+1.4 pp"

    def test_no_signed_zero_reaches_the_page(self) -> None:
        table = (
            render_preparation(self.prep(), ("BTCUSDT",))
            .split("prep-similar-scenarios")[1]
            .split("</table>")[0]
        )
        assert "-0.0 pp" not in table
        assert "+0.0 pp" not in table

    def test_it_never_claims_to_predict(self) -> None:
        text = plain(render_preparation(self.prep(), ("BTCUSDT",))).lower()
        assert "historical observations" in text
        for forbidden in ("expected return", "likely outcome", "prediction"):
            assert forbidden not in text

    def test_the_two_assets_are_computed_separately(self) -> None:
        """H. Membership can coincide -- the two move together most days -- so
        what has to differ is the measurement, which is what the table shows."""
        btc = self.prep("BTCUSDT")
        eth = self.prep("ETHUSDT")
        assert btc.intersections and eth.intersections
        shared = set(btc.intersections) & set(eth.intersections)
        assert shared
        assert any(
            btc.intersections[key].windows[HEADLINE_WINDOW].median_return_pct
            != eth.intersections[key].windows[HEADLINE_WINDOW].median_return_pct
            for key in shared
        )

    def test_an_upcoming_release_uses_the_run_up_to_now(self) -> None:
        """I."""
        prep = self.prep(event=NEXT_EVENT)
        assert not prep.released
        assert prep.entry_return_pct == prep.context.entry_return_pct

    def test_a_released_one_uses_its_own_run_up(self) -> None:
        """J. Not today's price action, which for a release months ago has
        nothing to do with how it began."""
        prep = self.prep(event=PAST_EVENT)
        assert prep.released
        assert prep.study is not None
        own = next(e for e in prep.study.events if e.event_id == PAST_EVENT)
        assert prep.entry_return_pct == own.windows[ENTRY_WINDOW.name].return_pct

    def test_a_released_one_is_not_in_its_own_groups(self) -> None:
        """E."""
        prep = self.prep(event=PAST_EVENT)
        for cohort in prep.intersections.values():
            assert PAST_EVENT not in cohort.event_ids

    def test_the_wider_scenario_statistics_are_untouched(self) -> None:
        """K. The intersection is an extra layer, not a replacement -- and the
        wider sample is what a reader should fall back to when one of these
        runs thin."""
        prep = self.prep()
        assert prep.study is not None
        values = load_releases()
        rule = SCENARIO_RULES[EventType.CPI]
        expected = split_study(
            prep.study, lambda item: rule(values.get(item.event_id))
        )
        by_key = {cohort.key: cohort for cohort in expected}
        for cohort in prep.scenarios:
            assert cohort.windows == by_key[cohort.key].windows
            assert cohort.event_ids == by_key[cohort.key].event_ids

    def test_it_costs_nothing(self, client: TestClient, no_ai: None) -> None:
        """L."""
        assert client.get(f"/event/{NEXT_EVENT}/prepare").status_code == 200
        assert client.get(f"/event/{PAST_EVENT}/prepare?symbol=ETHUSDT").status_code == 200


# --- N, O: nothing that worked before stopped --------------------------------


class TestTheExistingSurfacesStillWork:
    def test_the_research_page_still_opens(self, client: TestClient) -> None:
        assert client.get("/research").status_code == 200

    def test_the_research_box_can_arrive_prefilled(self, client: TestClient) -> None:
        """The follow-up link fills the box and stops. The reader still presses
        the button, so following a suggestion costs nothing until they decide
        it is worth something."""
        html = client.get("/research?q=Does+BTC+rise+after+cooler+CPI%3F").text
        assert "Does BTC rise after cooler CPI?" in html

    def test_a_prefill_cannot_paste_a_novel(self, client: TestClient) -> None:
        html = client.get("/research?q=" + "x" * 4000).text
        assert "x" * 600 not in html

    def test_the_curated_event_page_still_opens(self, client: TestClient) -> None:
        assert client.get(f"/event/{PAST_EVENT}").status_code == 200

    def test_the_calendar_still_opens(self, client: TestClient) -> None:
        assert client.get("/").status_code == 200

    def test_the_card_is_still_one_click_to_the_event(self, client: TestClient) -> None:
        """The card gained a second destination. The first one has to keep
        working, and a link inside a link would have been resolved by the
        browser guessing."""
        html = client.get("/").text
        assert 'class="event-open"' in html
        assert "<a" not in html.split('class="event-open"')[1].split("</a>")[0]


# --- helpers ---------------------------------------------------------------


def _now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def _event(event_id: str):  # type: ignore[no-untyped-def]
    return next(
        event for event in load_events(EventType.CPI) if event.event_id == event_id
    )


def _context(entry: float | None) -> MarketContext:
    windows = (
        {CONTEXT_ENTRY_WINDOW: window(CONTEXT_ENTRY_WINDOW, entry)}
        if entry is not None
        else {}
    )
    return MarketContext(
        symbol="BTCUSDT",
        as_of=_now(),
        interval=Interval.M5,
        price=100.0,
        windows=windows,
    )


def test_the_baseline_helper_declines_an_empty_study() -> None:
    """Nothing measured is no baseline, rather than a cohort of zero whose
    median would be a number with nothing behind it."""
    assert baseline_cohort(synthetic_study([])) is None


def test_the_run_up_is_measured_by_the_research_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current 24h return and every historical one come out of the same
    function, so the number a reader compares against twenty others was not
    computed by a different rule from theirs."""
    from backtool.research import preparation

    seen: list[str] = []
    original = preparation.compute_window_metrics

    def record(frame: object, resolved: object, interval: object) -> object:
        seen.append(getattr(resolved, "name", "?"))
        return original(frame, resolved, interval)  # type: ignore[arg-type]

    monkeypatch.setattr(preparation, "compute_window_metrics", record)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [dt.datetime(2026, 1, 1, tzinfo=dt.UTC)], utc=True
            ),
            "close_time": pd.to_datetime(
                [dt.datetime(2026, 1, 1, 0, 5, tzinfo=dt.UTC)], utc=True
            ),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    build_market_context(
        frame, symbol="BTCUSDT", now=_now(), interval=Interval.M5
    )
    assert seen == ["last_1h", "last_4h", "last_24h"]
