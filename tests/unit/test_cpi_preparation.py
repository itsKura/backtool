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

import pytest
from fastapi.testclient import TestClient

from backtool.core.types import EventType, Interval
from backtool.events import load_events
from backtool.events.releases import ReleaseValues, classify_cpi, load_releases
from backtool.research.aggregate import aggregate_window
from backtool.research.metrics import Direction, WindowMetrics, WindowStatus
from backtool.research.preparation import (
    CONTEXT_ENTRY_WINDOW,
    ENTRY_WINDOW,
    MarketContext,
    baseline_cohort,
    build_market_context,
    cohort_for_entry,
    entry_cohorts,
)
from backtool.research.results import EventResult, StudyResult
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.app import app
from backtool.web.prepare_page import (
    HEADLINE_WINDOW,
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
        assert "unavailable" in text or "no values recorded" in text

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
        assert "Historical scenarios" in plain(html)


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
        prep = build_preparation(_event(PAST_EVENT), symbol="BTCUSDT", now=_now())
        text = plain(render_preparation(prep, ("BTCUSDT",))).lower()
        assert "historically" in text
        for causal in ("causes", "will push", "leads to"):
            assert causal not in text


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
