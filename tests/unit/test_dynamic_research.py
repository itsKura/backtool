"""A research question about an event this engine has no calendar for.

The pieces existed separately: a resolver that establishes public dates from
cited sources, and an engine that measures reactions around a calendar. This is
the seam between them, and the claim it makes is narrow -- **a resolved concept
is measured by the same engine, through the same thesis, as a curated family.**

Three things have to hold for that, and each has a section here.

*A thesis knows which it is.* Either a curated family or a resolved concept,
never both and never neither, and every sentence it produces reads correctly
for both.

*The planner names the event but does not date it.* It can tell that a question
is about the Ethereum Merge; it cannot tell when the Merge happened, and a
thesis that let it try would be exactly the fabrication the whole provenance
layer exists to prevent.

*The reader sees the dates before confirming.* For a looked-up event the dates
*are* the research, so a specification card without them would be asking for
confirmation of the uninteresting half.

No API calls: every resolution here is a fixture. The end-to-end study reads
candles through the ordinary cache, like the other engine tests.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest

from backtool.ai.thesis_planner import THESIS_PLANNER_SYSTEM, PlannedThesis
from backtool.core.types import EventType
from backtool.events.models import SourceAuthority, SourceRef
from backtool.events.resolution import (
    Ambiguity,
    AmbiguitySeverity,
    AnchorPolicy,
    DayAnchor,
    ResolutionStatus,
    ResolvedEventConcept,
)
from backtool.research.thesis import Condition, Outcome, Thesis
from backtool.research.vocabulary import Operator
from backtool.web.app import _resolution_for
from backtool.web.research_page import (
    render_anchor_needed,
    render_event_provenance,
    render_spec_card,
    render_unresolved_event,
)
from fixtures.dynamic_events import collapse, eth_merge, midterms, unresolvable

DAY_START = DayAnchor(policy=AnchorPolicy.DAY_START)


def planned(**overrides: object) -> PlannedThesis:
    """A planner answer, with everything the schema requires."""
    base: dict[str, object] = {
        "restated_question": "What happened to ETH after the Ethereum Merge?",
        "asset": "ETHUSDT",
        "event_type": "",
        "event_concept": "Ethereum Merge",
        "conditions": [],
        "outcome_metric": "return_pct",
        "outcome_horizon": "24h",
        "aggregation": "median",
        "order": "",
        "limit": "",
        "outcome_threshold": "",
        "outcome_threshold_operator": "",
        "comparison_conditions": [],
        "compare_to_baseline": False,
        "assumptions": [],
        "clarifications": [],
        "unsupported": [],
    }
    base.update(overrides)
    return PlannedThesis.model_validate(base)


def dynamic_thesis(**overrides: object) -> Thesis:
    base: dict[str, object] = {
        "question": "What happened to ETH after the Ethereum Merge?",
        "asset": "ETHUSDT",
        "event_type": None,
        "resolution": eth_merge(),
        "outcome": Outcome(),
    }
    base.update(overrides)
    return Thesis(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------


class TestAThesisKnowsWhichKindItIs:
    """A family or a resolved concept, and the sentences work for both."""

    def test_a_resolved_concept_is_a_valid_target(self) -> None:
        thesis = dynamic_thesis()
        assert thesis.event_type is None
        assert thesis.is_runnable

    def test_both_at_once_is_refused(self) -> None:
        """Ambiguous about what gets measured, which is not a state the engine
        should have to interpret."""
        with pytest.raises(ValueError, match="Both were given"):
            dynamic_thesis(event_type=EventType.CPI)

    def test_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Neither was given"):
            dynamic_thesis(resolution=None)

    def test_a_curated_thesis_is_unchanged(self) -> None:
        """The common case says nothing and keeps the default."""
        thesis = Thesis(question="q")
        assert thesis.event_type is EventType.CPI
        assert thesis.resolution is None
        assert thesis.event_phrase == "CPI releases"

    def test_the_phrase_reads_for_both(self) -> None:
        assert dynamic_thesis().event_phrase == "Ethereum Merge"
        assert Thesis(question="q").event_phrase == "CPI releases"

    def test_the_label_comes_from_the_sources(self) -> None:
        assert dynamic_thesis().event_label == "Ethereum Merge"

    def test_the_identity_carries_the_resolution_not_the_name(self) -> None:
        """A study of two midterms must never be served for a resolution that
        found three, so the identity is the events and not the slug."""
        two = dynamic_thesis(resolution=midterms(), anchor=DAY_START)
        three = dynamic_thesis(
            resolution=midterms(
                dates=(dt.date(2018, 11, 6), dt.date(2022, 11, 8), dt.date(2026, 11, 3))
            ),
            anchor=DAY_START,
        )
        assert two.event_set != three.event_set
        assert two.event_set.startswith("dynamic:")

    def test_a_curated_identity_names_the_family(self) -> None:
        assert Thesis(question="q").event_set == "curated:cpi"

    def test_a_day_only_resolution_will_not_run_without_an_anchor(self) -> None:
        """Measuring from a calendar day is a decision, and an unmade decision
        is not a thesis that can execute."""
        thesis = dynamic_thesis(resolution=midterms())
        assert not thesis.is_runnable

    def test_the_same_resolution_runs_once_anchored(self) -> None:
        thesis = dynamic_thesis(resolution=midterms(), anchor=DAY_START)
        assert thesis.is_runnable

    def test_an_anchor_without_dates_to_anchor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="only a resolved concept can need"):
            Thesis(question="q", anchor=DAY_START)

    def test_an_unsettled_resolution_is_not_runnable(self) -> None:
        thesis = dynamic_thesis(resolution=unresolvable(), anchor=None)
        assert not thesis.is_runnable

    def test_a_blocking_question_stops_it(self) -> None:
        contested = eth_merge().model_copy(
            update={"ambiguities": (Ambiguity(question="Which Merge?"),)}
        )
        assert not dynamic_thesis(resolution=contested).is_runnable

    def test_a_disclosure_does_not(self) -> None:
        noted = eth_merge().model_copy(
            update={
                "ambiguities": (
                    Ambiguity(
                        question="Two sources differ by seconds.",
                        severity=AmbiguitySeverity.INFORMATIONAL,
                    ),
                )
            }
        )
        assert dynamic_thesis(resolution=noted).is_runnable


class TestConditionsHaveToMeanSomething:
    """A filter computed from a calendar this engine does not maintain."""

    def test_a_market_state_condition_applies_to_anything(self) -> None:
        """Computed from price, so it works for any event with a timestamp --
        which is most of the vocabulary, and why this seam is small."""
        thesis = dynamic_thesis(
            conditions=(
                Condition(field="btc_trend_7d", operator=Operator.IS, value="bearish"),
            )
        )
        assert thesis.is_runnable

    def test_a_curated_field_is_refused(self) -> None:
        """Nobody publishes a consensus for a protocol upgrade, so a surprise
        against one cannot be computed -- and must not read as 'missing'."""
        with pytest.raises(ValueError, match="derived from a curated calendar"):
            dynamic_thesis(
                conditions=(
                    Condition(field="cpi_surprise", operator=Operator.IS, value="hotter"),
                )
            )

    def test_the_curated_rule_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match="cannot be tested against FOMC"):
            Thesis(
                question="q",
                event_type=EventType.FOMC,
                conditions=(
                    Condition(field="cpi_surprise", operator=Operator.IS, value="hotter"),
                ),
            )


class TestThePlannerNamesButDoesNotDate:
    """It can say which event. It must not say when."""

    def test_it_reports_that_a_lookup_is_needed(self) -> None:
        assert planned().needs_resolution
        assert planned().concept == "Ethereum Merge"

    def test_a_curated_question_needs_no_lookup(self) -> None:
        assert not planned(event_type="CPI", event_concept="").needs_resolution

    def test_a_concept_without_a_lookup_cannot_become_a_thesis(self) -> None:
        """The fabrication this prevents: a thesis with dates nobody sourced."""
        with pytest.raises(ValueError, match="has to be looked up"):
            planned().to_thesis("q")

    def test_a_resolution_completes_it(self) -> None:
        thesis = planned().to_thesis("q", resolution=eth_merge())
        assert thesis.event_type is None
        assert thesis.resolution is not None
        assert thesis.is_runnable

    def test_naming_both_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has to be one"):
            planned(event_type="CPI").to_thesis("q", resolution=eth_merge())

    def test_naming_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no event"):
            planned(event_concept="").to_thesis("q")

    def test_a_curated_question_still_converts(self) -> None:
        thesis = planned(event_type="CPI", event_concept="").to_thesis("q")
        assert thesis.event_type is EventType.CPI

    def test_the_anchor_travels_with_the_thesis(self) -> None:
        thesis = planned().to_thesis(
            "q", resolution=midterms(), anchor=DAY_START
        )
        assert thesis.anchor == DAY_START
        assert thesis.is_runnable

    def test_the_prompt_says_to_name_the_event_not_the_question(self) -> None:
        assert "Events with no calendar here" in THESIS_PLANNER_SYSTEM
        assert "Name the event, not the question" in THESIS_PLANNER_SYSTEM

    def test_the_prompt_keeps_the_curated_fields_off_a_concept(self) -> None:
        assert "derived from a curated calendar cannot be tested" in THESIS_PLANNER_SYSTEM


class TestTheReaderSeesWhereTheDatesCameFrom:
    """For a looked-up event, the dates are the research."""

    def test_a_curated_thesis_shows_no_provenance_block(self) -> None:
        assert render_event_provenance(Thesis(question="q")) == ""

    def test_the_occurrence_and_its_sources_are_shown(self) -> None:
        html = render_event_provenance(dynamic_thesis())
        assert "2022-09-15" in html
        assert "blog.ethereum.org" in html
        assert "official" in html

    def test_a_published_instant_says_so(self) -> None:
        """The distinction that matters most: a sourced timestamp is a fact."""
        assert "as published" in render_event_provenance(dynamic_thesis())

    def test_a_chosen_instant_says_that_instead(self) -> None:
        """A date measured from the start of its day is a decision somebody
        made, and it must not look like a published time."""
        html = render_event_provenance(
            dynamic_thesis(resolution=midterms(), anchor=DAY_START)
        )
        assert "measured from" in html
        assert "start of the calendar day" in html
        assert "as published" not in html

    def test_milestones_are_named_as_not_measured(self) -> None:
        html = render_event_provenance(
            dynamic_thesis(resolution=collapse(), anchor=DAY_START)
        )
        assert "deliberately not measured" in html
        assert "Rescue announcement" in html

    def test_only_the_cited_sources_are_listed(self) -> None:
        """The search returns plenty the answer never relied on, and listing
        those beside the evidence would overstate the provenance."""
        extra = SourceRef(
            url="https://nobody-relied-on-this.example/",
            title="Unused page",
            authority=SourceAuthority.SECONDARY,
        )
        concept = eth_merge()
        concept = concept.model_copy(update={"sources": (*concept.sources, extra)})
        html = render_event_provenance(dynamic_thesis(resolution=concept))
        assert "nobody-relied-on-this" not in html

    def test_a_disclosure_is_shown_without_blocking(self) -> None:
        noted = eth_merge().model_copy(
            update={
                "ambiguities": (
                    Ambiguity(
                        question="Two sources differ by seconds.",
                        why_it_matters="Smaller than one candle.",
                        severity=AmbiguitySeverity.INFORMATIONAL,
                    ),
                )
            }
        )
        html = render_event_provenance(dynamic_thesis(resolution=noted))
        assert "Worth knowing" in html
        assert "differ by seconds" in html

    def test_the_spec_card_carries_it_all(self) -> None:
        """One card: what was understood, what was looked up, and the button."""
        html = render_spec_card(dynamic_thesis(), planned())
        assert "Dates established from published sources" in html
        assert "2022-09-15" in html
        assert 'data-run="1"' in html

    def test_the_thesis_round_trips_through_the_page(self) -> None:
        """The page sends the thesis back to run it, so the resolution has to
        survive the trip -- otherwise the run would re-resolve, and pay again."""
        html = render_spec_card(dynamic_thesis(), planned())
        start = html.index('class="thesis">') + len('class="thesis">')
        payload = html[start : html.index("</script>", start)]
        restored = Thesis.model_validate_json(payload)
        assert restored.resolution is not None
        assert restored.resolution.canonical_name == "Ethereum Merge"
        assert restored.is_runnable


class TestAnUnsettledLookupIsAQuestion:
    """Four different outcomes, four different sentences, no error page."""

    def test_an_ambiguous_concept_asks_which(self) -> None:
        html = render_unresolved_event(unresolvable())
        assert "matches more than one event" in html
        assert "Which banking crisis" in html

    def test_a_conflict_says_the_sources_disagree(self) -> None:
        contested = eth_merge().model_copy(
            update={"status": ResolutionStatus.CONFLICTING}
        )
        assert "sources disagree" in render_unresolved_event(contested)

    def test_nothing_found_says_so_plainly(self) -> None:
        missing = eth_merge().model_copy(
            update={"status": ResolutionStatus.NOT_FOUND, "candidate_events": ()}
        )
        assert "could not establish" in render_unresolved_event(missing)

    def test_a_period_is_distinguished_from_a_moment(self) -> None:
        period = eth_merge().model_copy(
            update={"status": ResolutionStatus.NOT_RESOLVABLE}
        )
        assert "over a period rather than at a moment" in render_unresolved_event(period)

    def test_it_invites_a_follow_up(self) -> None:
        assert "follow-up" in render_unresolved_event(unresolvable())

    def test_what_was_found_is_still_shown(self) -> None:
        """Even when it cannot be measured. A reader deciding how to narrow the
        question needs to see what the search turned up."""
        html = render_unresolved_event(
            eth_merge().model_copy(update={"status": ResolutionStatus.CONFLICTING})
        )
        assert "2022-09-15" in html


class TestTheLookupIsNotPaidForTwice:
    """A follow-up about the same event must not buy the same dates again.

    The resolver is stubbed throughout. These tests are about which branch is
    taken, and a test that can reach a paid web search to find out is a test
    that bills the project for its own coverage -- which this one did, once,
    before the stub went in.
    """

    @pytest.fixture
    def attempted(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Records every concept a lookup was attempted for, and calls nothing."""
        calls: list[str] = []

        def stub(concept: str, **_: object) -> ResolvedEventConcept:
            calls.append(concept)
            return eth_merge()

        monkeypatch.setattr(
            "backtool.ai.event_resolver.resolve_event_concept", stub
        )
        return calls

    def on_screen(self) -> Thesis:
        """A thesis whose resolution was looked up under its own name."""
        return dynamic_thesis(
            resolution=eth_merge().model_copy(update={"query": "Ethereum Merge"})
        )

    def test_the_resolution_on_screen_is_reused(self, attempted: list[str]) -> None:
        previous = self.on_screen()
        assert _resolution_for("Ethereum Merge", previous) is previous.resolution
        assert attempted == []

    def test_the_match_ignores_case_and_spacing(self, attempted: list[str]) -> None:
        previous = self.on_screen()
        assert _resolution_for("  ethereum merge  ", previous) is previous.resolution
        assert attempted == []

    def test_a_different_event_gets_its_own_lookup(
        self, attempted: list[str]
    ) -> None:
        """The failure that would matter: silently measuring the wrong event
        because something was already on screen."""
        _resolution_for("United States midterm elections", self.on_screen())
        assert attempted == ["United States midterm elections"]

    def test_a_curated_thesis_on_screen_is_not_reused(
        self, attempted: list[str]
    ) -> None:
        _resolution_for("Ethereum Merge", Thesis(question="q"))
        assert attempted == ["Ethereum Merge"]

    def test_nothing_on_screen_means_a_lookup(self, attempted: list[str]) -> None:
        _resolution_for("Ethereum Merge", None)
        assert attempted == ["Ethereum Merge"]


class TestTwoDifferentKindsOfNo:
    """A question this system cannot answer, versus one it has no calendar for.

    Confusing these costs in both directions. Treating an unlisted event as
    unsupported refuses questions the engine can answer perfectly well -- which
    it did, in production, because one paragraph of the prompt still said every
    event had to come from the vocabulary verbatim. Treating genuinely missing
    data as a lookup would promise a measurement nothing can produce.

    The test is about the *data*, not about the event: price is connected and a
    date can be looked up, so an obscure protocol upgrade is measurable; yields
    are not connected, so they stay unsupported however famous the event is.
    """

    def test_a_curated_question_takes_the_curated_path(self) -> None:
        answer = planned(event_type="CPI", event_concept="")
        assert not answer.needs_resolution
        assert answer.concept == ""
        assert answer.to_thesis("q").event_type is EventType.CPI

    def test_a_public_event_is_a_lookup_not_a_refusal(self) -> None:
        answer = planned(event_concept="Ethereum Merge", event_type="")
        assert answer.needs_resolution
        assert answer.concept == "Ethereum Merge"
        assert answer.unsupported == []

    def test_missing_data_is_still_unsupported(self) -> None:
        answer = planned(
            event_type="CPI",
            event_concept="",
            unsupported=["Treasury yields are not connected"],
        )
        assert answer.unsupported
        assert not answer.needs_resolution

    def test_a_question_can_need_both_answers(self) -> None:
        """A looked-up event asked about an asset nobody connected. The event
        is fine and the data is not, and both have to be said."""
        answer = planned(
            event_concept="FTX bankruptcy",
            event_type="",
            unsupported=["equity prices are not connected"],
        )
        assert answer.needs_resolution
        assert answer.unsupported

    def test_the_prompt_separates_the_two(self) -> None:
        assert "Two different kinds of 'no'" in THESIS_PLANNER_SYSTEM
        assert "it does not belong in `unsupported`" in THESIS_PLANNER_SYSTEM

    def test_the_prompt_no_longer_demands_a_listed_event(self) -> None:
        """The exact sentence that caused the production refusal."""
        assert "Every asset, event, condition field" not in THESIS_PLANNER_SYSTEM

    def test_the_prompt_frames_the_test_as_being_about_data(self) -> None:
        assert "about the *data*, not about the event" in THESIS_PLANNER_SYSTEM

    def test_the_unsupported_field_says_what_it_is_not_for(self) -> None:
        """A model reads the field description even when it skims the prompt."""
        description = PlannedThesis.model_fields["unsupported"].description
        assert description is not None
        assert "does NOT belong here" in description

    def test_naming_an_event_changes_nothing_else(self) -> None:
        """The new field widens *selection* only. The same question read as a
        concept and as a family must produce the same measurement."""
        concept = planned(event_concept="Ethereum Merge", event_type="").to_thesis(
            "q", resolution=eth_merge()
        )
        family = planned(event_concept="", event_type="CPI").to_thesis("q")
        assert concept.outcome == family.outcome
        assert concept.asset == family.asset
        assert concept.conditions == family.conditions
        assert concept.versus == family.versus
        assert concept.event_count == family.event_count


class TestTheInstantIsTheReadersToChoose:
    """A day with no published time is a question, and now an answerable one.

    Midnight local is not a neutral reading of "the day of the election": it
    puts the whole trading day after the event, so a 1h or 24h window from it
    answers a different question than one from the close. Nothing here picks.
    What changed is that the reader can now answer -- and that answering is
    free, because the resolution travels with the question rather than being
    looked up again.
    """

    def test_a_day_only_resolution_stops_before_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "backtool.research.experiment.run_study",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran anyway")),
        )
        thesis = dynamic_thesis(resolution=midterms(), anchor=None)
        assert thesis.needs_an_anchor
        assert not thesis.is_runnable

    def test_compilation_refuses_it_too(self) -> None:
        """Same refusal at the layer below, so neither depends on the other."""
        from backtool.events.resolution import compile_event_set

        with pytest.raises(ValueError, match="needs a decision"):
            compile_event_set(midterms())

    def test_the_reader_is_told_what_is_needed_and_why(self) -> None:
        html = render_anchor_needed(dynamic_thesis(resolution=midterms()))
        assert "no published time" in html
        assert "2022-11-08" in html
        assert "changes the answer" in html
        assert "I will not" in html

    def test_it_offers_no_default(self) -> None:
        """The thing this exists to prevent: a midnight that reads like a
        published fact once it reaches a report.

        Checked against what a reader can see, not the raw markup -- the card
        embeds the thesis as JSON, and a retrieval timestamp in there is not an
        anchor being offered.
        """
        html = render_anchor_needed(dynamic_thesis(resolution=midterms()))
        visible = html.split('<script')[0]
        assert "00:00" not in visible
        assert "start of the calendar day (" not in visible

    def test_the_question_carries_the_resolution(self) -> None:
        """What makes answering free. Without this the follow-up would be
        planned from scratch and buy the same dates a second time."""
        html = render_anchor_needed(dynamic_thesis(resolution=midterms()))
        start = html.index('class="thesis">') + len('class="thesis">')
        restored = Thesis.model_validate_json(
            html[start : html.index("</script>", start)]
        )
        assert restored.resolution is not None
        assert restored.needs_an_anchor

    def test_it_invites_the_three_forms_it_can_read(self) -> None:
        html = render_anchor_needed(dynamic_thesis(resolution=midterms()))
        assert "from the close of that day" in html
        assert "from the start of the day" in html
        assert "from 22:00" in html

    @pytest.mark.parametrize(
        ("spoken", "policy"),
        [
            ("end of day", AnchorPolicy.DAY_END),
            ("close of that day", AnchorPolicy.DAY_END),
            ("the close", AnchorPolicy.DAY_END),
            ("start of day", AnchorPolicy.DAY_START),
            ("market open", AnchorPolicy.DAY_START),
            ("22:00", AnchorPolicy.EXPLICIT_TIME),
            ("09:30", AnchorPolicy.EXPLICIT_TIME),
        ],
    )
    def test_the_planner_reads_what_the_reader_said(
        self, spoken: str, policy: AnchorPolicy
    ) -> None:
        answer = planned(event_anchor=spoken)
        assert answer.anchor is not None
        assert answer.anchor.policy is policy

    def test_an_explicit_time_keeps_the_time(self) -> None:
        anchor = planned(event_anchor="22:00").anchor
        assert anchor is not None
        assert anchor.local_time == dt.time(22, 0)

    def test_saying_nothing_chooses_nothing(self) -> None:
        assert planned(event_anchor="").anchor is None

    def test_an_unreadable_answer_asks_again(self) -> None:
        """Rather than guessing. A spelling nobody recognised turning into an
        instant is how a research decision becomes an invisible one."""
        for spoken in ("whenever", "25:99", "sometime after lunch", "the vibes"):
            assert planned(event_anchor=spoken).anchor is None, spoken

    def test_answering_makes_it_runnable(self) -> None:
        thesis = planned(event_anchor="end of day").to_thesis(
            "q", resolution=midterms()
        )
        assert thesis.anchor is not None
        assert thesis.anchor.policy is AnchorPolicy.DAY_END
        assert not thesis.needs_an_anchor
        assert thesis.is_runnable

    def test_the_choice_reaches_the_measurement(self) -> None:
        """Not just the thesis: the compiled instant moves with it, which is
        the whole reason the question was worth asking."""
        from backtool.events.resolution import compile_event_set

        at_close = compile_event_set(
            midterms(), anchor=DayAnchor(policy=AnchorPolicy.DAY_END)
        )
        at_open = compile_event_set(
            midterms(), anchor=DayAnchor(policy=AnchorPolicy.DAY_START)
        )
        assert at_close.events[0].timestamp_utc != at_open.events[0].timestamp_utc
        assert at_close.fingerprint != at_open.fingerprint

    def test_the_choice_is_carried_into_the_report(self) -> None:
        thesis = dynamic_thesis(
            resolution=midterms(), anchor=DayAnchor(policy=AnchorPolicy.DAY_END)
        )
        html = render_event_provenance(thesis)
        assert "measured from" in html
        assert "end of the calendar day" in html

    def test_a_timestamp_resolution_needs_no_decision(self) -> None:
        thesis = dynamic_thesis(resolution=eth_merge())
        assert not thesis.needs_an_anchor
        assert thesis.anchor is None
        assert thesis.is_runnable

    def test_an_anchor_carries_forward_across_a_follow_up(self) -> None:
        """The reader answers once. A later question about the same event does
        not have to restate it."""
        from backtool.web.app import _anchor_from

        answered = dynamic_thesis(
            resolution=midterms(), anchor=DayAnchor(policy=AnchorPolicy.DAY_END)
        )
        carried = _anchor_from(planned(event_anchor=""), answered)
        assert carried is not None
        assert carried.policy is AnchorPolicy.DAY_END

    def test_a_new_answer_overrides_the_carried_one(self) -> None:
        """So a reader can change their mind."""
        from backtool.web.app import _anchor_from

        answered = dynamic_thesis(
            resolution=midterms(), anchor=DayAnchor(policy=AnchorPolicy.DAY_END)
        )
        carried = _anchor_from(planned(event_anchor="start of day"), answered)
        assert carried is not None
        assert carried.policy is AnchorPolicy.DAY_START

    def test_the_prompt_tells_the_planner_not_to_choose(self) -> None:
        assert "Events dated to the day" in THESIS_PLANNER_SYSTEM
        assert "Do not choose for the reader" in THESIS_PLANNER_SYSTEM
        assert "records a decision; it never makes one" in THESIS_PLANNER_SYSTEM


class TestTheProductUsesTheOptimisedResolver:
    """The orchestration must not quietly get its own, dearer resolver.

    Unit tests passing says nothing about this: the resolver's defaults could be
    perfect while the web layer constructed a client of its own, or passed a
    wider search budget, and every test would still be green. So this asserts
    the seam itself -- that `_resolution_for` reaches `resolve_event_concept`
    with no overrides, and therefore inherits whatever the optimised
    configuration is rather than a copy of it.
    """

    def test_the_orchestration_passes_no_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def stub(*args: object, **kwargs: object) -> ResolvedEventConcept:
            seen.append((args, kwargs))
            return eth_merge()

        monkeypatch.setattr(
            "backtool.ai.event_resolver.resolve_event_concept", stub
        )
        _resolution_for("Some public event", None)

        args, kwargs = seen[0]
        assert args == ("Some public event",)
        # No client, no budget, no model: whatever the resolver decides is what
        # the product gets.
        assert kwargs == {}

    def test_the_budget_the_product_inherits_is_the_optimised_one(self) -> None:
        from backtool.ai import event_resolver

        assert event_resolver.MAX_SEARCHES == 2
        assert event_resolver.search_tool()["max_uses"] == 2
        assert (
            inspect.signature(event_resolver.resolve_event_concept)
            .parameters["max_searches"]
            .default
            == event_resolver.MAX_SEARCHES
        )

    def test_the_models_the_product_inherits_are_the_cheap_pair(self) -> None:
        from backtool.ai import event_resolver

        assert "sonnet" in event_resolver.DEFAULT_SEARCH_MODEL
        assert "haiku" in event_resolver.DEFAULT_EXTRACTION_MODEL

    def test_there_is_exactly_one_resolver_entry_point(self) -> None:
        """A second path is how a configuration drifts out of use. Any module
        that wants a resolution has to come through this function."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "src" / "backtool"
        callers: set[str] = set()
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                named = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if named == "resolve_event_concept":
                    callers.add(path.name)
        assert callers == {"app.py"}

    def test_the_web_layer_builds_no_ai_client_of_its_own(self) -> None:
        """The clients are made inside the resolver, where the per-stage models
        are chosen. One built here would bypass both."""
        import ast
        from pathlib import Path

        web = Path(__file__).resolve().parents[2] / "src" / "backtool" / "web"
        for path in web.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                named = getattr(node.func, "id", None)
                assert named != "AIClient", f"{path.name} builds a client"


class TestTheProseDoesNotCallItARelease:
    """A curated family publishes releases. A protocol upgrade happened once.

    The engine said "release" either way, so a report about the Merge read
    "across 1 release with usable data". Clumsy rather than wrong, and on
    every dynamic answer. The noun is defaulted, so curated wording is
    unchanged byte for byte -- which is what the curated tests go on asserting.
    """

    def test_a_resolved_concept_has_events_not_releases(self) -> None:
        assert dynamic_thesis().event_noun == "event"

    def test_a_curated_family_still_has_releases(self) -> None:
        assert Thesis(question="q").event_noun == "release"

    def test_the_specification_rows_follow(self) -> None:
        rows = dict(dynamic_thesis().describe_items())
        assert rows["Sample"] == "up to the last 40 events"
        assert rows["Condition"] == "none -- every usable event"

    def test_the_curated_rows_are_untouched(self) -> None:
        rows = dict(Thesis(question="q").describe_items())
        assert rows["Sample"] == "up to the last 40 releases"
        assert rows["Condition"] == "none -- every usable release"

    def test_the_summary_follows_too(self) -> None:
        from backtool.research.interpretation import build_facts

        facts = build_facts(_finding())
        assert facts.event_noun == "event"

    def test_the_noun_pluralises(self) -> None:
        from backtool.research.interpretation import _releases

        assert _releases(1, "event") == "event"
        assert _releases(2, "event") == "events"
        assert _releases(1) == "release"
        assert _releases(2) == "releases"


def _finding():  # type: ignore[no-untyped-def]
    from backtool.research.experiment import run_thesis

    return run_thesis(dynamic_thesis())


class TestTheEngineMeasuresIt:
    """The whole point: no new runner, no adapter, no bespoke path."""

    @pytest.fixture(scope="class")
    @classmethod
    def finding(cls):  # type: ignore[no-untyped-def]
        from backtool.research.experiment import run_thesis

        return run_thesis(dynamic_thesis())

    def test_it_produces_a_measurement(self, finding) -> None:  # type: ignore[no-untyped-def]
        assert finding.sample_size == 1
        assert finding.value is not None

    def test_the_spec_records_which_resolution_it_measured(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        """Not the slug. A cached study keyed on the name alone could be served
        for a different set of dates under the same question."""
        assert finding.spec.event_set.startswith("dynamic:ethereum-merge@")
        assert finding.spec.event_type is None

    def test_the_prose_does_not_call_it_a_release(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        """A one-off protocol upgrade is not a release, and the engine's own
        vocabulary should not leak into an answer about one."""
        assert "Ethereum Merge" in finding.cohort.label
