"""Tests that a date stays a date until somebody decides otherwise, on purpose.

Two different things can produce the same instant, and conflating them is the
failure this file exists to prevent:

* **A source fact.** The Ethereum Foundation published 06:42 UTC. Nothing was
  decided; the time is evidence.
* **A research choice.** The Federal Election Commission published 8 November.
  Measuring from it requires an instant the source never gave, so whichever
  instant a study uses was chosen by this project.

A report that cannot tell those apart presents a decision as evidence. Worse,
the decision is consequential: anchoring an election at the start of its day
puts the whole session *after* the event, anchoring at the end puts all of it
*before*, and the same question then returns opposite answers with nothing on
screen to explain why.

So precision travels with the event, an anchor is refused rather than defaulted,
and both participate in identity.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.core.types import EventType, Interval
from backtool.events.calendar import load_events
from backtool.events.models import MarketEvent, TemporalPrecision
from backtool.events.resolution import (
    AnchorPolicy,
    CandidateEvent,
    DayAnchor,
    compile_event_set,
    curated_set_key,
)
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS
from fixtures.dynamic_events import FEC, eth_merge, midterms

CUTOFF = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)

DAY_START = DayAnchor(policy=AnchorPolicy.DAY_START)
DAY_END = DayAnchor(policy=AnchorPolicy.DAY_END)
POLLS_CLOSE = DayAnchor(
    policy=AnchorPolicy.EXPLICIT_TIME, local_time=dt.time(20, 0)
)


def spec_for(events) -> ResearchSpec:  # type: ignore[no-untyped-def]
    return ResearchSpec(
        symbol="BTCUSDT",
        event_type=None,
        event_set=events.identity,
        event_count=10,
        interval=Interval.M5,
        windows=REACTION_WINDOWS,
        as_of=CUTOFF,
    )


class TestSourcedTimestampsAreUntouched:
    """1. An exact timestamp survives resolution and compilation unchanged."""

    def test_a_timestamped_event_needs_no_anchor(self) -> None:
        """Nothing was left open, so nothing has to be decided."""
        merged = compile_event_set(eth_merge())
        assert len(merged.events) == 1

    def test_the_published_instant_is_preserved_exactly(self) -> None:
        event = compile_event_set(eth_merge()).events[0]
        assert event.local_time == dt.time(6, 42)
        assert event.timestamp_utc == dt.datetime(2022, 9, 15, 6, 42, tzinfo=dt.UTC)

    def test_a_sourced_timestamp_carries_no_anchor(self) -> None:
        event = compile_event_set(eth_merge()).events[0]
        assert event.temporal_precision is TemporalPrecision.TIMESTAMP
        assert event.time_is_sourced
        assert event.anchor == ""

    def test_an_anchor_cannot_override_a_sourced_time(self) -> None:
        """Supplying one is harmless, not authoritative. A research choice does
        not get to overrule a published fact."""
        anchored = compile_event_set(eth_merge(), anchor=POLLS_CLOSE)
        assert anchored.events[0].local_time == dt.time(6, 42)
        assert anchored.events[0].anchor == ""
        assert anchored.fingerprint == compile_event_set(eth_merge()).fingerprint

    def test_the_timing_line_says_it_was_published(self) -> None:
        assert "as published" in compile_event_set(eth_merge()).events[0].describe_timing()


class TestDateOnlyStaysVisible:
    """2. A date-only fact remains visibly date-only, before and after."""

    def test_the_candidate_declares_its_precision(self) -> None:
        candidate = midterms().candidate_events[0]
        assert candidate.precision is TemporalPrecision.DATE_ONLY
        assert candidate.is_date_only
        assert candidate.local_time is None

    def test_precision_is_derived_when_omitted(self) -> None:
        timed = CandidateEvent(
            label="x", local_date=dt.date(2022, 11, 8),
            local_time=dt.time(20, 0), timezone="UTC", sources=(FEC,),
        )
        dated = CandidateEvent(
            label="x", local_date=dt.date(2022, 11, 8), timezone="UTC", sources=(FEC,)
        )
        assert timed.precision is TemporalPrecision.TIMESTAMP
        assert dated.precision is TemporalPrecision.DATE_ONLY

    def test_a_declared_precision_is_checked_against_the_time(self) -> None:
        """The same failure arriving from the other direction: a resolver
        claiming a timestamp it never supplied a time for."""
        with pytest.raises(ValueError, match="carries no time of day"):
            CandidateEvent(
                label="x", local_date=dt.date(2022, 11, 8), timezone="UTC",
                sources=(FEC,), precision=TemporalPrecision.TIMESTAMP,
            )
        with pytest.raises(ValueError, match="carries a time of day"):
            CandidateEvent(
                label="x", local_date=dt.date(2022, 11, 8),
                local_time=dt.time(20, 0), timezone="UTC", sources=(FEC,),
                precision=TemporalPrecision.DATE_ONLY,
            )

    def test_the_compiled_event_still_says_date_only(self) -> None:
        """The instant exists because the engine needs one. The record of what
        the source actually established survives it."""
        for event in compile_event_set(midterms(), anchor=DAY_START).events:
            assert event.temporal_precision is TemporalPrecision.DATE_ONLY
            assert not event.time_is_sourced


class TestNoSilentMidnight:
    """3. A date cannot enter a point-anchored study as midnight by default."""

    def test_compilation_refuses_without_an_anchor(self) -> None:
        with pytest.raises(ValueError, match="will not make silently"):
            compile_event_set(midterms())

    def test_the_refusal_names_the_events_it_cannot_time(self) -> None:
        with pytest.raises(ValueError) as caught:
            compile_event_set(midterms())
        message = str(caught.value)
        assert "2018 United States midterm elections" in message
        assert "2022 United States midterm elections" in message
        assert "DayAnchor" in message

    def test_a_date_only_event_cannot_be_hand_built_without_an_anchor(self) -> None:
        """The rule lives on the model, not only in the compiler, so there is
        no second route to an unexplained midnight."""
        with pytest.raises(ValueError, match="research choice rather than a fact"):
            MarketEvent(
                event_id="sneaky",
                event_set_key="dynamic:x",
                local_date=dt.date(2022, 11, 8),
                local_time=dt.time(0, 0),
                timezone="UTC",
                temporal_precision=TemporalPrecision.DATE_ONLY,
            )

    def test_a_sourced_timestamp_cannot_claim_an_anchor(self) -> None:
        """The inverse: an anchor on a sourced time would misreport evidence as
        a decision."""
        with pytest.raises(ValueError, match="would override the source"):
            MarketEvent(
                event_id="x",
                event_set_key="dynamic:x",
                local_date=dt.date(2022, 9, 15),
                local_time=dt.time(6, 42),
                timezone="UTC",
                temporal_precision=TemporalPrecision.TIMESTAMP,
                anchor="start of the calendar day",
            )

    def test_midnight_is_reachable_only_by_asking_for_it(self) -> None:
        """Not that midnight is wrong -- that it has to be chosen."""
        event = compile_event_set(midterms(), anchor=DAY_START).events[0]
        assert event.local_time == dt.time(0, 0)
        assert event.anchor


class TestAnExplicitAnchorCompiles:
    """4. With a stated anchor, a date-only event becomes measurable."""

    @pytest.mark.parametrize(
        ("anchor", "expected_local"),
        [
            (DAY_START, dt.time(0, 0)),
            (DAY_END, dt.time(23, 59, 59)),
            (POLLS_CLOSE, dt.time(20, 0)),
        ],
    )
    def test_each_policy_produces_its_instant(
        self, anchor: DayAnchor, expected_local: dt.time
    ) -> None:
        event = compile_event_set(midterms(), anchor=anchor).events[0]
        assert event.local_time == expected_local
        assert event.timestamp_utc.tzinfo is not None

    def test_the_anchor_is_recorded_on_the_event(self) -> None:
        event = compile_event_set(midterms(), anchor=DAY_END).events[0]
        assert "end of the calendar day" in event.anchor
        assert "America/New_York" in event.anchor

    def test_the_anchor_may_move_the_day_to_another_clock(self) -> None:
        """Anchoring to a market's clock rather than the event's."""
        utc_close = DayAnchor(
            policy=AnchorPolicy.EXPLICIT_TIME,
            local_time=dt.time(21, 0),
            timezone="UTC",
        )
        events = compile_event_set(midterms(), anchor=utc_close).events
        event = next(e for e in events if e.local_date.year == 2022)
        assert event.timezone == "UTC"
        assert event.timestamp_utc == dt.datetime(2022, 11, 8, 21, 0, tzinfo=dt.UTC)

    def test_an_explicit_policy_without_a_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs the local time"):
            DayAnchor(policy=AnchorPolicy.EXPLICIT_TIME)

    def test_a_fixed_policy_rejects_a_time_it_would_ignore(self) -> None:
        with pytest.raises(ValueError, match="would be ignored"):
            DayAnchor(policy=AnchorPolicy.DAY_START, local_time=dt.time(20, 0))

    def test_different_anchors_measure_different_instants(self) -> None:
        """The reason this cannot be a default: the choice changes the answer."""
        start = compile_event_set(midterms(), anchor=DAY_START).events[0]
        end = compile_event_set(midterms(), anchor=DAY_END).events[0]
        assert end.timestamp_utc - start.timestamp_utc == dt.timedelta(
            hours=23, minutes=59, seconds=59
        )


class TestProvenanceSurvivesAnchoring:
    """5. The record still says the source supplied only a date."""

    def test_the_source_is_still_attached(self) -> None:
        for event in compile_event_set(midterms(), anchor=POLLS_CLOSE).events:
            assert event.source is not None
            assert "fec.gov" in event.source.url

    def test_the_timing_line_separates_fact_from_choice(self) -> None:
        line = compile_event_set(midterms(), anchor=POLLS_CLOSE).events[0].describe_timing()
        assert "date only" in line
        assert "anchored at" in line
        assert "20:00" in line
        assert "as published" not in line

    def test_a_reader_can_tell_the_two_kinds_apart(self) -> None:
        sourced = compile_event_set(eth_merge()).events[0]
        chosen = compile_event_set(midterms(), anchor=DAY_START).events[0]
        assert sourced.time_is_sourced
        assert not chosen.time_is_sourced
        assert sourced.describe_timing() != chosen.describe_timing()


class TestIdentityIncludesTheChoice:
    """6. Precision and anchor participate in the fingerprint."""

    def test_different_anchors_give_different_set_fingerprints(self) -> None:
        prints = {
            compile_event_set(midterms(), anchor=a).fingerprint
            for a in (DAY_START, DAY_END, POLLS_CLOSE)
        }
        assert len(prints) == 3

    def test_different_anchors_give_different_spec_fingerprints(self) -> None:
        """The level that actually decides whether a cached study is reused."""
        prints = {
            spec_for(compile_event_set(midterms(), anchor=a)).fingerprint
            for a in (DAY_START, DAY_END, POLLS_CLOSE)
        }
        assert len(prints) == 3

    def test_the_same_anchor_is_stable(self) -> None:
        first = compile_event_set(midterms(), anchor=DAY_END)
        second = compile_event_set(midterms(), anchor=DAY_END)
        assert first.fingerprint == second.fingerprint
        assert spec_for(first).fingerprint == spec_for(second).fingerprint

    def test_a_sourced_instant_and_an_anchored_one_are_not_the_same_fact(self) -> None:
        """The case the brief singles out. Both land on 2022-11-08 20:00 ET;
        one is evidence and one is a decision, and a study of one must never be
        served for the other."""
        anchored = compile_event_set(midterms(), anchor=POLLS_CLOSE)

        timed = midterms().model_copy(
            update={
                "candidate_events": tuple(
                    CandidateEvent(
                        label=c.label,
                        local_date=c.local_date,
                        local_time=dt.time(20, 0),
                        timezone=c.timezone,
                        sources=c.sources,
                    )
                    for c in midterms().candidate_events
                )
            }
        )
        sourced = compile_event_set(timed)

        assert [e.timestamp_utc for e in anchored.events] == [
            e.timestamp_utc for e in sourced.events
        ]
        assert anchored.fingerprint != sourced.fingerprint
        assert spec_for(anchored).fingerprint != spec_for(sourced).fingerprint


class TestCuratedEventsAreUnaffected:
    """7. CPI and FOMC behave exactly as before."""

    def test_curated_events_are_timestamped_and_unanchored(self) -> None:
        """Both are published at a stated time -- 08:30 ET and 14:00 ET -- so
        nothing about them was ever a research choice."""
        for event_type in (EventType.CPI, EventType.FOMC):
            for event in load_events(event_type)[:5]:
                assert event.temporal_precision is TemporalPrecision.TIMESTAMP
                assert event.time_is_sourced
                assert event.anchor == ""

    def test_a_curated_event_still_describes_itself_as_published(self) -> None:
        event = load_events(EventType.CPI)[0]
        assert "as published" in event.describe_timing()

    def test_curated_instants_are_unchanged(self) -> None:
        event = load_events(EventType.CPI)[0]
        assert event.local_time == dt.time(8, 30)
        assert event.timezone == "America/New_York"

    def test_a_curated_spec_fingerprint_does_not_depend_on_anchoring(self) -> None:
        shared = {
            "symbol": "BTCUSDT",
            "event_type": EventType.CPI,
            "event_set": curated_set_key(EventType.CPI),
            "event_count": 10,
            "interval": Interval.M5,
            "windows": REACTION_WINDOWS,
            "as_of": CUTOFF,
        }
        assert ResearchSpec(**shared).fingerprint == ResearchSpec(**shared).fingerprint
