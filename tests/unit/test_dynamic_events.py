"""Tests that externally-resolved event sets fit the deterministic engine.

The claim under test is narrow and load-bearing: **a set of events resolved at
runtime is measured by the same engine, through the same types, as a curated
one.** If that holds, adding "what happened around X" is a resolution problem
rather than an engine problem, and Backtool stops needing every event compiled
in before anybody can ask about it.

Three things have to be true for that claim, and each has a section here.

*Curated families keep their meaning.* ``EventType`` continues to mean a family
with a calendar file, metadata and a classifier. A midterm election does not
pretend to be one, and nothing that depends on a family's machinery can be
reached from a set that has none.

*Identity is content, not name.* A dynamic set is identified by the events it
actually resolved to, so a study cached when "midterms" meant two dates can
never be served for a resolution that found three.

*The engine needs no new path.* ``run_study`` already takes a calendar as a
parameter; a compiled set goes in unchanged. No adapter, no second runner.

No network and no API calls: the resolution is a fixture, which is also how the
later web resolver will be held to this same schema.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.config import Settings
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events.calendar import load_events, select_last_n
from backtool.events.models import MarketEvent, SourceAuthority, SourceRef, TemporalPrecision
from backtool.events.releases import SCENARIO_RULES
from backtool.events.resolution import (
    AnchorPolicy,
    CandidateEvent,
    DayAnchor,
    ResolutionStatus,
    compile_event_set,
    curated_set_key,
    fingerprint,
    slugify,
)
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.vocabulary import CONDITION_FIELDS
from backtool.research.windows import REACTION_WINDOWS
from fixtures.dynamic_events import eth_merge, midterms, unresolvable

CUTOFF = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)


#: The anchor the existing tests compile with. Election days are date-only, so
#: there is no compiling them without one -- which is the point.
DAY_START = DayAnchor(policy=AnchorPolicy.DAY_START)


def compiled(anchor: DayAnchor = DAY_START, **kwargs: object):  # type: ignore[no-untyped-def]
    return compile_event_set(midterms(**kwargs), anchor=anchor)  # type: ignore[arg-type]


class TestCuratedFamiliesAreUnchanged:
    """``EventType`` still means what it meant."""

    def test_curated_events_still_carry_their_type(self) -> None:
        for event in load_events(EventType.CPI)[:3]:
            assert event.event_type is EventType.CPI
            assert event.is_curated
            assert event.curated_type is EventType.CPI

    def test_the_set_key_is_derived_not_stored(self) -> None:
        """So no calendar file had to change."""
        for event_type in (EventType.CPI, EventType.FOMC):
            event = load_events(event_type)[0]
            assert event.event_set_key == curated_set_key(event_type)

    def test_curated_and_dynamic_keys_cannot_collide(self) -> None:
        assert curated_set_key(EventType.CPI).startswith("curated:")
        assert compiled().key.startswith("dynamic:")

    def test_a_curated_study_still_runs(self) -> None:
        """The regression that matters most: nothing above was for free."""
        spec = ResearchSpec(
            symbol="BTCUSDT",
            event_type=EventType.CPI,
            event_set=curated_set_key(EventType.CPI),
            event_count=10,
            interval=Interval.M5,
            windows=REACTION_WINDOWS,
            as_of=CUTOFF,
        )
        with MarketDataService.from_settings(Settings.from_env()) as service:
            study = run_study(spec, service, load_events(EventType.CPI))
        assert len(study.events) == 10


class TestDynamicEventsDoNotPretend:
    """A resolved event is not a curated family wearing its name."""

    def test_a_dynamic_event_has_no_event_type(self) -> None:
        for event in compiled().events:
            assert event.event_type is None
            assert not event.is_curated

    def test_asking_a_dynamic_event_for_a_curated_type_raises(self) -> None:
        """Rather than returning None and letting a classifier lookup silently
        miss. The surfaces built around a family's metadata say so here."""
        with pytest.raises(ValueError, match="not a curated event type"):
            _ = compiled().events[0].curated_type

    def test_an_event_with_no_identity_at_all_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="belongs to no event set"):
            MarketEvent(
                event_id="orphan",
                local_date=dt.date(2022, 11, 8),
                local_time=dt.time(0, 0),
                timezone="UTC",
            )

    def test_every_dynamic_event_carries_provenance(self) -> None:
        """The whole point of keeping the source models from the start."""
        for event in compiled().events:
            assert event.source is not None
            assert event.source.url.startswith("https://")
            assert event.source.authority is SourceAuthority.OFFICIAL

    def test_a_candidate_without_a_source_is_refused(self) -> None:
        """A date with no provenance is exactly what must not reach a study."""
        with pytest.raises(ValueError, match="needs at least one source"):
            CandidateEvent(
                label="made up",
                local_date=dt.date(2022, 11, 8),
                timezone="UTC",
                sources=(),
            )

    def test_an_untimed_date_keeps_its_precision(self) -> None:
        """The instant exists because the engine needs one, but the event still
        records that the source gave only a day."""
        for event in compiled().events:
            assert event.temporal_precision is TemporalPrecision.DATE_ONLY
            assert not event.time_is_sourced
            assert event.anchor

    def test_a_timed_event_keeps_its_timestamp(self) -> None:
        merge = compile_event_set(eth_merge())
        assert len(merge.events) == 1
        assert merge.events[0].local_time == dt.time(6, 42)
        assert merge.events[0].timestamp_utc == dt.datetime(
            2022, 9, 15, 6, 42, tzinfo=dt.UTC
        )


class TestNoCuratedMachineryLeaks:
    """A set with no family cannot reach a family's machinery."""

    def test_a_dynamic_set_has_no_scenario_classifier(self) -> None:
        """`.get`, not `[]`: the absence is ordinary. Nobody publishes a
        consensus forecast for an election."""
        assert SCENARIO_RULES.get(None) is None
        assert SCENARIO_RULES.get(EventType.CPI) is not None

    def test_event_specific_conditions_are_unavailable(self) -> None:
        """A CPI surprise for a midterm election would select nothing and read
        like a result."""
        assert not CONDITION_FIELDS["cpi_surprise"].applies_to(None)

    def test_event_agnostic_conditions_remain_available(self) -> None:
        """The BTC regime conditions describe the market, not the event, so
        they are exactly as meaningful around an election."""
        for key in ("btc_trend_7d", "btc_vol_regime", "btc_vs_200d_ma", "btc_return_7d"):
            assert CONDITION_FIELDS[key].applies_to(None), key

    def test_curated_conditions_still_apply_to_their_family(self) -> None:
        assert CONDITION_FIELDS["cpi_surprise"].applies_to(EventType.CPI)
        assert not CONDITION_FIELDS["cpi_surprise"].applies_to(EventType.FOMC)


class TestSetIdentity:
    """Identity is the events resolved, not the name they were resolved from."""

    def test_the_same_resolution_fingerprints_the_same(self) -> None:
        assert compiled().fingerprint == compiled().fingerprint

    def test_order_does_not_matter(self) -> None:
        """A calendar is a set. A source listing the same elections in a
        different order has not found different elections."""
        forward = compiled(dates=(dt.date(2018, 11, 6), dt.date(2022, 11, 8)))
        reversed_ = compiled(dates=(dt.date(2022, 11, 8), dt.date(2018, 11, 6)))
        assert forward.fingerprint == reversed_.fingerprint

    def test_a_changed_date_changes_the_fingerprint(self) -> None:
        shifted = compiled(dates=(dt.date(2018, 11, 6), dt.date(2022, 11, 9)))
        assert shifted.fingerprint != compiled().fingerprint

    def test_an_added_event_changes_the_fingerprint(self) -> None:
        wider = compiled(
            dates=(dt.date(2014, 11, 4), dt.date(2018, 11, 6), dt.date(2022, 11, 8))
        )
        assert wider.fingerprint != compiled().fingerprint

    def test_a_removed_event_changes_the_fingerprint(self) -> None:
        narrower = compiled(dates=(dt.date(2022, 11, 8),))
        assert narrower.fingerprint != compiled().fingerprint

    def test_two_concepts_sharing_a_label_do_not_collide(self) -> None:
        """Identity is content plus concept, never the label alone."""
        same_name_other_dates = compiled(dates=(dt.date(2010, 11, 2),))
        assert same_name_other_dates.fingerprint != compiled().fingerprint

    def test_two_concepts_with_the_same_dates_stay_distinct(self) -> None:
        """"Midterm elections" and "general elections" resolving to the same
        day are still different questions."""
        other = compiled(canonical_name="United States general elections")
        assert other.key != compiled().key
        assert other.fingerprint != compiled().fingerprint

    def test_identity_carries_both_halves(self) -> None:
        """The key says which concept; the fingerprint says which resolution."""
        identity = compiled().identity
        assert identity.startswith("dynamic:united-states-midterm-elections@")
        assert compiled().fingerprint in identity

    def test_the_fingerprint_is_the_utc_instant_not_the_wall_clock(self) -> None:
        """The same moment written in two timezones is one event, and must not
        fingerprint as two."""
        utc = MarketEvent(
            event_id="a",
            event_set_key="dynamic:x",
            local_date=dt.date(2022, 9, 15),
            local_time=dt.time(6, 42),
            timezone="UTC",
            notes="same moment",
        )
        london = MarketEvent(
            event_id="b",
            event_set_key="dynamic:x",
            local_date=dt.date(2022, 9, 15),
            local_time=dt.time(7, 42),
            timezone="Europe/London",
            notes="same moment",
        )
        assert utc.timestamp_utc == london.timestamp_utc
        assert fingerprint("x", (utc,)) == fingerprint("x", (london,))

    def test_slugs_are_stable_and_readable(self) -> None:
        assert (
            slugify("United States midterm elections")
            == "united-states-midterm-elections"
        )
        assert slugify("  The Ethereum Merge!  ") == "the-ethereum-merge"
        assert slugify("???") == "unnamed"


class TestCompilationRefusesWhatCannotBeMeasured:
    """An open question is for the reader, not for the engine."""

    def test_an_ambiguous_concept_does_not_compile(self) -> None:
        with pytest.raises(ValueError, match="resolved as ambiguous"):
            compile_event_set(unresolvable())

    def test_a_concept_with_no_dates_does_not_compile(self) -> None:
        with pytest.raises(ValueError, match="no dates"):
            compile_event_set(midterms(dates=()))

    def test_a_resolved_concept_with_open_questions_does_not_compile(self) -> None:
        """Status alone is not enough -- an unanswered question still blocks."""
        from backtool.events.resolution import Ambiguity

        with pytest.raises(ValueError, match="still has open questions"):
            compile_event_set(
                midterms(
                    ambiguities=(Ambiguity(question="Which midterms?"),),
                )
            )

    def test_a_not_found_concept_does_not_compile(self) -> None:
        with pytest.raises(ValueError, match="resolved as not found"):
            compile_event_set(midterms(status=ResolutionStatus.NOT_FOUND))

    def test_runnable_distinguishes_the_states(self) -> None:
        assert midterms().is_runnable
        assert not unresolvable().is_runnable
        assert not midterms(status=ResolutionStatus.CONFLICTING).is_runnable


class TestSelectionCannotCrossSets:
    """A study measures its own events and nothing else."""

    def test_the_set_key_filters(self) -> None:
        mixed = [*load_events(EventType.CPI), *compiled().events]
        picked = select_last_n(
            mixed, 10, before=CUTOFF, event_set_key="dynamic:united-states-midterm-elections"
        )
        assert picked
        assert all(e.event_set_key == "dynamic:united-states-midterm-elections" for e in picked)

    def test_a_curated_filter_excludes_dynamic_events(self) -> None:
        mixed = [*load_events(EventType.CPI), *compiled().events]
        picked = select_last_n(mixed, 50, before=CUTOFF, event_type=EventType.CPI)
        assert picked
        assert all(e.event_type is EventType.CPI for e in picked)


class TestTheEngineConsumesItUnchanged:
    """The claim: no new runner, no adapter, no bespoke path."""

    @pytest.fixture(scope="class")
    @classmethod
    def study(cls):  # type: ignore[no-untyped-def]
        events = compiled()
        spec = ResearchSpec(
            symbol="BTCUSDT",
            # No curated family. The set identity carries the question instead.
            event_type=None,
            event_set=events.identity,
            event_count=10,
            interval=Interval.M5,
            windows=REACTION_WINDOWS,
            as_of=CUTOFF,
        )
        with MarketDataService.from_settings(Settings.from_env()) as service:
            return run_study(spec, service, events.calendar)

    def test_the_existing_runner_accepts_a_dynamic_calendar(
        self, study
    ) -> None:  # type: ignore[no-untyped-def]
        """``run_study`` already took the calendar as a parameter. That seam is
        why this iteration is small."""
        assert study.events

    def test_it_measured_the_resolved_dates(
        self, study
    ) -> None:  # type: ignore[no-untyped-def]
        measured = {event.local_date for event in study.events}
        assert measured == {dt.date(2018, 11, 6), dt.date(2022, 11, 8)}

    def test_every_window_was_attempted(
        self, study
    ) -> None:  # type: ignore[no-untyped-def]
        """Including the ones with no candle data. An event outside the cached
        history is reported as unusable rather than dropped -- the same
        accounting a curated study gets."""
        for event in study.events:
            assert set(event.windows) == {w.name for w in REACTION_WINDOWS}

    def test_the_spec_identity_records_which_resolution_ran(
        self, study
    ) -> None:  # type: ignore[no-untyped-def]
        assert study.spec.event_set.startswith(
            "dynamic:united-states-midterm-elections@"
        )
        assert study.spec.event_type is None

    def test_a_different_resolution_gets_a_different_spec_fingerprint(self) -> None:
        """The cache-correctness property, at the level that actually caches."""
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

        two = spec_for(compiled())
        three = spec_for(
            compiled(
                dates=(dt.date(2014, 11, 4), dt.date(2018, 11, 6), dt.date(2022, 11, 8))
            )
        )
        assert two.fingerprint != three.fingerprint

    def test_the_same_resolution_gets_the_same_spec_fingerprint(self) -> None:
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

        assert spec_for(compiled()).fingerprint == spec_for(compiled()).fingerprint

    def test_a_curated_and_a_dynamic_spec_never_share_a_fingerprint(self) -> None:
        shared = {
            "symbol": "BTCUSDT",
            "event_count": 10,
            "interval": Interval.M5,
            "windows": REACTION_WINDOWS,
            "as_of": CUTOFF,
        }
        curated = ResearchSpec(
            event_type=EventType.CPI, event_set=curated_set_key(EventType.CPI), **shared
        )
        dynamic = ResearchSpec(
            event_type=None, event_set=compiled().identity, **shared
        )
        assert curated.fingerprint != dynamic.fingerprint

    def test_the_report_can_name_a_dynamic_set(self) -> None:
        """Display code must not reach into an enum a dynamic study lacks."""
        spec = ResearchSpec(event_type=None, event_set=compiled().identity)
        assert spec.event_label == "united-states-midterm-elections"
        assert ResearchSpec(event_type=EventType.CPI).event_label == "CPI"


class TestProvenanceSurvivesCompilation:
    """A report has to be able to show its working."""

    def test_the_set_keeps_its_sources(self) -> None:
        events = compiled()
        assert events.sources
        assert all(src.url for src in events.sources)

    def test_sources_are_deduplicated(self) -> None:
        """One citation per source, however many events it dated."""
        urls = [src.url for src in compiled().sources]
        assert len(urls) == len(set(urls))

    def test_a_source_describes_itself_for_a_citation_line(self) -> None:
        source = SourceRef(
            url="https://www.fec.gov/",
            title="FEC",
            publisher="US Federal Election Commission",
            authority=SourceAuthority.OFFICIAL,
        )
        described = source.describe()
        assert "Federal Election Commission" in described
        assert "official" in described
        assert "https://www.fec.gov/" in described

    def test_resolution_notes_survive(self) -> None:
        assert "Federal Election Commission" in compiled().resolution_notes

    def test_retrieval_time_survives(self) -> None:
        """A resolved date is a fact as of a moment, not a permanent one."""
        assert compiled().retrieved_at.year == 2026
