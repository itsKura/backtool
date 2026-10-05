"""Tests for the thesis layer: vocabulary, construction, selection, results.

The rule under test throughout is that a question either compiles into
something the engine can measure, or is refused with a reason. There is no
third path where an unsupported concept becomes an approximation.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.core.types import EventType
from backtool.events.releases import classify_cpi, load_releases
from backtool.research.experiment import ResearchFinding, run_thesis
from backtool.research.thesis import (
    Comparison,
    Condition,
    Outcome,
    Thesis,
    unsupported_reason,
)
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    OUTCOME_HORIZONS,
    Aggregation,
    Operator,
    describe_vocabulary,
)

COOLER_CPI = Thesis(
    question="Does BTC perform better after cooler-than-expected CPI releases?",
    asset="BTCUSDT",
    event_type=EventType.CPI,
    conditions=(Condition(field="cpi_surprise", value="cooler"),),
    outcome=Outcome(metric="return_pct", horizon="24h", aggregation=Aggregation.MEDIAN),
    # Explicit now that a baseline is optional: this thesis is a comparison,
    # and saying so beats a default that quietly decides what is being asked.
    versus=Comparison(),
)


class TestVocabulary:
    def test_every_field_declares_what_it_can_be_compared_against(self) -> None:
        """A categorical field declares its states; a numeric one declares its
        range. A field declaring neither would accept anything, which is the
        one thing a controlled vocabulary must not do."""
        for spec in CONDITION_FIELDS.values():
            assert spec.label and spec.source, spec.key
            if spec.is_numeric:
                assert spec.bounds, spec.key
                assert not spec.values, spec.key
            else:
                assert spec.values, spec.key
                assert spec.bounds is None, spec.key

    def test_the_planner_is_told_exactly_what_exists(self) -> None:
        """The prompt's vocabulary is generated from the same tables the
        validator checks, so the model cannot be told about a field the engine
        does not have."""
        described = describe_vocabulary()
        assert set(described["horizons"]) == set(OUTCOME_HORIZONS)
        for spec in CONDITION_FIELDS.values():
            assert any(entry.startswith(spec.key) for entry in described["condition_fields"])

    def test_cpi_surprise_only_applies_to_cpi(self) -> None:
        spec = CONDITION_FIELDS["cpi_surprise"]
        assert spec.applies_to(EventType.CPI)
        assert not spec.applies_to(EventType.FOMC)

    def test_market_conditions_apply_to_every_event_type(self) -> None:
        for key in ("btc_trend_7d", "btc_vol_regime", "btc_vs_200d_ma"):
            assert CONDITION_FIELDS[key].applies_to(EventType.FOMC)


class TestThesisConstruction:
    def test_the_supported_question_builds(self) -> None:
        assert COOLER_CPI.is_runnable
        assert COOLER_CPI.outcome.window == "post_event_24h"

    def test_an_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown condition field"):
            Condition(field="funding_rate", value="negative")

    def test_an_unknown_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be"):
            Condition(field="cpi_surprise", value="slightly warm")

    def test_an_unsupported_asset_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported asset"):
            Thesis(asset="SPY")

    def test_an_unsupported_horizon_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported horizon"):
            Outcome(horizon="3d")

    def test_a_cpi_condition_cannot_be_tested_against_fomc(self) -> None:
        """Forcing it would produce a cohort of zero and look like a result."""
        with pytest.raises(ValueError, match="cannot be tested against"):
            Thesis(
                event_type=EventType.FOMC,
                conditions=(Condition(field="cpi_surprise", value="cooler"),),
            )

    def test_an_unresolved_thesis_is_not_runnable(self) -> None:
        pending = COOLER_CPI.model_copy(
            update={"unresolved": ("What counts as recovering?",)}
        )
        assert not pending.is_runnable
        with pytest.raises(ValueError, match="still needs"):
            run_thesis(pending)

    def test_regime_work_is_skipped_when_nothing_asks_for_it(self) -> None:
        """A question about CPI surprise alone should not pay for two years of
        hourly history it will not look at."""
        assert not COOLER_CPI.needs_regime
        assert COOLER_CPI.with_condition(
            Condition(field="btc_trend_7d", value="bearish")
        ).needs_regime

    def test_unsupported_concepts_are_named_not_guessed(self) -> None:
        assert "not connected" in (unsupported_reason(asset="SPY") or "")
        assert "calendar" in (unsupported_reason(event="NFP") or "")
        assert "yet" in (unsupported_reason(field_key="open_interest") or "")
        assert unsupported_reason(asset="BTCUSDT") is None


class TestFollowUps:
    def test_a_follow_up_modifies_rather_than_replaces(self) -> None:
        """'What about only when BTC was down that week?' is a change to the
        question on screen, not a new one."""
        follow_up = COOLER_CPI.with_condition(
            Condition(field="btc_trend_7d", value="bearish")
        )
        assert len(follow_up.conditions) == 2
        assert follow_up.event_type is COOLER_CPI.event_type
        assert follow_up.outcome == COOLER_CPI.outcome
        fields = {c.field for c in follow_up.conditions}
        assert fields == {"cpi_surprise", "btc_trend_7d"}

    def test_a_condition_on_the_same_field_replaces_rather_than_stacks(self) -> None:
        """Two filters on one field are either redundant or contradictory."""
        changed = COOLER_CPI.with_condition(
            Condition(field="cpi_surprise", value="hotter")
        )
        assert len(changed.conditions) == 1
        assert changed.conditions[0].value == "hotter"

    def test_a_condition_can_be_dropped(self) -> None:
        assert COOLER_CPI.without_condition("cpi_surprise").conditions == ()

    def test_the_original_is_never_mutated(self) -> None:
        COOLER_CPI.with_condition(Condition(field="btc_trend_7d", value="bullish"))
        assert len(COOLER_CPI.conditions) == 1


class TestConditionMatching:
    def test_an_unmeasurable_value_never_matches_either_way(self) -> None:
        """A release whose 200-day average could not be computed is not 'not
        above' it -- it is unknown, and treating unknown as the negative case
        fills a cohort with releases selected for having less data."""
        is_above = Condition(field="btc_vs_200d_ma", value="above")
        is_not_above = Condition(
            field="btc_vs_200d_ma", operator=Operator.IS_NOT, value="above"
        )
        assert not is_above.matches(None)
        assert not is_not_above.matches(None)

    def test_negation_works(self) -> None:
        condition = Condition(
            field="cpi_surprise", operator=Operator.IS_NOT, value="cooler"
        )
        assert condition.matches("hotter")
        assert not condition.matches("cooler")


class TestCoolerCPIEndToEnd:
    """The milestone question, against the real dataset and real candles."""

    @pytest.fixture(scope="class")
    @classmethod
    def finding(cls) -> ResearchFinding:
        return run_thesis(COOLER_CPI)

    def test_it_produces_a_finding(self, finding: ResearchFinding) -> None:
        assert finding.cohort.sample_size > 0
        assert finding.comparison.sample_size > finding.cohort.sample_size

    def test_the_cohort_is_exactly_the_cooler_releases(
        self, finding: ResearchFinding
    ) -> None:
        values = load_releases()
        for event_id in finding.cohort.event_ids:
            verdict = classify_cpi(values[event_id])
            assert verdict is not None
            assert verdict[0] == "cooler"

    def test_no_unverified_release_enters_the_cohort(
        self, finding: ResearchFinding
    ) -> None:
        """Scenario classification requires a verified, undisputed consensus.
        A release without one has an unknown surprise, and unknown is not
        'cooler'."""
        values = load_releases()
        for event_id in finding.cohort.event_ids:
            release = values[event_id]
            assert release.consensus_verified
            assert not release.consensus_conflict
            assert release.consensus_yoy is not None

    def test_the_baseline_is_every_usable_release(
        self, finding: ResearchFinding
    ) -> None:
        assert set(finding.cohort.event_ids) <= set(finding.comparison.event_ids)

    def test_the_difference_is_the_two_medians(
        self, finding: ResearchFinding
    ) -> None:
        cohort = finding.cohort_metrics
        baseline = finding.comparison_metrics
        assert cohort is not None and baseline is not None
        assert finding.difference_pp == pytest.approx(
            cohort.median_return_pct - baseline.median_return_pct
        )

    def test_every_observation_reports_why_it_qualified(
        self, finding: ResearchFinding
    ) -> None:
        for observation in finding.observations:
            assert observation.conditions["CPI surprise"] == "cooler than expected"
            assert "24h" in observation.returns

    def test_exclusions_are_counted_by_reason(
        self, finding: ResearchFinding
    ) -> None:
        assert finding.excluded
        assert sum(finding.excluded.values()) > 0

    def test_the_baseline_overlap_is_disclosed(
        self, finding: ResearchFinding
    ) -> None:
        """The comparison group here is every usable release, so it contains
        the cohort. Saying so is the difference between a comparison a reader
        can judge and one they have to reverse-engineer."""
        assert any("including the ones selected here" in w for w in finding.warnings)

    def test_the_verdict_is_deterministic_not_written(
        self, finding: ResearchFinding
    ) -> None:
        """The headline sentence must be derivable from the two numbers above
        it, so the language layer can elaborate but never replace it."""
        verdict = finding.verdict()
        difference = finding.difference_pp
        assert difference is not None
        if abs(difference) < 0.1:
            assert "No meaningful difference" in verdict
        else:
            assert ("better" in verdict) == (difference > 0)

    def test_the_result_carries_a_reproducible_specification(
        self, finding: ResearchFinding
    ) -> None:
        assert finding.spec.fingerprint
        assert finding.spec.symbol == "BTCUSDT"
        assert finding.spec.event_type is EventType.CPI


class TestDeterminism:
    def test_wording_does_not_change_the_numbers(self) -> None:
        """Two differently-worded questions that compile to the same thesis
        must produce the same figures. The engine sees the thesis, never the
        sentence."""
        other = COOLER_CPI.model_copy(
            update={"question": "cooler CPI, what happens to bitcoin in a day?"}
        )
        first = run_thesis(COOLER_CPI)
        second = run_thesis(other, as_of=first.spec.as_of)

        assert first.spec.fingerprint == second.spec.fingerprint
        assert first.cohort.event_ids == second.cohort.event_ids
        assert first.difference_pp == second.difference_pp

    def test_a_pinned_cutoff_makes_a_run_reproducible(self) -> None:
        cutoff = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
        a = run_thesis(COOLER_CPI, as_of=cutoff)
        b = run_thesis(COOLER_CPI, as_of=cutoff)
        assert a.cohort.event_ids == b.cohort.event_ids
        assert a.cohort_metrics == b.cohort_metrics


class TestComparisonGroups:
    """A baseline is a group with no filters, not a separate concept."""

    def test_an_unfiltered_group_is_every_usable_release(self) -> None:
        assert Comparison().describe(EventType.CPI) == "All usable CPI releases"
        assert Comparison().is_baseline

    def test_a_filtered_group_is_named_by_its_filters(self) -> None:
        group = Comparison(
            conditions=(Condition(field="cpi_surprise", value="cooler"),)
        )
        assert group.describe(EventType.CPI) == "CPI surprise is cooler than expected"
        assert not group.is_baseline
