"""Tests for numeric conditions, and above all for their units.

The failure this file exists to prevent is silent and total: a threshold read
as ``-0.03`` when the reader said "down 3%" selects almost every release
instead of a handful, produces a plausible-looking cohort, and gives a
confident answer to a question nobody asked. Nothing about the output would
look wrong.

So the unit is pinned from three directions -- what a condition matches, what it
renders back to the reader, and what the planner is allowed to emit -- and the
sign is tested on both sides of zero, because ``<`` and ``>`` are not
symmetric around a negative threshold and that is where an off-by-one in a
comparison hides.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.ai.thesis_planner import PlannedCondition, PlannedThesis
from backtool.core.types import EventType
from backtool.research.regime import MarketSetup, Trend
from backtool.research.thesis import Condition, Outcome, Thesis
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    NUMERIC_FIELDS,
    Operator,
    ResolutionContext,
    describe_vocabulary,
)

FELL_3_PCT = Condition(field="btc_return_7d", operator=Operator.BELOW, threshold=-3.0)
GAINED_5_PCT = Condition(
    field="btc_return_30d", operator=Operator.AT_LEAST, threshold=5.0
)


class TestUnits:
    """One canonical scale: the number a reader says out loud."""

    def test_a_percent_threshold_is_the_spoken_number(self) -> None:
        """-3 means a three percent fall, so a 4.6% fall qualifies and a 0.04%
        fall does not. Under the fraction reading both would qualify."""
        assert FELL_3_PCT.matches_number(-4.6)
        assert not FELL_3_PCT.matches_number(-0.04)

    def test_the_fraction_reading_would_have_been_caught(self) -> None:
        """If -3 were ever read as -0.03, this is the assertion that fails:
        a flat week would start qualifying as a 3% drop."""
        assert not FELL_3_PCT.matches_number(-0.01)
        assert not FELL_3_PCT.matches_number(0.0)

    def test_a_threshold_outside_the_measurement_is_refused(self) -> None:
        """A 7-day return cannot be -400%. Rejecting it surfaces a unit or
        parse mistake instead of selecting nothing."""
        with pytest.raises(ValueError, match="outside the range"):
            Condition(field="btc_return_7d", operator=Operator.BELOW, threshold=-400.0)

    def test_a_percentile_is_zero_to_one_hundred(self) -> None:
        """67 means the 67th percentile, the same cut btc_vol_regime calls
        'high'. A reading expressed as a fraction would never clear it."""
        high = Condition(
            field="btc_vol_percentile", operator=Operator.ABOVE, threshold=67.0
        )
        assert high.matches_number(80.0)
        assert not high.matches_number(0.8)
        with pytest.raises(ValueError, match="outside the range"):
            Condition(
                field="btc_vol_percentile", operator=Operator.ABOVE, threshold=150.0
            )

    def test_the_threshold_is_rendered_back_with_its_unit(self) -> None:
        """The reader's only defence against a unit mistake is seeing the
        figure before anything runs."""
        assert FELL_3_PCT.describe() == "BTC 7-day return is below -3%"
        assert GAINED_5_PCT.describe() == "BTC 30-day return is at least +5%"

    def test_a_percentile_renders_unsigned(self) -> None:
        """'+67' would read as a change rather than a rank."""
        condition = Condition(
            field="btc_vol_percentile", operator=Operator.ABOVE, threshold=67.0
        )
        assert condition.describe() == "BTC volatility percentile is above 67"

    def test_the_planner_is_told_the_unit_rule(self) -> None:
        described = describe_vocabulary()
        rule = " ".join(described["units"]).lower()
        assert "0.03" in rule
        for key in NUMERIC_FIELDS:
            assert any(
                key in entry and "number" in entry
                for entry in described["condition_fields"]
            )


class TestSignedComparison:
    """Both sides of zero, because a negative threshold inverts the intuition."""

    @pytest.mark.parametrize(
        ("operator", "threshold", "reading", "expected"),
        [
            # Below a negative threshold: "fell more than 3%".
            (Operator.BELOW, -3.0, -3.01, True),
            (Operator.BELOW, -3.0, -3.0, False),
            (Operator.BELOW, -3.0, -2.99, False),
            (Operator.BELOW, -3.0, 10.0, False),
            # At most a negative threshold includes the boundary.
            (Operator.AT_MOST, -3.0, -3.0, True),
            (Operator.AT_MOST, -3.0, -2.99, False),
            # Above a positive threshold: "gained more than 5%".
            (Operator.ABOVE, 5.0, 5.01, True),
            (Operator.ABOVE, 5.0, 5.0, False),
            (Operator.ABOVE, 5.0, -20.0, False),
            (Operator.AT_LEAST, 5.0, 5.0, True),
            (Operator.AT_LEAST, 5.0, 4.99, False),
            # Across zero: "7-day return below zero" is a fall of any size.
            (Operator.BELOW, 0.0, -0.01, True),
            (Operator.BELOW, 0.0, 0.0, False),
            (Operator.BELOW, 0.0, 0.01, False),
        ],
    )
    def test_boundaries_are_exact(
        self, operator: Operator, threshold: float, reading: float, expected: bool
    ) -> None:
        condition = Condition(
            field="btc_return_7d", operator=operator, threshold=threshold
        )
        assert condition.matches_number(reading) is expected

    def test_an_unmeasured_reading_never_matches_either_way(self) -> None:
        """A release too early for a 7-day return is not 'below -3%' and not
        'above -3%'. Admitting it to the negative case would fill a cohort with
        releases selected for having less history."""
        below = Condition(field="btc_return_7d", operator=Operator.BELOW, threshold=-3.0)
        above = Condition(field="btc_return_7d", operator=Operator.ABOVE, threshold=-3.0)
        assert not below.matches_number(None)
        assert not above.matches_number(None)


class TestOperatorFieldPairing:
    """A mismatched pair selects nothing, which looks like a finding."""

    def test_a_number_cannot_be_compared_to_a_named_state(self) -> None:
        with pytest.raises(ValueError, match="named state, not a number"):
            Condition(field="cpi_surprise", operator=Operator.BELOW, threshold=-3.0)

    def test_a_named_state_cannot_be_compared_to_a_number(self) -> None:
        with pytest.raises(ValueError, match="is a number"):
            Condition(field="btc_return_7d", value="bearish")

    def test_a_numeric_condition_needs_a_threshold(self) -> None:
        with pytest.raises(ValueError, match="needs a number"):
            Condition(field="btc_return_7d", operator=Operator.BELOW)

    def test_a_categorical_condition_rejects_a_stray_threshold(self) -> None:
        with pytest.raises(ValueError, match="nothing to compare against"):
            Condition(field="cpi_surprise", value="cooler", threshold=3.0)

    def test_a_numeric_condition_rejects_a_stray_value(self) -> None:
        with pytest.raises(ValueError, match="carries a threshold"):
            Condition(
                field="btc_return_7d",
                operator=Operator.BELOW,
                threshold=-3.0,
                value="bearish",
            )


class TestRanges:
    """Two bounds on one field, which is the only honest form of a range."""

    def test_a_lower_and_an_upper_bound_coexist(self) -> None:
        """'Down between 2% and 5% over the week' is a conjunction. If adding
        the second bound evicted the first, this would answer a different
        question with no sign that it had."""
        thesis = (
            Thesis(event_type=EventType.CPI)
            .with_condition(
                Condition(
                    field="btc_return_7d", operator=Operator.AT_LEAST, threshold=-5.0
                )
            )
            .with_condition(
                Condition(
                    field="btc_return_7d", operator=Operator.AT_MOST, threshold=-2.0
                )
            )
        )
        assert len(thesis.conditions) == 2
        assert all(c.matches_number(-3.5) for c in thesis.conditions)
        assert not all(c.matches_number(-6.0) for c in thesis.conditions)
        assert not all(c.matches_number(-1.0) for c in thesis.conditions)

    def test_a_bound_replaces_one_on_the_same_end(self) -> None:
        """'Actually, more than 5%' changes the lower bound rather than adding
        a second contradictory one."""
        thesis = (
            Thesis()
            .with_condition(
                Condition(
                    field="btc_return_7d", operator=Operator.BELOW, threshold=-3.0
                )
            )
            .with_condition(
                Condition(
                    field="btc_return_7d", operator=Operator.AT_MOST, threshold=-5.0
                )
            )
        )
        assert len(thesis.conditions) == 1
        assert thesis.conditions[0].threshold == -5.0

    def test_removing_the_field_removes_both_bounds(self) -> None:
        thesis = (
            Thesis()
            .with_condition(
                Condition(
                    field="btc_return_7d", operator=Operator.AT_LEAST, threshold=-5.0
                )
            )
            .with_condition(
                Condition(
                    field="btc_return_7d", operator=Operator.AT_MOST, threshold=-2.0
                )
            )
        )
        assert thesis.without_condition("btc_return_7d").conditions == ()

    def test_a_categorical_condition_still_replaces_on_the_field(self) -> None:
        """The existing rule is unchanged: two named states on one field are
        contradictory, not a range."""
        thesis = Thesis(
            conditions=(Condition(field="cpi_surprise", value="cooler"),)
        ).with_condition(Condition(field="cpi_surprise", value="hotter"))
        assert len(thesis.conditions) == 1
        assert thesis.conditions[0].value == "hotter"


class TestResolution:
    """The readings come from the regime pass, which already computed them."""

    def test_a_setup_exposes_the_figures_behind_its_buckets(self) -> None:
        """-4.6% and -28% are both 'bearish'. A numeric condition needs the
        figure, and it is produced by the same pass either way."""
        setup = MarketSetup(
            as_of=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
            return_7d_pct=-4.6,
            trend_7d=Trend.BEARISH,
            vol_percentile=72.0,
        )
        assert setup.readings["return_7d_pct"] == -4.6
        assert setup.readings["vol_percentile"] == 72.0
        assert setup.measured["trend_7d"] == "bearish"

    def test_an_unmeasured_figure_is_absent_not_zero(self) -> None:
        setup = MarketSetup(as_of=dt.datetime(2024, 6, 1, tzinfo=dt.UTC))
        assert setup.readings == {}

    def test_the_context_maps_thesis_fields_to_regime_readings(self) -> None:
        context = ResolutionContext(
            readings={"CPI-2024-01-11": {"return_7d_pct": -4.6, "vol_percentile": 72.0}}
        )
        assert context.resolve_number("btc_return_7d", "CPI-2024-01-11") == -4.6
        assert context.resolve_number("btc_vol_percentile", "CPI-2024-01-11") == 72.0
        assert context.resolve_number("btc_return_30d", "CPI-2024-01-11") is None
        assert context.resolve_number("btc_return_7d", "CPI-2099-01-01") is None

    def test_a_categorical_field_has_no_numeric_reading(self) -> None:
        context = ResolutionContext(surprises={"CPI-2024-01-11": "cooler"})
        assert context.resolve_number("cpi_surprise", "CPI-2024-01-11") is None

    def test_every_numeric_field_requires_the_regime_pass(self) -> None:
        """All three read daily closes, so a thesis using one must trigger the
        history fetch rather than silently resolving to nothing."""
        for key in NUMERIC_FIELDS:
            assert CONDITION_FIELDS[key].requires_regime, key
            thesis = Thesis(
                conditions=(
                    Condition(
                        field=key, operator=Operator.ABOVE, threshold=0.0
                    ),
                )
            )
            assert thesis.needs_regime


class TestPlannerThresholds:
    """What the planner may emit, and what it may not."""

    @staticmethod
    def _planned(**condition: str) -> PlannedThesis:
        return PlannedThesis(
            restated_question="q",
            asset="BTCUSDT",
            event_type="CPI",
            conditions=[PlannedCondition(**condition)],
            outcome_metric="return",
            outcome_horizon="24h",
            assumptions=[],
            clarifications=[],
            unsupported=[],
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("-3", -3.0),
            ("-3%", -3.0),
            (" -3.5 ", -3.5),
            ("5", 5.0),
            ("−3", -3.0),  # unicode minus; float() rejects it unaided
            ("1,000", 1000.0),
        ],
    )
    def test_a_threshold_is_read_as_written(self, raw: str, expected: float) -> None:
        thesis = self._planned(
            field="btc_return_7d", operator="<", value="", threshold=raw
        ).to_thesis("q")
        assert thesis.conditions[0].threshold == expected

    def test_a_fraction_is_not_rescaled_into_a_percent(self) -> None:
        """-0.03 means three hundredths of a percent, because that is what it
        says. Guessing it meant -3 would redefine the question silently; the
        specification card shows the figure instead."""
        thesis = self._planned(
            field="btc_return_7d", operator="<", value="", threshold="-0.03"
        ).to_thesis("q")
        assert thesis.conditions[0].threshold == -0.03
        assert "-0.03%" in thesis.conditions[0].describe()

    def test_a_threshold_that_is_not_a_number_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not one"):
            self._planned(
                field="btc_return_7d", operator="<", value="", threshold="a lot"
            ).to_thesis("q")

    def test_word_operators_are_understood(self) -> None:
        thesis = self._planned(
            field="btc_return_30d", operator="at least", value="", threshold="5"
        ).to_thesis("q")
        assert thesis.conditions[0].operator is Operator.AT_LEAST

    def test_a_numeric_operator_on_a_categorical_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match="named state, not a number"):
            self._planned(
                field="cpi_surprise", operator="<", value="", threshold="-3"
            ).to_thesis("q")

    def test_a_categorical_operator_on_a_numeric_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is a number"):
            self._planned(
                field="btc_return_7d", operator="is", value="bearish", threshold=""
            ).to_thesis("q")


class TestSelectionEndToEnd:
    """A numeric thesis against the real dataset and real candles."""

    @pytest.fixture(scope="class")
    @classmethod
    def finding(cls):  # type: ignore[no-untyped-def]
        from backtool.research.experiment import run_thesis

        return run_thesis(
            Thesis(
                question="What happens after CPI when BTC fell more than 3% "
                "during the previous week?",
                event_type=EventType.CPI,
                conditions=(FELL_3_PCT,),
                outcome=Outcome(horizon="24h"),
            )
        )

    def test_it_runs(self, finding) -> None:  # type: ignore[no-untyped-def]
        assert finding.baseline.sample_size > 0

    def test_every_qualifying_release_really_fell_more_than_three_percent(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        """Read back from the observation rows, which carry the measured
        figure rather than the rule."""
        for observation in finding.observations:
            shown = observation.conditions["BTC 7-day return"]
            assert shown != "not measured"
            assert float(shown.rstrip("%")) < -3.0

    def test_the_cohort_is_a_subset_of_the_baseline(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        assert set(finding.cohort.event_ids) <= set(finding.baseline.event_ids)

    def test_the_rule_is_disclosed_in_the_cohort_label(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        assert "-3%" in finding.cohort.label
