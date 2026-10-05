"""Tests for A-versus-B comparisons: two filtered groups, measured the same way.

The generalisation under test is that a baseline was never a different kind of
thing from a cohort. "All usable CPI releases" is the group selected by no
filters; "cooler CPI releases" is the group selected by one. Representing both
as :class:`~backtool.research.thesis.Comparison` means a baseline question and
a hotter-versus-cooler question run through identical code, and these tests
assert that by driving both through the same functions rather than through
branches named after question types.

Three properties are defended.

**The two sides measure the same thing, structurally.** The metric and the
window live on one shared ``Outcome``. There is no field in which left and
right could disagree, so the rejection you would otherwise have to test for
cannot be written down -- and one test here asserts that absence directly.

**Neither side is the other's background.** A group the reader named is called
by its name. "Typical" and "baseline" belong only to a group that really is
every release, and the payload sent to the language layer carries the
baseline-specific verdict only in that case.

**The weaker side is never covered for.** Two groups in one comparison are
routinely lopsided. Each carries its own sample size and its own quality band,
in the facts, in the payload and in the prose.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.ai.research_interpreter import violations
from backtool.ai.thesis_planner import PlannedCondition, PlannedThesis
from backtool.core.types import EventType
from backtool.research.experiment import run_thesis
from backtool.research.interpretation import (
    build_facts,
    deterministic_summary,
    interpretation_payload,
)
from backtool.research.thesis import Comparison, Condition, Outcome, Thesis
from backtool.research.vocabulary import (
    Aggregation,
    Operator,
    Order,
    ResultShape,
)
from backtool.web.research_page import render_finding

CUTOFF = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)

HOTTER = Condition(field="cpi_surprise", value="hotter")
COOLER = Condition(field="cpi_surprise", value="cooler")
IN_LINE = Condition(field="cpi_surprise", value="in_line")
BULLISH = Condition(field="btc_trend_7d", value="bullish")
BEARISH = Condition(field="btc_trend_7d", value="bearish")
BELOW_MA = Condition(field="btc_vs_200d_ma", value="below")


def versus(
    left: tuple[Condition, ...],
    right: tuple[Condition, ...],
    *,
    metric: str = "return_pct",
    horizon: str = "24h",
) -> Thesis:
    """An A-versus-B thesis, built from two generic filter sets."""
    return Thesis(
        event_type=EventType.CPI,
        conditions=left,
        versus=Comparison(conditions=right),
        outcome=Outcome(metric=metric, horizon=horizon),
    )


BASELINE_THESIS = Thesis(
    question="Does a bearish 7-day BTC trend change the reaction to CPI?",
    event_type=EventType.CPI,
    conditions=(BEARISH,),
    versus=Comparison(),
    outcome=Outcome(horizon="24h"),
)


class TestTheModelIsGeneric:
    """No comparison type per question. Two groups, one shape."""

    def test_a_baseline_is_a_group_with_no_filters(self) -> None:
        """The whole generalisation in one assertion: the old baseline and a
        new named group are the same class, differing only in contents."""
        baseline = Comparison()
        named = Comparison(conditions=(COOLER,))
        assert type(baseline) is type(named)
        assert baseline.is_baseline
        assert not named.is_baseline

    def test_both_kinds_of_comparison_have_the_same_shape(self) -> None:
        assert BASELINE_THESIS.shape is ResultShape.COMPARISON
        assert versus((HOTTER,), (COOLER,)).shape is ResultShape.COMPARISON

    def test_the_metric_cannot_differ_between_sides(self) -> None:
        """Not validated -- unrepresentable. The outcome is one object shared
        by the thesis, so there is no second field to disagree with."""
        assert "metric" not in Comparison.model_fields
        assert "outcome" not in Comparison.model_fields
        assert "horizon" not in Comparison.model_fields
        assert set(Comparison.model_fields) == {"conditions"}

    def test_the_window_cannot_differ_between_sides(self) -> None:
        thesis = versus((HOTTER,), (COOLER,), horizon="4h")
        assert thesis.outcome.horizon == "4h"
        # One window, read by both groups.
        assert thesis.outcome.window == "post_event_4h"

    def test_both_sides_use_the_same_condition_vocabulary(self) -> None:
        thesis = versus((HOTTER,), (COOLER,))
        assert thesis.conditions[0].field == thesis.versus.conditions[0].field


class TestValidation:
    """The rejections from the brief, and the ones that cannot happen."""

    def test_identical_groups_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero by construction"):
            versus((HOTTER,), (HOTTER,))

    def test_identical_groups_are_rejected_whatever_the_order(self) -> None:
        """Conditions are a conjunction, so the same filters listed differently
        are the same group. Comparing tuples directly would let a reordering
        through as a real comparison."""
        with pytest.raises(ValueError, match="zero by construction"):
            versus((HOTTER, BELOW_MA), (BELOW_MA, HOTTER))

    def test_two_empty_groups_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero by construction"):
            Thesis(event_type=EventType.CPI, versus=Comparison())

    def test_an_unsupported_condition_on_the_right_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown condition field"):
            Comparison(conditions=(Condition(field="funding_rate", value="negative"),))

    def test_a_cpi_condition_on_the_right_cannot_face_fomc(self) -> None:
        """The left side already checked this; the right must too, or half the
        comparison would be a cohort of zero."""
        with pytest.raises(ValueError, match="cannot be tested against"):
            Thesis(
                event_type=EventType.FOMC,
                conditions=(BULLISH,),
                versus=Comparison(conditions=(COOLER,)),
            )

    @pytest.mark.parametrize("metric", ["return_pct", "range_pct", "mae_pct"])
    def test_post_event_metrics_cannot_filter_either_side(self, metric: str) -> None:
        """The look-ahead wall holds on the right exactly as on the left,
        because both sides hold the same Condition type."""
        with pytest.raises(ValueError, match="measured after the event"):
            Comparison(
                conditions=(
                    Condition(field=metric, operator=Operator.ABOVE, threshold=2.0),
                )
            )

    def test_a_ranked_query_cannot_have_a_second_group(self) -> None:
        with pytest.raises(ValueError, match="nothing to compare against"):
            Thesis(
                event_type=EventType.CPI,
                conditions=(HOTTER,),
                versus=Comparison(conditions=(COOLER,)),
                outcome=Outcome(
                    metric="range_pct",
                    aggregation=Aggregation.ROWS,
                    order=Order.DESCENDING,
                    limit=5,
                ),
            )


class TestFollowUps:
    """Editing a comparison, through the existing mutation helpers."""

    def test_a_shared_condition_lands_on_both_sides(self) -> None:
        """"What about only when BTC was below its 200-day average?" narrows
        the question, not one side of it."""
        narrowed = versus((HOTTER,), (COOLER,)).with_shared_condition(BELOW_MA)
        assert {c.field for c in narrowed.conditions} == {
            "cpi_surprise",
            "btc_vs_200d_ma",
        }
        assert {c.field for c in narrowed.versus.conditions} == {
            "cpi_surprise",
            "btc_vs_200d_ma",
        }
        # Each side keeps its own distinguishing filter.
        assert any(c.value == "hotter" for c in narrowed.conditions)
        assert any(c.value == "cooler" for c in narrowed.versus.conditions)

    def test_a_shared_condition_on_a_plain_thesis_still_works(self) -> None:
        plain = Thesis(event_type=EventType.CPI, conditions=(HOTTER,))
        assert len(plain.with_shared_condition(BELOW_MA).conditions) == 2

    def test_a_shared_condition_reuses_the_replace_rule(self) -> None:
        """A numeric bound added to both groups must behave the same way on
        each -- replacing the bound on the same end, not stacking."""
        base = versus((HOTTER,), (COOLER,)).with_shared_condition(
            Condition(field="btc_return_7d", operator=Operator.BELOW, threshold=-3.0)
        )
        tightened = base.with_shared_condition(
            Condition(field="btc_return_7d", operator=Operator.BELOW, threshold=-5.0)
        )
        for side in (tightened.conditions, tightened.versus.conditions):
            bounds = [c for c in side if c.field == "btc_return_7d"]
            assert len(bounds) == 1
            assert bounds[0].threshold == -5.0

    def test_changing_the_window_preserves_both_groups(self) -> None:
        changed = versus((HOTTER,), (COOLER,)).with_outcome(Outcome(horizon="4h"))
        assert changed.outcome.horizon == "4h"
        assert changed.conditions[0].value == "hotter"
        assert changed.versus.conditions[0].value == "cooler"

    def test_changing_only_the_right_group(self) -> None:
        """"Compare hotter with in-line instead" edits one side."""
        changed = versus((HOTTER,), (COOLER,)).versus_group((IN_LINE,))
        assert changed.conditions[0].value == "hotter"
        assert changed.versus.conditions[0].value == "in_line"

    def test_the_original_is_never_mutated(self) -> None:
        original = versus((HOTTER,), (COOLER,))
        original.with_shared_condition(BELOW_MA)
        original.versus_group((IN_LINE,))
        assert len(original.conditions) == 1
        assert original.versus.conditions[0].value == "cooler"

    def test_a_comparison_needs_the_regime_pass_for_either_side(self) -> None:
        """A condition on the right alone still requires the history fetch."""
        right_only = Thesis(
            event_type=EventType.CPI,
            conditions=(HOTTER,),
            versus=Comparison(conditions=(BEARISH,)),
        )
        assert right_only.needs_regime
        assert not versus((HOTTER,), (COOLER,)).needs_regime


class TestRealData:
    """Against the real dataset and real candles."""

    @pytest.fixture(scope="class")
    @classmethod
    def surprise(cls):  # type: ignore[no-untyped-def]
        return run_thesis(versus((HOTTER,), (COOLER,)), as_of=CUTOFF)

    @pytest.fixture(scope="class")
    @classmethod
    def trend(cls):  # type: ignore[no-untyped-def]
        return run_thesis(versus((BULLISH,), (BEARISH,)), as_of=CUTOFF)

    @pytest.fixture(scope="class")
    @classmethod
    def baseline(cls):  # type: ignore[no-untyped-def]
        return run_thesis(BASELINE_THESIS, as_of=CUTOFF)

    def test_hotter_against_cooler_produces_two_groups(
        self, surprise
    ) -> None:  # type: ignore[no-untyped-def]
        assert surprise.sample_size > 0
        assert surprise.comparison_sample_size > 0
        assert surprise.value is not None
        assert surprise.comparison_value is not None

    def test_the_two_groups_are_disjoint_here(
        self, surprise
    ) -> None:  # type: ignore[no-untyped-def]
        """A release cannot be both hotter and cooler. Nothing in the model
        guarantees disjointness -- the conditions do -- so it is checked."""
        assert not set(surprise.cohort.event_ids) & set(
            surprise.comparison.event_ids
        )

    def test_each_side_holds_only_its_own_releases(
        self, surprise
    ) -> None:  # type: ignore[no-untyped-def]
        from backtool.events.releases import classify_cpi, load_releases

        values = load_releases()
        for event_id in surprise.cohort.event_ids:
            assert classify_cpi(values[event_id])[0] == "hotter"
        for event_id in surprise.comparison.event_ids:
            assert classify_cpi(values[event_id])[0] == "cooler"

    def test_the_difference_is_the_two_values(
        self, surprise
    ) -> None:  # type: ignore[no-untyped-def]
        assert surprise.difference_pp == pytest.approx(
            surprise.value - surprise.comparison_value
        )

    def test_neither_group_is_called_typical(
        self, surprise
    ) -> None:  # type: ignore[no-untyped-def]
        assert surprise.cohort.label == "CPI surprise is hotter than expected"
        assert surprise.comparison.label == "CPI surprise is cooler than expected"
        assert "typical" not in surprise.verdict().lower()
        assert "baseline" not in surprise.verdict().lower()

    def test_the_verdict_names_the_higher_group(
        self, surprise
    ) -> None:  # type: ignore[no-untyped-def]
        verdict = surprise.verdict()
        if abs(surprise.difference_pp) >= 0.1:
            higher = (
                surprise.cohort.label
                if surprise.difference_pp > 0
                else surprise.comparison.label
            )
            assert higher in verdict

    def test_bullish_against_bearish_runs(
        self, trend
    ) -> None:  # type: ignore[no-untyped-def]
        assert trend.sample_size > 0
        assert trend.comparison_sample_size > 0
        assert not set(trend.cohort.event_ids) & set(trend.comparison.event_ids)

    def test_the_existing_baseline_comparison_is_unchanged(
        self, baseline
    ) -> None:  # type: ignore[no-untyped-def]
        """The regression that matters: the question this product already
        answered must give the same numbers it always gave."""
        assert baseline.shape is ResultShape.COMPARISON
        assert baseline.sample_size == 8
        assert baseline.value == pytest.approx(0.38, abs=0.01)
        assert baseline.comparison_sample_size == 31
        assert baseline.comparison_value == pytest.approx(0.95, abs=0.01)
        assert baseline.difference_pp == pytest.approx(-0.57, abs=0.01)
        assert baseline.comparison.label == "All usable CPI releases"

    def test_a_shared_condition_narrows_both_sides_in_practice(self) -> None:
        wide = run_thesis(versus((HOTTER,), (COOLER,)), as_of=CUTOFF)
        narrow = run_thesis(
            versus((HOTTER,), (COOLER,)).with_shared_condition(BELOW_MA),
            as_of=CUTOFF,
        )
        assert narrow.sample_size <= wide.sample_size
        assert narrow.comparison_sample_size <= wide.comparison_sample_size

    def test_one_measurement_serves_both_groups(
        self, surprise, trend
    ) -> None:  # type: ignore[no-untyped-def]
        """Two different comparisons over the same universe share a
        specification, so the second re-reads the first one's study."""
        assert surprise.spec.fingerprint == trend.spec.fingerprint


class TestFactsAndPayload:
    """Left and right, never a fabricated baseline."""

    @pytest.fixture(scope="class")
    @classmethod
    def named(cls):  # type: ignore[no-untyped-def]
        return build_facts(run_thesis(versus((HOTTER,), (COOLER,)), as_of=CUTOFF))

    @pytest.fixture(scope="class")
    @classmethod
    def against_all(cls):  # type: ignore[no-untyped-def]
        return build_facts(run_thesis(BASELINE_THESIS, as_of=CUTOFF))

    def test_the_payload_uses_left_and_right(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        payload = interpretation_payload(named)
        assert payload["left"]["label"] == "CPI surprise is hotter than expected"
        assert payload["right"]["label"] == "CPI surprise is cooler than expected"
        assert "cohort" not in payload
        assert "baseline" not in payload

    def test_a_named_comparison_sends_no_baseline_verdict(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        """`direction_vs_baseline` says better or worse, which only means
        something when one side is every release. Sending it here would hand
        the model a standard neither group is."""
        payload = interpretation_payload(named)
        assert "direction_vs_baseline" not in payload
        assert "right_is_baseline" not in payload
        assert payload["higher_side"] in {"left", "right", "neither"}

    def test_a_baseline_comparison_keeps_its_verdict(
        self, against_all
    ) -> None:  # type: ignore[no-untyped-def]
        payload = interpretation_payload(against_all)
        assert payload["right_is_baseline"] is True
        assert payload["direction_vs_baseline"] in {"better", "worse", "similar"}

    def test_each_side_carries_its_own_sample_quality(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        """Two groups in one comparison are routinely lopsided, and a single
        shared warning would describe the better-supported side."""
        payload = interpretation_payload(named)
        for side in ("left", "right"):
            assert payload[side]["sample_quality"]
            assert isinstance(payload[side]["small_sample"], bool)
            assert payload[side]["n"] > 0

    def test_the_higher_side_matches_the_sign(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        assert named.difference_pp is not None
        if abs(named.difference_pp) < 0.1:
            assert named.higher_side == "neither"
        else:
            assert named.higher_side == (
                "left" if named.difference_pp > 0 else "right"
            )

    def test_the_payload_carries_no_price_series(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        payload = interpretation_payload(named)

        def walk(node: object, path: str = "") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key.lower() not in {"open", "high", "low", "close"}, key
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                assert not [i for i in node if isinstance(i, int | float)], path
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]")

        walk(payload)

    def test_the_summary_reports_both_sample_sizes(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        text = " ".join(deterministic_summary(named))
        assert named.left is not None and named.right is not None
        assert f"{named.left.sample_size} release" in text
        assert f"{named.right.sample_size} release" in text

    def test_the_summary_does_not_call_either_group_a_baseline(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        text = " ".join(deterministic_summary(named)).lower()
        for word in ("baseline", "typical release", "benchmark"):
            assert word not in text, word

    def test_the_baseline_summary_still_says_baseline(
        self, against_all
    ) -> None:  # type: ignore[no-untyped-def]
        """There, one side really is everything, and saying so is clearer than
        pretending the comparison is symmetrical."""
        text = " ".join(deterministic_summary(against_all)).lower()
        assert "baseline" in text

    def test_a_weak_side_is_disclosed_in_the_prose(self) -> None:
        """The interpretation must not hide the weaker group."""
        facts = build_facts(
            run_thesis(
                versus((HOTTER, BELOW_MA), (COOLER,)),
                as_of=CUTOFF,
            )
        )
        text = " ".join(deterministic_summary(facts))
        assert facts.left is not None
        if facts.left.needs_sample_warning:
            assert facts.left.sample_quality in text

    def test_the_interpreter_may_not_call_a_named_group_a_baseline(
        self, named
    ) -> None:  # type: ignore[no-untyped-def]
        """The guard that replaces the better/worse check when neither side is
        a standard."""
        payload = interpretation_payload(named)
        assert violations(
            "Hotter prints outperformed the baseline over 24 hours.", payload
        )
        assert violations("Cooler CPI is the typical release here.", payload)
        assert not violations(
            "Hotter prints measured higher than cooler prints in this "
            "historical sample.",
            payload,
        )

    def test_the_baseline_guard_still_applies_where_it_means_something(
        self, against_all
    ) -> None:  # type: ignore[no-untyped-def]
        payload = interpretation_payload(against_all)
        assert payload["direction_vs_baseline"] == "worse"
        assert violations("The cohort outperformed the baseline.", payload)
        assert not violations(
            "The cohort underperformed the baseline in this historical sample.",
            payload,
        )


class TestRendering:
    """Both tiles carry their own group's name."""

    @pytest.fixture(scope="class")
    @classmethod
    def html(cls) -> str:
        finding = run_thesis(versus((HOTTER,), (COOLER,)), as_of=CUTOFF)
        return render_finding(finding, deterministic_summary(build_facts(finding)))

    def test_both_groups_are_named(self, html: str) -> None:
        assert "CPI surprise is hotter than expected" in html
        assert "CPI surprise is cooler than expected" in html

    def test_neither_group_is_called_typical(self, html: str) -> None:
        """Case-insensitive, over the reader-visible text. A capital-B check
        let "cohort minus baseline" through on the difference tile, which is
        the one place the old framing survived.

        Tags are stripped first: the distribution chart carries an SVG class
        called ``median-baseline``, which is a style hook no reader sees, and
        matching it would make this test fail for a reason that is not about
        what the page says.
        """
        import re as _re

        lowered = _re.sub(r"<[^>]+>", " ", html).lower()
        assert "typical release" not in lowered
        assert "baseline" not in lowered
        assert "benchmark" not in lowered

    def test_the_difference_tile_names_both_groups(self, html: str) -> None:
        """A reader has to be able to tell which way the subtraction ran."""
        assert "minus" in html
        assert "cohort minus baseline" not in html.lower()

    def test_the_difference_is_shown(self, html: str) -> None:
        assert "Difference" in html
        assert "compare-diff" in html

    def test_both_sample_sizes_are_shown(self, html: str) -> None:
        finding = run_thesis(versus((HOTTER,), (COOLER,)), as_of=CUTOFF)
        assert f"n={finding.sample_size}" in html
        assert f"n={finding.comparison_sample_size}" in html

    def test_the_baseline_case_still_says_typical(self) -> None:
        """Unchanged where it is accurate."""
        finding = run_thesis(BASELINE_THESIS, as_of=CUTOFF)
        assert "A typical release" in render_finding(finding)


class TestPlannerBoundary:
    """What the planner may emit, validated into two groups."""

    @staticmethod
    def planned(
        left: list[PlannedCondition],
        right: list[PlannedCondition],
        *,
        baseline: bool = False,
    ) -> PlannedThesis:
        return PlannedThesis(
            restated_question="q",
            asset="BTCUSDT",
            event_type="CPI",
            conditions=left,
            comparison_conditions=right,
            outcome_metric="return_pct",
            outcome_horizon="24h",
            aggregation="median",
            order="",
            limit="",
            outcome_threshold="",
            outcome_threshold_operator="",
            compare_to_baseline=baseline,
            assumptions=[],
            clarifications=[],
            unsupported=[],
        )

    @staticmethod
    def condition(field: str, value: str) -> PlannedCondition:
        return PlannedCondition(field=field, operator="is", value=value, threshold="")

    def test_two_groups_compile(self) -> None:
        thesis = self.planned(
            [self.condition("cpi_surprise", "hotter")],
            [self.condition("cpi_surprise", "cooler")],
        ).to_thesis("q")
        assert thesis.shape is ResultShape.COMPARISON
        assert thesis.versus is not None
        assert not thesis.versus.is_baseline

    def test_a_named_group_wins_over_a_baseline_flag(self) -> None:
        """The reader asked about two groups; measuring one against everything
        would answer a question they did not ask."""
        thesis = self.planned(
            [self.condition("cpi_surprise", "hotter")],
            [self.condition("cpi_surprise", "cooler")],
            baseline=True,
        ).to_thesis("q")
        assert thesis.versus is not None
        assert not thesis.versus.is_baseline

    def test_no_second_group_falls_back_to_the_baseline(self) -> None:
        thesis = self.planned(
            [self.condition("btc_trend_7d", "bearish")], [], baseline=True
        ).to_thesis("q")
        assert thesis.versus is not None
        assert thesis.versus.is_baseline

    def test_identical_groups_are_refused_at_the_boundary(self) -> None:
        with pytest.raises(ValueError, match="zero by construction"):
            self.planned(
                [self.condition("cpi_surprise", "hotter")],
                [self.condition("cpi_surprise", "hotter")],
            ).to_thesis("q")

    def test_a_post_event_metric_on_the_right_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measured after the event"):
            self.planned(
                [self.condition("cpi_surprise", "hotter")],
                [
                    PlannedCondition(
                        field="return_pct", operator=">", value="", threshold="2"
                    )
                ],
            ).to_thesis("q")
