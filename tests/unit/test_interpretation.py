"""Tests for the interpretation layer: facts, fallback, and what reaches the model.

The claim being defended is narrow and absolute: **the language layer explains
numbers, it never produces them, and it can never contradict them.** Three
separate mechanisms uphold it, and each is tested here.

1. The payload. Only computed statistics cross into ``ai/``, pinned key by key,
   with no candle or price path anywhere in it.
2. The verdict. Direction, magnitude and sample band are decided by
   deterministic code before prose exists, so the model is told the conclusion
   rather than asked for one.
3. The check. An explanation that reverses the direction, or explains *why* the
   market moved, is discarded and the deterministic summary shown instead.

Together they mean the worst outcome of a bad model response is prose that is
less good, never a report that is wrong.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from backtool.ai.client import AIError
from backtool.ai.research_interpreter import (
    RESEARCH_INTERPRETER_SYSTEM,
    ResearchExplanation,
    explain_research,
    violations,
)
from backtool.core.types import EventType, Interval
from backtool.research.aggregate import WindowAggregate
from backtool.research.cohort import Cohort
from backtool.research.experiment import Observation, ResearchFinding
from backtool.research.interpretation import (
    Direction,
    build_facts,
    deterministic_summary,
    interpretation_payload,
    methodology_summary,
)
from backtool.research.spec import ResearchSpec
from backtool.research.thesis import (
    Comparison,
    Condition,
    Outcome,
    Thesis,
    describe_group,
)
from backtool.research.vocabulary import Operator, reaction_windows

WINDOW = "post_event_24h"


def aggregate(
    *, n: int, median: float, up: int, window: str = WINDOW
) -> WindowAggregate:
    """A window aggregate with the fields the facts layer reads.

    Hand-built rather than measured: this file is testing the layer *above* the
    statistics, and constructing them here keeps a change in the engine from
    being reported as a failure of the prose.
    """
    return WindowAggregate(
        window=window,
        sample_size=n,
        excluded=0,
        mean_return_pct=median,
        median_return_pct=median,
        stdev_return_pct=1.0,
        min_return_pct=median - 5,
        max_return_pct=median + 5,
        up_rate_pct=round(up / n * 100, 1) if n else 0.0,
        down_rate_pct=round((n - up) / n * 100, 1) if n else 0.0,
        flat_count=0,
        up_count=up,
        down_count=n - up,
        p10_return_pct=median - 3,
        p25_return_pct=median - 1,
        p75_return_pct=median + 1,
        p90_return_pct=median + 3,
        mean_abs_move_pct=abs(median),
        mean_mfe_pct=abs(median) + 1,
        mean_mae_pct=-1.0,
        mean_range_pct=2.0,
        median_range_pct=2.0,
        mean_realized_vol_pct=0.5,
        move_over_0_5_pct=50.0,
        move_over_1_pct=25.0,
        move_over_0_5_count=n // 2,
        move_over_1_count=n // 4,
        mean_coverage_pct=100.0,
    )


def finding(
    *,
    cohort_n: int = 8,
    cohort_median: float = 0.38,
    cohort_up: int = 6,
    baseline_n: int = 31,
    baseline_median: float = 0.95,
    baseline_up: int = 22,
    conditions: tuple[Condition, ...] = (
        Condition(field="btc_trend_7d", value="bearish"),
    ),
    excluded: dict[str, int] | None = None,
    warnings: tuple[str, ...] = ("The baseline includes the cohort.",),
    outcome: Outcome | None = None,
    baseline: bool = True,
) -> ResearchFinding:
    """The worked example from the brief, as a finding.

    Defaults reproduce the real result that prompted this work: a bearish
    7-day trend, 8 releases, +0.38% against +0.95%, -0.57 pp.
    """
    thesis = Thesis(
        question="Does a bearish 7-day BTC trend change the reaction to CPI?",
        event_type=EventType.CPI,
        conditions=conditions,
        outcome=outcome or Outcome(horizon="24h"),
        versus=Comparison() if baseline and conditions else None,
    )
    spec = ResearchSpec(
        symbol="BTCUSDT",
        event_type=EventType.CPI,
        event_count=40,
        interval=Interval.M5,
        windows=reaction_windows(),
        as_of=dt.datetime(2026, 10, 1, tzinfo=dt.UTC),
    )
    wants_baseline = thesis.versus is not None
    return ResearchFinding(
        thesis=thesis,
        spec=spec,
        generated_at=dt.datetime(2026, 10, 1, tzinfo=dt.UTC),
        cohort=Cohort(
            key="left",
            # Derived from the conditions, as the engine derives it, so a
            # fixture cannot assert a label the engine would never produce.
            label=describe_group(conditions, EventType.CPI),
            description="CPI releases entered in a bearish 7-day trend",
            event_ids=tuple(f"CPI-2024-{m:02d}-10" for m in range(1, cohort_n + 1)),
            windows={WINDOW: aggregate(n=cohort_n, median=cohort_median, up=cohort_up)},
        ),
        comparison=Cohort(
            key="right",
            label="All usable CPI releases",
            description="Every CPI release with usable price data",
            event_ids=tuple(f"CPI-B-{i:02d}" for i in range(baseline_n)),
            windows={
                WINDOW: aggregate(n=baseline_n, median=baseline_median, up=baseline_up)
            },
        )
        if wants_baseline
        else None,
        observations=tuple(
            Observation(
                event_id=f"CPI-2024-{i:02d}-10",
                event_date=dt.date(2024, (i - 1) % 12 + 1, 11),
                conditions={"BTC 7-day trend": "bearish"},
                returns={"24h": cohort_median},
                value=cohort_median + i / 100,
            )
            for i in range(1, cohort_n + 1)
        ),
        value=cohort_median if cohort_n else None,
        comparison_value=baseline_median if wants_baseline and baseline_n else None,
        sample_size=cohort_n,
        comparison_sample_size=baseline_n if wants_baseline else 0,
        excluded=excluded if excluded is not None else {"did not match: x": 17},
        warnings=warnings,
    )


class StubClient:
    """An :class:`AIClient` stand-in that returns fixed paragraphs.

    Lets the whole interpretation path be tested with no key, no network and no
    cost -- which is also the configuration the fallback has to work in, so the
    tests and the degraded mode exercise the same code.
    """

    def __init__(self, paragraphs: list[str]) -> None:
        self._paragraphs = paragraphs
        self.seen: dict[str, object] = {}
        # A faithful double reports these, because the caller prices the call
        # it just made. A stand-in missing them would make the cost accounting
        # untestable by the very tests that exercise the path it runs on.
        self.model = "claude-opus-5"
        self.last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def structured(self, *, system: str, user: str, schema: type, **_: object):  # type: ignore[no-untyped-def]
        self.seen = {"system": system, "user": user}
        return ResearchExplanation(paragraphs=self._paragraphs)


class ExplodingClient:
    """Whatever can go wrong with an API call, as one object."""

    def structured(self, **_: object) -> ResearchExplanation:
        raise AIError("upstream is unavailable")


class TestDeterministicFacts:
    """The verdict is settled before any prose exists."""

    def test_a_negative_difference_is_worse(self) -> None:
        facts = build_facts(finding())
        assert facts.direction is Direction.WORSE
        assert facts.difference_pp == pytest.approx(-0.57)

    def test_a_positive_difference_is_better(self) -> None:
        facts = build_facts(finding(cohort_median=1.60))
        assert facts.direction is Direction.BETTER
        assert facts.difference_pp == pytest.approx(0.65)

    def test_an_immaterial_difference_picks_no_side(self) -> None:
        """Two samples this size differ by a tenth of a point from noise.
        Calling that 'better' would be the first dishonest sentence."""
        facts = build_facts(finding(cohort_median=0.99))
        assert facts.direction is Direction.SIMILAR

    def test_an_empty_cohort_is_undetermined_not_worse(self) -> None:
        facts = build_facts(finding(cohort_n=0, cohort_up=0))
        assert facts.direction is Direction.UNDETERMINED
        assert facts.left is None

    def test_the_counts_are_carried_not_recomputed_from_rates(self) -> None:
        """6/8 printed beside 75% has to be the same 6."""
        facts = build_facts(finding())
        assert facts.left is not None
        assert facts.left.positive_count == 6
        assert facts.left.sample_size == 8

    def test_a_small_sample_is_flagged(self) -> None:
        facts = build_facts(finding())
        assert facts.needs_sample_warning
        assert facts.sample_quality == "limited"

    def test_an_adequate_sample_is_not_flagged(self) -> None:
        facts = build_facts(finding(cohort_n=25, cohort_up=18))
        assert not facts.needs_sample_warning
        assert facts.sample_quality == "adequate"

    def test_the_engine_warnings_survive_verbatim(self) -> None:
        facts = build_facts(finding(warnings=("Limited sample: 8 releases.",)))
        assert facts.warnings == ("Limited sample: 8 releases.",)


class TestDeterministicSummary:
    """The report's floor: useful with no model involved at all."""

    def test_it_states_the_numbers_it_was_given(self) -> None:
        text = " ".join(deterministic_summary(build_facts(finding())))
        assert "8 releases" in text
        assert "+0.38%" in text
        assert "+0.95%" in text
        assert "0.57 percentage points" in text
        assert "6 of 8" in text

    def test_it_names_the_direction_the_facts_decided(self) -> None:
        worse = " ".join(deterministic_summary(build_facts(finding())))
        better = " ".join(
            deterministic_summary(build_facts(finding(cohort_median=1.60)))
        )
        assert "underperformed" in worse and "outperformed" not in worse
        assert "outperformed" in better and "underperformed" not in better

    def test_it_declines_to_pick_a_side_on_an_immaterial_difference(self) -> None:
        text = " ".join(deterministic_summary(build_facts(finding(cohort_median=0.99))))
        assert "no material difference" in text
        assert "outperformed" not in text and "underperformed" not in text

    def test_it_warns_about_a_small_sample(self) -> None:
        text = " ".join(deterministic_summary(build_facts(finding())))
        assert "only 8 releases" in text
        assert "no statistical test has been run" in text

    def test_it_makes_no_causal_claim(self) -> None:
        """A template has no excuse for one, and having a clean fallback is a
        quieter argument for writing the fallback first."""
        for median in (0.38, 1.60, 0.99):
            facts = build_facts(finding(cohort_median=median))
            text = " ".join(deterministic_summary(facts))
            assert not violations(text, interpretation_payload(facts)), text

    def test_an_empty_cohort_says_so_plainly(self) -> None:
        text = " ".join(deterministic_summary(build_facts(finding(cohort_n=0, cohort_up=0))))
        assert "nothing to compare" in text
        assert "BTC 7-day trend is bearish" in text

    def test_unclassifiable_releases_are_disclosed(self) -> None:
        facts = build_facts(
            finding(excluded={"BTC 7-day trend could not be determined": 4})
        )
        text = " ".join(deterministic_summary(facts))
        assert "4 further releases" in text
        assert "excluded rather than assumed" in text


class TestPayloadBoundary:
    """What may cross into the AI layer, and nothing else."""

    def test_the_payload_keys_are_pinned(self) -> None:
        """Pinned deliberately. This function decides what reaches the model,
        so adding a key has to be a conscious edit here too."""
        payload = interpretation_payload(build_facts(finding()))
        assert set(payload) == {
            # Shared by every shape.
            "question",
            "asset",
            "event_type",
            "horizon",
            "measure",
            "metric",
            "result_shape",
            "sample_size",
            "warnings",
            "excluded_counts",
            "methodology",
            # Any scalar result.
            "cohort_rule",
            "result",
            "sample_quality",
            "small_sample",
            # A comparison, and only a comparison. Named left and right, not
            # cohort and baseline: in an A-versus-B question neither side is
            # the other's background.
            "left",
            "right",
            "difference_pp",
            "difference_is_material",
            "higher_side",
            # Sent only because the right group here IS every usable release.
            # A comparison of two named groups carries neither of these.
            "right_is_baseline",
            "direction_vs_baseline",
        }

    def test_the_payload_carries_no_price_data(self) -> None:
        """Not a candle, not a price path, not a per-release row. The model has
        nothing to 'analyse', only statistics to describe.

        Checked structurally rather than by keyword, because the methodology
        block legitimately contains the *words* "candle" and "price" while
        describing how the figures were produced. What must not appear is a
        price *value*: any sequence of numbers, or an OHLCV key.
        """
        payload = interpretation_payload(build_facts(finding()))

        def walk(node: object, path: str = "") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key.lower() not in {
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "candles_raw",
                        "observations",
                        "per_event",
                        "series",
                    }, f"{path}.{key}"
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                numbers = [item for item in node if isinstance(item, int | float)]
                assert not numbers, f"{path} carries a numeric series: {numbers[:5]}"
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]")

        walk(payload)

    def test_the_payload_carries_the_verdict_not_the_raw_material(self) -> None:
        payload = interpretation_payload(build_facts(finding()))
        assert payload["direction_vs_baseline"] == "worse"
        assert payload["difference_is_material"] is True

    def test_the_payload_is_small(self) -> None:
        """Token discipline is a property of the payload, not of a reminder in
        the prompt. If this grows past a kilobyte or two something started
        sending evidence instead of conclusions."""
        payload = json.dumps(interpretation_payload(build_facts(finding())))
        assert len(payload) < 2000, len(payload)

    def test_each_group_reports_its_own_count(self) -> None:
        payload = interpretation_payload(build_facts(finding()))
        assert payload["left"]["n"] == 8
        assert payload["right"]["n"] == 31
        assert payload["left"]["positive_count"] == 6
        # Each side carries its own band, so a lopsided comparison cannot hide
        # the weaker half behind the stronger one's sample size.
        assert payload["left"]["sample_quality"] == "limited"
        assert payload["right"]["sample_quality"] == "adequate"
        assert payload["left"]["small_sample"] is True
        assert payload["right"]["small_sample"] is False

    def test_a_missing_cohort_is_null_not_zero(self) -> None:
        payload = interpretation_payload(build_facts(finding(cohort_n=0, cohort_up=0)))
        assert payload["left"] is None

    def test_the_methodology_summary_is_short_and_factual(self) -> None:
        summary = methodology_summary(finding())
        assert summary["statistic"] == "median across releases"
        assert "5m" in summary["candles"]
        assert len(json.dumps(summary)) < 1000

    def test_the_ai_interpreter_cannot_reach_the_engine(self) -> None:
        """The structural guarantee. ``research.experiment`` imports the data
        layer, so an interpreter that accepted a ``ResearchFinding`` would put
        candles one attribute access away from the prompt. It takes a dict."""
        import ast
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "backtool"
            / "ai"
            / "research_interpreter.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        leaked = {m for m in imported if m.startswith("backtool.research")}
        assert not leaked, f"interpreter imports the engine: {leaked}"


class TestExplanationChecks:
    """A sentence may not contradict the table beneath it."""

    def test_outperformance_cannot_be_claimed_over_a_worse_result(self) -> None:
        payload = interpretation_payload(build_facts(finding()))
        problems = violations(
            "The cohort outperformed the baseline over 24 hours.", payload
        )
        assert problems
        assert "outperform" in problems[0]

    def test_underperformance_cannot_be_claimed_over_a_better_result(self) -> None:
        payload = interpretation_payload(build_facts(finding(cohort_median=1.60)))
        assert violations("These releases underperformed the baseline.", payload)

    def test_the_correct_direction_passes(self) -> None:
        payload = interpretation_payload(build_facts(finding()))
        assert not violations(
            "In this historical sample the condition was associated with "
            "weaker 24-hour performance than CPI releases overall.",
            payload,
        )

    @pytest.mark.parametrize(
        "sentence",
        [
            "Bearish momentum caused institutions to sell into the print.",
            "This proves the condition is tradeable.",
            "The pattern guarantees a positive outcome.",
            "BTC will continue higher after the next release.",
            "The move was due to positioning ahead of the release.",
            "Weakness persisted because of de-risking by funds.",
        ],
    )
    def test_causal_and_predictive_claims_are_rejected(self, sentence: str) -> None:
        """Nothing in an event study establishes a mechanism or a forecast."""
        payload = interpretation_payload(build_facts(finding()))
        assert violations(sentence, payload), sentence

    def test_approved_hedging_language_is_not_rejected(self) -> None:
        """The check must not punish the phrasing the prompt asks for."""
        payload = interpretation_payload(build_facts(finding()))
        assert not violations(
            "The data shows the observed difference was negative. This does "
            "not provide strong evidence of a stable relationship, and the "
            "approval of any trade on it would need more observations.",
            payload,
        )

    def test_a_rejected_explanation_raises_rather_than_being_shown(self) -> None:
        facts = build_facts(finding())
        client = StubClient(["The cohort outperformed the baseline."])
        with pytest.raises(AIError, match="contradicted the facts"):
            explain_research(interpretation_payload(facts), client=client)  # type: ignore[arg-type]


class TestExplanationPath:
    """The call itself, with a stub in place of the API."""

    def test_the_model_sees_the_payload_and_nothing_more(self) -> None:
        facts = build_facts(finding())
        payload = interpretation_payload(facts)
        client = StubClient(
            ["In this historical sample the median was weaker than the baseline."]
        )
        explain_research(payload, client=client)  # type: ignore[arg-type]

        sent = json.loads(str(client.seen["user"]))
        assert sent == payload

    def test_the_prompt_forbids_new_numbers_and_causes(self) -> None:
        lowered = RESEARCH_INTERPRETER_SYSTEM.lower()
        assert "do not compute new ones" in lowered
        assert "direction_vs_baseline" in lowered
        assert "100-250 words" in lowered

    def test_blank_paragraphs_are_dropped(self) -> None:
        client = StubClient(["In this historical sample it was weaker.", "", "   "])
        result = explain_research(
            interpretation_payload(build_facts(finding())),
            client=client,  # type: ignore[arg-type]
        )
        assert len(result) == 1


class TestRenderingWithoutAI:
    """Requirement: the report renders whatever the AI does."""

    def test_the_report_renders_from_the_deterministic_summary_alone(self) -> None:
        from backtool.web.research_page import render_finding

        result = finding()
        facts = build_facts(result)
        html = render_finding(result, deterministic_summary(facts))
        assert "What this suggests" in html
        assert "8 releases" in html
        assert "+0.38%" in html

    def test_the_report_renders_with_no_interpretation_at_all(self) -> None:
        from backtool.web.research_page import render_finding

        html = render_finding(finding())
        assert "What this suggests" not in html
        # The numbers are still all there. The reading is an addition to the
        # report, not a part of it.
        assert "+0.38%" in html
        assert "n=8" in html
        assert "against A typical release" in html

    def test_a_failed_call_leaves_the_deterministic_summary_in_place(self) -> None:
        """The wiring in the route catches this; here it is at the source."""
        facts = build_facts(finding())
        with pytest.raises(AIError):
            explain_research(
                interpretation_payload(facts),
                client=ExplodingClient(),  # type: ignore[arg-type]
            )
        assert deterministic_summary(facts)

    def test_the_interpretation_sits_above_the_comparison(self) -> None:
        """Hierarchy: the reader should understand the answer before studying
        the charts."""
        from backtool.web.research_page import render_finding

        result = finding()
        html = render_finding(result, deterministic_summary(build_facts(result)))
        assert html.index("What this suggests") < html.index("compare-tile")
        assert html.index("compare-tile") < html.index("against A typical release")
        assert html.index("against A typical release") < html.index(
            "The releases behind this"
        )
        assert html.index("The releases behind this") < html.index(
            "How was this tested?"
        )


class TestWordingCannotChangeTheNumbers:
    """Requirement 9, stated as an invariant rather than a hope."""

    def test_two_different_explanations_leave_the_finding_identical(self) -> None:
        result = finding()
        before = result.model_dump_json()
        facts = build_facts(result)
        payload = interpretation_payload(facts)

        for paragraphs in (
            ["In this sample the median was weaker than the baseline."],
            ["The observed difference was negative across eight releases."],
        ):
            explain_research(payload, client=StubClient(paragraphs))  # type: ignore[arg-type]

        assert result.model_dump_json() == before
        assert build_facts(result).difference_pp == facts.difference_pp
        assert result.verdict() == finding().verdict()

    def test_the_headline_verdict_is_the_engines_own_sentence(self) -> None:
        """Whatever the prose says, the first line comes from the two numbers
        above it."""
        result = finding()
        assert "lower" in result.verdict()
        assert "0.57 percentage points" in result.verdict()


class TestConditionAwareFacts:
    """Numeric conditions reach the prose with their thresholds intact."""

    def test_a_numeric_rule_is_described_with_its_unit(self) -> None:
        facts = build_facts(
            finding(
                conditions=(
                    Condition(
                        field="btc_return_7d", operator=Operator.BELOW, threshold=-3.0
                    ),
                )
            )
        )
        assert facts.cohort_rule == "BTC 7-day return is below -3%"
        assert facts.left is not None
        assert facts.left.label == "BTC 7-day return is below -3%"
        assert "-3%" in " ".join(deterministic_summary(facts))

    def test_a_range_is_described_as_both_bounds(self) -> None:
        facts = build_facts(
            finding(
                conditions=(
                    Condition(
                        field="btc_return_7d",
                        operator=Operator.AT_LEAST,
                        threshold=-5.0,
                    ),
                    Condition(
                        field="btc_return_7d", operator=Operator.AT_MOST, threshold=-2.0
                    ),
                )
            )
        )
        assert facts.cohort_rule == (
            "BTC 7-day return is at least -5% and BTC 7-day return is at most -2%"
        )
