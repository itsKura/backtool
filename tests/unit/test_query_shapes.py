"""Tests for the composable query model: projection, shape, and the look-ahead wall.

Three claims are defended here, and they are the ones that would be expensive
to discover were false.

**A question is a query, not a type.** Largest, top five, median and frequency
differ only in the values a few fields take. There is no handler per question,
so these tests assert behaviour through the same code path with different
fields rather than through branches named after questions.

**A measurement is reused, not repeated.** Ordering, limiting and aggregating
are projections over per-event readings the study already produced, so the same
specification fingerprint answers all of them. Changing "the largest" to "the
five largest" must not measure anything twice.

**A cohort can never be selected by its own outcome.** The most dangerous bug
available in this product is a filter on a post-event metric: the arithmetic
stays correct, the sample size stays honest, and the result is chosen by
hindsight with nothing in the output to show it. That is prevented at
construction, not in a prompt, and these tests push on the wall from several
directions.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.core.types import EventType
from backtool.research.aggregate import aggregate_values, count_matching, metric_series
from backtool.research.experiment import run_thesis
from backtool.research.interpretation import (
    Direction,
    build_facts,
    deterministic_summary,
    interpretation_payload,
)
from backtool.research.metrics import Direction as Direction_
from backtool.research.metrics import WindowMetrics, WindowStatus
from backtool.research.results import EventResult
from backtool.research.thesis import Comparison, Condition, Outcome, Thesis
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    METRICS,
    Aggregation,
    Operator,
    Order,
    ResultShape,
    TemporalScope,
)
from backtool.web.research_page import render_finding

CUTOFF = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)


def ranked(metric: str = "range_pct", limit: int = 1, descending: bool = True) -> Thesis:
    return Thesis(
        event_type=EventType.CPI,
        outcome=Outcome(
            metric=metric,
            aggregation=Aggregation.ROWS,
            order=Order.DESCENDING if descending else Order.ASCENDING,
            limit=limit,
        ),
    )


def scalar(metric: str = "range_pct", aggregation: Aggregation = Aggregation.MEDIAN) -> Thesis:
    return Thesis(
        event_type=EventType.CPI, outcome=Outcome(metric=metric, aggregation=aggregation)
    )


class TestTemporalScope:
    """The wall between what selects and what is measured."""

    def test_every_condition_field_is_pre_event(self) -> None:
        """The rule, asserted rather than trusted. A field added without a
        scope inherits PRE_EVENT, so this is what catches a post-event one
        being added to the wrong table."""
        for spec in CONDITION_FIELDS.values():
            assert spec.scope is TemporalScope.PRE_EVENT, spec.key

    def test_every_metric_is_post_event(self) -> None:
        for spec in METRICS.values():
            assert spec.scope is TemporalScope.POST_EVENT, spec.key

    def test_the_two_vocabularies_are_disjoint(self) -> None:
        """A key in both would be selectable and measurable, which is the
        ambiguity the split exists to remove."""
        assert not set(METRICS) & set(CONDITION_FIELDS)

    @pytest.mark.parametrize(
        "metric", ["return_pct", "range_pct", "mfe_pct", "mae_pct", "abs_move_pct"]
    )
    def test_a_post_event_metric_cannot_select_the_cohort(self, metric: str) -> None:
        """The headline guarantee. Each of these would produce a cohort chosen
        by the answer, with arithmetically correct statistics over it."""
        with pytest.raises(ValueError, match="measured after the event"):
            Condition(field=metric, operator=Operator.ABOVE, threshold=2.0)

    def test_the_refusal_names_the_problem_not_a_typo(self) -> None:
        """'Unknown field' would send the reader looking for a spelling
        mistake instead of telling them the cohort would be circular."""
        with pytest.raises(ValueError) as caught:
            Condition(field="range_pct", operator=Operator.ABOVE, threshold=2.0)
        message = str(caught.value)
        assert "chosen by the very thing being measured" in message
        assert "frequency" in message

    def test_the_categorical_form_is_refused_too(self) -> None:
        """Closing the other door: `is` rather than `>` must not slip past."""
        with pytest.raises(ValueError, match="measured after the event"):
            Condition(field="return_pct", value="positive")

    def test_the_moved_two_percent_question_has_no_condition_form(self) -> None:
        """"How often did BTC move more than 2%?" must become a frequency.
        Expressed as a filter it would report 100% of a cohort selected for
        moving 2%."""
        with pytest.raises(ValueError, match="measured after the event"):
            Thesis(
                event_type=EventType.CPI,
                conditions=(
                    Condition(
                        field="abs_move_pct", operator=Operator.ABOVE, threshold=2.0
                    ),
                ),
            )

        # The correct form, which compiles.
        thesis = Thesis(
            event_type=EventType.CPI,
            outcome=Outcome(
                metric="abs_move_pct",
                aggregation=Aggregation.FREQUENCY,
                threshold=2.0,
                threshold_operator=Operator.ABOVE,
            ),
        )
        assert thesis.conditions == ()
        assert thesis.shape is ResultShape.SCALAR


class TestQueryValidation:
    """Combinations that would be silently ignored are refused instead."""

    def test_rows_without_an_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs a sort order"):
            Outcome(aggregation=Aggregation.ROWS, limit=1)

    def test_rows_without_a_limit_is_refused(self) -> None:
        """Dropping this requirement is how a fourth shape -- every
        observation as a series -- would arrive. Until then it must not be
        reachable by omission."""
        with pytest.raises(ValueError, match="needs a limit"):
            Outcome(aggregation=Aggregation.ROWS, order=Order.DESCENDING)

    def test_ordering_a_scalar_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to act on"):
            Outcome(aggregation=Aggregation.MEDIAN, order=Order.DESCENDING, limit=5)

    def test_a_frequency_needs_a_threshold_and_a_comparison(self) -> None:
        with pytest.raises(ValueError, match="not a question until"):
            Outcome(metric="abs_move_pct", aggregation=Aggregation.FREQUENCY)
        with pytest.raises(ValueError, match="not a question until"):
            Outcome(
                metric="abs_move_pct",
                aggregation=Aggregation.FREQUENCY,
                threshold=2.0,
            )

    def test_a_threshold_on_a_non_frequency_is_refused(self) -> None:
        with pytest.raises(ValueError, match="counts nothing against a threshold"):
            Outcome(aggregation=Aggregation.MEDIAN, threshold=2.0)

    def test_an_unknown_metric_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unsupported metric"):
            Outcome(metric="sharpe_ratio")


class TestBaselineIsOptional:
    """A comparison is a choice, and a meaningless one is refused."""

    def test_a_non_comparison_query_needs_no_baseline(self) -> None:
        thesis = scalar()
        assert thesis.versus is None
        assert thesis.shape is ResultShape.SCALAR

    def test_a_ranked_query_cannot_have_a_baseline(self) -> None:
        """The largest range has nothing to be larger *than*."""
        with pytest.raises(ValueError, match="nothing to compare against"):
            Thesis(
                event_type=EventType.CPI,
                conditions=(Condition(field="btc_trend_7d", value="bearish"),),
                outcome=Outcome(
                    metric="range_pct",
                    aggregation=Aggregation.ROWS,
                    order=Order.DESCENDING,
                    limit=5,
                ),
                versus=Comparison(),
            )

    def test_two_unfiltered_groups_are_refused(self) -> None:
        """Both sides would be every usable release and the difference zero by
        construction, which reads as a finding."""
        with pytest.raises(ValueError, match="zero by construction"):
            Thesis(event_type=EventType.CPI, versus=Comparison())

    def test_a_comparison_declares_its_shape(self) -> None:
        thesis = Thesis(
            event_type=EventType.CPI,
            conditions=(Condition(field="btc_trend_7d", value="bearish"),),
            versus=Comparison(),
        )
        assert thesis.shape is ResultShape.COMPARISON


class TestAggregationPrimitives:
    """The collapse itself, away from any data fetching."""

    def test_mean_and_median_differ_on_a_skewed_column(self) -> None:
        values = [1.0, 1.0, 1.0, 1.0, 96.0]
        assert aggregate_values(values, Aggregation.MEAN) == pytest.approx(20.0)
        assert aggregate_values(values, Aggregation.MEDIAN) == pytest.approx(1.0)

    def test_count_counts_rows_not_values(self) -> None:
        assert aggregate_values([3.0, 3.0, 3.0], Aggregation.COUNT) == 3.0

    def test_an_empty_column_has_no_mean_but_has_a_count(self) -> None:
        """Zero is a count. It is not a median."""
        assert aggregate_values([], Aggregation.COUNT) == 0.0
        assert aggregate_values([], Aggregation.MEDIAN) is None

    def test_rows_has_no_scalar_form(self) -> None:
        assert aggregate_values([1.0, 2.0], Aggregation.ROWS) is None

    def test_frequency_is_a_share_in_percent(self) -> None:
        values = [1.0, 2.5, 3.0, 0.5]
        share = aggregate_values(
            values, Aggregation.FREQUENCY, threshold=2.0, operator=Operator.ABOVE
        )
        assert share == pytest.approx(50.0)
        assert count_matching(values, Operator.ABOVE, 2.0) == 2

    def test_frequency_respects_the_boundary(self) -> None:
        values = [2.0, 2.0]
        assert (
            aggregate_values(
                values, Aggregation.FREQUENCY, threshold=2.0, operator=Operator.ABOVE
            )
            == 0.0
        )
        assert (
            aggregate_values(
                values, Aggregation.FREQUENCY, threshold=2.0, operator=Operator.AT_LEAST
            )
            == 100.0
        )

    def test_a_frequency_without_a_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="needs a threshold"):
            aggregate_values([1.0], Aggregation.FREQUENCY)


class TestProjectionOverRealData:
    """Against the real study, because the claim is about reuse."""

    @pytest.fixture(scope="class")
    @classmethod
    def largest(cls):  # type: ignore[no-untyped-def]
        return run_thesis(ranked(limit=1), as_of=CUTOFF)

    @pytest.fixture(scope="class")
    @classmethod
    def top_five(cls):  # type: ignore[no-untyped-def]
        return run_thesis(ranked(limit=5), as_of=CUTOFF)

    @pytest.fixture(scope="class")
    @classmethod
    def median(cls):  # type: ignore[no-untyped-def]
        return run_thesis(scalar(), as_of=CUTOFF)

    def test_changing_the_limit_does_not_change_the_measurement(
        self, largest, top_five
    ) -> None:  # type: ignore[no-untyped-def]
        """The specification is what gets measured; the limit is what gets
        reported. Identical fingerprints mean the second question re-read the
        first one's study rather than running another."""
        assert largest.spec.fingerprint == top_five.spec.fingerprint
        assert largest.sample_size == top_five.sample_size
        assert len(largest.observations) == len(top_five.observations)

    def test_a_wider_limit_is_the_same_ranking_extended(
        self, largest, top_five
    ) -> None:  # type: ignore[no-untyped-def]
        assert len(largest.rows) == 1
        assert len(top_five.rows) == 5
        assert top_five.rows[0].event_id == largest.rows[0].event_id
        assert [row.event_id for row in top_five.observations[:1]] == [
            row.event_id for row in largest.observations[:1]
        ]

    def test_one_fingerprint_serves_median_rank_and_frequency(
        self, largest, median
    ) -> None:  # type: ignore[no-untyped-def]
        """The separation that makes the composable model affordable: the spec
        covers symbol, events, interval, windows and cutoff, so changing the
        projection does not change what must be fetched."""
        frequency = run_thesis(
            Thesis(
                event_type=EventType.CPI,
                outcome=Outcome(
                    metric="abs_move_pct",
                    aggregation=Aggregation.FREQUENCY,
                    threshold=2.0,
                    threshold_operator=Operator.ABOVE,
                ),
            ),
            as_of=CUTOFF,
        )
        assert largest.spec.fingerprint == median.spec.fingerprint
        assert median.spec.fingerprint == frequency.spec.fingerprint

    def test_descending_really_returns_the_maximum(
        self, largest, top_five
    ) -> None:  # type: ignore[no-untyped-def]
        """MAX via order+limit must agree with the maximum of the column, and
        must name the release it came from."""
        values = [o.value for o in top_five.observations if o.value is not None]
        assert largest.rows[0].value == pytest.approx(max(values))
        assert largest.rows[0].event_id
        assert largest.rows[0].event_date

    def test_ascending_returns_the_minimum(self) -> None:
        smallest = run_thesis(ranked(limit=1, descending=False), as_of=CUTOFF)
        values = [o.value for o in smallest.observations if o.value is not None]
        assert smallest.rows[0].value == pytest.approx(min(values))

    def test_the_worst_return_is_the_most_negative_one(self) -> None:
        """Ascending on a signed metric means worst, not smallest magnitude."""
        worst = run_thesis(
            ranked(metric="return_pct", limit=1, descending=False), as_of=CUTOFF
        )
        values = [o.value for o in worst.observations if o.value is not None]
        assert worst.rows[0].value == pytest.approx(min(values))
        assert worst.rows[0].value is not None and worst.rows[0].value < 0

    def test_the_ordering_is_total(self, top_five) -> None:  # type: ignore[no-untyped-def]
        """Ties broken by recency, so two releases with the same reading
        cannot swap places between runs."""
        again = run_thesis(ranked(limit=5), as_of=CUTOFF)
        assert [r.event_id for r in again.rows] == [r.event_id for r in top_five.rows]

    def test_the_median_matches_the_projected_column(
        self, median
    ) -> None:  # type: ignore[no-untyped-def]
        import statistics

        values = [o.value for o in median.observations if o.value is not None]
        assert median.value == pytest.approx(statistics.median(values))

    def test_a_range_query_reports_ranges_not_returns(
        self, median
    ) -> None:  # type: ignore[no-untyped-def]
        """The regression that matters most for the projection: asking about
        range must not quietly report the return it used to hardcode."""
        returns = run_thesis(scalar(metric="return_pct"), as_of=CUTOFF)
        assert median.metric.key == "range_pct"
        assert returns.metric.key == "return_pct"
        assert median.value != pytest.approx(returns.value)
        # A trading range is a high-to-low span and cannot be negative; a
        # median return over this sample is a fraction of a percent. Confusing
        # the two would be visible here.
        assert median.value is not None and median.value > 1.0


class TestRenderingByShape:
    """Layout follows the query, not a catalogue of question types."""

    @pytest.fixture(scope="class")
    @classmethod
    def findings(cls):  # type: ignore[no-untyped-def]
        return {
            ResultShape.RANKED: run_thesis(ranked(limit=5), as_of=CUTOFF),
            ResultShape.SCALAR: run_thesis(scalar(), as_of=CUTOFF),
            ResultShape.COMPARISON: run_thesis(
                Thesis(
                    event_type=EventType.CPI,
                    conditions=(Condition(field="btc_trend_7d", value="bearish"),),
                    outcome=Outcome(horizon="24h"),
                    versus=Comparison(),
                ),
                as_of=CUTOFF,
            ),
        }

    def test_each_query_produces_its_declared_shape(
        self, findings
    ) -> None:  # type: ignore[no-untyped-def]
        for shape, finding in findings.items():
            assert finding.shape is shape

    def test_a_ranked_result_renders_rows_and_no_difference_tile(
        self, findings
    ) -> None:  # type: ignore[no-untyped-def]
        html = render_finding(findings[ResultShape.RANKED])
        assert "cohort-table ranked" in html
        assert "Difference" not in html
        assert "A typical release" not in html

    def test_a_scalar_result_renders_one_figure_and_no_baseline(
        self, findings
    ) -> None:  # type: ignore[no-untyped-def]
        html = render_finding(findings[ResultShape.SCALAR])
        assert "compare-single" in html
        assert "A typical release" not in html
        assert "Difference" not in html
        assert "Baseline" not in html

    def test_a_comparison_still_renders_as_before(
        self, findings
    ) -> None:  # type: ignore[no-untyped-def]
        """The shape that existed before this change must be untouched."""
        html = render_finding(findings[ResultShape.COMPARISON])
        assert "A typical release" in html
        assert "Difference" in html
        assert "against A typical release" in html

    def test_a_range_query_never_renders_a_return_headline(
        self, findings
    ) -> None:  # type: ignore[no-untyped-def]
        html = render_finding(findings[ResultShape.SCALAR])
        assert "trading range" in html
        assert "median 24h return" not in html

    def test_every_shape_keeps_the_methodology_and_the_releases(
        self, findings
    ) -> None:  # type: ignore[no-untyped-def]
        """Shape changes the hero, not the evidence beneath it."""
        for finding in findings.values():
            html = render_finding(finding)
            assert "How was this tested?" in html
            assert "The releases behind this" in html
            assert finding.spec.fingerprint in html


class TestFactsAndPayloadByShape:
    """Meaningless comparison fields must not reach a non-comparison summary."""

    @pytest.fixture(scope="class")
    @classmethod
    def facts(cls):  # type: ignore[no-untyped-def]
        return {
            ResultShape.RANKED: build_facts(run_thesis(ranked(limit=3), as_of=CUTOFF)),
            ResultShape.SCALAR: build_facts(run_thesis(scalar(), as_of=CUTOFF)),
            ResultShape.COMPARISON: build_facts(
                run_thesis(
                    Thesis(
                        event_type=EventType.CPI,
                        conditions=(Condition(field="btc_trend_7d", value="bearish"),),
                        versus=Comparison(),
                    ),
                    as_of=CUTOFF,
                )
            ),
        }

    def test_a_ranked_payload_carries_no_direction(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        """Nothing to have a direction against. Sending one set to
        'undetermined' would hand the model a vocabulary of outperformance to
        reach for."""
        payload = interpretation_payload(facts[ResultShape.RANKED])
        for absent in (
            "direction_vs_baseline",
            "difference_pp",
            "difference_is_material",
            "baseline",
            "cohort",
            "small_sample",
        ):
            assert absent not in payload, absent
        assert payload["ranking"]["extreme"] == "largest"
        assert len(payload["ranking"]["rows"]) == 3

    def test_a_scalar_payload_carries_a_result_but_no_comparison(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        payload = interpretation_payload(facts[ResultShape.SCALAR])
        assert payload["result"]
        assert "ranking" not in payload
        for absent in ("direction_vs_baseline", "difference_pp", "baseline"):
            assert absent not in payload, absent

    def test_a_comparison_payload_carries_the_verdict(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        payload = interpretation_payload(facts[ResultShape.COMPARISON])
        assert payload["direction_vs_baseline"] in {"better", "worse", "similar"}
        assert payload["left"]["n"] > 0
        assert payload["right"]["n"] > 0
        assert payload["higher_side"] in {"left", "right", "neither"}

    def test_only_a_comparison_has_a_direction(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        assert facts[ResultShape.RANKED].direction is Direction.UNDETERMINED
        assert facts[ResultShape.SCALAR].direction is Direction.UNDETERMINED
        assert facts[ResultShape.COMPARISON].direction is not (
            Direction.UNDETERMINED
        )

    def test_a_hit_rate_appears_only_for_a_signed_return(self) -> None:
        """A trading range is never negative, so "31/31 positive" over one
        would be true, useless, and read as a finding."""
        comparison = build_facts(
            run_thesis(
                Thesis(
                    event_type=EventType.CPI,
                    conditions=(Condition(field="btc_trend_7d", value="bearish"),),
                    outcome=Outcome(metric="range_pct"),
                    versus=Comparison(),
                ),
                as_of=CUTOFF,
            )
        )
        assert comparison.left is not None
        assert comparison.left.positive_count is None

    def test_an_extremum_gets_no_small_sample_warning(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        """The largest of 31 observed releases is a fact about the sample, not
        an estimate. Telling the reader it is "limited evidence" would be
        telling them to doubt something that happened."""
        ranked_facts = facts[ResultShape.RANKED]
        assert not ranked_facts.needs_sample_warning
        text = " ".join(deterministic_summary(ranked_facts))
        assert "limited historical evidence" not in text
        assert "observed rather than estimated" in text

    def test_a_small_median_still_warns(self) -> None:
        narrow = build_facts(
            run_thesis(
                Thesis(
                    event_type=EventType.CPI,
                    conditions=(
                        Condition(
                            field="btc_return_7d",
                            operator=Operator.BELOW,
                            threshold=-5.0,
                        ),
                    ),
                ),
                as_of=CUTOFF,
            )
        )
        assert narrow.needs_sample_warning
        assert "statistical test" in " ".join(deterministic_summary(narrow))

    def test_each_shape_gets_its_own_sentences(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        ranked_text = " ".join(deterministic_summary(facts[ResultShape.RANKED]))
        scalar_text = " ".join(deterministic_summary(facts[ResultShape.SCALAR]))
        compare_text = " ".join(
            deterministic_summary(facts[ResultShape.COMPARISON])
        )

        assert "largest" in ranked_text and "on " in ranked_text
        assert "outperformed" not in ranked_text
        assert "underperformed" not in ranked_text

        assert "Across" in scalar_text
        assert "baseline" not in scalar_text.lower()

        assert "baseline" in compare_text.lower()

    def test_no_payload_carries_a_price_series(
        self, facts
    ) -> None:  # type: ignore[no-untyped-def]
        """Ranked rows name dates and formatted values, never a path."""
        for fact in facts.values():
            payload = interpretation_payload(fact)

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


class TestMetricSeries:
    """The projection, on a hand-built study."""

    @staticmethod
    def event(
        event_id: str,
        *,
        usable: bool = True,
        range_pct: float = 3.0,
        vol: float | None = 0.4,
    ) -> EventResult:
        """One event with one window, built by hand.

        An OK window must carry its whole outcome -- ``WindowMetrics`` enforces
        that -- so the "missing reading" case uses realised volatility, which
        is genuinely optional on a complete window because a short one cannot
        support it.
        """
        complete = {
            "start_price": 43000.0,
            "end_price": 43430.0,
            "return_pct": 1.0,
            "abs_move_pct": 1.0,
            "direction": Direction_.UP,
            "high": 44000.0,
            "low": 42710.0,
            "mfe_pct": 2.3,
            "mae_pct": -0.7,
            "range_pct": range_pct,
            "realized_vol_pct": vol,
        }
        metrics = WindowMetrics(
            name="post_event_24h",
            start_utc=dt.datetime(2024, 1, 11, 13, 30, tzinfo=dt.UTC),
            end_utc=dt.datetime(2024, 1, 12, 13, 30, tzinfo=dt.UTC),
            status=WindowStatus.OK if usable else WindowStatus.INSUFFICIENT_DATA,
            **(complete if usable else {}),  # type: ignore[arg-type]
        )
        return EventResult(
            event_id=event_id,
            event_time_utc=dt.datetime(2024, 1, 11, 13, 30, tzinfo=dt.UTC),
            local_date=dt.date(2024, 1, 11),
            is_scheduled=True,
            windows={"post_event_24h": metrics},
        )

    def test_the_series_reads_the_metric_the_spec_names(self) -> None:
        readings = metric_series(
            [self.event("a", range_pct=4.5)], "post_event_24h", METRICS["range_pct"]
        )
        assert [r.value for r in readings] == [4.5]

    def test_an_unusable_window_is_omitted(self) -> None:
        readings = metric_series(
            [self.event("a"), self.event("b", usable=False)],
            "post_event_24h",
            METRICS["range_pct"],
        )
        assert [r.event.event_id for r in readings] == ["a"]

    def test_a_missing_reading_is_omitted_not_zeroed(self) -> None:
        """A metric that could not be computed is not zero. A zero would drag
        a mean down and win a 'smallest' sort outright."""
        readings = metric_series(
            [self.event("a", vol=None), self.event("b", vol=0.9)],
            "post_event_24h",
            METRICS["realized_vol_pct"],
        )
        assert [r.value for r in readings] == [0.9]
        assert [r.event.event_id for r in readings] == ["b"]

    def test_an_unknown_window_yields_nothing(self) -> None:
        assert (
            metric_series([self.event("a")], "post_event_5m", METRICS["range_pct"])
            == []
        )
