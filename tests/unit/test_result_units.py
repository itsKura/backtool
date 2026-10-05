"""Tests that an aggregate is displayed in the units it is actually in.

``frequency(return_pct > 0)`` carries three different units at once:

* the **metric** is a return, in percent of price
* the **threshold** is a return, in percent of price
* the **result** is a share of observations, in percent of releases

The first two belong on a price axis. The third does not, and the bug these
tests exist to prevent was the renderer treating all three as the same thing --
plotting ``75.0`` on a distribution of returns, labelling it "median", and
captioning the comparison chart "return (%)" while it showed shares.

The failure is quiet and total: every number was right, every chart drew, and
the page said BTC's median return was +75%. So the check cannot be "does it
render" -- it has to be "is this figure allowed on that axis", which is a
question about the aggregation, asked before anything is drawn.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from backtool.core.types import EventType
from backtool.reporting.charts import BarItem, distribution_chart, diverging_bar_chart
from backtool.research.experiment import run_thesis
from backtool.research.interpretation import build_facts, deterministic_summary
from backtool.research.thesis import Comparison, Condition, Outcome, Thesis
from backtool.research.vocabulary import Aggregation, Operator
from backtool.web.research_page import render_finding

CUTOFF = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)

BEARISH = Condition(field="btc_trend_7d", value="bearish")
#: Every usable release -- the baseline, as a group with no filters.
ALL_RELEASES = Comparison()


def visible(html: str) -> str:
    """Reader-visible text, with tags and entities stripped.

    Checks run against this rather than the raw markup: an SVG class attribute
    is not something anyone reads, and matching one would fail a test for a
    reason that has nothing to do with what the page says.
    """
    return re.sub(r"&[a-z]+;", " ", re.sub(r"<[^>]+>", " ", html))


def frequency_thesis(
    *,
    metric: str = "return_pct",
    threshold: float = 0.0,
    operator: Operator = Operator.ABOVE,
    conditions: tuple[Condition, ...] = (BEARISH,),
    versus: Comparison | None = ALL_RELEASES,
) -> Thesis:
    return Thesis(
        event_type=EventType.CPI,
        conditions=conditions,
        versus=versus,
        outcome=Outcome(
            metric=metric,
            aggregation=Aggregation.FREQUENCY,
            threshold=threshold,
            threshold_operator=operator,
        ),
    )


class TestAggregationDeclaresItsUnits:
    """The semantics the renderer asks for, at the source."""

    @pytest.mark.parametrize(
        ("aggregation", "metric_units"),
        [
            (Aggregation.MEAN, True),
            (Aggregation.MEDIAN, True),
            (Aggregation.FREQUENCY, False),
            (Aggregation.COUNT, False),
            (Aggregation.ROWS, False),
        ],
    )
    def test_each_aggregation_says_whether_it_yields_metric_units(
        self, aggregation: Aggregation, metric_units: bool
    ) -> None:
        assert aggregation.yields_metric_units is metric_units

    def test_a_non_metric_aggregation_names_what_it_measures(self) -> None:
        assert Aggregation.FREQUENCY.result_noun == "share of releases"
        assert Aggregation.COUNT.result_noun == "releases"

    def test_a_metric_aggregation_defers_to_the_metric(self) -> None:
        """Empty, because the metric's own label is the honest caption and this
        enum has no business knowing it."""
        assert Aggregation.MEDIAN.result_noun == ""
        assert Aggregation.MEAN.result_noun == ""


class TestChartsDescribeWhatTheyPlot:
    """The captions were hardcoded to "return" whatever they were handed."""

    def test_the_bar_chart_caption_is_not_hardcoded(self) -> None:
        svg = diverging_bar_chart(
            [BarItem(label="a", value=75.0)], quantity="share of releases"
        )
        assert "share of releases (%)" in svg
        assert "return (%)" not in svg

    def test_the_distribution_caption_is_not_hardcoded(self) -> None:
        svg = distribution_chart([1.0, 2.0, 3.0], quantity="trading range")
        assert "trading range (%)" in svg
        assert "return (%)" not in svg

    def test_the_default_caption_is_unchanged(self) -> None:
        """Existing callers that really do plot returns keep working."""
        assert "return (%)" in diverging_bar_chart([BarItem(label="a", value=1.0)])
        assert "return (%)" in distribution_chart([1.0, 2.0, 3.0])


class TestFrequencyRendering:
    """The reported bug, end to end on real data."""

    @pytest.fixture(scope="class")
    @classmethod
    def finding(cls):  # type: ignore[no-untyped-def]
        return run_thesis(frequency_thesis(), as_of=CUTOFF)

    @pytest.fixture(scope="class")
    @classmethod
    def html(cls, finding) -> str:  # type: ignore[no-untyped-def]
        return render_finding(finding, deterministic_summary(build_facts(finding)))

    def test_the_deterministic_numbers_are_unchanged(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        """The fix is to presentation. If these move, something else broke."""
        assert finding.matched_count == 6
        assert finding.sample_size == 8
        assert finding.value == pytest.approx(75.0)
        assert finding.comparison_matched_count == 22
        assert finding.comparison_sample_size == 31
        assert finding.comparison_value == pytest.approx(70.97, abs=0.01)
        assert finding.difference_pp == pytest.approx(4.03, abs=0.01)

    def test_six_of_eight_renders_as_a_share_not_a_return(
        self, html: str
    ) -> None:
        """75% of releases, never "+75.00%" of price."""
        assert "75%" in html
        assert "+75.00%" not in html
        assert "+75%" not in html

    def test_both_sides_show_the_fraction_behind_the_share(
        self, html: str
    ) -> None:
        text = visible(html)
        assert "6 of 8" in text
        assert "22 of 31" in text

    def test_the_comparison_chart_axis_does_not_say_return(
        self, html: str
    ) -> None:
        captions = re.findall(r'class="axis-caption"[^>]*>(.*?)</text>', html)
        assert captions, "no axis caption rendered"
        assert "share of releases (%)" in captions
        assert "return (%)" not in captions

    def test_the_frequency_is_never_labelled_a_median(
        self, finding, html: str
    ) -> None:  # type: ignore[no-untyped-def]
        """The precise failure: "median +75.00%" over a chart of returns."""
        text = visible(html).lower()
        share = f"{finding.value:.2f}"
        for phrase in (f"median +{share}", f"median {share}", "median +75"):
            assert phrase not in text, phrase

    def test_the_aggregate_never_reaches_the_distribution_chart(
        self, finding, html: str
    ) -> None:  # type: ignore[no-untyped-def]
        """A share of releases is not an observation of the metric, so it
        cannot be drawn on a chart of observations."""
        assert not finding.result_in_metric_units
        assert "Distribution of" not in html

    def test_the_headline_reports_both_sides_as_shares(
        self, finding
    ) -> None:  # type: ignore[no-untyped-def]
        verdict = finding.verdict()
        assert "75%" in verdict and "71%" in verdict
        assert "6 of 8" in verdict and "22 of 31" in verdict
        assert "+4.03 percentage points" in verdict
        assert "median" not in verdict.lower()

    def test_the_tile_subtitle_is_not_called_a_kind_of_return(
        self, html: str
    ) -> None:
        """"frequency 24h return" reads as a return. The measure line now says
        what was actually counted."""
        text = visible(html)
        assert "how often the 24h return was above 0.00%" in text
        assert "frequency 24h return" not in text

    def test_the_full_statistics_may_still_show_real_medians(
        self, html: str
    ) -> None:
        """The cohort table's "Median return" column is a genuine median of
        returns. Suppressing it would be overcorrecting -- the bug was calling
        the *frequency* a median, not showing medians at all."""
        assert "Median return" in html


class TestFrequencyGenerally:
    """Not a patch for one question."""

    def test_a_different_metric_and_threshold_works(self) -> None:
        """"How often did BTC move more than 2% after CPI?" -- a different
        metric, a different threshold, no comparison group."""
        finding = run_thesis(
            frequency_thesis(
                metric="abs_move_pct",
                threshold=2.0,
                conditions=(),
                versus=None,
            ),
            as_of=CUTOFF,
        )
        html = render_finding(finding)

        assert not finding.result_in_metric_units
        assert finding.result_quantity == "share of releases"
        assert "Distribution of" not in html
        assert "+39.00%" not in html
        assert "39%" in html
        assert "12 of 31" in visible(html)

    def test_a_count_is_also_not_in_metric_units(self) -> None:
        """COUNT returns a number of releases. It gets no marker either."""
        finding = run_thesis(
            Thesis(
                event_type=EventType.CPI,
                outcome=Outcome(metric="range_pct", aggregation=Aggregation.COUNT),
            ),
            as_of=CUTOFF,
        )
        assert not finding.result_in_metric_units
        assert finding.result_quantity == "releases"
        html = render_finding(finding)
        # The distribution still renders -- a count does not make the spread of
        # the metric meaningless -- but nothing is marked on it.
        assert "Distribution of" in html
        assert "median-line" not in html or f"{finding.value:.2f}" not in visible(html)


class TestMetricAggregationsAreUntouched:
    """The shapes that were already right must stay right."""

    def test_a_median_return_still_marks_its_distribution(self) -> None:
        finding = run_thesis(
            Thesis(
                event_type=EventType.CPI,
                conditions=(BEARISH,),
                versus=Comparison(),
                outcome=Outcome(metric="return_pct"),
            ),
            as_of=CUTOFF,
        )
        html = render_finding(finding)

        assert finding.result_in_metric_units
        assert finding.result_quantity == "return"
        assert "Distribution of return" in html
        captions = re.findall(r'class="axis-caption"[^>]*>(.*?)</text>', html)
        assert "return (%)" in captions
        # The marker is the median, and it is a return.
        assert finding.value == pytest.approx(0.38, abs=0.01)
        assert "+0.38%" in html

    def test_a_median_range_still_marks_its_distribution(self) -> None:
        finding = run_thesis(
            Thesis(
                event_type=EventType.CPI,
                outcome=Outcome(metric="range_pct"),
            ),
            as_of=CUTOFF,
        )
        html = render_finding(finding)

        assert finding.result_in_metric_units
        assert finding.result_quantity == "trading range"
        assert "Distribution of trading range" in html
        captions = re.findall(r'class="axis-caption"[^>]*>(.*?)</text>', html)
        assert "trading range (%)" in captions
        assert "return (%)" not in captions

    def test_the_existing_comparison_numbers_are_unchanged(self) -> None:
        """The question this product has answered since the beginning."""
        finding = run_thesis(
            Thesis(
                event_type=EventType.CPI,
                conditions=(BEARISH,),
                versus=Comparison(),
                outcome=Outcome(horizon="24h"),
            ),
            as_of=CUTOFF,
        )
        assert finding.sample_size == 8
        assert finding.value == pytest.approx(0.38, abs=0.01)
        assert finding.comparison_sample_size == 31
        assert finding.comparison_value == pytest.approx(0.95, abs=0.01)
        assert finding.difference_pp == pytest.approx(-0.57, abs=0.01)
        assert "lower" in finding.verdict()

    def test_a_ranked_result_is_still_unmarked(self) -> None:
        from backtool.research.vocabulary import Order

        finding = run_thesis(
            Thesis(
                event_type=EventType.CPI,
                outcome=Outcome(
                    metric="range_pct",
                    aggregation=Aggregation.ROWS,
                    order=Order.DESCENDING,
                    limit=5,
                ),
            ),
            as_of=CUTOFF,
        )
        html = render_finding(finding)
        assert "Distribution of trading range" in html
        assert not finding.result_in_metric_units
