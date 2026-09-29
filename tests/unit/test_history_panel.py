"""Tests for the distribution chart and the redesigned history section.

Two things are being protected here. The first is arithmetic: a histogram whose
bins straddle zero would colour a mixed bucket as one sign, which is exactly the
thing the colour is supposed to encode. The second is vocabulary: the engine's
own names are correct and belong in the methodology block, and nowhere above it.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from backtool.reporting.charts import _bin_counts, _bin_step, distribution_chart
from backtool.research.metrics import Direction, WindowMetrics, WindowStatus
from backtool.research.results import EventResult, StudyResult
from backtool.research.spec import ResearchSpec
from backtool.research.windows import DEFAULT_WINDOWS
from backtool.web.history_panel import render_history

NOW = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC)


def _metrics(name: str, return_pct: float | None) -> WindowMetrics:
    """One window's metrics, usable or not, without touching a candle."""
    if return_pct is None:
        return WindowMetrics(
            name=name,
            start_utc=NOW,
            end_utc=NOW + dt.timedelta(hours=1),
            status=WindowStatus.INSUFFICIENT_DATA,
            reason="No candles in range.",
        )
    end_price = 100.0 * (1 + return_pct / 100)
    direction = (
        Direction.UP
        if return_pct > 0
        else Direction.DOWN
        if return_pct < 0
        else Direction.FLAT
    )
    return WindowMetrics(
        name=name,
        start_utc=NOW,
        end_utc=NOW + dt.timedelta(hours=1),
        status=WindowStatus.OK,
        start_price=100.0,
        end_price=end_price,
        return_pct=return_pct,
        direction=direction,
        high=max(100.0, end_price) + 1.0,
        low=min(100.0, end_price) - 1.0,
        abs_move_pct=abs(return_pct),
        mfe_pct=abs(return_pct),
        mae_pct=-abs(return_pct),
        range_pct=abs(return_pct) * 2,
    )


def _study(returns: list[float], *, symbol: str = "BTCUSDT") -> StudyResult:
    spec = ResearchSpec(
        symbol=symbol, event_count=len(returns), windows=DEFAULT_WINDOWS, as_of=NOW
    )
    events = tuple(
        EventResult(
            event_id=f"FOMC-2026-01-{index + 1:02d}",
            event_time_utc=NOW - dt.timedelta(days=index),
            local_date=dt.date(2026, 1, index + 1),
            is_scheduled=True,
            windows={
                window.name: _metrics(window.name, value) for window in DEFAULT_WINDOWS
            },
        )
        for index, value in enumerate(returns)
    )
    return StudyResult(spec=spec, generated_at=NOW, events=events)


class TestBinning:
    def test_a_bin_edge_always_lands_on_zero(self) -> None:
        """A bucket spanning -0.2% to +0.3% would have to be coloured as one
        sign or the other, misreporting the thing colour encodes."""
        values = [-2.4, -0.3, 0.1, 1.7, 3.9]
        step = _bin_step(values)
        low = int(min(min(values), 0.0) // step)
        high = int(max(max(values), 0.0) // step)
        edges = [(low + i) * step for i in range(high - low + 2)]
        assert any(abs(edge) < 1e-9 for edge in edges)

    def test_every_observation_lands_in_exactly_one_bin(self) -> None:
        values = [-4.88, -1.55, 0.0, 0.02, 1.24, 4.54]
        step = _bin_step(values)
        low = int(min(min(values), 0.0) // step)
        high = int(max(max(values), 0.0) // step)
        assert sum(_bin_counts(values, step, low, high)) == len(values)

    def test_zero_counts_as_positive(self) -> None:
        """Consistent with how direction is defined everywhere else."""
        step = _bin_step([0.0, 1.0])
        assert _bin_counts([0.0], step, 0, 1)[0] == 1

    def test_an_outlier_does_not_produce_hundreds_of_bins(self) -> None:
        values = [0.1, 0.2, -0.1, 90.0]
        step = _bin_step(values)
        span = max(values) - min(min(values), 0.0)
        assert span / step <= 24


class TestDistributionChart:
    def test_colours_split_on_the_sign_of_the_bucket(self) -> None:
        svg = distribution_chart([-3.0, -2.0, 2.0, 3.0])
        assert "bar-up" in svg
        assert "bar-down" in svg

    def test_marks_the_median_it_is_given(self) -> None:
        """The chart never computes a statistic; it is handed one."""
        svg = distribution_chart([1.0, 2.0, 3.0], median=2.0)
        assert "median-line" in svg
        assert "median +2.00%" in svg

    def test_omits_the_median_rule_when_not_given(self) -> None:
        assert "median-line" not in distribution_chart([1.0, 2.0])

    def test_empty_input_renders_a_placeholder_not_a_crash(self) -> None:
        svg = distribution_chart([])
        assert svg.startswith("<svg")
        assert "no usable observations" in svg

    def test_identical_values_still_render(self) -> None:
        """A zero-width span must not divide by zero."""
        assert distribution_chart([2.0, 2.0, 2.0]).startswith("<svg")

    def test_bar_counts_sum_to_the_sample(self) -> None:
        values = [-1.2, -0.4, 0.3, 0.9, 1.1, 2.7]
        svg = distribution_chart(values)
        labels = re.findall(r'class="value-label" text-anchor="middle">(\d+)<', svg)
        assert sum(int(n) for n in labels) == len(values)


class TestHistoryFragment:
    @pytest.fixture
    def html(self) -> str:
        return render_history(_study([-4.88, 1.24, -1.55, 1.83, 0.63, 3.39, -0.82]))

    def test_returns_three_addressable_slots(self, html: str) -> None:
        for slot in ("summary", "evidence", "methodology"):
            assert f'data-slot="{slot}"' in html

    def test_one_window_is_selected_and_the_rest_are_hidden(self, html: str) -> None:
        """Every window is rendered so switching needs no round trip, but only
        one is on screen."""
        panels = re.findall(r'class="window-panel" data-window="\d+"( hidden)?>', html)
        assert len(panels) == len(DEFAULT_WINDOWS)
        assert panels.count("") == 1

        tabs = re.findall(r'class="window-tab"[^>]*aria-selected="(\w+)"', html)
        assert tabs.count("true") == 1
        assert len(tabs) == len(DEFAULT_WINDOWS)

    def test_the_reaction_window_opens_first(self, html: str) -> None:
        """A reader who clicked an event came for the reaction, not the run-up."""
        selected = re.search(
            r'data-window="(\d+)" aria-selected="true"', html
        )
        assert selected is not None
        assert DEFAULT_WINDOWS[int(selected.group(1))].start_offset.total_seconds() >= 0

    def test_window_names_appear_only_in_the_methodology(self, html: str) -> None:
        above = html[: html.index('data-slot="methodology"')]
        for window in DEFAULT_WINDOWS:
            assert window.name not in above

    def test_statistics_carry_their_product_names(self, html: str) -> None:
        for label in (
            "Median return",
            "Positive events",
            "Average trading range",
            "Events analyzed",
            "Return variability",
        ):
            assert label in html

    def test_standard_deviation_is_not_called_std_dev(self, html: str) -> None:
        assert "std dev" not in html

    def test_a_move_that_rounds_to_nothing_is_not_shown_as_negative(self) -> None:
        """'-0.00%' reads as a typo, and colouring it red reports a direction
        the displayed number does not have."""
        html = render_history(_study([-0.004, -0.003]))
        assert "-0.00%" not in html
        assert "0.00%" in html

    def test_evidence_and_methodology_are_collapsed(self, html: str) -> None:
        """Both are secondary: kept so a number can be checked, not so it has
        to be read first."""
        assert html.count("<details") == 2
        assert "<details open" not in html

    def test_evidence_columns_use_window_labels(self, html: str) -> None:
        evidence = html[html.index('data-slot="evidence"') :]
        assert "<th>24h before</th>" in evidence
        assert "<th>1h after</th>" in evidence

    def test_methodology_keeps_the_exact_technical_wording(self, html: str) -> None:
        """A reader who opens a section called methodology wants the anchor
        rule stated precisely, not paraphrased."""
        method = html[html.index('data-slot="methodology"') :]
        assert "anchor rule" in method
        assert "spec fingerprint" in method
        assert "no look-ahead" in method

    def test_coverage_is_always_stated(self, html: str) -> None:
        assert "Coverage." in html

    def test_an_excluded_event_is_declared(self) -> None:
        """An average over twenty releases that quietly rested on nineteen is
        the one genuinely dishonest thing this page could do."""
        study = _study([1.0, 2.0, None, 3.0])  # type: ignore[list-item]
        html = render_history(study)
        assert "1 excluded" in html

    def test_a_window_with_no_data_says_so(self) -> None:
        html = render_history(_study([None, None]))  # type: ignore[list-item]
        assert "No usable observations" in html
