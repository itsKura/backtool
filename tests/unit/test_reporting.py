"""Tests for SVG chart generation and the HTML report.

A chart that misplaces a label produces a wrong-looking page, not an exception,
so the geometry rules are asserted rather than eyeballed.
"""

from __future__ import annotations

import re

import pytest

from backtool.reporting.charts import (
    BAR_HEIGHT,
    BarItem,
    _place_value_label,
    _ticks,
    diverging_bar_chart,
)


class TestBarGeometry:
    def test_renders_one_bar_per_item(self) -> None:
        svg = diverging_bar_chart(
            [BarItem("a", 1.0), BarItem("b", -2.0), BarItem("c", 3.0)]
        )
        assert svg.count("bar-up") == 2
        assert svg.count("bar-down") == 1

    def test_positive_and_negative_get_different_classes(self) -> None:
        """Colour carries polarity, so the sign must drive the class."""
        assert 'class="bar-up"' in diverging_bar_chart([BarItem("a", 5.0)])
        assert 'class="bar-down"' in diverging_bar_chart([BarItem("a", -5.0)])

    def test_zero_is_treated_as_positive(self) -> None:
        assert 'class="bar-up"' in diverging_bar_chart([BarItem("a", 0.0)])

    def test_bar_never_exceeds_the_thickness_cap(self) -> None:
        assert BAR_HEIGHT <= 24

    def test_empty_input_renders_a_placeholder_not_a_crash(self) -> None:
        svg = diverging_bar_chart([])
        assert "no usable observations" in svg
        assert svg.startswith("<svg")

    def test_all_equal_values_still_render(self) -> None:
        """A degenerate range must not divide by zero."""
        svg = diverging_bar_chart([BarItem("a", 2.0), BarItem("b", 2.0)])
        assert svg.count("bar-up") == 2

    def test_labels_are_xml_escaped(self) -> None:
        svg = diverging_bar_chart([BarItem("a<b>&c", 1.0)])
        assert "a<b>&c" not in svg
        assert "&lt;b&gt;" in svg

    def test_every_bar_carries_a_hover_title(self) -> None:
        svg = diverging_bar_chart([BarItem("a", 1.0, tooltip="detail here")])
        assert "<title>detail here</title>" in svg

    def test_baseline_is_drawn(self) -> None:
        assert 'class="baseline"' in diverging_bar_chart([BarItem("a", 1.0)])

    def test_bar_path_is_rounded_on_the_data_end_only(self) -> None:
        """Two arcs, both at the far end; the baseline end stays square so the
        zero line reads as a hard reference."""
        svg = diverging_bar_chart([BarItem("a", 5.0)])
        path = re.search(r'<path d="([^"]+)" class="bar-up"', svg)
        assert path is not None
        assert path.group(1).count("A ") == 2


class TestValueLabelPlacement:
    """A label that collides with the row gutter or runs off the canvas is a
    layout bug that never raises, so placement is asserted directly."""

    PLOT_LEFT = 148.0
    PLOT_RIGHT = 652.0

    def test_short_positive_bar_labels_outside(self) -> None:
        x, anchor, inside = _place_value_label(
            "+1.00", bar_x=300.0, zero_x=250.0, positive=True,
            plot_left=self.PLOT_LEFT, plot_right=self.PLOT_RIGHT,
        )
        assert inside is False
        assert anchor == "start"
        assert x > 300.0

    def test_long_positive_bar_flips_the_label_inside(self) -> None:
        x, anchor, inside = _place_value_label(
            "+12.34", bar_x=650.0, zero_x=200.0, positive=True,
            plot_left=self.PLOT_LEFT, plot_right=self.PLOT_RIGHT,
        )
        assert inside is True
        assert anchor == "end"
        assert x < 650.0

    def test_long_negative_bar_flips_the_label_inside(self) -> None:
        """This is the case that collided with the row label before the fix."""
        x, anchor, inside = _place_value_label(
            "-12.34", bar_x=150.0, zero_x=600.0, positive=False,
            plot_left=self.PLOT_LEFT, plot_right=self.PLOT_RIGHT,
        )
        assert inside is True
        assert anchor == "start"
        assert x > 150.0

    def test_short_negative_bar_labels_outside(self) -> None:
        _, anchor, inside = _place_value_label(
            "-1.00", bar_x=400.0, zero_x=450.0, positive=False,
            plot_left=self.PLOT_LEFT, plot_right=self.PLOT_RIGHT,
        )
        assert inside is False
        assert anchor == "end"

    def test_tiny_bar_keeps_the_label_outside(self) -> None:
        """A bar too narrow to hold the text must not swallow it."""
        _, _, inside = _place_value_label(
            "-0.01", bar_x=645.0, zero_x=650.0, positive=False,
            plot_left=self.PLOT_LEFT, plot_right=self.PLOT_RIGHT,
        )
        assert inside is False

    def test_inside_labels_use_the_contrasting_class(self) -> None:
        svg = diverging_bar_chart([BarItem("x", -9.9), BarItem("y", 0.01)])
        assert "value-label-inside" in svg


class TestTicks:
    def test_ticks_are_round_numbers(self) -> None:
        ticks = _ticks(-5.0, 5.0)
        assert all(abs(t * 2 - round(t * 2)) < 1e-9 for t in ticks)

    def test_ticks_lie_within_the_range(self) -> None:
        ticks = _ticks(-3.2, 7.8)
        assert all(-3.2 <= t <= 7.8 for t in ticks)

    def test_degenerate_range_returns_something(self) -> None:
        assert _ticks(0.0, 0.0) == [0.0]


class TestHtmlReport:
    @pytest.fixture
    def study(self, tmp_path):  # type: ignore[no-untyped-def]
        import datetime as dt

        from tests.unit.test_runner_and_aggregate import ScriptedSource, make_event

        from backtool.data.service import MarketDataService
        from backtool.data.storage.sqlite import SQLiteCandleRepository
        from backtool.research.runner import run_study
        from backtool.research.spec import ResearchSpec
        from backtool.research.windows import WindowSpec

        as_of = dt.datetime(2024, 6, 1, tzinfo=dt.UTC)
        source = ScriptedSource({3: 1.5, 4: -2.5, 5: None})
        service = MarketDataService(
            source, SQLiteCandleRepository(tmp_path / "candles.db")
        )
        spec = ResearchSpec(
            event_count=3,
            windows=(WindowSpec(name="post_event_1h", start="0h", end="1h"),),
            as_of=as_of,
        )
        return run_study(
            spec, service, [make_event(3), make_event(4), make_event(5)], now=as_of
        )

    def test_produces_a_complete_html_document(self, study) -> None:  # type: ignore[no-untyped-def]
        from backtool.reporting.html import render_report

        html = render_report(study)
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert "<svg" in html

    def test_renders_assumptions_with_values(self, study) -> None:  # type: ignore[no-untyped-def]
        """Regression: describe() used to be re-parsed from formatted strings,
        which dropped the fingerprint's value."""
        from backtool.reporting.html import render_report

        html = render_report(study)
        assert f"<dd>{study.spec.fingerprint}</dd>" in html
        assert "no look-ahead" in html

    def test_states_coverage(self, study) -> None:  # type: ignore[no-untyped-def]
        from backtool.reporting.html import render_report

        assert "3 event(s) selected of 3 requested" in render_report(study)

    def test_evidence_table_lists_every_event(self, study) -> None:  # type: ignore[no-untyped-def]
        """Including the unusable one -- that is the point of the table."""
        from backtool.reporting.html import render_report

        html = render_report(study)
        for day in (3, 4, 5):
            assert f"FOMC-2024-01-{day:02d}" in html
        assert ">n/a<" in html

    def test_carries_the_exploratory_caveat(self, study) -> None:  # type: ignore[no-untyped-def]
        from backtool.reporting.html import render_report

        html = render_report(study)
        assert "not a trading signal" in html.lower()
        assert "multiple comparisons" in html

    def test_dark_is_the_default_theme(self, study) -> None:  # type: ignore[no-untyped-def]
        """Dark is unconditional, not OS-conditional -- the page should not
        change appearance based on a system setting the reader did not choose
        for this app."""
        from backtool.reporting.html import render_report

        html = render_report(study)
        base = html[html.index(":root {") : html.index("* { box-sizing")]
        assert "color-scheme: dark" in base
        assert "@media (prefers-color-scheme" not in html

    def test_light_remains_available_as_an_opt_in(self, study) -> None:  # type: ignore[no-untyped-def]
        """A reader who needs light (print, bright room, low vision) is not
        locked out."""
        from backtool.reporting.html import render_report

        assert ':root[data-theme="light"]' in render_report(study)

    def test_is_self_contained(self, study) -> None:  # type: ignore[no-untyped-def]
        """No network requests, no external assets -- the file must work
        offline and survive being emailed."""
        from backtool.reporting.html import render_report

        html = render_report(study)
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "<script" not in html

    def test_write_report_creates_the_file(self, study, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from backtool.reporting.html import write_report

        path = write_report(study, tmp_path / "nested" / "report.html")
        assert path.endswith("report.html")
