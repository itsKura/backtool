"""Self-contained HTML report for a completed study.

One file, no assets, no JavaScript, no network. Opens in any browser and can be
emailed or committed as a record of a run.

The layout follows the product's core promise: assumptions first, then the
headline statistics, then the individual observations behind them. A reader who
distrusts a number can always scroll to the evidence that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from backtool.reporting.charts import BarItem, diverging_bar_chart
from backtool.research.aggregate import WindowAggregate, aggregate_study
from backtool.research.results import StudyResult

# Palette: the data-viz reference instance's diverging pair, used unchanged.
# Blue/red rather than green/red -- see charts.py for why.
#
# Dark is the default, not an OS-conditional variant. The steps below are the
# reference palette's *dark* column, chosen and contrast-checked against the
# dark surface rather than produced by inverting the light ones -- an automatic
# flip drops saturated hues below the contrast floor. Light remains available
# under an explicit `data-theme="light"` stamp, so a reader who needs it (print,
# a bright room, low vision) is not locked out.
_STYLE = """
:root {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --surface-2: #262624;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e85;
  --grid: #343431;
  --up: #3987e5;
  --down: #e66767;
  --accent: #3987e5;
}
:root[data-theme="light"] {
  color-scheme: light;
  --surface-0: #f4f3f0;
  --surface-1: #fcfcfb;
  --surface-2: #eceae5;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #77756f;
  --grid: #e3e1db;
  --up: #2a78d6;
  --down: #e34948;
  --accent: #2a78d6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px 64px;
  background: var(--surface-0);
  color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 40px 0 12px; letter-spacing: -0.005em; }
.sub { color: var(--text-secondary); margin: 0 0 24px; font-size: 14px; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--grid);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 13px; }
.assumptions { display: grid; grid-template-columns: max-content 1fr; gap: 4px 20px; }
.assumptions dt { color: var(--text-muted); }
.assumptions dd { margin: 0; color: var(--text-primary); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; }
.kpi { background: var(--surface-2); border-radius: 8px; padding: 12px 14px; }
.kpi .label { color: var(--text-muted); font-size: 12px; text-transform: none; }
.kpi .value { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; margin-top: 2px; }
.kpi .value.pos { color: var(--up); }
.kpi .value.neg { color: var(--down); }
.legend { display: flex; gap: 18px; align-items: center; margin: 4px 0 8px; font-size: 13px;
          color: var(--text-secondary); }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block;
          margin-right: 6px; vertical-align: -1px; }
.swatch.up { background: var(--up); }
.swatch.down { background: var(--down); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-muted); font-weight: 500; white-space: nowrap; }
td.pos { color: var(--up); }
td.neg { color: var(--down); }
td.na { color: var(--text-muted); }
.caveat { border-left: 3px solid var(--down); padding: 10px 14px; margin: 14px 0;
          background: var(--surface-2); border-radius: 0 8px 8px 0;
          color: var(--text-secondary); font-size: 13px; }
.chart .gridline { stroke: var(--grid); stroke-width: 1; }
.chart .baseline { stroke: var(--text-muted); stroke-width: 1; }
.chart .bar-up { fill: var(--up); }
.chart .bar-down { fill: var(--down); }
.chart .row-label { fill: var(--text-secondary); font-size: 11.5px;
                    font-family: ui-monospace, Consolas, monospace; }
.chart .value-label { fill: var(--text-secondary); font-size: 11.5px;
                      font-family: ui-monospace, Consolas, monospace; }
.chart .value-label-inside { fill: var(--surface-1); font-size: 11.5px; font-weight: 600;
                             font-family: ui-monospace, Consolas, monospace; }
.chart .axis-label { fill: var(--text-muted); font-size: 11px; }
.chart .axis-caption { fill: var(--text-muted); font-size: 11px; }
.chart .bar-group { cursor: default; }
.chart .bar-group:hover .bar-up,
.chart .bar-group:hover .bar-down { filter: brightness(1.15); }
.chart .bar-group:hover .row-label { fill: var(--text-primary); }
footer { color: var(--text-muted); font-size: 12px; margin-top: 40px;
         border-top: 1px solid var(--grid); padding-top: 14px; }
.card.ai { border-left: 3px solid var(--accent); }
.ai-badge { color: var(--text-muted); font-size: 12px; margin: 0 0 12px;
            text-transform: none; letter-spacing: 0.01em; }
.ai-headline { font-size: 17px; font-weight: 600; margin: 0 0 16px;
               letter-spacing: -0.01em; }
.ai-label { color: var(--text-muted); font-size: 12px; margin: 14px 0 4px; }
.card.ai ul { margin: 0; padding-left: 20px; }
.card.ai li { margin-bottom: 5px; color: var(--text-secondary); }
"""


def render_body(study: StudyResult, interpretation: Any | None = None) -> str:
    """Render just the report sections, with no document wrapper.

    Split out from :func:`render_report` so the identical markup can be written
    to a standalone file or injected into a live page. The two must never drift:
    a served result that differs from a saved one would make the saved file a
    poor record of what was seen.

    Args:
        study: The finished run.
        interpretation: Optional AI reading of the results. Typed loosely so the
            reporting layer does not import ``backtool.ai`` -- reports must
            render identically with or without it, and the AI layer is optional.
    """
    aggregates = aggregate_study(study)

    sections = [
        _header(study),
        _assumptions(study),
        _coverage(study),
    ]
    if interpretation is not None:
        sections.append(_interpretation(interpretation))
    for window_spec in study.spec.windows:
        sections.append(_window_section(study, aggregates[window_spec.name]))
    sections.append(_evidence_table(study))
    sections.append(_footer(study))
    return "".join(sections)


def render_report(study: StudyResult, interpretation: Any | None = None) -> str:
    """Render a complete study as a standalone HTML document."""
    spec = study.spec
    body = render_body(study, interpretation)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(spec.symbol)} around {escape(spec.event_type.value)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        f'<div class="wrap">\n{body}\n</div>\n</body>\n</html>\n'
    )


def _header(study: StudyResult) -> str:
    spec = study.spec
    return (
        f"<h1>{escape(spec.symbol)} price action around "
        f"{escape(spec.event_type.value)}</h1>"
        f'<p class="sub">{len(study.events)} events · {spec.interval.value} candles · '
        f"generated {study.generated_at:%Y-%m-%d %H:%M} UTC · "
        f'spec <span class="mono">{spec.fingerprint}</span></p>'
    )


def _assumptions(study: StudyResult) -> str:
    rows = [
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"
        for label, value in study.spec.describe_items()
    ]
    return (
        "<h2>Assumptions</h2>"
        '<div class="card"><dl class="assumptions mono">'
        f'{"".join(rows)}</dl></div>'
    )


def _coverage(study: StudyResult) -> str:
    return f'<div class="card"><strong>Coverage.</strong> {escape(study.coverage_note())}.</div>'


def render_interpretation_card(interpretation: Any) -> str:
    """The interpretation card without the surrounding heading.

    Public so the web page can render a summary on demand and get markup
    identical to what a saved report contains -- the same words should not look
    like two different things depending on where they are read.
    """
    return _interpretation_body(interpretation)


def _interpretation(interpretation: Any) -> str:
    """Render the AI reading, clearly marked as interpretation not measurement.

    Placed after the assumptions and before the charts, and visually distinct,
    so a reader is never in doubt about which parts of the page were computed
    and which were written.
    """
    return "<h2>Interpretation</h2>" + _interpretation_body(interpretation)


def _interpretation_body(interpretation: Any) -> str:
    """The card itself, shared by the report and the live page."""

    def block(title: str, items: list[str]) -> str:
        if not items:
            return ""
        entries = "".join(f"<li>{escape(item)}</li>" for item in items)
        return f"<p class='ai-label'>{escape(title)}</p><ul>{entries}</ul>"

    return (
        '<div class="card ai">'
        '<p class="ai-badge">Written by a language model from the statistics '
        "on this page. It computed none of them.</p>"
        f'<p class="ai-headline">{escape(interpretation.headline)}</p>'
        f"{block('Findings', list(interpretation.findings))}"
        f"{block('Caveats', list(interpretation.caveats))}"
        f"{block('Worth running next', list(interpretation.suggested_followups))}"
        "</div>"
    )


def _window_section(study: StudyResult, agg: WindowAggregate) -> str:
    items = [
        BarItem(
            label=event.event_id.replace("FOMC-", ""),
            value=event.windows[agg.window].return_pct or 0.0,
            tooltip=_tooltip(event.event_id, event.windows[agg.window]),
        )
        for event in study.usable_events(agg.window)
    ]

    caveat = (
        f'<div class="caveat"><strong>Small sample.</strong> {escape(agg.caveat)}</div>'
        if agg.caveat
        else ""
    )

    return (
        f"<h2>{escape(agg.window)}</h2>"
        f'<div class="card">{_kpis(agg)}{caveat}'
        '<div class="legend">'
        '<span><span class="swatch up"></span>positive return</span>'
        '<span><span class="swatch down"></span>negative return</span>'
        "</div>"
        f"{diverging_bar_chart(items)}"
        "</div>"
    )


def _kpis(agg: WindowAggregate) -> str:
    if agg.sample_size == 0:
        return (
            '<div class="kpi"><div class="label">observations</div>'
            '<div class="value">0</div></div>'
        )

    stdev = (
        f"{agg.stdev_return_pct:.2f}" if agg.stdev_return_pct is not None else "&mdash;"
    )
    tiles = [
        ("observations", f"{agg.sample_size}", ""),
        ("mean return", f"{agg.mean_return_pct:+.2f}%", _sign(agg.mean_return_pct)),
        ("median return", f"{agg.median_return_pct:+.2f}%", _sign(agg.median_return_pct)),
        ("positive", f"{agg.up_rate_pct:.0f}%", ""),
        ("std dev", f"{stdev}%", ""),
        ("mean range", f"{agg.mean_range_pct:.2f}%", ""),
    ]
    cells = "".join(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value {css}">{value}</div></div>'
        for label, value, css in tiles
    )
    return f'<div class="kpis">{cells}</div>'


def _evidence_table(study: StudyResult) -> str:
    """Every observation behind every statistic above.

    This is the part that makes a conclusion checkable: a reader who doubts a
    mean can read the numbers it was computed from, including the ones that were
    excluded and why.
    """
    names = [window.name for window in study.spec.windows]
    head = "".join(f"<th>{escape(name)}</th>" for name in names)

    rows = []
    for event in study.events:
        cells = []
        for name in names:
            metrics = event.windows[name]
            if metrics.is_usable and metrics.return_pct is not None:
                cells.append(
                    f'<td class="{_sign(metrics.return_pct)}">{metrics.return_pct:+.3f}</td>'
                )
            else:
                cells.append(f'<td class="na" title="{escape(metrics.reason or "")}">n/a</td>')
        rows.append(
            f'<tr><td class="mono">{escape(event.event_id)}</td>{"".join(cells)}</tr>'
        )

    return (
        "<h2>Evidence &mdash; every observation</h2>"
        '<div class="card"><table>'
        f'<thead><tr><th>event</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _footer(study: StudyResult) -> str:
    return (
        "<footer>"
        "<p><strong>Exploratory statistics.</strong> These are historical "
        "observations over a small sample, not a trading signal. No correction "
        "has been applied for multiple comparisons: testing several windows "
        "raises the chance that one looks significant purely by chance.</p>"
        "<p>Every number was computed by deterministic Python from exchange "
        "candles. Prices are anchored to the last candle that closed at or "
        "before each window boundary, so no measurement contains information "
        "from after the instant it claims to describe.</p>"
        f'<p>Reproduce with spec <span class="mono">{study.spec.fingerprint}</span>.</p>'
        "</footer>"
    )


def _sign(value: float) -> str:
    return "pos" if value >= 0 else "neg"


def _tooltip(event_id: str, metrics: object) -> str:
    return_pct = getattr(metrics, "return_pct", None)
    high = getattr(metrics, "high", None)
    low = getattr(metrics, "low", None)
    coverage = getattr(metrics, "coverage_pct", None)
    parts = [event_id]
    if return_pct is not None:
        parts.append(f"return {return_pct:+.3f}%")
    if high is not None and low is not None:
        parts.append(f"high {high:,.2f} / low {low:,.2f}")
    if coverage is not None:
        parts.append(f"coverage {coverage:.0f}%")
    return " · ".join(parts)


def write_report(
    study: StudyResult, path: str | Path, *, interpretation: Any | None = None
) -> str:
    """Write the report to ``path`` and return the absolute path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(study, interpretation), encoding="utf-8")
    return str(target.resolve())


__all__ = [
    "render_body",
    "render_interpretation_card",
    "render_report",
    "write_report",
]
