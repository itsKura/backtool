"""Rendering for cohorts: the reaction grid, the setup readout, the ladder.

Shared by Similar Setups and Historical Scenarios, because they show the same
thing. A cohort is a named group of past releases; what a reader wants from one
is always the same block of numbers, and rendering it once means the two
features cannot end up describing their samples differently.

Two rules run through everything here:

* **The sample size is never separated from the statistics.** It sits in the
  section heading, again in the table caption, and again on every row, because
  a median over three releases and a median over thirty look identical.
* **Nothing is computed.** Every figure comes from
  :class:`~backtool.research.aggregate.WindowAggregate`.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from backtool.research.cohort import Cohort
from backtool.research.regime import DIMENSION_LABELS, MarketSetup
from backtool.research.setups import SetupMatch
from backtool.research.windows import WindowSpec, compact_duration


def render_cohort_table(cohort: Cohort, windows: tuple[WindowSpec, ...]) -> str:
    """The reaction grid: one row per horizon, the same columns every time.

    Rows are horizons rather than columns because a reader reads down them --
    a move that shows at 5m and survives to 24h is a different story from one
    that is gone by 1h, and that story is a vertical scan.
    """
    if cohort.sample_size == 0:
        return (
            '<p class="muted">No historical releases in this group, so there is '
            "nothing to measure.</p>"
        )

    rows = []
    for window in windows:
        agg = cohort.windows.get(window.name)
        if agg is None or agg.sample_size == 0:
            rows.append(
                f'<tr><td>{escape(compact_duration(window.end_offset))}</td>'
                f'<td class="na" colspan="5">not measurable</td></tr>'
            )
            continue

        median, median_css = _signed(agg.median_return_pct)
        n = agg.sample_size
        rows.append(
            f"<tr><td>{escape(compact_duration(window.end_offset))}</td>"
            f'<td class="{median_css}">{median}</td>'
            f"<td>{_rate(agg.up_count, n, agg.up_rate_pct)}</td>"
            f"<td>{agg.median_range_pct:.2f}%</td>"
            f"<td>{_rate(agg.move_over_0_5_count, n, agg.move_over_0_5_pct)}</td>"
            f"<td>{_rate(agg.move_over_1_count, n, agg.move_over_1_pct)}</td></tr>"
        )

    return (
        '<table class="cohort-table">'
        "<thead><tr><th>After</th><th>Median return</th><th>Positive</th>"
        "<th>Median range</th><th>Moved &gt;0.5%</th><th>Moved &gt;1%</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
        f"{_extremes(cohort, windows)}"
    )


def _extremes(cohort: Cohort, windows: tuple[WindowSpec, ...]) -> str:
    """Best and worst single outcome, at the longest horizon measured.

    One horizon rather than all five: the largest gain at 5m and the largest
    gain at 24h are usually different releases, and a grid of ten extremes
    invites reading them as one story.
    """
    for window in reversed(windows):
        agg = cohort.windows.get(window.name)
        if agg is not None and agg.sample_size:
            horizon = compact_duration(window.end_offset)
            gain, _ = _signed(agg.max_return_pct)
            loss, _ = _signed(agg.min_return_pct)
            return (
                f'<p class="cohort-extremes">Over {escape(horizon)}: '
                f'best <span class="pos">{gain}</span>, '
                f'worst <span class="neg">{loss}</span>, '
                f"measured over {agg.sample_size} release"
                f'{"" if agg.sample_size == 1 else "s"}.</p>'
            )
    return ""


#: Below this sample size a percentage is shown with the count that produced it.
#: "100%" over five releases and "100%" over fifty read identically and mean
#: entirely different things; the fraction is what separates them at a glance.
COUNT_ALONGSIDE_BELOW = 20


def _rate(count: int, total: int, pct: float) -> str:
    """A percentage, with its underlying fraction while the sample is small."""
    if total == 0:
        return "&mdash;"
    if total < COUNT_ALONGSIDE_BELOW:
        return f"{count}/{total} ({pct:.0f}%)"
    return f"{pct:.0f}%"


def render_sample_badge(cohort: Cohort) -> str:
    """The count, its quality band, and what that band means.

    The band is a word, never a score. A "68% confidence" next to these figures
    would be a statistic nobody computed -- no test has been run on any of them
    -- and it would be the most authoritative-looking number on the page.
    """
    quality = cohort.quality
    band = (
        f'<span class="cohort-quality quality-{quality.name.lower()}">'
        f"{escape(quality.value)} sample</span>"
        if quality.needs_warning
        else ""
    )
    caveat = (
        f'<span class="cohort-warn">{escape(cohort.caveat)}</span>'
        if cohort.caveat
        else ""
    )
    return (
        f'<span class="cohort-n">n={cohort.sample_size}</span>{band}'
        f'<span class="cohort-desc">{escape(cohort.description)}</span>{caveat}'
    )


def render_setup(setup: MarketSetup, *, title: str, note: str = "") -> str:
    """The four readings and the buckets they produced.

    Both halves are shown on every line. A reader given only "bearish" has to
    trust the threshold; given "-4.6% -> bearish" they can disagree with it.

    Args:
        note: Timing context. For an upcoming release this says how far off it
            is and that conditions may move before then, which is the
            difference between a reading and a forecast.
    """
    items = setup.describe_items()
    stamp = (
        f'<span class="setup-stamp">as of {setup.as_of:%d %b %Y, %H:%M} UTC</span>'
    )

    if not items:
        return (
            f'<div class="setup-card"><p class="setup-title">{escape(title)}'
            f"{stamp}</p>"
            '<p class="muted">Not enough price history before this point to read '
            "the market state.</p></div>"
        )

    lines = "".join(
        f'<div class="setup-row"><span class="setup-label">{escape(label)}</span>'
        f'<span class="setup-value">{escape(value)}</span></div>'
        for label, value in items
    )
    tail = f'<p class="setup-note">{escape(note)}</p>' if note else ""
    return (
        f'<div class="setup-card"><p class="setup-title">{escape(title)}{stamp}</p>'
        f"{lines}{tail}</div>"
    )


def render_ladder(match: SetupMatch) -> str:
    """Every rung of the relaxation, with the used one marked.

    Shown in full rather than only reporting the rung that was used. "Six
    matches" means something different when the exact match found five than
    when it found none, and only the ladder distinguishes them.
    """
    if not match.ladder:
        return ""

    rows = "".join(
        f'<tr class="{"rung-used" if rung.used else ""}">'
        f"<td>{escape(rung.describe())}</td>"
        f"<td>{rung.sample_size}</td>"
        f'<td>{"used" if rung.used else ""}</td></tr>'
        for rung in match.ladder
    )
    return (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">How the sample was chosen</span>'
        f'<span class="disclosure-hint">{match.exact_count} exact, '
        f"{match.sample_size} used</span></summary>"
        '<div class="disclosure-body">'
        "<p>Conditions are dropped one at a time, slowest-moving first, until "
        "enough releases match. Every rung is listed so the cost of the sample "
        "is visible, not just its size.</p>"
        '<table class="cohort-table"><thead><tr><th>Conditions</th>'
        "<th>Releases</th><th></th></tr></thead>"
        f'<tbody>{rows}</tbody></table>'
        "</div></details>"
    )


def render_matched_releases(match: SetupMatch) -> str:
    """Each matched release with its own readings, collapsed.

    This is what makes a match checkable. A reader who doubts that a release
    belonged in the sample can read the four figures that put it there.
    """
    if not match.matches:
        return ""

    rows = []
    for release in match.matches:
        buckets = release.setup.measured
        cells = "".join(
            f'<td class="{"bucket-on" if dim in match.matched_on else "bucket-off"}">'
            f'{escape(buckets.get(dim, "—"))}</td>'
            for dim in DIMENSION_LABELS
        )
        rows.append(
            f'<tr><td class="mono">{escape(release.event_id)}</td>{cells}</tr>'
        )

    heads = "".join(f"<th>{escape(name)}</th>" for name in DIMENSION_LABELS.values())
    return (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">The matched releases</span>'
        f'<span class="disclosure-hint">{len(match.matches)} shown with their '
        "own conditions</span></summary>"
        '<div class="disclosure-body">'
        "<p>Highlighted cells are the conditions this sample required. Dimmed "
        "cells were not required, and may differ from today.</p>"
        f'<table class="cohort-table"><thead><tr><th>Release</th>{heads}</tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table>'
        "</div></details>"
    )


def _signed(value: float) -> tuple[str, str]:
    """Format a percentage and the class matching what is displayed.

    Same rule as the history panel: a figure rounding to zero is shown without
    a sign and without a colour, because -0.00% asserts a direction the printed
    digits do not have.
    """
    rounded = round(value, 2)
    if rounded == 0:
        return "0.00%", ""
    return f"{rounded:+.2f}%", "pos" if rounded > 0 else "neg"


COHORT_STYLE = """
.cohort-table { width: 100%; border-collapse: collapse; font-size: 13px;
                margin: 10px 0 6px; }
.cohort-table th { text-align: right; color: var(--text-muted); font-weight: 500;
                   padding: 6px 10px; border-bottom: 1px solid var(--grid);
                   white-space: nowrap; }
.cohort-table td { text-align: right; padding: 6px 10px;
                   border-bottom: 1px solid var(--grid);
                   font-variant-numeric: tabular-nums; }
.cohort-table th:first-child, .cohort-table td:first-child { text-align: left; }
.cohort-table td.pos, .cohort-table span.pos { color: var(--up); }
.cohort-table td.neg, .cohort-table span.neg { color: var(--down); }
.cohort-table td.na { color: var(--text-muted); text-align: left; }
.cohort-table tr.rung-used td { color: var(--text-primary); font-weight: 500; }
.cohort-table td.bucket-off { color: var(--text-muted); opacity: 0.6; }
.cohort-table td.bucket-on { color: var(--text-primary); }
.cohort-extremes { font-size: 12px; color: var(--text-muted); margin: 6px 0 0; }

.cohort-n { font-size: 12px; font-weight: 600; color: var(--text-primary);
            background: var(--surface-2); border-radius: 5px; padding: 2px 7px;
            margin-right: 8px; font-variant-numeric: tabular-nums; }
.cohort-desc { font-size: 12px; color: var(--text-muted); }
.cohort-quality { font-size: 11px; padding: 2px 7px; border-radius: 99px;
                  margin-right: 8px; text-transform: uppercase;
                  letter-spacing: 0.04em; font-weight: 600; }
.quality-very_limited { background: #4a1f1f; color: #f3a9a8; }
.quality-limited { background: #43341a; color: #e9c274; }
.quality-moderate { background: #1e3350; color: #9cc0ea; }
:root[data-theme="light"] .quality-very_limited { background: #fde8e7; color: #a2302f; }
:root[data-theme="light"] .quality-limited { background: #fdf0d8; color: #8a5d00; }
:root[data-theme="light"] .quality-moderate { background: #e6edf6; color: #2a5490; }
.cohort-warn { display: block; font-size: 12px; color: var(--down); margin-top: 6px; }

.setup-card { border: 1px solid var(--grid); border-radius: 8px; padding: 12px 14px;
              margin-bottom: 12px; }
.setup-title { font-size: 12px; color: var(--text-muted); margin: 0 0 8px;
               display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;
               justify-content: space-between; }
.setup-stamp { font-size: 11px; color: var(--text-muted);
               font-family: ui-monospace, Consolas, monospace; }
.setup-note { font-size: 12px; color: var(--text-muted); margin: 10px 0 0;
              padding-top: 8px; border-top: 1px solid var(--grid); }
.setup-row { display: flex; justify-content: space-between; gap: 16px;
             padding: 3px 0; font-size: 13px; }
.setup-label { color: var(--text-secondary); }
.setup-value { color: var(--text-primary); font-variant-numeric: tabular-nums;
               text-align: right; }

.cohort-tabs { display: flex; flex-wrap: wrap; gap: 2px; padding: 2px;
               background: var(--surface-2); border-radius: 8px;
               margin: 0 0 14px; width: fit-content; max-width: 100%; }
.cohort-tab { padding: 6px 14px; border: none; border-radius: 6px;
              background: transparent; color: var(--text-secondary);
              font: inherit; font-size: 13px; cursor: pointer; white-space: nowrap; }
.cohort-tab:hover { color: var(--text-primary); }
.cohort-tab[aria-selected="true"] { background: var(--surface-0);
                                    color: var(--text-primary); font-weight: 500; }
.cohort-panel[hidden] { display: none; }
"""


__all__ = [
    "COHORT_STYLE",
    "render_cohort_table",
    "render_ladder",
    "render_matched_releases",
    "render_sample_badge",
    "render_setup",
]
