"""The historical-reaction section of an event page.

Deliberately *not* :func:`backtool.reporting.html.render_body`. That function
renders a research report -- assumptions first, every window at full size, every
observation in a table -- and it is right for a saved artefact a reader opened
on purpose. Dropped into a product page it reads as an internal tool: four
full-height charts before the reader reaches anything they asked for, and the
engine's own vocabulary on the surface.

Same numbers, same aggregation layer, different presentation:

* one window visible at a time, switched client-side because all four are
  already computed and a round trip to re-render them would be theatre;
* three statistics given weight and the rest given a line;
* the individual observations and the methodology kept, but collapsed.

Nothing here computes anything. Every figure comes from
:func:`~backtool.research.aggregate.aggregate_study`, which is the only place in
the project that produces a summary statistic.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from backtool.events.releases import scenario_methodology
from backtool.reporting.charts import BarItem, distribution_chart, diverging_bar_chart
from backtool.research.aggregate import WindowAggregate, aggregate_study
from backtool.research.regime import MATCHING_METHODOLOGY, SETUP_SYMBOL
from backtool.research.results import StudyResult
from backtool.research.windows import (
    WindowSpec,
    window_label,
    window_sentence,
    window_title,
)


def _default_window_index(windows: tuple[WindowSpec, ...]) -> int:
    """Which window opens first.

    A reader who clicked an event came for the reaction, not the run-up, so the
    first window starting at or after the announcement wins. Falling back to the
    last window rather than the first matters for a spec made entirely of
    pre-event windows, where the one nearest the event is the interesting one.
    """
    for index, window in enumerate(windows):
        if window.start_offset.total_seconds() >= 0:
            return index
    return len(windows) - 1


def render_history(study: StudyResult) -> str:
    """Render the whole historical block as slotted fragments.

    Returns three ``data-slot`` containers rather than one blob. The page
    interleaves other sections -- key findings, similar setups, the explainer --
    between them, and the client distributes these into the matching
    placeholders. One fragment would have forced the page into the order this
    function happened to emit.
    """
    aggregates = aggregate_study(study)
    windows = study.spec.windows
    selected = _default_window_index(windows)

    return (
        f'<div data-slot="summary">{_summary(study, aggregates, selected)}</div>'
        f'<div data-slot="evidence">{_evidence(study)}</div>'
        f'<div data-slot="methodology">{_methodology(study)}</div>'
    )


def _summary(
    study: StudyResult, aggregates: dict[str, WindowAggregate], selected: int
) -> str:
    spec = study.spec
    windows = spec.windows

    tabs = "".join(
        f'<button type="button" class="window-tab" role="tab" '
        f'data-window="{index}" aria-selected="{str(index == selected).lower()}">'
        f"{escape(window_title(window))}</button>"
        for index, window in enumerate(windows)
    )

    panels = "".join(
        _window_panel(study, window, aggregates[window.name], index, selected)
        for index, window in enumerate(windows)
    )

    return (
        '<div class="section-head">'
        "<h2>Historical reaction</h2>"
        f'<p class="section-sub">Last {len(study.events)} '
        f"{escape(spec.event_type.value)} releases &middot; "
        f'<span class="mono">{escape(spec.symbol)}</span> &middot; '
        f"{escape(spec.interval.value)} candles</p>"
        "</div>"
        f'<div class="window-tabs" role="tablist">{tabs}</div>'
        f"{panels}"
    )


def _window_panel(
    study: StudyResult,
    window: WindowSpec,
    agg: WindowAggregate,
    index: int,
    selected: int,
) -> str:
    hidden = "" if index == selected else " hidden"

    if agg.sample_size == 0:
        return (
            f'<div class="window-panel" data-window="{index}"{hidden}>'
            '<div class="panel-card"><p class="muted">No usable observations in '
            "this window. The releases were selected, but there were not enough "
            "candles around them to measure.</p></div></div>"
        )

    return (
        f'<div class="window-panel" data-window="{index}"{hidden}>'
        f'<p class="window-caption">Measured over {escape(window_sentence(window))}.</p>'
        f"{_hero_stats(agg)}"
        f"{_stat_strip(agg)}"
        f"{_notes(agg)}"
        f"{_charts(study, window, agg, index)}"
        "</div>"
    )


def _hero_stats(agg: WindowAggregate) -> str:
    """The three figures a reader should leave with.

    Median rather than mean leads, because one outsized release drags a mean
    somewhere no individual event went. The mean is still shown, a line below.
    """
    median, median_css = _signed_pct(agg.median_return_pct)
    tiles = [
        (
            "Median return",
            median,
            median_css,
            "Half the releases moved more than this, half less.",
        ),
        (
            "Positive events",
            f"{agg.up_rate_pct:.0f}%",
            "",
            f"{agg.up_count} of {agg.sample_size} releases closed higher over "
            "this window.",
        ),
        (
            "Average trading range",
            f"{agg.mean_range_pct:.2f}%",
            "",
            "Mean distance from the window's low to its high, whichever way "
            "price ended up going.",
        ),
    ]
    cells = "".join(
        f'<div class="hero-stat" title="{escape(hint)}">'
        f'<span class="hero-label">{escape(label)}</span>'
        f'<span class="hero-value {css}">{value}</span></div>'
        for label, value, css, hint in tiles
    )
    return f'<div class="hero-stats">{cells}</div>'


def _stat_strip(agg: WindowAggregate) -> str:
    """Everything else worth knowing, on one line rather than in six more cards."""
    variability = (
        f"{agg.stdev_return_pct:.2f}%" if agg.stdev_return_pct is not None else "&mdash;"
    )
    items = [
        ("Events analyzed", str(agg.sample_size), "Releases with usable candle data."),
        (
            "Average return",
            _signed_pct(agg.mean_return_pct)[0],
            "The arithmetic mean, which a single large move can dominate.",
        ),
        (
            "Largest gain",
            _signed_pct(agg.max_return_pct)[0],
            "The strongest single outcome in this window.",
        ),
        (
            "Largest loss",
            _signed_pct(agg.min_return_pct)[0],
            "The weakest single outcome in this window.",
        ),
        (
            "Return variability",
            variability,
            "Standard deviation of the returns: how far a typical release "
            "landed from the average.",
        ),
    ]
    cells = "".join(
        f'<div class="strip-item" title="{escape(hint)}">'
        f'<span class="strip-label">{escape(label)}</span>'
        f'<span class="strip-value">{value}</span></div>'
        for label, value, hint in items
    )
    return f'<div class="stat-strip">{cells}</div>'


def _notes(agg: WindowAggregate) -> str:
    """Sample-size and coverage warnings, when there is something to warn about.

    Coverage is not decoration here. An average presented over twenty releases
    that quietly rested on seventeen would be the one genuinely dishonest thing
    this page could do, so an exclusion is always stated.
    """
    notes: list[str] = []
    if agg.caveat:
        notes.append(f"<strong>Small sample.</strong> {escape(agg.caveat)}.")
    if agg.excluded:
        notes.append(
            f"<strong>{agg.excluded} excluded.</strong> Selected, but without "
            "enough candle data to measure this window. They are not in the "
            "figures above."
        )
    if not notes:
        return ""
    return "".join(f'<p class="note">{note}</p>' for note in notes)


def _charts(
    study: StudyResult, window: WindowSpec, agg: WindowAggregate, index: int
) -> str:
    """The distribution, with the per-release bars available behind a toggle.

    The histogram leads because it answers the shape questions -- spread, lean,
    outliers -- without the reader tallying twenty bars themselves. The bars
    stay one click away, because "which release was that?" is the immediate
    follow-up and the distribution cannot answer it.
    """
    usable = study.usable_events(window.name)
    returns = [
        metrics.return_pct
        for event in usable
        if (metrics := event.windows[window.name]).return_pct is not None
    ]
    bars = [
        BarItem(
            label=event.local_date.strftime("%d %b %Y"),
            value=event.windows[window.name].return_pct or 0.0,
            tooltip=_tooltip(event.local_date.strftime("%d %b %Y"), event.windows[window.name]),
        )
        for event in usable
    ]

    return (
        '<div class="panel-card chart-panel">'
        f'<div class="chart-switch" role="tablist" data-panel="{index}">'
        '<button type="button" role="tab" data-view="distribution" '
        'aria-selected="true">Distribution</button>'
        '<button type="button" role="tab" data-view="events" '
        'aria-selected="false">By release</button>'
        "</div>"
        f'<div class="chart-view" data-view="distribution">'
        f"{distribution_chart(returns, median=agg.median_return_pct)}"
        '<p class="chart-note">Each bar counts the releases whose return fell in '
        "that bucket. Blue is positive, red negative; the dashed rule is the "
        "median.</p>"
        "</div>"
        f'<div class="chart-view" data-view="events" hidden>'
        f"{diverging_bar_chart(bars)}"
        "</div>"
        "</div>"
    )


def _evidence(study: StudyResult) -> str:
    """Every observation, collapsed.

    Kept because it is what makes a conclusion checkable -- a reader who doubts
    a median can read the numbers it came from -- and collapsed because most
    readers do not doubt it, and twenty rows of six decimals is not what the
    page is for.
    """
    windows = study.spec.windows
    head = "".join(f"<th>{escape(window_label(window))}</th>" for window in windows)

    rows: list[str] = []
    for event in study.events:
        cells: list[str] = []
        for window in windows:
            metrics = event.windows[window.name]
            if metrics.is_usable and metrics.return_pct is not None:
                text, css = _signed_pct(metrics.return_pct)
                cells.append(f'<td class="{css}">{text}</td>')
            else:
                reason = metrics.reason or "Not enough candle data."
                cells.append(
                    f'<td class="na" title="{escape(reason)}">&mdash;</td>'
                )
        rows.append(
            f"<tr><td>{escape(event.local_date.strftime('%d %b %Y'))}</td>"
            f'{"".join(cells)}</tr>'
        )

    count = len(study.events)
    return (
        '<details class="disclosure">'
        "<summary><span class=\"disclosure-title\">Historical events</span>"
        f'<span class="disclosure-hint">{count} releases &middot; '
        "return over each window</span></summary>"
        '<div class="disclosure-body"><table>'
        f"<thead><tr><th>Release</th>{head}</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<p class="chart-note">A dash means that window could not be measured '
        "for that release. Hover it for the reason.</p>"
        "</div></details>"
    )


def _methodology(study: StudyResult) -> str:
    """Exact technical wording, preserved and out of the way.

    None of this is softened for the collapsed view. A reader who opens a
    section called methodology wants the anchor rule stated precisely, not
    paraphrased -- and the fingerprint is what makes a set of numbers
    reproducible, so it stays verbatim.
    """
    rows = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"
        for label, value in study.spec.describe_items()
    )
    # The setup-matching rule is stated here too. It decides which releases a
    # reader is shown under "Similar setups", so it belongs with the other
    # definitions that decide what the numbers are, not only in the section it
    # happens to drive.
    scenarios = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"
        for label, value in scenario_methodology()
    )
    matching = "".join(
        f"<dt>{escape(label)}</dt>"
        f"<dd>{escape(value.replace('{symbol}', SETUP_SYMBOL))}</dd>"
        for label, value in MATCHING_METHODOLOGY
    )

    return (
        '<details class="disclosure">'
        '<summary><span class="disclosure-title">Data &amp; methodology</span>'
        '<span class="disclosure-hint">Definitions, coverage and the exact '
        "specification</span></summary>"
        '<div class="disclosure-body">'
        f'<p class="note"><strong>Coverage.</strong> '
        f"{escape(study.coverage_note())}.</p>"
        f'<dl class="assumptions mono">{rows}</dl>'
        "<p><strong>Historical scenarios.</strong> How releases are sorted "
        "by what they said.</p>"
        f'<dl class="assumptions mono">{scenarios}</dl>'
        "<p><strong>Similar-setup matching.</strong> How past releases are "
        "selected as comparable to now.</p>"
        f'<dl class="assumptions mono">{matching}</dl>'
        "<p><strong>Exploratory statistics.</strong> These are historical "
        "observations over a small sample, not a trading signal. No correction "
        "has been applied for multiple comparisons: testing several windows "
        "raises the chance that one looks significant purely by chance.</p>"
        "<p>Every number was computed by deterministic Python from exchange "
        "candles. Prices are anchored to the last candle that closed at or "
        "before each window boundary, so no measurement contains information "
        "from after the instant it claims to describe.</p>"
        "</div></details>"
    )


def _tooltip(label: str, metrics: object) -> str:
    return_pct = getattr(metrics, "return_pct", None)
    high = getattr(metrics, "high", None)
    low = getattr(metrics, "low", None)
    parts = [label]
    if return_pct is not None:
        parts.append(f"return {_signed_pct(return_pct)[0]}")
    if high is not None and low is not None:
        parts.append(f"high {high:,.2f} / low {low:,.2f}")
    return " · ".join(parts)


def _signed_pct(value: float, decimals: int = 2) -> tuple[str, str]:
    """Format a percentage with its sign, plus the CSS class matching what shows.

    A mean of -0.004% formats as ``-0.00%``, which reads as a typo and gets
    coloured red for a move that rounded to nothing. Taking the sign from the
    *rounded* figure keeps the number, its sign and its colour telling the same
    story, rather than colouring by a precision the reader cannot see.
    """
    rounded = round(value, decimals)
    if rounded == 0:
        return f"{0.0:.{decimals}f}%", ""
    return f"{rounded:+.{decimals}f}%", "pos" if rounded > 0 else "neg"


HISTORY_SCRIPT = """
(function () {
  const root = document.getElementById('reaction-summary');
  if (!root) return;

  function select(group, attr, value) {
    for (const el of group) el.setAttribute('aria-selected', String(el.dataset[attr] === value));
  }

  root.addEventListener('click', (ev) => {
    const tab = ev.target.closest('.window-tab');
    if (tab) {
      const chosen = tab.dataset.window;
      select(root.querySelectorAll('.window-tab'), 'window', chosen);
      for (const panel of root.querySelectorAll('.window-panel')) {
        panel.hidden = panel.dataset.window !== chosen;
      }
      return;
    }

    const view = ev.target.closest('.chart-switch button');
    if (view) {
      const panel = view.closest('.chart-panel');
      const chosen = view.dataset.view;
      select(panel.querySelectorAll('.chart-switch button'), 'view', chosen);
      for (const pane of panel.querySelectorAll('.chart-view')) {
        pane.hidden = pane.dataset.view !== chosen;
      }
    }
  });
})();
"""
