"""The event detail page.

Built around the question a reader actually arrives with, in the order they
arrive with it: what is this event, what does the market expect, what has
happened around it before, what stands out, and -- only if they go looking --
the individual observations and the exact method behind them.

That ordering is the whole design. An earlier version of this page led with the
research report: assumptions, four full-height charts, every observation. All of
it accurate, none of it what someone clicking "FOMC" wanted first. The numbers
are unchanged; what changed is which ones are given weight, and what vocabulary
they are given in.

The price chart is TradingView's Lightweight Charts, fed from **our own candle
cache**. That is the point: the embeddable TradingView widget renders
TradingView's aggregated feed, which can disagree with Binance candle by candle.
A reader seeing +2.59% in the table beside a differently-shaped candle would
have no way to tell which is right, and the whole premise here is that every
number is traceable. Charting the same data the statistics came from makes them
one thing rather than two.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

from backtool.core.time import NEW_YORK, utc_to_local
from backtool.core.types import Interval
from backtool.events.meta import meta_for
from backtool.events.models import MarketEvent
from backtool.research.windows import DEFAULT_WINDOWS, window_label, window_title
from backtool.web.calendar_page import format_delta

if TYPE_CHECKING:  # pandas is only needed for the annotation
    import pandas as pd

#: TradingView's open-source charting library. Pinned to a major version: v5
#: replaced `addCandlestickSeries` with `addSeries(CandlestickSeries, ...)`, so
#: an unpinned range would break the page on the next major release.
LIGHTWEIGHT_CHARTS_CDN = (
    "https://cdn.jsdelivr.net/npm/lightweight-charts@5/dist/"
    "lightweight-charts.standalone.production.js"
)

#: How far either side of the event the price chart reaches.
CHART_SPAN = dt.timedelta(hours=24)

#: Zoom presets offered above the chart, in hours either side of the release.
#:
#: The bands below are drawn from the measurement windows, and at the full
#: 48-hour view a one-hour window is about two percent of the width -- a sliver
#: too narrow to carry its own label. Rather than leave those two windows
#: effectively invisible, the reader can pull the view in to where they are
#: legible. Native scroll-zoom does the same thing; these just make it findable.
CHART_ZOOMS: tuple[int, ...] = (24, 4, 1)


def chart_window_bands(
    event_time: dt.datetime,
    candles: pd.DataFrame | None = None,
    interval: Interval = Interval.M5,
) -> list[dict[str, object]]:
    """The measurement windows as boxes: a time span and a price range.

    Derived from the same ``DEFAULT_WINDOWS`` the statistics are computed over,
    so a shaded region cannot describe a different window than the numbers do.
    Clipped to the charted span: a window reaching past the last candle would
    otherwise shade empty canvas.

    The high and low come from :func:`compute_window_metrics` -- the same
    function the research layer uses -- rather than a max/min taken in the
    browser. A window's high is a *statistic*, with a specific rule about which
    candles fall inside the window (close time after the start, at or before
    the end), and recomputing it in JavaScript would put a second definition of
    it in the codebase. The one that drew the box would then be the one nobody
    tested.

    Args:
        event_time: The announcement instant.
        candles: Normalised candle frame around the event. Without it the bands
            carry time spans only and the client draws them full height.
        interval: Candle interval, needed for the coverage figure.

    Returns:
        One entry per window: label, start and end as UNIX seconds, whether it
        sits before or after the release, and ``high``/``low`` when measurable.
    """
    low_bound = event_time - CHART_SPAN
    high_bound = event_time + CHART_SPAN
    bands: list[dict[str, object]] = []

    for window in DEFAULT_WINDOWS:
        start = max(event_time + window.start_offset, low_bound)
        end = min(event_time + window.end_offset, high_bound)
        if end <= start:
            continue

        band: dict[str, object] = {
            "label": window_label(window),
            "title": window_title(window),
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "phase": "before" if window.end_offset <= dt.timedelta(0) else "after",
            "high": None,
            "low": None,
        }

        if candles is not None:
            from backtool.research.metrics import compute_window_metrics

            metrics = compute_window_metrics(
                candles, window.resolve(event_time), interval
            )
            if metrics.is_usable:
                band["high"] = metrics.high
                band["low"] = metrics.low

        bands.append(band)
    return bands

#: Display names for the symbols offered. The exchange symbol stays visible in
#: technical contexts -- the methodology block, the section subtitle -- because
#: there it is the precise thing being described. On a selector it is noise.
SYMBOL_NAMES: dict[str, str] = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}

#: How many findings and caveats the summary shows before collapsing the rest.
#: The model is allowed to return more; a page that leads with nine bullets is
#: not a summary of anything.
VISIBLE_FINDINGS = 3
VISIBLE_CAVEATS = 1


def render_event_header(event: MarketEvent, *, now: dt.datetime) -> str:
    """Title, timing, impact, and one line on what the release is."""
    meta = meta_for(event.event_type)
    local = utc_to_local(event.timestamp_utc, NEW_YORK)
    delta = event.timestamp_utc - now
    upcoming = delta.total_seconds() > 0

    facts = [
        f"{local:%b} {local.day} &middot; {_clock(local)} ET",
        f'<span class="badge badge-{escape(meta.importance.value)}">'
        f"{escape(meta.importance.value)} impact</span>",
        escape(format_delta(delta, upcoming)),
    ]

    return (
        '<p class="breadcrumb"><a href="/">&larr; Calendar</a></p>'
        '<header class="event-header">'
        f"<h1>{escape(meta.label)}</h1>"
        f'<p class="event-facts">{" &middot; ".join(facts)}</p>'
        f'<p class="event-summary">{escape(meta.summary)}</p>'
        "</header>"
        f"{_schedule_note(event)}"
    )


def _clock(local: dt.datetime) -> str:
    """Twelve-hour time without a leading zero: ``2:00 PM``.

    ``%-I`` is not portable to Windows, so the hour is trimmed by hand rather
    than by a format code that works on one developer's machine.
    """
    return f"{local:%I:%M %p}".lstrip("0")


#: Hover text for each expectations cell. All three are unavailable today, but
#: for different reasons, and the difference is the honest part: two are a
#: wiring job against a public agency feed, one has no free source at all.
_EXPECTATION_CELLS: tuple[tuple[str, str], ...] = (
    ("Previous", "The figure published at the preceding release."),
    (
        "Expected",
        "Consensus forecasts are compiled by commercial survey providers and "
        "are not available under a free licence, so no substitute is shown in "
        "their place.",
    ),
    ("Actual", "The figure published at the release instant."),
)


def render_expectations() -> str:
    """Previous, expected and actual for this release.

    All three read as em dashes today, and stating that is better than omitting
    the row: a reader can see what the product tracks and that it has nothing
    yet, rather than wondering whether it was measured and found to be nothing.
    """
    rendered = "".join(
        f'<div class="expect-cell" title="{escape(hint)}">'
        f'<span class="expect-label">{escape(label)}</span>'
        '<span class="expect-value">&mdash;</span></div>'
        for label, hint in _EXPECTATION_CELLS
    )
    return f'<div class="expectations">{rendered}</div>'


def render_asset_selector(event: MarketEvent, symbols: tuple[str, ...], chosen: str) -> str:
    """Which market to measure. Display names on the control, tickers underneath."""
    tabs = "".join(
        f'<a class="asset-tab" href="/event/{escape(event.event_id)}?symbol={symbol}" '
        f'aria-selected="{str(symbol == chosen).lower()}">'
        f'{escape(SYMBOL_NAMES.get(symbol, symbol))}</a>'
        for symbol in symbols
    )
    return (
        '<div class="asset-row">'
        f'<div class="asset-tabs">{tabs}</div>'
        f'<span class="asset-ticker mono">{escape(chosen)}</span>'
        "</div>"
    )


def render_event_explainer(event: MarketEvent) -> str:
    """What the release is and why markets move on it.

    Near the foot of the page. A reader who clicked a CPI event already knows
    roughly what CPI is; they came for the price. This is background, so it
    reads as prose rather than as a specification.
    """
    meta = meta_for(event.event_type)
    watch = "".join(f"<li>{escape(item)}</li>" for item in meta.what_to_watch)

    return (
        '<section class="about">'
        "<h2>About this event</h2>"
        '<div class="panel-card prose">'
        "<h3>What it is</h3>"
        f"<p>{escape(meta.what_it_is)}</p>"
        "<h3>Why markets care</h3>"
        f"<p>{escape(meta.why_markets_care)}</p>"
        "<h3>What to watch</h3>"
        f"<ul>{watch}</ul>"
        f'<p class="muted">{escape(meta.source)} &middot; '
        f"{escape(meta.release_time)}</p>"
        "</div></section>"
    )


def render_price_chart(event: MarketEvent, symbol: str, *, available: bool) -> str:
    """The candle chart slot, or an explanation of why there is no chart.

    An event that has not happened has no candles around it. Saying so is better
    than an empty frame the reader has to interpret.
    """
    if not available:
        return (
            '<section class="price">'
            "<h2>Price</h2>"
            '<div class="panel-card"><p class="muted">This release has not '
            "happened yet, so there are no candles around it to chart. The "
            "historical record below covers previous occurrences.</p></div>"
            "</section>"
        )

    hours = int(CHART_SPAN.total_seconds() // 3600)
    zooms = "".join(
        f'<button type="button" data-hours="{value}" '
        f'aria-selected="{str(value == CHART_ZOOMS[0]).lower()}">&plusmn;{value}h</button>'
        for value in CHART_ZOOMS
    )
    return (
        '<section class="price">'
        '<div class="section-head">'
        "<h2>Price around the release</h2>"
        f'<p class="section-sub"><span class="mono">{escape(symbol)}</span> '
        f"&middot; {hours}h either side &middot; shaded bands are the windows "
        "the statistics below are measured over</p>"
        "</div>"
        '<div class="panel-card chart-card">'
        '<div class="chart-toolbar">'
        '<div class="band-legend">'
        '<span><i class="band-key band-key-before"></i>before the release</span>'
        '<span><i class="band-key band-key-after"></i>after the release</span>'
        "</div>"
        f'<div class="chart-range" role="group" aria-label="Zoom">{zooms}</div>'
        "</div>"
        '<div id="price-chart"><div id="price-bands" aria-hidden="true"></div></div>'
        '<p class="status" id="price-chart-status">'
        '<span class="spinner"></span>Loading candles&hellip;</p>'
        "</div></section>"
    )


def _schedule_note(event: MarketEvent) -> str:
    """Flag an event that did not follow the usual schedule.

    A rescheduled or emergency release is precisely the one a reader should not
    assume behaved like the others, so it is called out rather than buried in a
    notes column.
    """
    if event.is_scheduled and not event.notes:
        return ""

    label = "Scheduled" if event.is_scheduled else "Not a normal release"
    css = "note" if event.is_scheduled else "note note-warn"
    detail = event.notes or "Followed the usual schedule."
    return f'<p class="{css}"><strong>{escape(label)}.</strong> {escape(detail)}</p>'


def render_similar_setups() -> str:
    """The Similar Setups slot, filled asynchronously.

    Deliberately a slot rather than inline content: building it reads two years
    of hourly history to measure the market state before every candidate
    release, and that is not something the page should block on.
    """
    return (
        '<section class="setups">'
        '<div class="section-head">'
        "<h2>Similar setups</h2>"
        '<p class="section-sub">Past releases that arrived with the market in '
        "the same condition as today &mdash; matched on four measured "
        "conditions, not on a similarity score.</p>"
        "</div>"
        '<div id="slot-setups"><p class="status"><span class="spinner"></span>'
        "Reading the market state before each past release&hellip;</p></div>"
        "</section>"
    )


def render_scenarios_slot() -> str:
    """The Historical Scenarios slot, filled asynchronously."""
    return (
        '<section class="scenarios">'
        '<div class="section-head">'
        "<h2>Historical scenarios</h2>"
        '<p class="section-sub">How past releases behaved depending on what the '
        "release itself said, rather than on when it happened.</p>"
        "</div>"
        '<div id="slot-scenarios"><p class="status"><span class="spinner"></span>'
        "Splitting past releases by outcome&hellip;</p></div>"
        "</section>"
    )


def scenarios_load_script(event_id: str, symbol: str) -> str:
    """Fetch the Historical Scenarios fragment into its slot."""
    return f"""
(function () {{
  const slot = document.getElementById('slot-scenarios');
  if (!slot) return;
  fetch('/api/event/{event_id}/scenarios?symbol={symbol}')
    .then(r => r.text())
    .then(html => {{ slot.innerHTML = html; }})
    .catch(e => {{
      slot.innerHTML = '<div class="error"><h3>Could not load scenarios</h3><p>'
        + String(e) + '</p></div>';
    }});
}})();
"""


def setups_load_script(event_id: str, symbol: str) -> str:
    """Fetch the Similar Setups fragment into its slot."""
    return f"""
(function () {{
  const slot = document.getElementById('slot-setups');
  if (!slot) return;
  fetch('/api/event/{event_id}/setups?symbol={symbol}')
    .then(r => r.text())
    .then(html => {{ slot.innerHTML = html; }})
    .catch(e => {{
      slot.innerHTML = '<div class="error"><h3>Could not load setups</h3><p>'
        + String(e) + '</p></div>';
    }});
}})();
"""


def render_ai_summary(*, available: bool, has_history: bool) -> str:
    """The key-findings slot: a button, and somewhere to put the answer.

    On demand rather than automatic. Each summary is a paid API call taking
    tens of seconds, and a reader who came to look at the chart should not be
    billed for prose they did not ask for.

    The attribution line is not decoration. A reader has to be able to tell at a
    glance which parts of the page were computed and which were written, and the
    only reliable way to do that is to say so next to the writing.
    """
    if not has_history:
        return ""
    if not available:
        return (
            '<section class="findings">'
            "<h2>Key findings</h2>"
            '<div class="panel-card"><p class="muted">Set '
            "<code>ANTHROPIC_API_KEY</code> and restart to have the statistics "
            "above summarised in plain language. Everything else on this page "
            "works without it.</p></div></section>"
        )

    return (
        '<section class="findings">'
        '<div class="section-head">'
        "<h2>Key findings</h2>"
        '<p class="section-sub">A language model reads the statistics above and '
        "explains them. It computes nothing, and every figure it quotes appears "
        "in the numbers on this page.</p>"
        "</div>"
        '<div class="panel-card" id="explain-card">'
        '<button type="button" class="btn" id="explain-btn">Summarise these '
        "results</button>"
        '<span class="status" id="explain-status"></span>'
        '<div id="explain-out"></div>'
        "</div></section>"
    )


def render_key_findings(interpretation: Any) -> str:
    """Render the model's reading: a headline, a few findings, one caveat.

    Typed loosely for the same reason the reporting layer does it -- this module
    must not import :mod:`backtool.ai`, so the page renders with or without that
    layer installed.

    The model is asked for two to five findings and as many caveats as apply.
    Showing all of them produced a wall of text that buried the one sentence
    worth reading, so the rest moves behind a disclosure. Nothing is discarded:
    the full reading is one click away, and it is the same text.
    """
    findings = list(interpretation.findings)
    caveats = list(interpretation.caveats)
    followups = list(interpretation.suggested_followups)

    overflow = findings[VISIBLE_FINDINGS:] + caveats[VISIBLE_CAVEATS:]
    full = (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">View full analysis</span>'
        f'<span class="disclosure-hint">{len(overflow)} further point'
        f'{"" if len(overflow) == 1 else "s"}</span></summary>'
        f'<div class="disclosure-body">{_bullets(overflow)}</div></details>'
        if overflow
        else ""
    )

    explore = (
        f'<div class="explore"><h3>Explore further</h3>{_bullets(followups)}</div>'
        if followups
        else ""
    )

    return (
        f'<p class="finding-headline">{escape(interpretation.headline)}</p>'
        f"{_bullets(findings[:VISIBLE_FINDINGS])}"
        f"{_keep_in_mind(caveats[:VISIBLE_CAVEATS])}"
        f"{full}{explore}"
        '<p class="attribution">Written by a language model from the statistics '
        "on this page. It computed none of them.</p>"
    )


def _bullets(items: list[str]) -> str:
    if not items:
        return ""
    entries = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f'<ul class="findings-list">{entries}</ul>'


def _keep_in_mind(caveats: list[str]) -> str:
    if not caveats:
        return ""
    body = " ".join(escape(caveat) for caveat in caveats)
    return f'<p class="note"><strong>Keep in mind.</strong> {body}</p>'


def ai_summary_script(event_id: str, symbol: str) -> str:
    """Client code for the on-demand summary button."""
    return f"""
(function () {{
  const btn = document.getElementById('explain-btn');
  if (!btn) return;
  const status = document.getElementById('explain-status');
  const out = document.getElementById('explain-out');

  btn.addEventListener('click', async () => {{
    btn.disabled = true;
    status.innerHTML = '<span class="spinner"></span>Reading the statistics…';
    out.innerHTML = '';
    try {{
      const r = await fetch('/api/event/{event_id}/explain?symbol={symbol}');
      out.innerHTML = await r.text();
      if (r.ok) btn.remove();
    }} catch (e) {{
      out.innerHTML = '<div class="error"><h3>Could not reach the server</h3><p>'
        + String(e) + '</p></div>';
    }} finally {{
      btn.disabled = false;
      status.textContent = '';
    }}
  }});
}})();
"""


def history_load_script(event_id: str, symbol: str) -> str:
    """Fetch the history fragment and distribute it into its placeholders.

    The response carries several ``data-slot`` blocks rather than one, because
    other sections sit between them on the page. An error response has no slots
    at all, so it falls through to the summary placeholder and is shown there
    instead of vanishing.
    """
    return f"""
(function () {{
  const fallback = document.getElementById('slot-summary');
  fetch('/api/event/{event_id}/history?symbol={symbol}')
    .then(r => r.text())
    .then(html => {{
      const holder = document.createElement('div');
      holder.innerHTML = html;
      const slots = holder.querySelectorAll('[data-slot]');
      if (!slots.length) {{ fallback.innerHTML = html; return; }}
      for (const slot of slots) {{
        const target = document.getElementById('slot-' + slot.dataset.slot);
        if (target) target.innerHTML = slot.innerHTML;
      }}
    }})
    .catch(e => {{
      fallback.innerHTML = '<div class="error"><h3>Could not load history</h3><p>'
        + String(e) + '</p></div>';
    }});
}})();
"""


def render_upcoming_notice(event: MarketEvent) -> str:
    """Make clear that a future event has no outcome yet.

    The history below it is real; the event itself has not happened. Without
    this the two read as one thing.
    """
    return (
        '<p class="note"><strong>This release has not happened yet.</strong> '
        "Everything below describes how "
        f"{escape(meta_for(event.event_type).label)} releases behaved in the "
        "past. It is a record, not a forecast.</p>"
    )


def price_chart_script(event_id: str, symbol: str) -> str:
    """Client code that fetches our candles and renders them.

    Colours are read from the page's CSS custom properties rather than
    hardcoded, so the chart follows the theme instead of being a second source
    of truth for it.
    """
    default_zoom = CHART_ZOOMS[0]
    return f"""
(function () {{
  const el = document.getElementById('price-chart');
  const status = document.getElementById('price-chart-status');
  if (!el || typeof LightweightCharts === 'undefined') {{
    if (status) status.textContent = 'Chart library unavailable.';
    return;
  }}

  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => (css.getPropertyValue(name) || fallback).trim();
  let loaded = false;

  const chart = LightweightCharts.createChart(el, {{
    height: 340,
    layout: {{
      background: {{ color: 'transparent' }},
      textColor: v('--text-muted', '#8f8e85'),
      fontSize: 11,
    }},
    grid: {{
      vertLines: {{ color: v('--grid', '#343431') }},
      horzLines: {{ color: v('--grid', '#343431') }},
    }},
    rightPriceScale: {{ borderColor: v('--grid', '#343431') }},
    // minBarSpacing is lowered because the default floor (0.5px) refuses to
    // shrink 579 five-minute candles into a narrow pane: the library clamps,
    // the series overflows the width, and the view silently shows only its
    // tail -- which reads as "the chart is wrong" rather than "too narrow".
    timeScale: {{
      borderColor: v('--grid', '#343431'),
      timeVisible: true,
      minBarSpacing: 0.05,
    }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});

  const up = v('--up', '#3987e5');
  const down = v('--down', '#e66767');
  const series = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor: up, borderUpColor: up, wickUpColor: up,
    downColor: down, borderDownColor: down, wickDownColor: down,
  }});

  const bandLayer = document.getElementById('price-bands');
  const timeScale = chart.timeScale();
  let allBands = [];
  let bands = [];
  let allCandles = [];
  let candleTimes = [];
  let eventTime = null;

  fetch('/api/event/{event_id}/candles?symbol={symbol}')
    .then(r => r.json())
    .then(data => {{
      if (!data.candles || !data.candles.length) {{
        status.textContent = data.message || 'No candles available for this window.';
        return;
      }}
      series.setData(data.candles);
      LightweightCharts.createSeriesMarkers(series, [{{
        time: data.event_time,
        position: 'aboveBar',
        color: v('--text-primary', '#ffffff'),
        shape: 'arrowDown',
        text: 'release',
      }}]);

      eventTime = data.event_time;
      allCandles = data.candles;
      // Widest first, so a nested shorter window paints on top of the one that
      // contains it and reads as the stronger of the two.
      allBands = (data.windows || []).slice().sort(
        (a, b) => (b.end - b.start) - (a.end - a.start)
      );

      loaded = true;
      setZoom({default_zoom});
      status.remove();
    }})
    .catch(e => {{ status.textContent = 'Could not load candles: ' + e; }});

  // A band boundary is snapped to a real candle before it is asked for a
  // coordinate. timeToCoordinate answers for points on the data grid; an
  // instant between two candles can come back null, which would silently drop
  // the band rather than misplace it -- harder to notice than a wrong pixel.
  function nearestIndex(t) {{
    if (!candleTimes.length) return null;
    let best = 0;
    let bestGap = Math.abs(t - candleTimes[0]);
    for (let i = 1; i < candleTimes.length; i++) {{
      const gap = Math.abs(t - candleTimes[i]);
      if (gap < bestGap) {{ best = i; bestGap = gap; }}
    }}
    return best;
  }}

  function nearestCandle(t) {{
    const index = nearestIndex(t);
    return index === null ? null : candleTimes[index];
  }}

  function buildBands() {{
    bandLayer.innerHTML = '';
    // Rank within each phase: the longest window of a phase is the outer wash,
    // anything shorter sits inside it and is drawn stronger.
    const seen = {{}};
    for (const band of bands) {{
      const depth = seen[band.phase] = (seen[band.phase] || 0) + 1;
      const node = document.createElement('div');
      node.className = 'band band-' + band.phase +
        (depth === 1 ? ' band-outer' : ' band-inner');
      node.innerHTML = '<span class="band-label"></span>';
      node.querySelector('.band-label').textContent = band.title;
      band.node = node;
      bandLayer.appendChild(node);
    }}
  }}

  function drawBands() {{
    if (!bands.length) return;
    const paneWidth = timeScale.width ? timeScale.width() : el.clientWidth;
    const axisHeight = timeScale.height ? timeScale.height() : 28;
    bandLayer.style.width = paneWidth + 'px';
    bandLayer.style.height = Math.max(0, el.clientHeight - axisHeight) + 'px';

    for (const band of bands) {{
      const node = band.node;
      const from = nearestCandle(band.start);
      const to = nearestCandle(band.end);
      if (from === null || to === null) {{ node.style.display = 'none'; continue; }}
      const a = timeScale.timeToCoordinate(from);
      const b = timeScale.timeToCoordinate(to);
      if (a === null || b === null) {{ node.style.display = 'none'; continue; }}

      const left = Math.max(0, Math.min(a, b));
      const right = Math.min(paneWidth, Math.max(a, b));
      const width = right - left;
      if (width <= 1) {{ node.style.display = 'none'; continue; }}

      node.style.display = '';
      node.style.left = left + 'px';
      node.style.width = width + 'px';

      // Vertical extent is the window's own high and low, so the box frames
      // the price the window actually covered instead of the whole canvas.
      // Both figures were computed server-side by the research layer.
      const layerHeight = bandLayer.clientHeight;
      let top = 0;
      let height = layerHeight;
      if (band.high !== null && band.low !== null) {{
        const yHigh = series.priceToCoordinate(band.high);
        const yLow = series.priceToCoordinate(band.low);
        if (yHigh !== null && yLow !== null) {{
          top = Math.min(yHigh, yLow);
          height = Math.abs(yLow - yHigh);
        }}
      }}
      // Keep the box on the canvas when autoscale puts an extreme off-screen,
      // and keep it thick enough to see when a quiet window is a flat line.
      const clampedTop = Math.max(0, Math.min(top, layerHeight - 2));
      node.style.top = clampedTop + 'px';
      node.style.height = Math.max(2, Math.min(height, layerHeight - clampedTop)) + 'px';

      node.classList.toggle('band-label-inside', clampedTop < 20);
      // A label wider than its band would overflow into the neighbouring one
      // and read as belonging to it, so it is dropped instead of clipped.
      node.classList.toggle('band-narrow', width < band.title.length * 7 + 18);
    }}
  }}

  // Zoom by changing what the series holds, not by moving the viewport.
  // fitContent, setVisibleRange and setVisibleLogicalRange are all accepted
  // and then ignored by this build of the library -- the range stays pinned
  // wherever it started. Narrowing the data is the one lever that does move
  // the view, and it rescales the price axis to the window being examined,
  // which is what a reader zooming in wants anyway.
  //
  // The opening view calls this too, so "what the chart shows on load" and
  // "what the 24h button does" are the same code path and cannot drift apart.
  function setZoom(hours) {{
    if (eventTime === null) return;
    const low = eventTime - hours * 3600;
    const high = eventTime + hours * 3600;

    const visible = allCandles.filter(c => c.time >= low && c.time <= high);
    if (!visible.length) return;
    series.setData(visible);
    candleTimes = visible.map(c => c.time);

    // A window only partly inside the view would be drawn as though it ended
    // at the edge of the canvas, so a band labelled "24 hours before" would
    // cover one hour. Drop those rather than mislabel them.
    bands = allBands.filter(b => b.start >= low && b.end <= high);
    buildBands();

    // Narrowing the data alone does not move the viewport -- the range stays
    // wherever it was pinned, which for a smaller set is off the end of it.
    // fit() is what binds the view to the data now present, and it applies the
    // container width first: asking for a coordinate from a chart that has not
    // been told how wide it is returns null for every one of them.
    fit();
  }}

  // fitContent() is accepted and ignored by this build of the library: the
  // chart stays at its default six-pixel bar spacing, showing the last twelve
  // hours of a forty-eight hour series. Sizing the bars to the pane by hand
  // gets there through applyOptions, which does work, and scrollToPosition(0)
  // pins the last bar to the right edge so the series exactly fills the width.
  function fitAll() {{
    // Both, deliberately. fitContent is the library's own answer and is what
    // works in a normal browser; sizing the bars by hand is the fallback for
    // builds where it is accepted and ignored. When both work they agree --
    // barSpacing = paneWidth / bars is the definition of fitting -- so running
    // the second after the first costs nothing and removes a dependency on
    // which of them happens to be functional.
    try {{ timeScale.fitContent(); }} catch (e) {{ /* fall through */ }}

    const count = candleTimes.length;
    const paneWidth = timeScale.width ? timeScale.width() : el.clientWidth;
    if (count < 2 || paneWidth < 2) return;
    try {{
      timeScale.applyOptions({{ barSpacing: paneWidth / count, rightOffset: 0 }});
      timeScale.scrollToPosition(0, false);
    }} catch (e) {{ /* fitContent above may already have done it */ }}
  }}

  // Width and fit must be applied together. Fitting against a stale width
  // leaves the candles bunched to one side of the canvas, which is what
  // happened when the observer resized the chart after fitting had run.
  function fit() {{
    chart.applyOptions({{ width: el.clientWidth }});
    if (loaded) fitAll();
    drawBands();
    // A chart laid out at zero width -- a collapsed container, a tab that was
    // never shown -- yields no coordinates, and every band hides itself. The
    // observer re-fires when width appears, and this second pass catches the
    // case where the width lands in the same frame the chart was sized in.
    requestAnimationFrame(drawBands);
  }}

  timeScale.subscribeVisibleLogicalRangeChange(drawBands);

  for (const btn of document.querySelectorAll('.chart-range button')) {{
    btn.addEventListener('click', () => {{
      for (const other of document.querySelectorAll('.chart-range button')) {{
        other.setAttribute('aria-selected', String(other === btn));
      }}
      setZoom(Number(btn.dataset.hours));
    }});
  }}

  new ResizeObserver(fit).observe(el);
  fit();
}})();
"""


EVENT_STYLE = """
.breadcrumb { margin: 0 0 18px; font-size: 13px; }
.breadcrumb a { color: var(--text-muted); text-decoration: none; }
.breadcrumb a:hover { color: var(--text-primary); }

.event-header { margin: 0 0 20px; }
.event-header h1 { font-size: 28px; margin: 0 0 8px; letter-spacing: -0.02em; }
.event-facts { margin: 0 0 12px; font-size: 13px; color: var(--text-secondary);
               display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.event-summary { margin: 0; color: var(--text-secondary); font-size: 15px;
                 max-width: 68ch; line-height: 1.6; }

.expectations { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                border: 1px solid var(--grid); border-radius: 8px;
                overflow: hidden; margin: 0 0 28px; }
.expect-cell { padding: 12px 16px; border-right: 1px solid var(--grid); }
.expect-cell:last-child { border-right: none; }
.expect-label { display: block; font-size: 11px; color: var(--text-muted);
                letter-spacing: 0.04em; text-transform: uppercase; }
.expect-value { display: block; font-size: 20px; font-weight: 600; margin-top: 3px;
                font-variant-numeric: tabular-nums; }

.asset-row { display: flex; align-items: center; gap: 12px; margin: 0 0 8px; }
.asset-tabs { display: inline-flex; gap: 2px; padding: 2px;
              background: var(--surface-2); border-radius: 8px; }
.asset-tab { padding: 5px 16px; border-radius: 6px; font-size: 13px;
             font-weight: 500; color: var(--text-secondary); text-decoration: none; }
.asset-tab:hover { color: var(--text-primary); }
.asset-tab[aria-selected="true"] { background: var(--accent); color: #fff; }
.asset-ticker { color: var(--text-muted); font-size: 12px; }

.section-head { margin: 36px 0 14px; }
.section-head h2 { margin: 0 0 3px; }
.section-sub { margin: 0; color: var(--text-muted); font-size: 13px; max-width: 70ch; }

.panel-card { background: var(--surface-1); border: 1px solid var(--grid);
              border-radius: 10px; padding: 18px 20px; }
.chart-card { padding: 14px; }
#price-chart { width: 100%; position: relative; }

.chart-toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
                 justify-content: space-between; margin: 0 0 10px; }
.band-legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px;
               color: var(--text-muted); }
.band-legend span { display: inline-flex; align-items: center; gap: 6px; }
.band-key { width: 18px; height: 10px; border-radius: 2px; display: inline-block; }
.band-key-before { background: rgba(var(--band-before), 0.20);
                   border: 1px solid rgba(var(--band-before), 0.65); }
.band-key-after { background: rgba(var(--band-after), 0.20);
                  border: 1px solid rgba(var(--band-after), 0.65); }
.chart-range { display: inline-flex; gap: 2px; padding: 2px;
               background: var(--surface-2); border-radius: 7px; }
.chart-range button { padding: 4px 11px; border: none; border-radius: 5px;
                      background: transparent; color: var(--text-muted);
                      font: inherit; font-size: 12px; cursor: pointer; }
.chart-range button[aria-selected="true"] { background: var(--surface-0);
                                            color: var(--text-primary); }

/* The band layer sits over the candles but must never swallow a crosshair or
   a drag, so the whole overlay is transparent to the pointer. */
#price-bands { position: absolute; inset: 0 auto 0 0; overflow: hidden;
               pointer-events: none; z-index: 1; }
/* Top and height are set from the window's own high and low, so the box
   frames the price the window covered rather than the whole canvas. */
.band { position: absolute; border-radius: 2px; }
.band-before { background: rgba(var(--band-before), 0.13);
               border: 1px solid rgba(var(--band-before), 0.40); }
.band-after { background: rgba(var(--band-after), 0.13);
              border: 1px solid rgba(var(--band-after), 0.40); }
.band-inner.band-before { background: rgba(var(--band-before), 0.22);
                          border-color: rgba(var(--band-before), 0.70); }
.band-inner.band-after { background: rgba(var(--band-after), 0.22);
                         border-color: rgba(var(--band-after), 0.70); }
.band-label { position: absolute; bottom: 100%; margin-bottom: 3px; left: 50%;
              transform: translateX(-50%); font-size: 11px; white-space: nowrap;
              letter-spacing: 0.01em; padding: 1px 6px; border-radius: 4px;
              background: var(--surface-0); opacity: 0.94; }
/* A box hugging the top of the canvas has no room above it for its label. */
.band-label-inside .band-label { bottom: auto; top: 3px; }
.band-before .band-label { color: var(--band-before-text); }
.band-after .band-label { color: var(--band-after-text); }
.band-narrow .band-label { display: none; }

.window-tabs { display: flex; flex-wrap: wrap; gap: 2px; padding: 2px;
               background: var(--surface-2); border-radius: 8px;
               margin: 0 0 18px; width: fit-content; max-width: 100%; }
.window-tab { padding: 6px 14px; border: none; border-radius: 6px;
              background: transparent; color: var(--text-secondary);
              font: inherit; font-size: 13px; cursor: pointer; white-space: nowrap; }
.window-tab:hover { color: var(--text-primary); }
.window-tab[aria-selected="true"] { background: var(--surface-0);
                                    color: var(--text-primary); font-weight: 500; }
.window-panel[hidden] { display: none; }
.window-caption { margin: 0 0 14px; font-size: 13px; color: var(--text-muted); }

.hero-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
              border: 1px solid var(--grid); border-radius: 10px;
              overflow: hidden; margin-bottom: 10px; }
.hero-stat { padding: 16px 20px; border-right: 1px solid var(--grid); }
.hero-stat:last-child { border-right: none; }
.hero-label { display: block; font-size: 12px; color: var(--text-muted); }
.hero-value { display: block; font-size: 30px; font-weight: 600; margin-top: 4px;
              letter-spacing: -0.025em; font-variant-numeric: tabular-nums; }
.hero-value.pos { color: var(--up); }
.hero-value.neg { color: var(--down); }

.stat-strip { display: flex; flex-wrap: wrap; gap: 0 28px; padding: 12px 20px;
              border: 1px solid var(--grid); border-radius: 10px;
              margin-bottom: 14px; }
.strip-item { display: flex; flex-direction: column; gap: 1px; }
.strip-label { font-size: 11px; color: var(--text-muted); }
.strip-value { font-size: 14px; font-weight: 500;
               font-variant-numeric: tabular-nums; }

.note { background: var(--surface-2); border-radius: 8px; padding: 10px 14px;
        font-size: 13px; color: var(--text-secondary); margin: 0 0 12px; }
.note-warn { border-left: 3px solid var(--down); border-radius: 0 8px 8px 0; }
.muted { color: var(--text-muted); font-size: 13px; }

.chart-panel { padding: 14px 16px 10px; }
.chart-switch { display: inline-flex; gap: 2px; padding: 2px;
                background: var(--surface-2); border-radius: 7px; margin-bottom: 6px; }
.chart-switch button { padding: 4px 12px; border: none; border-radius: 5px;
                       background: transparent; color: var(--text-muted);
                       font: inherit; font-size: 12px; cursor: pointer; }
.chart-switch button[aria-selected="true"] { background: var(--surface-0);
                                             color: var(--text-primary); }
.chart-view[hidden] { display: none; }
.chart-note { color: var(--text-muted); font-size: 12px; margin: 4px 0 0;
              max-width: 70ch; }

.disclosure { border: 1px solid var(--grid); border-radius: 10px;
              margin: 14px 0; background: var(--surface-1); }
.disclosure.inline { background: var(--surface-2); border: none; margin: 14px 0; }
.disclosure > summary { cursor: pointer; padding: 14px 18px; display: flex;
                        flex-wrap: wrap; align-items: baseline; gap: 10px;
                        list-style: none; }
.disclosure > summary::-webkit-details-marker { display: none; }
.disclosure > summary::before { content: "\\203A"; color: var(--text-muted);
                                font-size: 16px; line-height: 1;
                                transition: transform 0.15s ease; }
.disclosure[open] > summary::before { transform: rotate(90deg); }
@media (prefers-reduced-motion: reduce) {
  .disclosure > summary::before { transition: none; }
}
.disclosure-title { font-size: 14px; font-weight: 500; }
.disclosure-hint { font-size: 12px; color: var(--text-muted); }
.disclosure-body { padding: 0 18px 18px; }
.disclosure-body p { font-size: 13px; color: var(--text-secondary); }
.disclosure-body .assumptions { font-size: 12.5px; margin: 14px 0; }

.finding-headline { font-size: 17px; font-weight: 600; margin: 0 0 14px;
                    letter-spacing: -0.01em; line-height: 1.45; }
.findings-list { margin: 0 0 12px; padding-left: 18px; }
.findings-list li { margin-bottom: 6px; color: var(--text-secondary);
                    font-size: 14px; }
.explore { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--grid); }
.explore h3 { font-size: 13px; margin: 0 0 8px; color: var(--text-muted);
              font-weight: 500; }
.attribution { color: var(--text-muted); font-size: 12px; margin: 16px 0 0;
               padding-top: 12px; border-top: 1px solid var(--grid); }

.empty-state p { font-size: 13px; color: var(--text-secondary); margin: 0 0 8px; }
.empty-state p:last-child { margin-bottom: 0; }

.prose h3 { font-size: 13px; color: var(--text-muted); font-weight: 500;
            margin: 18px 0 6px; }
.prose h3:first-child { margin-top: 0; }
.prose p { margin: 0 0 4px; color: var(--text-secondary); max-width: 72ch;
           line-height: 1.65; }
.prose ul { margin: 0; padding-left: 18px; }
.prose li { margin-bottom: 5px; color: var(--text-secondary); }
.prose .muted { margin-top: 18px; padding-top: 12px;
                border-top: 1px solid var(--grid); }

.btn { background: var(--accent); color: #fff; border: none; border-radius: 7px;
       padding: 9px 18px; font: inherit; font-size: 14px; font-weight: 500;
       cursor: pointer; }
.btn:hover { filter: brightness(1.08); }
.btn:disabled { opacity: 0.55; cursor: progress; }

@media (max-width: 680px) {
  .hero-stats, .expectations { grid-template-columns: 1fr; }
  .hero-stat, .expect-cell { border-right: none;
                             border-bottom: 1px solid var(--grid); }
  .hero-stat:last-child, .expect-cell:last-child { border-bottom: none; }
  .stat-strip { gap: 12px 24px; }
  .window-tabs { width: 100%; }
  .window-tab { flex: 1 1 auto; text-align: center; }
  .event-header h1 { font-size: 23px; }
}
"""
