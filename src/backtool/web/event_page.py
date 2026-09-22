"""The event detail page.

Order follows what a reader actually wants when they click a past event: show
me the price first, then how it compared to previous occurrences, then -- if I
still care -- what this release is and why it matters. The explainer sits at the
bottom because someone checking a CPI print does not need CPI explained to them
before they can see the chart.

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
from xml.sax.saxutils import escape

from backtool.core.time import NEW_YORK, utc_to_local
from backtool.events.meta import meta_for
from backtool.events.models import MarketEvent
from backtool.web.calendar_page import format_delta

#: TradingView's open-source charting library. Pinned to a major version: v5
#: replaced `addCandlestickSeries` with `addSeries(CandlestickSeries, ...)`, so
#: an unpinned range would break the page on the next major release.
LIGHTWEIGHT_CHARTS_CDN = (
    "https://cdn.jsdelivr.net/npm/lightweight-charts@5/dist/"
    "lightweight-charts.standalone.production.js"
)

#: How far either side of the event the price chart reaches.
CHART_SPAN = dt.timedelta(hours=24)


def render_event_header(event: MarketEvent, *, now: dt.datetime) -> str:
    """Title, timing and any schedule irregularity. Nothing else."""
    meta = meta_for(event.event_type)
    local = utc_to_local(event.timestamp_utc, NEW_YORK)
    delta = event.timestamp_utc - now
    upcoming = delta.total_seconds() > 0

    return (
        '<p class="breadcrumb"><a href="/">&larr; Calendar</a></p>'
        f"<h1>{escape(meta.label)}</h1>"
        f'<p class="sub">'
        f"{local:%A %d %B %Y}, {local:%H:%M} ET &middot; "
        f"{event.timestamp_utc:%H:%M} UTC &middot; "
        f'<span class="badge badge-{escape(meta.importance.value)}">'
        f"{escape(meta.importance.value)}</span> &middot; "
        f"{escape(format_delta(delta, upcoming))}"
        f"</p>"
        f"{_schedule_note(event)}"
    )


def render_event_explainer(event: MarketEvent) -> str:
    """What the release is and why markets move on it.

    Rendered at the foot of the page. A reader who clicked a CPI event already
    knows roughly what CPI is; they came for the price. This is reference
    material, not the lead.
    """
    meta = meta_for(event.event_type)
    watch = "".join(f"<li>{escape(item)}</li>" for item in meta.what_to_watch)

    return (
        "<h2>About this event</h2>"
        '<div class="card">'
        f'<p class="ai-label">What it is</p><p>{escape(meta.what_it_is)}</p>'
        f'<p class="ai-label">Why markets care</p><p>{escape(meta.why_markets_care)}</p>'
        f'<p class="ai-label">What to watch</p><ul>{watch}</ul>'
        f'<p class="hint">{escape(meta.source)} &middot; {escape(meta.release_time)}</p>'
        "</div>"
    )


def render_price_chart(event: MarketEvent, symbol: str, *, available: bool) -> str:
    """The candle chart slot, or an explanation of why there is no chart.

    An event that has not happened has no candles around it. Saying so is better
    than an empty frame the reader has to interpret.
    """
    if not available:
        return (
            "<h2>Price</h2>"
            '<div class="card"><p class="hint">This event has not happened yet, '
            "so there are no candles around it to chart. The historical record "
            "below covers previous occurrences.</p></div>"
        )

    hours = int(CHART_SPAN.total_seconds() // 3600)
    return (
        "<h2>Price around the event</h2>"
        f'<p class="chart-caption">{escape(symbol)} &middot; '
        f"{hours}h either side of the release &middot; "
        "the marker is the announcement instant</p>"
        '<div class="card chart-card">'
        '<div id="price-chart"></div>'
        '<p class="status" id="price-chart-status">'
        '<span class="spinner"></span>Loading candles&hellip;</p>'
        "</div>"
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
    css = "notice" if event.is_scheduled else "caveat"
    detail = event.notes or "Followed the usual schedule."
    return f'<div class="{css}"><strong>{escape(label)}.</strong> {escape(detail)}</div>'


def render_ai_summary(*, available: bool, has_history: bool) -> str:
    """The AI summary slot: a button, and somewhere to put the answer.

    On demand rather than automatic. Each summary is a paid API call taking
    tens of seconds, and a reader who came to look at the chart should not be
    billed for prose they did not ask for.

    The badge is not decoration. A reader has to be able to tell at a glance
    which parts of the page were computed and which were written, and the only
    reliable way to do that is to say so next to the writing.
    """
    if not has_history:
        return ""
    if not available:
        return (
            "<h2>AI summary</h2>"
            '<div class="card"><p class="hint">Set <code>ANTHROPIC_API_KEY</code> '
            "in <code>.env</code> and restart to have the statistics above "
            "summarised in plain language. Everything else on this page works "
            "without it.</p></div>"
        )

    return (
        "<h2>AI summary</h2>"
        '<div class="card ai">'
        '<p class="ai-badge">A language model reads the statistics above and '
        "explains them. It computes nothing, and every figure it quotes appears "
        "in the numbers on this page.</p>"
        '<button type="button" class="run" id="explain-btn">Explain these results</button>'
        '<span class="status" id="explain-status"></span>'
        '<div id="explain-out"></div>'
        "</div>"
    )


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


def render_outcome_placeholder() -> str:
    """State plainly which outcome figures exist and which do not.

    Actual and previous values are obtainable from the publishing agency. A
    consensus *forecast* is not: those come from Bloomberg and Reuters surveys
    with no free, licence-clean feed. Rather than substitute something else and
    label it "forecast" -- the exact silent redefinition this project exists to
    avoid -- the gap is shown.
    """
    return (
        "<h2>Actual / forecast / previous</h2>"
        '<div class="card">'
        "<p>Not yet wired up. Actual and previous values are available from the "
        "publishing agency and are planned. A consensus <em>forecast</em> has no "
        "free licence-clean source, so it is shown as unavailable rather than "
        "replaced by a substitute labelled as one.</p>"
        "</div>"
    )


def render_upcoming_notice(event: MarketEvent) -> str:
    """Make clear that a future event has no outcome yet.

    The history below it is real; the event itself has not happened. Without
    this the two read as one thing.
    """
    return (
        '<div class="notice"><strong>This event has not happened yet.</strong> '
        "Everything below describes how "
        f"{escape(meta_for(event.event_type).label)} releases behaved in the past. "
        "It is a record, not a forecast.</div>"
    )


def price_chart_script(event_id: str, symbol: str) -> str:
    """Client code that fetches our candles and renders them.

    Colours are read from the page's CSS custom properties rather than
    hardcoded, so the chart follows the theme instead of being a second source
    of truth for it.
    """
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
    height: 360,
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
    timeScale: {{ borderColor: v('--grid', '#343431'), timeVisible: true }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});

  const up = v('--up', '#3987e5');
  const down = v('--down', '#e66767');
  const series = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor: up, borderUpColor: up, wickUpColor: up,
    downColor: down, borderDownColor: down, wickDownColor: down,
  }});

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
      loaded = true;
      fit();
      status.remove();
    }})
    .catch(e => {{ status.textContent = 'Could not load candles: ' + e; }});

  // Width and fit must be applied together. Fitting against a stale width
  // leaves the candles bunched to one side of the canvas, which is what
  // happened when the observer resized the chart after fitContent had run.
  function fit() {{
    chart.applyOptions({{ width: el.clientWidth }});
    if (loaded) chart.timeScale().fitContent();
  }}

  new ResizeObserver(fit).observe(el);
  fit();
}})();
"""


EVENT_STYLE = """
.breadcrumb { margin: 0 0 12px; font-size: 13px; }
.breadcrumb a { color: var(--accent); text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
.card p { margin: 0 0 12px; }
.card ul { margin: 0 0 12px; padding-left: 20px; }
.card li { margin-bottom: 4px; color: var(--text-secondary); }
.chart-caption { color: var(--text-muted); font-size: 12px; margin: -6px 0 10px; }
.chart-card { padding: 14px; }
#price-chart { width: 100%; }
.symbol-tabs { display: flex; gap: 6px; margin: 24px 0 12px; }
.symbol-tab {
  padding: 7px 16px; border: 1px solid var(--grid); border-radius: 7px;
  background: var(--surface-1); color: var(--text-secondary);
  font: inherit; font-size: 14px; cursor: pointer; text-decoration: none;
}
.symbol-tab[aria-selected="true"] {
  background: var(--accent); color: #fff; border-color: transparent; font-weight: 500;
}
"""
