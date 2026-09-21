"""The event detail page: what this event is, and what happened the last N times.

Structure follows what a reader needs in order: what the event is, why it moves
markets, then the historical record, then the evidence behind it. Explanation
before numbers, numbers before conclusions.

The history is produced by the same engine the CLI uses. Nothing here computes a
statistic.
"""

from __future__ import annotations

import datetime as dt
from xml.sax.saxutils import escape

from backtool.core.time import NEW_YORK, utc_to_local
from backtool.events.meta import meta_for
from backtool.events.models import MarketEvent
from backtool.web.calendar_page import format_delta


def render_event_header(event: MarketEvent, *, now: dt.datetime) -> str:
    """Render the explanatory part of an event page.

    Everything above the charts: identity, timing, what the event is, and the
    mechanism by which it affects prices.
    """
    meta = meta_for(event.event_type)
    local = utc_to_local(event.timestamp_utc, NEW_YORK)
    delta = event.timestamp_utc - now
    upcoming = delta.total_seconds() > 0

    watch = "".join(f"<li>{escape(item)}</li>" for item in meta.what_to_watch)

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
        '<div class="card">'
        f'<p class="ai-label">What it is</p><p>{escape(meta.what_it_is)}</p>'
        f'<p class="ai-label">Why markets care</p><p>{escape(meta.why_markets_care)}</p>'
        f'<p class="ai-label">What to watch</p><ul>{watch}</ul>'
        f'<p class="hint">{escape(meta.source)} &middot; {escape(meta.release_time)}</p>'
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


def render_outcome_placeholder() -> str:
    """State plainly which outcome figures exist and which do not.

    Actual and previous values are obtainable from the publishing agency. A
    consensus *forecast* is not: those come from Bloomberg and Reuters surveys
    with no free, licence-clean feed. Rather than substitute something else and
    label it "forecast" -- the exact silent redefinition this project exists to
    avoid -- the gap is shown.
    """
    return (
        '<div class="card">'
        '<p class="ai-label">Actual / forecast / previous</p>'
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


EVENT_STYLE = """
.breadcrumb { margin: 0 0 12px; font-size: 13px; }
.breadcrumb a { color: var(--accent); text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
.card p { margin: 0 0 12px; }
.card ul { margin: 0 0 12px; padding-left: 20px; }
.card li { margin-bottom: 4px; color: var(--text-secondary); }
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
