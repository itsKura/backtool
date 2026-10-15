"""The calendar: what is coming, what just happened, and a way into each.

This is the front door. A form asking for event counts and window offsets
assumes the reader already knew what to ask; a calendar answers the question
they actually have, which is "what is coming up, and does it matter?"
"""

from __future__ import annotations

import datetime as dt
from xml.sax.saxutils import escape

from backtool.core.time import NEW_YORK, utc_to_local
from backtool.core.types import EventType
from backtool.events.meta import meta_for
from backtool.events.models import MarketEvent
from backtool.web.timefmt import local_stamp

#: How many past events to list. Enough to show recent context without turning
#: the page into an archive.
PAST_LIMIT = 8

#: Between two clocks shown on one line. Carried inside the second one, so a
#: stamp the browser drops as redundant takes its separator with it rather than
#: leaving one pointing at nothing.
SEPARATOR = " · "


def render_calendar(
    events: list[MarketEvent],
    *,
    now: dt.datetime,
) -> str:
    """Render the calendar body: upcoming events first, then recent ones.

    Args:
        events: The full merged calendar, any order.
        now: The current instant, used to split upcoming from past.

    Returns:
        HTML for the page body (no document wrapper).
    """
    ordered = sorted(events, key=lambda event: event.timestamp_utc)
    upcoming = [event for event in ordered if event.timestamp_utc >= now]
    past = [event for event in ordered if event.timestamp_utc < now][-PAST_LIMIT:]

    return (
        "<h2>Upcoming</h2>"
        f"{_event_list(upcoming, now, empty='No scheduled events ahead in the calendar.')}"
        "<h2>Recent</h2>"
        f"{_event_list(list(reversed(past)), now, empty='No past events in the calendar.')}"
    )


def _event_list(events: list[MarketEvent], now: dt.datetime, *, empty: str) -> str:
    if not events:
        return f'<div class="card"><p class="hint">{escape(empty)}</p></div>'
    return "".join(_event_card(event, now) for event in events)


def _event_card(event: MarketEvent, now: dt.datetime) -> str:
    meta = meta_for(event.curated_type)
    local = utc_to_local(event.timestamp_utc, NEW_YORK)
    delta = event.timestamp_utc - now
    upcoming = delta.total_seconds() > 0

    # The same instant on the reader's own clock. The browser drops it for
    # anyone already in the release's timezone, where it would otherwise print
    # the same time twice.
    in_your_time = local_stamp(
        event.timestamp_utc, lead=SEPARATOR, instead_of=f"{local:%H:%M}"
    )

    # A container rather than one big link, because the card now carries a
    # second destination. The main link still covers the whole card -- through
    # a stretched ::after in the stylesheet -- so the target is unchanged for
    # anyone who just clicks it; Prepare sits above that layer.
    return (
        f'<div class="event-card">'
        f'<a class="event-open" href="/event/{escape(event.event_id)}">'
        f'<span class="event-when">'
        f'<span class="event-date">{local:%a %d %b %Y}</span>'
        f'<span class="event-time">{local:%H:%M} ET{in_your_time}</span>'
        f"</span>"
        f'<span class="event-body">'
        f'<span class="event-title">{escape(meta.label)}</span>'
        f'<span class="event-sub">{escape(meta.source)}'
        f"{_note_suffix(event)}</span>"
        f"</span>"
        f"</a>"
        f'<span class="event-meta">'
        f"{_prepare_link(event, upcoming=upcoming)}"
        f'<span class="badge badge-{escape(meta.importance.value)}">'
        f"{escape(meta.importance.value)}</span>"
        f'<span class="event-countdown">{escape(format_delta(delta, upcoming))}</span>'
        f"</span>"
        f"</div>"
    )


def _prepare_link(event: MarketEvent, *, upcoming: bool) -> str:
    """The action on a release there is still time to prepare for.

    Two conditions, for two different reasons.

    Only CPI has the page at all: it reads a CPI values file and a CPI scenario
    rule, and offering the button on an event type with neither would promise
    something the route correctly refuses.

    And only ahead of the release, because that is what the word means. A
    release that has already printed is read about rather than prepared for,
    and the event page is where that reading lives. The preparation page still
    answers on its own URL -- a link kept from last month goes on working, and
    what it shows afterwards is deliberate -- it simply stops being offered
    here, where every other card leads to the event page.
    """
    if event.curated_type is not EventType.CPI or not upcoming:
        return ""
    return (
        f'<a class="prep-prepare" href="/event/{escape(event.event_id)}/prepare">'
        "Prepare</a>"
    )


def _note_suffix(event: MarketEvent) -> str:
    """Surface a calendar irregularity on the card itself.

    An event that was delayed or rescheduled is exactly the one a reader should
    not assume followed the usual pattern.
    """
    if event.is_scheduled:
        return ""
    return ' <span class="event-flag">rescheduled</span>'


def format_delta(delta: dt.timedelta, upcoming: bool) -> str:
    """Render a time delta as a short human phrase.

    Coarse on purpose: "in 3 weeks" is what a reader scanning a calendar wants,
    not "in 22 days, 4 hours".
    """
    seconds = abs(delta.total_seconds())
    days = seconds / 86400

    if days >= 14:
        value, unit = round(days / 7), "week"
    elif days >= 1:
        value, unit = int(days), "day"
    elif seconds >= 3600:
        value, unit = int(seconds // 3600), "hour"
    else:
        value, unit = max(1, int(seconds // 60)), "minute"

    plural = "" if value == 1 else "s"
    return f"in {value} {unit}{plural}" if upcoming else f"{value} {unit}{plural} ago"


CALENDAR_STYLE = """
.event-card {
  display: grid; grid-template-columns: 1fr auto; gap: 18px;
  align-items: center; color: inherit; position: relative;
  background: var(--surface-1); border: 1px solid var(--grid);
  border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
}
.event-card:hover { border-color: var(--accent); background: var(--surface-2); }
/* The card's own grid, kept on the link so the date and title stay in the
   columns they were in before the card gained a second destination. */
.event-open {
  display: grid; grid-template-columns: 190px 1fr; gap: 18px;
  align-items: center; text-decoration: none; color: inherit; min-width: 0;
}
/* Covers the whole card, so clicking anywhere still opens the event. A nested
   link would have been invalid; this is the same affordance without one. */
.event-open::after { content: ""; position: absolute; inset: 0;
                     border-radius: 10px; }
/* Above the stretched layer, or Prepare would be unclickable. */
.event-meta { position: relative; z-index: 1; }
.event-when { display: flex; flex-direction: column; }
.event-date { font-weight: 600; font-size: 14px; }
.event-time { color: var(--text-muted); font-size: 12px;
              font-family: ui-monospace, Consolas, monospace; }
.event-body { display: flex; flex-direction: column; min-width: 0; }
.event-title { font-size: 15px; font-weight: 500; }
.event-sub { color: var(--text-muted); font-size: 12px; }
.event-flag { color: var(--down); }
.event-meta { display: flex; align-items: center; gap: 14px; white-space: nowrap; }
.event-countdown { color: var(--text-secondary); font-size: 13px; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 99px;
         text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
.badge-high { background: #4a1f1f; color: #f3a9a8; }
.badge-medium { background: #43341a; color: #e9c274; }
.badge-low { background: #1e3350; color: #9cc0ea; }
:root[data-theme="light"] .badge-high { background: #fde8e7; color: #a2302f; }
:root[data-theme="light"] .badge-medium { background: #fdf0d8; color: #8a5d00; }
:root[data-theme="light"] .badge-low { background: #e6edf6; color: #2a5490; }
@media (max-width: 640px) {
  .event-card { grid-template-columns: 1fr; gap: 10px; }
  .event-open { grid-template-columns: 1fr; gap: 6px; }
  .event-meta { justify-content: flex-start; flex-wrap: wrap; gap: 10px; }
}
"""
