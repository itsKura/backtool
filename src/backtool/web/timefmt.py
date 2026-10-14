"""Instants shown in the reader's own timezone.

"as of 21:26 UTC" answers the wrong question. Somebody looking at a price wants
to know whether it is fresh, and that is a comparison against the clock on
their own wall -- so a UTC stamp asks them to do arithmetic in their head, with
an offset they have to remember, against the most time-sensitive number on the
page.

The server cannot know that timezone. It is a property of the browser, and
guessing it from an IP address or an Accept-Language header is how a page ends
up confidently wrong about something trivially knowable on the other side. So
the server writes UTC -- correct, unambiguous and readable on its own -- and
the browser rewrites it in place. With scripting off the page still states a
true time rather than leaving an empty element.

This is formatting, not calculation. The instant does not change; only the
clock it is read against does. No statistic is computed in the browser, and the
UTC original stays on the element's ``title``, so nothing is actually lost.

Its own module because both :mod:`backtool.web.page` and the page modules it
imports need it, and putting it in either would make a cycle.
"""

from __future__ import annotations

import datetime as dt
from xml.sax.saxutils import escape, quoteattr


def local_stamp(
    instant: dt.datetime,
    *,
    with_date: bool = False,
    lead: str = "",
    instead_of: str = "",
) -> str:
    """An instant the browser rewrites into the reader's own timezone.

    Args:
        instant: The moment, timezone-aware.
        with_date: Include the date. Worth it for a stamp that could be more
            than a day old, where "21:26" alone invites the reader to assume
            today.
        lead: Text carried *inside* the element, such as a separator. Inside
            rather than beside it, so that dropping a redundant stamp takes its
            separator with it instead of leaving one pointing at nothing.
        instead_of: A 24-hour ``HH:MM`` this stamp would merely repeat -- the
            event's own published local time, on a page that already shows it.
            A reader in that timezone then sees one clock rather than the same
            one twice, which on a US release page is most of them.

    Returns:
        A ``<time>`` element carrying the UTC reading as its text.
    """
    pattern = "%d %b %Y, %H:%M" if with_date else "%H:%M"
    kind = "datetime" if with_date else "time"
    duplicate = f" data-instead-of={quoteattr(instead_of)}" if instead_of else ""
    return (
        f"<time datetime={quoteattr(instant.isoformat())} "
        f'data-local="{kind}"{duplicate}>'
        f"{escape(lead)}{instant:{pattern}} UTC</time>"
    )


#: Rewrites every stamp above into the clock the reader is actually looking at.
#:
#: Deliberately small and failure-tolerant: an unparseable datetime, a locale
#: with no short zone name, or no scripting at all each leave the server's UTC
#: text exactly as it was written. The worst case is the page as it was before.
LOCAL_TIME_SCRIPT = """
(function () {
  var stamps = document.querySelectorAll('time[data-local]');
  for (var i = 0; i < stamps.length; i++) {
    var el = stamps[i];
    var when = new Date(el.getAttribute('datetime'));
    if (isNaN(when.getTime())) continue;

    // 24-hour regardless of locale, because every other time on these pages is
    // written that way -- "08:30 ET" beside "07:48 PM EDT" reads as two
    // different products. The zone name still comes from the reader's locale.
    var opts = { hour: '2-digit', minute: '2-digit', hour12: false,
                 timeZoneName: 'short' };
    if (el.getAttribute('data-local') === 'datetime') {
      opts.day = '2-digit'; opts.month = 'short'; opts.year = 'numeric';
    }

    // A stamp that would only repeat a time already on the line goes away
    // rather than appearing twice. Its separator is inside it, so it goes too.
    var same = el.getAttribute('data-instead-of');
    if (same) {
      var clock = when.toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', hour12: false });
      if (clock === same) { el.parentNode.removeChild(el); continue; }
    }

    try {
      var text = el.textContent;
      var lead = text.match(/^[^0-9]*/);
      // The server's reading, kept on hover. It already says UTC.
      el.title = text.trim();
      el.textContent = (lead ? lead[0] : '') + when.toLocaleString([], opts);
    } catch (err) {
      /* Leave the server's UTC text alone. */
    }
  }
})();
"""
