"""The Similar Setups section: today's conditions, and what happened last time.

This module owns the assembly -- fetch, measure, match, render -- so the route
stays a route. The work is blocking and network-bound, so the caller runs it off
the event loop.

The candle fetch here is the expensive part of the whole feature. Reading the
market state before a 2024 release needs two years of hourly history *before*
that release, so the range starts from the oldest release in the study rather
than from today. That is one large fetch the first time and a cache hit after,
which is why it is worth doing once for the whole set rather than per release.
"""

from __future__ import annotations

import datetime as dt
from xml.sax.saxutils import escape

from backtool.config import Settings
from backtool.core.time import UTC, ensure_utc
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events import load_events
from backtool.research.regime import (
    SETUP_INTERVAL,
    SETUP_SYMBOL,
    MarketSetup,
    compute_setup,
    history_window,
)
from backtool.research.runner import run_study
from backtool.research.setups import SetupMatch, match_setups
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.calendar_page import format_delta
from backtool.web.cohort_view import (
    render_cohort_table,
    render_ladder,
    render_matched_releases,
    render_sample_badge,
    render_setup,
)

#: How many past releases of the event's type to consider as candidates. Larger
#: than the history section's twenty: matching discards most candidates, so the
#: pool has to be wider than the sample it is expected to yield.
SETUP_CANDIDATES = 40


def build_setup_match(
    *,
    event_type: EventType,
    symbol: str,
    as_of: dt.datetime,
    interval: Interval = Interval.M5,
) -> tuple[SetupMatch, int]:
    """Measure today, measure every past release, and match.

    Args:
        event_type: Which release type to search.
        symbol: The market whose reaction is measured. The *conditions* are
            always read from BTC -- a setup describes the market, not the pair.
        as_of: The instant "today's conditions" are read at.
        interval: Candle interval for the reaction windows.

    Returns:
        The match, and how many candidate releases had a measurable setup.
    """
    now = dt.datetime.now(tz=UTC)
    spec = ResearchSpec(
        symbol=symbol,
        event_type=event_type,
        event_count=SETUP_CANDIDATES,
        interval=interval,
        windows=REACTION_WINDOWS,
        as_of=as_of,
    )

    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(event_type), now=now)
        if not study.events:
            return _empty_match(as_of), 0

        # One fetch covering every release's own lookback. Starting at today's
        # lookback instead would leave the oldest releases without a 200-day
        # average, and they would drop out of matching for a reason that has
        # nothing to do with the market.
        oldest = min(event.event_time_utc for event in study.events)
        start, _ = history_window(oldest)
        frame = service.get_candles(SETUP_SYMBOL, SETUP_INTERVAL, start, as_of)

    current = compute_setup(frame, as_of)
    history: dict[str, MarketSetup] = {}
    for event in study.events:
        setup = compute_setup(frame, event.event_time_utc)
        if setup.measured:
            history[event.event_id] = setup

    match = match_setups(
        study,
        current,
        history,
        label="Similar setups",
        description=(
            f"Past {event_type.value} releases that arrived with the market in "
            "the same condition as today."
        ),
    )
    return match, len(history)


def _empty_match(as_of: dt.datetime) -> SetupMatch:
    return SetupMatch(
        current=MarketSetup(as_of=as_of),
        matched_on=(),
        relaxed=(),
        exact_count=0,
        candidate_count=0,
        matches=(),
        cohort=None,
    )


#: Inside this many hours of a release, the reading stops being "current
#: conditions" and starts being the state the release will actually land into.
#: Two days is roughly the point where a 7-day trend and a 30-day trend can no
#: longer flip from where they sit now without an unusual move.
PRE_EVENT_HOURS = 48


def setup_framing(*, as_of: dt.datetime, event_time: dt.datetime) -> tuple[str, str]:
    """How to title and caveat a setup, given how far the release still is.

    Three cases, and the distinction is the honest part. For a past release the
    reading is settled history. Within two days it is what the release will
    land into. Weeks out it is a reading of today that has no particular claim
    on release day, and saying "pre-event setup" there would dress up a current
    observation as a prediction.

    Returns:
        ``(title, note)``. The note is empty when there is nothing to qualify.
    """
    lead = (ensure_utc(event_time) - ensure_utc(as_of)).total_seconds() / 3600.0

    if lead <= 0:
        return "Market setup before the release", ""

    remaining = format_delta(dt.timedelta(hours=lead), True)
    if lead <= PRE_EVENT_HOURS:
        return (
            "Pre-event setup",
            f"The release is {remaining}. These are the conditions it is "
            "arriving into, barring an unusual move between now and then.",
        )
    return (
        "Current market setup",
        f"The release is {remaining}. This is a reading of the market now, not "
        "a description of what it will look like on release day -- any of these "
        "four conditions can change before then.",
    )


def render_setups_section(
    match: SetupMatch,
    *,
    measurable: int,
    symbol: str,
    event_time: dt.datetime | None = None,
) -> str:
    """The whole section, as a fragment.

    Order follows the argument: here is what today looks like, here is how many
    past releases looked like it, here is what happened, here is how the sample
    was chosen, here are the releases themselves.
    """
    title, note = (
        setup_framing(as_of=match.current.as_of, event_time=event_time)
        if event_time is not None
        else ("Market setup", "")
    )
    current = render_setup(match.current, title=title, note=note)

    if match.cohort is None or match.sample_size == 0:
        return (
            current
            + '<div class="panel-card"><p class="muted">No past release arrived '
            "with the market in a condition close enough to today's to compare "
            f"against. {match.candidate_count} release"
            f'{"" if match.candidate_count == 1 else "s"} were considered.</p>'
            "</div>"
        )

    relaxed = match.relaxation_note()
    relaxed_note = f'<p class="note">{escape(relaxed)}</p>' if relaxed else ""

    return (
        current
        + '<div class="panel-card">'
        + f"{render_sample_badge(match.cohort)}"
        + relaxed_note
        + '<p class="chart-note">Reaction of <span class="mono">'
        + f"{escape(symbol)}</span> across these {match.sample_size} release"
        + f'{"" if match.sample_size == 1 else "s"}, measured from the release '
        + "price at each horizon.</p>"
        + render_cohort_table(match.cohort, REACTION_WINDOWS)
        + render_ladder(match)
        + render_matched_releases(match)
        + f'<p class="chart-note">{measurable} of {match.candidate_count} '
        + "candidate releases had enough price history before them to read the "
        + "market state. Conditions are always read from BTC, whichever market "
        + "the reaction is measured in.</p>"
        + "</div>"
    )


__all__ = [
    "PRE_EVENT_HOURS",
    "SETUP_CANDIDATES",
    "build_setup_match",
    "render_setups_section",
    "setup_framing",
]
