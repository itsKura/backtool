"""Loading and selecting macroeconomic events.

One loader for every event type. Adding a new one is a CSV plus a row in
:data:`CALENDAR_FILES` -- the model, the time handling, and the research engine
need no changes, because an event is fully described by a local wall-clock time
and an IANA zone.

CPI proved that out: it is released at 08:30 ET rather than 14:00, monthly
rather than eight times a year, and nothing above the data file had to change.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

from backtool.core.time import UTC, ensure_utc
from backtool.core.types import EventType
from backtool.events.models import MarketEvent

logger = logging.getLogger(__name__)

#: Packaged calendar file for each supported event type.
CALENDAR_FILES: dict[EventType, str] = {
    EventType.FOMC: "fomc_meetings.csv",
    EventType.CPI: "cpi_releases.csv",
}


class EventDataError(ValueError):
    """A calendar file is malformed or internally inconsistent."""


def load_events(
    event_type: EventType | None = None,
    *,
    path: Path | None = None,
) -> list[MarketEvent]:
    """Load a calendar, validated and sorted by announcement instant.

    Args:
        event_type: Which calendar to load. ``None`` loads and merges every
            supported type, which is what a cross-event-type study needs.
        path: Optional override for a single calendar file. Requires
            ``event_type``.

    Returns:
        Events sorted ascending by :attr:`~MarketEvent.timestamp_utc`.

    Raises:
        EventDataError: on a malformed row, a duplicate event id, or a
            wall-clock time that does not map to exactly one UTC instant.
        ValueError: if ``path`` is given without ``event_type``.
    """
    if path is not None and event_type is None:
        raise ValueError("path requires an explicit event_type")

    types = [event_type] if event_type is not None else list(CALENDAR_FILES)
    events: list[MarketEvent] = []
    for kind in types:
        events.extend(_load_one(kind, path))

    _assert_unique_ids(events)
    events.sort(key=lambda event: event.timestamp_utc)

    if events:
        logger.info(
            "Loaded %d event(s) across %s (%s to %s)",
            len(events),
            ", ".join(kind.value for kind in types),
            events[0].local_date,
            events[-1].local_date,
        )
    return events


def _load_one(event_type: EventType, path: Path | None) -> list[MarketEvent]:
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    else:
        filename = CALENDAR_FILES.get(event_type)
        if filename is None:
            raise EventDataError(f"No calendar file registered for {event_type.value}")
        text = (
            resources.files("backtool.events.data")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )

    events: list[MarketEvent] = []
    for line_number, row in enumerate(csv.DictReader(text.splitlines()), start=2):
        try:
            events.append(MarketEvent.model_validate(row))
        except ValidationError as exc:
            raise EventDataError(
                f"Invalid {event_type.value} event on line {line_number}: {exc}"
            ) from exc

    # Touch timestamp_utc now so a bad wall-clock time surfaces at load time
    # rather than midway through a research run.
    for event in events:
        _ = event.timestamp_utc
    return events


def _assert_unique_ids(events: list[MarketEvent]) -> None:
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            raise EventDataError(f"Duplicate event id: {event.event_id}")
        seen.add(event.event_id)


def load_fomc_events(path: Path | None = None) -> list[MarketEvent]:
    """Load the FOMC calendar. Thin wrapper over :func:`load_events`."""
    return load_events(EventType.FOMC, path=path)


def select_last_n(
    events: list[MarketEvent],
    n: int,
    *,
    before: dt.datetime | None = None,
    event_type: EventType | None = None,
) -> list[MarketEvent]:
    """Return the ``n`` most recent events that had already occurred.

    Future-dated calendar entries are excluded. Calendars intentionally contain
    scheduled events that have not happened yet; including one would be a
    look-ahead error of the most literal kind.

    Args:
        events: Candidate events, in any order.
        n: Maximum number of events to return.
        before: Cutoff instant; events at or after it are excluded. Defaults to
            the current time. Pass an explicit value to make a run reproducible.
        event_type: Optional filter on event type.

    Returns:
        Up to ``n`` events, sorted ascending by announcement instant. Fewer than
        ``n`` are returned if the calendar does not contain enough -- callers
        must check the length rather than assume it.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    cutoff = ensure_utc(before) if before is not None else dt.datetime.now(tz=UTC)

    candidates = [
        event
        for event in events
        if event.timestamp_utc < cutoff
        and (event_type is None or event.event_type == event_type)
    ]
    candidates.sort(key=lambda event: event.timestamp_utc)

    selected = candidates[-n:]
    if len(selected) < n:
        logger.warning(
            "Requested %d events but only %d occurred before %s", n, len(selected), cutoff
        )
    return selected
