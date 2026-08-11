"""Loading and selection of FOMC calendar events."""

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

_PACKAGED_CSV = "fomc_meetings.csv"


class EventDataError(ValueError):
    """The event calendar file is malformed or internally inconsistent."""


def load_fomc_events(path: Path | None = None) -> list[MarketEvent]:
    """Load the FOMC calendar, validated and sorted by announcement instant.

    Args:
        path: Optional override for the CSV location. Defaults to the file
            packaged inside ``backtool.events.data``, resolved via
            ``importlib.resources`` so it works from an installed wheel as well
            as from a source checkout.

    Returns:
        Events sorted ascending by :attr:`~MarketEvent.timestamp_utc`.

    Raises:
        EventDataError: on a malformed row, a duplicate event id, or a
            wall-clock time that does not map to exactly one UTC instant.
    """
    if path is None:
        text = resources.files("backtool.events.data").joinpath(_PACKAGED_CSV).read_text(
            encoding="utf-8"
        )
    else:
        text = Path(path).read_text(encoding="utf-8")

    events: list[MarketEvent] = []
    reader = csv.DictReader(text.splitlines())

    for line_number, row in enumerate(reader, start=2):  # start=2 accounts for the header
        try:
            events.append(MarketEvent.model_validate(row))
        except ValidationError as exc:
            raise EventDataError(f"Invalid event on line {line_number}: {exc}") from exc

    _assert_unique_ids(events)

    # Touch timestamp_utc on every event now so a bad wall-clock time surfaces
    # at load time rather than midway through a research run.
    events.sort(key=lambda event: event.timestamp_utc)

    logger.info(
        "Loaded %d FOMC events (%s to %s)",
        len(events),
        events[0].local_date if events else "n/a",
        events[-1].local_date if events else "n/a",
    )
    return events


def _assert_unique_ids(events: list[MarketEvent]) -> None:
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            raise EventDataError(f"Duplicate event id: {event.event_id}")
        seen.add(event.event_id)


def select_last_n(
    events: list[MarketEvent],
    n: int,
    *,
    before: dt.datetime | None = None,
    event_type: EventType | None = None,
) -> list[MarketEvent]:
    """Return the ``n`` most recent events that had already occurred.

    Future-dated calendar entries are excluded. The calendar intentionally
    contains scheduled meetings that have not happened yet; including one in an
    analysis would be a look-ahead error of the most literal kind.

    Args:
        events: Candidate events, in any order.
        n: Maximum number of events to return.
        before: Cutoff instant; events at or after it are excluded. Defaults to
            the current time. Pass an explicit value to make a research run
            reproducible.
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
