"""Shared domain types with no I/O dependencies."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum


class Interval(StrEnum):
    """A supported candle interval.

    A :class:`StrEnum` so the member serialises as its exchange-native string
    (``"5m"``) in JSON, Parquet paths, and Binance query parameters without any
    conversion layer.

    V1 deliberately supports only these three. Sub-minute intervals invite
    microstructure noise that this system is not designed to reason about, and
    intervals above 1h are too coarse to resolve a 1-hour post-event window.
    """

    M5 = "5m"
    M15 = "15m"
    H1 = "1h"

    @property
    def duration(self) -> dt.timedelta:
        """Wall-clock length of one candle of this interval."""
        return _DURATIONS[self]

    @property
    def milliseconds(self) -> int:
        """Length of one candle in milliseconds (Binance's native unit)."""
        return int(self.duration.total_seconds() * 1000)


_DURATIONS: dict[Interval, dt.timedelta] = {
    Interval.M5: dt.timedelta(minutes=5),
    Interval.M15: dt.timedelta(minutes=15),
    Interval.H1: dt.timedelta(hours=1),
}


class EventType(StrEnum):
    """A category of scheduled market event.

    An event is fully described by a local wall-clock time and an IANA zone, so
    adding a type needs a calendar file and an entry here -- nothing in the time
    layer or research engine changes. CPI demonstrated that: released 08:30 ET
    rather than 14:00, monthly rather than eight times a year, no code changes
    above the data file.
    """

    #: US Federal Reserve interest-rate decisions. Statement at 14:00 ET on the
    #: second day of a two-day meeting.
    FOMC = "FOMC"
    #: US Consumer Price Index, published by the Bureau of Labor Statistics at
    #: 08:30 ET. The publication date is the event, not the reference month.
    CPI = "CPI"
