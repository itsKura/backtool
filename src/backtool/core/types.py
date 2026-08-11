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

    V1 ships FOMC only. The enum exists now so that event identity is typed
    rather than stringly-typed the moment CPI or NFP is added.
    """

    FOMC = "FOMC"
