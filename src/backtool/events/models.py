"""Event domain model."""

from __future__ import annotations

import datetime as dt
from functools import cached_property
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator

from backtool.core.time import local_to_utc
from backtool.core.types import EventType


class MarketEvent(BaseModel):
    """A single scheduled (or unscheduled) macroeconomic event.

    The announcement instant is stored as **local wall-clock time plus IANA
    timezone**, never as a UTC column. UTC is derived on demand by
    :attr:`timestamp_utc`.

    This is deliberate. A stored UTC column can silently disagree with the
    local time beside it -- for example if a source is re-imported after a DST
    rule change, or if someone edits one field and not the other. Deriving UTC
    from a single source of truth makes that disagreement unrepresentable.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: EventType
    local_date: dt.date
    local_time: dt.time
    timezone: str
    is_scheduled: bool = True
    notes: str = ""

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        """Reject unknown IANA zone keys at load time, not at analysis time."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone {value!r}") from exc
        return value

    @cached_property
    def zone(self) -> ZoneInfo:
        """The event's timezone as a :class:`ZoneInfo`."""
        return ZoneInfo(self.timezone)

    @cached_property
    def timestamp_utc(self) -> dt.datetime:
        """Exact UTC instant of the announcement.

        Raises:
            LocalTimeError: if the recorded wall-clock time is non-existent or
                ambiguous in its timezone (a data error worth failing on).
        """
        return local_to_utc(self.local_date, self.local_time, self.zone)

    def __str__(self) -> str:
        return (
            f"{self.event_id} ({self.local_date} {self.local_time:%H:%M} "
            f"{self.timezone} = {self.timestamp_utc:%Y-%m-%d %H:%M} UTC)"
        )
