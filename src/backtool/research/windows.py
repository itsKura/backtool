"""Event-relative time windows.

A window is expressed as two offsets from an event, e.g. ``-24h`` to ``0h``.
Resolving it against a concrete event instant produces two absolute UTC
timestamps. Keeping windows relative is what lets one specification apply to
twenty different FOMC meetings.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backtool.core.time import ensure_utc

#: ``-24h``, ``+90m``, ``0h``, ``3d``. Sign optional (defaults to positive),
#: unit required. Requiring the unit avoids the "is 24 hours or days?" trap.
_OFFSET_PATTERN = re.compile(r"^([+-]?)(\d+(?:\.\d+)?)([mhd])$")

_UNIT_TO_TIMEDELTA = {
    "m": dt.timedelta(minutes=1),
    "h": dt.timedelta(hours=1),
    "d": dt.timedelta(days=1),
}


def parse_offset(text: str) -> dt.timedelta:
    """Parse an event-relative offset string into a :class:`~datetime.timedelta`.

    Args:
        text: Offset such as ``"-24h"``, ``"+1h"``, ``"0h"``, ``"30m"``, ``"3d"``.

    Returns:
        The signed duration. ``"-24h"`` is 24 hours *before* the event.

    Raises:
        ValueError: if the string is not a recognised offset. Failing loudly
            matters here because a silently-misparsed offset would shift the
            entire analysis without any other symptom.
    """
    match = _OFFSET_PATTERN.match(text.strip())
    if match is None:
        raise ValueError(
            f"Invalid offset {text!r}. Expected a signed number with a unit, "
            "e.g. '-24h', '+90m', '0h', '3d'."
        )

    sign, magnitude, unit = match.groups()
    delta = float(magnitude) * _UNIT_TO_TIMEDELTA[unit]
    return -delta if sign == "-" else delta


def format_offset(delta: dt.timedelta) -> str:
    """Render a timedelta back as a compact offset string, for display."""
    total_seconds = delta.total_seconds()
    sign = "-" if total_seconds < 0 else ""
    seconds = abs(total_seconds)

    for unit, unit_seconds in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= unit_seconds and seconds % unit_seconds == 0:
            return f"{sign}{int(seconds // unit_seconds)}{unit}"
    return f"{sign}{seconds / 60:g}m"


@dataclass(frozen=True)
class ResolvedWindow:
    """A window bound to one specific event: two absolute UTC instants."""

    name: str
    start: dt.datetime
    end: dt.datetime

    @property
    def duration(self) -> dt.timedelta:
        return self.end - self.start

    def __str__(self) -> str:
        return f"{self.name}[{self.start:%Y-%m-%d %H:%M} .. {self.end:%Y-%m-%d %H:%M} UTC]"


class WindowSpec(BaseModel):
    """A named, event-relative window in a research specification.

    This is the model the AI planner will eventually emit, which is why it is a
    Pydantic model rather than a plain dataclass: the same class becomes the
    LLM's structured-output schema with no translation layer.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    start: str
    end: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Window name must not be empty")
        return value

    @field_validator("start", "end")
    @classmethod
    def _validate_offsets(cls, value: str) -> str:
        parse_offset(value)  # raises on malformed input
        return value

    @model_validator(mode="after")
    def _validate_ordering(self) -> WindowSpec:
        if self.start_offset >= self.end_offset:
            raise ValueError(
                f"Window {self.name!r} has start {self.start!r} at or after end "
                f"{self.end!r}. Windows must span forward in time."
            )
        return self

    @property
    def start_offset(self) -> dt.timedelta:
        return parse_offset(self.start)

    @property
    def end_offset(self) -> dt.timedelta:
        return parse_offset(self.end)

    def resolve(self, event_time: dt.datetime) -> ResolvedWindow:
        """Bind this window to a concrete event instant.

        Args:
            event_time: The announcement instant. Must be timezone-aware.

        Returns:
            The window as two absolute UTC timestamps.

        Raises:
            ValueError: if ``event_time`` is timezone-naive.
        """
        anchor = ensure_utc(event_time)
        return ResolvedWindow(
            name=self.name,
            start=anchor + self.start_offset,
            end=anchor + self.end_offset,
        )


#: The default V1 window set. Deliberately small: two pre-event windows to
#: characterise the run-up, two post-event windows to characterise the reaction.
#: Every one of these is an explicit, editable research choice, not a constant.
DEFAULT_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(name="pre_event_24h", start="-24h", end="0h"),
    WindowSpec(name="pre_event_1h", start="-1h", end="0h"),
    WindowSpec(name="post_event_1h", start="0h", end="1h"),
    WindowSpec(name="post_event_24h", start="0h", end="24h"),
)
