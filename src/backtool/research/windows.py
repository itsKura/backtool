"""Event-relative time windows.

A window is expressed as two offsets from an event, e.g. ``-24h`` to ``0h``.
Resolving it against a concrete event instant produces two absolute UTC
timestamps. Keeping windows relative is what lets one specification apply to
twenty different FOMC meetings.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
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


_ZERO = dt.timedelta(0)

#: Units tried largest-first when rendering a duration, each with the smallest
#: value at which it is worth using. A window boundary is always a whole number
#: of one of these in practice, but the fallback below keeps an odd offset
#: readable rather than crashing.
#:
#: Days start at two, not one. A 24-hour window written as "1 day" is correct
#: and useless: it sits in a tab strip beside "1 hour before" and "1 hour
#: after", where the reader is comparing spans and the unit switching under
#: them hides that 24h is twenty-four times the others.
_UNITS: tuple[tuple[str, str, int, int], ...] = (
    # A year is 365 days exactly. Nothing here is a calendar operation -- an
    # offset is a duration added to an instant -- so the only thing a leap day
    # would change is which arbitrary number gets printed.
    ("year", "y", 31536000, 1),
    ("day", "d", 86400, 2),
    ("hour", "h", 3600, 1),
    ("minute", "m", 60, 1),
)


def humanise_duration(delta: dt.timedelta) -> str:
    """Render a duration in words: ``24 hours``, ``1 hour``, ``30 minutes``.

    Used in headings and prose, where ``24h`` reads as jargon.
    """
    seconds = abs(delta.total_seconds())
    for name, _, size, minimum in _UNITS:
        if seconds >= size * minimum and seconds % size == 0:
            value = int(seconds // size)
            return f"{value} {name}{'' if value == 1 else 's'}"
    minutes = seconds / 60
    return f"{minutes:g} minutes"


def compact_duration(delta: dt.timedelta) -> str:
    """Render a duration tersely: ``24h``, ``1h``, ``30m``.

    Used in table headers and chips, where a full phrase would force the
    column wider than the numbers it describes.
    """
    seconds = abs(delta.total_seconds())
    for _, short, size, minimum in _UNITS:
        if seconds >= size * minimum and seconds % size == 0:
            return f"{int(seconds // size)}{short}"
    return f"{seconds / 60:g}m"


def window_title(spec: WindowSpec) -> str:
    """The window as a heading: ``24 hours before``, ``1 hour after``."""
    return _describe(spec, humanise_duration)


def window_label(spec: WindowSpec) -> str:
    """The window as a compact chip or column header: ``24h before``."""
    return _describe(spec, compact_duration)


def _describe(spec: WindowSpec, fmt: Callable[[dt.timedelta], str]) -> str:
    """Shared shape for both label widths.

    ``fmt`` is the duration formatter; the five cases below are the same either
    way, so the phrasing lives in one place and the two public functions cannot
    drift apart.
    """
    start, end = spec.start_offset, spec.end_offset

    if start < _ZERO and end == _ZERO:
        return f"{fmt(start)} before"
    if start == _ZERO and end > _ZERO:
        return f"{fmt(end)} after"
    if start < _ZERO and end > _ZERO:
        return f"{fmt(start)} before to {fmt(end)} after"
    if start < _ZERO and end < _ZERO:
        return f"{fmt(start)} to {fmt(end)} before"
    return f"{fmt(start)} to {fmt(end)} after"


def window_sentence(spec: WindowSpec) -> str:
    """The window as a phrase inside a sentence: ``the 24 hours before it``."""
    start, end = spec.start_offset, spec.end_offset
    if start < _ZERO and end == _ZERO:
        return f"the {humanise_duration(start)} before the release"
    if start == _ZERO and end > _ZERO:
        return f"the {humanise_duration(end)} after the release"
    return window_title(spec)


#: The default V1 window set. Deliberately small: two pre-event windows to
#: characterise the run-up, two post-event windows to characterise the reaction.
#: Every one of these is an explicit, editable research choice, not a constant.
DEFAULT_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(name="pre_event_24h", start="-24h", end="0h"),
    WindowSpec(name="pre_event_1h", start="-1h", end="0h"),
    WindowSpec(name="post_event_1h", start="0h", end="1h"),
    WindowSpec(name="post_event_24h", start="0h", end="24h"),
)

#: The reaction grid used by scenario and setup cohorts: how far price had moved
#: at five points after the release, all measured from the same anchor.
#:
#: Every one of these is a *separate* measurement from the release price, not a
#: leg of a path -- the 24h figure is not the 4h figure plus something. Reading
#: them as a sequence is the point: a move that appears at 5m and holds to 24h
#: is a different event from one that appears at 5m and is gone by 1h.
#:
#: The 5m window is exactly one candle at the 5m interval. Anything finer would
#: need an interval this project deliberately does not support.
REACTION_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(name="post_event_5m", start="0h", end="5m"),
    WindowSpec(name="post_event_15m", start="0h", end="15m"),
    WindowSpec(name="post_event_1h", start="0h", end="1h"),
    WindowSpec(name="post_event_4h", start="0h", end="4h"),
    WindowSpec(name="post_event_24h", start="0h", end="24h"),
)

#: The windows a long-horizon study measures, spanning a week to two years.
#:
#: Separate from :data:`REACTION_WINDOWS` rather than appended to it, because a
#: study has one candle interval and no interval serves both: two years at five
#: minutes is 210,240 candles per event, and five minutes at one hour is not
#: resolvable at all. Which family a thesis runs against follows from the
#: horizon it asked about.
#:
#: The same reasoning as the reaction grid applies within this one. Each is
#: measured from the event, not from the previous step -- the 2y figure is not
#: the 1y figure plus something -- and reading them in sequence is the point: a
#: move still present after two years is a different event from one that was
#: there at ninety days and gone by one.
LONG_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(name="post_event_7d", start="0h", end="7d"),
    WindowSpec(name="post_event_30d", start="0h", end="30d"),
    WindowSpec(name="post_event_90d", start="0h", end="90d"),
    WindowSpec(name="post_event_180d", start="0h", end="180d"),
    WindowSpec(name="post_event_1y", start="0h", end="365d"),
    WindowSpec(name="post_event_2y", start="0h", end="730d"),
)

