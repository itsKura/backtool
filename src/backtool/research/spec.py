"""The research specification: a complete, reproducible definition of a study.

Everything that affects a result lives here -- symbol, event type, how many
events, candle interval, the exact windows, and the as-of cutoff. Two runs of
the same spec against the same data must produce identical numbers, so the spec
carries a content hash that appears alongside every result.

This is also the schema the AI planner will emit in Milestone 5. It is a
Pydantic model rather than a dataclass for exactly that reason: the same class
validates a handwritten spec today and a model-generated one later, with no
translation layer and no second definition to drift out of sync.
"""

from __future__ import annotations

import datetime as dt
import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtool.core.types import EventType, Interval
from backtool.research.windows import DEFAULT_WINDOWS, WindowSpec


class ResearchSpec(BaseModel):
    """A fully-specified, reproducible research question."""

    model_config = ConfigDict(frozen=True)

    symbol: str = "BTCUSDT"
    event_type: EventType = EventType.FOMC
    event_count: int = Field(default=20, ge=1, le=200)
    interval: Interval = Interval.M5
    windows: tuple[WindowSpec, ...] = DEFAULT_WINDOWS

    #: Analysis cutoff. Events at or after this instant are excluded. Pinning it
    #: is what makes a run reproducible -- left as ``None`` it means "now", and
    #: the same spec would select different events tomorrow.
    as_of: dt.datetime | None = None

    @field_validator("symbol")
    @classmethod
    def _normalise_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol.isalnum():
            raise ValueError(f"Symbol {value!r} must be alphanumeric")
        return symbol

    @field_validator("windows")
    @classmethod
    def _validate_windows(cls, value: tuple[WindowSpec, ...]) -> tuple[WindowSpec, ...]:
        if not value:
            raise ValueError("A spec must define at least one window")
        names = [window.name for window in value]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate window name(s): {sorted(duplicates)}")
        return value

    @property
    def earliest_offset(self) -> dt.timedelta:
        """How far before an event any window reaches."""
        return min(window.start_offset for window in self.windows)

    @property
    def latest_offset(self) -> dt.timedelta:
        """How far after an event any window reaches."""
        return max(window.end_offset for window in self.windows)

    @property
    def fingerprint(self) -> str:
        """Short content hash identifying this exact specification.

        Reported with every result so a set of numbers can be traced back to the
        question that produced them. Changing any field -- including a single
        window boundary -- changes the fingerprint.
        """
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> list[str]:
        """Human-readable assumptions, for display above any result.

        The product promise is that no definition is ever applied silently, so
        every choice that affects the numbers is rendered here.
        """
        if self.as_of is not None:
            cutoff = f"before {self.as_of:%Y-%m-%d %H:%M UTC}"
        else:
            cutoff = "as of now (not reproducible - pin --as-of)"

        lines = [
            f"symbol           {self.symbol}",
            f"event type       {self.event_type.value}",
            f"events           last {self.event_count}, {cutoff}",
            f"interval         {self.interval.value}",
            "anchor rule      last candle closed at or before the boundary (no look-ahead)",
            "MFE / MAE        relative to a long opened at the window start price",
            "realized vol     root sum of squared log returns over the window, not annualised",
        ]
        for window in self.windows:
            lines.append(f"window           {window.name}: {window.start} -> {window.end}")
        lines.append(f"spec fingerprint {self.fingerprint}")
        return lines
