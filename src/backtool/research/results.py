"""Result models for a completed study.

Separate from :mod:`backtool.research.runner` on purpose. The runner needs the
market-data layer to *produce* these; anything that only *reads* them -- the
reporting layer, the AI interpreter -- must not drag that dependency along.

Keeping the models here is what lets ``backtool.ai`` import study results
without gaining a path to a candle. Enforced by
``tests/unit/test_ai_boundary.py``.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict

from backtool.research.metrics import WindowMetrics
from backtool.research.spec import ResearchSpec


class EventResult(BaseModel):
    """Metrics for every window of a single event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_time_utc: dt.datetime
    local_date: dt.date
    is_scheduled: bool
    windows: dict[str, WindowMetrics]

    @property
    def is_complete(self) -> bool:
        """True when every window produced usable numbers."""
        return all(window.is_usable for window in self.windows.values())

    @property
    def usable_window_count(self) -> int:
        return sum(1 for window in self.windows.values() if window.is_usable)

    def window(self, name: str) -> WindowMetrics:
        return self.windows[name]


class StudyResult(BaseModel):
    """Everything one run of a specification produced.

    Carries the spec itself so results are never separated from the assumptions
    that generated them.
    """

    model_config = ConfigDict(frozen=True)

    spec: ResearchSpec
    generated_at: dt.datetime
    events: tuple[EventResult, ...]

    @property
    def requested_count(self) -> int:
        return self.spec.event_count

    @property
    def complete_events(self) -> tuple[EventResult, ...]:
        return tuple(event for event in self.events if event.is_complete)

    def usable_events(self, window_name: str) -> tuple[EventResult, ...]:
        """Events whose named window produced usable numbers."""
        return tuple(
            event
            for event in self.events
            if window_name in event.windows and event.windows[window_name].is_usable
        )

    def coverage_note(self) -> str:
        """One-line honesty statement about what the numbers actually rest on.

        An aggregate reported over "20 events" that quietly used 17 is a lie, so
        this is rendered with every result.
        """
        found = len(self.events)
        complete = len(self.complete_events)
        parts = [f"{found} event(s) selected of {self.requested_count} requested"]
        if complete != found:
            parts.append(f"{complete} with all windows usable")
        return "; ".join(parts)
