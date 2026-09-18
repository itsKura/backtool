"""Schemas for what the language model is allowed to produce.

These are deliberately separate from :class:`~backtool.research.spec.ResearchSpec`.
The spec is the minimal set of parameters that change a number; a *plan* is the
spec plus the reasoning that produced it -- which definitions were chosen for
ambiguous terms, and what had to be assumed. Keeping them apart stops
explanation text leaking into the thing that gets fingerprinted and
reproduced.

``PlannedStudy.to_spec()`` is the bridge, and it is the only place a model's
output becomes something the deterministic engine will act on.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field

from backtool.core.types import EventType, Interval
from backtool.research.spec import ResearchSpec
from backtool.research.windows import WindowSpec


class PlannedWindow(BaseModel):
    """One window the planner chose, with its reason for existing."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="snake_case identifier, e.g. pre_event_24h")
    start: str = Field(description="Offset from the event, e.g. '-24h', '0h', '-90m'")
    end: str = Field(description="Offset from the event; must be after start")
    rationale: str = Field(description="One sentence: what this window is measuring")


class PlannedStudy(BaseModel):
    """A research question turned into an executable, explicit specification.

    ``interpretation`` is the load-bearing field. Financial language is
    underspecified -- "just before the announcement", "a big move", "recently"
    all have several defensible readings that produce different numbers. The
    planner must resolve each one into a concrete rule and say so here, so the
    reader can disagree with the definition rather than silently inherit it.
    """

    model_config = ConfigDict(frozen=True)

    restated_question: str = Field(
        description="The question as the planner understood it, in one sentence"
    )
    symbol: str = Field(description="Exchange symbol; only BTCUSDT is supported")
    event_type: str = Field(description="Event type; only FOMC is supported")
    event_count: int = Field(ge=1, le=200, description="How many recent events to analyse")
    interval: str = Field(description="Candle interval: 5m, 15m, or 1h")
    windows: list[PlannedWindow] = Field(
        description="Windows to measure, 1-8 of them, each anchored to the event"
    )
    interpretation: list[str] = Field(
        description=(
            "Every ambiguous term in the question and the explicit rule chosen "
            "for it. One entry per decision, phrased as 'X = <testable rule>'. "
            "Empty only if the question contained no ambiguity at all."
        )
    )
    assumptions: list[str] = Field(
        description=(
            "Anything the question did not specify that had to be filled in, "
            "such as an unstated interval or event count."
        )
    )

    def to_spec(self, *, as_of: dt.datetime | None = None) -> ResearchSpec:
        """Convert to the deterministic engine's specification.

        Raises:
            ValueError: if the planner produced a symbol, event type, interval,
                or window that the engine does not support. Validation happens
                here rather than being trusted from the model.
        """
        return ResearchSpec(
            symbol=self.symbol,
            event_type=EventType(self.event_type.upper()),
            event_count=self.event_count,
            interval=Interval(self.interval.lower()),
            windows=tuple(
                WindowSpec(name=window.name, start=window.start, end=window.end)
                for window in self.windows
            ),
            as_of=as_of,
        )


class Interpretation(BaseModel):
    """The model's reading of results it did not compute.

    Structured rather than free prose so the same object renders into the
    terminal, the HTML report, or anything added later, and so each claim can
    be checked against the statistics it came from.
    """

    model_config = ConfigDict(frozen=True)

    headline: str = Field(
        description="One sentence stating the single most notable finding, with its number"
    )
    findings: list[str] = Field(
        description=(
            "2-5 observations. Every number cited must appear verbatim in the "
            "supplied results. No estimates, no derived figures."
        )
    )
    caveats: list[str] = Field(
        description=(
            "What limits these conclusions: sample size, excluded events, "
            "multiple comparisons, regime changes over the period."
        )
    )
    suggested_followups: list[str] = Field(
        description="1-3 concrete follow-up studies this system could actually run"
    )
