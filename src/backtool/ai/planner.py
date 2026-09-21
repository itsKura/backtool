"""Natural language in, executable research specification out.

The planner is the only component that reads a human question. It produces a
:class:`PlannedStudy` -- a specification plus the reasoning that justifies it --
which the deterministic engine then executes without further reference to the
model.

Nothing here touches market data. That is enforced architecturally: this package
does not import ``backtool.data``, so no path exists from a model response to a
candle.
"""

from __future__ import annotations

import datetime as dt
import logging

from backtool.ai.client import AIClient, AIError
from backtool.ai.models import PlannedStudy
from backtool.ai.prompts import PLANNER_SYSTEM
from backtool.core.types import EventType
from backtool.research.spec import ResearchSpec

logger = logging.getLogger(__name__)

MAX_WINDOWS = 8

#: Symbols the engine is scoped to. The planner prompt says BTCUSDT only, and
#: this enforces it -- a model can emit any alphanumeric string, and Binance
#: would happily return data for ETHUSDT, producing a study outside the scope
#: the project actually validated.
SUPPORTED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})

#: Event types with a calendar behind them. Enforced for the same reason as
#: symbols: structured outputs guarantee shape, not sense.
SUPPORTED_EVENT_TYPES = frozenset(kind.value for kind in EventType)


def plan_study(
    question: str,
    *,
    client: AIClient | None = None,
    as_of: dt.datetime | None = None,
) -> tuple[PlannedStudy, ResearchSpec]:
    """Turn a research question into a plan and a validated specification.

    Args:
        question: The user's question in plain language.
        client: Optional pre-built client, for testing.
        as_of: Analysis cutoff applied to the resulting spec. Pin it for a
            reproducible run; the model never chooses this.

    Returns:
        ``(plan, spec)``. The plan carries the interpretation and assumptions
        for display; the spec is what actually gets executed.

    Raises:
        ValueError: if the question is empty, or the model produced a
            specification the engine cannot execute.
        AIError: on any API failure.
    """
    text = question.strip()
    if not text:
        raise ValueError("Question must not be empty")

    resolved = client or AIClient()
    plan = resolved.structured(
        system=PLANNER_SYSTEM,
        user=f"Research question:\n\n{text}",
        schema=PlannedStudy,
    )

    _validate(plan)
    spec = plan.to_spec(as_of=as_of)

    logger.info(
        "Planned %d window(s) over %d %s event(s) at %s (spec %s)",
        len(spec.windows),
        spec.event_count,
        spec.event_type.value,
        spec.interval.value,
        spec.fingerprint,
    )
    return plan, spec


def _validate(plan: PlannedStudy) -> None:
    """Check the model's output before it reaches the engine.

    Structured outputs guarantee the *shape* of a response, not that its
    contents make sense. A model that invents a symbol or emits nine windows
    produces valid JSON, so the semantic limits are enforced here rather than
    trusted.
    """
    symbol = plan.symbol.strip().upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"The planner chose symbol {symbol!r}, which is outside V1 scope. "
            f"Supported: {sorted(SUPPORTED_SYMBOLS)}"
        )

    kind = plan.event_type.strip().upper()
    if kind not in SUPPORTED_EVENT_TYPES:
        raise ValueError(
            f"The planner chose event type {kind!r}, which has no calendar. "
            f"Supported: {sorted(SUPPORTED_EVENT_TYPES)}"
        )

    if not plan.windows:
        raise ValueError("The planner returned no windows to measure")
    if len(plan.windows) > MAX_WINDOWS:
        raise ValueError(
            f"The planner returned {len(plan.windows)} windows; the maximum is "
            f"{MAX_WINDOWS}. Each additional window is another comparison."
        )

    names = [window.name for window in plan.windows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"The planner reused window name(s): {duplicates}")


def render_plan(plan: PlannedStudy, spec: ResearchSpec) -> list[str]:
    """Format a plan for display before the study runs.

    Shown before execution on purpose: the reader gets the chance to reject a
    definition before spending time on results computed under it.
    """
    lines = [f"Understood as: {plan.restated_question}", ""]

    if plan.interpretation:
        lines.append("Ambiguous terms resolved:")
        lines.extend(f"  {entry}" for entry in plan.interpretation)
        lines.append("")

    if plan.assumptions:
        lines.append("Assumed (not specified in the question):")
        lines.extend(f"  {entry}" for entry in plan.assumptions)
        lines.append("")

    lines.append("Windows:")
    for window in plan.windows:
        lines.append(f"  {window.name:<20} {window.start} -> {window.end}")
        lines.append(f"  {'':<20} {window.rationale}")

    lines.extend(["", f"Spec fingerprint: {spec.fingerprint}"])
    return lines


__all__ = ["AIError", "plan_study", "render_plan"]
