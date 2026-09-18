"""Structured results in, explanation out.

The interpreter receives statistics that deterministic Python already computed
and explains what they mean. It never sees a candle, never computes a number,
and cannot reach the market-data layer -- this package does not import
``backtool.data``, and a test enforces that.

That constraint is the product: an explanation that cannot fabricate the figures
it is explaining.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backtool.ai.client import AIClient
from backtool.ai.models import Interpretation
from backtool.ai.prompts import INTERPRETER_SYSTEM
from backtool.research.aggregate import WindowAggregate, aggregate_study
from backtool.research.results import StudyResult

logger = logging.getLogger(__name__)


def interpret_study(
    study: StudyResult,
    *,
    client: AIClient | None = None,
    question: str | None = None,
) -> Interpretation:
    """Explain a completed study.

    Args:
        study: The finished run.
        client: Optional pre-built client, for testing.
        question: The original question, if there was one, so the explanation
            answers what was actually asked.

    Returns:
        A structured :class:`Interpretation`.

    Raises:
        AIError: on any API failure or refusal.
    """
    resolved = client or AIClient()
    payload = build_payload(study, question=question)

    logger.info("Interpreting study %s (%d events)", study.spec.fingerprint, len(study.events))
    return resolved.structured(
        system=INTERPRETER_SYSTEM,
        user=json.dumps(payload, indent=2, default=str),
        schema=Interpretation,
    )


def build_payload(study: StudyResult, *, question: str | None = None) -> dict[str, Any]:
    """Assemble exactly what the model is allowed to see.

    Deliberately explicit rather than dumping the whole result object: this
    function is the boundary that decides what reaches the model, so what is
    excluded should be as reviewable as what is included. Raw candles are not
    here and never will be.
    """
    aggregates = aggregate_study(study)

    return {
        "question": question,
        "assumptions": dict(study.spec.describe_items()),
        "coverage": {
            "events_requested": study.requested_count,
            "events_selected": len(study.events),
            "events_with_all_windows_usable": len(study.complete_events),
            "note": study.coverage_note(),
        },
        "aggregates": {
            name: _aggregate_payload(aggregate) for name, aggregate in aggregates.items()
        },
        "per_event": [
            {
                "event_id": event.event_id,
                "date": event.local_date.isoformat(),
                "scheduled": event.is_scheduled,
                "windows": {
                    name: (
                        {
                            "return_pct": round(metrics.return_pct, 4)
                            if metrics.return_pct is not None
                            else None,
                            "direction": metrics.direction.value
                            if metrics.direction
                            else None,
                            "coverage_pct": round(metrics.coverage_pct, 1)
                            if metrics.coverage_pct is not None
                            else None,
                        }
                        if metrics.is_usable
                        else {"status": metrics.status.value, "reason": metrics.reason}
                    )
                    for name, metrics in event.windows.items()
                },
            }
            for event in study.events
        ],
    }


def _aggregate_payload(aggregate: WindowAggregate) -> dict[str, Any]:
    """One window's statistics, rounded to what is meaningful to report."""
    if aggregate.sample_size == 0:
        return {
            "sample_size": 0,
            "excluded": aggregate.excluded,
            "note": "no usable observations",
        }

    def rounded(value: float | None, places: int = 3) -> float | None:
        return round(value, places) if value is not None else None

    return {
        "sample_size": aggregate.sample_size,
        "excluded": aggregate.excluded,
        "reliable_sample": aggregate.is_reliable,
        "caveat": aggregate.caveat,
        "mean_return_pct": rounded(aggregate.mean_return_pct),
        "median_return_pct": rounded(aggregate.median_return_pct),
        "stdev_return_pct": rounded(aggregate.stdev_return_pct),
        "min_return_pct": rounded(aggregate.min_return_pct),
        "max_return_pct": rounded(aggregate.max_return_pct),
        "up_rate_pct": rounded(aggregate.up_rate_pct, 1),
        "down_rate_pct": rounded(aggregate.down_rate_pct, 1),
        "p10_return_pct": rounded(aggregate.p10_return_pct),
        "p25_return_pct": rounded(aggregate.p25_return_pct),
        "p75_return_pct": rounded(aggregate.p75_return_pct),
        "p90_return_pct": rounded(aggregate.p90_return_pct),
        "mean_abs_move_pct": rounded(aggregate.mean_abs_move_pct),
        "mean_mfe_pct": rounded(aggregate.mean_mfe_pct),
        "mean_mae_pct": rounded(aggregate.mean_mae_pct),
        "mean_range_pct": rounded(aggregate.mean_range_pct),
        "mean_realized_vol_pct": rounded(aggregate.mean_realized_vol_pct),
        "mean_coverage_pct": rounded(aggregate.mean_coverage_pct, 1),
    }


def render_interpretation(interpretation: Interpretation) -> list[str]:
    """Format an interpretation for terminal output."""
    lines = [interpretation.headline, ""]

    if interpretation.findings:
        lines.append("Findings:")
        lines.extend(f"  - {finding}" for finding in interpretation.findings)
        lines.append("")

    if interpretation.caveats:
        lines.append("Caveats:")
        lines.extend(f"  - {caveat}" for caveat in interpretation.caveats)
        lines.append("")

    if interpretation.suggested_followups:
        lines.append("Worth running next:")
        lines.extend(f"  - {item}" for item in interpretation.suggested_followups)

    return lines


__all__ = ["build_payload", "interpret_study", "render_interpretation"]
