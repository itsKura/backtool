"""AI layer: research planning and results interpretation.

Architectural constraint, enforced by ``tests/unit/test_ai_boundary.py``: this
package must never import ``backtool.data``. The model emits specifications and
consumes computed results. It cannot reach a candle, so it cannot invent a
number that looks like one.
"""

from backtool.ai.client import AIClient, AIError, AINotConfiguredError, is_configured
from backtool.ai.interpreter import build_payload, interpret_study, render_interpretation
from backtool.ai.models import Interpretation, PlannedStudy, PlannedWindow
from backtool.ai.planner import plan_study, render_plan

__all__ = [
    "AIClient",
    "AIError",
    "AINotConfiguredError",
    "Interpretation",
    "PlannedStudy",
    "PlannedWindow",
    "build_payload",
    "interpret_study",
    "is_configured",
    "plan_study",
    "render_interpretation",
    "render_plan",
]
