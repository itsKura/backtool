"""Research engine: specification -> windows -> metrics -> aggregation."""

from backtool.research.aggregate import WindowAggregate, aggregate_study, aggregate_window
from backtool.research.anchoring import Anchor, anchor_at
from backtool.research.metrics import (
    Direction,
    WindowMetrics,
    WindowStatus,
    compute_window_metrics,
)
from backtool.research.runner import EventResult, StudyResult, run_study
from backtool.research.spec import ResearchSpec
from backtool.research.windows import (
    DEFAULT_WINDOWS,
    ResolvedWindow,
    WindowSpec,
    parse_offset,
)

__all__ = [
    "DEFAULT_WINDOWS",
    "Anchor",
    "Direction",
    "EventResult",
    "ResearchSpec",
    "ResolvedWindow",
    "StudyResult",
    "WindowAggregate",
    "WindowMetrics",
    "WindowSpec",
    "WindowStatus",
    "aggregate_study",
    "aggregate_window",
    "anchor_at",
    "compute_window_metrics",
    "parse_offset",
    "run_study",
]
