"""Research engine: specification -> windows -> metrics -> aggregation."""

from backtool.research.anchoring import Anchor, anchor_at
from backtool.research.metrics import (
    Direction,
    WindowMetrics,
    WindowStatus,
    compute_window_metrics,
)
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
    "ResolvedWindow",
    "WindowMetrics",
    "WindowSpec",
    "WindowStatus",
    "anchor_at",
    "compute_window_metrics",
    "parse_offset",
]
