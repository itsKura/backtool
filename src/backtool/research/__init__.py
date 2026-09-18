"""Research engine: specification -> windows -> metrics -> aggregation.

Everything exported here is pure: given results or candles already in memory, it
computes. The one component that performs I/O is :mod:`backtool.research.runner`,
which needs the market-data layer to fetch candles, and it is deliberately **not**
re-exported.

Importing any submodule executes this file, so re-exporting the runner would give
every importer of ``backtool.research.spec`` -- including ``backtool.ai`` -- a
transitive path to ``backtool.data``. Import it explicitly instead::

    from backtool.research.runner import run_study

Enforced by ``tests/unit/test_ai_boundary.py``.
"""

from backtool.research.aggregate import WindowAggregate, aggregate_study, aggregate_window
from backtool.research.anchoring import Anchor, anchor_at
from backtool.research.metrics import (
    Direction,
    WindowMetrics,
    WindowStatus,
    compute_window_metrics,
)
from backtool.research.results import EventResult, StudyResult
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
]
