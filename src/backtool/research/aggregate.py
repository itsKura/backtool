"""Cross-event aggregation.

Turns N per-event observations into the summary statistics a reader actually
wants: central tendency, dispersion, how often the move was up, and the shape of
the distribution.

Every aggregate reports the sample size it was computed from. A mean over four
observations and a mean over forty look identical on a page, and only one of
them means anything.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from backtool.research.metrics import Direction
from backtool.research.results import StudyResult

#: Below this many observations, a summary statistic is too noisy to lead with.
#: Not a hard cut -- the aggregate is still produced, but flagged.
MIN_RELIABLE_SAMPLE = 10


class WindowAggregate(BaseModel):
    """Summary statistics for one window across all usable events."""

    model_config = ConfigDict(frozen=True)

    window: str
    #: Events whose metrics entered these statistics.
    sample_size: int
    #: Events selected but excluded from this window for lack of data.
    excluded: int

    mean_return_pct: float
    median_return_pct: float
    stdev_return_pct: float | None
    min_return_pct: float
    max_return_pct: float

    #: Share of events with a positive return, in percent.
    up_rate_pct: float
    down_rate_pct: float
    flat_count: int
    #: The same thing as a count. Carried rather than recovered from the rate,
    #: because rounding a percentage back into a count over a five-event sample
    #: produces the wrong integer often enough to matter, and "4 of 5" printed
    #: beside "100%" would be worse than showing neither.
    up_count: int
    down_count: int

    p10_return_pct: float
    p25_return_pct: float
    p75_return_pct: float
    p90_return_pct: float

    mean_abs_move_pct: float
    mean_mfe_pct: float
    mean_mae_pct: float
    mean_range_pct: float
    #: Median rather than mean high-to-low range. One violent release drags a
    #: mean range somewhere no typical release went, and "how far does this
    #: usually travel" is a question about the typical case.
    median_range_pct: float
    mean_realized_vol_pct: float | None

    #: Share of events whose absolute move cleared a threshold, in percent.
    #: A median of +0.1% over events that individually moved 2% either way is a
    #: different fact from a median of +0.1% over events that barely moved, and
    #: these two are what separate them.
    move_over_0_5_pct: float
    move_over_1_pct: float
    move_over_0_5_count: int
    move_over_1_count: int

    #: Mean share of expected candles actually present, in percent.
    mean_coverage_pct: float | None

    @property
    def is_reliable(self) -> bool:
        """Whether the sample is large enough to lead with."""
        return self.sample_size >= MIN_RELIABLE_SAMPLE

    @property
    def caveat(self) -> str | None:
        if self.sample_size == 0:
            return "no usable observations"
        if not self.is_reliable:
            return (
                f"only {self.sample_size} observation(s) - treat as indicative, "
                "not established"
            )
        return None


def aggregate_study(study: StudyResult) -> dict[str, WindowAggregate]:
    """Aggregate every window of a study across its events.

    Args:
        study: A completed run.

    Returns:
        One :class:`WindowAggregate` per window in the spec, keyed by name.
        Windows with no usable observations still appear, with
        ``sample_size == 0``, so a reader sees that they were attempted.
    """
    return {
        window_spec.name: aggregate_window(study, window_spec.name)
        for window_spec in study.spec.windows
    }


def aggregate_window(study: StudyResult, window_name: str) -> WindowAggregate:
    """Aggregate a single named window across all events in a study."""
    usable = [
        event.windows[window_name]
        for event in study.usable_events(window_name)
    ]
    excluded = len(study.events) - len(usable)

    if not usable:
        return _empty_aggregate(window_name, excluded)

    returns = np.array([m.return_pct for m in usable], dtype="float64")
    moves = np.array([m.abs_move_pct for m in usable], dtype="float64")
    directions = [m.direction for m in usable]
    coverages = [m.coverage_pct for m in usable if m.coverage_pct is not None]
    vols = [m.realized_vol_pct for m in usable if m.realized_vol_pct is not None]

    return WindowAggregate(
        window=window_name,
        sample_size=len(usable),
        excluded=excluded,
        mean_return_pct=float(np.mean(returns)),
        median_return_pct=float(np.median(returns)),
        # Sample standard deviation (ddof=1): we are describing a sample drawn
        # from a wider population of events, not the population itself. Needs at
        # least two observations.
        stdev_return_pct=float(np.std(returns, ddof=1)) if len(returns) > 1 else None,
        min_return_pct=float(np.min(returns)),
        max_return_pct=float(np.max(returns)),
        up_rate_pct=_share(directions, Direction.UP),
        down_rate_pct=_share(directions, Direction.DOWN),
        flat_count=sum(1 for d in directions if d is Direction.FLAT),
        up_count=sum(1 for d in directions if d is Direction.UP),
        down_count=sum(1 for d in directions if d is Direction.DOWN),
        p10_return_pct=float(np.percentile(returns, 10)),
        p25_return_pct=float(np.percentile(returns, 25)),
        p75_return_pct=float(np.percentile(returns, 75)),
        p90_return_pct=float(np.percentile(returns, 90)),
        mean_abs_move_pct=float(np.mean(moves)),
        move_over_0_5_pct=float(np.mean(moves >= 0.5) * 100.0),
        move_over_1_pct=float(np.mean(moves >= 1.0) * 100.0),
        move_over_0_5_count=int(np.count_nonzero(moves >= 0.5)),
        move_over_1_count=int(np.count_nonzero(moves >= 1.0)),
        median_range_pct=float(np.median([m.range_pct for m in usable])),
        mean_mfe_pct=float(np.mean([m.mfe_pct for m in usable])),
        mean_mae_pct=float(np.mean([m.mae_pct for m in usable])),
        mean_range_pct=float(np.mean([m.range_pct for m in usable])),
        mean_realized_vol_pct=float(np.mean(vols)) if vols else None,
        mean_coverage_pct=float(np.mean(coverages)) if coverages else None,
    )


def _share(directions: list[Direction | None], target: Direction) -> float:
    if not directions:
        return 0.0
    return sum(1 for d in directions if d is target) / len(directions) * 100.0


def _empty_aggregate(window_name: str, excluded: int) -> WindowAggregate:
    zeros = dict.fromkeys(
        (
            "mean_return_pct",
            "median_return_pct",
            "min_return_pct",
            "max_return_pct",
            "up_rate_pct",
            "down_rate_pct",
            "p10_return_pct",
            "p25_return_pct",
            "p75_return_pct",
            "p90_return_pct",
            "mean_abs_move_pct",
            "mean_mfe_pct",
            "mean_mae_pct",
            "mean_range_pct",
            "median_range_pct",
            "move_over_0_5_pct",
            "move_over_1_pct",
        ),
        0.0,
    )
    return WindowAggregate(
        window=window_name,
        sample_size=0,
        excluded=excluded,
        stdev_return_pct=None,
        flat_count=0,
        up_count=0,
        down_count=0,
        move_over_0_5_count=0,
        move_over_1_count=0,
        mean_realized_vol_pct=None,
        mean_coverage_pct=None,
        **zeros,
    )
