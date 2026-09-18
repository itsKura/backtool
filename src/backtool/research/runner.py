"""Running a research specification across many events.

This is the orchestration layer: it selects events, asks the data service for
the candles each one needs, computes window metrics, and collects the results
with an explicit account of what could not be computed and why.
"""

from __future__ import annotations

import datetime as dt
import logging

from backtool.core.time import UTC, ensure_utc
from backtool.data.service import MarketDataService
from backtool.events import MarketEvent, select_last_n
from backtool.research.metrics import WindowStatus, compute_window_metrics
from backtool.research.results import EventResult, StudyResult
from backtool.research.spec import ResearchSpec

logger = logging.getLogger(__name__)


def run_study(
    spec: ResearchSpec,
    service: MarketDataService,
    calendar: list[MarketEvent],
    *,
    now: dt.datetime | None = None,
) -> StudyResult:
    """Execute a specification and return per-event metrics.

    Args:
        spec: The research question, fully specified.
        service: Cache-aware market data source.
        calendar: Candidate events, typically the whole FOMC calendar.
        now: Override for the current instant, for deterministic runs.

    Returns:
        A :class:`StudyResult`. Events that could not be computed are still
        present, with per-window ``status`` explaining why, so the caller can
        report what was excluded rather than silently shrinking the sample.
    """
    current = ensure_utc(now) if now is not None else dt.datetime.now(tz=UTC)
    cutoff = ensure_utc(spec.as_of) if spec.as_of is not None else current

    selected = select_last_n(
        calendar, spec.event_count, before=cutoff, event_type=spec.event_type
    )
    logger.info(
        "Running spec %s over %d event(s) at %s",
        spec.fingerprint,
        len(selected),
        spec.interval.value,
    )

    results = [
        _run_single_event(spec, service, event, current)
        for event in selected
    ]

    return StudyResult(
        spec=spec,
        generated_at=current,
        events=tuple(results),
    )


def _run_single_event(
    spec: ResearchSpec,
    service: MarketDataService,
    event: MarketEvent,
    now: dt.datetime,
) -> EventResult:
    """Fetch the candles one event needs and compute all its windows."""
    anchor = event.timestamp_utc
    candles = service.get_candles(
        spec.symbol,
        spec.interval,
        anchor + spec.earliest_offset,
        anchor + spec.latest_offset,
        now=now,
    )

    windows = {
        window_spec.name: compute_window_metrics(
            candles, window_spec.resolve(anchor), spec.interval
        )
        for window_spec in spec.windows
    }

    unusable = [name for name, metrics in windows.items() if not metrics.is_usable]
    if unusable:
        logger.info(
            "%s: %d/%d window(s) unusable (%s)",
            event.event_id,
            len(unusable),
            len(windows),
            ", ".join(unusable),
        )

    return EventResult(
        event_id=event.event_id,
        event_time_utc=anchor,
        local_date=event.local_date,
        is_scheduled=event.is_scheduled,
        windows=windows,
    )


__all__ = ["EventResult", "StudyResult", "WindowStatus", "run_study"]
