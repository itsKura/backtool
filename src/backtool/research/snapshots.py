"""Recording what the market looked like before a release, at the time.

A setup computed for an upcoming release is a reading taken now. Once the
release happens, that reading becomes unrecoverable in one specific way: the
*numbers* can be recomputed from candles forever, but the fact that this is what
Backtool was showing a reader beforehand cannot. Recomputing later would produce
the same figures from a system that may have changed its thresholds, its
lookbacks, or its bucket boundaries in the meantime.

So a snapshot stores the reading **and the rule that produced it**. Comparing a
snapshot against a fresh computation later answers two different questions:
whether the market moved between the snapshot and the release, and whether the
classification logic changed underneath it.

Nothing writes here automatically. A page view is not an event worth recording,
and recording one per view would fill the table with duplicates that say nothing
about what a reader saw. Capture is an explicit call.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from backtool.core.time import UTC, ensure_utc
from backtool.research.regime import (
    MA_DAYS,
    TREND_BEAR_PCT,
    TREND_BULL_PCT,
    VOL_HIGH_PCTL,
    VOL_LOW_PCTL,
    VOL_PERCENTILE_LOOKBACK_DAYS,
    MarketSetup,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS setup_snapshots (
    event_id     TEXT NOT NULL,
    captured_at  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    hours_before REAL NOT NULL,
    setup_json   TEXT NOT NULL,
    rules_json   TEXT NOT NULL,
    PRIMARY KEY (event_id, captured_at, symbol)
);
"""


def current_rules() -> dict[str, float | int]:
    """The thresholds a snapshot was classified under.

    Stored with every snapshot. A reading of "bearish" means nothing a year
    later without the number that made it bearish, and these constants are
    research choices that may well be revised.
    """
    return {
        "trend_bull_pct": TREND_BULL_PCT,
        "trend_bear_pct": TREND_BEAR_PCT,
        "vol_low_pctl": VOL_LOW_PCTL,
        "vol_high_pctl": VOL_HIGH_PCTL,
        "vol_lookback_days": VOL_PERCENTILE_LOOKBACK_DAYS,
        "ma_days": MA_DAYS,
    }


class SetupSnapshot(BaseModel):
    """One recorded reading, with when it was taken and under what rules."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    captured_at: dt.datetime
    symbol: str
    #: Hours between the capture and the release. Negative would mean captured
    #: after the fact, which :func:`record_snapshot` refuses.
    hours_before: float
    setup: MarketSetup
    rules: dict[str, float]

    @property
    def rules_match_current(self) -> bool:
        """Whether today's thresholds are the ones this was classified under."""
        return self.rules == {k: float(v) for k, v in current_rules().items()}


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(_SCHEMA)
    return connection


def record_snapshot(
    path: Path,
    *,
    event_id: str,
    event_time: dt.datetime,
    symbol: str,
    setup: MarketSetup,
    captured_at: dt.datetime | None = None,
) -> SetupSnapshot | None:
    """Store a pre-release reading.

    Returns:
        The stored snapshot, or ``None`` if the release had already happened.
        Recording a reading taken after the event would defeat the purpose --
        the point is a record of what was shown *beforehand* -- so it is
        refused rather than stored with a negative lead time.
    """
    moment = ensure_utc(captured_at) if captured_at else dt.datetime.now(tz=UTC)
    release = ensure_utc(event_time)
    lead = (release - moment).total_seconds() / 3600.0
    if lead <= 0:
        return None

    snapshot = SetupSnapshot(
        event_id=event_id,
        captured_at=moment,
        symbol=symbol,
        hours_before=lead,
        setup=setup,
        rules={k: float(v) for k, v in current_rules().items()},
    )

    connection = _connect(path)
    try:
        connection.execute(
            "INSERT OR REPLACE INTO setup_snapshots "
            "(event_id, captured_at, symbol, hours_before, setup_json, rules_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                snapshot.event_id,
                snapshot.captured_at.isoformat(),
                snapshot.symbol,
                snapshot.hours_before,
                snapshot.setup.model_dump_json(),
                json.dumps(snapshot.rules, sort_keys=True),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return snapshot


def load_snapshots(path: Path, event_id: str) -> list[SetupSnapshot]:
    """Every reading recorded for one release, nearest to the release last."""
    if not path.exists():
        return []

    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT event_id, captured_at, symbol, hours_before, setup_json, "
            "rules_json FROM setup_snapshots WHERE event_id = ? "
            "ORDER BY hours_before DESC",
            (event_id,),
        ).fetchall()
    finally:
        connection.close()

    return [
        SetupSnapshot(
            event_id=row[0],
            captured_at=dt.datetime.fromisoformat(row[1]),
            symbol=row[2],
            hours_before=row[3],
            setup=MarketSetup.model_validate_json(row[4]),
            rules=json.loads(row[5]),
        )
        for row in rows
    ]


__all__ = [
    "SetupSnapshot",
    "current_rules",
    "load_snapshots",
    "record_snapshot",
]
