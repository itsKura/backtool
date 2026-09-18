"""SQLite implementation of :class:`CandleRepository`.

One database file holds every symbol and interval, plus the coverage metadata
that records which ranges have been fetched. Keeping both in one file is the
point: a fetch commits its candles and its coverage record in a single
transaction, so the two cannot disagree.

Timestamps are stored as integer epoch milliseconds -- the exchange's own unit.
That avoids any date parsing on the storage boundary, sorts correctly as
integers, and sidesteps the timestamp-resolution ambiguity that bit us once
already (see ``CANDLE_TIME_DTYPE``).
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from backtool.core.time import ensure_utc, utc_ms
from backtool.core.types import Interval
from backtool.data.candles import empty_candles, normalise_candles
from backtool.data.storage.coverage import CoverageIndex, TimeRange

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol       TEXT    NOT NULL,
    interval     TEXT    NOT NULL,
    open_time_ms INTEGER NOT NULL,
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    volume       REAL    NOT NULL,
    PRIMARY KEY (symbol, interval, open_time_ms)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS coverage (
    symbol   TEXT    NOT NULL,
    interval TEXT    NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms   INTEGER NOT NULL,
    PRIMARY KEY (symbol, interval, start_ms)
);
"""


class SQLiteCandleRepository:
    """Stores candles and fetch coverage in a single SQLite database.

    ``WITHOUT ROWID`` on ``candles`` makes the primary key the clustering key,
    so a range scan over ``(symbol, interval, open_time_ms)`` -- the only read
    pattern this system has -- walks contiguous storage instead of hopping
    through an index into a heap.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            yield connection
        finally:
            connection.close()

    # --- CandleRepository -------------------------------------------------

    def read(
        self, symbol: str, interval: Interval, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if start_utc >= end_utc:
            return empty_candles(interval)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT open_time_ms, open, high, low, close, volume
                  FROM candles
                 WHERE symbol = ? AND interval = ?
                   AND open_time_ms >= ? AND open_time_ms < ?
                 ORDER BY open_time_ms
                """,
                (symbol.upper(), interval.value, utc_ms(start_utc), utc_ms(end_utc)),
            ).fetchall()

        if not rows:
            return empty_candles(interval)

        frame = pd.DataFrame(
            rows, columns=["open_time_ms", "open", "high", "low", "close", "volume"]
        )
        frame["open_time"] = pd.to_datetime(frame.pop("open_time_ms"), unit="ms", utc=True)
        return normalise_candles(frame, interval)

    def store(
        self,
        symbol: str,
        interval: Interval,
        candles: pd.DataFrame,
        covered_start: dt.datetime,
        covered_end: dt.datetime,
    ) -> None:
        normalised_symbol = symbol.upper()
        new_range = TimeRange(ensure_utc(covered_start), ensure_utc(covered_end))

        # Converted via total_seconds() rather than raw epoch integers so the
        # result does not depend on the frame's backing timestamp resolution.
        open_time_ms: list[int] = (
            ((candles["open_time"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds() * 1000)
            .round()
            .astype("int64")
            .tolist()
        )
        candle_rows = [
            (normalised_symbol, interval.value, timestamp, open_, high, low, close, volume)
            for timestamp, open_, high, low, close, volume in zip(
                open_time_ms,
                candles["open"].astype("float64").tolist(),
                candles["high"].astype("float64").tolist(),
                candles["low"].astype("float64").tolist(),
                candles["close"].astype("float64").tolist(),
                candles["volume"].astype("float64").tolist(),
                strict=True,
            )
        ]

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if candle_rows:
                    # INSERT OR REPLACE makes re-fetching a range idempotent and
                    # lets a corrected candle overwrite a stale one.
                    connection.executemany(
                        "INSERT OR REPLACE INTO candles "
                        "(symbol, interval, open_time_ms, open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        candle_rows,
                    )

                merged = self._read_coverage(connection, normalised_symbol, interval).with_range(
                    new_range.start, new_range.end
                )
                connection.execute(
                    "DELETE FROM coverage WHERE symbol = ? AND interval = ?",
                    (normalised_symbol, interval.value),
                )
                connection.executemany(
                    "INSERT INTO coverage (symbol, interval, start_ms, end_ms) VALUES (?, ?, ?, ?)",
                    [
                        (normalised_symbol, interval.value, utc_ms(r.start), utc_ms(r.end))
                        for r in merged.ranges
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        logger.debug(
            "Stored %d candle(s) and coverage %s for %s %s",
            len(candle_rows),
            new_range,
            normalised_symbol,
            interval.value,
        )

    def coverage(self, symbol: str, interval: Interval) -> CoverageIndex:
        with self._connect() as connection:
            return self._read_coverage(connection, symbol.upper(), interval)

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _read_coverage(
        connection: sqlite3.Connection, symbol: str, interval: Interval
    ) -> CoverageIndex:
        rows = connection.execute(
            "SELECT start_ms, end_ms FROM coverage WHERE symbol = ? AND interval = ? "
            "ORDER BY start_ms",
            (symbol, interval.value),
        ).fetchall()
        return CoverageIndex(
            tuple(
                TimeRange(
                    dt.datetime.fromtimestamp(start / 1000, tz=dt.UTC),
                    dt.datetime.fromtimestamp(end / 1000, tz=dt.UTC),
                )
                for start, end in rows
            )
        )

    # --- introspection ----------------------------------------------------

    def candle_count(self, symbol: str, interval: Interval) -> int:
        """How many candles are stored for this symbol and interval."""
        with self._connect() as connection:
            (count,) = connection.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol = ? AND interval = ?",
                (symbol.upper(), interval.value),
            ).fetchone()
        return int(count)

    def symbols(self) -> list[tuple[str, str]]:
        """Distinct ``(symbol, interval)`` pairs present in the cache."""
        with self._connect() as connection:
            return [
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT DISTINCT symbol, interval FROM candles ORDER BY symbol, interval"
                ).fetchall()
            ]

    @property
    def path(self) -> Path:
        return self._path
