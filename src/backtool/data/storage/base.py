"""The storage contract the rest of the system codes against.

Storage is an implementation detail behind this Protocol, not an architectural
commitment. V1 ships a SQLite implementation; see
``docs/decisions/001-candle-storage.md`` and ``004-sqlite-over-parquet.md``.
Swapping in PostgreSQL or TimescaleDB later is a new class plus one line of
wiring, not a rewrite.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import pandas as pd

from backtool.core.types import Interval
from backtool.data.storage.coverage import CoverageIndex


class CandleRepository(Protocol):
    """Persistent storage for normalised candles plus fetch-coverage metadata."""

    def read(
        self, symbol: str, interval: Interval, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        """Return stored candles whose ``open_time`` falls in ``[start, end)``.

        Returns an empty normalised frame when nothing is stored, never
        ``None``.
        """
        ...

    def store(
        self,
        symbol: str,
        interval: Interval,
        candles: pd.DataFrame,
        covered_start: dt.datetime,
        covered_end: dt.datetime,
    ) -> None:
        """Persist candles and mark ``[covered_start, covered_end)`` as fetched.

        Both happen atomically. If the candles were written but the coverage
        record was not, the next run would re-download data it already has; if
        coverage were written but the candles were not, the range would be
        permanently skipped and research would silently run on missing data.
        The second failure is much worse, so the two must not be separable.

        Coverage is recorded even when ``candles`` is empty: an empty response
        means the market genuinely had nothing in that range -- a pre-listing
        period or an exchange outage -- and asking again tomorrow will not
        change that.

        Must be idempotent: storing the same candles twice leaves one copy.
        """
        ...

    def coverage(self, symbol: str, interval: Interval) -> CoverageIndex:
        """Return the ranges already fetched for this symbol and interval."""
        ...
