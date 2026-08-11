"""Resolving a price at an exact instant, without look-ahead.

Implements ``docs/decisions/002-event-anchoring.md``. This module is small and
boring on purpose -- it is the single place where "the price at time T" is
defined, and the correctness of every downstream statistic rests on it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from backtool.core.time import ensure_utc


@dataclass(frozen=True)
class Anchor:
    """A price observation pinned to a requested instant."""

    #: Close time of the candle actually used.
    time: dt.datetime
    #: That candle's close price.
    price: float
    #: ``requested - time``. Never negative. Zero means the candle closed
    #: exactly at the requested instant.
    lag: dt.timedelta
    #: The instant that was asked for.
    requested: dt.datetime

    @property
    def lag_seconds(self) -> float:
        return self.lag.total_seconds()


def anchor_at(candles: pd.DataFrame, target: dt.datetime) -> Anchor | None:
    """Return the last price observable at or before ``target``.

    Takes the close of the last candle whose ``close_time`` is ``<= target``.
    That price was fully formed and public at ``target``, so using it cannot
    leak information from after ``target`` -- which is exactly the trap that
    makes naive event studies produce impressive nonsense.

    Args:
        candles: A normalised candle frame (see
            :func:`backtool.data.candles.normalise_candles`), sorted ascending
            by ``open_time``.
        target: The instant to price. Must be timezone-aware.

    Returns:
        An :class:`Anchor`, or ``None`` if no candle closed at or before
        ``target`` -- meaning the request predates available history. ``None``
        is a legitimate outcome the caller must handle, not an error.

    Raises:
        ValueError: if ``target`` is timezone-naive.
    """
    requested = ensure_utc(target)

    if candles.empty:
        return None

    # Use pandas' own dtype-aware binary search rather than converting both
    # sides to integers. Raw epoch integers require knowing the column's
    # resolution -- pandas 3.0 may store microseconds while pd.Timestamp.value
    # is always nanoseconds, and mixing the two silently returns the last
    # candle in the frame for every query rather than raising.
    position = int(
        candles["close_time"].searchsorted(pd.Timestamp(requested), side="right")
    ) - 1
    if position < 0:
        return None

    anchor_time = candles["close_time"].iloc[position].to_pydatetime()
    return Anchor(
        time=anchor_time,
        price=float(candles["close"].iloc[position]),
        lag=requested - anchor_time,
        requested=requested,
    )
