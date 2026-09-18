"""Normalised candle (OHLCV) representation and its integrity checks.

Interval convention
-------------------
A candle covers the half-open interval ``[open_time, close_time)`` where
``close_time == open_time + interval``. The ``close`` price becomes known to the
market **at** ``close_time``, not before.

Binance reports ``close_time`` as ``open_time + interval - 1ms``, which creates
awkward off-by-one-millisecond arithmetic at every boundary. We normalise it
away: ``close_time`` here is the exact instant the candle closes, so consecutive
candles share a boundary (``close_time[i] == open_time[i+1]``).

Everything downstream -- anchoring, window selection, metrics -- depends on this
convention. It is the reason an FOMC announcement at exactly 19:00:00 UTC can be
anchored with zero staleness: the candle ``[18:55, 19:00)`` closes precisely
then.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

import pandas as pd

from backtool.core.time import expected_candle_count as _expected_candle_count
from backtool.core.types import Interval

#: The one timestamp representation used throughout the system.
#:
#: pandas infers a resolution from its input -- pandas 3.0 produces
#: ``datetime64[us]`` from Python ``datetime`` objects where pandas 2.x produced
#: ``datetime64[ns]``. Any code that converts timestamps to integers must know
#: which unit it is getting, so we pin it here rather than inheriting whatever
#: was inferred upstream. See ``_check_timestamps`` for why this matters.
#:
#: Declared ``Final`` so type checkers narrow it to a string literal, which is
#: what ``Series.astype`` overloads match against.
CANDLE_TIME_DTYPE: Final = "datetime64[ns, UTC]"

#: Columns a caller must supply to :func:`normalise_candles`.
REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")

#: Columns present on a normalised candle frame, in order.
CANDLE_COLUMNS = ("open_time", "close_time", "open", "high", "low", "close", "volume")

_PRICE_COLUMNS = ("open", "high", "low", "close")


class CandleDataError(ValueError):
    """Candle data is structurally invalid or internally inconsistent."""


def normalise_candles(frame: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    """Validate, sort, and complete a raw candle frame.

    Single entry point for candles from any source -- the Binance client, the
    Parquet cache, or a test fixture. Centralising it means every candle in the
    system has passed the same checks.

    Args:
        frame: Must contain :data:`REQUIRED_COLUMNS`. ``open_time`` must be
            timezone-aware. Row order does not matter.
        interval: The interval these candles represent, used to derive
            ``close_time`` and to check spacing.

    Returns:
        A new frame with :data:`CANDLE_COLUMNS`, sorted ascending by
        ``open_time``, ``open_time``/``close_time`` in UTC, prices as float64,
        and a fresh ``RangeIndex``.

    Raises:
        CandleDataError: on missing columns, naive timestamps, duplicate
            ``open_time`` values, timestamps off the interval grid, missing
            prices, or OHLC values that cannot coexist (e.g. ``high < low``).
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CandleDataError(f"Candle frame is missing required column(s): {missing}")

    result = frame.loc[:, list(REQUIRED_COLUMNS)].copy()

    open_time = pd.to_datetime(result["open_time"])
    if open_time.dt.tz is None:
        raise CandleDataError(
            "open_time is timezone-naive. All backtool timestamps must be "
            "timezone-aware; see docs/decisions/003-utc-internal-time-standard.md"
        )
    result["open_time"] = open_time.dt.tz_convert("UTC").astype(CANDLE_TIME_DTYPE)

    for column in _PRICE_COLUMNS + ("volume",):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")

    result = result.sort_values("open_time").reset_index(drop=True)

    _check_timestamps(result, interval)
    _check_prices(result)

    result["close_time"] = (result["open_time"] + interval.duration).astype(CANDLE_TIME_DTYPE)
    return result.loc[:, list(CANDLE_COLUMNS)]


def _check_timestamps(frame: pd.DataFrame, interval: Interval) -> None:
    open_time = frame["open_time"]

    duplicates = open_time[open_time.duplicated()]
    if not duplicates.empty:
        raise CandleDataError(
            f"Duplicate open_time values ({len(duplicates)}), first at {duplicates.iloc[0]}. "
            "Exchange pagination usually overlaps at range boundaries; deduplicate on ingest."
        )

    # Every candle must sit on the interval grid measured from the Unix epoch,
    # which is how Binance aligns them. A timestamp off-grid means the data was
    # resampled, shifted, or came from a different interval than claimed.
    #
    # Measured in seconds rather than raw epoch integers: `total_seconds()`
    # always means seconds, whatever backing resolution pandas chose for the
    # column, so this cannot silently break the way a hardcoded nanosecond
    # assumption did.
    seconds_since_epoch = (open_time - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    remainder = seconds_since_epoch % interval.duration.total_seconds()
    off_grid = frame.loc[remainder != 0, "open_time"]
    if not off_grid.empty:
        raise CandleDataError(
            f"{len(off_grid)} candle(s) are not aligned to the {interval.value} grid, "
            f"first at {off_grid.iloc[0]}. Was this frame built with a different interval?"
        )


def _check_prices(frame: pd.DataFrame) -> None:
    for column in _PRICE_COLUMNS:
        if frame[column].isna().any():
            count = int(frame[column].isna().sum())
            raise CandleDataError(f"Column {column!r} has {count} missing value(s)")

    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)

    # These invariants hold for any real candle. A violation means the columns
    # were mislabelled or the source is corrupt -- silently averaging such rows
    # would produce plausible, wrong statistics.
    if (frame["high"] < frame["low"]).any():
        raise CandleDataError("Found candle(s) where high < low")
    if (frame["high"] < body_high).any():
        raise CandleDataError("Found candle(s) where high is below open or close")
    if (frame["low"] > body_low).any():
        raise CandleDataError("Found candle(s) where low is above open or close")
    if (frame[list(_PRICE_COLUMNS)] <= 0).to_numpy().any():
        raise CandleDataError("Found non-positive price(s); spot prices must be > 0")


def empty_candles(interval: Interval) -> pd.DataFrame:
    """An empty frame carrying the normalised candle schema and dtypes.

    Returned instead of ``pd.DataFrame()`` so that callers can slice, filter and
    concatenate a "no data" result without special-casing it, and so dtype
    checks downstream still hold.
    """
    frame = pd.DataFrame(
        {
            "open_time": pd.Series(dtype=CANDLE_TIME_DTYPE),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    return normalise_candles(frame, interval)


def expected_candle_count(
    start: dt.datetime, end: dt.datetime, interval: Interval
) -> int:
    """Interval-typed wrapper over :func:`backtool.core.time.expected_candle_count`."""
    return _expected_candle_count(start, end, interval.duration)
