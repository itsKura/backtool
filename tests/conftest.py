"""Shared test fixtures and synthetic candle construction.

Synthetic candles let us assert against hand-computed expected values. If a
metric is wrong, the test fails for an arithmetic reason, not because an
exchange returned something unexpected.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pandas as pd
import pytest

from backtool.core.time import UTC
from backtool.core.types import Interval
from backtool.data.candles import normalise_candles


def build_candles(
    start: dt.datetime,
    closes: Sequence[float],
    interval: Interval = Interval.M5,
    *,
    opens: Sequence[float] | None = None,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    volume: float = 1.0,
) -> pd.DataFrame:
    """Build a normalised candle frame from a list of close prices.

    By default each candle is a flat "doji" with ``open == high == low ==
    close``, which keeps return tests free of incidental high/low noise. Pass
    ``highs``/``lows`` explicitly when testing excursion metrics.

    Args:
        start: ``open_time`` of the first candle. Must be aligned to the
            interval grid and timezone-aware.
        closes: One close price per candle; determines the candle count.
        interval: Spacing between candles.

    Returns:
        A frame that has already passed :func:`normalise_candles`.
    """
    resolved_opens = list(opens) if opens is not None else list(closes)
    resolved_highs = (
        list(highs)
        if highs is not None
        else [max(o, c) for o, c in zip(resolved_opens, closes, strict=True)]
    )
    resolved_lows = (
        list(lows)
        if lows is not None
        else [min(o, c) for o, c in zip(resolved_opens, closes, strict=True)]
    )

    frame = pd.DataFrame(
        {
            "open_time": [start + index * interval.duration for index in range(len(closes))],
            "open": resolved_opens,
            "high": resolved_highs,
            "low": resolved_lows,
            "close": list(closes),
            "volume": [volume] * len(closes),
        }
    )
    return normalise_candles(frame, interval)


#: A convenient aligned reference instant: 2024-01-31 FOMC, 14:00 ET = 19:00 UTC.
EVENT_TIME = dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)

#: Three hours of 5m candles centred on that event: 18:00 .. 21:00 UTC.
CANDLE_START = dt.datetime(2024, 1, 31, 18, 0, tzinfo=UTC)


@pytest.fixture
def flat_candles() -> pd.DataFrame:
    """36 five-minute candles, all priced at 100.0, spanning 18:00-21:00 UTC."""
    return build_candles(CANDLE_START, [100.0] * 36)
