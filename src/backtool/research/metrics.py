"""Per-event, per-window metrics.

Everything here is deterministic arithmetic over candles. No statistic in this
project is produced anywhere else, and nothing here consults a language model.

Explicit definitions
--------------------
Several of these metrics have more than one defensible definition. Rather than
pick one silently, each is stated here and surfaced to the user alongside
results:

* **return** -- simple percentage change between the two anchor prices, not log
  return. Log returns are better for aggregation; simple returns are what a
  reader means by "BTC rose 2%". Aggregation converts where it matters.
* **MFE / MAE** -- maximum favourable and adverse excursion, defined relative to
  a **long position opened at the window's start price**. MFE can be negative if
  price never traded above entry; it is not floored at zero, because "the best
  it ever got was -0.4%" is information worth keeping.
* **realized volatility** -- square root of the sum of squared log returns
  across the window. This is volatility *over the window*, deliberately not
  annualised: annualising a one-hour window produces a large number that invites
  false comparison with daily figures.
* **direction** -- sign of the window return, with exact equality treated as
  flat.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from backtool.core.types import Interval
from backtool.data.candles import expected_candle_count
from backtool.research.anchoring import anchor_at
from backtool.research.windows import ResolvedWindow


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class WindowStatus(StrEnum):
    """Whether a window produced usable numbers."""

    OK = "ok"
    #: Not enough candles to price the window. Metrics are ``None``. The window
    #: is still reported so it can be counted -- an aggregate over "20 events"
    #: that quietly used 17 is a lie.
    INSUFFICIENT_DATA = "insufficient_data"


class WindowMetrics(BaseModel):
    """Deterministic results for one window of one event.

    Metric fields are ``None`` when ``status`` is not ``OK``. Callers must check
    ``status`` rather than assume a number is present.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    status: WindowStatus
    reason: str | None = None

    # --- anchors (see docs/decisions/002-event-anchoring.md) ---
    start_price: float | None = None
    end_price: float | None = None
    start_anchor_utc: dt.datetime | None = None
    end_anchor_utc: dt.datetime | None = None
    #: How stale each anchor was, in seconds. Zero on aligned, gap-free data.
    #: Larger than one interval means candles were missing at that boundary.
    start_anchor_lag_s: float | None = None
    end_anchor_lag_s: float | None = None

    # --- outcome ---
    return_pct: float | None = None
    abs_move_pct: float | None = None
    direction: Direction | None = None

    # --- path ---
    high: float | None = None
    low: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    range_pct: float | None = None
    realized_vol_pct: float | None = None

    # --- coverage ---
    candle_count: int = 0
    expected_candles: int = 0
    coverage_pct: float | None = None

    @property
    def is_usable(self) -> bool:
        return self.status is WindowStatus.OK


def compute_window_metrics(
    candles: pd.DataFrame,
    window: ResolvedWindow,
    interval: Interval,
) -> WindowMetrics:
    """Compute all metrics for one resolved window.

    Args:
        candles: Normalised candle frame covering at least the window. It may
            extend well beyond it -- only the relevant slice is used.
        window: The window, already bound to a concrete event instant.
        interval: Candle interval, needed to compute expected coverage.

    Returns:
        A :class:`WindowMetrics`. Always returns a value; data problems are
        reported via ``status`` and ``reason`` rather than raised, so one bad
        event cannot abort a twenty-event study.

    What can go wrong:
        * History starts after the window -- no anchor exists, so
          ``INSUFFICIENT_DATA``.
        * An exchange outage removes candles mid-window -- metrics are still
          computed, but ``coverage_pct`` drops and the anchor lags grow.
        * The window is shorter than one candle -- no candle closes inside it,
          so ``INSUFFICIENT_DATA``.
    """
    expected = expected_candle_count(window.start, window.end, interval)

    start_anchor = anchor_at(candles, window.start)
    end_anchor = anchor_at(candles, window.end)

    if start_anchor is None or end_anchor is None:
        missing, boundary = (
            ("start", window.start) if start_anchor is None else ("end", window.end)
        )
        return WindowMetrics(
            name=window.name,
            start_utc=window.start,
            end_utc=window.end,
            expected_candles=expected,
            status=WindowStatus.INSUFFICIENT_DATA,
            reason=(
                f"No candle closed at or before the window {missing} "
                f"({boundary:%Y-%m-%d %H:%M} UTC)"
            ),
        )

    # Candles whose close falls inside (start, end]. The start anchor supplies
    # the opening price, so including a candle that closed exactly at `start`
    # would double-count it.
    in_window = candles[
        (candles["close_time"] > pd.Timestamp(window.start))
        & (candles["close_time"] <= pd.Timestamp(window.end))
    ]

    if in_window.empty:
        return WindowMetrics(
            name=window.name,
            start_utc=window.start,
            end_utc=window.end,
            expected_candles=expected,
            status=WindowStatus.INSUFFICIENT_DATA,
            reason=(
                f"No candles closed inside the window; expected {expected} at "
                f"{interval.value}"
            ),
            start_price=start_anchor.price,
            end_price=end_anchor.price,
            start_anchor_utc=start_anchor.time,
            end_anchor_utc=end_anchor.time,
            start_anchor_lag_s=start_anchor.lag_seconds,
            end_anchor_lag_s=end_anchor.lag_seconds,
            coverage_pct=0.0 if expected else None,
        )

    start_price = start_anchor.price
    end_price = end_anchor.price
    return_pct = (end_price / start_price - 1.0) * 100.0

    high = float(in_window["high"].max())
    low = float(in_window["low"].min())

    return WindowMetrics(
        name=window.name,
        start_utc=window.start,
        end_utc=window.end,
        expected_candles=expected,
        status=WindowStatus.OK,
        start_price=start_price,
        end_price=end_price,
        start_anchor_utc=start_anchor.time,
        end_anchor_utc=end_anchor.time,
        start_anchor_lag_s=start_anchor.lag_seconds,
        end_anchor_lag_s=end_anchor.lag_seconds,
        return_pct=return_pct,
        abs_move_pct=abs(return_pct),
        direction=_direction(return_pct),
        high=high,
        low=low,
        mfe_pct=(high / start_price - 1.0) * 100.0,
        mae_pct=(low / start_price - 1.0) * 100.0,
        range_pct=(high - low) / start_price * 100.0,
        realized_vol_pct=_realized_volatility(start_price, in_window["close"].to_numpy()),
        candle_count=len(in_window),
        coverage_pct=(len(in_window) / expected * 100.0) if expected else None,
    )


def _direction(return_pct: float) -> Direction:
    if return_pct > 0:
        return Direction.UP
    if return_pct < 0:
        return Direction.DOWN
    return Direction.FLAT


def _realized_volatility(start_price: float, closes: np.ndarray) -> float | None:
    """Root sum of squared log returns across the window.

    The series runs from the anchor price through every close inside the
    window, so the first move out of the anchor is included rather than
    discarded.

    Returns ``None`` when fewer than two prices are available, since a single
    observation has no dispersion.
    """
    prices = np.concatenate(([start_price], closes.astype("float64")))
    if prices.size < 3:
        return None
    log_returns = np.diff(np.log(prices))
    return float(np.sqrt(np.sum(log_returns**2)) * 100.0)
