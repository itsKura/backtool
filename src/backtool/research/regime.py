"""What the market looked like immediately before a given instant.

Four deterministic readings -- a one-week return, a one-month return, a
volatility percentile, and position relative to the 200-day average -- each
bucketed into a handful of named states. Together they are the "setup" that
Similar Setups matches on.

Every number here is arithmetic over daily closes. There is no similarity score,
no embedding, no model. A reader who disagrees with a match can read the four
figures that produced it and check them, which is the entire point of building
it this way: an opaque score that says "83% similar" cannot be argued with, and
a wrong one cannot be spotted.

Point-in-time, strictly
-----------------------
Every figure for a historical release is computed from candles that had already
closed **before that release**. This matters most for the volatility percentile:
ranking a 2024 release against the full 2017-2026 distribution would classify it
using years of data that had not happened yet, and the resulting "high
volatility" cohort would be selected partly by hindsight. The percentile here is
against the trailing window available at the time, and nothing else.

Daily closes are resampled from 1h candles rather than fetched as daily bars.
The interval enum deliberately stops at 1h -- adding a daily member to support
this would also let someone select it for a one-hour event window, which is the
thing that enum exists to prevent.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from backtool.core.time import ensure_utc
from backtool.core.types import Interval

#: Lookbacks the setup is built from, in calendar days.
TREND_SHORT_DAYS = 7
TREND_LONG_DAYS = 30
VOL_WINDOW_DAYS = 30
MA_DAYS = 200

#: How much history to rank a volatility reading against. Two years is long
#: enough for the percentile to mean something and short enough that a regime
#: from five years ago does not define "normal" for today.
VOL_PERCENTILE_LOOKBACK_DAYS = 730

#: How far back candles must be fetched to produce every reading. The 200-day
#: average is the binding constraint; the percentile ranks values that each
#: already need 30 days, so its own lookback adds to that.
REQUIRED_HISTORY_DAYS = VOL_PERCENTILE_LOOKBACK_DAYS + VOL_WINDOW_DAYS + 10

#: Trend thresholds, in percent. Stated as constants because they are a
#: research choice, not a fact: "bullish" is a definition this project picked
#: and shows, and changing it here changes it everywhere it is displayed.
TREND_BULL_PCT = 3.0
TREND_BEAR_PCT = -3.0

#: Volatility percentile cuts.
VOL_LOW_PCTL = 33.0
VOL_HIGH_PCTL = 67.0


class Trend(StrEnum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class VolRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class MAPosition(StrEnum):
    ABOVE = "above"
    BELOW = "below"


class MarketSetup(BaseModel):
    """The four readings, their buckets, and what was measurable.

    Any field may be ``None``: a release 100 days after the symbol started
    trading has no 200-day average, and saying so is the honest answer. A
    ``None`` bucket never matches anything, so a setup with gaps narrows the
    comparison rather than silently widening it.
    """

    model_config = ConfigDict(frozen=True)

    as_of: dt.datetime
    close: float | None = None

    return_7d_pct: float | None = None
    trend_7d: Trend | None = None

    return_30d_pct: float | None = None
    trend_30d: Trend | None = None

    realized_vol_30d_pct: float | None = None
    vol_percentile: float | None = None
    vol_regime: VolRegime | None = None

    ma_200d: float | None = None
    ma_200d_gap_pct: float | None = None
    ma_position: MAPosition | None = None

    @property
    def measured(self) -> dict[str, str]:
        """The buckets that exist, keyed by dimension name.

        A dimension that could not be measured is absent rather than defaulted,
        so matching iterates over what is actually known.
        """
        found: dict[str, str] = {}
        if self.trend_7d is not None:
            found["trend_7d"] = self.trend_7d.value
        if self.trend_30d is not None:
            found["trend_30d"] = self.trend_30d.value
        if self.vol_regime is not None:
            found["vol_regime"] = self.vol_regime.value
        if self.ma_position is not None:
            found["ma_position"] = self.ma_position.value
        return found

    @property
    def readings(self) -> dict[str, float]:
        """The numeric figures behind the buckets, where they exist.

        The buckets above are a lossy summary: ``-4.6%`` and ``-28%`` are both
        "bearish", and a reader asking about a fall of more than 3% means the
        number, not the nearest bucket edge. Both are returned from the same
        computation, so exposing the figures costs nothing and lets a condition
        test either one.

        An absent key means the figure could not be computed from the history
        available before ``as_of``. It is not zero.
        """
        found: dict[str, float] = {}
        if self.return_7d_pct is not None:
            found["return_7d_pct"] = self.return_7d_pct
        if self.return_30d_pct is not None:
            found["return_30d_pct"] = self.return_30d_pct
        if self.realized_vol_30d_pct is not None:
            found["realized_vol_30d_pct"] = self.realized_vol_30d_pct
        if self.vol_percentile is not None:
            found["vol_percentile"] = self.vol_percentile
        if self.ma_200d_gap_pct is not None:
            found["ma_200d_gap_pct"] = self.ma_200d_gap_pct
        return found

    def describe_items(self) -> list[tuple[str, str]]:
        """The setup as label/value pairs, for display.

        Each line carries both the measurement and the bucket it produced, so a
        reader sees ``-4.6% -> bearish`` rather than a bare verdict they have to
        take on trust.
        """
        items: list[tuple[str, str]] = []
        if self.return_7d_pct is not None and self.trend_7d is not None:
            items.append(
                ("BTC 7-day", f"{self.return_7d_pct:+.1f}% → {self.trend_7d.value}")
            )
        if self.return_30d_pct is not None and self.trend_30d is not None:
            items.append(
                ("BTC 30-day", f"{self.return_30d_pct:+.1f}% → {self.trend_30d.value}")
            )
        if self.vol_percentile is not None and self.vol_regime is not None:
            items.append(
                (
                    "Volatility",
                    f"{_ordinal(self.vol_percentile)} percentile "
                    f"→ {self.vol_regime.value}",
                )
            )
        if self.ma_position is not None and self.ma_200d_gap_pct is not None:
            items.append(
                (
                    "BTC vs 200-day average",
                    f"{self.ma_200d_gap_pct:+.1f}% → {self.ma_position.value}",
                )
            )
        return items


#: Display names for the four dimensions, used when explaining a match or a
#: relaxation. Defined next to the buckets so a renamed dimension cannot show
#: up under its engine name in a sentence about why something matched.
DIMENSION_LABELS: dict[str, str] = {
    "trend_7d": "BTC 7-day trend",
    "trend_30d": "BTC 30-day trend",
    "vol_regime": "volatility regime",
    "ma_position": "position vs the 200-day average",
}

#: The conditions that define "like today", in two explicit tiers.
#:
#: Primary conditions are what a reader means by a similar setup: which way the
#: market has been moving this week, and how violently it is moving at all.
#: Those two decide whether a release arrived into a calm drift or a panic.
#:
#: Secondary conditions are context. The 200-day average and the monthly trend
#: describe the wider position, and two releases that differ only on those are
#: still the most alike of any pair that differs at all.
#:
#: The split is declared here rather than left implicit in an ordering, because
#: the ordering is derived from it -- reclassifying a dimension changes what
#: gets relaxed first without anyone having to notice a tuple was reordered.
PRIMARY_DIMENSIONS: tuple[str, ...] = ("trend_7d", "vol_regime")
SECONDARY_DIMENSIONS: tuple[str, ...] = ("ma_position", "trend_30d")

#: Order dimensions are dropped in when an exact match is too small: every
#: secondary condition before any primary one. Within a tier, the slower-moving
#: condition goes first.
RELAXATION_ORDER: tuple[str, ...] = (*SECONDARY_DIMENSIONS, *PRIMARY_DIMENSIONS)

#: How the matching rule is described in Data & methodology. Kept beside the
#: tiers so a change to one cannot leave the other describing the old rule.
MATCHING_METHODOLOGY: tuple[tuple[str, str], ...] = (
    (
        "match rule",
        "exact agreement on every active condition; no distance score, no weighting",
    ),
    ("primary conditions", "BTC 7-day trend, volatility regime"),
    ("secondary conditions", "BTC 30-day trend, position vs the 200-day average"),
    (
        "relaxation",
        "secondary conditions dropped first, one at a time, until the sample "
        "reaches 5 releases or one condition remains",
    ),
    (
        "7-day / 30-day trend",
        f"bullish above {TREND_BULL_PCT:+g}%, bearish below {TREND_BEAR_PCT:+g}%, "
        "otherwise neutral",
    ),
    (
        "volatility regime",
        f"30-day realized volatility ranked against the prior "
        f"{VOL_PERCENTILE_LOOKBACK_DAYS // 365} years; low below the "
        f"{VOL_LOW_PCTL:g}th percentile, high above the {VOL_HIGH_PCTL:g}th",
    ),
    ("200-day average", f"mean of the last {MA_DAYS} daily closes"),
    (
        "point-in-time",
        "every reading uses only candles that closed before the instant "
        "measured; nothing after it is visible to the classification",
    ),
    ("conditions read from", "{symbol} daily closes resampled from 1h candles"),
)


def daily_closes(candles: pd.DataFrame) -> pd.Series:
    """Resample an hourly frame to one close per UTC day.

    Returns:
        A series indexed by day-end timestamp. Days with no candles are absent
        rather than forward-filled: an exchange outage is missing data, and
        carrying the previous close across it would invent a flat day that
        deflates every volatility figure spanning it.
    """
    if candles.empty:
        return pd.Series(dtype="float64")

    frame = candles.set_index("close_time").sort_index()
    return frame["close"].resample("1D").last().dropna()


def compute_setup(candles: pd.DataFrame, as_of: dt.datetime) -> MarketSetup:
    """Read the market state as it stood immediately before ``as_of``.

    Args:
        candles: Hourly candles reaching back far enough for the longest
            lookback. Anything after ``as_of`` is discarded before use.
        as_of: The instant to measure at, typically a release time.

    Returns:
        A :class:`MarketSetup`. Every field that could not be computed from the
        supplied history is ``None`` rather than approximated.
    """
    moment = ensure_utc(as_of)
    closes = daily_closes(candles)
    # Strictly before: a candle that closes exactly at the release instant
    # contains the release, and using it would price the setup with the
    # reaction it is supposed to precede.
    closes = closes[closes.index < pd.Timestamp(moment)]
    if closes.empty:
        return MarketSetup(as_of=moment)

    last = float(closes.iloc[-1])
    return_7d, trend_7d = _trend(closes, TREND_SHORT_DAYS, moment)
    return_30d, trend_30d = _trend(closes, TREND_LONG_DAYS, moment)
    vol, pctl, regime = _volatility(closes)
    ma, gap, position = _moving_average(closes, last)

    return MarketSetup(
        as_of=moment,
        close=last,
        return_7d_pct=return_7d,
        trend_7d=trend_7d,
        return_30d_pct=return_30d,
        trend_30d=trend_30d,
        realized_vol_30d_pct=vol,
        vol_percentile=pctl,
        vol_regime=regime,
        ma_200d=ma,
        ma_200d_gap_pct=gap,
        ma_position=position,
    )


def _trend(
    closes: pd.Series, days: int, moment: dt.datetime
) -> tuple[float | None, Trend | None]:
    """Return over the trailing ``days``, and the bucket it falls in.

    The reference close is the last one at or before ``moment - days``, not the
    close ``days`` rows back. Rows are trading days only where data exists, and
    counting rows would silently stretch the lookback across every gap.
    """
    cutoff = pd.Timestamp(moment - dt.timedelta(days=days))
    earlier = closes[closes.index <= cutoff]
    if earlier.empty:
        return None, None

    start = float(earlier.iloc[-1])
    if start <= 0:
        return None, None

    change = (float(closes.iloc[-1]) / start - 1.0) * 100.0
    if change > TREND_BULL_PCT:
        bucket = Trend.BULLISH
    elif change < TREND_BEAR_PCT:
        bucket = Trend.BEARISH
    else:
        bucket = Trend.NEUTRAL
    return change, bucket


def _volatility(
    closes: pd.Series,
) -> tuple[float | None, float | None, VolRegime | None]:
    """Trailing 30-day realized volatility, and where it ranks historically.

    The rank is against the same statistic computed over the preceding two
    years *of this series only*, which contains nothing after the instant being
    measured. Percentile rather than a raw threshold because a volatility that
    counted as high in 2018 is unremarkable now, and a fixed cut would sort
    releases by era rather than by regime.
    """
    series = _rolling_vol(closes)
    if series.empty:
        return None, None, None

    current = float(series.iloc[-1])
    if not np.isfinite(current):
        return None, None, None

    history = series.iloc[-_rows_for(VOL_PERCENTILE_LOOKBACK_DAYS) :]
    # A percentile over a handful of readings is not a percentile. Report the
    # volatility itself and decline the bucket rather than manufacture a rank.
    if len(history) < 60:
        return current, None, None

    percentile = float((history <= current).mean() * 100.0)
    if percentile < VOL_LOW_PCTL:
        regime = VolRegime.LOW
    elif percentile > VOL_HIGH_PCTL:
        regime = VolRegime.HIGH
    else:
        regime = VolRegime.NORMAL
    return current, percentile, regime


def _rolling_vol(closes: pd.Series) -> pd.Series:
    """Rolling standard deviation of daily log returns, in percent.

    Not annualised, matching the convention the rest of this project uses for
    realized volatility: annualising invites comparison with figures computed
    over entirely different horizons.
    """
    if len(closes) < 2:
        return pd.Series(dtype="float64")

    # Built through numpy explicitly rather than np.log(series): the ufunc on a
    # Series is typed as returning an array, and the index has to survive for
    # the percentile lookback below to slice by date.
    ratio = (closes / closes.shift(1)).dropna()
    log_returns = pd.Series(np.log(ratio.to_numpy(dtype="float64")), index=ratio.index)

    window = _rows_for(VOL_WINDOW_DAYS)
    if len(log_returns) < window:
        return pd.Series(dtype="float64")

    rolling: pd.Series = log_returns.rolling(window).std() * 100.0
    return rolling.dropna()


def _moving_average(
    closes: pd.Series, last: float
) -> tuple[float | None, float | None, MAPosition | None]:
    """The 200-day average and which side of it price sits on.

    Returns ``None`` when fewer than 200 daily closes exist. A 140-day average
    labelled as a 200-day one would be a quiet redefinition, and it is exactly
    the sort of thing that is impossible to notice from the output.
    """
    window = _rows_for(MA_DAYS)
    if len(closes) < window:
        return None, None, None

    average = float(closes.iloc[-window:].mean())
    if average <= 0:
        return None, None, None

    gap = (last / average - 1.0) * 100.0
    position = MAPosition.ABOVE if last >= average else MAPosition.BELOW
    return average, gap, position


def _rows_for(days: int) -> int:
    """Daily rows spanning ``days``. One row per day, by construction."""
    return days


def _ordinal(value: float) -> str:
    """``78`` -> ``78th``. Percentiles read as ranks, not as measurements."""
    number = int(round(value))
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def history_window(as_of: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """The candle range :func:`compute_setup` needs to produce every reading."""
    moment = ensure_utc(as_of)
    return moment - dt.timedelta(days=REQUIRED_HISTORY_DAYS), moment


#: Interval the setup is built from. Hourly candles resampled to daily closes.
SETUP_INTERVAL = Interval.H1

#: The symbol whose regime defines "the market". Setups describe conditions, and
#: BTC is the condition even when the release is being measured against ETH.
SETUP_SYMBOL = "BTCUSDT"


__all__ = [
    "DIMENSION_LABELS",
    "MA_DAYS",
    "RELAXATION_ORDER",
    "REQUIRED_HISTORY_DAYS",
    "SETUP_INTERVAL",
    "SETUP_SYMBOL",
    "TREND_BEAR_PCT",
    "TREND_BULL_PCT",
    "MAPosition",
    "MarketSetup",
    "Trend",
    "VolRegime",
    "compute_setup",
    "daily_closes",
    "history_window",
]
