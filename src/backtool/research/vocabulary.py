"""The controlled vocabulary a research question may be expressed in.

This module is the boundary between "what a person can ask" and "what this
system can actually measure". Everything a thesis may refer to -- an asset, an
event, a condition, an outcome, a baseline -- is declared here, with the values
it accepts and the resolver that answers it from measured data.

Why a vocabulary rather than generated code
-------------------------------------------
The obvious way to answer arbitrary questions is to let a model write the query.
That approach can express anything, including things that are wrong in ways
nobody can see: a filter that silently matched nothing, a statistic computed
over the wrong subset, a comparison against a baseline that quietly included the
cohort. None of those raise.

A fixed vocabulary cannot express those questions at all. The cost is real --
an unsupported question has to be refused rather than approximated -- and that
refusal is the feature. "I cannot test Treasury yields, that dataset is not
connected" is a worse answer than a number and a better answer than a wrong
number.

How to extend it
----------------
Adding funding rates, open interest, or drawdown later means adding a
:class:`ConditionField` entry with its allowed values and a resolver. Nothing
about the thesis model, the planner, the executor or the UI changes. That is the
whole point of declaring the vocabulary in one place.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from backtool.core.types import EventType
from backtool.research.windows import WindowSpec

#: Markets that can be measured. Both list on Binance from 2017-08-17, so
#: neither truncates the history relative to the other.
SUPPORTED_ASSETS: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
}


class FieldKind(StrEnum):
    """Whether a field resolves to a named state or to a number.

    The distinction decides which operators are legal and which resolver
    answers it, and it is declared per field rather than inferred, so a field
    added without saying which it is fails loudly instead of defaulting into
    the wrong comparison.
    """

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class Operator(StrEnum):
    """How a condition compares a field to a value.

    Two families, and a condition uses exactly one. Categorical operators test
    a named state a classifier produced; numeric operators test a measured
    reading against a threshold. Mixing them -- ``cpi_surprise > 3`` or
    ``btc_return_7d is bearish`` -- is rejected at construction rather than
    producing an empty cohort that looks like a result.
    """

    IS = "is"
    IS_NOT = "is not"
    ABOVE = ">"
    AT_LEAST = ">="
    BELOW = "<"
    AT_MOST = "<="

    @property
    def is_numeric(self) -> bool:
        return self in NUMERIC_OPERATORS

    @property
    def bound(self) -> str:
        """Which side of a range this operator constrains.

        ``"lower"``, ``"upper"``, or ``""`` for a categorical operator. Used so
        a follow-up adding an upper bound does not evict the lower bound
        already in place -- which is what makes ``between -5% and -2%``
        expressible as two conditions rather than unsupported.
        """
        if self in {Operator.ABOVE, Operator.AT_LEAST}:
            return "lower"
        if self in {Operator.BELOW, Operator.AT_MOST}:
            return "upper"
        return ""

    def phrase(self) -> str:
        """How the comparison reads in a sentence a reader confirms."""
        return _OPERATOR_PHRASES[self]


#: Operators that compare against a number rather than a named state.
NUMERIC_OPERATORS: frozenset[Operator] = frozenset(
    {Operator.ABOVE, Operator.AT_LEAST, Operator.BELOW, Operator.AT_MOST}
)

#: Operators that compare against one of a field's declared values.
CATEGORICAL_OPERATORS: frozenset[Operator] = frozenset(
    {Operator.IS, Operator.IS_NOT}
)

_OPERATOR_PHRASES: dict[Operator, str] = {
    Operator.IS: "is",
    Operator.IS_NOT: "is not",
    Operator.ABOVE: "is above",
    Operator.AT_LEAST: "is at least",
    Operator.BELOW: "is below",
    Operator.AT_MOST: "is at most",
}

#: The canonical representation of every numeric field in this vocabulary:
#: **the number a reader would say out loud**. A 3% fall over the week is
#: ``-3.0``, never ``-0.03``; a volatility percentile of 67 is ``67.0``, never
#: ``0.67``. This matches the ``*_pct`` convention the rest of the project
#: already uses, so a threshold and the reading it is compared against are
#: always on the same scale.
#:
#: Nothing rescales a threshold on the reader's behalf. ``-0.03`` is a legal
#: threshold meaning three hundredths of one percent, and guessing that it
#: "really meant" ``-3`` would be exactly the silent redefinition this layer
#: exists to prevent. The defence is disclosure: every threshold is rendered
#: back in the specification card, with its unit, before anything runs.
NUMERIC_UNIT_RULE = (
    "Numbers are in the units a reader speaks: a 3% move is 3, not 0.03. "
    "A percentile is 0 to 100."
)


class OutcomeMetric(StrEnum):
    """What is measured after the event."""

    RETURN = "return"
    POSITIVE_RATE = "positive rate"
    RANGE = "trading range"


class BaselineKind(StrEnum):
    """What the cohort is compared against.

    V1 offers one: every usable release of the same type. That is the
    comparison that answers "better than what?" -- without it, "cooler CPI
    gives +1.2% over 24h" is a number with nothing to be better or worse than.
    """

    ALL_EVENTS = "all events of the same type"


@dataclass(frozen=True)
class FieldSpec:
    """One thing a condition may test, and how to answer it.

    Args:
        key: Stable identifier used in a thesis and by the planner.
        label: What a reader sees.
        values: Allowed values, each mapped to its display form. A thesis
            naming a value outside this set is rejected rather than coerced.
        source: Which measurement supplies it, for the methodology block.
        requires_regime: Whether answering it needs the market-regime pass,
            which is an expensive extra fetch and so is only run when asked for.
    """

    key: str
    label: str
    values: dict[str, str]
    source: str
    requires_regime: bool = False
    #: Event types this field can be tested against. Empty means all of them.
    event_types: frozenset[EventType] = field(default_factory=frozenset)

    #: Whether this field resolves to a named state or a number.
    kind: FieldKind = FieldKind.CATEGORICAL
    #: Suffix shown after a numeric reading or threshold, e.g. ``"%"``. Empty
    #: for a dimensionless number such as a percentile.
    unit: str = ""
    #: Range a threshold must fall inside to be accepted. Not a guess at what
    #: the reader meant -- a filter on values that cannot describe this
    #: measurement at all, so a parse failure is caught at construction rather
    #: than silently producing a cohort of zero.
    bounds: tuple[float, float] | None = None

    def accepts(self, value: str) -> bool:
        return value in self.values

    def accepts_threshold(self, threshold: float) -> bool:
        if self.bounds is None:
            return True
        low, high = self.bounds
        return low <= threshold <= high

    def display(self, value: str) -> str:
        return self.values.get(value, value)

    def applies_to(self, event_type: EventType) -> bool:
        return not self.event_types or event_type in self.event_types

    @property
    def is_numeric(self) -> bool:
        return self.kind is FieldKind.NUMERIC

    def format_number(self, reading: float) -> str:
        """A reading or threshold as a reader sees it, unit included.

        Signed for a field that can go either way, because ``3%`` and ``-3%``
        are opposite findings and an unsigned figure forces the reader to go
        looking for the sign elsewhere. The sign is taken from the *rounded*
        value, so a reading that rounds to nothing does not print as ``-0.0``.
        """
        rounded = round(reading, 2)
        if self.bounds is not None and self.bounds[0] >= 0:
            return f"{rounded:g}{self.unit}"
        if rounded == 0:
            return f"0{self.unit}"
        return f"{rounded:+g}{self.unit}"


#: Every condition a V1 thesis may express.
#:
#: The CPI surprise states come from :mod:`backtool.events.releases`; the four
#: market states from :mod:`backtool.research.regime`. Both already existed and
#: are already tested -- this table names them, it does not reimplement them.
CONDITION_FIELDS: dict[str, FieldSpec] = {
    "cpi_surprise": FieldSpec(
        key="cpi_surprise",
        label="CPI surprise",
        values={
            "hotter": "hotter than expected",
            "in_line": "in line with expectations",
            "cooler": "cooler than expected",
        },
        source="headline CPI-U year-over-year actual against consensus",
        event_types=frozenset({EventType.CPI}),
    ),
    "btc_trend_7d": FieldSpec(
        key="btc_trend_7d",
        label="BTC 7-day trend",
        values={"bullish": "bullish", "neutral": "neutral", "bearish": "bearish"},
        source="BTC return over the 7 days before the release",
        requires_regime=True,
    ),
    "btc_trend_30d": FieldSpec(
        key="btc_trend_30d",
        label="BTC 30-day trend",
        values={"bullish": "bullish", "neutral": "neutral", "bearish": "bearish"},
        source="BTC return over the 30 days before the release",
        requires_regime=True,
    ),
    "btc_vol_regime": FieldSpec(
        key="btc_vol_regime",
        label="BTC volatility regime",
        values={"low": "low", "normal": "normal", "high": "high"},
        source="30-day realised volatility ranked against the prior two years",
        requires_regime=True,
    ),
    "btc_vs_200d_ma": FieldSpec(
        key="btc_vs_200d_ma",
        label="BTC vs 200-day average",
        values={"above": "above", "below": "below"},
        source="BTC close against the mean of the last 200 daily closes",
        requires_regime=True,
    ),
    # Numeric fields. Each reads a figure the regime pass already computes and
    # currently discards after bucketing it -- no new measurement, no new
    # threshold invented by this layer. A reader who asks "fell more than 3%"
    # gets 3% rather than the nearest bucket edge.
    "btc_return_7d": FieldSpec(
        key="btc_return_7d",
        label="BTC 7-day return",
        values={},
        source="BTC return over the 7 days before the release, in percent",
        requires_regime=True,
        kind=FieldKind.NUMERIC,
        unit="%",
        # Wide enough to contain anything BTC has ever done over a week and
        # narrow enough to reject a figure that cannot be a weekly return.
        bounds=(-100.0, 1000.0),
    ),
    "btc_return_30d": FieldSpec(
        key="btc_return_30d",
        label="BTC 30-day return",
        values={},
        source="BTC return over the 30 days before the release, in percent",
        requires_regime=True,
        kind=FieldKind.NUMERIC,
        unit="%",
        bounds=(-100.0, 2000.0),
    ),
    # The percentile, not the raw realised volatility. The underlying figure is
    # a 30-day standard deviation of daily log returns, un-annualised, and a
    # threshold on it ("above 3") means nothing to a reader without knowing
    # that convention. The percentile is self-describing, already point-in-time
    # ranked, and shares its cuts with btc_vol_regime -- so `> 67` is the same
    # boundary as "high", visibly rather than by coincidence.
    "btc_vol_percentile": FieldSpec(
        key="btc_vol_percentile",
        label="BTC volatility percentile",
        values={},
        source=(
            "30-day realised volatility ranked against the prior two years; "
            "the same ranking btc_vol_regime buckets at the 33rd and 67th"
        ),
        requires_regime=True,
        kind=FieldKind.NUMERIC,
        bounds=(0.0, 100.0),
    ),
}

#: Numeric fields only, in declaration order.
NUMERIC_FIELDS: tuple[str, ...] = tuple(
    key for key, spec in CONDITION_FIELDS.items() if spec.is_numeric
)

#: Horizons an outcome may be measured over, mapped to the window the engine
#: already measures. Keys are what a person says; values are what runs.
OUTCOME_HORIZONS: dict[str, str] = {
    "5m": "post_event_5m",
    "15m": "post_event_15m",
    "1h": "post_event_1h",
    "4h": "post_event_4h",
    "24h": "post_event_24h",
}

#: The horizon assumed when a question does not name one. Long enough to
#: contain a full reaction rather than the first spike, short enough that the
#: next day's unrelated news has not yet dominated it.
DEFAULT_HORIZON = "24h"

#: Event types a thesis may name.
SUPPORTED_EVENTS: dict[str, str] = {
    EventType.CPI.value: "US CPI release",
    EventType.FOMC.value: "FOMC rate decision",
}


def field_spec(key: str) -> FieldSpec | None:
    return CONDITION_FIELDS.get(key)


def window_for(horizon: str) -> str | None:
    return OUTCOME_HORIZONS.get(horizon)


def asset_label(symbol: str) -> str:
    return SUPPORTED_ASSETS.get(symbol, symbol)


def describe_vocabulary() -> dict[str, list[str]]:
    """The whole supported surface, for the planner prompt and the UI.

    Generated from the tables above rather than written out again, so a field
    added here cannot be missing from what the planner is told it may use.
    """
    return {
        "assets": sorted(SUPPORTED_ASSETS),
        "events": sorted(SUPPORTED_EVENTS),
        # One list rather than two. Categorical and numeric fields are the same
        # kind of thing to whoever is asking the question, and splitting them
        # would invite a field being described in one place and enforced from
        # the other.
        "condition_fields": [_describe_field(spec) for spec in CONDITION_FIELDS.values()],
        "categorical_operators": [op.value for op in CATEGORICAL_OPERATORS],
        "numeric_operators": sorted(op.value for op in NUMERIC_OPERATORS),
        "units": [NUMERIC_UNIT_RULE],
        "outcome_metrics": [metric.value for metric in OutcomeMetric],
        "horizons": list(OUTCOME_HORIZONS),
        "baselines": [kind.value for kind in BaselineKind],
    }


def _describe_field(spec: FieldSpec) -> str:
    """One line a planner can act on: name, kind, and what it may be compared to."""
    if spec.is_numeric:
        low, high = spec.bounds or (float("-inf"), float("inf"))
        unit = spec.unit or "percentile points"
        return (
            f"{spec.key} (number in {unit}, between {low:g} and {high:g}; "
            f"use >, >=, < or <= with a threshold) -- {spec.source}"
        )
    return (
        f"{spec.key} ({', '.join(sorted(spec.values))}; use is or is not) "
        f"-- {spec.source}"
    )


#: Resolver signature: given an event id and the measured context, return the
#: field's value for that event, or ``None`` when it could not be determined.
#: ``None`` never matches, so an unmeasurable event narrows a cohort rather
#: than silently widening it.
Resolver = Callable[[str], str | None]


@dataclass(frozen=True)
class ResolutionContext:
    """Everything the resolvers need, gathered once per run.

    Assembled by the executor rather than fetched per condition: reading the
    market state before forty releases is one large fetch, and doing it once
    per condition would repeat it.
    """

    surprises: dict[str, str] = field(default_factory=dict)
    regimes: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Numeric readings per event, keyed by the regime layer's own field names.
    #: Separate from ``regimes`` rather than a widened value type, so every
    #: call site stays typed on one of the two and no caller has to ask which
    #: kind of answer it got.
    readings: dict[str, dict[str, float]] = field(default_factory=dict)
    as_of: dt.datetime | None = None

    def resolve(self, field_key: str, event_id: str) -> str | None:
        """The named state of a categorical field, or ``None`` if unmeasured."""
        if field_key == "cpi_surprise":
            return self.surprises.get(event_id)
        return self.regimes.get(event_id, {}).get(_REGIME_KEYS.get(field_key, ""))

    def resolve_number(self, field_key: str, event_id: str) -> float | None:
        """The reading of a numeric field, or ``None`` if unmeasured.

        ``None`` fails every comparison, in both directions. A release whose
        7-day return could not be computed is not "below -3%" and not "above
        -3%" either; it is unknown, and admitting it to the negative case would
        fill a cohort with releases selected for having less history.
        """
        key = _NUMERIC_KEYS.get(field_key)
        if key is None:
            return None
        return self.readings.get(event_id, {}).get(key)


#: Thesis field key -> the key the regime layer stores it under.
_REGIME_KEYS: dict[str, str] = {
    "btc_trend_7d": "trend_7d",
    "btc_trend_30d": "trend_30d",
    "btc_vol_regime": "vol_regime",
    "btc_vs_200d_ma": "ma_position",
}

#: Numeric thesis field key -> the attribute :class:`MarketSetup` carries it in.
#: Both halves of each pair are already in percent, so nothing here converts.
_NUMERIC_KEYS: dict[str, str] = {
    "btc_return_7d": "return_7d_pct",
    "btc_return_30d": "return_30d_pct",
    "btc_vol_percentile": "vol_percentile",
}


def reaction_windows() -> tuple[WindowSpec, ...]:
    """The windows every thesis measures.

    All five horizons are measured whatever the thesis asked for, because the
    cost is one study either way and a reader looking at a 24-hour answer
    almost always wants to know whether it was already there at one hour.
    """
    from backtool.research.windows import REACTION_WINDOWS

    return REACTION_WINDOWS


__all__ = [
    "CATEGORICAL_OPERATORS",
    "CONDITION_FIELDS",
    "DEFAULT_HORIZON",
    "NUMERIC_FIELDS",
    "NUMERIC_OPERATORS",
    "NUMERIC_UNIT_RULE",
    "OUTCOME_HORIZONS",
    "SUPPORTED_ASSETS",
    "SUPPORTED_EVENTS",
    "BaselineKind",
    "FieldKind",
    "FieldSpec",
    "Operator",
    "OutcomeMetric",
    "ResolutionContext",
    "Resolver",
    "asset_label",
    "describe_vocabulary",
    "field_spec",
    "reaction_windows",
    "window_for",
]
