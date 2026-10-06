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


class TemporalScope(StrEnum):
    """When a quantity becomes knowable, relative to the event.

    The most important distinction in this module, and the reason it is a
    declared field rather than a convention. A cohort may only be selected on
    things that were already true *before* the release. Selecting on something
    measured afterwards -- "the releases where BTC moved more than 2%" -- picks
    the sample using the answer, and the resulting statistic describes a group
    assembled by hindsight while looking exactly like one that was not.

    That failure is invisible in the output: the numbers are arithmetically
    correct, the sample size is honest, and nothing raises. So it is prevented
    at construction instead. See :class:`~backtool.research.thesis.Condition`.
    """

    PRE_EVENT = "pre-event"
    POST_EVENT = "post-event"


class Aggregation(StrEnum):
    """How a column of per-event readings becomes a result.

    ``ROWS`` is the identity: it keeps the readings as rows, which is what
    ordering and limiting operate on. Largest, smallest and "top 5" are all
    ``ROWS`` with an order and a limit, so there is no MAX or MIN here -- a
    maximum is the first row of a descending sort, and expressing it that way
    hands back the release as well as the number.
    """

    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    #: Share of readings satisfying a threshold, in percent. The only
    #: aggregation that takes a parameter.
    FREQUENCY = "frequency"
    ROWS = "rows"

    @property
    def is_scalar(self) -> bool:
        """Whether this collapses the rows to a single number."""
        return self is not Aggregation.ROWS

    @property
    def yields_metric_units(self) -> bool:
        """Whether the result is a value *of the metric*, or a fact *about the set*.

        The distinction the renderer needs and did not have. A median return is
        a return: it shares an axis with the observations, can be marked on
        their distribution, and is read in percent of price. A frequency is a
        share of releases -- the same ``75.0`` means something entirely
        different, and putting it on a chart of returns labels it as a 75%
        move.

        ``frequency(return_pct > 0)`` has three separate units in play: the
        metric is a return, the threshold is a return, and the result is a
        proportion of observations. Only the first two belong on a price axis.
        """
        return self in {Aggregation.MEAN, Aggregation.MEDIAN}

    @property
    def result_noun(self) -> str:
        """What the result *is*, for an axis caption.

        Empty when the result is a value of the metric, because then the
        metric's own label is the honest caption and this enum has no business
        knowing it.
        """
        if self is Aggregation.FREQUENCY:
            return "share of releases"
        if self is Aggregation.COUNT:
            return "releases"
        return ""


class ResultShape(StrEnum):
    """The kind of answer a query produces, and therefore how it is laid out.

    Derived from the query -- whether the aggregation collapses the rows, and
    whether a baseline was asked for -- rather than from a catalogue of
    question types. That is the difference between a product where a new kind
    of question needs a new handler and one where it needs a new row in a
    table.

    A fourth shape, every observation as a series, is what ``ROWS`` without a
    limit would mean. It is deliberately not built yet; the limit is required
    so the half-built case cannot be reached by accident.
    """

    #: One number, set against the same number for another group. That group
    #: may be every usable release -- the classic baseline -- or a second named
    #: selection; both are the same shape and the same code.
    COMPARISON = "comparison"
    #: One number, standing on its own.
    SCALAR = "scalar"
    #: Individual releases, ordered and cut to a limit.
    RANKED = "ranked rows"


class Order(StrEnum):
    """Direction of a sort over per-event readings."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True)
class MetricSpec:
    """One measured quantity a question may ask about.

    Every one of these is already computed for every event and every window by
    :class:`~backtool.research.metrics.WindowMetrics`. This table names them and
    says how to read them; it adds no measurement. ``attribute`` is the field to
    project out of a ``WindowMetrics``, which is why adding a metric that the
    engine already produces costs one entry here and nothing else.
    """

    key: str
    label: str
    #: The ``WindowMetrics`` attribute this projects.
    attribute: str
    unit: str
    description: str
    #: Whether the reading can be negative, which decides whether it is
    #: rendered with an explicit sign.
    signed: bool = True
    #: Always post-event. Declared rather than assumed, so the validator has
    #: something to check instead of a naming convention to trust.
    scope: TemporalScope = TemporalScope.POST_EVENT

    def format(self, value: float) -> str:
        """A reading as a reader sees it, with its unit and sign."""
        rounded = round(value, 2)
        if not self.signed:
            return f"{rounded:,.2f}{self.unit}"
        if rounded == 0:
            return f"0.00{self.unit}"
        return f"{rounded:+.2f}{self.unit}"


#: Every quantity a thesis may measure, keyed as the planner names it.
#:
#: All eight already exist on ``WindowMetrics`` for every event and window. A
#: study retains them whatever the question asked, so median, maximum, top-five
#: and frequency over the same universe are projections of one measurement
#: rather than four runs of it.
METRICS: dict[str, MetricSpec] = {
    "return_pct": MetricSpec(
        key="return_pct",
        label="return",
        attribute="return_pct",
        unit="%",
        description="change from the price at the event to the price at the end of the window",
    ),
    "abs_move_pct": MetricSpec(
        key="abs_move_pct",
        label="absolute move",
        attribute="abs_move_pct",
        unit="%",
        description="size of the move regardless of direction",
        signed=False,
    ),
    "range_pct": MetricSpec(
        key="range_pct",
        label="trading range",
        attribute="range_pct",
        unit="%",
        description="high to low across the window, as a share of the price at the event",
        signed=False,
    ),
    "mfe_pct": MetricSpec(
        key="mfe_pct",
        label="best gain reached",
        attribute="mfe_pct",
        unit="%",
        description="furthest the price travelled up before the window ended",
        signed=False,
    ),
    "mae_pct": MetricSpec(
        key="mae_pct",
        label="worst drawdown reached",
        attribute="mae_pct",
        unit="%",
        description="furthest the price travelled down before the window ended",
    ),
    "realized_vol_pct": MetricSpec(
        key="realized_vol_pct",
        label="realised volatility",
        attribute="realized_vol_pct",
        unit="%",
        description="standard deviation of candle returns within the window",
        signed=False,
    ),
    "high": MetricSpec(
        key="high",
        label="highest price",
        attribute="high",
        unit="",
        description="highest price touched during the window",
        signed=False,
    ),
    "low": MetricSpec(
        key="low",
        label="lowest price",
        attribute="low",
        unit="",
        description="lowest price touched during the window",
        signed=False,
    ),
}

#: The metric assumed when a question does not name one.
DEFAULT_METRIC = "return_pct"


def metric_spec(key: str) -> MetricSpec | None:
    return METRICS.get(key)


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
    #: When this becomes knowable. Every condition field must be
    #: :attr:`TemporalScope.PRE_EVENT`: a cohort selected on anything else is
    #: selected by hindsight. Declared per field and checked by a test, so the
    #: rule survives someone adding a field without reading this comment.
    scope: TemporalScope = TemporalScope.PRE_EVENT
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

    def applies_to(self, event_type: EventType | None) -> bool:
        """Whether this field can be tested against a given event set.

        ``None`` means a dynamically resolved set, which belongs to no curated
        family. An event-agnostic field -- the BTC regime conditions, which
        describe the market rather than the event -- still applies. A field
        tied to specific families does not: there is no CPI surprise for a
        midterm election, and offering one would produce a cohort of zero that
        reads like a result.
        """
        if not self.event_types:
            return True
        return event_type is not None and event_type in self.event_types

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
        # Sorted, not iterated. Both of these are frozensets, and a frozenset
        # of strings iterates in an order that changes between processes --
        # so without this the planner received a vocabulary whose ordering
        # varied run to run. Harmless to the meaning, but this project
        # otherwise goes to some trouble to make the same question produce the
        # same request, and an unstable payload also makes it impossible to
        # tell whether the vocabulary has really changed.
        "categorical_operators": sorted(op.value for op in CATEGORICAL_OPERATORS),
        "numeric_operators": sorted(op.value for op in NUMERIC_OPERATORS),
        "units": [NUMERIC_UNIT_RULE],
        # Metrics are post-event by construction and therefore never appear in
        # condition_fields. The two lists are disjoint, and a test asserts it.
        "metrics": [
            f"{spec.key} ({spec.label}{', in ' + spec.unit if spec.unit else ''}) "
            f"-- {spec.description}"
            for spec in METRICS.values()
        ],
        "aggregations": [
            f"{agg.value} -- {_AGGREGATION_NOTES[agg]}" for agg in Aggregation
        ],
        "orders": [order.value for order in Order],
        "horizons": list(OUTCOME_HORIZONS),
    }


#: What each aggregation does, for the planner prompt. Kept beside the enum so a
#: member cannot be advertised without a description, or described without
#: existing.
_AGGREGATION_NOTES: dict[Aggregation, str] = {
    Aggregation.MEAN: "average across the matching releases; one number",
    Aggregation.MEDIAN: "middle value across the matching releases; one number",
    Aggregation.COUNT: "how many releases matched; one number",
    Aggregation.FREQUENCY: (
        "share of matching releases whose reading clears a threshold, as a "
        "percentage; requires a threshold and a comparison operator"
    ),
    Aggregation.ROWS: (
        "the releases themselves rather than a summary; use with an order and a "
        "limit. Largest is descending with limit 1, smallest is ascending with "
        "limit 1, top five is descending with limit 5"
    ),
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
    "DEFAULT_METRIC",
    "METRICS",
    "NUMERIC_FIELDS",
    "NUMERIC_OPERATORS",
    "NUMERIC_UNIT_RULE",
    "OUTCOME_HORIZONS",
    "SUPPORTED_ASSETS",
    "SUPPORTED_EVENTS",
    "Aggregation",
    "FieldKind",
    "FieldSpec",
    "MetricSpec",
    "Operator",
    "Order",
    "ResultShape",
    "ResolutionContext",
    "Resolver",
    "TemporalScope",
    "asset_label",
    "describe_vocabulary",
    "field_spec",
    "metric_spec",
    "reaction_windows",
    "window_for",
]
