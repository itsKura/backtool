"""A market thesis, structured.

This is the contract between what somebody asked and what gets measured. It is
deliberately *not* :class:`~backtool.research.spec.ResearchSpec`, which already
exists and means something narrower -- the parameters of one measurement run:
symbol, interval, windows, cutoff. A thesis sits a layer above that and
compiles down to one.

    Thesis           "does BTC do better after cooler CPI?"
      |  compile
    ResearchSpec     measure BTCUSDT over the last N CPI releases at 5m
      |  run_study
    StudyResult      per-release, per-window numbers
      |  split on Thesis.conditions
    Cohort + baseline

Keeping the two apart is what lets the question be edited without touching the
engine, and lets the engine stay reproducible: a ``ResearchSpec`` fingerprint
still identifies exactly one set of numbers, regardless of which thesis
produced it.

Follow-ups
----------
A thesis is immutable and composes. "What about only when BTC was down that
week?" is :meth:`Thesis.with_condition` on the thesis already on screen, not a
new question parsed from nothing. That is why conversation state holds a thesis
rather than a transcript: the second question is a modification of the first,
and the structure makes it one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backtool.core.types import EventType
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    DEFAULT_HORIZON,
    NUMERIC_UNIT_RULE,
    OUTCOME_HORIZONS,
    SUPPORTED_ASSETS,
    BaselineKind,
    Operator,
    OutcomeMetric,
    asset_label,
    field_spec,
)


class Condition(BaseModel):
    """One filter that decides whether a release belongs to the cohort.

    Two shapes in one model, chosen by the operator: a categorical test against
    one of a field's declared states, or a numeric test against a threshold.
    One model rather than a union because every consumer -- selection,
    description, follow-up editing -- keys off ``field``, and a union would make
    each of them ask which variant it was handed before doing the same thing.

    The pairing is enforced, not assumed. ``cpi_surprise > 3`` and
    ``btc_return_7d is bearish`` are both rejected at construction, because
    each would otherwise select nothing and a cohort of zero looks like a
    finding rather than a mistake.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    operator: Operator = Operator.IS
    #: The named state tested, for a categorical condition. Empty for numeric.
    value: str = ""
    #: The number tested against, for a numeric condition. ``None`` for
    #: categorical. In the units of :data:`NUMERIC_UNIT_RULE`: a 3% fall is
    #: ``-3.0``, never ``-0.03``.
    threshold: float | None = None

    @field_validator("field")
    @classmethod
    def _known_field(cls, value: str) -> str:
        if value not in CONDITION_FIELDS:
            raise ValueError(
                f"Unknown condition field {value!r}. "
                f"Supported: {', '.join(sorted(CONDITION_FIELDS))}"
            )
        return value

    @model_validator(mode="after")
    def _operator_suits_the_field(self) -> Condition:
        spec = CONDITION_FIELDS[self.field]

        if self.operator.is_numeric:
            if not spec.is_numeric:
                raise ValueError(
                    f"{spec.label} is a named state, not a number, so it "
                    f"cannot be compared with {self.operator.value!r}. "
                    f"Use is or is not with one of: "
                    f"{', '.join(sorted(spec.values))}."
                )
            if self.threshold is None:
                raise ValueError(
                    f"{spec.label} {self.operator.value} needs a number to "
                    "compare against."
                )
            if not spec.accepts_threshold(self.threshold):
                low, high = spec.bounds or (0.0, 0.0)
                raise ValueError(
                    f"{self.threshold:g} is outside the range {spec.label} "
                    f"can take ({low:g} to {high:g}). {NUMERIC_UNIT_RULE}"
                )
            if self.value:
                raise ValueError(
                    f"A numeric condition on {spec.label} carries a threshold, "
                    f"not the value {self.value!r}."
                )
            return self

        if spec.is_numeric:
            raise ValueError(
                f"{spec.label} is a number, so it cannot be tested with "
                f"{self.operator.value!r}. Use >, >=, < or <= with a "
                f"threshold. {NUMERIC_UNIT_RULE}"
            )
        if self.threshold is not None:
            raise ValueError(
                f"{spec.label} is a named state, so a threshold of "
                f"{self.threshold:g} has nothing to compare against."
            )
        if not spec.accepts(self.value):
            raise ValueError(
                f"{spec.label} cannot be {self.value!r}. "
                f"Supported: {', '.join(sorted(spec.values))}"
            )
        return self

    @property
    def is_numeric(self) -> bool:
        return self.operator.is_numeric

    def describe(self) -> str:
        """The condition as a reader would say it.

        For a numeric condition this is the only place the threshold is shown
        back before the test runs, which makes it the defence against a unit
        mistake: ``BTC 7-day return is below -0.03%`` reads obviously wrong to
        anyone who meant a 3% fall.
        """
        spec = CONDITION_FIELDS[self.field]
        if self.is_numeric:
            assert self.threshold is not None  # guaranteed by the validator
            return (
                f"{spec.label} {self.operator.phrase()} "
                f"{spec.format_number(self.threshold)}"
            )
        return f"{spec.label} {self.operator.phrase()} {spec.display(self.value)}"

    def supersedes(self, other: Condition) -> bool:
        """Whether adding this condition should remove ``other``.

        Same field and same role. Two categorical filters on one field are
        redundant or contradictory, so one replaces the other. Two numeric
        filters are only in conflict when they constrain the *same end* of the
        range: ``>= -5`` and ``<= -2`` together mean "down between 2% and 5%",
        and evicting one of them would answer a question nobody asked.
        """
        if other.field != self.field:
            return False
        return self.operator.bound == other.operator.bound

    def matches(self, observed: str | None) -> bool:
        """Whether a release with this observed state belongs in the cohort.

        An unmeasurable value never matches, in either direction. A release
        whose 200-day average could not be computed is not "not above" it --
        it is unknown, and treating unknown as the negative case would fill a
        cohort with releases selected for having less data.
        """
        if observed is None:
            return False
        if self.operator is Operator.IS:
            return observed == self.value
        return observed != self.value

    def matches_number(self, observed: float | None) -> bool:
        """Whether a release with this reading belongs in the cohort.

        An unmeasurable reading never matches, for the same reason as above.
        Comparison is on the reader's scale throughout: both sides are percent
        where the field is a percent, so nothing is rescaled here and there is
        no second place for a unit to be reinterpreted.
        """
        if observed is None or self.threshold is None:
            return False
        if self.operator is Operator.ABOVE:
            return observed > self.threshold
        if self.operator is Operator.AT_LEAST:
            return observed >= self.threshold
        if self.operator is Operator.BELOW:
            return observed < self.threshold
        if self.operator is Operator.AT_MOST:
            return observed <= self.threshold
        return False


class Outcome(BaseModel):
    """What is measured after the event."""

    model_config = ConfigDict(frozen=True)

    metric: OutcomeMetric = OutcomeMetric.RETURN
    horizon: str = DEFAULT_HORIZON

    @field_validator("horizon")
    @classmethod
    def _known_horizon(cls, value: str) -> str:
        if value not in OUTCOME_HORIZONS:
            raise ValueError(
                f"Unsupported horizon {value!r}. "
                f"Supported: {', '.join(OUTCOME_HORIZONS)}"
            )
        return value

    def describe(self) -> str:
        return f"{self.horizon} {self.metric.value}"

    @property
    def window(self) -> str:
        return OUTCOME_HORIZONS[self.horizon]


class Baseline(BaseModel):
    """What the cohort is measured against."""

    model_config = ConfigDict(frozen=True)

    kind: BaselineKind = BaselineKind.ALL_EVENTS

    def describe(self, event_type: EventType) -> str:
        return f"All usable {event_type.value} releases"


class Thesis(BaseModel):
    """A question this system can actually answer, or say why it cannot."""

    model_config = ConfigDict(frozen=True)

    #: The question as asked, kept verbatim so the result can be shown beside it.
    question: str = ""
    asset: str = "BTCUSDT"
    event_type: EventType = EventType.CPI
    conditions: tuple[Condition, ...] = ()
    outcome: Outcome = Outcome()
    baseline: Baseline = Baseline()
    #: How many releases of this type to consider.
    event_count: int = Field(default=40, ge=1, le=200)

    #: Anything the planner could not determine and would have had to guess at.
    #: A thesis carrying these is shown to the reader as a question, never run:
    #: a silent default on a field that changes the answer is the failure this
    #: whole layer exists to prevent.
    unresolved: tuple[str, ...] = ()

    @field_validator("asset")
    @classmethod
    def _known_asset(cls, value: str) -> str:
        symbol = value.strip().upper()
        if symbol not in SUPPORTED_ASSETS:
            raise ValueError(
                f"Unsupported asset {value!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_ASSETS))}"
            )
        return symbol

    @model_validator(mode="after")
    def _conditions_apply(self) -> Thesis:
        for condition in self.conditions:
            spec = CONDITION_FIELDS[condition.field]
            if not spec.applies_to(self.event_type):
                raise ValueError(
                    f"{spec.label} cannot be tested against "
                    f"{self.event_type.value} releases."
                )
        return self

    @property
    def is_runnable(self) -> bool:
        """Whether this can be executed, or still needs the reader."""
        return not self.unresolved

    @property
    def needs_regime(self) -> bool:
        """Whether any condition requires the market-regime pass.

        Checked so a thesis about CPI surprise alone does not pay for two years
        of hourly history it will not look at.
        """
        return any(
            CONDITION_FIELDS[condition.field].requires_regime
            for condition in self.conditions
        )

    def describe_items(self) -> list[tuple[str, str]]:
        """The thesis as label/value pairs, for the specification preview.

        This is what a reader confirms before anything runs. Every field that
        changes the answer appears here; nothing that changes the answer is
        decided anywhere else.
        """
        conditions = (
            " and ".join(condition.describe() for condition in self.conditions)
            if self.conditions
            else "none -- every usable release"
        )
        return [
            ("Asset", f"{asset_label(self.asset)} ({self.asset})"),
            ("Event", f"{self.event_type.value} releases"),
            ("Condition", conditions),
            ("Outcome", f"{self.outcome.horizon} {self.asset} {self.outcome.metric.value}"),
            ("Baseline", self.baseline.describe(self.event_type)),
            ("Sample", f"up to the last {self.event_count} releases"),
        ]

    def headline(self) -> str:
        """A one-line restatement, used when the question itself is missing."""
        if self.question:
            return self.question
        conditions = (
            " and ".join(condition.describe() for condition in self.conditions)
            or "all releases"
        )
        return (
            f"{asset_label(self.asset)} {self.outcome.horizon} "
            f"{self.outcome.metric.value} after {self.event_type.value} "
            f"when {conditions}"
        )

    def with_condition(self, condition: Condition) -> Thesis:
        """A copy with one more filter, replacing any it supersedes.

        Replacing rather than appending where the two conflict: a follow-up
        that says "only when BTC was down that week" means *instead of*
        whatever the trend filter was, not *as well as*. But a numeric bound
        only conflicts with a bound on the same end of the range, so
        ``>= -5`` and ``<= -2`` coexist and express a range.
        See :meth:`Condition.supersedes`.
        """
        kept = tuple(c for c in self.conditions if not condition.supersedes(c))
        return self.model_copy(update={"conditions": (*kept, condition)})

    def without_condition(self, field_key: str) -> Thesis:
        """A copy with one filter removed."""
        return self.model_copy(
            update={
                "conditions": tuple(
                    c for c in self.conditions if c.field != field_key
                )
            }
        )

    def with_outcome(self, outcome: Outcome) -> Thesis:
        return self.model_copy(update={"outcome": outcome})


def unsupported_reason(
    *, asset: str | None = None, event: str | None = None, field_key: str | None = None
) -> str | None:
    """Why a requested concept cannot be tested, phrased for a reader.

    Returned instead of a guess. "That dataset is not connected" is a worse
    answer than a number and a far better one than a wrong number.
    """
    if asset and asset.strip().upper() not in SUPPORTED_ASSETS:
        return (
            f"{asset} is not connected. Backtool can measure "
            f"{' and '.join(sorted(SUPPORTED_ASSETS))} today."
        )
    if event and event.strip().upper() not in {e.value for e in EventType}:
        return (
            f"{event} is not in the calendar. Backtool covers CPI and FOMC "
            "releases today."
        )
    if field_key and field_spec(field_key) is None:
        return (
            f"{field_key} is not a condition Backtool can test yet. "
            f"Available: {', '.join(sorted(CONDITION_FIELDS))}."
        )
    return None


__all__ = [
    "Baseline",
    "Condition",
    "Outcome",
    "Thesis",
    "unsupported_reason",
]
