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
    DEFAULT_METRIC,
    METRICS,
    NUMERIC_UNIT_RULE,
    OUTCOME_HORIZONS,
    SUPPORTED_ASSETS,
    Aggregation,
    MetricSpec,
    Operator,
    Order,
    ResultShape,
    TemporalScope,
    asset_label,
    field_spec,
    metric_spec,
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
        """Reject anything that is not a pre-event condition field.

        The metric check comes first and has its own message, because
        ``Condition(field="range_pct", ...)`` is not a typo -- it is a
        look-ahead selection, and "unknown field" would send whoever wrote it
        looking for a spelling mistake instead of telling them the cohort would
        have been chosen by its own outcome.
        """
        spec = metric_spec(value)
        if spec is not None:
            raise ValueError(
                f"{spec.label} ({value}) is measured after the event, so it "
                "cannot select which releases to study: the cohort would be "
                "chosen by the very thing being measured. Ask for it as the "
                "outcome instead -- a frequency over a threshold answers "
                '"how often did this happen?" without selecting on it.'
            )
        if value not in CONDITION_FIELDS:
            raise ValueError(
                f"Unknown condition field {value!r}. "
                f"Supported: {', '.join(sorted(CONDITION_FIELDS))}"
            )
        if CONDITION_FIELDS[value].scope is not TemporalScope.PRE_EVENT:
            raise ValueError(
                f"{CONDITION_FIELDS[value].label} is not knowable before the "
                "release, so it cannot select the cohort."
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
    """What is measured after the event, and how it is reported.

    The query projection: a metric column, a window to read it over, and an
    operation that turns that column into an answer. Deliberately generic --
    "largest range", "median return" and "how often did it move 2%" differ only
    in which values these fields take, so none of them needs code of its own.

    There is no MAX or MIN. A maximum is ``ROWS`` ordered descending with a
    limit of one, which returns the release as well as the number and so is
    strictly more useful than a bare statistic.
    """

    model_config = ConfigDict(frozen=True)

    #: Key into :data:`~backtool.research.vocabulary.METRICS`.
    metric: str = DEFAULT_METRIC
    horizon: str = DEFAULT_HORIZON
    aggregation: Aggregation = Aggregation.MEDIAN
    #: Sort direction. Required by ``ROWS``, meaningless otherwise.
    order: Order | None = None
    #: How many rows to report. Required by ``ROWS`` in this iteration; making
    #: it optional is how a fourth shape -- every observation, as a series --
    #: would arrive, which is why the restriction lives here rather than being
    #: assumed downstream.
    limit: int | None = Field(default=None, ge=1, le=100)
    #: The level a ``FREQUENCY`` counts against, in the metric's own units.
    threshold: float | None = None
    #: How the reading is compared with the threshold.
    threshold_operator: Operator | None = None

    @field_validator("horizon")
    @classmethod
    def _known_horizon(cls, value: str) -> str:
        if value not in OUTCOME_HORIZONS:
            raise ValueError(
                f"Unsupported horizon {value!r}. "
                f"Supported: {', '.join(OUTCOME_HORIZONS)}"
            )
        return value

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in METRICS:
            raise ValueError(
                f"Unsupported metric {value!r}. "
                f"Supported: {', '.join(METRICS)}"
            )
        return value

    @model_validator(mode="after")
    def _parts_fit_the_aggregation(self) -> Outcome:
        """Reject combinations that have no meaning.

        Each one would otherwise be silently ignored, which is the failure mode
        this layer exists to avoid: a question asking for the five largest and
        a question asking for the median must not be able to produce the same
        result because a field was dropped on the way through.
        """
        if self.aggregation is Aggregation.ROWS:
            if self.order is None:
                raise ValueError(
                    "Reporting the releases themselves needs a sort order: "
                    "ascending for the smallest, descending for the largest."
                )
            if self.limit is None:
                raise ValueError(
                    "Reporting the releases themselves needs a limit, such as "
                    "1 for the largest or 5 for the top five."
                )
        else:
            if self.order is not None or self.limit is not None:
                raise ValueError(
                    f"A {self.aggregation.value} is one number, so ordering "
                    "and limiting have nothing to act on. Ask for rows "
                    "instead to rank individual releases."
                )

        if self.aggregation is Aggregation.FREQUENCY:
            if self.threshold is None or self.threshold_operator is None:
                raise ValueError(
                    "A frequency needs a threshold and a comparison -- "
                    '"how often" is not a question until "how often what?" '
                    "is settled."
                )
            if not self.threshold_operator.is_numeric:
                raise ValueError(
                    f"{self.threshold_operator.value!r} cannot compare a "
                    "reading with a threshold. Use >, >=, < or <=."
                )
        elif self.threshold is not None or self.threshold_operator is not None:
            raise ValueError(
                f"A {self.aggregation.value} counts nothing against a "
                "threshold, so supplying one would have no effect."
            )
        return self

    @property
    def window(self) -> str:
        return OUTCOME_HORIZONS[self.horizon]

    @property
    def spec(self) -> MetricSpec:
        return METRICS[self.metric]

    def describe(self) -> str:
        """The projection as a reader would say it.

        This is the line they confirm before anything runs, so it has to carry
        every part that changes the answer: the metric, the window, the
        operation, and any threshold or limit.
        """
        spec = self.spec
        subject = f"{self.horizon} {spec.label}"

        if self.aggregation is Aggregation.ROWS:
            assert self.limit is not None and self.order is not None
            extreme = "largest" if self.order is Order.DESCENDING else "smallest"
            if self.limit == 1:
                return f"the {extreme} {subject}"
            return f"the {self.limit} {extreme} {subject} readings"

        if self.aggregation is Aggregation.FREQUENCY:
            assert self.threshold is not None and self.threshold_operator is not None
            return (
                f"how often the {subject} was "
                f"{self.threshold_operator.phrase().removeprefix('is ')} "
                f"{spec.format(self.threshold)}"
            )

        if self.aggregation is Aggregation.COUNT:
            return f"how many releases had a usable {subject}"

        return f"{self.aggregation.value} {subject}"


class Comparison(BaseModel):
    """The group a result is measured against.

    A second filtered group, nothing more. That is the whole generalisation:
    "all usable releases" was never a different kind of thing from "cooler
    releases", it was the group selected by *no* filters. Representing it as
    one means a baseline comparison and a hotter-versus-cooler comparison run
    through identical code, and there is no pair of concepts to keep in step.

    What is deliberately absent is a metric or a window. Both live on
    :class:`Outcome`, shared by the whole thesis, so the two sides cannot be
    measuring different things -- not as a rule that is checked, but as a
    statement that cannot be written down.
    """

    model_config = ConfigDict(frozen=True)

    #: Filters selecting this group. Empty means every usable release, which
    #: is the classic baseline.
    conditions: tuple[Condition, ...] = ()

    @property
    def is_baseline(self) -> bool:
        """Whether this group is 'everything', rather than a named selection."""
        return not self.conditions

    def describe(self, event_type: EventType) -> str:
        return describe_group(self.conditions, event_type)

    def with_condition(self, condition: Condition) -> Comparison:
        """A copy with one more filter, under the same rule the left side uses.

        Shares :meth:`Condition.supersedes`, so a numeric bound added to both
        groups replaces the bound on the same end for each of them rather than
        behaving one way on the left and another on the right.
        """
        kept = tuple(c for c in self.conditions if not condition.supersedes(c))
        return Comparison(conditions=(*kept, condition))


def describe_group(conditions: tuple[Condition, ...], event_type: EventType) -> str:
    """What a filtered group is called, from its filters alone.

    Used for both sides, which is what stops one of them being labelled
    "baseline" or "typical" when the reader named two groups. Only a group with
    no filters can be described as everything, because only that group is.
    """
    if not conditions:
        return f"All usable {event_type.value} releases"
    return " and ".join(condition.describe() for condition in conditions)


class Thesis(BaseModel):
    """A question this system can actually answer, or say why it cannot."""

    model_config = ConfigDict(frozen=True)

    #: The question as asked, kept verbatim so the result can be shown beside it.
    question: str = ""
    asset: str = "BTCUSDT"
    event_type: EventType = EventType.CPI
    conditions: tuple[Condition, ...] = ()
    outcome: Outcome = Outcome()
    #: The group this one is measured against, when that question makes sense.
    #: ``None`` for a result that is not a comparison: "the largest 24h range"
    #: has nothing to be larger *than*. An empty ``Comparison`` is every usable
    #: release, which is the baseline; a filtered one is the other half of an
    #: A-versus-B question.
    versus: Comparison | None = None
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

    @model_validator(mode="after")
    def _comparison_is_meaningful(self) -> Thesis:
        """Reject a comparison that cannot say anything.

        Every rejected case would otherwise produce a confident-looking
        difference that is zero or nonsensical by construction, which is worse
        than an error because it reads as a finding rather than as a mistake.
        """
        if self.versus is None:
            return self
        if self.outcome.aggregation is Aggregation.ROWS:
            raise ValueError(
                "A ranked list of releases has nothing to compare against. "
                "Drop the comparison, or ask for a summary statistic instead."
            )
        if self._same_conditions(self.conditions, self.versus.conditions):
            if not self.conditions:
                raise ValueError(
                    "Both sides are every usable release, so the difference "
                    "would be zero by construction. Add a condition to at "
                    "least one side."
                )
            raise ValueError(
                "Both sides select the same releases "
                f"({describe_group(self.conditions, self.event_type)}), so the "
                "difference would be zero by construction. Change one side."
            )
        for condition in self.versus.conditions:
            spec = CONDITION_FIELDS[condition.field]
            if not spec.applies_to(self.event_type):
                raise ValueError(
                    f"{spec.label} cannot be tested against "
                    f"{self.event_type.value} releases."
                )
        return self

    @staticmethod
    def _same_conditions(
        left: tuple[Condition, ...], right: tuple[Condition, ...]
    ) -> bool:
        """Whether two filter sets select identically.

        Order-insensitive: conditions are a conjunction, so the same filters
        listed in a different order are the same group, and comparing tuples
        directly would let a reordering slip past as a real comparison.
        """
        return {c.model_dump_json() for c in left} == {
            c.model_dump_json() for c in right
        }

    @property
    def is_runnable(self) -> bool:
        """Whether this can be executed, or still needs the reader."""
        return not self.unresolved

    @property
    def shape(self) -> ResultShape:
        """What kind of answer this question produces.

        Derived from the query rather than declared, which is what keeps result
        presentation from needing a case per question. Two fields decide it:
        whether the aggregation collapses the rows, and whether there is
        something to compare against.
        """
        if self.outcome.aggregation is Aggregation.ROWS:
            return ResultShape.RANKED
        if self.versus is not None:
            return ResultShape.COMPARISON
        return ResultShape.SCALAR

    @property
    def needs_regime(self) -> bool:
        """Whether any condition requires the market-regime pass.

        Checked so a thesis about CPI surprise alone does not pay for two years
        of hourly history it will not look at.
        """
        return any(
            CONDITION_FIELDS[condition.field].requires_regime
            for condition in self.all_conditions
        )

    @property
    def all_conditions(self) -> tuple[Condition, ...]:
        """Every condition either side uses.

        Both groups are measured from one study, so anything either of them
        filters on has to be resolved -- a comparison of bullish against
        bearish needs the regime pass even though neither side alone would
        look different from a question that did not.
        """
        if self.versus is None:
            return self.conditions
        return (*self.conditions, *self.versus.conditions)

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
        items = [
            ("Asset", f"{asset_label(self.asset)} ({self.asset})"),
            ("Event", f"{self.event_type.value} releases"),
        ]
        # Two named groups are shown as two rows rather than as a cohort and a
        # baseline, because the reader asked about both and neither is the
        # other's background. A plain baseline keeps its old wording.
        if self.versus is not None and not self.versus.is_baseline:
            items.append(("Left group", conditions))
            items.append(
                ("Right group", describe_group(self.versus.conditions, self.event_type))
            )
        else:
            items.append(("Condition", conditions))
            if self.versus is not None:
                items.append(("Compared with", self.versus.describe(self.event_type)))
        items.append(("Measure", f"{self.outcome.describe()} for {self.asset}"))
        items.append(("Sample", f"up to the last {self.event_count} releases"))
        return items

    def headline(self) -> str:
        """A one-line restatement, used when the question itself is missing."""
        if self.question:
            return self.question
        where = (
            " where " + " and ".join(c.describe() for c in self.conditions)
            if self.conditions
            else ""
        )
        return (
            f"{asset_label(self.asset)} {self.outcome.describe()} after "
            f"{self.event_type.value} releases{where}"
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

    def with_shared_condition(self, condition: Condition) -> Thesis:
        """A copy with one more filter on *both* groups.

        "What about only when BTC was below its 200-day average?" narrows the
        question, not one side of it. Applying it to one group would silently
        turn a comparison of two conditions into a comparison of two different
        questions.

        Reuses :meth:`with_condition` on each side rather than reimplementing
        the replace-or-append rule, so a shared numeric bound behaves the same
        as a single-group one.
        """
        narrowed = self.with_condition(condition)
        if self.versus is None:
            return narrowed
        return narrowed.model_copy(
            update={"versus": self.versus.with_condition(condition)}
        )

    def versus_group(self, conditions: tuple[Condition, ...]) -> Thesis:
        """A copy whose right-hand group is the one given.

        "Compare hotter with in-line instead" replaces the right group and
        leaves the left one alone, which is the behaviour that makes a
        comparison editable rather than restated.
        """
        return self.model_copy(update={"versus": Comparison(conditions=conditions)})


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
    "Comparison",
    "Condition",
    "Outcome",
    "Thesis",
    "describe_group",
    "unsupported_reason",
]
