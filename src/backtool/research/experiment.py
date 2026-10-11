"""Running a thesis: compile, measure, split, compare.

Every number this produces comes from code that already existed.
:func:`~backtool.research.runner.run_study` measures, :func:`build_cohort`
groups, :func:`~backtool.research.aggregate.aggregate_window` computes. This
module decides *which releases go in which group* and nothing else, which is
the one thing a question actually determines.

The comparison is the point
---------------------------
"Cooler CPI gives +1.2% over 24 hours" is a number with nothing to be better or
worse than. BTC drifts upward over most 24-hour periods, so a positive figure
is unremarkable until it is set against what the same releases did on average.
Every finding therefore carries a baseline and the difference between them, and
the difference is what the headline reports.

The baseline is every usable release of the same type -- *including* the cohort.
Excluding them would compare cooler releases against hotter-and-in-line ones,
which is a different question from "better than a typical release of this kind"
and would overstate the gap. The overlap is stated in the methodology.
"""

from __future__ import annotations

import datetime as dt
import logging

from pydantic import BaseModel, ConfigDict

from backtool.config import Settings
from backtool.core.time import UTC
from backtool.core.types import Interval
from backtool.data.service import MarketDataService
from backtool.events import load_events
from backtool.events.models import MarketEvent
from backtool.events.releases import SCENARIO_RULES, load_releases
from backtool.events.resolution import compile_event_set, curated_set_key
from backtool.research.aggregate import (
    MetricReading,
    WindowAggregate,
    aggregate_values,
    count_matching,
    metric_series,
)
from backtool.research.cohort import Cohort, build_cohort, sample_quality
from backtool.research.regime import (
    SETUP_INTERVAL,
    SETUP_SYMBOL,
    compute_setup,
    history_window,
)
from backtool.research.results import EventResult, StudyResult
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.thesis import Condition, Outcome, Thesis, describe_group
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    Aggregation,
    MetricSpec,
    Order,
    ResolutionContext,
    ResultShape,
    interval_for,
    reaction_windows,
)

logger = logging.getLogger(__name__)

#: Candle interval a short-horizon thesis is measured at. Fine enough to
#: resolve a five-minute window, which is the shortest the vocabulary offers.
#:
#: No longer the only one: a thesis asking about a week or more runs on hourly
#: candles instead, because two years at five minutes is a quarter of a million
#: candles for a single event. Which one applies follows from the horizon --
#: see :func:`~backtool.research.vocabulary.interval_for` -- so the interval and
#: the windows cannot be set to disagree.
THESIS_INTERVAL = Interval.M5


class Observation(BaseModel):
    """One release that qualified, with what put it there and what followed."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_date: dt.date
    #: The value of every condition field for this release, so the reader can
    #: check membership rather than take it on trust.
    conditions: dict[str, str]
    #: Return at each measured horizon, keyed by horizon. Kept whatever the
    #: query asked for, because a reader looking at a 24-hour figure almost
    #: always wants to know whether it was already there at one hour.
    returns: dict[str, float | None]
    #: The queried metric for this release, at the queried window. The column
    #: ordering and limiting act on; ``None`` when it could not be measured.
    value: float | None = None


class ResearchFinding(BaseModel):
    """Everything one answered question produced.

    Shaped so the same object can be rendered in chat today and saved, shared,
    re-run or compared later without the UI having recomputed anything. The
    renderer reads this; it does not reach back into the engine.
    """

    model_config = ConfigDict(frozen=True)

    thesis: Thesis
    #: The execution spec that produced the numbers. Carries the fingerprint,
    #: so a finding can always be traced to a reproducible run.
    spec: ResearchSpec
    generated_at: dt.datetime

    cohort: Cohort
    #: ``None`` when the question is not a comparison.
    #: The group this one is set against, when the question is a comparison.
    #: Every usable release when the comparison is a baseline, a second named
    #: selection when the reader asked about two groups.
    comparison: Cohort | None = None
    #: Every qualifying release, already in the query's order. Ordering happens
    #: once, here; the limit is applied when the rows are read, so changing
    #: "largest" to "top five" re-slices this tuple and measures nothing again.
    observations: tuple[Observation, ...] = ()

    #: The scalar answer, for a query that has one. In the metric's own units,
    #: or percent for a frequency.
    value: float | None = None
    #: The same statistic over the comparison group, where there is one.
    comparison_value: float | None = None
    #: Releases behind ``value``: the ones whose reading entered it.
    sample_size: int = 0
    comparison_sample_size: int = 0
    #: For a frequency, how many readings cleared the threshold. Carried rather
    #: than recovered from the percentage, which rounds back to the wrong
    #: integer often enough to matter over a small sample.
    matched_count: int | None = None
    #: The same count for the comparison group, so both sides of a frequency
    #: comparison can show the fraction behind their percentage.
    comparison_matched_count: int | None = None

    #: Releases considered but not in the cohort, counted by reason.
    excluded: dict[str, int] = {}
    #: Anything the reader should know before trusting the figures.
    warnings: tuple[str, ...] = ()

    @property
    def shape(self) -> ResultShape:
        return self.thesis.shape

    @property
    def metric(self) -> MetricSpec:
        return self.thesis.outcome.spec

    @property
    def result_in_metric_units(self) -> bool:
        """Whether the scalar result is a value of the metric.

        The question anything displaying ``value`` has to ask first. A median
        return belongs on the same axis as the observations it summarises; a
        frequency is a share of releases and belongs nowhere near them.
        """
        return self.thesis.outcome.aggregation.yields_metric_units

    @property
    def result_quantity(self) -> str:
        """What the scalar result *is*, for an axis caption.

        Falls back to the metric's own label, which is right exactly when the
        result is a value of that metric.
        """
        return self.thesis.outcome.aggregation.result_noun or self.metric.label

    def describe_count(self, matched: int | None, size: int) -> str:
        """The fraction behind a percentage, where one exists.

        A share is far easier to judge beside the count it came from: 75% of
        eight releases and 75% of eighty are the same number and different
        evidence.
        """
        if matched is None:
            return ""
        return f"{matched} of {size}"

    @property
    def window(self) -> str:
        return self.thesis.outcome.window

    @property
    def rows(self) -> tuple[Observation, ...]:
        """The releases this result reports, cut to the limit.

        A slice of ``observations`` rather than a separate measurement, which
        is what makes "the largest" and "the five largest" the same run.
        """
        limit = self.thesis.outcome.limit
        return self.observations[:limit] if limit else self.observations

    @property
    def cohort_metrics(self) -> WindowAggregate | None:
        return self.cohort.windows.get(self.window)

    @property
    def comparison_metrics(self) -> WindowAggregate | None:
        return (
            self.comparison.windows.get(self.window) if self.comparison else None
        )

    @property
    def difference_pp(self) -> float | None:
        """Left statistic minus right statistic, in the metric's units.

        Only exists for a comparison. Both sides share one ``Outcome``, so the
        subtraction is always between like and like -- not because that is
        checked, but because there is no field in which they could differ.
        """
        if self.value is None or self.comparison_value is None:
            return None
        return self.value - self.comparison_value

    def format_value(self, value: float | None) -> str:
        """A result in the units it was measured in.

        A frequency is a percentage of releases whatever the metric was, so it
        is formatted as one rather than inheriting the metric's unit -- "72% of
        releases" and "72% range" are different claims.
        """
        if value is None:
            return "not measured"
        aggregation = self.thesis.outcome.aggregation
        if aggregation is Aggregation.FREQUENCY:
            return f"{value:.0f}%"
        if aggregation is Aggregation.COUNT:
            return f"{value:.0f}"
        return self.metric.format(value)

    def counted(self, n: int) -> str:
        """``3 releases``, ``1 event`` -- the count with the right noun.

        Both halves vary. The noun follows the thesis, because a concept
        resolved from the web is not a release of anything; the plural follows
        the count, which stopped being academic when long horizons made a
        sample of one the ordinary outcome rather than a degenerate one.
        """
        noun = self.thesis.event_noun
        return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    @property
    def _plural(self) -> str:
        """The event noun, pluralised, for sentences with no count in them."""
        return f"{self.thesis.event_noun}s"

    def verdict(self) -> str:
        """A neutral one-line reading, derived from the result and its shape.

        Deterministic, not written by a model: this sentence is the first thing
        a reader sees and it has to be derivable from the numbers beneath it.
        Three shapes, three sentences -- not three handlers, because the shape
        is read off the query rather than carried as a question type.
        """
        outcome = self.thesis.outcome
        horizon = outcome.horizon
        subject = f"{horizon} {self.metric.label}"

        if self.shape is ResultShape.RANKED:
            if not self.rows:
                return (
                    f"No {self.thesis.event_noun} had a usable reading, so "
                    "there is nothing to rank."
                )
            top = self.rows[0]
            extreme = "largest" if outcome.order is Order.DESCENDING else "smallest"
            return (
                f"The {extreme} {subject} was "
                f"{self.metric.format(top.value or 0.0)} on "
                f"{top.event_date:%d %B %Y}."
            )

        if self.value is None:
            return (
                f"No {self.thesis.event_noun} matched with a usable reading, "
                "so there is nothing to report."
            )

        # Gated on the shape, so a frequency *comparison* falls through to the
        # two-group sentence below. Reporting one side's share and stopping
        # would answer half a question the reader asked all of.
        if (
            outcome.aggregation is Aggregation.FREQUENCY
            and self.shape is not ResultShape.COMPARISON
        ):
            assert outcome.threshold is not None
            comparison = (
                outcome.threshold_operator.phrase().removeprefix("is ")
                if outcome.threshold_operator
                else "above"
            )
            counted = self.describe_count(self.matched_count, self.sample_size)
            return (
                f"The {subject} was {comparison} "
                f"{self.metric.format(outcome.threshold)} after "
                f"{self.value:.0f}% of {self._plural}"
                f"{f' ({counted})' if counted else ''}."
            )

        if (
            outcome.aggregation is Aggregation.COUNT
            and self.shape is not ResultShape.COMPARISON
        ):
            return f"{self.counted(int(self.value))} had a usable {subject}."

        if self.shape is ResultShape.SCALAR:
            return (
                f"The {outcome.aggregation.value} {subject} was "
                f"{self.format_value(self.value)} across "
                f"{self.counted(self.sample_size)}."
            )

        difference = self.difference_pp
        if difference is None:
            return (
                "The other group could not be measured, so there is nothing "
                "to compare against."
            )
        size = abs(difference)
        measure = f"{outcome.aggregation.value} {horizon} {self.metric.label}"

        # A group the reader named is called by its name. Only an unfiltered
        # group can be "a typical release", because only an unfiltered group
        # is every release.
        against = (
            f"a typical {self.thesis.event_noun}"
            if self.comparison is not None and self.comparison.key == "right"
            and self.thesis.versus is not None
            and self.thesis.versus.is_baseline
            else self.comparison.label
            if self.comparison
            else "the other group"
        )

        # A result that is not in the metric's units cannot be described as a
        # higher or lower *return*: both sides are shares of the sample, so the
        # sentence states the two shares and the gap between them.
        if not self.result_in_metric_units:
            left = self.describe_count(self.matched_count, self.sample_size)
            right = self.describe_count(
                self.comparison_matched_count, self.comparison_sample_size
            )
            return (
                f"{self.cohort.label}: {self.format_value(self.value)}"
                f"{f' ({left})' if left else ''} against {against}: "
                f"{self.format_value(self.comparison_value)}"
                f"{f' ({right})' if right else ''}, a difference of "
                f"{difference:+.2f} percentage points."
            )

        if size < 0.1:
            return (
                f"No meaningful difference: the {measure} for "
                f"{self.cohort.label} is within 0.1 percentage points of "
                f"{against}."
            )
        direction = "higher" if difference > 0 else "lower"
        return (
            f"{self.cohort.label}: {measure} was {direction} than "
            f"{against}, by {size:.2f} percentage points."
        )


def run_thesis(thesis: Thesis, *, as_of: dt.datetime | None = None) -> ResearchFinding:
    """Measure a thesis against real data.

    Args:
        thesis: A runnable thesis. One carrying unresolved fields is a question
            for the reader, not a job for this function.
        as_of: Cutoff; releases at or after it are excluded. Defaults to now.

    Returns:
        A :class:`ResearchFinding`. An empty cohort is a result, not an error:
        "nothing matched" is an answer, and the reader needs to see which
        conditions produced it.

    Raises:
        ValueError: if the thesis still has unresolved fields.
    """
    if not thesis.is_runnable:
        raise ValueError(
            "This thesis still needs: " + "; ".join(thesis.unresolved)
        )

    now = dt.datetime.now(tz=UTC)
    cutoff = as_of or now

    calendar, event_set = _calendar(thesis)
    spec = ResearchSpec(
        symbol=thesis.asset,
        event_type=thesis.event_type,
        event_set=event_set,
        event_count=thesis.event_count,
        interval=interval_for(thesis.outcome.horizon),
        windows=reaction_windows(thesis.outcome.horizon),
        as_of=cutoff,
    )

    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, calendar, now=now)
        context = _context(study, service, thesis, cutoff)

    # Both sides are selected the same way, from the same study, by the same
    # function. A baseline is the call with no conditions -- there is no second
    # path for it, which is what stops the two drifting apart in what they
    # count as usable.
    members, excluded = _select(study, thesis.conditions, thesis, context)

    cohort = build_cohort(
        study,
        key="left",
        label=describe_group(thesis.conditions, thesis.event_phrase),
        description=_group_description(thesis.conditions, thesis.event_phrase),
        members=members,
    )
    comparison: Cohort | None = None
    comparison_members: list[EventResult] = []
    if thesis.versus is not None:
        comparison_members, _ = _select(study, thesis.versus.conditions, thesis, context)
        comparison = build_cohort(
            study,
            key="right",
            label=thesis.versus.describe(thesis.event_phrase),
            description=_group_description(
                thesis.versus.conditions, thesis.event_phrase
            ),
            members=comparison_members,
        )

    # The projection. Everything above decided *which* releases; this decides
    # which column of them to read and what to do with it. No measurement
    # happens here -- every value already exists on the events.
    outcome = thesis.outcome
    readings = metric_series(members, outcome.window, outcome.spec)
    values = [reading.value for reading in readings]

    ordered = _ordered(readings, outcome)
    value = aggregate_values(
        values,
        outcome.aggregation,
        threshold=outcome.threshold,
        operator=outcome.threshold_operator,
    )

    comparison_value: float | None = None
    comparison_values: list[float] = []
    if comparison is not None:
        comparison_values = [
            reading.value
            for reading in metric_series(
                comparison_members, outcome.window, outcome.spec
            )
        ]
        comparison_value = aggregate_values(
            comparison_values,
            outcome.aggregation,
            threshold=outcome.threshold,
            operator=outcome.threshold_operator,
        )

    counts_matches = (
        outcome.aggregation is Aggregation.FREQUENCY
        and outcome.threshold is not None
        and outcome.threshold_operator is not None
    )
    matched = (
        count_matching(values, outcome.threshold_operator, outcome.threshold)
        if counts_matches and outcome.threshold_operator and outcome.threshold is not None
        else None
    )
    comparison_matched = (
        count_matching(
            comparison_values, outcome.threshold_operator, outcome.threshold
        )
        if counts_matches
        and comparison is not None
        and outcome.threshold_operator
        and outcome.threshold is not None
        else None
    )

    return ResearchFinding(
        thesis=thesis,
        spec=spec,
        generated_at=now,
        cohort=cohort,
        comparison=comparison,
        observations=_observations(ordered, thesis, context),
        value=value,
        comparison_value=comparison_value,
        sample_size=len(values),
        comparison_sample_size=len(comparison_values),
        matched_count=matched,
        comparison_matched_count=comparison_matched,
        excluded=excluded,
        warnings=_warnings(thesis, len(values), len(comparison_values)),
    )


def _ordered(
    readings: list[MetricReading], outcome: Outcome
) -> list[tuple[EventResult, float | None]]:
    """Put the qualifying releases in the order the query asked for.

    Sorted by the queried metric when one was requested, newest first
    otherwise. Ordering happens once over the whole set and the limit is
    applied when the rows are read, so asking for five instead of one is a
    different slice of the same list rather than a second run.
    """
    if outcome.order is None:
        return [
            (reading.event, reading.value)
            for reading in sorted(
                readings, key=lambda r: r.event.event_time_utc, reverse=True
            )
        ]
    # Ties broken by recency so the order is total: two releases with the same
    # range must not swap places between runs.
    return [
        (reading.event, reading.value)
        for reading in sorted(
            readings,
            key=lambda r: (r.value, r.event.event_time_utc),
            reverse=outcome.order is Order.DESCENDING,
        )
    ]


def thesis_calendar(thesis: Thesis) -> tuple[list[MarketEvent], str]:
    """The events this thesis measures, for a caller outside the engine.

    Public because the chart needs exactly what the study measured: the same
    dates, compiled the same way, against the same anchor. Anything that built
    its own calendar from the resolution would be a second definition of which
    instant an event happened at, and the one drawing the picture would be the
    one nobody tested.
    """
    return _calendar(thesis)


def _calendar(thesis: Thesis) -> tuple[list[MarketEvent], str]:
    """The events this thesis measures, and the identity of that set.

    One crossing point for both kinds of question. A curated family loads its
    calendar file; a resolved concept is compiled here, which is where every
    date is validated and where a day-only occurrence meets the anchor the
    reader chose. Either way what comes back is a plain list of
    ``MarketEvent``, because the engine below this knows nothing about where a
    calendar came from -- and that is the whole claim the dynamic event work
    was making.
    """
    if thesis.resolution is not None:
        compiled = compile_event_set(thesis.resolution, anchor=thesis.anchor)
        return compiled.calendar, compiled.identity
    assert thesis.event_type is not None  # guaranteed by the thesis validator
    return load_events(thesis.event_type), curated_set_key(thesis.event_type)


def _context(
    study: StudyResult,
    service: MarketDataService,
    thesis: Thesis,
    cutoff: dt.datetime,
) -> ResolutionContext:
    """Gather everything the conditions need, once.

    The regime pass reads two years of hourly history and is skipped entirely
    when no condition asks for it -- a question purely about CPI surprise
    should not pay for it.
    """
    surprises: dict[str, str] = {}
    values = load_releases()
    # `.get`, not `[]`. A dynamically resolved set has no classifier and never
    # will -- nobody publishes a consensus for an election -- so the absence is
    # ordinary rather than an error.
    rule = (
        None
        if thesis.event_type is None
        else SCENARIO_RULES.get(thesis.event_type)
    )
    if rule is not None:
        for event in study.events:
            verdict = rule(values.get(event.event_id))
            if verdict is not None:
                surprises[event.event_id] = verdict[0]

    regimes: dict[str, dict[str, str]] = {}
    readings: dict[str, dict[str, float]] = {}
    if thesis.needs_regime and study.events:
        oldest = min(event.event_time_utc for event in study.events)
        start, _ = history_window(oldest)
        frame = service.get_candles(SETUP_SYMBOL, SETUP_INTERVAL, start, cutoff)
        for event in study.events:
            setup = compute_setup(frame, event.event_time_utc)
            # Buckets and figures come out of the same computation, so both are
            # kept: a numeric condition tests the figure, a categorical one the
            # bucket, and neither triggers a second pass over the history.
            if setup.measured:
                regimes[event.event_id] = setup.measured
            if setup.readings:
                readings[event.event_id] = setup.readings

    return ResolutionContext(
        surprises=surprises, regimes=regimes, readings=readings, as_of=cutoff
    )


def _select(
    study: StudyResult,
    conditions: tuple[Condition, ...],
    thesis: Thesis,
    context: ResolutionContext,
) -> tuple[list[EventResult], dict[str, int]]:
    """Split releases into one group and a tally of why the rest are out.

    Takes the conditions rather than reading them off the thesis, so the same
    function selects the left group, the right group and a baseline. An empty
    tuple selects every release with usable data, which is what a baseline is.

    Each exclusion reason is kept distinct. "No consensus was published" and
    "the surprise was hotter, not cooler" are different facts, and one count
    covering both would hide that the sample is limited by data as well as by
    the question.
    """
    members: list[EventResult] = []
    excluded: dict[str, int] = {}

    def note(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for event in study.events:
        metrics = event.windows.get(thesis.outcome.window)
        if metrics is None or not metrics.is_usable:
            note("no usable price data at this horizon")
            continue

        failed = False
        for condition in conditions:
            spec = CONDITION_FIELDS[condition.field]
            if condition.is_numeric:
                reading = context.resolve_number(condition.field, event.event_id)
                measurable = reading is not None
                held = condition.matches_number(reading)
            else:
                observed = context.resolve(condition.field, event.event_id)
                measurable = observed is not None
                held = condition.matches(observed)

            if not measurable:
                note(f"{spec.label} could not be determined")
                failed = True
                break
            if not held:
                note(f"did not match: {condition.describe()}")
                failed = True
                break
        if not failed:
            members.append(event)

    return members, excluded


def spec_label(field_key: str) -> str:
    return CONDITION_FIELDS[field_key].label


def _observations(
    ordered: list[tuple[EventResult, float | None]],
    thesis: Thesis,
    context: ResolutionContext,
) -> tuple[Observation, ...]:
    """Every qualifying release, in query order, with what qualified it.

    Takes the ordered pairs rather than re-sorting, so the sequence a reader
    sees and the sequence the limit cuts are the same one.
    """
    horizons = {window: horizon for horizon, window in _horizon_windows().items()}
    return tuple(
        Observation(
            event_id=event.event_id,
            event_date=event.local_date,
            conditions=_observed_values(thesis, context, event.event_id),
            returns={
                horizons[name]: (
                    metrics.return_pct if metrics.is_usable else None
                )
                for name, metrics in event.windows.items()
                if name in horizons
            },
            value=value,
        )
        for event, value in ordered
    )


def _observed_values(
    thesis: Thesis, context: ResolutionContext, event_id: str
) -> dict[str, str]:
    """What each condition field actually read for one release.

    The *reading*, not the rule. The reader already knows the rule -- it is on
    the specification card -- and what they cannot otherwise check is whether
    this particular release really satisfied it. ``-4.6%`` answers that;
    repeating "below -3%" does not.

    Keyed by field label, so two bounds on one field (a range) produce one
    entry showing the single reading both were tested against.
    """
    observed: dict[str, str] = {}
    for condition in thesis.conditions:
        spec = CONDITION_FIELDS[condition.field]
        if condition.is_numeric:
            reading = context.resolve_number(condition.field, event_id)
            observed[spec.label] = (
                spec.format_number(reading) if reading is not None else "not measured"
            )
        else:
            observed[spec.label] = spec.display(
                context.resolve(condition.field, event_id) or ""
            )
    return observed


def _horizon_windows() -> dict[str, str]:
    from backtool.research.vocabulary import OUTCOME_HORIZONS

    return dict(OUTCOME_HORIZONS)


def _group_description(
    conditions: tuple[Condition, ...], event_phrase: str
) -> str:
    """One sentence saying what a group holds. Used for both sides."""
    if not conditions:
        return f"All {event_phrase} with usable price data."
    return (
        f"{event_phrase} where "
        + " and ".join(condition.describe() for condition in conditions)
    )


def _warnings(thesis: Thesis, measured: int, compared: int) -> tuple[str, ...]:
    """What a reader should know before trusting the result.

    Shape-aware, because the same sample size means different things to
    different results. A median over eight releases is an estimate that eight
    different outcomes would move; the largest of thirty-one observed releases
    is not an estimate at all, it is a fact about the sample, and warning that
    it is "limited evidence" would be telling the reader to doubt an
    observation rather than an inference.
    """
    notes: list[str] = []
    noun = thesis.event_noun

    def counted(n: int) -> str:
        return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    if measured == 0:
        notes.append(f"No {noun} matched with a usable reading.")
        return tuple(notes)

    if thesis.shape is ResultShape.RANKED:
        # Stated as context rather than as a caveat: a reader ranking releases
        # needs to know how many were in the running.
        notes.append(
            f"Ranked across {counted(measured)} with a usable reading. These "
            "are the observed extremes of that sample, not an estimate of how "
            f"extreme a future {noun} could be."
        )
    else:
        quality = sample_quality(measured)
        if quality.needs_warning:
            notes.append(
                f"{quality.value.capitalize()} sample: {counted(measured)}. "
                "No statistical test has been run, and a handful of different "
                "outcomes would move these figures."
            )

    if thesis.versus is not None:
        if thesis.versus.is_baseline:
            notes.append(
                f"The comparison group is every usable {noun}, including the "
                f"ones selected here. Comparing against the remaining {noun}s "
                "instead would answer a different question and would widen "
                "the gap."
            )
        else:
            # Two named groups are disjoint whenever their conditions
            # contradict, which is the usual case, but nothing guarantees it --
            # so the overlap is reported rather than assumed away.
            notes.append(
                f"The two groups are selected independently; a {noun} meeting "
                "both sets of conditions would appear in both."
            )
        quality = sample_quality(compared)
        if compared == 0:
            notes.append(f"No {noun} matched the comparison group.")
        elif quality.needs_warning:
            notes.append(
                f"Comparison group: {counted(compared)}, a "
                f"{quality.value} sample on that side."
            )

    return tuple(notes)


__all__ = [
    "THESIS_INTERVAL",
    "Observation",
    "ResearchFinding",
    "run_thesis",
    "thesis_calendar",
]
