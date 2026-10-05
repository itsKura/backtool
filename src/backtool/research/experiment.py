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
from backtool.events.releases import SCENARIO_RULES, load_releases
from backtool.research.aggregate import WindowAggregate
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
from backtool.research.thesis import Thesis
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    ResolutionContext,
    reaction_windows,
)

logger = logging.getLogger(__name__)

#: Candle interval every thesis is measured at. Fine enough to resolve a
#: five-minute window, which is the shortest horizon the vocabulary offers.
THESIS_INTERVAL = Interval.M5


class Observation(BaseModel):
    """One release that qualified, with what put it there and what followed."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_date: dt.date
    #: The value of every condition field for this release, so the reader can
    #: check membership rather than take it on trust.
    conditions: dict[str, str]
    #: Return at each measured horizon, keyed by horizon.
    returns: dict[str, float | None]


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
    baseline: Cohort
    observations: tuple[Observation, ...]

    #: Releases considered but not in the cohort, counted by reason.
    excluded: dict[str, int] = {}
    #: Anything the reader should know before trusting the figures.
    warnings: tuple[str, ...] = ()

    @property
    def window(self) -> str:
        return self.thesis.outcome.window

    @property
    def cohort_metrics(self) -> WindowAggregate | None:
        return self.cohort.windows.get(self.window)

    @property
    def baseline_metrics(self) -> WindowAggregate | None:
        return self.baseline.windows.get(self.window)

    @property
    def difference_pp(self) -> float | None:
        """Cohort median minus baseline median, in percentage points.

        Medians rather than means on both sides. One violent release drags a
        mean somewhere no individual release went, and a difference between two
        means can be produced entirely by a single outlier on either side.
        """
        cohort, baseline = self.cohort_metrics, self.baseline_metrics
        if cohort is None or baseline is None:
            return None
        if cohort.sample_size == 0 or baseline.sample_size == 0:
            return None
        return cohort.median_return_pct - baseline.median_return_pct

    def verdict(self) -> str:
        """A neutral one-line reading of the comparison.

        Deterministic, not written by a model: this sentence is the first thing
        a reader sees, and it has to be derivable from the two numbers above it.
        The language layer may elaborate on it but never replaces it.
        """
        difference = self.difference_pp
        cohort = self.cohort_metrics
        if cohort is None or cohort.sample_size == 0:
            return "No releases matched these conditions, so there is nothing to compare."
        if difference is None:
            return "The baseline could not be measured, so there is nothing to compare against."

        horizon = self.thesis.outcome.horizon
        size = abs(difference)
        if size < 0.1:
            return (
                f"No meaningful difference: the {horizon} median is within "
                f"0.1 percentage points of a typical release."
            )
        direction = "better" if difference > 0 else "worse"
        return (
            f"These releases performed {direction} than a typical one over "
            f"{horizon}, by {size:.2f} percentage points of median return."
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

    spec = ResearchSpec(
        symbol=thesis.asset,
        event_type=thesis.event_type,
        event_count=thesis.event_count,
        interval=THESIS_INTERVAL,
        windows=reaction_windows(),
        as_of=cutoff,
    )

    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(thesis.event_type), now=now)
        context = _context(study, service, thesis, cutoff)

    members, excluded = _select(study, thesis, context)
    usable = [
        event
        for event in study.events
        if event.windows.get(thesis.outcome.window, None) is not None
        and event.windows[thesis.outcome.window].is_usable
    ]

    cohort = build_cohort(
        study,
        key="cohort",
        label=_cohort_label(thesis),
        description=_cohort_description(thesis),
        members=members,
    )
    baseline = build_cohort(
        study,
        key="baseline",
        label=thesis.baseline.describe(thesis.event_type),
        description=(
            f"Every {thesis.event_type.value} release with usable price data, "
            "whatever the conditions around it."
        ),
        members=usable,
    )

    return ResearchFinding(
        thesis=thesis,
        spec=spec,
        generated_at=now,
        cohort=cohort,
        baseline=baseline,
        observations=_observations(members, thesis, context),
        excluded=excluded,
        warnings=_warnings(cohort, baseline, thesis),
    )


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
    rule = SCENARIO_RULES[thesis.event_type]
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
    study: StudyResult, thesis: Thesis, context: ResolutionContext
) -> tuple[list[EventResult], dict[str, int]]:
    """Split releases into the cohort and a tally of why the rest are out.

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
        for condition in thesis.conditions:
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
    members: list[EventResult], thesis: Thesis, context: ResolutionContext
) -> tuple[Observation, ...]:
    """Every qualifying release, with the values that qualified it."""
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
        )
        for event in sorted(members, key=lambda e: e.event_time_utc, reverse=True)
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


def _cohort_label(thesis: Thesis) -> str:
    if not thesis.conditions:
        return f"All {thesis.event_type.value} releases"
    return " and ".join(condition.describe() for condition in thesis.conditions)


def _cohort_description(thesis: Thesis) -> str:
    return (
        f"{thesis.event_type.value} releases where "
        + " and ".join(condition.describe() for condition in thesis.conditions)
        if thesis.conditions
        else f"Every {thesis.event_type.value} release with usable price data."
    )


def _warnings(cohort: Cohort, baseline: Cohort, thesis: Thesis) -> tuple[str, ...]:
    """What a reader should know before trusting the comparison."""
    notes: list[str] = []

    if cohort.sample_size == 0:
        notes.append("No release matched these conditions.")
    else:
        quality = sample_quality(cohort.sample_size)
        if quality.needs_warning:
            notes.append(
                f"{quality.value.capitalize()} sample: {cohort.sample_size} "
                "releases. No statistical test has been run, and a handful of "
                "different outcomes would move these figures."
            )

    if 0 < cohort.sample_size == baseline.sample_size:
        notes.append(
            "Every usable release met these conditions, so the cohort and the "
            "baseline are the same set and the difference is zero by "
            "construction."
        )
    elif cohort.sample_size:
        notes.append(
            "The baseline includes the cohort. Comparing against the remaining "
            "releases instead would answer a different question and would widen "
            "the gap."
        )

    return tuple(notes)


__all__ = [
    "THESIS_INTERVAL",
    "Observation",
    "ResearchFinding",
    "run_thesis",
]
