"""Finding historical releases that happened under today's market conditions.

Matching is exact-on-buckets, not scored. Two releases either shared a bucket on
a dimension or they did not, and a release is a match when it agrees on every
dimension still active. There is no distance function and no weighting, because
a weighted score cannot be checked: a reader shown "83% similar" has no way to
disagree with it, and no way to notice when it is wrong.

The cost of exactness is that four dimensions over ~30 releases frequently
produces two matches, or none. That is handled by **relaxing**, not by fuzzing:
dimensions are dropped one at a time, in a fixed order, and every drop is
reported. A sample that grew from 2 to 6 because the 200-day-average condition
was ignored is a different claim from a sample of 6 that matched on everything,
and the reader is told which one they are looking at.

Relaxation never happens silently. :class:`SetupMatch` carries what was dropped
and what the exact count was, and the page renders both.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from backtool.research.cohort import Cohort, build_cohort
from backtool.research.regime import DIMENSION_LABELS, RELAXATION_ORDER, MarketSetup
from backtool.research.results import EventResult, StudyResult

#: Fewest matched releases worth reporting without relaxing further. Below this
#: the ladder takes another step, if it has one left.
TARGET_SAMPLE = 5


class MatchedRelease(BaseModel):
    """One historical release that matched, and the readings that made it match."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    #: The release's own setup, so the page can show why it matched rather than
    #: asserting that it did.
    setup: MarketSetup


class LadderStep(BaseModel):
    """One rung of the relaxation ladder: what was dropped, and what it bought.

    Recorded for every rung, including the ones that were not used. Four
    conditions over thirty releases is sparse enough that the exact match is
    often empty, and a reader deciding whether to trust the final sample needs
    to see how much had to be given up to reach it -- not just that something
    was.
    """

    model_config = ConfigDict(frozen=True)

    #: Dimensions still required at this rung.
    active: tuple[str, ...]
    #: Dimensions dropped to get here, cumulative.
    dropped: tuple[str, ...]
    sample_size: int
    #: Whether this is the rung the reported statistics came from.
    used: bool = False

    def describe(self) -> str:
        """The rung as a sentence, in the reader's vocabulary."""
        if not self.dropped:
            return "All conditions"
        names = ", ".join(DIMENSION_LABELS.get(d, d) for d in self.dropped)
        return f"Ignoring {names}"


class SetupMatch(BaseModel):
    """The outcome of matching, including everything given up to get there."""

    model_config = ConfigDict(frozen=True)

    #: Today's conditions, the thing being matched against.
    current: MarketSetup
    #: Dimensions every matched release agreed on.
    matched_on: tuple[str, ...]
    #: Dimensions dropped to reach this sample, in the order they were dropped.
    relaxed: tuple[str, ...]
    #: How many releases matched on *every* measurable dimension, before any
    #: relaxation. Reported even when it is zero -- especially when it is zero.
    exact_count: int
    #: How many releases were considered at all.
    candidate_count: int
    matches: tuple[MatchedRelease, ...]
    #: Statistics over the matched releases, or ``None`` if nothing matched.
    cohort: Cohort | None
    #: Every rung considered, tightest first, with the used one flagged.
    ladder: tuple[LadderStep, ...] = ()

    @property
    def sample_size(self) -> int:
        return len(self.matches)

    @property
    def was_relaxed(self) -> bool:
        return bool(self.relaxed)

    def relaxation_note(self) -> str | None:
        """What was given up, and what still holds. ``None`` if nothing was.

        Both halves matter. "Ignoring the 200-day average" tells a reader the
        sample is looser than it looks; "7-day trend and volatility still
        match" tells them it is not meaningless. A note carrying only the first
        reads as a disclaimer, and one carrying only the second hides the cost.
        """
        if not self.relaxed:
            return None
        dropped = _join(DIMENSION_LABELS.get(d, d) for d in self.relaxed)
        kept = _join(DIMENSION_LABELS.get(d, d) for d in self.matched_on)
        # "Still matching:" rather than starting a sentence with the list. Some
        # labels begin with an acronym and some with a lowercase word, so any
        # sentence-initial position needs capitalising -- and str.capitalize
        # lowercases the rest, turning "BTC 7-day trend" into "Btc 7-day trend".
        still = f" Still matching: {kept}." if kept else ""
        return (
            f"Exact matches on all conditions: {self.exact_count}. "
            f"Expanded to {self.sample_size} releases by ignoring {dropped}.{still}"
        )


def _join(names: Iterable[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` -- a list a person would read aloud."""
    items = list(names)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def match_setups(
    study: StudyResult,
    current: MarketSetup,
    history: dict[str, MarketSetup],
    *,
    target_sample: int = TARGET_SAMPLE,
    restrict_to: Iterable[str] | None = None,
    label: str = "Similar setups",
    description: str = "",
) -> SetupMatch:
    """Find releases whose conditions matched today's, relaxing if too few.

    Args:
        study: The completed run supplying candidate releases and their metrics.
        current: Today's conditions.
        history: Each candidate release's own conditions, keyed by event id.
            A release absent here was never measurable and is not a candidate.
        target_sample: Relax until at least this many releases match, or until
            the ladder runs out.
        restrict_to: Optional event ids to consider, which is how a scenario is
            composed with a setup -- pass the scenario's members and the match
            runs inside it.
        label: Display name for the resulting cohort.
        description: One sentence on what the cohort represents.

    Returns:
        A :class:`SetupMatch`. Always returns: no match is a result, not an
        error, and the caller shows "nothing like today" rather than nothing.
    """
    allowed = set(restrict_to) if restrict_to is not None else None
    candidates = [
        event
        for event in study.events
        if event.event_id in history
        and (allowed is None or event.event_id in allowed)
    ]

    active: tuple[str, ...] = tuple(current.measured)
    exact = _matching(candidates, current, history, active)
    matched = exact
    relaxed: tuple[str, ...] = ()

    rungs = [LadderStep(active=active, dropped=(), sample_size=len(exact))]
    settled = len(exact) >= target_sample

    # Drop conditions one at a time, slowest-moving first. The whole ladder is
    # walked even after a rung is settled on, because the rungs below it are
    # what let a reader judge whether the reported sample was bought cheaply or
    # by giving up most of the definition. Never below one active dimension:
    # "matched on nothing" is every release, which is not a setup.
    trial_active, trial_dropped = active, relaxed
    for dimension in RELAXATION_ORDER:
        if dimension not in trial_active or len(trial_active) <= 1:
            continue
        trial_active = tuple(d for d in trial_active if d != dimension)
        trial_dropped = (*trial_dropped, dimension)
        found = _matching(candidates, current, history, trial_active)
        rungs.append(
            LadderStep(
                active=trial_active, dropped=trial_dropped, sample_size=len(found)
            )
        )
        if not settled:
            active, relaxed, matched = trial_active, trial_dropped, found
            settled = len(found) >= target_sample

    ladder = tuple(
        rung.model_copy(update={"used": rung.dropped == relaxed}) for rung in rungs
    )

    cohort = (
        build_cohort(
            study,
            key="similar_setup",
            label=label,
            description=description,
            members=matched,
        )
        if matched
        else None
    )

    return SetupMatch(
        current=current,
        matched_on=active,
        relaxed=relaxed,
        exact_count=len(exact),
        candidate_count=len(candidates),
        matches=tuple(
            MatchedRelease(event_id=event.event_id, setup=history[event.event_id])
            for event in sorted(matched, key=lambda e: e.event_time_utc, reverse=True)
        ),
        cohort=cohort,
        ladder=ladder,
    )


def _matching(
    candidates: list[EventResult],
    current: MarketSetup,
    history: dict[str, MarketSetup],
    active: tuple[str, ...],
) -> list[EventResult]:
    """Releases agreeing with ``current`` on every active dimension.

    A release missing one of the active dimensions does not match. Treating an
    unmeasurable 200-day average as "whatever today is" would pad the sample
    with releases selected for having less data, not for being alike.
    """
    if not active:
        return []

    wanted = current.measured
    result = []
    for event in candidates:
        buckets = history[event.event_id].measured
        if all(buckets.get(dim) == wanted.get(dim) for dim in active):
            result.append(event)
    return result


__all__ = [
    "LadderStep",
    "TARGET_SAMPLE",
    "MatchedRelease",
    "SetupMatch",
    "match_setups",
]
