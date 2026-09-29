"""Splitting a study's events into named groups, and describing each one.

Both of the questions this product exists to answer turn out to be the same
operation:

* *"How did BTC behave when CPI came in cooler than expected?"*
* *"How did BTC behave when the market looked like it does today?"*

Each takes the same set of historical releases, splits it into named groups, and
reports an identical block of statistics per group. Only the rule that assigns a
release to a group differs. Building that once means the two features cannot
drift apart in what they measure, and it is what makes the third question --
*"cooler-than-expected CPI while the market looked like today"* -- a composition
of two rules rather than a third system.

Nothing here computes a statistic. Grouping happens here; the numbers come from
:func:`~backtool.research.aggregate.aggregate_window`, which is the only place
in the project that produces one.

**Sample size is never hidden.** A cohort of three is reported as a cohort of
three, with its statistics, and the caller is expected to show the count beside
them. Suppressing a small group would leave the reader assuming it did not
exist; showing it without the count would let three observations read like
thirty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from backtool.research.aggregate import WindowAggregate, aggregate_window
from backtool.research.results import EventResult, StudyResult

#: Below this many observations a cohort is labelled indicative rather than
#: informative. Deliberately the same threshold the single-window aggregate
#: uses, so "small sample" means one thing across the product.
MIN_COHORT_SAMPLE = 5


class SampleQuality(StrEnum):
    """How much weight a sample of a given size can carry.

    Bands, not a score. A confidence percentage would be a statistic nobody
    computed -- there is no test behind it, and attaching a number to a hunch
    is the precise thing this project exists not to do. A band says only what
    the count already says, in words a reader will not skim past.
    """

    VERY_LIMITED = "very limited"
    LIMITED = "limited"
    MODERATE = "moderate"
    ADEQUATE = "adequate"

    @property
    def needs_warning(self) -> bool:
        return self is not SampleQuality.ADEQUATE


#: Lower bound of each band, checked largest-first.
_QUALITY_BANDS: tuple[tuple[int, SampleQuality], ...] = (
    (20, SampleQuality.ADEQUATE),
    (10, SampleQuality.MODERATE),
    (5, SampleQuality.LIMITED),
    (0, SampleQuality.VERY_LIMITED),
)


def sample_quality(count: int) -> SampleQuality:
    """Which band a sample size falls in."""
    for floor, band in _QUALITY_BANDS:
        if count >= floor:
            return band
    return SampleQuality.VERY_LIMITED


class Cohort(BaseModel):
    """One named group of historical releases, and how price behaved in it."""

    model_config = ConfigDict(frozen=True)

    #: Stable identifier, used for selection in the UI.
    key: str
    #: What a reader sees, e.g. "Cooler than expected".
    label: str
    #: Why these releases are grouped together, in one sentence.
    description: str
    #: The releases in this cohort, most recent first.
    event_ids: tuple[str, ...]
    #: One aggregate per window of the spec, keyed by window name.
    windows: dict[str, WindowAggregate]

    @property
    def sample_size(self) -> int:
        """Releases assigned to this cohort.

        This is the count of releases *selected*, which is not necessarily the
        count behind any given window -- a release can land in the cohort and
        still lack the candles to measure its 24h window. Each window carries
        its own ``sample_size`` for that reason, and the two are reported
        separately rather than reconciled into one reassuring number.
        """
        return len(self.event_ids)

    @property
    def is_reliable(self) -> bool:
        return self.sample_size >= MIN_COHORT_SAMPLE

    @property
    def quality(self) -> SampleQuality:
        return sample_quality(self.sample_size)

    @property
    def caveat(self) -> str | None:
        """What the sample size means, in the reader's terms.

        Scaled to the band rather than a single on/off warning: twelve
        releases and three releases are both short of convincing, but they are
        not short in the same way, and one warning for both would either
        overstate the risk at twelve or understate it at three.
        """
        count = self.sample_size
        if count == 0:
            return "no historical releases match this group"
        if count < 5:
            return (
                f"Very limited sample: {count} release"
                f'{"" if count == 1 else "s"}. One different outcome would move '
                "every figure here. Read these as examples, not as a rate."
            )
        if count < 10:
            return (
                f"Limited sample: {count} releases. Indicative only -- a couple "
                "of different outcomes would change the direction of these "
                "figures."
            )
        if count < 20:
            return (
                f"Moderate sample: {count} releases. Enough to describe a "
                "tendency, not enough to establish one."
            )
        return None


def build_cohort(
    study: StudyResult,
    *,
    key: str,
    label: str,
    description: str,
    members: Sequence[EventResult],
) -> Cohort:
    """Describe one group of already-selected releases.

    Args:
        study: The completed run the members came from. Supplies the window
            list, so every cohort reports the same windows in the same order.
        key: Stable identifier for selection.
        label: Display name.
        description: One sentence on what defines membership.
        members: The releases in this cohort.

    Returns:
        A :class:`Cohort` whose per-window statistics were computed over
        exactly these members and nothing else.
    """
    subset = StudyResult(
        spec=study.spec,
        generated_at=study.generated_at,
        events=tuple(members),
    )
    return Cohort(
        key=key,
        label=label,
        description=description,
        # Most recent first: a reader scanning the matched releases wants the
        # ones closest to today's conditions at the top.
        event_ids=tuple(
            event.event_id
            for event in sorted(members, key=lambda e: e.event_time_utc, reverse=True)
        ),
        windows={
            window.name: aggregate_window(subset, window.name)
            for window in study.spec.windows
        },
    )


def split_study(
    study: StudyResult,
    classify: Callable[[EventResult], tuple[str, str, str] | None],
) -> list[Cohort]:
    """Split every release in a study into cohorts by a classification rule.

    Args:
        study: The completed run.
        classify: Given one release, returns ``(key, label, description)`` for
            the cohort it belongs to, or ``None`` to leave it unclassified.
            Returning ``None`` is how a rule declines to guess -- a release
            with no consensus figure is not "in line", it is unknown, and the
            two must not collapse into one bucket.

    Returns:
        Cohorts in first-seen order. Releases the rule declined are simply
        absent; the caller is responsible for reporting how many, which
        :func:`unclassified_count` supplies.
    """
    order: list[str] = []
    grouped: dict[str, tuple[str, str, list[EventResult]]] = {}

    for event in study.events:
        verdict = classify(event)
        if verdict is None:
            continue
        key, label, description = verdict
        if key not in grouped:
            order.append(key)
            grouped[key] = (label, description, [])
        grouped[key][2].append(event)

    return [
        build_cohort(
            study,
            key=key,
            label=grouped[key][0],
            description=grouped[key][1],
            members=grouped[key][2],
        )
        for key in order
    ]


def unclassified_count(
    study: StudyResult,
    classify: Callable[[EventResult], tuple[str, str, str] | None],
) -> int:
    """How many releases the rule declined to classify.

    Reported alongside the cohorts. Twelve cohorted releases out of twelve is a
    different claim from twelve out of thirty, and only one of them is visible
    without this.
    """
    return sum(1 for event in study.events if classify(event) is None)


__all__ = [
    "MIN_COHORT_SAMPLE",
    "Cohort",
    "SampleQuality",
    "sample_quality",
    "build_cohort",
    "split_study",
    "unclassified_count",
]
