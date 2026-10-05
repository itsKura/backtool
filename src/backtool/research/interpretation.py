"""Turning a finding into facts, and facts into a sentence a reader understands.

Two layers, deliberately separated, because they fail differently.

:class:`ResearchFacts` is *derived*, not written. Every field on it is a
comparison or a count read straight off a
:class:`~backtool.research.experiment.ResearchFinding` -- did the cohort do
better or worse, by how much, over how many releases, with what missing. It
computes no statistic of its own; the arithmetic it does is subtraction that
``ResearchFinding`` already exposes. Because it is derived, it cannot disagree
with the numbers on screen.

:func:`deterministic_summary` writes those facts out in plain English with no
model involved. It is not a placeholder for the AI reading -- it is the floor
the report stands on. The AI explanation is an improvement on the prose, never
a source of a fact, and when it is missing, slow, unaffordable or wrong the
report reads exactly as well without it.

Why the facts are computed here rather than asked for
-----------------------------------------------------
"Did this cohort outperform?" is a comparison between two numbers. Handing both
numbers to a language model and asking which is bigger introduces a failure
mode with no upside: a model that says "outperformed" about a negative
difference produces a report that contradicts the table printed beneath it. So
the direction, the magnitude, the sample band and the warnings are all settled
before any prose exists, and the prose layer is given the verdict rather than
the raw material to form one.

:func:`interpretation_payload` is the exact set of facts the model may see. It
is built from this module's facts and nothing else: no candles, no per-release
price path, no HTML, no source. See
:mod:`backtool.ai.research_interpreter` for the consumer, which cannot import
this package at all.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from backtool.research.cohort import sample_quality
from backtool.research.experiment import ResearchFinding
from backtool.research.vocabulary import CONDITION_FIELDS, asset_label

#: Below this magnitude in percentage points, a difference between two medians
#: is not worth describing as a direction. Two samples of this size will differ
#: by a tenth of a point from noise alone, and calling that "better" would be
#: the first dishonest sentence in the report.
MATERIAL_DIFFERENCE_PP = 0.1

#: How many qualifying releases to name in the payload. Enough for the reader
#: to recognise a period; few enough that the payload stays small and the model
#: cannot be tempted to analyse the list instead of the statistics.
NOTABLE_OBSERVATIONS = 5


class Direction(StrEnum):
    """How the cohort compared with the baseline.

    Decided here, from the numbers, once. Every downstream sentence -- template
    or model-written -- is checked against this rather than forming its own
    view, which is what makes "underperformed" impossible to print above a
    positive difference.
    """

    BETTER = "better"
    WORSE = "worse"
    SIMILAR = "similar"
    #: The cohort was empty, or the baseline could not be measured.
    UNDETERMINED = "undetermined"


class GroupFacts(BaseModel):
    """One side of the comparison, as counted."""

    model_config = ConfigDict(frozen=True)

    label: str
    sample_size: int
    median_return_pct: float
    positive_count: int
    positive_pct: float


class ResearchFacts(BaseModel):
    """Everything true about a finding that a sentence might want to state.

    Derived from a :class:`ResearchFinding` and never from anywhere else, so
    the prose and the tables cannot drift. Nothing on here is an opinion: the
    one judgement call, what counts as a material difference, is a named
    constant above.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    asset: str
    event_type: str
    horizon: str
    cohort_rule: str

    cohort: GroupFacts | None
    baseline: GroupFacts | None
    difference_pp: float | None

    direction: Direction
    #: ``"very limited"`` through ``"adequate"``, from the shared bands, so the
    #: phrase means the same thing here as on the cohort badge.
    sample_quality: str
    #: Whether the sample is small enough that the reader should be told.
    needs_sample_warning: bool

    #: Releases considered but not in the cohort, counted by reason.
    excluded: dict[str, int] = {}
    #: Engine warnings, verbatim, including the baseline-overlap disclosure.
    warnings: tuple[str, ...] = ()
    #: How the figures were produced, in a handful of lines. Carried on the
    #: facts rather than fetched separately so that the facts are the complete
    #: input to the payload, and the payload builder never needs the finding.
    methodology: dict[str, str] = {}

    @property
    def difference_is_material(self) -> bool:
        return (
            self.difference_pp is not None
            and abs(self.difference_pp) >= MATERIAL_DIFFERENCE_PP
        )


def build_facts(finding: ResearchFinding) -> ResearchFacts:
    """Read a finding into the facts a sentence may rest on.

    No statistic is computed here that ``ResearchFinding`` does not already
    expose. The direction is a sign test on its ``difference_pp``; the bands
    come from :func:`~backtool.research.cohort.sample_quality`, the same
    function the cohort badge uses.
    """
    thesis = finding.thesis
    cohort_metrics = finding.cohort_metrics
    baseline_metrics = finding.baseline_metrics
    difference = finding.difference_pp

    cohort = (
        GroupFacts(
            label=finding.cohort.label,
            sample_size=cohort_metrics.sample_size,
            median_return_pct=cohort_metrics.median_return_pct,
            positive_count=cohort_metrics.up_count,
            positive_pct=cohort_metrics.up_rate_pct,
        )
        if cohort_metrics is not None and cohort_metrics.sample_size > 0
        else None
    )
    baseline = (
        GroupFacts(
            label=finding.baseline.label,
            sample_size=baseline_metrics.sample_size,
            median_return_pct=baseline_metrics.median_return_pct,
            positive_count=baseline_metrics.up_count,
            positive_pct=baseline_metrics.up_rate_pct,
        )
        if baseline_metrics is not None and baseline_metrics.sample_size > 0
        else None
    )

    if cohort is None or difference is None:
        direction = Direction.UNDETERMINED
    elif abs(difference) < MATERIAL_DIFFERENCE_PP:
        direction = Direction.SIMILAR
    elif difference > 0:
        direction = Direction.BETTER
    else:
        direction = Direction.WORSE

    size = cohort.sample_size if cohort else 0
    quality = sample_quality(size)

    return ResearchFacts(
        question=thesis.headline(),
        asset=f"{asset_label(thesis.asset)} ({thesis.asset})",
        event_type=thesis.event_type.value,
        horizon=thesis.outcome.horizon,
        cohort_rule=(
            " and ".join(condition.describe() for condition in thesis.conditions)
            or "no conditions -- every usable release"
        ),
        cohort=cohort,
        baseline=baseline,
        difference_pp=difference,
        direction=direction,
        sample_quality=quality.value,
        needs_sample_warning=bool(size) and quality.needs_warning,
        excluded=dict(finding.excluded),
        warnings=finding.warnings,
        methodology=methodology_summary(finding),
    )


def deterministic_summary(facts: ResearchFacts) -> tuple[str, ...]:
    """The reading of a finding, in sentences, with no model involved.

    This is what the reader gets when there is no API key, no budget, no
    network, or no confidence in what came back. It is held to the same
    standard as the AI version and for the same reason: it states what was
    measured and declines to explain *why* it happened, because nothing here
    tested why. There is no "investors de-risked" sentence available to a
    template, which is one of the quieter arguments for having one.

    Returns:
        One paragraph per element, in reading order. Empty-cohort findings
        return a single sentence saying so.
    """
    cohort, baseline = facts.cohort, facts.baseline

    if cohort is None:
        return (
            f"No {facts.event_type} release in the sample matched "
            f"{facts.cohort_rule}, so there is nothing to compare. "
            + _exclusion_sentence(facts),
        )

    paragraphs: list[str] = []

    hit_rate = (
        f"{cohort.positive_count} of {cohort.sample_size} qualifying "
        f"{_releases(cohort.sample_size)} had a positive "
        f"{facts.horizon} return"
    )
    paragraphs.append(
        f"{cohort.sample_size} {_releases(cohort.sample_size)} matched "
        f"{facts.cohort_rule}. Their median {facts.horizon} return was "
        f"{_signed(cohort.median_return_pct)}, and {hit_rate}."
    )

    if baseline is not None and facts.difference_pp is not None:
        comparison = _comparison_sentence(facts, baseline)
        paragraphs.append(comparison)

    closing = _caveat_sentence(facts)
    if closing:
        paragraphs.append(closing)

    return tuple(paragraphs)


def _comparison_sentence(facts: ResearchFacts, baseline: GroupFacts) -> str:
    """The cohort set against the baseline, in the direction already decided."""
    assert facts.difference_pp is not None
    difference = facts.difference_pp
    against = (
        f"The baseline of {baseline.sample_size} usable "
        f"{_releases(baseline.sample_size)} had a median of "
        f"{_signed(baseline.median_return_pct)}"
    )

    if facts.direction is Direction.SIMILAR:
        return (
            f"{against}, within {MATERIAL_DIFFERENCE_PP:g} percentage points of "
            "the cohort. In this historical sample the condition made no "
            "material difference to the median outcome."
        )

    word = "outperformed" if facts.direction is Direction.BETTER else "underperformed"
    return (
        f"{against}, so the cohort {word} it by "
        f"{abs(difference):.2f} percentage points of median return. That is "
        "the observed difference in this historical sample, not a measure of "
        "how reliably it would repeat."
    )


def _caveat_sentence(facts: ResearchFacts) -> str:
    """What the reader should hold back, stated as plainly as the result."""
    cohort = facts.cohort
    if cohort is None:
        return ""

    parts: list[str] = []
    if facts.needs_sample_warning:
        parts.append(
            f"The sample contains only {cohort.sample_size} "
            f"{_releases(cohort.sample_size)}, which is {facts.sample_quality} "
            "historical evidence rather than a stable relationship; no "
            "statistical test has been run"
        )
    if facts.excluded:
        missing = sum(
            count
            for reason, count in facts.excluded.items()
            if "could not be determined" in reason or "no usable price data" in reason
        )
        if missing:
            parts.append(
                f"{missing} further {_releases(missing)} could not be "
                "classified or priced and are excluded rather than assumed"
            )
    if not parts:
        return ""
    return _join(parts) + "."


def _exclusion_sentence(facts: ResearchFacts) -> str:
    if not facts.excluded:
        return "No releases were considered."
    total = sum(facts.excluded.values())
    return (
        f"{total} {_releases(total)} were considered and each fell out for a "
        "stated reason, listed under the methodology."
    )


def interpretation_payload(facts: ResearchFacts) -> dict[str, Any]:
    """Exactly what the language model is allowed to see.

    Statistics and verdicts, already settled. No candles, no per-release price
    path, no HTML, no application source, no conversation history. The model's
    job is to say these facts well, and anything beyond them would only give it
    room to form a second opinion about numbers that are already final.

    The ``direction`` and ``difference_is_material`` fields are the important
    ones: they are the verdict, not evidence for one, and the explanation is
    checked against them afterwards.
    """
    payload: dict[str, Any] = {
        "question": facts.question,
        "asset": facts.asset,
        "event_type": facts.event_type,
        "horizon": facts.horizon,
        "cohort_rule": facts.cohort_rule,
        "direction_vs_baseline": facts.direction.value,
        "difference_pp": _round(facts.difference_pp),
        "difference_is_material": facts.difference_is_material,
        "sample_quality": facts.sample_quality,
        "warnings": list(facts.warnings),
        "excluded_counts": dict(facts.excluded),
        "methodology": dict(facts.methodology),
    }
    payload["cohort"] = _group_payload(facts.cohort)
    payload["baseline"] = _group_payload(facts.baseline)
    return payload


def _group_payload(group: GroupFacts | None) -> dict[str, Any] | None:
    if group is None:
        return None
    return {
        "label": group.label,
        "n": group.sample_size,
        "median_return_pct": _round(group.median_return_pct),
        "positive_count": group.positive_count,
        "positive_pct": _round(group.positive_pct, 1),
    }


def methodology_summary(finding: ResearchFinding) -> dict[str, str]:
    """A short, fixed description of how the figures were produced.

    Sent with the payload so the explanation can say "measured from the
    release price at 5-minute resolution" without inventing a method. Kept to
    a handful of lines; the full methodology block already lives in the report
    and the model does not need it to write three paragraphs.
    """
    thesis = finding.thesis
    conditions = [CONDITION_FIELDS[c.field].source for c in thesis.conditions]
    return {
        "outcome": (
            f"{thesis.outcome.horizon} {thesis.outcome.metric.value} from the "
            "release price"
        ),
        "anchoring": (
            "the last candle closing at or before each boundary, so no "
            "measurement contains information from after the instant it "
            "describes"
        ),
        "candles": f"{finding.spec.interval.value} candles from Binance",
        "statistic": "median across releases, not mean",
        "conditions_read_from": "; ".join(conditions) or "no conditions applied",
    }


def _round(value: float | None, places: int = 2) -> float | None:
    return None if value is None else round(value, places)


def _signed(value: float) -> str:
    """A percentage with its sign taken from the rounded figure.

    ``-0.001`` prints as ``0.00%``, not ``-0.00%``: a move that rounds to
    nothing should not be given a direction by its last decimal place.
    """
    rounded = round(value, 2)
    return f"{rounded:+.2f}%" if rounded else "0.00%"


def _releases(count: int) -> str:
    return "release" if count == 1 else "releases"


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0][0].upper() + parts[0][1:]
    head = parts[0][0].upper() + parts[0][1:]
    return head + ", and " + ", and ".join(parts[1:])


__all__ = [
    "MATERIAL_DIFFERENCE_PP",
    "Direction",
    "GroupFacts",
    "ResearchFacts",
    "build_facts",
    "deterministic_summary",
    "interpretation_payload",
    "methodology_summary",
]
