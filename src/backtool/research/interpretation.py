"""Turning a finding into facts, and facts into a sentence a reader understands.

Two layers, deliberately separated, because they fail differently.

:class:`ResearchFacts` is *derived*, not written. Every field on it is a
comparison or a count read straight off a
:class:`~backtool.research.experiment.ResearchFinding`. It computes no statistic
of its own, so it cannot disagree with the numbers on screen.

:func:`deterministic_summary` writes those facts out in plain English with no
model involved. It is not a placeholder for the AI reading -- it is the floor
the report stands on. The AI explanation is an improvement on the prose, never
a source of a fact.

Shape, not question type
------------------------
A question can produce three kinds of answer, and they need different
sentences: a statistic set against a baseline, a statistic standing alone, and
a ranked list of individual releases. Those three come from
:class:`~backtool.research.vocabulary.ResultShape`, which is derived from the
query -- so adding "the five smallest 4h drawdowns after FOMC" needs no code
here at all.

What changes with shape is not only the wording. A comparison has a direction;
a ranked list does not, and passing it one would invite a sentence about
outperformance in a result with nothing to outperform. A median over eight
releases is an estimate that deserves a small-sample warning; the largest of
thirty-one observed releases is a fact about the sample, and warning that it is
"limited evidence" would tell the reader to doubt an observation rather than an
inference. Both distinctions are made here, once.

:func:`interpretation_payload` is the exact set of facts the model may see, and
it carries only the fields its shape gives meaning to. See
:mod:`backtool.ai.research_interpreter` for the consumer, which cannot import
this package at all.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from backtool.research.cohort import sample_quality
from backtool.research.experiment import ResearchFinding
from backtool.research.vocabulary import (
    CONDITION_FIELDS,
    Aggregation,
    Order,
    ResultShape,
    asset_label,
)

#: Below this magnitude in percentage points, a difference between two
#: aggregates is not worth describing as a direction. Two samples of this size
#: will differ by a tenth of a point from noise alone, and calling that "better"
#: would be the first dishonest sentence in the report.
MATERIAL_DIFFERENCE_PP = 0.1

#: How many ranked rows to name in the payload. Enough for the reader to see
#: the shape of the tail; few enough that the payload stays small.
MAX_PAYLOAD_ROWS = 5


class Direction(StrEnum):
    """How the cohort compared with the baseline.

    Decided here, from the numbers, once. Every downstream sentence is checked
    against this rather than forming its own view, which is what makes
    "underperformed" impossible to print above a positive difference.
    """

    BETTER = "better"
    WORSE = "worse"
    SIMILAR = "similar"
    #: No comparison exists: the cohort was empty, the baseline could not be
    #: measured, or the question was never a comparison in the first place.
    UNDETERMINED = "undetermined"


class GroupFacts(BaseModel):
    """One side of a comparison, as counted.

    Carries its own sample size and its own quality band. Two groups in one
    comparison are routinely lopsided -- hotter CPI may have ten releases and
    cooler three -- and a single shared warning would describe the better-
    supported side and quietly cover for the other.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    sample_size: int
    #: The aggregated statistic, in the metric's own units.
    value: float
    value_text: str
    #: ``"very limited"`` through ``"adequate"``, for this group alone.
    sample_quality: str = ""
    #: Whether this group's size needs saying out loud.
    needs_sample_warning: bool = False
    #: How often the move was upward. Only meaningful when the metric is a
    #: signed return -- a trading range is never negative, so a "positive rate"
    #: over one would be 100% and would mean nothing.
    positive_count: int | None = None
    positive_pct: float | None = None


class RowFacts(BaseModel):
    """One ranked release."""

    model_config = ConfigDict(frozen=True)

    rank: int
    event_date: str
    value: float
    value_text: str


class ResearchFacts(BaseModel):
    """Everything true about a finding that a sentence might want to state.

    Carries the blocks its shape gives meaning to and leaves the rest empty.
    ``shape`` says which is which, so a consumer never has to infer whether a
    ``None`` means "not applicable" or "could not be measured".
    """

    model_config = ConfigDict(frozen=True)

    shape: ResultShape
    question: str
    asset: str
    event_type: str
    horizon: str
    #: What was measured and how, e.g. "the 5 largest 24h trading range
    #: readings".
    measure: str
    metric_label: str
    cohort_rule: str

    #: Releases whose reading entered the result.
    sample_size: int = 0

    # --- scalar shapes ---
    value: float | None = None
    value_text: str = ""
    #: For a frequency: how many readings cleared the threshold.
    matched_count: int | None = None

    # --- comparison only ---
    left: GroupFacts | None = None
    right: GroupFacts | None = None
    difference_pp: float | None = None
    direction: Direction = Direction.UNDETERMINED
    #: Whether the right-hand group is every usable release rather than a
    #: second named selection. Decides whether "a typical release" is an
    #: honest description of it.
    right_is_baseline: bool = False

    # --- ranked only ---
    rows: tuple[RowFacts, ...] = ()
    #: How many releases the ranking was taken from.
    ranked_from: int = 0
    extreme: str = ""

    sample_quality: str = ""
    #: Whether the sample is small enough that the reader should be told. False
    #: for a ranked result, which reports observations rather than estimates.
    needs_sample_warning: bool = False

    excluded: dict[str, int] = {}
    warnings: tuple[str, ...] = ()
    methodology: dict[str, str] = {}

    @property
    def is_comparison(self) -> bool:
        return self.shape is ResultShape.COMPARISON

    @property
    def higher_side(self) -> str:
        """Which group measured higher: ``left``, ``right`` or ``neither``.

        The neutral form of the verdict. ``direction`` says better or worse,
        which only means something when one side is a baseline; this means the
        same thing for two named groups without implying either is the
        standard the other is judged against.
        """
        if self.difference_pp is None or not self.difference_is_material:
            return "neither"
        return "left" if self.difference_pp > 0 else "right"

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
    shape = finding.shape
    quality = sample_quality(finding.sample_size)

    # Written out rather than splatted from a dict: every field here is part of
    # the contract with the prose layer, and a keyword argument is checked
    # while a `**kwargs` is not.
    facts = ResearchFacts(
        shape=shape,
        question=thesis.headline(),
        asset=f"{asset_label(thesis.asset)} ({thesis.asset})",
        event_type=thesis.event_label,
        horizon=thesis.outcome.horizon,
        measure=thesis.outcome.describe(),
        metric_label=finding.metric.label,
        cohort_rule=(
            " and ".join(condition.describe() for condition in thesis.conditions)
            or "no conditions -- every usable release"
        ),
        sample_size=finding.sample_size,
        sample_quality=quality.value,
        excluded=dict(finding.excluded),
        warnings=finding.warnings,
        methodology=methodology_summary(finding),
    )

    if shape is ResultShape.RANKED:
        return facts.model_copy(
            update={
                "rows": tuple(
                    RowFacts(
                        rank=index,
                        event_date=f"{row.event_date:%d %B %Y}",
                        value=row.value if row.value is not None else 0.0,
                        value_text=finding.metric.format(row.value or 0.0),
                    )
                    for index, row in enumerate(finding.rows, start=1)
                ),
                "ranked_from": finding.sample_size,
                "extreme": (
                    "largest" if thesis.outcome.order is Order.DESCENDING else "smallest"
                ),
                # Left False deliberately. An extremum is an observation, not an
                # estimate, so the caveat that belongs on a median would be
                # telling the reader to doubt something that happened.
                "needs_sample_warning": False,
            }
        )

    facts = facts.model_copy(
        update={
            "value": finding.value,
            "value_text": finding.format_value(finding.value),
            "matched_count": finding.matched_count,
            "needs_sample_warning": bool(finding.sample_size) and quality.needs_warning,
        }
    )

    if shape is ResultShape.SCALAR:
        return facts

    left = _group(finding, finding.value, finding.cohort.label, finding.sample_size)
    right = (
        _group(
            finding,
            finding.comparison_value,
            finding.comparison.label,
            finding.comparison_sample_size,
        )
        if finding.comparison is not None
        else None
    )
    difference = finding.difference_pp

    if left is None or difference is None:
        direction = Direction.UNDETERMINED
    elif abs(difference) < MATERIAL_DIFFERENCE_PP:
        direction = Direction.SIMILAR
    elif difference > 0:
        direction = Direction.BETTER
    else:
        direction = Direction.WORSE

    return facts.model_copy(
        update={
            "left": left,
            "right": right,
            "difference_pp": difference,
            "direction": direction,
            "right_is_baseline": (
                thesis.versus is not None and thesis.versus.is_baseline
            ),
        }
    )


def _group(
    finding: ResearchFinding, value: float | None, label: str, size: int
) -> GroupFacts | None:
    """One side of a comparison, counted on its own terms.

    The quality band is computed from this group's size alone. The hit rate
    appears only for a signed return, because "100% positive" over a trading
    range would be true and meaningless.
    """
    if value is None or size == 0:
        return None
    metrics = (
        finding.cohort_metrics
        if label == finding.cohort.label
        else finding.comparison_metrics
    )
    signed = finding.metric.key == "return_pct"
    quality = sample_quality(size)
    return GroupFacts(
        label=label,
        sample_size=size,
        value=value,
        value_text=finding.format_value(value),
        sample_quality=quality.value,
        needs_sample_warning=quality.needs_warning,
        positive_count=metrics.up_count if signed and metrics else None,
        positive_pct=metrics.up_rate_pct if signed and metrics else None,
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
        One paragraph per element, in reading order.
    """
    if facts.shape is ResultShape.RANKED:
        return _ranked_summary(facts)
    if facts.shape is ResultShape.SCALAR:
        return _scalar_summary(facts)
    return _comparison_summary(facts)


def _ranked_summary(facts: ResearchFacts) -> tuple[str, ...]:
    """Individual releases, named and dated. No direction, no estimate."""
    if not facts.rows:
        return (
            f"No {facts.event_type} release had a usable "
            f"{facts.horizon} {facts.metric_label}, so there is nothing to rank.",
        )

    top = facts.rows[0]
    paragraphs = [
        f"The {facts.extreme} {facts.horizon} {facts.metric_label} after a "
        f"{facts.event_type} release was {top.value_text}, on {top.event_date}."
    ]

    rest = facts.rows[1:]
    if rest:
        listed = ", ".join(f"{row.value_text} on {row.event_date}" for row in rest)
        paragraphs.append(f"The next {facts.extreme} were {listed}.")

    paragraphs.append(
        f"These are the {facts.extreme} readings among {facts.ranked_from} "
        f"{_releases(facts.ranked_from)} with usable data, observed rather "
        "than estimated. "
        + _exclusion_clause(facts)
    )
    return tuple(p.strip() for p in paragraphs)


def _scalar_summary(facts: ResearchFacts) -> tuple[str, ...]:
    """One statistic, standing on its own, with the population behind it."""
    if facts.value is None:
        return (
            f"No {facts.event_type} release produced a usable "
            f"{facts.horizon} {facts.metric_label}, so there is nothing to "
            "report. " + _exclusion_clause(facts),
        )

    paragraphs: list[str] = []
    if facts.matched_count is not None:
        paragraphs.append(
            f"{facts.value_text} of {facts.event_type} releases met this "
            f"condition: {facts.matched_count} of {facts.sample_size} "
            f"{_releases(facts.sample_size)} with usable data."
        )
    else:
        paragraphs.append(
            f"Across {facts.sample_size} {_releases(facts.sample_size)} with "
            f"usable data, the {facts.measure} was {facts.value_text}."
        )

    caveat = _caveat_sentence(facts)
    if caveat:
        paragraphs.append(caveat)
    return tuple(paragraphs)


def _comparison_summary(facts: ResearchFacts) -> tuple[str, ...]:
    """Two groups set against each other.

    Symmetrical when both sides are named, because neither is the other's
    background. The baseline case keeps its old asymmetric phrasing -- "the
    cohort underperformed the baseline" -- because there one side really is
    everything and saying so is clearer than pretending otherwise.
    """
    left, right = facts.left, facts.right

    if left is None:
        return (
            f"No {facts.event_type} release in the sample matched "
            f"{facts.cohort_rule}, so there is nothing to compare. "
            + _exclusion_clause(facts),
        )
    if right is None:
        return (
            f"{left.sample_size} {_releases(left.sample_size)} matched "
            f"{left.label} and their {facts.measure} was {left.value_text}, "
            "but no release matched the group it was to be compared with, so "
            "there is no comparison to report.",
        )

    paragraphs = [_group_sentence(facts, left), _group_sentence(facts, right)]
    if facts.difference_pp is not None:
        paragraphs.append(_difference_sentence(facts, left, right))

    caveat = _caveat_sentence(facts)
    if caveat:
        paragraphs.append(caveat)
    return tuple(paragraphs)


def _group_sentence(facts: ResearchFacts, group: GroupFacts) -> str:
    """One group's own figures, with its own sample size beside them."""
    sentence = (
        f"{group.label}: {group.sample_size} "
        f"{_releases(group.sample_size)}, {facts.measure} "
        f"{group.value_text}"
    )
    if group.positive_count is not None:
        sentence += (
            f", {group.positive_count} of {group.sample_size} positive over "
            f"{facts.horizon}"
        )
    if group.needs_sample_warning:
        sentence += f" ({group.sample_quality} sample)"
    return sentence + "."


def _difference_sentence(
    facts: ResearchFacts, left: GroupFacts, right: GroupFacts
) -> str:
    assert facts.difference_pp is not None
    size = abs(facts.difference_pp)

    if facts.direction is Direction.SIMILAR:
        return (
            f"The two differ by less than {MATERIAL_DIFFERENCE_PP:g} percentage "
            "points. In this historical sample the distinction made no "
            "material difference to the outcome."
        )

    if facts.right_is_baseline:
        word = (
            "outperformed" if facts.direction is Direction.BETTER else "underperformed"
        )
        return (
            f"The cohort {word} the baseline by {size:.2f} percentage points. "
            "That is the observed difference in this historical sample, not a "
            "measure of how reliably it would repeat."
        )

    higher, lower = (left, right) if facts.difference_pp > 0 else (right, left)
    return (
        f"{higher.label} measured {size:.2f} percentage points higher than "
        f"{lower.label}. That is the observed difference between these two "
        "groups in this historical sample, not a measure of how reliably it "
        "would repeat, and no statistical test was run on it."
    )


def _caveat_sentence(facts: ResearchFacts) -> str:
    """What the reader should hold back, stated as plainly as the result."""
    parts: list[str] = []
    if facts.needs_sample_warning:
        parts.append(
            f"the sample contains only {facts.sample_size} "
            f"{_releases(facts.sample_size)}, which is {facts.sample_quality} "
            "historical evidence rather than a stable relationship, and no "
            "statistical test has been run"
        )
    missing = _unmeasured(facts)
    if missing:
        parts.append(
            f"{missing} further {_releases(missing)} could not be classified "
            "or priced and are excluded rather than assumed"
        )
    if not parts:
        return ""
    return _join(parts) + "."


def _exclusion_clause(facts: ResearchFacts) -> str:
    if not facts.excluded:
        return ""
    total = sum(facts.excluded.values())
    return (
        f"{total} {_releases(total)} fell out for a stated reason, listed "
        "under the methodology."
    )


def _unmeasured(facts: ResearchFacts) -> int:
    """Releases lost to missing data rather than to the question."""
    return sum(
        count
        for reason, count in facts.excluded.items()
        if "could not be determined" in reason or "no usable price data" in reason
    )


def interpretation_payload(facts: ResearchFacts) -> dict[str, Any]:
    """Exactly what the language model is allowed to see, for this shape.

    Statistics and verdicts, already settled. No candles, no per-release price
    path, no HTML, no application source, no conversation history.

    The payload differs by shape, and that is the point rather than an
    optimisation. A ranked result carries no ``direction_vs_baseline``, because
    there is nothing to have a direction *against*; sending one set to
    ``undetermined`` would hand the model a field to reason about and a
    vocabulary of outperformance to reach for. What is not in the payload
    cannot be said.
    """
    payload: dict[str, Any] = {
        "question": facts.question,
        "asset": facts.asset,
        "event_type": facts.event_type,
        "horizon": facts.horizon,
        "measure": facts.measure,
        "metric": facts.metric_label,
        "result_shape": facts.shape.value,
        "sample_size": facts.sample_size,
        "warnings": list(facts.warnings),
        "excluded_counts": dict(facts.excluded),
        "methodology": dict(facts.methodology),
    }

    if facts.shape is ResultShape.RANKED:
        payload["ranking"] = {
            "extreme": facts.extreme,
            "ranked_from": facts.ranked_from,
            "rows": [
                {"rank": row.rank, "date": row.event_date, "value": row.value_text}
                for row in facts.rows[:MAX_PAYLOAD_ROWS]
            ],
        }
        # No direction, no baseline, no sample warning: an observed extremum is
        # not an estimate and must not be discussed as one.
        return payload

    payload["cohort_rule"] = facts.cohort_rule
    payload["result"] = facts.value_text
    payload["sample_quality"] = facts.sample_quality
    payload["small_sample"] = facts.needs_sample_warning
    if facts.matched_count is not None:
        payload["matched_count"] = facts.matched_count

    if facts.shape is ResultShape.SCALAR:
        return payload

    payload["left"] = _group_payload(facts.left)
    payload["right"] = _group_payload(facts.right)
    payload["difference_pp"] = _round(facts.difference_pp)
    payload["difference_is_material"] = facts.difference_is_material
    #: Neutral, and always present: which group measured higher, decided here.
    payload["higher_side"] = facts.higher_side

    # `direction_vs_baseline` says better or worse, which only means something
    # when one side is every release. Sending it for two named groups would
    # hand the model a standard neither of them is, so it goes only where it
    # is true -- and takes the outperform/underperform vocabulary with it.
    if facts.right_is_baseline:
        payload["right_is_baseline"] = True
        payload["direction_vs_baseline"] = facts.direction.value
    return payload


def _group_payload(group: GroupFacts | None) -> dict[str, Any] | None:
    if group is None:
        return None
    payload: dict[str, Any] = {
        "label": group.label,
        "n": group.sample_size,
        "value": group.value_text,
        "sample_quality": group.sample_quality,
        "small_sample": group.needs_sample_warning,
    }
    if group.positive_count is not None:
        payload["positive_count"] = group.positive_count
        payload["positive_pct"] = _round(group.positive_pct, 1)
    return payload


def methodology_summary(finding: ResearchFinding) -> dict[str, str]:
    """A short, fixed description of how the figures were produced.

    Sent with the payload so the explanation can say "measured from the release
    price at 5-minute resolution" without inventing a method. Kept to a handful
    of lines; the full methodology block already lives in the report.
    """
    thesis = finding.thesis
    conditions = [CONDITION_FIELDS[c.field].source for c in thesis.conditions]
    aggregation = thesis.outcome.aggregation
    statistic = (
        "individual releases, ranked -- no statistic computed across them"
        if aggregation is Aggregation.ROWS
        else f"{aggregation.value} across releases"
    )
    return {
        "outcome": f"{thesis.outcome.describe()}, from the release price",
        "anchoring": (
            "the last candle closing at or before each boundary, so no "
            "measurement contains information from after the instant it "
            "describes"
        ),
        "candles": f"{finding.spec.interval.value} candles from Binance",
        "statistic": statistic,
        "conditions_read_from": "; ".join(conditions) or "no conditions applied",
    }


def _round(value: float | None, places: int = 2) -> float | None:
    return None if value is None else round(value, places)


def _releases(count: int) -> str:
    return "release" if count == 1 else "releases"


def _join(parts: list[str]) -> str:
    joined = parts[0] if len(parts) == 1 else ", and ".join(parts)
    return joined[0].upper() + joined[1:]


__all__ = [
    "MATERIAL_DIFFERENCE_PP",
    "Direction",
    "GroupFacts",
    "ResearchFacts",
    "RowFacts",
    "build_facts",
    "deterministic_summary",
    "interpretation_payload",
    "methodology_summary",
]
