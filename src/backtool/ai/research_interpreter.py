"""Explaining a finished finding in plain language.

This module writes prose about statistics somebody else computed. That is its
entire remit, and the import list is the proof: it takes a ``dict`` of already
settled facts and cannot reach :mod:`backtool.research` at all, let alone
:mod:`backtool.data`. There is no path from here to a candle, so there is no
path from here to a number that was not measured.

What it is allowed to do
------------------------
Say the facts well. A reader looking at ``+0.38%``, ``+0.95%`` and ``-0.57 pp``
has to assemble the meaning themselves; a paragraph that says "usually positive,
but weaker than a typical release, over only eight observations" is the same
information with the work done.

What it is not allowed to do
----------------------------
Decide anything. The direction of the comparison arrives pre-computed in the
payload, so the model is told the verdict rather than asked for one. And the
explanation is *checked* before it is shown: see :func:`violations`. An
explanation claiming outperformance over a negative difference, or explaining
the result by a cause nothing tested, is discarded in favour of the
deterministic summary. The caller always has that fallback, which is what
makes rejecting cheap enough to do on suspicion.

Configuration
-------------
Independent of the planner. ``BACKTOOL_INTERPRETER_MODEL`` selects the model for
this call alone, so summarising can move to something cheap and fast while
question-planning -- where a mistake silently changes what gets measured --
stays on the stronger model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backtool.ai.client import AIClient, AIError

logger = logging.getLogger(__name__)

#: Model used for interpretation when nothing else is set. Falls back to the
#: client's own default, so an account with one model configured keeps working.
INTERPRETER_MODEL_ENV = "BACKTOOL_INTERPRETER_MODEL"

#: Thinking effort for this call. Low deliberately: the reasoning was done by
#: the engine, and what remains is writing three paragraphs about six numbers.
INTERPRETER_EFFORT = "low"


class ResearchExplanation(BaseModel):
    """A reading of one finding, in the reader's language."""

    model_config = ConfigDict(frozen=True)

    paragraphs: list[str] = Field(
        description=(
            "Two or three short paragraphs, 100-250 words in total, explaining "
            "the supplied statistics in plain language. No new numbers beyond "
            "those supplied. No causal explanation of why the market moved."
        )
    )


def explain_research(
    payload: dict[str, Any],
    *,
    client: AIClient | None = None,
) -> tuple[str, ...]:
    """Explain settled statistics, or raise.

    Args:
        payload: The facts, built by
            :func:`backtool.research.interpretation.interpretation_payload`.
            A plain dict rather than a model, which is what keeps this module
            unable to import the engine that produced it.
        client: Optional pre-built client, for testing.

    Returns:
        The paragraphs, in reading order.

    Raises:
        AIError: on any API failure or refusal, or when the explanation
            contradicts the facts it was given. The caller falls back to the
            deterministic summary in both cases, so a rejection here costs the
            reader nothing but the better prose.
    """
    resolved = client or AIClient(model=os.getenv(INTERPRETER_MODEL_ENV) or None)

    explanation = resolved.structured(
        system=RESEARCH_INTERPRETER_SYSTEM,
        user=json.dumps(payload, indent=2, default=str),
        schema=ResearchExplanation,
        effort=INTERPRETER_EFFORT,
    )

    text = " ".join(explanation.paragraphs)
    problems = violations(text, payload)
    if problems:
        # Logged, not shown. The reader gets the deterministic summary and no
        # indication that a model was involved, because a notice about a
        # rejected explanation is noise to everyone except whoever maintains
        # this prompt.
        logger.warning("Discarding explanation: %s", "; ".join(problems))
        raise AIError(f"Explanation contradicted the facts: {'; '.join(problems)}")

    return tuple(p.strip() for p in explanation.paragraphs if p.strip())


#: Words asserting a cause, a certainty, or a prediction. None of them can be
#: supported by an event study over a few dozen observations: the test measures
#: what followed a release, never why, and never what will follow the next one.
#:
#: Matched on word boundaries so "proves" is caught and "approves" is not.
_FORBIDDEN_CLAIMS: tuple[tuple[str, str], ...] = (
    (r"\bcaused?\b", "asserts a cause"),
    (r"\bcausing\b", "asserts a cause"),
    (r"\bbecause of\b", "asserts a cause"),
    (r"\bdue to\b", "asserts a cause"),
    (r"\bproves?\b", "claims proof"),
    (r"\bproven\b", "claims proof"),
    (r"\bguarantees?\b", "claims a guarantee"),
    (r"\bwill (?:rise|fall|drop|rally|continue|repeat|happen)\b", "predicts"),
    (r"\bshould expect\b", "predicts"),
    (r"\bimplies that\b", "asserts an inference nothing tested"),
)

#: Phrases that assert the cohort beat the baseline, and phrases that assert
#: the reverse. Checked against the direction already computed, so a sentence
#: cannot disagree with the table printed beneath it.
_BETTER_CLAIMS = (
    "outperform",
    "performed better",
    "did better",
    "stronger than the baseline",
    "beat the baseline",
    "above the baseline",
)
_WORSE_CLAIMS = (
    "underperform",
    "performed worse",
    "did worse",
    "weaker than the baseline",
    "below the baseline",
    "lagged the baseline",
)


#: Words that cast one group as the standard the other is judged against.
#: Fine when the right-hand group really is every usable release; wrong when
#: the reader named two groups, where neither is the other's background.
_BASELINE_WORDS = (
    "baseline",
    "benchmark",
    "control group",
    "typical release",
    "the norm",
)


def violations(text: str, payload: dict[str, Any]) -> list[str]:
    """Reasons this explanation must not be shown.

    Two classes of problem, both cheap to detect and both fatal:

    * It contradicts the comparison. The direction was decided by the engine;
      prose that reverses it would make the report argue with itself.
    * It explains the result. No test here established a mechanism, so a
      sentence about institutions de-risking is invention with a confident
      tone, and confident invention is the specific thing this product is
      built not to produce.

    Deliberately literal. A keyword list cannot catch every bad sentence, and
    it is not trying to -- it catches the failures that would be most damaging
    and least visible, and the prompt handles the rest. False positives cost
    one fallback to the deterministic summary, which is an acceptable price.
    """
    lowered = text.lower()
    found: list[str] = []

    for pattern, reason in _FORBIDDEN_CLAIMS:
        match = re.search(pattern, lowered)
        if match:
            found.append(f"{reason} ({match.group(0)!r})")

    # The better/worse check only applies where "the cohort" is unambiguous,
    # which is when the other side is every usable release. For two named
    # groups there is no implied subject, so the equivalent guard is that
    # neither may be presented as the standard.
    if payload.get("right_is_baseline"):
        direction = str(payload.get("direction_vs_baseline", ""))
        if direction == "worse":
            found += [
                f"claims outperformance against a worse result ({claim!r})"
                for claim in _BETTER_CLAIMS
                if claim in lowered
            ]
        elif direction == "better":
            found += [
                f"claims underperformance against a better result ({claim!r})"
                for claim in _WORSE_CLAIMS
                if claim in lowered
            ]
    elif str(payload.get("result_shape", "")) == "comparison":
        found += [
            f"calls a named group a baseline ({word!r})"
            for word in _BASELINE_WORDS
            if word in lowered
        ]

    return found


RESEARCH_INTERPRETER_SYSTEM = """\
You explain the result of a completed quantitative study to a trader. The study \
is already finished. Every number you are given was measured from exchange data \
by deterministic code, and your job is to say what those numbers mean in plain \
language -- not to analyse, re-derive, extend or second-guess them.

## Hard rules

Use only the numbers in the payload. Do not compute new ones: no ratios, no \
annualised figures, no percentages of percentages, no counts you worked out \
yourself. If a number is not in the payload, it does not go in your answer.

`result_shape` says what kind of answer you are explaining, and the payload \
carries only the fields that shape gives meaning to. Explain what is there and \
never reach for a field that is absent.

* `scalar with baseline` -- a cohort set against every release of the same \
type. `direction_vs_baseline` is the verdict and it is already decided. If it \
says `worse`, the cohort underperformed: say so, and never write that it \
outperformed, beat, or was stronger than the baseline. If it says `similar`, \
the difference was immaterial and you must not pick a side. Contradicting this \
field makes your whole explanation unusable.
* `scalar` -- one figure describing every release measured. There is no \
baseline and no comparison, so do not invent one, and do not describe the \
result as strong, weak, high or low: nothing here establishes what normal is.
* `ranked rows` -- specific releases and their dates, under `ranking`. These \
are observed extremes of the sample, not estimates, so do not call them \
limited evidence, do not hedge them as though they might be wrong, and do not \
suggest what a future release might do. Name the dates. Say how many releases \
they were drawn from.

Do not explain *why* the market behaved this way. The study tested what \
followed a release; it tested nothing about mechanism, positioning, \
institutions, liquidity or sentiment. Any such sentence would be invention in \
a confident voice.

## Language

Write "was associated with", "in this historical sample", "the observed \
difference", "the data shows", "does not provide strong evidence".

Never write "caused", "because of", "proves", "guarantees", "will rise", \
"will continue", or any claim about what happens next.

## What to cover

Two or three short paragraphs, 100-250 words in total.

1. The result itself, said as a trader would read it -- the figure and what it \
was measured over, or the ranked releases and their dates.
2. For a comparison only: how that sits against the baseline, in the direction \
the payload gives, with the difference in percentage points.
3. What holds the finding back: sample size where `small_sample` is true, and \
anything in `warnings` or `excluded_counts` a reader would want to know before \
acting. If nothing holds it back, say so briefly and stop -- do not manufacture \
a caveat, and never apply a sample-size caveat to a ranked result, where the \
figures are observations rather than estimates.

Plain prose. No headings, no bullet points, no field names from the payload: \
write "median 24-hour return", never "median_return_pct". Address the reader \
directly about what was measured, and let the restraint do the persuading.
"""


__all__ = [
    "INTERPRETER_EFFORT",
    "INTERPRETER_MODEL_ENV",
    "RESEARCH_INTERPRETER_SYSTEM",
    "ResearchExplanation",
    "explain_research",
    "violations",
]
