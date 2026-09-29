"""Published CPI values, and the scenarios they sort a release into.

What a release *said*, as opposed to when it happened. The calendar in
:mod:`backtool.events.calendar` answers the second; this answers the first, and
the two are separate files because they have completely different provenance. A
release date is announced a year ahead and never changes. A consensus forecast
is a survey of economists, transcribed by hand from contemporaneous reporting,
and every row carries the URL it came from.

One metric, deliberately
------------------------
:data:`CPI_METRIC` is the whole V1 definition: **headline CPI-U, all items,
year-over-year, not seasonally adjusted**. Nothing here mixes it with core CPI,
or with month-over-month. Those are different series that surprise in different
directions on the same day, and a cohort built from a mixture of them would be
measuring nothing in particular while looking exactly like one that measured
something.

What is allowed to be classified
--------------------------------
A release enters a scenario cohort only when it has a verified consensus that is
not disputed between sources. Three ways a row is held back, each reported
rather than hidden:

* **No consensus.** Four rows have none. They stay in the dataset and remain
  available for ordinary event studies and Similar Setups; they simply cannot
  have a surprise, and an unknown surprise is not "in line".
* **Conflicting consensus.** One row has Reuters at 2.5% and CNBC/Dow Jones at
  2.4%. Those classify the release differently, so V1 excludes it by default
  rather than picking the source that happens to be stored.
* **Unverified.** Anything without ``consensus_verified``.

Rows are never dropped for any of these. They are excluded from cohorts and
counted, because a dataset of 27 that quietly reported on 20 would be the one
genuinely dishonest thing this module could do.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib import resources

from backtool.core.types import EventType

#: The single series every figure in this module describes.
CPI_METRIC = "Headline CPI-U, all items, year-over-year, not seasonally adjusted"

#: The file rows are read from.
RELEASES_FILE = "releases.csv"

#: Percentages are compared as integer hundredths rather than as floats.
#:
#: The rule is a strict three-way comparison with no tolerance, so a release at
#: 3.0% against a consensus of 2.9% is hotter and one at 2.9% against 2.9% is in
#: line. Neither 2.9 nor 3.0 is exactly representable in binary floating point,
#: and comparing them through a subtraction is what produced an asymmetry in an
#: earlier tolerance-based version of this rule. Scaling to integers removes the
#: question entirely: two figures published to one decimal place either are the
#: same number or they are not.
_SCALE = 100


class CPIScenario(StrEnum):
    """How a CPI print landed against what was forecast."""

    HOTTER = "hotter"
    IN_LINE = "in_line"
    COOLER = "cooler"


class ConsensusMethod(StrEnum):
    """How the stored consensus figure was obtained.

    ``DERIVED`` means no source printed the consensus directly; it was
    reconstructed arithmetically from a statement about the gap -- CNBC said a
    2.4% actual was "0.1 percentage point above forecast", which fixes the
    forecast at 2.3%. That is sound, and it is still a different kind of fact
    from a quoted number, so it is recorded as such and disclosed.
    """

    EXPLICIT = "explicit"
    DERIVED = "derived"


class FOMCScenario(StrEnum):
    """How an FOMC decision landed. Defined, not yet produced.

    Needs the target rate before and after each meeting for hike/hold/cut, and
    pre-meeting market pricing for hawkish/dovish versus expectations. Neither
    is stored. Naming them keeps the shape of the eventual answer visible
    without inventing one.
    """

    HIKE = "hike"
    HOLD = "hold"
    CUT = "cut"
    HAWKISH = "hawkish_surprise"
    AS_EXPECTED = "as_expected"
    DOVISH = "dovish_surprise"


@dataclass(frozen=True)
class ReleaseValues:
    """One CPI release: what was expected, what printed, and how we know."""

    event_id: str
    #: The identifier as supplied in the dataset, kept for traceability back to
    #: the source file.
    source_id: str = ""
    release_date: dt.date | None = None
    reference_month: str = ""

    previous_yoy: float | None = None
    consensus_yoy: float | None = None
    actual_yoy: float | None = None

    actual_source: str = ""
    consensus_source: str = ""
    consensus_quote: str = ""

    first_release_verified: bool = False
    consensus_verified: bool = False
    consensus_method: ConsensusMethod | None = None
    consensus_conflict: bool = False
    notes: str = ""

    @property
    def surprise_pp(self) -> float | None:
        """Actual minus consensus, in percentage points, or ``None``.

        Never defaulted to zero. A missing consensus means the surprise is
        unknown; zero means the print landed exactly on forecast. Those are
        opposite claims, and collapsing them would file every unknown release
        under "in line".
        """
        if self.actual_yoy is None or self.consensus_yoy is None:
            return None
        return (
            round(self.actual_yoy * _SCALE) - round(self.consensus_yoy * _SCALE)
        ) / _SCALE

    def is_classifiable(self, *, include_conflicts: bool = False) -> bool:
        """Whether this release may enter a scenario cohort."""
        return self.exclusion_reason(include_conflicts=include_conflicts) is None

    def exclusion_reason(self, *, include_conflicts: bool = False) -> str | None:
        """Why this release is not in a cohort, in the reader's terms.

        Returned rather than merely implied by a ``False``, so the page can say
        which of three different things happened instead of showing one count
        of releases that "did not qualify".
        """
        if self.consensus_yoy is None:
            return "no verified consensus for this release"
        if not self.consensus_verified:
            return "consensus not verified"
        if self.consensus_conflict and not include_conflicts:
            return "sources disagree on the consensus figure"
        if self.actual_yoy is None:
            return "no first-release actual for this release"
        return None


def classify_cpi(
    values: ReleaseValues | None, *, include_conflicts: bool = False
) -> tuple[str, str, str] | None:
    """Sort a CPI release by a strict three-way comparison.

    No tolerance band. Actual above consensus is hotter, equal is in line,
    below is cooler -- the comparison a reader makes themselves when they see
    "3.0% versus 2.9% expected", and the one they can check.

    Args:
        include_conflicts: Admit releases whose sources disagree on the
            consensus. Off by default: the stored figure and the disputed one
            classify this release differently, so including it would be
            choosing a source silently.

    Returns:
        ``(key, label, description)``, or ``None`` when the release cannot be
        classified from what is stored.
    """
    if values is None or not values.is_classifiable(include_conflicts=include_conflicts):
        return None

    assert values.actual_yoy is not None  # guaranteed by is_classifiable
    assert values.consensus_yoy is not None
    actual = round(values.actual_yoy * _SCALE)
    consensus = round(values.consensus_yoy * _SCALE)

    if actual > consensus:
        return (
            CPIScenario.HOTTER.value,
            "Hotter than expected",
            "The published figure came in above the consensus forecast.",
        )
    if actual < consensus:
        return (
            CPIScenario.COOLER.value,
            "Cooler than expected",
            "The published figure came in below the consensus forecast.",
        )
    return (
        CPIScenario.IN_LINE.value,
        "In line",
        "The published figure matched the consensus forecast exactly.",
    )


def classify_fomc(values: ReleaseValues | None) -> tuple[str, str, str] | None:
    """Sort an FOMC decision. Always ``None`` today.

    Wired into :data:`SCENARIO_RULES` and exercised on every FOMC page, so the
    moment target rates are stored the cohorts appear without touching the UI.
    CPI's rule is deliberately not reused: a rate decision is a choice among
    three options, not a number that missed a forecast.
    """
    return None


#: A classifier maps a release's values to ``(key, label, description)``, or
#: ``None`` when it cannot be classified from what is stored.
Classifier = Callable[["ReleaseValues | None"], "tuple[str, str, str] | None"]

#: One classifier per event type.
SCENARIO_RULES: dict[EventType, Classifier] = {
    EventType.CPI: classify_cpi,
    EventType.FOMC: classify_fomc,
}

#: Scenario order on the page: hotter, in line, cooler. Fixed rather than
#: first-seen, so the tabs do not reorder themselves when the dataset changes.
SCENARIO_ORDER: tuple[str, ...] = (
    CPIScenario.HOTTER.value,
    CPIScenario.IN_LINE.value,
    CPIScenario.COOLER.value,
)


def scenario_gap(event_type: EventType) -> str:
    """Why an event type has no scenarios, in the reader's terms."""
    if event_type is EventType.CPI:
        return (
            "No CPI releases in the dataset carry a verified consensus figure, "
            "so none can be sorted by how they landed against expectations."
        )
    return (
        "Splitting decisions into hike, hold and cut needs the target rate "
        "before and after each meeting, and splitting them into hawkish or "
        "dovish versus expectations needs what the market had priced in "
        "beforehand. Neither is stored yet."
    )


def scenario_methodology() -> tuple[tuple[str, str], ...]:
    """How scenarios are defined, for the Data & methodology block."""
    return (
        ("CPI metric", CPI_METRIC),
        (
            "scenario rule",
            "actual above consensus is hotter, equal is in line, below is "
            "cooler; no tolerance band",
        ),
        (
            "comparison",
            "figures compared as integer hundredths of a percentage point, so "
            "two values published to one decimal are equal or they are not",
        ),
        ("actual values", "first-release figures from BLS news releases"),
        ("consensus sources", "Reuters, and CNBC citing Dow Jones"),
        (
            "missing consensus",
            "release stays in the dataset, is excluded from surprise "
            "classification, and is counted as excluded",
        ),
        (
            "conflicting consensus",
            "where two outlets published different forecasts for the same "
            "release, it is excluded by default rather than resolved silently",
        ),
        (
            "derived consensus",
            "where no source printed the forecast directly but stated the gap "
            "to it, the figure is reconstructed and marked derived",
        ),
    )


def to_event_id(source_id: str) -> str:
    """``cpi_2026_09_11`` -> ``CPI-2026-09-11``.

    The dataset and the calendar were built independently and name releases
    differently. Translating in one place means the join is either right for
    every row or visibly wrong for all of them, rather than silently dropping
    the rows whose spelling happened to differ.
    """
    return source_id.upper().replace("_", "-")


@lru_cache(maxsize=1)
def load_releases() -> dict[str, ReleaseValues]:
    """Published values keyed by **calendar** event id.

    Returns an empty mapping when the file is absent or header-only. Cached
    because the file is static for the life of a process.
    """
    try:
        source = resources.files("backtool.events.data").joinpath(RELEASES_FILE)
        text = source.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return {}

    values: dict[str, ReleaseValues] = {}
    for row in csv.DictReader(text.splitlines()):
        source_id = (row.get("event_id") or "").strip()
        if not source_id:
            continue
        values[to_event_id(source_id)] = ReleaseValues(
            event_id=to_event_id(source_id),
            source_id=source_id,
            release_date=_date(row.get("release_date")),
            reference_month=(row.get("reference_month") or "").strip(),
            previous_yoy=_percent(row.get("previous_yoy")),
            consensus_yoy=_percent(row.get("consensus_yoy")),
            actual_yoy=_percent(row.get("actual_yoy")),
            actual_source=_text(row.get("actual_source")),
            consensus_source=_text(row.get("consensus_source")),
            consensus_quote=_text(row.get("consensus_quote_or_paraphrase")),
            first_release_verified=_flag(row.get("first_release_verified")),
            consensus_verified=_flag(row.get("consensus_verified")),
            consensus_method=_method(row.get("consensus_method")),
            consensus_conflict=_flag(row.get("consensus_conflict")),
            notes=_text(row.get("notes")),
        )
    return values


def _percent(raw: str | None) -> float | None:
    """Parse ``3.4%`` into ``3.4``. Blank and ``NULL`` are missing, not zero.

    Raises:
        ValueError: on a cell that is neither blank nor a percentage. A typo in
            a consensus figure would otherwise become a silent ``None`` and drop
            that release out of every cohort without explanation.
    """
    text = (raw or "").strip()
    if not text or text.upper() == "NULL":
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError as exc:
        raise ValueError(f"Invalid percentage {raw!r} in {RELEASES_FILE}") from exc


def _date(raw: str | None) -> dt.date | None:
    text = (raw or "").strip()
    if not text or text.upper() == "NULL":
        return None
    return dt.date.fromisoformat(text)


def _flag(raw: str | None) -> bool:
    return (raw or "").strip().lower() == "true"


def _text(raw: str | None) -> str:
    text = (raw or "").strip()
    return "" if text.upper() == "NULL" else text


def _method(raw: str | None) -> ConsensusMethod | None:
    text = (raw or "").strip().lower()
    if not text or text == "null":
        return None
    return ConsensusMethod(text)


__all__ = [
    "CPI_METRIC",
    "SCENARIO_ORDER",
    "SCENARIO_RULES",
    "CPIScenario",
    "Classifier",
    "ConsensusMethod",
    "FOMCScenario",
    "ReleaseValues",
    "classify_cpi",
    "classify_fomc",
    "load_releases",
    "scenario_methodology",
    "scenario_gap",
    "to_event_id",
]
