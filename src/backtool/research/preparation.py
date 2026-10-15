"""Everything a reader needs before a release, computed from candles and files.

The event page answers "what is this release and how has it behaved". This
answers the question a trader actually arrives with the morning of a print:
*what is about to happen, what has happened in each of the ways it could land,
and what does today look like against those.*

Nothing here is new arithmetic. The scenario split is
:func:`~backtool.research.cohort.split_study`, every statistic is
:func:`~backtool.research.aggregate.aggregate_window`, and the market going into
the release is measured by :func:`~backtool.research.metrics.compute_window_metrics`
-- the same function, with the same anchoring rule and the same staleness guard,
that produced every historical figure on the page. A "current 24h return"
computed any other way would be a second definition of a number the reader is
about to compare against twenty historical ones.

No model is called from here. The whole page is a dataset, a candle cache and a
comparison, and it costs nothing to open.

The pre-event condition
-----------------------
:func:`entry_cohorts` splits past releases by how the asset entered them, which
is the one join between today and history that is worth making. It reads
``pre_event_24h`` -- a window that ends *at* the release -- so a release is
selected by what was knowable before it printed and never by how it turned out.
That is the whole of the look-ahead argument, and it is why the condition is a
window rather than a filter on the outcome.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backtool.core.types import Interval
from backtool.research.cohort import Cohort, split_study
from backtool.research.metrics import WindowMetrics, compute_window_metrics
from backtool.research.results import EventResult, StudyResult
from backtool.research.windows import WindowSpec

if TYPE_CHECKING:  # pandas is only needed for the annotation
    import pandas as pd

#: How the asset entered a release, measured over the day before it.
#:
#: Appended to the reaction grid rather than replacing it, because the study has
#: to carry both: the outcome windows are what the page reports, and this is
#: what selects which releases it reports them over.
ENTRY_WINDOW = WindowSpec(name="pre_event_24h", start="-24h", end="0h")

#: What the market has done coming into *now*, for the panel beside the history.
#:
#: Expressed as windows ending at the present instant so they go through
#: :func:`compute_window_metrics` unchanged. The alternative -- a hand-rolled
#: "price now over price then" -- would be a second definition of a return,
#: differing from the historical one exactly at the boundaries that are hardest
#: to reason about and easiest to get silently wrong.
CONTEXT_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(name="last_1h", start="-1h", end="0h"),
    WindowSpec(name="last_4h", start="-4h", end="0h"),
    WindowSpec(name="last_24h", start="-24h", end="0h"),
)

#: Labels for the context windows, in the reader's terms.
CONTEXT_LABELS: dict[str, str] = {
    "last_1h": "Last hour",
    "last_4h": "Last 4 hours",
    "last_24h": "Last 24 hours",
}

#: The window whose sign decides which entry cohort today resembles.
CONTEXT_ENTRY_WINDOW = "last_24h"


@dataclass(frozen=True)
class MarketContext:
    """What the asset has done coming into the present instant.

    Every field may be absent, and absence is reported rather than defaulted.
    A symbol whose cache ends an hour ago has no usable last hour, and a zero
    there would read as "unchanged" -- the opposite of "unknown" and the kind
    of wrong number a reader has no way to detect.
    """

    symbol: str
    as_of: dt.datetime
    interval: Interval
    price: float | None
    windows: dict[str, WindowMetrics]

    def ret(self, name: str) -> float | None:
        """The return over one context window, or ``None`` if unmeasurable."""
        window = self.windows.get(name)
        if window is None or not window.is_usable:
            return None
        return window.return_pct

    @property
    def entry_return_pct(self) -> float | None:
        """How the asset is entering the release, over the last 24 hours."""
        return self.ret(CONTEXT_ENTRY_WINDOW)

    @property
    def entry_direction(self) -> str | None:
        """``"down"`` or ``"up"``, or ``None`` when the day is unmeasurable.

        Exactly zero counts as up, matching :func:`entry_cohorts`. The two have
        to agree or the panel would describe today as one thing and select
        history by another; a tie in a float return is vanishingly rare and the
        shared constant is cheaper than a third bucket holding one release.
        """
        value = self.entry_return_pct
        if value is None:
            return None
        return "down" if value < 0 else "up"


def build_market_context(
    candles: pd.DataFrame | None,
    *,
    symbol: str,
    now: dt.datetime,
    interval: Interval,
) -> MarketContext:
    """Measure the run-up to ``now`` from an already-fetched frame.

    Pure: the caller owns the fetch, so this is testable without a cache and
    cannot quietly reach the network while a page is rendering.

    The price is the last close in the frame rather than a separate lookup. It
    is therefore the same number the windows were measured against, which is
    worth more than being a few minutes fresher than they are.
    """
    if candles is None or candles.empty:
        return MarketContext(
            symbol=symbol, as_of=now, interval=interval, price=None, windows={}
        )

    windows = {
        spec.name: compute_window_metrics(candles, spec.resolve(now), interval)
        for spec in CONTEXT_WINDOWS
    }
    return MarketContext(
        symbol=symbol,
        as_of=now,
        interval=interval,
        price=float(candles["close"].iloc[-1]),
        windows=windows,
    )


def baseline_cohort(study: StudyResult) -> Cohort | None:
    """Every measured release as one cohort, for scenarios to be read against.

    A median of +1.6% after cooler prints means nothing until it sits beside
    what the release does in general. Built through :func:`split_study` rather
    than aggregated directly so the baseline and the cohorts it is compared
    with come out of the same code path -- a baseline computed a second way is
    a difference that can be an artefact of the arithmetic.
    """
    cohorts = split_study(
        study,
        lambda _event: (
            "all",
            "All releases",
            "Every release in the dataset that could be measured, "
            "whatever it printed.",
        ),
    )
    return cohorts[0] if cohorts else None


#: A rule that sorts one release into a named group, or declines to.
#:
#: Already bound to whatever the rule needs to read -- the CPI classifier wants
#: published figures, which live in a file this module has no business knowing
#: about. Taking it pre-bound keeps this about composition and nothing else.
ScenarioOf = Callable[[EventResult], "tuple[str, str, str] | None"]

#: The two ways an asset can have gone into a release.
#:
#: Two rather than three: the dataset has no tolerance band around zero that
#: would make "flat" mean anything, and inventing one would put releases in a
#: bucket whose boundaries nobody chose. An exactly unchanged day is counted as
#: up, which is stated rather than hidden, and is vanishingly rare in a float
#: return over 24 hours.
ENTERED_DOWN = "down"
ENTERED_UP = "up"


def entry_direction(event: EventResult) -> str | None:
    """How the asset went into this release, or ``None`` if unmeasurable.

    Reads ``pre_event_24h``, a window that ends *at* the release, so the answer
    was knowable before the figure printed. An unmeasurable day is declined
    rather than bucketed: it is not a flat one.
    """
    window = event.windows.get(ENTRY_WINDOW.name)
    if window is None or not window.is_usable or window.return_pct is None:
        return None
    return ENTERED_DOWN if window.return_pct < 0 else ENTERED_UP


def scenario_and_entry(scenario: ScenarioOf, *, direction: str) -> ScenarioOf:
    """A classifier for releases that both printed a given way and began one.

    Composed from the two splits rather than written as a third, so the
    intersection cannot drift from the groups it claims to intersect: the CPI
    rule here is the identical callable the scenario cards use, and the entry
    rule is the identical window the setup panel reads.

    Args:
        scenario: The outcome rule, keyed on the release's own figures. Returns
            ``None`` for a release it declines to classify -- no verified
            consensus, or sources that disagree -- and a declined release is
            absent from every intersection rather than defaulting into one.
        direction: :data:`ENTERED_DOWN` or :data:`ENTERED_UP`. Only releases
            the asset went into that way are kept.

    Returns:
        A classifier whose key is the scenario's, so the caller can look a
        cohort up by the scenario it belongs to and report *no observations*
        for one the intersection emptied.
    """

    def classify(event: EventResult) -> tuple[str, str, str] | None:
        if entry_direction(event) != direction:
            return None
        landed = scenario(event)
        if landed is None:
            return None
        key, label, _description = landed
        moved = "fallen" if direction == ENTERED_DOWN else "risen"
        return (
            key,
            label,
            f"Releases that printed {label.lower()} and which the asset had "
            f"{moved} into over the previous 24 hours.",
        )

    return classify


def intersection_cohorts(
    study: StudyResult, *, scenario: ScenarioOf, direction: str
) -> dict[str, Cohort]:
    """Intersection cohorts, keyed by the scenario each belongs to.

    A dictionary rather than a list because the interesting case is the empty
    one. A scenario with no matching release produces no cohort at all, and the
    page has to say *no observations matched both* rather than quietly omitting
    a row -- an absent row reads as an oversight, and a zero would be a
    statistic nobody computed.
    """
    return {
        cohort.key: cohort
        for cohort in split_study(
            study, scenario_and_entry(scenario, direction=direction)
        )
    }


def entry_cohorts(study: StudyResult) -> list[Cohort]:
    """Past releases split by how the asset entered them.

    The condition is the 24 hours *before* each release, so membership is fixed
    by what the market had already done when the figure printed. Nothing about
    the outcome participates in the selection, which is what keeps the
    comparison honest: a cohort chosen by its own result would report a
    spectacular statistic about nothing.

    A release whose entry window could not be measured is declined rather than
    bucketed. An unmeasurable day is not a flat one.
    """

    def classify(event: EventResult) -> tuple[str, str, str] | None:
        window = event.windows.get(ENTRY_WINDOW.name)
        if window is None or not window.is_usable or window.return_pct is None:
            return None
        if window.return_pct < 0:
            return (
                "entered_down",
                "Entered the release down",
                "Releases where the asset had fallen over the 24 hours "
                "immediately before the figure printed.",
            )
        return (
            "entered_up",
            "Entered the release up",
            "Releases where the asset had risen, or was unchanged, over the "
            "24 hours immediately before the figure printed.",
        )

    return split_study(study, classify)


def cohort_for_entry(study: StudyResult, context: MarketContext) -> Cohort | None:
    """The entry cohort today belongs to, or ``None`` when today is unknown.

    The join between the panel on the left and the history on the right. If the
    current day cannot be measured there is no comparable set to show, and
    showing the larger of the two would be answering a question nobody asked.
    """
    direction = context.entry_direction
    if direction is None:
        return None
    wanted = "entered_down" if direction == "down" else "entered_up"
    return next(
        (cohort for cohort in entry_cohorts(study) if cohort.key == wanted), None
    )
