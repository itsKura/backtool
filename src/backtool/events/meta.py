"""Descriptive metadata about each event type.

Static editorial copy, kept out of the calendar CSVs because it describes the
*type*, not any individual occurrence. A reader landing on an event page should
understand what it is and why it moves markets without leaving the page.

Deliberately factual rather than predictive. Describing the mechanism by which
an event affects prices is explanation; asserting what prices will do next is a
trading call, which this project does not make.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from backtool.core.types import EventType


class Importance(StrEnum):
    """How much a release typically moves markets.

    A coarse editorial judgement, not a computed figure. It exists to let a
    reader triage a calendar at a glance, and it is deliberately not derived
    from realised volatility -- that would make it look like a measurement when
    it is an opinion.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class EventMeta:
    """Everything an event page needs to explain one event type."""

    label: str
    importance: Importance
    #: Who publishes it, and on what schedule.
    source: str
    release_time: str
    #: Two or three sentences: what the number actually is.
    what_it_is: str
    #: Why the market reacts to it. Mechanism, not prediction.
    why_markets_care: str
    #: What a reader should look at on the day.
    what_to_watch: tuple[str, ...]


EVENT_META: dict[EventType, EventMeta] = {
    EventType.FOMC: EventMeta(
        label="FOMC rate decision",
        importance=Importance.HIGH,
        source="US Federal Reserve",
        release_time="14:00 America/New_York, second day of a two-day meeting",
        what_it_is=(
            "The Federal Open Market Committee sets the target range for the "
            "federal funds rate, the overnight rate US banks charge each other. "
            "Eight scheduled meetings a year, each ending with a statement and, "
            "at most meetings, a press conference 30 minutes later."
        ),
        why_markets_care=(
            "The federal funds rate is the anchor for the price of dollars. "
            "Raising it makes holding cash more attractive and borrowing more "
            "expensive, which tends to pull capital out of risk assets; cutting "
            "does the reverse. Crypto has no earnings to discount, so it tends "
            "to trade on liquidity conditions, which makes it unusually "
            "sensitive to this particular number. Most of the move usually "
            "comes not from the decision itself -- which is often widely "
            "anticipated -- but from the wording of the statement and what the "
            "Chair says afterwards."
        ),
        what_to_watch=(
            "The decision versus what was already priced in, not the decision alone.",
            "Changes in statement wording from the previous meeting.",
            "The 14:30 ET press conference, which often moves price more than the 14:00 statement.",
            "The dot plot, at the four meetings that include projections.",
        ),
    ),
    EventType.CPI: EventMeta(
        label="US Consumer Price Index",
        importance=Importance.HIGH,
        source="US Bureau of Labor Statistics",
        release_time="08:30 America/New_York, monthly",
        what_it_is=(
            "The Consumer Price Index measures the average change in prices "
            "paid by US urban consumers for a basket of goods and services. "
            "Published monthly, reporting on the previous month. Headline CPI "
            "includes everything; core CPI strips out food and energy, which "
            "are volatile for reasons unrelated to underlying inflation."
        ),
        why_markets_care=(
            "CPI is the most closely watched input to what the Fed does next. "
            "A hotter print pushes expectations toward tighter policy and "
            "higher rates; a cooler one does the opposite. Because it arrives "
            "monthly while FOMC decisions arrive eight times a year, CPI is "
            "often where rate expectations actually move -- the FOMC meeting "
            "then confirms what the CPI prints already implied."
        ),
        what_to_watch=(
            "Core CPI more than headline; the Fed weights it more heavily.",
            "The month-over-month change, which is more current than year-over-year.",
            "The release is at 08:30 ET, before the US equity open -- five and a half "
            "hours earlier than an FOMC statement.",
            "Revisions to prior months, which can change the trend without changing the headline.",
        ),
    ),
}


def meta_for(event_type: EventType) -> EventMeta:
    """Descriptive metadata for an event type.

    Raises:
        KeyError: if the type has no metadata. Every ``EventType`` must have an
            entry; a test asserts it, because a missing one would produce an
            event page with no explanation on it.
    """
    return EVENT_META[event_type]
