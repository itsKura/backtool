"""Checking a model's claims against the evidence it was given.

A schema stops a web page changing what the system *does*. It does not stop a
page changing what the system *believes*. A source asserting "the Merge
happened on 1 January 2020" produces a perfectly schema-valid answer, and
everything downstream then treats that date as fact.

So the structure is necessary and not sufficient, and this module is the rest:
deterministic checks, in Python, that every claim traces back to something
actually retrieved.

What it can do
--------------
* Give each retrieved source a stable id, assigned here, so a claim references
  evidence rather than supplying it. A source the model invented has no id and
  cannot be referenced.
* Read dates and times out of the excerpts the API returned, in the forms
  English prose actually writes them, and refuse a claim that the evidence does
  not contain.
* Notice that a source cited for one occurrence states a date the resolution
  cannot explain, which is what a contradicted source looks like from the
  outside -- scoped to that occurrence, so a page naming a neighbouring
  milestone is not mistaken for a rival claim about this one.

What it cannot do
-----------------
Tell a truthful source from a lying one. A single plausible page, uncontradicted
by anything else retrieved, will pass every check here -- the claim traces to
real evidence, and the evidence is wrong. Nothing short of judging source
credibility fixes that, and judging credibility is the thing being poisoned.

What it can notice is a contradiction that was *retrieved and then dropped*:
see :func:`unused_contradictions`. That is weaker than fact-checking and
stronger than nothing.

What remains is narrower and worth having: a claim must come from somewhere, and
when the retrieved sources disagree the disagreement survives rather than being
quietly resolved. The honest mitigation for the rest is that provenance reaches
the report, so a reader can check the page that produced a date.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

# One constant, not a second convention. The transport layer decides how several
# passages quoted from one page are written into a single string, and this module
# reads what it produced. Nothing here calls the client.
from backtool.ai.client import PASSAGE_SEPARATOR

#: Prefix for ids assigned to retrieved sources. Deliberately not a URL: the
#: model references evidence by a handle it was given, so there is no field in
#: which it could supply a source of its own.
SOURCE_ID_PREFIX = "source_"

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}

_MONTH_ALTERNATION = "|".join(sorted(_MONTHS, key=len, reverse=True))

#: An ordinal suffix on a day number: 1st, 2nd, 3rd, 10th.
#:
#: Written out rather than shared between the two patterns by accident. Prose
#: uses the ordinal in either order -- "August 10th, 2022" and "10th August
#: 2022" are the same date -- and supporting it on one side only is how a real
#: source date went unverified: "the 1st of December 2020" was unreadable while
#: "December 1st, 2020" was fine.
_ORDINAL = r"(?:st|nd|rd|th)"

#: ISO: 2022-09-15
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
#: British: 15 September 2022 / 15th Sept 2022 / 1st of December 2020
_DAY_FIRST = re.compile(
    rf"\b(\d{{1,2}}){_ORDINAL}?\s+(?:of\s+)?({_MONTH_ALTERNATION})\.?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
#: American: September 15, 2022 / Sep 15 2022 / September 15th, 2022
_MONTH_FIRST = re.compile(
    rf"\b({_MONTH_ALTERNATION})\.?\s+(\d{{1,2}}){_ORDINAL}?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
#: A time of day: 06:42, 14:00, 6:42, 11:34am, 11:34:47 AM, 06:42:42 UTC.
#:
#: Matches loosely and validates in Python, the same way dates do. A regex that
#: encoded the hour and minute ranges itself would be unreadable and would still
#: need a second rule for the meridiem, where 12am is hour zero and 12pm is
#: hour twelve. Nothing here reads a timezone: the trailing "UTC" in "06:42
#: UTC" is left where it is, because which zone an event is in comes from the
#: resolver, not from a quoted fragment.
_TIME = re.compile(
    r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:([ap])\.?m\.?)?", re.IGNORECASE
)


@dataclass(frozen=True)
class EvidenceRecord:
    """One retrieved source, with the id the model will refer to it by.

    Built from what the API returned, never from what a model said. ``excerpt``
    is the ``cited_text`` the search tool supplied -- up to about 150
    characters, and often absent, which is the main limit on how much can be
    verified here.
    """

    source_id: str
    url: str
    title: str = ""
    excerpt: str = ""

    @property
    def can_ground(self) -> bool:
        """Whether this source carries text a claim could be checked against."""
        return bool(self.excerpt.strip())

    def merged_with(self, other: EvidenceRecord) -> EvidenceRecord:
        """One record of this page, keeping the richest form of every part.

        Defensive: :func:`~backtool.ai.client._read_blocks` already merges per
        url, so a duplicate normally cannot reach here. It is implemented anyway
        because the alternative -- dropping the second record -- is the exact
        defect that made this layer inert against a real response, and it should
        not be waiting one level up for the next producer to rediscover.

        The identity stays with the record that was already numbered: ids are
        positional, and a merge joins two readings of one page rather than
        introducing another.
        """
        if other.url != self.url:
            raise ValueError(
                f"Refusing to merge records of different pages: {self.url!r} "
                f"and {other.url!r}."
            )
        passages = sorted(
            {
                passage.strip()
                for text in (self.excerpt, other.excerpt)
                for passage in text.split(PASSAGE_SEPARATOR)
                if passage.strip()
            }
        )
        return replace(
            self,
            title=max(
                self.title, other.title, key=lambda text: (len(text), text)
            ),
            excerpt=PASSAGE_SEPARATOR.join(passages),
        )

    def dates(self) -> set[dt.date]:
        return excerpt_dates(self.excerpt)

    def has_time(self) -> bool:
        return bool(excerpt_times(self.excerpt))

    def times(self) -> set[dt.time]:
        return excerpt_times(self.excerpt)


def build_evidence(
    citations: Iterable[Any], *, limit: int = 40
) -> tuple[EvidenceRecord, ...]:
    """Assign ids to the sources a search actually returned.

    Ids are positional and stable within one resolution. They exist so that an
    event cites evidence instead of describing it: the model picks from this
    list or cites nothing, and there is no third option.

    One page gets one id however many times it was reported, and the readings
    are merged rather than raced -- see :meth:`EvidenceRecord.merged_with`.

    Args:
        citations: ``Citation`` objects from the search response.
        limit: Most sources to pass on, so a broad search cannot inflate the
            extraction payload without bound.
    """
    merged: dict[str, EvidenceRecord] = {}
    for citation in list(citations)[: limit * 2]:
        url = getattr(citation, "url", "")
        if not url:
            continue
        incoming = EvidenceRecord(
            source_id=f"{SOURCE_ID_PREFIX}{len(merged) + 1}",
            url=url,
            title=getattr(citation, "title", "") or "",
            excerpt=getattr(citation, "cited_text", "") or "",
        )
        existing = merged.get(url)
        merged[url] = existing.merged_with(incoming) if existing else incoming
        if len(merged) >= limit:
            break
    return tuple(merged.values())


def excerpt_dates(text: str) -> set[dt.date]:
    """Every calendar date stated in a piece of source text.

    Three orderings -- ISO, day-first and month-first -- each accepting an
    optional ordinal suffix on the day and an optional "of", which is how
    English prose actually writes a date. Anything unparseable is skipped rather
    than guessed at: this is a guard, and a guard that invents readings is worse
    than no guard.
    """
    found: set[dt.date] = set()

    for year, month, day in _ISO.findall(text):
        _add(found, int(year), int(month), int(day))
    for day, month, year in _DAY_FIRST.findall(text):
        _add(found, int(year), _MONTHS[month.lower().rstrip(".")], int(day))
    for month, day, year in _MONTH_FIRST.findall(text):
        _add(found, int(year), _MONTHS[month.lower().rstrip(".")], int(day))

    return found


def excerpt_times(text: str) -> set[dt.time]:
    """Every time of day stated in a piece of source text.

    Written forms, not a clock library: ``06:42``, ``06:42:42``, ``11:34am``,
    ``11:34 AM``, ``11:34:47AM``, and the same with a timezone written after
    them. The meridiem is optional and may be attached, spaced, or dotted,
    because sources write it every one of those ways -- and a form the parser
    cannot read makes a page that does state a time read as silent on it, which
    silently downgrades a real timestamp.

    Validated rather than trusted. An hour of 25 or a minute of 70 is a typo or
    something that was never a time, and a guard that accepts those is worse
    than no guard: ``has_time`` is what decides whether a claimed timestamp
    survives.
    """
    found: set[dt.time] = set()
    for hour, minute, second, meridiem in _TIME.findall(text):
        _add_time(found, hour, minute, second, meridiem)
    return found


def _add_time(
    found: set[dt.time], hour: str, minute: str, second: str, meridiem: str
) -> None:
    """Add a time, ignoring anything that is not one.

    The meridiem decides which hours are legal. With one, the hour is a clock
    face and must be 1 to 12, so "13:00pm" is not a time anybody wrote on
    purpose. Without one it is a 24-hour reading.
    """
    value, minutes = int(hour), int(minute)
    seconds = int(second) if second else 0
    if minutes > 59 or seconds > 59:
        return

    if meridiem:
        if not 1 <= value <= 12:
            return
        if meridiem.lower() == "a":
            value = 0 if value == 12 else value
        else:
            value = 12 if value == 12 else value + 12
    elif value > 23:
        return

    with contextlib.suppress(ValueError):
        found.add(dt.time(value, minutes, seconds))


def _add(found: set[dt.date], year: int, month: int, day: int) -> None:
    """Add a date, ignoring one that does not exist.

    A regex will happily match 31 February. Skipping it is right: this reads
    dates out of prose, and prose contains typos.
    """
    with contextlib.suppress(ValueError):
        found.add(dt.date(year, month, day))


def supports_date(records: list[EvidenceRecord], claimed: dt.date) -> bool | None:
    """Whether the cited evidence states the claimed date.

    Returns:
        ``True`` when an excerpt contains it, ``False`` when the excerpts
        contain dates but not this one, and ``None`` when there is nothing to
        check against -- no excerpt, or no date in it.

        The three-way answer matters. ``False`` is a claim the evidence
        contradicts and is refused; ``None`` is a claim the evidence is silent
        on, which is the common case given how short an excerpt is, and
        refusing those would reject almost everything true.
    """
    usable = [record for record in records if record.can_ground]
    if not usable:
        return None

    stated: set[dt.date] = set()
    for record in usable:
        stated |= record.dates()
    if not stated:
        return None
    return claimed in stated


def supports_time(records: list[EvidenceRecord]) -> bool | None:
    """Whether any cited excerpt states a time of day.

    The check that keeps the temporal-precision model honest. A claimed
    timestamp with no time anywhere in its evidence is precision the sources
    did not supply, and the only safe direction to fail is downward.

    Returns:
        ``True`` if an excerpt contains a time, ``False`` if excerpts exist and
        none does, ``None`` if there is nothing to check.
    """
    usable = [record for record in records if record.can_ground]
    if not usable:
        return None
    return any(record.has_time() for record in usable)


def distinct_dates(records: tuple[EvidenceRecord, ...]) -> set[dt.date]:
    """Every date stated anywhere in the retrieved evidence."""
    found: set[dt.date] = set()
    for record in records:
        found |= record.dates()
    return found


def evidence_exceeds_claims(
    records: tuple[EvidenceRecord, ...], claimed_dates: set[dt.date]
) -> set[dt.date]:
    """Dates the evidence states that the answer does not account for.

    Reported as a caveat, not as a verdict. Pages retrieved for one concept
    mention neighbouring milestones constantly -- a Merge page names Bellatrix,
    an election page names the previous election -- so "the evidence contains a
    date the answer does not" is far too weak to mean disagreement. It is still
    worth telling a reader, because the alternative is evidence quietly going
    unexplained.

    Whether sources actually *disagree* is :func:`disagreeing_sources`, which is
    scoped to one occurrence and its own citations.
    """
    return distinct_dates(records) - claimed_dates


def disagreeing_sources(
    records: Iterable[EvidenceRecord],
    claimed: dt.date,
    *,
    accounted: set[dt.date],
) -> tuple[EvidenceRecord, ...]:
    """The cited sources that dispute *this* occurrence's date.

    Conflict detection, scoped. A source disputes an occurrence when all three
    hold:

    * it was cited **for that occurrence** -- a page nothing relied on is
      background, not a competing claim;
    * it states dates but **not** the claimed one -- a page that contains the
      date agrees, however many other dates it also mentions, which is what
      keeps a timeline page from reading as a contradiction;
    * at least one of its dates is **unexplained by the resolution as a whole**
      -- so a page cited for the Merge that also dates Bellatrix is not in
      dispute when Bellatrix is itself one of the resolved events.

    That last clause is what the previous rule was missing. It compared the
    evidence against one event's date and called everything left over a
    conflict, which made "The Merge was September 15. Bellatrix was September
    6." look like two rival claims about one event.

    Args:
        records: The sources this occurrence cites.
        claimed: The date claimed for it.
        accounted: Every date the resolution explains, across all of its
            events, related ones included.

    Returns:
        The disputing sources, so a reader can be told which page says what.
    """
    disputing: list[EvidenceRecord] = []
    for record in records:
        stated = record.dates()
        if not stated or claimed in stated:
            continue
        if stated - accounted:
            disputing.append(record)
    return tuple(disputing)


def unused_contradictions(
    records: Iterable[EvidenceRecord],
    *,
    cited_ids: set[str],
    accounted: set[dt.date],
) -> tuple[EvidenceRecord, ...]:
    """Retrieved sources the answer neither cited nor explained.

    Evidence that was paid for and then left out. Reported as a warning rather
    than a verdict, because what it cannot tell is whether the leftover date
    belongs to *this* occurrence.

    A worked example of why. Asked about the Ethereum Merge, the search returns
    pages reading "Goerli Testnet merged August 10, 2022" and "The Merge
    occurred on January 1, 2020". Both name the concept, neither shares a date
    with the answer, and neither was cited. One is a different event; the other
    is a lie about this one -- and the only thing separating them is knowing
    that Goerli is a testnet, which is world knowledge and not in a
    150-character quotation. So this reports them and lets a reader look.
    Narrow enough to be usable. A page something relied on is not here, because
    whatever else it dates is context rather than evidence nobody used; a page
    restating a date the resolution already has is not here either. What is
    left is evidence in hand that the answer passed over.

    Associating an uncited source with a particular occurrence is not something
    this module can do; that comes from the extraction step, which read the
    whole page summary and can say which occurrence a source dates. See
    ``_associated`` in :mod:`backtool.ai.event_resolver`.
    """
    return tuple(
        record
        for record in records
        if record.source_id not in cited_ids and record.dates() - accounted
    )


__all__ = [
    "SOURCE_ID_PREFIX",
    "EvidenceRecord",
    "build_evidence",
    "disagreeing_sources",
    "distinct_dates",
    "evidence_exceeds_claims",
    "excerpt_dates",
    "excerpt_times",
    "supports_date",
    "supports_time",
    "unused_contradictions",
]
