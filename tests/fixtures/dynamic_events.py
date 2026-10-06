"""A resolved event concept, as a fixture rather than from the web.

Stands in for whatever eventually performs the resolution. It exists so the
deterministic half can be built and tested at zero cost, and so that the later
resolver has a worked example of the schema it must populate -- not a second
path of its own.

The dates are real US midterm election days and the citations are the real
agencies that publish them, because a fixture that cheats on provenance would
let the provenance models drift out of shape unnoticed.
"""

from __future__ import annotations

import datetime as dt

from backtool.core.time import UTC
from backtool.events.models import SourceAuthority, SourceRef
from backtool.events.resolution import (
    Ambiguity,
    CandidateEvent,
    ResolutionStatus,
    ResolvedEventConcept,
)

RETRIEVED = dt.datetime(2026, 10, 5, 12, 0, tzinfo=UTC)

FEC = SourceRef(
    url="https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/",
    title="Federal Election Commission election results",
    publisher="US Federal Election Commission",
    authority=SourceAuthority.OFFICIAL,
    retrieved_at=RETRIEVED,
)

EAC = SourceRef(
    url="https://www.eac.gov/",
    title="US Election Assistance Commission",
    publisher="US Election Assistance Commission",
    authority=SourceAuthority.OFFICIAL,
    retrieved_at=RETRIEVED,
)


def midterms(
    *,
    dates: tuple[dt.date, ...] = (dt.date(2018, 11, 6), dt.date(2022, 11, 8)),
    canonical_name: str = "United States midterm elections",
    ambiguities: tuple[Ambiguity, ...] = (),
    status: ResolutionStatus = ResolutionStatus.RESOLVED,
) -> ResolvedEventConcept:
    """US midterm elections, resolved.

    Parameterised so the fingerprint tests can vary one thing at a time --
    a date, an extra occurrence, the name -- without hand-building a concept
    each time and risking an unintended second difference.
    """
    return ResolvedEventConcept(
        query="midterm elections",
        canonical_name=canonical_name,
        status=status,
        candidate_events=tuple(
            CandidateEvent(
                label=f"{day.year} United States midterm elections",
                local_date=day,
                # Dated, not timed. Polls close at different hours by state, so
                # no single announcement instant exists and the source does not
                # claim one. Compilation refuses to invent it: measuring this
                # needs an anchor the caller states.
                local_time=None,
                timezone="America/New_York",
                sources=(FEC,),
            )
            for day in dates
        ),
        sources=(EAC,),
        ambiguities=ambiguities,
        resolution_notes=(
            "Election days taken from the Federal Election Commission's "
            "published results pages; the Election Assistance Commission "
            "confirms the same dates."
        ),
        retrieved_at=RETRIEVED,
    )


def eth_merge() -> ResolvedEventConcept:
    """A one-off event with a known timestamp, rather than a recurring family.

    The counterpart case: one occurrence, timed to the minute, from the
    project's own announcement. Included so the models are exercised against
    both shapes before either reaches a resolver.
    """
    source = SourceRef(
        url="https://blog.ethereum.org/2022/09/15/merge-blog",
        title="The Merge",
        publisher="Ethereum Foundation",
        authority=SourceAuthority.OFFICIAL,
        retrieved_at=RETRIEVED,
    )
    return ResolvedEventConcept(
        query="the Ethereum Merge",
        canonical_name="Ethereum Merge",
        status=ResolutionStatus.RESOLVED,
        candidate_events=(
            CandidateEvent(
                label="Ethereum transitions to proof of stake",
                local_date=dt.date(2022, 9, 15),
                # Timed to the minute by the project itself, so nothing here is
                # a research choice and no anchor applies.
                local_time=dt.time(6, 42),
                timezone="UTC",
                sources=(source,),
            ),
        ),
        sources=(source,),
        resolution_notes="Timestamp from the Ethereum Foundation's own announcement.",
        retrieved_at=RETRIEVED,
    )


def unresolvable() -> ResolvedEventConcept:
    """Something real with no single public date -- the shape a period takes."""
    return ResolvedEventConcept(
        query="the banking crisis",
        canonical_name="2023 United States regional banking crisis",
        status=ResolutionStatus.AMBIGUOUS,
        candidate_events=(),
        ambiguities=(
            Ambiguity(
                question="Which banking crisis, and what period counts as it?",
                options=(
                    "2023 US regional banks (SVB, Signature, First Republic)",
                    "2008 global financial crisis",
                    "a custom date range",
                ),
                why_it_matters=(
                    "A crisis is a period rather than an announcement, so the "
                    "dates measured decide the answer."
                ),
            ),
        ),
        missing_information=("No single announcement instant exists for a period.",),
        retrieved_at=RETRIEVED,
    )


__all__ = ["EAC", "FEC", "RETRIEVED", "eth_merge", "midterms", "unresolvable"]
