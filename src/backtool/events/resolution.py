"""Events resolved at runtime, from outside the application.

Backtool cannot scale by compiling in every event somebody might ask about.
CPI and FOMC earn their place as *curated families*: each has a calendar file,
metadata, and a classifier that reads its surprise. Midterm elections, the
Ethereum Merge and a bank failure have none of those, and never will -- but
their dates are public facts, and once the dates are known the market question
is the same question the engine already answers.

This module is the shape those facts take. It contains no AI, no network, and
no measurement: pure models, so the thing that eventually fills them in has
somewhere to put facts and nowhere to put prose.

The trust boundary
------------------
Whatever resolves these -- a web search, a pasted list, a fixture -- produces a
:class:`ResolvedEventConcept` and nothing else. Only :func:`compile_event_set`
turns one into :class:`~backtool.events.models.MarketEvent` objects the engine
will measure, and it validates every date and zone on the way through. Webpage
text has no path into a study: it can only become a date, a label, and a
citation, and anything it cannot become is dropped at the schema.

That is why the provenance models live here from the beginning, while the only
caller is a test fixture. A later resolver has to populate *this* schema rather
than growing a second path beside it.

What this module deliberately does not decide
---------------------------------------------
Whether a resolution is good enough to run. A concept can be ambiguous,
contradicted by its own sources, or simply not found, and each of those is a
different thing to tell a reader. The status is recorded; acting on it belongs
to the layer that can ask a question.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backtool.core.time import UTC, floor_to_interval
from backtool.core.types import EventType, Interval
from backtool.events.models import MarketEvent, SourceRef, TemporalPrecision

#: Prefix for the identity of a set that was resolved rather than compiled in.
#: Kept distinct from ``curated:`` so no dynamic set can ever be mistaken for a
#: family with built-in metadata.
DYNAMIC_PREFIX = "dynamic"

#: How many characters of the digest to keep. Long enough that two genuinely
#: different event sets will not collide; short enough to print in a report.
FINGERPRINT_LENGTH = 16

#: The finest candle the engine has data for.
#:
#: Derived from :class:`~backtool.core.types.Interval` rather than written down,
#: so it follows the data the engine actually has. Add a finer interval to that
#: enum and every judgement built on this tightens by itself.
FINEST_MEASURABLE_SPAN = min(interval.duration for interval in Interval)


def resolves_to_one_candle(
    instants: Iterable[dt.datetime], *, span: dt.timedelta = FINEST_MEASURABLE_SPAN
) -> bool:
    """Whether every instant is priced from the same candle.

    The question "could the engine tell these apart" is about buckets, not
    about duration, and the difference is not academic: 12:04:55 and 12:05:05
    are ten seconds apart and anchor to different five-minute candles, while
    12:01 and 12:04 are three minutes apart and anchor to the same one. A rule
    comparing elapsed time to the candle length gets both of those wrong.

    Uses :func:`~backtool.core.time.floor_to_interval`, which is what the data
    layer uses to find the last closed candle boundary, so this agrees with how
    a study actually resolves a price rather than approximating it. One
    implementation, one answer.

    Args:
        instants: Timezone-aware instants. Fewer than two is not a comparison
            and trivially shares a bucket; a caller deciding whether two
            candidates differ must check that it has two.
        span: Candle length to bucket by. Defaults to the finest the engine
            has, which is the only span at which a difference could ever
            matter -- anything coarser buckets more things together.

    Returns:
        ``True`` when all of them fall in one bucket.
    """
    return len({floor_to_interval(instant, span) for instant in instants}) == 1


class ResolutionStatus(StrEnum):
    """How far a resolution got.

    Separate states rather than a boolean, because they call for different
    answers. "I found several things with that name" is a question for the
    reader; "trustworthy sources disagree about the date" is a warning that
    must not be resolved by picking one; "no such event" is a refusal.
    """

    #: Dates found and agreed on. Ready to compile.
    RESOLVED = "resolved"
    #: Found, but the question does not say which occurrences are meant.
    AMBIGUOUS = "ambiguous"
    #: Sources of comparable authority give materially different dates.
    CONFLICTING = "conflicting"
    #: Nothing matching was found.
    NOT_FOUND = "not found"
    #: Real, but with no publicly resolvable date -- a gradual period, a
    #: private event, something that never had a timestamp.
    NOT_RESOLVABLE = "not publicly resolvable"

    @property
    def is_usable(self) -> bool:
        return self is ResolutionStatus.RESOLVED


class EventRole(StrEnum):
    """Whether an occurrence *is* the concept, or merely belongs to its story.

    Resolving a public concept turns up more dates than the concept has. Ask
    about the Ethereum Merge and the sources also date Bellatrix, three testnet
    rehearsals and the Beacon Chain genesis; ask about a bank failure and they
    date the deposit run, the regulator's closure and the receivership. Each of
    those is real, dated and worth showing -- and none of them is the thing the
    reader asked about.

    Measuring them all would answer a question nobody asked, so the distinction
    is carried in the schema rather than left to whoever reads the output.
    :func:`compile_event_set` measures from :attr:`PRIMARY` occurrences only,
    while :attr:`RELATED` ones stay attached to the resolution as the context
    and provenance that explain it.

    Nothing here is domain-specific, because the distinction is not: it is the
    difference between an occurrence of the concept and a milestone around it,
    and that holds for an election, a protocol upgrade and a collapse alike. A
    recurring concept simply has many primary occurrences.
    """

    #: An occurrence of the concept itself. These become measurable events.
    PRIMARY = "primary"
    #: A milestone that led up to, followed from, or forms part of the story.
    #: Kept for explanation; never a study anchor unless asked for by name.
    RELATED = "related"


class AmbiguitySeverity(StrEnum):
    """Whether an open question has to be answered before measuring.

    Resolving a public concept raises questions of very different weight, and
    treating them alike makes the resolver useless. "Which banking crisis?"
    decides what gets measured. "Two sources disagree about a testnet rehearsal
    three months earlier" does not, and blocking a study over it wastes the
    reader's time on a question whose answer cannot change the result.

    The distinction is about consequence, not confidence. An open question
    blocks when answering it could change which occurrence is measured, or when
    it is measured; otherwise it is something the reader should be told and not
    something they must adjudicate first.
    """

    #: Answering it could change what gets measured. The study waits.
    BLOCKING = "blocking"
    #: Worth disclosing, cannot change the measurement. The study proceeds.
    INFORMATIONAL = "informational"


class Ambiguity(BaseModel):
    """Something the question leaves open that changes which events qualify.

    Carries its options and why it matters, so the reader is asked one specific
    question rather than told the request was unclear.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    options: tuple[str, ...] = ()
    why_it_matters: str = ""
    #: Defaults to blocking. An unlabelled question is one nobody has decided
    #: the consequence of, and the safe reading of that is "ask the reader":
    #: a spurious question costs one answer, a skipped one costs a wrong study.
    severity: AmbiguitySeverity = AmbiguitySeverity.BLOCKING
    #: How far apart the candidate instants were, when the question was about
    #: which instant an occurrence happened at. Computed from the candidates,
    #: never supplied: it is reported so a reader can see the size of what was
    #: set aside, and it is not what decided the severity -- see
    #: :func:`resolves_to_one_candle`.
    timing_slack: dt.timedelta | None = None

    @property
    def is_blocking(self) -> bool:
        return self.severity is AmbiguitySeverity.BLOCKING


class CandidateEvent(BaseModel):
    """One occurrence, as reported by a source.

    A candidate, not yet an event: it has not been validated into the engine's
    model and carries no identity. :func:`compile_event_set` does that.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    local_date: dt.date
    #: Time of day where the source gave one. ``None`` for something dated but
    #: not timed -- an election day, a ruling with no announced hour.
    local_time: dt.time | None = None
    timezone: str = "UTC"
    #: What the source established. Derived from ``local_time`` when omitted,
    #: and cross-checked against it either way, so a resolver cannot claim a
    #: precision it did not supply a time for.
    precision: TemporalPrecision | None = None
    #: Whether this is an occurrence of the concept or a milestone around it.
    #: Defaults to ``PRIMARY`` so a caller that knows it is building the
    #: occurrences themselves -- a pasted list, a fixture -- says nothing.
    role: EventRole = EventRole.PRIMARY
    #: Where this specific date came from. Required: a date with no source is
    #: exactly what this module exists to prevent reaching a study.
    sources: tuple[SourceRef, ...]
    notes: str = ""

    @field_validator("sources")
    @classmethod
    def _needs_a_source(cls, value: tuple[SourceRef, ...]) -> tuple[SourceRef, ...]:
        if not value:
            raise ValueError(
                "A resolved event needs at least one source. An undated claim "
                "with no provenance is not a fact this engine can measure."
            )
        return value

    @model_validator(mode="after")
    def _precision_matches_what_was_given(self) -> CandidateEvent:
        """Derive precision, or check a declared one against the time.

        Derived when omitted, so the common case stays terse. Checked when
        stated, so a resolver claiming ``TIMESTAMP`` without a time -- the
        exact failure this iteration exists to prevent, arriving from the other
        direction -- cannot get through.
        """
        implied = (
            TemporalPrecision.TIMESTAMP
            if self.local_time is not None
            else TemporalPrecision.DATE_ONLY
        )
        if self.precision is None:
            object.__setattr__(self, "precision", implied)
            return self
        if self.precision is not implied:
            raise ValueError(
                f"{self.label!r} is declared {self.precision.value} but "
                + (
                    "carries no time of day."
                    if self.local_time is None
                    else f"carries a time of day ({self.local_time:%H:%M})."
                )
            )
        return self

    @property
    def is_date_only(self) -> bool:
        return self.precision is TemporalPrecision.DATE_ONLY

    @property
    def is_primary(self) -> bool:
        return self.role is EventRole.PRIMARY


class ResolvedEventConcept(BaseModel):
    """What was asked for, what was found, and what is still open.

    Deliberately richer than a list of dates. A caller has to be able to tell
    "here are the two midterms" from "which midterms?" from "these sources
    disagree", and a bare list collapses all three into the same shape.
    """

    model_config = ConfigDict(frozen=True)

    #: The phrase the reader used, kept verbatim.
    query: str
    #: The name the sources use for it.
    canonical_name: str
    status: ResolutionStatus
    candidate_events: tuple[CandidateEvent, ...] = ()
    #: Sources consulted for the concept as a whole, beyond the per-event ones.
    sources: tuple[SourceRef, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    #: Facts that would have been needed and could not be established.
    missing_information: tuple[str, ...] = ()
    #: How the resolution was reached, in one or two sentences, for the report.
    resolution_notes: str = ""
    retrieved_at: dt.datetime = dt.datetime(1970, 1, 1, tzinfo=UTC)

    @property
    def slug(self) -> str:
        """A stable, readable key for this concept."""
        return slugify(self.canonical_name or self.query)

    @property
    def event_set_key(self) -> str:
        return f"{DYNAMIC_PREFIX}:{self.slug}"

    @property
    def primary_events(self) -> tuple[CandidateEvent, ...]:
        """The occurrences of the concept itself -- what a study measures.

        One for a one-time concept, many for a recurring one, and that
        difference needs no flag: it falls out of how many occurrences the
        sources establish.
        """
        return tuple(c for c in self.candidate_events if c.is_primary)

    @property
    def related_events(self) -> tuple[CandidateEvent, ...]:
        """Dated milestones around the concept.

        Kept, cited and reportable, but never measured by default. They are
        here so a reader can see what the resolution rests on and ask for one
        by name, not so a study silently acquires six extra anchors.
        """
        return tuple(c for c in self.candidate_events if not c.is_primary)

    @property
    def blocking_ambiguities(self) -> tuple[Ambiguity, ...]:
        """Questions a reader must answer before anything can be measured."""
        return tuple(item for item in self.ambiguities if item.is_blocking)

    @property
    def informational_ambiguities(self) -> tuple[Ambiguity, ...]:
        """Questions to disclose, which cannot change the measurement.

        Kept on the concept rather than folded into the notes, so a report can
        show them as the open questions they are without a reader having to
        answer them first.
        """
        return tuple(item for item in self.ambiguities if not item.is_blocking)

    @property
    def is_runnable(self) -> bool:
        """Whether this can be measured without asking the reader anything.

        Three conditions, and no more: the concept resolved, at least one
        occurrence of it survived validation, and nothing is left open that
        could change what gets measured. Informational questions do not appear
        here -- they are disclosed alongside the result, which is what they are
        for.

        Note what this does *not* cover: a date-only occurrence still needs an
        anchor, and :func:`compile_event_set` refuses without one. That is a
        decision the caller makes rather than a question the reader answers, so
        it belongs there and not in this property.
        """
        return (
            self.status.is_usable
            and bool(self.primary_events)
            and not self.blocking_ambiguities
        )


class DynamicEventSet(BaseModel):
    """A resolved concept, validated into events the engine can measure.

    The output of :func:`compile_event_set` and the only thing a study ever
    sees. By this point every date has survived timezone validation and every
    event carries a citation; nothing that reaches here is still prose.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    #: Identity of *these* events, not of the name. See :func:`fingerprint`.
    fingerprint: str
    events: tuple[MarketEvent, ...]
    sources: tuple[SourceRef, ...] = ()
    retrieved_at: dt.datetime = dt.datetime(1970, 1, 1, tzinfo=UTC)
    resolution_notes: str = ""

    @property
    def calendar(self) -> list[MarketEvent]:
        """The events as the engine takes them.

        A plain list of ``MarketEvent``, which is exactly what
        :func:`~backtool.research.runner.run_study` already accepts. No adapter,
        no second runner -- that is the whole claim this module is making.
        """
        return list(self.events)

    @property
    def identity(self) -> str:
        """Key and fingerprint together, for a research specification.

        Both halves matter. The key says which concept, the fingerprint says
        which resolution of it -- so a study of midterms resolved to two dates
        can never be served from a cache built when it resolved to three.
        """
        return f"{self.key}@{self.fingerprint}"


def slugify(name: str) -> str:
    """A stable key from a human name. ``"US Midterm Elections"`` -> ``us-midterm-elections``."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    return cleaned.strip("-") or "unnamed"


def curated_set_key(event_type: EventType) -> str:
    """The set key of a curated family."""
    return f"curated:{event_type.value.lower()}"


def fingerprint(slug: str, events: tuple[MarketEvent, ...]) -> str:
    """Identity of a specific set of events.

    Hashes the concept's slug together with the **UTC instant and label of
    every event**, sorted. Three properties follow, and each is a test:

    * Order does not matter. A calendar is a set; a source listing the same
      elections in a different order has not found different elections.
    * Content does. Change a date, add an occurrence, drop one, and the
      fingerprint moves -- so a cached study can never be served for a
      different resolution of the same question.
    * The name alone cannot collide. Two concepts that happen to share a label
      differ here as soon as their events differ, and the slug keeps them apart
      even if the events are identical.

    Normalised to UTC instants rather than local wall-clock, because the instant
    is what the engine measures from: the same moment written in two timezones
    is the same event, and would otherwise fingerprint as two.
    """
    # Precision and anchor are part of identity, not decoration. A sourced
    # 20:00 timestamp and a date-only day anchored to 20:00 land on the same
    # instant and are not the same fact -- one is evidence, the other a
    # decision -- so a study of one must never be served for the other.
    rows = sorted(
        "|".join(
            (
                event.timestamp_utc.isoformat(),
                event.temporal_precision.value,
                event.anchor,
                event.notes or event.event_id,
            )
        )
        for event in events
    )
    payload = json.dumps({"slug": slug, "events": rows}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


class AnchorPolicy(StrEnum):
    """How a calendar day is turned into an instant a study can measure from.

    Every member is a research choice, which is why they are named rather than
    defaulted. The choice changes the answer: anchoring an election at the
    start of its day puts the day's whole session after the event, anchoring at
    the end puts all of it before.
    """

    #: Midnight local. The whole calendar day falls after the anchor, which is
    #: usually what "after the election" means.
    DAY_START = "start of the calendar day"
    #: The last second of the day. Everything that happened during it falls
    #: before the anchor.
    DAY_END = "end of the calendar day"
    #: A specific local time the reader chose -- poll close, a session close,
    #: whatever they can defend.
    EXPLICIT_TIME = "a chosen local time"


class DayAnchor(BaseModel):
    """The research decision that turns a date into an instant.

    Kept as an object rather than a bare time so that what reaches a report is
    the *reason* -- "anchored at the end of the calendar day" -- rather than a
    number a reader would have to reverse-engineer.
    """

    model_config = ConfigDict(frozen=True)

    policy: AnchorPolicy = AnchorPolicy.DAY_START
    #: Required by, and only by, :attr:`AnchorPolicy.EXPLICIT_TIME`.
    local_time: dt.time | None = None
    #: Overrides the candidate's own timezone when the reader anchors to a
    #: market's clock rather than the event's.
    timezone: str = ""

    @model_validator(mode="after")
    def _time_matches_policy(self) -> DayAnchor:
        if self.policy is AnchorPolicy.EXPLICIT_TIME and self.local_time is None:
            raise ValueError(
                "An explicit anchor needs the local time to anchor at."
            )
        if self.policy is not AnchorPolicy.EXPLICIT_TIME and self.local_time:
            raise ValueError(
                f"{self.policy.value} fixes its own time, so "
                f"{self.local_time:%H:%M} would be ignored. Use "
                "AnchorPolicy.EXPLICIT_TIME to choose one."
            )
        return self

    def time_for(self) -> dt.time:
        """The local time this policy anchors at."""
        if self.policy is AnchorPolicy.DAY_START:
            return dt.time(0, 0)
        if self.policy is AnchorPolicy.DAY_END:
            return dt.time(23, 59, 59)
        assert self.local_time is not None  # guaranteed by the validator
        return self.local_time

    def zone_for(self, candidate_timezone: str) -> str:
        return self.timezone or candidate_timezone

    def describe(self, candidate_timezone: str = "") -> str:
        """The decision in words, for the event record and the methodology."""
        zone = self.zone_for(candidate_timezone)
        where = f" {zone}" if zone else ""
        if self.policy is AnchorPolicy.EXPLICIT_TIME:
            assert self.local_time is not None
            return f"{self.local_time:%H:%M}{where} (chosen)"
        return f"{self.policy.value} ({self.time_for():%H:%M}{where})"

    @property
    def key(self) -> str:
        """A stable identity for fingerprinting."""
        return f"{self.policy.value}|{self.local_time or ''}|{self.timezone}"


def compile_event_set(
    concept: ResolvedEventConcept, *, anchor: DayAnchor | None = None
) -> DynamicEventSet:
    """Turn a resolved concept into events the existing engine can measure.

    The single crossing point between resolved facts and measurement. Every
    date is validated here -- the timezone through ``MarketEvent``'s own
    validator, the instant through its UTC derivation -- so a malformed date
    fails at compilation rather than halfway through a study.

    This is also where the distinction between a source fact and a research
    choice is enforced. A study measures from an instant; a source that gave
    only a calendar day did not supply one. Rather than filling the gap with a
    default, compilation **refuses** until the caller states an anchor, and
    then records the anchor on every event it applied to.

    Only :attr:`EventRole.PRIMARY` occurrences are compiled. The related
    milestones a resolution turns up are real and dated, and measuring them
    would answer a question nobody asked.

    Informational questions do not stop compilation. They travel with the
    resolution to be disclosed beside the result, because a disagreement about
    a related milestone cannot change what an anchor on the concept measures.

    Args:
        concept: A resolution. Must be runnable; an ambiguous or conflicting
            one is a question for the reader, not an input to a measurement.
        anchor: How to turn a calendar day into an instant. Required if any
            primary occurrence is date-only, ignored otherwise -- a sourced
            timestamp is never overridden.

    Returns:
        A :class:`DynamicEventSet` whose ``calendar`` goes straight into
        :func:`~backtool.research.runner.run_study`.

    Raises:
        ValueError: if the concept is not runnable, carries no events, or
            contains a date-only event with no anchor to resolve it.
    """
    if not concept.status.is_usable:
        raise ValueError(
            f"{concept.canonical_name!r} resolved as {concept.status.value}, "
            "which is something to tell the reader rather than to measure."
        )
    if not concept.primary_events:
        related = len(concept.related_events)
        raise ValueError(
            f"{concept.canonical_name!r} resolved with no occurrence of the "
            "concept itself, so there is nothing to measure"
            + (
                f" -- only {related} related milestone(s), which are context "
                "rather than study anchors."
                if related
                else "."
            )
        )
    blocking = concept.blocking_ambiguities
    if blocking:
        raise ValueError(
            f"{concept.canonical_name!r} still has open questions: "
            + "; ".join(item.question for item in blocking)
        )

    undated = [c.label for c in concept.primary_events if c.is_date_only]
    if undated and anchor is None:
        raise ValueError(
            "These were established only to a calendar date, so measuring them "
            "needs a decision this engine will not make silently: "
            + "; ".join(undated)
            + ". Supply a DayAnchor saying where in the day to measure from."
        )

    slug = concept.slug
    key = concept.event_set_key
    # Primary occurrences only. A related milestone keeps its place in the
    # resolution's provenance and stays out of the calendar the engine measures.
    events = tuple(
        _compile_one(slug, key, candidate, anchor)
        for candidate in concept.primary_events
    )

    return DynamicEventSet(
        key=key,
        label=concept.canonical_name or concept.query,
        fingerprint=fingerprint(slug, events),
        events=events,
        sources=tuple({src.url: src for src in _all_sources(concept)}.values()),
        retrieved_at=concept.retrieved_at,
        resolution_notes=concept.resolution_notes,
    )


def _compile_one(
    slug: str, key: str, candidate: CandidateEvent, anchor: DayAnchor | None
) -> MarketEvent:
    """One candidate as a measurable event, with its timing provenance intact."""
    if candidate.is_date_only:
        assert anchor is not None  # guaranteed by the caller
        local_time = anchor.time_for()
        timezone = anchor.zone_for(candidate.timezone)
        anchor_note = anchor.describe(candidate.timezone)
    else:
        assert candidate.local_time is not None  # guaranteed by the validator
        local_time = candidate.local_time
        timezone = candidate.timezone
        anchor_note = ""

    return MarketEvent(
        event_id=_event_id(slug, candidate),
        event_type=None,
        event_set_key=key,
        local_date=candidate.local_date,
        local_time=local_time,
        timezone=timezone,
        is_scheduled=True,
        notes=candidate.label,
        source=candidate.sources[0],
        temporal_precision=candidate.precision or TemporalPrecision.TIMESTAMP,
        anchor=anchor_note,
    )


def _event_id(slug: str, candidate: CandidateEvent) -> str:
    """A readable, stable id. ``us-midterm-elections-2022-11-08``."""
    return f"{slug}-{candidate.local_date.isoformat()}"


def _all_sources(concept: ResolvedEventConcept) -> list[SourceRef]:
    """Concept-level and per-event sources, in a stable order."""
    found = list(concept.sources)
    for candidate in concept.candidate_events:
        found.extend(candidate.sources)
    return found


__all__ = [
    "DYNAMIC_PREFIX",
    "FINEST_MEASURABLE_SPAN",
    "FINGERPRINT_LENGTH",
    "Ambiguity",
    "AmbiguitySeverity",
    "AnchorPolicy",
    "CandidateEvent",
    "DayAnchor",
    "EventRole",
    "DynamicEventSet",
    "ResolutionStatus",
    "ResolvedEventConcept",
    "TemporalPrecision",
    "compile_event_set",
    "curated_set_key",
    "fingerprint",
    "resolves_to_one_candle",
    "slugify",
]
