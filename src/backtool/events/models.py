"""Event domain model."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from functools import cached_property
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backtool.core.time import local_to_utc
from backtool.core.types import EventType


class TemporalPrecision(StrEnum):
    """How precisely a source established *when* an event happened.

    Part of provenance, not of measurement. A source that says "the Merge
    happened at 06:42 UTC" and one that says "election day was 8 November" have
    established different kinds of fact, and flattening both into a timestamp
    loses the difference exactly where it matters -- in a report that claims to
    show its working.

    An event study needs an instant. A calendar day can be turned into one, but
    that is a research choice made by this project, and it has to stay
    distinguishable from what the source actually said.
    """

    #: The source gave an instant. Nothing was decided here.
    TIMESTAMP = "timestamp"
    #: The source gave a calendar day. Any instant is this project's choice.
    DATE_ONLY = "date only"


class SourceAuthority(StrEnum):
    """How much weight a source carries for a date.

    Ranked because conflicts have to be resolvable without guessing. A federal
    agency publishing its own schedule outranks an encyclopedia summarising it,
    and when two sources of the *same* rank disagree the honest answer is to
    report the conflict rather than pick one.
    """

    #: The organisation the event belongs to, publishing its own schedule.
    OFFICIAL = "official"
    #: A primary record of the event -- a press release, a filing, a
    #: contemporaneous announcement.
    PRIMARY = "primary"
    #: A reputable secondary account. Good for confirmation, not for
    #: establishing a date on its own.
    SECONDARY = "secondary"


class SourceRef(BaseModel):
    """Where a fact came from, kept so a report can show its working.

    Carried from the first iteration, while the only producer is a fixture, so
    that whatever resolves events later has to populate this rather than
    growing a parallel path with no provenance.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    title: str
    publisher: str = ""
    authority: SourceAuthority = SourceAuthority.SECONDARY
    retrieved_at: dt.datetime | None = None

    def describe(self) -> str:
        """One line for a citation list."""
        who = self.publisher or self.title
        return f"{who} ({self.authority.value}) -- {self.url}"


class MarketEvent(BaseModel):
    """A single scheduled (or unscheduled) macroeconomic event.

    The announcement instant is stored as **local wall-clock time plus IANA
    timezone**, never as a UTC column. UTC is derived on demand by
    :attr:`timestamp_utc`.

    This is deliberate. A stored UTC column can silently disagree with the
    local time beside it -- for example if a source is re-imported after a DST
    rule change, or if someone edits one field and not the other. Deriving UTC
    from a single source of truth makes that disagreement unrepresentable.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    #: The curated family this belongs to, or ``None`` for an event that was
    #: resolved at runtime rather than compiled in.
    #:
    #: ``None`` rather than a widened ``str``: the enum means "a family with its
    #: own calendar file, metadata and scenario classifier", and a midterm
    #: election has none of those. Letting one claim to be an ``EventType``
    #: would make every classifier lookup a lie that type-checks.
    event_type: EventType | None = None
    #: Stable identity of the set this event belongs to, independent of
    #: ``event_type``: ``"curated:cpi"`` or ``"dynamic:us-midterm-elections"``.
    #: Derived for curated events, so no calendar file changes.
    event_set_key: str = ""
    local_date: dt.date
    local_time: dt.time
    timezone: str
    is_scheduled: bool = True
    notes: str = ""
    #: What the *source* established. ``TIMESTAMP`` for every curated event --
    #: CPI is published at 08:30 ET and FOMC at 14:00 ET, both on the record --
    #: and for a dynamic event whose source gave a time.
    temporal_precision: TemporalPrecision = TemporalPrecision.TIMESTAMP
    #: How a calendar day was turned into the instant above, where it was.
    #: Empty when the time came from the source.
    #:
    #: The whole point of the pair: ``local_time`` alone cannot say whether
    #: 00:00 is a fact or a decision, and a report that cannot tell them apart
    #: presents a choice as evidence.
    anchor: str = ""
    #: Where the date came from. ``None`` for curated events, whose provenance
    #: is the checked-in calendar file. A dynamically resolved event must carry
    #: one, so a report can say why these dates were used.
    source: SourceRef | None = None

    @model_validator(mode="after")
    def _derive_set_key(self) -> MarketEvent:
        """Fill in the set key for a curated event.

        Means the CSV loaders and the calendar files need no change at all:
        anything with an ``EventType`` belongs to that family's set by
        definition.
        """
        if not self.event_set_key and self.event_type is not None:
            object.__setattr__(
                self, "event_set_key", f"curated:{self.event_type.value.lower()}"
            )
        return self

    @model_validator(mode="after")
    def _identity_is_complete(self) -> MarketEvent:
        """Every event belongs to some set.

        An event with neither a curated type nor a set key has no identity at
        all, which would let it be selected into any study.
        """
        if not self.event_set_key:
            raise ValueError(
                f"{self.event_id} has no event_type and no event_set_key, so "
                "it belongs to no event set."
            )
        return self

    @model_validator(mode="after")
    def _date_only_events_state_their_anchor(self) -> MarketEvent:
        """A date-only event must say how it got a time.

        This is the rule that stops a date becoming midnight silently. The
        instant exists -- the engine needs one -- but it cannot exist without a
        stated reason, so there is nowhere for an unexamined default to hide.
        """
        if self.temporal_precision is TemporalPrecision.DATE_ONLY and not self.anchor:
            raise ValueError(
                f"{self.event_id} was established only to a calendar date, so "
                f"{self.local_time:%H:%M} is a research choice rather than a "
                "fact. Compile it with an explicit anchor."
            )
        if self.temporal_precision is TemporalPrecision.TIMESTAMP and self.anchor:
            raise ValueError(
                f"{self.event_id} has a sourced timestamp, so anchoring it to "
                f"{self.anchor!r} would override the source rather than "
                "supplement it."
            )
        return self

    @property
    def is_curated(self) -> bool:
        return self.event_type is not None

    @property
    def time_is_sourced(self) -> bool:
        """Whether the instant came from the source rather than from us."""
        return self.temporal_precision is TemporalPrecision.TIMESTAMP

    def describe_timing(self) -> str:
        """How this event's instant was arrived at, for a methodology line."""
        if self.time_is_sourced:
            return (
                f"{self.local_date} {self.local_time:%H:%M} {self.timezone}, "
                "as published"
            )
        return (
            f"{self.local_date} ({self.timezone}), date only; "
            f"anchored at {self.anchor}"
        )

    @property
    def curated_type(self) -> EventType:
        """The curated family, for code that only ever handles curated events.

        Raises rather than returning ``None`` so the assumption is stated where
        it is relied on. The calendar and event pages are built around a
        family's metadata and classifiers; a dynamic event reaching one is a
        bug, and this is where it surfaces.

        Raises:
            ValueError: if this event was resolved at runtime.
        """
        if self.event_type is None:
            raise ValueError(
                f"{self.event_id} is a dynamically resolved event "
                f"({self.event_set_key}), not a curated event type."
            )
        return self.event_type

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        """Reject unknown IANA zone keys at load time, not at analysis time."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone {value!r}") from exc
        return value

    @cached_property
    def zone(self) -> ZoneInfo:
        """The event's timezone as a :class:`ZoneInfo`."""
        return ZoneInfo(self.timezone)

    @cached_property
    def timestamp_utc(self) -> dt.datetime:
        """Exact UTC instant of the announcement.

        Raises:
            LocalTimeError: if the recorded wall-clock time is non-existent or
                ambiguous in its timezone (a data error worth failing on).
        """
        return local_to_utc(self.local_date, self.local_time, self.zone)

    def __str__(self) -> str:
        return (
            f"{self.event_id} ({self.local_date} {self.local_time:%H:%M} "
            f"{self.timezone} = {self.timestamp_utc:%Y-%m-%d %H:%M} UTC)"
        )
