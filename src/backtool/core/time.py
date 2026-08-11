"""Time primitives and the project-wide UTC standard.

Internal standard
-----------------
Every timestamp that crosses a module boundary inside backtool is a
timezone-aware ``datetime`` in UTC. Naive datetimes are treated as bugs, not as
"probably UTC" -- see :func:`ensure_utc`.

Conversions between UTC and a local wall-clock time (e.g. the 2:00 PM
``America/New_York`` FOMC announcement) happen only here, and only explicitly.
The reason is that ET is UTC-5 in winter and UTC-4 in summer: hardcoding either
offset silently corrupts roughly half of all event timestamps, which in turn
shifts every event window by an hour without producing any visible error.

See ``docs/decisions/003-utc-internal-time-standard.md``.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

#: Re-exported so every module imports UTC from one place, making the internal
#: standard greppable rather than an unstated convention.
UTC = dt.UTC

#: Timezone of US macroeconomic releases (FOMC, CPI, NFP). Uses the IANA
#: database so historical DST rule changes are handled correctly.
NEW_YORK = ZoneInfo("America/New_York")


class LocalTimeError(ValueError):
    """Base class for wall-clock times that do not map cleanly onto UTC."""


class NonExistentLocalTimeError(LocalTimeError):
    """The local time never occurred (it fell in a DST spring-forward gap)."""


class AmbiguousLocalTimeError(LocalTimeError):
    """The local time occurred twice (it fell in a DST fall-back overlap)."""


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """Return ``value`` as a UTC-aware datetime.

    Naive datetimes raise rather than being assumed to be UTC. Silently
    adopting a timezone is how an hour-shifted backtest looks correct.

    Raises:
        ValueError: if ``value`` is timezone-naive.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"Naive datetime {value!r} crossed a module boundary. "
            "All backtool timestamps must be timezone-aware."
        )
    return value.astimezone(UTC)


def local_to_utc(
    local_date: dt.date,
    local_time: dt.time,
    timezone: ZoneInfo,
) -> dt.datetime:
    """Convert a local wall-clock date/time into an exact UTC instant.

    Unlike a bare ``datetime.replace(tzinfo=...)``, this rejects the two
    wall-clock times that do not correspond to exactly one UTC instant:

    * **Non-existent** -- clocks jumped forward over it (e.g. 02:30 on a US
      spring-forward Sunday). Python would silently invent an instant for it.
    * **Ambiguous** -- clocks fell back and it happened twice. Python would
      silently pick the first occurrence via ``fold=0``.

    Neither case arises for a 14:00 ET FOMC release, but an event source that
    ever emits one must fail loudly rather than produce a plausible wrong
    answer.

    Args:
        local_date: Calendar date in ``timezone``.
        local_time: Wall-clock time in ``timezone``.
        timezone: IANA timezone the date/time are expressed in.

    Returns:
        The equivalent UTC-aware datetime.

    Raises:
        NonExistentLocalTimeError: the wall-clock time never occurred.
        AmbiguousLocalTimeError: the wall-clock time occurred twice.
    """
    naive = dt.datetime.combine(local_date, local_time)

    earlier = naive.replace(tzinfo=timezone, fold=0)
    later = naive.replace(tzinfo=timezone, fold=1)

    # A DST gap: the two folds disagree on offset AND the value does not
    # survive a round trip through UTC (Python fabricates a nominal instant).
    if earlier.utcoffset() != later.utcoffset():
        round_tripped = earlier.astimezone(UTC).astimezone(timezone)
        if round_tripped.replace(tzinfo=None) != naive:
            raise NonExistentLocalTimeError(
                f"{naive} does not exist in {timezone.key}: clocks jumped "
                "forward over this wall-clock time."
            )
        raise AmbiguousLocalTimeError(
            f"{naive} is ambiguous in {timezone.key}: clocks fell back, so "
            "this wall-clock time occurred twice."
        )

    return earlier.astimezone(UTC)


def utc_to_local(instant: dt.datetime, timezone: ZoneInfo) -> dt.datetime:
    """Render a UTC instant as wall-clock time in ``timezone``.

    Used for display and for timezone-relative rules ("Monday open in New
    York"). Never used as an intermediate step in arithmetic.
    """
    return ensure_utc(instant).astimezone(timezone)


def utc_ms(instant: dt.datetime) -> int:
    """Return ``instant`` as integer milliseconds since the Unix epoch.

    Binance expresses every kline timestamp in epoch milliseconds, so this is
    the exchange-boundary representation.
    """
    return int(ensure_utc(instant).timestamp() * 1000)


def from_utc_ms(epoch_ms: int) -> dt.datetime:
    """Inverse of :func:`utc_ms`: epoch milliseconds to a UTC-aware datetime."""
    return dt.datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)
