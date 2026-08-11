# 003 — UTC everywhere internally; local time only at the edges

**Status:** Accepted · **Date:** 2026-08-11

## Problem

The system mixes three time regimes:

- Binance candles, timestamped in epoch milliseconds (UTC).
- FOMC announcements, published at a wall-clock time in `America/New_York`.
- Future analysis concepts such as "Monday open", which are timezone-relative.

New York is **UTC−5 in winter (EST)** and **UTC−4 in summer (EDT)**. A 14:00 ET
announcement is therefore 19:00 UTC for part of the year and 18:00 UTC for the
rest. Roughly half of all FOMC events fall on each side.

An hour of error here shifts every window. A "1 hour after the announcement"
measurement becomes an hour of unrelated price action. Nothing crashes and no
value looks obviously wrong.

## Options

1. Store a fixed UTC offset per event source. Fails on DST, and on historical
   DST rule changes.
2. Store a precomputed UTC column alongside the local time. Works until the two
   disagree — a re-import, a hand edit, a rule change.
3. Store local wall-clock time plus IANA zone; derive UTC on demand.

## Decision

Option 3, with three supporting rules:

1. **Every timestamp crossing a module boundary is timezone-aware UTC.** Naive
   datetimes raise (`ensure_utc`) rather than being assumed to be UTC.
2. **Event sources store local wall-clock time and an IANA zone key.** UTC is a
   derived property (`MarketEvent.timestamp_utc`), never a stored column, so a
   stored UTC value cannot drift out of sync with the local time beside it.
3. **Conversion lives in exactly one module** (`backtool.core.time`), and
   rejects wall-clock times that do not map to exactly one UTC instant.

## Reason

Rule 3 covers two cases a bare `datetime.replace(tzinfo=...)` handles silently
and wrongly:

- **Non-existent times** in a spring-forward gap (02:30 on a US March
  changeover). Python fabricates a nominal instant.
- **Ambiguous times** in a fall-back overlap (01:30 on a US November
  changeover), which occur twice. Python picks the first via `fold=0`.

Neither arises for a 14:00 ET release, but an event source that ever emits one
must fail loudly. `NonExistentLocalTimeError` and `AmbiguousLocalTimeError` make
that failure explicit.

Deriving UTC rather than storing it makes an entire class of bug
unrepresentable. There is one source of truth, so there is nothing to
desynchronise.

## Tradeoffs

- Derivation costs a `zoneinfo` lookup per access; mitigated with
  `functools.cached_property`, and irrelevant at this data scale.
- Requires an up-to-date IANA database. Python ships `tzdata` on Windows; on
  Linux the system database is used. Historical rules are stable, so this
  affects future dates far more than past ones.

## Verified by

`tests/unit/test_time.py` pins six real FOMC dates across DST boundaries —
including 2024-03-20 (ten days after spring-forward) and 2024-11-07 (four days
after fall-back) — and asserts both DST edge-case errors.
