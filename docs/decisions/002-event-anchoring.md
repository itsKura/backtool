# 002 — Event anchoring: the last candle that closed at or before T

**Status:** Accepted · **Date:** 2026-08-11

## Problem

An event announcement is an exact instant `T` (e.g. FOMC at 19:00:00 UTC).
Candles are intervals. To compute "the return in the 24 hours before the event"
we need a single price that represents the market *at* `T`.

`T` rarely aligns with the candle grid, and the obvious choice is wrong.

## Options

Suppose `T = 19:03:00 UTC` and we are using 5-minute candles. The candle
covering `T` is `[19:00, 19:05)`, which closes at 19:05.

1. **Close of the candle containing `T`.**
   This price is formed at 19:05 — two minutes *after* the announcement. Every
   "pre-event" return computed from it silently includes post-event price
   action. The pre-event window appears to predict the event.
2. **Open of the candle containing `T`.**
   The 19:00 open is clean, but it is up to one interval *stale*, and on an
   aligned event (`T = 19:00:00`) it is the first post-event tick — the same
   leak as option 1, one candle earlier.
3. **Close of the last candle that closed at or before `T`.**
   For `T = 19:03` that is the close of `[18:55, 19:00)`, known at 19:00.
   Contains no information from after `T`, by construction.

## Decision

Option 3. The anchor price is the **close of the last candle whose close time
is less than or equal to `T`**.

Every event result additionally records:

- `anchor_time` — the close time of the candle actually used;
- `anchor_lag` — `T − anchor_time`, the staleness of the anchor.

## Reason

Only option 3 is provably free of look-ahead: the anchor price was fully
observable to a market participant standing at instant `T`. Options 1 and 2 both
admit information from after the announcement, which is the single most common
way an event study produces impressive and completely fictitious results.

Recording `anchor_lag` makes the compromise auditable. On an aligned event with
complete data, `anchor_lag` is zero. A lag larger than one interval means
candles were missing, and the caller can decide whether to trust the
observation or drop it — rather than never learning it happened.

Note the convenient case: Binance candle `[18:55, 19:00)` closes exactly at
19:00, so an FOMC announcement at 19:00:00 UTC yields `anchor_lag = 0` with no
leakage. Alignment is a happy accident of the data, not something the code may
assume.

## Tradeoffs

- The anchor can be up to one full interval stale. On 1h candles around a
  volatile release, that staleness is material — which is an argument for
  running the analysis at 5m, not for changing the rule.
- Two intervals can produce two different anchor prices for the same event. This
  is correct and expected; the interval is part of the research specification
  and must be reported alongside results.

## Applies to

Window boundaries as well as the event itself. A window `[-24h, 0h]` resolves to
concrete UTC instants relative to `T`, and each boundary is then anchored by
this same rule.
