# 004 — SQLite instead of Parquet for the candle cache

**Status:** Accepted · **Date:** 2026-09-18 · **Amends:** [001](001-candle-storage.md)

## Problem

ADR 001 chose Parquet files for candle storage. On first contact with real
data, `pyarrow` could not load:

```
DLL load failed while importing _parquet:
An Application Control policy has blocked this file.
```

`pyarrow.parquet` and `pyarrow.feather` are both blocked by Windows Application
Control on the development machine. `fastparquet` would not install. Parquet is
simply unavailable in this environment.

## Options

1. **Weaken the Application Control policy** to permit the pyarrow DLLs.
   Rejected: trading a machine-wide security control for a storage format is a
   bad exchange, and it makes the project undeployable anywhere with a similar
   policy.
2. **gzipped CSV.** Works everywhere, but no indexing, slow range scans, and
   every read re-parses text into floats.
3. **SQLite.** Python standard library, no native dependency of our own, no
   installation step, and indexed range queries.

## Decision

SQLite, one database file holding every symbol and interval plus the coverage
metadata. `pyarrow` is dropped from the dependency list entirely.

Candles are stored in a `WITHOUT ROWID` table keyed on
`(symbol, interval, open_time_ms)`, so the primary key is the clustering key
and the only read pattern this system has — a contiguous time-range scan for
one symbol and interval — walks contiguous storage.

Timestamps are integer epoch milliseconds: the exchange's own unit, correct
under integer sort, and immune to the timestamp-resolution ambiguity documented
in `CANDLE_TIME_DTYPE`.

## Reason

The environment made Parquet non-viable, but SQLite is the better choice on the
merits anyway:

**It fixes an atomicity hole in the original design.** Parquet files plus a
separate `_coverage.json` manifest are two writes that can fail independently.
The dangerous ordering is coverage recorded without candles: that range is then
marked fetched forever, and research silently runs on data that was never
downloaded. With SQLite both commit in one transaction, so the failure mode
cannot occur. This is verified by
`test_failed_write_leaves_no_coverage_behind`.

**Merging becomes trivial.** `INSERT OR REPLACE` makes re-fetching idempotent
and lets a corrected candle overwrite a stale one. The Parquet path needed
read-concat-deduplicate-rewrite of a whole partition for every update.

**Zero installation risk.** `sqlite3` ships with Python. There is no native
wheel to be blocked, no compiler, no platform-specific build.

The reasoning in ADR 001 is unchanged and still points here: the dataset is
~50MB, there is a single writer, no transactions across entities, and no joins.
SQLite satisfies all of that without the operational cost of a server.

## Tradeoffs

- Larger on disk than columnar Parquet, and no column pruning. Irrelevant at
  this volume; we read whole rows anyway.
- Slower for full-history analytical scans. Our access pattern is narrow time
  ranges, which is exactly what the clustered index serves.
- One writer at a time. Already true of the V1 CLI.

## Unchanged

The `CandleRepository` Protocol from ADR 001 did its job: swapping the entire
storage backend touched one class and one line of wiring in
`MarketDataService.from_settings`. Nothing in the research engine changed.

## Revisit when

Same triggers as ADR 001 — concurrent writers, a hosted deployment, or a cache
beyond a few GB. SQLite will handle considerably more than V1 needs before any
of that becomes true.
