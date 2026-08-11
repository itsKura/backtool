# 001 — Candle storage: Parquet files, not PostgreSQL

**Status:** Accepted · **Date:** 2026-08-11

## Problem

The system needs to persist historical OHLCV candles so that repeated research
runs do not re-download identical data. What storage engine should hold them?

## Options

1. **PostgreSQL** — the obvious default. Requires Docker, a schema, Alembic
   migrations, connection management, and a test database.
2. **TimescaleDB** — Postgres plus time-series partitioning and compression.
   Everything above, plus an extension to install and operate.
3. **Parquet files on local disk**, partitioned by symbol and interval, read
   with pyarrow/pandas.
4. **SQLite** — a middle ground; no server, but row-oriented and weaker at
   columnar range scans.

## Decision

Parquet files on local disk, reached only through a `CandleRepository`
protocol. Storage is an implementation detail behind that interface, not an
architectural commitment.

## Reason

The V1 dataset is small enough that a database earns nothing:

| Dataset | Rows | Approx. Parquet size |
| --- | ---: | ---: |
| BTCUSDT 5m, Aug 2017 → today | ~950,000 | ~50 MB |
| BTCUSDT 15m, same span | ~320,000 | ~18 MB |
| BTCUSDT 1h, same span | ~79,000 | ~5 MB |

That fits in memory several times over. Pandas reads the largest of these in
roughly 200 ms — comparable to, or faster than, the equivalent Postgres range
query once network and serialisation costs are counted.

Against that, Postgres would cost us, on day one: Docker (not installed on the
development machine), a schema, a migration tool, connection pooling, and CI
that starts a container. All of it before a single return is computed.

The access pattern also suits columnar files. Research reads a contiguous time
slice of a few columns for one symbol and interval — exactly what Parquet
predicate pushdown and column pruning are built for. There are no concurrent
writers, no transactions, and no relational joins on this data. Candles are an
append-only immutable time series, which is the weakest possible case for a
relational database.

## Tradeoffs

**What we give up**

- No concurrent multi-process writes. Acceptable: V1 is a single-process CLI.
- No SQL over candles. Acceptable: pandas is the query layer.
- Deleting or correcting rows means rewriting a partition file.

**What we accept as future work**

Postgres is likely to enter the project later, but for a different job: storing
**research runs** — specification, parameters, code version, and results — so
that a past analysis can be reproduced and audited. That is genuinely
relational, multi-user, and transactional. Candles are not.

## Revisit when

- Multiple processes or users need to write candles concurrently.
- The cache exceeds roughly 5 GB, or a single partition stops fitting in memory.
- The system is deployed as a hosted service rather than run locally.

The `CandleRepository` protocol exists so that this switch is a new
implementation plus one line of wiring, not a rewrite.
