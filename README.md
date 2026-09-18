# backtool

A deterministic event-study research engine for crypto markets, with an AI
planning and interpretation layer on top.

> **Status: in development.** The deterministic engine works end to end on
> real market data: it identifies FOMC events, fetches and caches Binance
> candles, and computes pre/post-event metrics with a no-look-ahead guarantee.
> Cross-event aggregation and the AI layer are next. See [Roadmap](#roadmap).

## The problem

You can already ask a chatbot "how does BTC behave around FOMC meetings?" and
get a fluent answer. The answer is usually made up. Language models do not
compute medians over twenty observations; they produce numbers that look like
medians.

backtool splits the job along the line where each side is actually good:

- **Deterministic Python** computes every number. Returns, volatility, hit
  rates, percentiles — all of it, from real candles, reproducibly.
- **The LLM** turns a vague research question into an explicit specification,
  and turns structured results into an explanation. It never sees a raw candle
  and never computes a statistic.

The output is not just a conclusion. It is a conclusion, the assumptions behind
it, and the individual observations supporting it.

## The ambiguity problem

Financial language is underspecified. "BTC was in an uptrend" could mean a
positive 30-day return, price above a 200-period moving average, a sequence of
higher highs, or a positive regression slope. These disagree, often materially.

backtool never picks silently. Every ambiguous concept is resolved into an
explicit, visible, editable rule before anything is computed:

```
Interpretation
  Uptrend     = 30-day return > 5%
  Countermove = Monday open → FOMC return < -1%
  Anchor      = last candle close at or before the announcement
```

If the definition is wrong, you can see that it is wrong and change it.

## Architecture

Dependencies point in one direction only:

```
core/        pure logic, zero I/O   (UTC standard, DST conversion, shared types)
   ↑
data/  events/     the I/O boundary (Binance client, SQLite cache, event calendar)
   ↑
research/    specification → windows → per-event metrics → aggregation
   ↑
reporting/   tables and charts
   ↑
cli/         entry point
   ↑
ai/          research planner + results interpreter
```

The load-bearing constraint: **`ai/` may never import `data/`.** The AI emits
structured specifications and consumes structured results. It cannot touch a
candle, so it cannot invent a number that looks like one.

### Key design decisions

| Decision | Why |
| --- | --- |
| [No PostgreSQL](docs/decisions/001-candle-storage.md) | ~50 MB of data. A server database would buy nothing and cost Docker, migrations, and a test DB. Hidden behind a `CandleRepository` protocol so it can change. |
| [SQLite, not Parquet](docs/decisions/004-sqlite-over-parquet.md) | pyarrow is blocked by Application Control on the dev machine — but SQLite is better anyway: stdlib, and candles + fetch-coverage commit in one transaction, closing an atomicity hole the file-based design had. |
| [Anchor = last close at or before `T`](docs/decisions/002-event-anchoring.md) | The obvious alternatives leak post-event prices into pre-event windows, which is how event studies produce impressive fiction. |
| [UTC internally, local time only at edges](docs/decisions/003-utc-internal-time-standard.md) | 2 PM ET is 19:00 UTC in winter and 18:00 UTC in summer. Getting this wrong shifts every window by an hour and looks completely fine. |

## Stack

Python 3.12 · pandas · NumPy · Pydantic · httpx · SQLite (stdlib) · pytest · ruff · mypy (strict)

FastAPI, a React front end, and Docker are deliberately **not** here yet. They
arrive when there is a working research engine worth serving.

## Setup

```bash
git clone <your-repo-url>
cd backtool
python -m venv .venv
```

Activate the environment — on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Then install in editable mode with the dev extras:

```bash
pip install -e ".[dev]"
```

Copy the example configuration:

```bash
cp .env.example .env
```

No API keys are required — Binance's public kline endpoints are
unauthenticated.

> **Note on the Binance endpoint.** `api.binance.com` returns HTTP 451 from a
> number of jurisdictions. The default base URL is therefore
> `https://data-api.binance.vision`, Binance's public market-data mirror, which
> serves the identical `/api/v3/klines` contract without geo-restriction.
> Override with `BINANCE_BASE_URL` if you need a different host.

## Testing

```bash
pytest
```

Tests that perform real network I/O are marked. To run only offline tests:

```bash
pytest -m "not network"
```

Lint and type-check:

```bash
ruff check . && mypy
```

### What is tested, and why

Correctness of financial calculation comes before everything else — a polished
UI over wrong arithmetic is worth less than nothing, because it is convincing.
Tests concentrate on:

- DST conversion pinned against six real FOMC dates on both sides of both
  changeovers
- Non-existent and ambiguous wall-clock times (DST gap and overlap)
- Structural invariants of the event calendar (uniqueness, weekday, count/year)
- Future-dated events being excluded from any analysis

## Data caveats

The FOMC calendar (81 events, 2017–2026) was verified row by row against the
Federal Reserve's published calendar on 2026-08-11, with no discrepancies. Full
provenance, verified exceptions, and the list of Fed calendar entries we
deliberately exclude are in [docs/data-sources.md](docs/data-sources.md).

Known limits:

- BTCUSDT spot on Binance begins 2017-08-17. Earlier events have no price data
  and must be reported as `insufficient_data`, never silently dropped.
- The scheduled 17–18 March 2020 meeting was cancelled and replaced by the
  unscheduled 15 March Sunday action, so 2020 has seven scheduled meetings.
- Non-rate-decision FOMC actions (framework votes, emergency facility
  authorisations) are excluded by design — see the data-sources doc.
- Announcement times before 2013 were 2:15 PM ET, not 2:00 PM. Out of range for
  this calendar, but relevant if it is ever extended backwards.

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| **1a** | UTC time standard, DST conversion, FOMC calendar | ✅ Done |
| **1b** | Window resolution and per-event metrics, on synthetic candles | ✅ Done |
| **1c** | Binance client, SQLite cache, gap detection | ✅ Done |
| **1d** | End-to-end run on one real event | ✅ Done |
| **2** | Many events; explicit coverage reporting | Next |
| **3** | Cross-event aggregation (mean, median, hit rate, percentiles) | |
| **4** | Charts and a CLI report | |
| **5** | AI planner (NL → specification) and interpreter (results → prose) | |

Beyond V1: more assets, more event types, funding/open-interest data, and
exploratory pattern discovery — the last of which needs careful handling of
multiple-hypothesis testing before it is trustworthy.

## Scope

V1 is deliberately narrow: **BTCUSDT**, **FOMC**, **5m/15m/1h**. Breadth is
easy to add and easy to get wrong. Correctness on one asset and one event type
comes first.
