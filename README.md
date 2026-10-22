# backtool

A deterministic event-study research engine for crypto markets, with an AI
planning and interpretation layer that is never allowed to touch a number.

Ask what BTC has done around CPI and FOMC releases, get an answer with the
assumptions, the sample, and the individual observations attached — or open a
preparation page before the next release and see what each way it could land
has historically meant.

```
Calendar → upcoming CPI → Prepare → scenarios, current setup, evidence
Research → ask in plain language → see the specification → run it
```

## The problem

You can already ask a chatbot "how does BTC behave around FOMC meetings?" and
get a fluent answer. The answer is usually made up. Language models do not
compute medians over twenty observations; they produce numbers that look like
medians.

backtool splits the job along the line where each side is actually good:

- **Deterministic Python** computes every number. Returns, volatility, hit
  rates, percentiles, excursions — all of it, from real candles, reproducibly.
- **The LLM** turns a vague research question into an explicit specification,
  and turns structured results into prose. It never sees a raw candle and never
  computes a statistic.

The output is not just a conclusion. It is a conclusion, the assumptions behind
it, and the individual observations supporting it.

## The ambiguity problem

Financial language is underspecified. "BTC was in an uptrend" could mean a
positive 30-day return, price above a 200-period moving average, a sequence of
higher highs, or a positive regression slope. These disagree, often materially.

backtool never picks silently. Every ambiguous concept is resolved into an
explicit, visible rule before anything is computed:

```
Interpretation
  Uptrend     = 30-day return > 5%
  Countermove = Monday open → FOMC return < -1%
  Anchor      = last candle close at or before the announcement
```

If the definition is wrong, you can see that it is wrong and change it.

## What it does

### Event preparation (`/event/{id}/prepare`)

The main surface. For an upcoming CPI release:

- **Published figures** — previous, consensus, actual, each stating *why* it is
  absent when it is. A forecast nobody has published and one this dataset has
  not collected are different facts and are not both rendered as a dash.
- **Market going in** — current price and 1h / 4h / 24h returns, measured by the
  same function that produced every historical figure beside them.
- **Scenario playbook** — how the asset behaved after hotter, in-line and cooler
  prints, against an all-releases baseline, at five horizons.
- **Similar historical setups** — those same scenarios narrowed to releases the
  asset *entered the same way* as today. Both conditions at once.
- **The releases behind the numbers** — every observation, with the figures that
  classified it, so the arithmetic can be checked by hand.

After the release the same page reports what actually happened, horizon by
horizon, and withholds any window that has not closed.

Opening it costs **nothing**: no planner, no resolver, no model call.

### Research (`/research`)

Ask in plain language. The planner turns the question into a specification,
shows it for confirmation *before* running anything, then the engine measures
it and the interpreter explains the result. Supports horizons from 5 minutes to
2 years, conditional cohorts, A-vs-B comparisons and ranked queries.

### Calendar (`/`) and event pages

Upcoming and recent FOMC and CPI releases. Each event page explains what the
release is, shows its published figures, and charts what BTC or ETH did around
the last twenty occurrences.

## Architecture

Dependencies point in one direction only:

```
core/        pure logic, zero I/O   (UTC standard, DST conversion, shared types)
   ↑
data/  events/     the I/O boundary (Binance client, SQLite cache, calendars,
   ↑                                 BLS release figures)
research/    specification → windows → per-event metrics → aggregation → cohorts
   ↑
reporting/   tables and charts
   ↑
web/  cli/   FastAPI app and command line
   ↑
ai/          planner, event resolver, interpreters
```

The load-bearing constraint: **`ai/` may never import `data/`.** The AI emits
structured specifications and consumes structured results. It cannot touch a
candle, so it cannot invent a number that looks like one.

This is enforced by [`test_ai_boundary.py`](tests/unit/test_ai_boundary.py),
which checks the import graph both statically (AST) and at runtime (no
`backtool.data` module loads when `backtool.ai` is imported). It has caught two
real violations — a type import that dragged in the data layer, and a package
`__init__` that did the same transitively.

### Key design decisions

| Decision | Why |
| --- | --- |
| [Anchor = last close at or before `T`](docs/decisions/002-event-anchoring.md) | The obvious alternatives leak post-event prices into pre-event windows, which is how event studies produce impressive fiction. |
| [UTC internally, local time only at edges](docs/decisions/003-utc-internal-time-standard.md) | 2 PM ET is 19:00 UTC in winter and 18:00 UTC in summer. Getting this wrong shifts every window by an hour and looks completely fine. |
| [No PostgreSQL](docs/decisions/001-candle-storage.md) | ~50 MB of data. A server database would buy nothing and cost Docker, migrations, and a test DB. Hidden behind a `CandleRepository` protocol so it can change. |
| [SQLite, not Parquet](docs/decisions/004-sqlite-over-parquet.md) | Candles and fetch-coverage commit in one transaction, closing an atomicity hole the file-based design had. |
| [Anthropic API over HTTP, not the SDK](docs/decisions/005-anthropic-over-http.md) | Removes the last compiled dependency; the project now builds anywhere Python does. |

A few rules that are not ADRs but hold everywhere:

- **A statistic is computed once.** The browser never recomputes a high; it is
  given one. Two definitions of a number is one definition too many.
- **Absence is reported, never defaulted.** An unmeasurable window is not zero,
  a missing consensus is not "in line", and an unpublished figure is not a gap
  in the dataset.
- **A cohort is never selected by its own outcome.** Pre-event conditions use
  windows that end at the release; the look-ahead tests flip every outcome's
  sign and assert membership does not move.

## Stack

Python 3.12 · pandas · NumPy · Pydantic · httpx · FastAPI · SQLite (stdlib) ·
TradingView Lightweight Charts · pytest · ruff · mypy (strict)

No compiled dependencies beyond the scientific stack, no build step for the
front end, no Docker. The web layer renders server-side HTML and ships one
charting library from a CDN.

## Setup

```bash
git clone https://github.com/itsKura/backtool.git
cd backtool
python -m venv .venv
```

Activate the environment — on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install in editable mode with the dev extras, and copy the example config:

```bash
pip install -e ".[dev]"
```

```bash
cp .env.example .env
```

**No API key is required.** Binance's public kline endpoints are
unauthenticated, the BLS series is public, and every page except the
conversational research view works without a model.

> **Note on the Binance endpoint.** `api.binance.com` returns HTTP 451 from a
> number of jurisdictions. The default base URL is therefore
> `https://data-api.binance.vision`, Binance's public market-data mirror, which
> serves the identical `/api/v3/klines` contract without geo-restriction.
> Override with `BINANCE_BASE_URL` if you need a different host.

## Usage

### Web

```bash
backtool serve
```

Then open <http://127.0.0.1:8000>.

| Page | What it is |
| --- | --- |
| `/` | Calendar — upcoming and recent releases, with **Prepare** on the next CPI |
| `/event/{id}/prepare` | Event preparation: scenarios, current setup, evidence |
| `/event/{id}` | One event: what it is, its figures, and history around it |
| `/research` | Ask a question in plain language |
| `/study` | Advanced: set the specification by hand |

### Command line

```bash
backtool events -n 25
```

Run a study and write a self-contained HTML report with charts:

```bash
backtool analyse -n 20 --interval 5m --as-of 2026-09-18 --html reports/fomc-20.html
```

Ask in plain language — the AI plans the study, the engine runs it, the AI
explains the result:

```bash
backtool analyse --ask "How does BTC behave in the hour after FOMC?" --html reports/ask.html --open
```

Custom windows, repeatable:

```bash
backtool analyse -n 30 --window "pre_2h:-2h:0h" --window "post_4h:0h:4h"
```

Fill CPI actual and previous from the BLS published series (reports by default,
`--write` applies):

```bash
backtool refresh-cpi
```

Inspect the local candle cache:

```bash
backtool cache
```

Pin `--as-of` for a reproducible run; without it the event selection changes as
new releases happen. Every report prints its assumptions and a spec
fingerprint, so a set of numbers can always be traced back to the question that
produced it.

## The AI layer

Three roles, all optional — the engine and the preparation pages work fully
without an API key.

**Planner** turns a question into a specification, resolving every ambiguous
term into an explicit rule and showing them *before* running, so a wrong
definition can be rejected rather than silently inherited.

**Event resolver** handles events that are not in a curated calendar — a
halving, an election — by searching, then assigning claim-level provenance in
Python. The model cites sources; it never supplies them. Resolution runs in two
stages: the first has web search and produces prose, the second has no tools
and must emit a strict schema. Nothing a search returned can reach the
specification except through that second, toolless stage.

**Interpreter** explains computed statistics. It is given aggregates and
per-event returns — never candles — and every number it cites must appear
verbatim in its input.

Set `ANTHROPIC_API_KEY` in `.env` to enable them. A planned question costs
roughly 2–6 cents; curated pages cost nothing.

## Testing

```bash
pytest
```

**1,825 tests.** Two are skipped by default: a date-conditional one, and the
live planner matrix, which spends real money and is opt-in.

```bash
pytest -m "not network"     # offline only
ruff check . && mypy        # lint and strict type-check
```

### What is tested, and why

Correctness of financial calculation comes before everything else — a polished
UI over wrong arithmetic is worth less than nothing, because it is convincing.
Tests concentrate on the invariants whose violation would be *invisible*:

- DST conversion pinned against six real FOMC dates on both sides of both
  changeovers, plus non-existent and ambiguous wall-clock times
- No-look-ahead: cohort membership must not move when every outcome's sign is
  flipped
- Windows that have not closed stay unpriced rather than becoming zero
- A release is excluded from the comparison group explaining it
- The AI import boundary, statically and at runtime
- Recorded planner answers, so prompt changes are visible as diffs

The live planner suite replays 28 recorded answers for free and can be re-run
against the real model when the prompt changes.

## Data

| Source | Coverage |
| --- | --- |
| FOMC calendar | 81 meetings, 2017–2026, verified row by row against the Fed's published calendar |
| CPI calendar | 34 releases, 2024–2026 |
| CPI figures | previous and actual from the BLS published series (`CUUR0000SA0`); consensus hand-transcribed with a cited source per row |
| Candles | Binance BTCUSDT / ETHUSDT, 5m / 15m / 1h, cached in SQLite |

Full provenance, verified exceptions and deliberate exclusions are in
[docs/data-sources.md](docs/data-sources.md).

### Known limits

- **Consensus forecasts have no free source.** They are surveys sold by
  Bloomberg, Reuters and Dow Jones. `refresh-cpi` fills previous and actual
  automatically; consensus stays hand-transcribed, and a release without one is
  excluded from the scenario cohorts rather than counted as "in line".
- BTCUSDT spot on Binance begins 2017-08-17. Earlier events have no price data
  and are reported as `insufficient_data`, never silently dropped.
- The CPI calendar covers 2024 onwards. Earlier BLS archives were unreachable,
  and dates are omitted rather than reconstructed — a wrong release date does
  not fail, it silently measures the wrong hour.
- The 2025 US government shutdown delayed three CPI releases and cancelled one
  entirely; October 2025 was never published.
- The 17–18 March 2020 meeting was cancelled and replaced by the unscheduled
  15 March Sunday action, so 2020 has seven scheduled meetings.
- Non-rate-decision FOMC actions are excluded by design.
- Scenario samples are small — 3 to 10 releases per CPI outcome. Every cohort
  shows its count and a quality band, and no significance claim is made
  anywhere, because none is computed.

## Scope

Deliberately narrow: **BTCUSDT and ETHUSDT**, **FOMC and CPI**, **5m/15m/1h**
for short horizons and **1h** out to two years. Breadth is easy to add and easy
to get wrong, so correctness came first.

CPI was the test of whether the architecture generalises. It is released at
08:30 ET rather than 14:00, monthly rather than eight times a year — and adding
it took a calendar file and one enum entry. Nothing in the time layer, the
metrics, or the research engine changed, because an event is fully described by
a local wall-clock time and an IANA zone.

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 | UTC time standard, DST conversion, FOMC calendar, window metrics | ✅ |
| 2 | Binance client, SQLite cache, gap detection, many events | ✅ |
| 3 | Cross-event aggregation, charts, HTML report | ✅ |
| 4 | AI planner and interpreter | ✅ |
| 5 | Web interface: calendar, event pages, scenario and setup cohorts | ✅ |
| 6 | Conversational research, dynamic event resolution, long horizons | ✅ |
| 7 | CPI event-preparation workflow, intersection cohorts, BLS integration | ✅ |
| 8 | FOMC preparation, more assets, funding and open-interest data | Next |

Exploratory pattern discovery is deliberately last: it needs careful handling
of multiple-hypothesis testing before it is trustworthy.

## Licence

[MIT](LICENSE).
