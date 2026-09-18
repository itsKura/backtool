# Data sources and provenance

## FOMC calendar

**File:** `src/backtool/events/data/fomc_meetings.csv`
**Source:** Federal Reserve Board, <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm>
and the per-year historical pages (`fomchistorical<YEAR>.htm`).
**Verified:** 2026-08-11 — all 81 rows checked against the Fed's published
calendar. No discrepancies found.

### What the file records

The Fed's calendar lists **meeting dates**. This project needs **announcement
instants**, which is a different thing:

- A scheduled FOMC meeting runs two days. The policy statement is released at
  **2:00 PM ET on the second day**. Our `local_date` is that second day, so the
  Fed's "January 30-31" becomes our `FOMC-2024-01-31`.
- Unscheduled actions have their own release times, taken from the individual
  press release.

Announcement times were 2:15 PM ET before 2013 and 2:00 PM ET from 2013 onward.
Every event in this file postdates the change, so 2:00 PM applies uniformly to
scheduled meetings. Extending the calendar before 2013 requires handling both.

### Verified exceptions

| Event | Detail | Evidence |
| --- | --- | --- |
| `FOMC-2020-03-03` | Emergency 50bp cut, 10:00 ET (not 14:00) | Press release: "For release at 10:00 a.m. EST" |
| `FOMC-2020-03-15` | Emergency 100bp cut, Sunday, 17:00 ET | Press release: "For release at 5:00 p.m. EDT" |
| 2020 has 7 scheduled meetings | The scheduled 17–18 March meeting was cancelled; its business folded into the 15 March action. The Fed's 2020 historical page lists no March 17–18 meeting. | Fed 2020 historical page |
| `FOMC-2024-11-07` | Thursday, not Wednesday — meeting shifted for the US general election | Fed calendar: "November 6-7" |

### Deliberately excluded

These appear on the Fed's calendar but are **not interest-rate announcements**,
so including them would pollute an FOMC rate-decision event study:

| Date | What it was | Why excluded |
| --- | --- | --- |
| 2019-10-04 | Unscheduled conference call | Reserve-management operations, no rate decision |
| 2020-03-19, 03-23, 03-31 | Notation votes | Emergency lending facilities and swap lines, no rate decision |
| 2020-08-27 | Notation vote | Revised Statement on Longer-Run Goals (average inflation targeting) |
| 2025-08-22 | Single-day meeting | Approval of the updated Statement on Longer-Run Goals, released ~10:00 ET alongside the Jackson Hole speech |

The last two were genuinely market-moving despite not being rate decisions. If
the project later adds a `FED_FRAMEWORK` event type, they belong there — not in
the FOMC rate-decision series.

### Re-verifying

```bash
backtool events --all
```

Compare against the Fed calendar. Structural invariants
(`tests/unit/test_events.py`) catch transcription errors — duplicate ids,
non-midweek scheduled dates, wrong meetings-per-year — but cannot catch a date
that is simply the wrong Wednesday. That requires the source above.

## CPI calendar

**File:** `src/backtool/events/data/cpi_releases.csv`
**Source:** US Bureau of Labor Statistics —
<https://www.bls.gov/schedule/news_release/cpi.htm>,
<https://www.bls.gov/bls/news-release/cpi.htm>, and
<https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm>
**Verified:** 2026-09-18 — every row checked against BLS publication records.

### What the file records

**The publication instant, not the reference month.** CPI for a given month is
published weeks later, and it is the publication that moves markets. So
`CPI-2024-02-13` is the release *on* 13 February 2024, which reported January
2024 data; the reference month is recorded in the notes.

Releases are at **08:30 America/New_York**, five and a half hours earlier than
an FOMC statement. Anything phrased relative to the US session means something
different for the two.

### Coverage

| Years | Status |
| --- | --- |
| 2024 – 2026 | Verified, 34 rows (31 occurred as of 2026-09-18) |
| 2017 – 2023 | **Not included.** BLS archive pages for those years returned 404. |

The earlier years are omitted rather than reconstructed. A calendar of 31
verified releases is worth more than one of 100 where 70 are recalled, because
a wrong date does not fail — it silently measures the wrong hour. Adding a year
is a matter of appending verified rows.

### The 2025 appropriations lapse

The US government shutdown damaged this calendar in three distinct ways, none of
which a "monthly, around the 12th" rule would capture:

| Reference month | What happened |
| --- | --- |
| September 2025 | Published **24 October**, not ~15 October |
| **October 2025** | **Never published.** The survey data could not be collected, and could not be collected retroactively. There is no event. |
| November 2025 | Published **18 December**, not 10 December |
| January 2026 | Published **13 February**, not 11 February |

The three delayed releases are marked `is_scheduled=false` with the reason in
their notes. `tests/unit/test_cpi_calendar.py` asserts both irregular gaps (43
and 55 days) explicitly, so a regenerated calendar that smoothed them over
would fail.

## Market data

**Source:** Binance public REST API (`/api/v3/klines`), unauthenticated.

- BTCUSDT spot trading begins **2017-08-17**. FOMC events before that date have
  no price data and are reported as `insufficient_data`, never silently dropped.
- Binance is geo-restricted in some jurisdictions. `BINANCE_BASE_URL` in `.env`
  allows pointing at a regional mirror.
