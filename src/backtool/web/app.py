"""FastAPI server: the research engine as a web page.

A thin layer over the existing engine. It parses a request, calls the same
``run_study`` and ``render_body`` the CLI uses, and returns the result. No
research logic lives here.

Two entry points by design:

* **Ask** -- a plain-language question, planned by the AI layer. Requires an
  API key.
* **Build** -- explicit controls for event count, interval, and windows. Needs
  no key at all.

The second is not a fallback for the first. The deterministic engine is the
product; the AI is a convenience on top of it, and the page stays fully usable
without it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse

from backtool.config import Settings
from backtool.core.time import UTC
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events import load_events
from backtool.events.models import MarketEvent
from backtool.reporting.html import render_body
from backtool.research.results import StudyResult
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.windows import DEFAULT_WINDOWS, WindowSpec
from backtool.web.calendar_page import render_calendar
from backtool.web.event_page import (
    CHART_SPAN,
    LIGHTWEIGHT_CHARTS_CDN,
    ai_summary_script,
    price_chart_script,
    render_ai_summary,
    render_event_explainer,
    render_event_header,
    render_outcome_placeholder,
    render_price_chart,
    render_upcoming_notice,
)
from backtool.web.page import render_error, render_page, shell

logger = logging.getLogger(__name__)

#: A study fetches candles for each event, so a cold run over many events is
#: slow. Capped so one request cannot tie up the server indefinitely.
MAX_EVENTS = 60

#: Symbols offered on an event page. Both list on Binance from 2017-08-17, so
#: neither truncates the history relative to the other.
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")

#: How many past occurrences an event page summarises by default.
DEFAULT_HISTORY_EVENTS = 20

app = FastAPI(
    title="backtool",
    description="Deterministic event-study research engine for crypto markets.",
    docs_url="/api/docs",
)


def _ai_available() -> bool:
    """Whether the AI layer is configured, checked per request.

    Not cached: a key added to the environment should take effect on the next
    page load rather than needing a restart.
    """
    from backtool.ai import is_configured

    return is_configured()


@app.get("/", response_class=HTMLResponse)
async def calendar() -> HTMLResponse:
    """The front door: what is coming up, and what just happened."""
    now = dt.datetime.now(tz=UTC)
    body = (
        '<div class="masthead"><h1>Calendar</h1>'
        "<p>Scheduled macroeconomic events, and what crypto did around them "
        "last time. Click any event for its history.</p></div>"
        + render_calendar(load_events(), now=now)
    )
    return HTMLResponse(shell("backtool - calendar", body, active="calendar"))


@app.get("/study", response_class=HTMLResponse)
async def study_form() -> HTMLResponse:
    """Advanced mode: full control over the specification."""
    return HTMLResponse(render_page(ai_available=_ai_available()))


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def event_detail(event_id: str, symbol: str = "BTCUSDT") -> HTMLResponse:
    """One event: what it is, and how the last occurrences behaved.

    The history is loaded asynchronously rather than inline. A cold cache means
    fetching candles for twenty events, and a reader should get the explanation
    immediately rather than staring at a blank tab.
    """
    event = _find_event(event_id)
    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]
    now = dt.datetime.now(tz=UTC)

    tabs = "".join(
        f'<a class="symbol-tab" href="/event/{event.event_id}?symbol={candidate}" '
        f'aria-selected="{str(candidate == chosen).lower()}">{candidate}</a>'
        for candidate in SYMBOLS
    )

    upcoming = event.timestamp_utc > now

    # Price first, explainer last. Someone who clicked a CPI release came to
    # see what price did, not to have CPI explained to them first.
    body = (
        render_event_header(event, now=now)
        + (render_upcoming_notice(event) if upcoming else "")
        + f'<div class="symbol-tabs">{tabs}</div>'
        + render_price_chart(event, chosen, available=not upcoming)
        + "<h2>Historical reaction</h2>"
        + '<div id="history"><p class="status"><span class="spinner"></span>'
        + "Loading the last "
        + str(DEFAULT_HISTORY_EVENTS)
        + " occurrences&hellip;</p></div>"
        + render_ai_summary(available=_ai_available(), has_history=True)
        + render_outcome_placeholder()
        + render_event_explainer(event)
    )

    history_script = (
        "fetch('/api/event/"
        + event.event_id
        + "/history?symbol="
        + chosen
        + "')"
        ".then(r => r.text())"
        ".then(html => { document.getElementById('history').innerHTML = html; })"
        ".catch(e => { document.getElementById('history').innerHTML = "
        "'<div class=\"error\"><h3>Could not load history</h3><p>' + e + '</p></div>'; });"
    )
    script = (
        history_script
        + ("" if upcoming else price_chart_script(event.event_id, chosen))
        + ai_summary_script(event.event_id, chosen)
    )

    return HTMLResponse(
        shell(
            f"backtool - {event.event_id}",
            body,
            active="calendar",
            script=script,
            # The chart library is only loaded on pages that actually chart.
            head_scripts=() if upcoming else (LIGHTWEIGHT_CHARTS_CDN,),
        )
    )


@app.get("/api/event/{event_id}/explain", response_class=HTMLResponse)
async def event_explain(
    event_id: str, symbol: str = "BTCUSDT", interval: str = "5m"
) -> HTMLResponse:
    """Have the model explain this event's historical statistics.

    Runs the same study the history section shows, then hands the *computed
    results* to the model -- never candles. The interpretation is rendered by
    the reporting layer, so a summary shown here looks identical to one saved
    into a report.
    """
    event = _find_event(event_id)
    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]

    if not _ai_available():
        return HTMLResponse(
            render_error("ANTHROPIC_API_KEY is not set on the server."),
            status_code=503,
        )

    try:
        html = await asyncio.to_thread(
            _event_explain, event=event, symbol=chosen, interval=interval
        )
    except Exception as exc:  # noqa: BLE001 - the panel must show the reason
        logger.exception("Explanation failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=502)

    return HTMLResponse(html)


def _event_explain(*, event: MarketEvent, symbol: str, interval: str) -> str:
    """Run the study, interpret it, and render the interpretation."""
    from backtool.ai import interpret_study
    from backtool.reporting.html import render_interpretation_card

    now = dt.datetime.now(tz=UTC)
    spec = ResearchSpec(
        symbol=symbol,
        event_type=event.event_type,
        event_count=DEFAULT_HISTORY_EVENTS,
        interval=Interval(interval.lower()),
        windows=DEFAULT_WINDOWS,
        as_of=min(event.timestamp_utc, now),
    )
    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(spec.event_type), now=now)

    question = (
        f"Summarise how {symbol} behaved around the last "
        f"{DEFAULT_HISTORY_EVENTS} {event.event_type.value} releases, for a "
        f"reader looking at the {event.local_date.isoformat()} release."
    )
    return render_interpretation_card(interpret_study(study, question=question))


@app.get("/api/event/{event_id}/candles")
async def event_candles(
    event_id: str, symbol: str = "BTCUSDT", interval: str = "5m"
) -> dict[str, Any]:
    """Candles around one event, in the shape Lightweight Charts expects.

    Served from the same cache the statistics are computed from, so the chart
    and the numbers cannot disagree.
    """
    event = _find_event(event_id)
    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]
    now = dt.datetime.now(tz=UTC)

    if event.timestamp_utc > now:
        return {
            "symbol": chosen,
            "candles": [],
            "message": "This event has not happened yet.",
        }

    try:
        candles = await asyncio.to_thread(
            _event_candles, event=event, symbol=chosen, interval=interval
        )
    except Exception as exc:  # noqa: BLE001 - the chart shows the reason
        logger.exception("Candle fetch failed")
        return {"symbol": chosen, "candles": [], "message": f"{type(exc).__name__}: {exc}"}

    return {
        "symbol": chosen,
        "interval": interval,
        "event_time": int(event.timestamp_utc.timestamp()),
        "candles": candles,
    }


def _event_candles(
    *, event: MarketEvent, symbol: str, interval: str
) -> list[dict[str, float | int]]:
    """Fetch and shape the candles surrounding an event.

    Times are UNIX seconds, which is what Lightweight Charts wants for
    intraday data. Converted via ``timestamp()`` rather than integer epoch
    arithmetic so the result does not depend on the frame's backing
    resolution -- the mistake that cost a debugging session earlier.
    """
    import pandas as pd

    resolved = Interval(interval.lower())
    start = event.timestamp_utc - CHART_SPAN
    end = event.timestamp_utc + CHART_SPAN

    with MarketDataService.from_settings(Settings.from_env()) as service:
        frame = service.get_candles(symbol, resolved, start, end)

    seconds: list[int] = (
        (frame["open_time"] - pd.Timestamp(0, tz="UTC"))
        .dt.total_seconds()
        .astype("int64")
        .tolist()
    )
    return [
        {"time": t, "open": o, "high": h, "low": low, "close": c}
        for t, o, h, low, c in zip(
            seconds,
            frame["open"].astype("float64").tolist(),
            frame["high"].astype("float64").tolist(),
            frame["low"].astype("float64").tolist(),
            frame["close"].astype("float64").tolist(),
            strict=True,
        )
    ]


@app.get("/api/event/{event_id}/history", response_class=HTMLResponse)
async def event_history(
    event_id: str, symbol: str = "BTCUSDT", interval: str = "5m"
) -> HTMLResponse:
    """The study body for one event type, as an HTML fragment."""
    event = _find_event(event_id)
    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]

    try:
        html = await asyncio.to_thread(
            _event_history, event=event, symbol=chosen, interval=interval
        )
    except ValueError as exc:
        return HTMLResponse(render_error(str(exc)), status_code=400)
    except Exception as exc:  # noqa: BLE001 - the page must show something
        logger.exception("Event history failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=500)

    return HTMLResponse(html)


def _find_event(event_id: str) -> MarketEvent:
    for event in load_events():
        if event.event_id == event_id:
            return event
    raise HTTPException(status_code=404, detail=f"No event {event_id!r} in the calendar")


def _event_history(*, event: MarketEvent, symbol: str, interval: str) -> str:
    """Run the standard study for this event's type and render it.

    The cutoff is the event itself for a past occurrence, so its own outcome is
    excluded -- a page about one release must not quietly include that release
    in the history it presents as prior context.
    """
    now = dt.datetime.now(tz=UTC)
    as_of = min(event.timestamp_utc, now)

    spec = ResearchSpec(
        symbol=symbol,
        event_type=event.event_type,
        event_count=DEFAULT_HISTORY_EVENTS,
        interval=Interval(interval.lower()),
        windows=DEFAULT_WINDOWS,
        as_of=as_of,
    )
    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(spec.event_type), now=now)

    return render_body(study)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness plus what the server can currently do."""
    return {
        "status": "ok",
        "ai_available": _ai_available(),
        "max_events": MAX_EVENTS,
        "intervals": [interval.value for interval in Interval],
        "event_types": [event.value for event in EventType],
    }


@app.post("/api/analyse", response_class=HTMLResponse)
async def analyse(
    mode: str = Form("build"),
    question: str = Form(""),
    event_type: str = Form("FOMC"),
    count: int = Form(20),
    interval: str = Form("5m"),
    windows: str = Form(""),
    explain: str = Form(""),
) -> HTMLResponse:
    """Run a study and return the report markup.

    Returns an HTML fragment rather than JSON: the result *is* markup, and the
    page injects it directly. Errors come back as markup too, so the client
    never has to branch on content type.
    """
    try:
        # The engine is synchronous and does blocking network I/O, so it runs
        # in a worker thread. Without this one slow study would stall the whole
        # event loop and block every other request.
        html = await asyncio.to_thread(
            _run,
            mode=mode,
            question=question.strip(),
            event_type=event_type,
            count=count,
            interval=interval,
            windows=windows,
            explain=bool(explain),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        return HTMLResponse(render_error(str(exc)), status_code=400)
    except Exception as exc:  # noqa: BLE001 - the page must show *something*
        logger.exception("Study failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=500)

    return HTMLResponse(html)


def _run(
    *,
    mode: str,
    question: str,
    event_type: str,
    count: int,
    interval: str,
    windows: str,
    explain: bool,
) -> str:
    """Build a spec, execute it, and render the result. Runs off the event loop."""
    settings = Settings.from_env()
    # Pinned so a result is reproducible from its fingerprint. Without it the
    # same spec would select different events tomorrow.
    as_of = dt.datetime.now(tz=UTC)

    plan = None
    if mode == "ask":
        if not question:
            raise ValueError("Enter a question, or switch to Build a study.")
        from backtool.ai import plan_study

        plan, spec = plan_study(question, as_of=as_of)
    else:
        spec = _build_spec(event_type, count, interval, windows, as_of)

    with MarketDataService.from_settings(settings) as service:
        study = run_study(spec, service, load_events(spec.event_type), now=as_of)

    interpretation = None
    if mode == "ask" or explain:
        interpretation = _interpret(study, question or None)

    return _plan_block(plan, spec) + render_body(study, interpretation)


def _interpret(study: StudyResult, question: str | None) -> Any | None:
    """Interpret the results, returning ``None`` if the AI layer fails.

    A failed interpretation must not lose the study: the numbers are the
    product, and they are already computed by this point.
    """
    from backtool.ai import AIError, interpret_study

    try:
        return interpret_study(study, question=question)
    except AIError as exc:
        logger.warning("Interpretation unavailable: %s", exc)
        return None


def _build_spec(
    event_type: str,
    count: int,
    interval: str,
    windows: str,
    as_of: dt.datetime,
) -> ResearchSpec:
    if not 1 <= count <= MAX_EVENTS:
        raise ValueError(f"Event count must be between 1 and {MAX_EVENTS}.")

    parsed = tuple(_parse_window(line) for line in windows.splitlines() if line.strip())

    try:
        return ResearchSpec(
            event_type=EventType(event_type.upper()),
            event_count=count,
            interval=Interval(interval.lower()),
            windows=parsed or DEFAULT_WINDOWS,
            as_of=as_of,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid specification: {exc}") from exc


def _parse_window(line: str) -> WindowSpec:
    """Parse one ``name:start:end`` line from the windows textarea."""
    parts = [part.strip() for part in line.split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid window {line.strip()!r}. Use name:start:end, e.g. post_1h:0h:1h"
        )
    return WindowSpec(name=parts[0], start=parts[1], end=parts[2])


def _plan_block(plan: Any | None, spec: ResearchSpec) -> str:
    """Show the AI's interpretation of the question above the results.

    Rendered before the numbers so a reader can reject a definition rather than
    silently inherit it.
    """
    if plan is None:
        return ""

    from xml.sax.saxutils import escape

    def section(title: str, items: list[str]) -> str:
        if not items:
            return ""
        entries = "".join(f"<li>{escape(item)}</li>" for item in items)
        return f"<p class='ai-label'>{escape(title)}</p><ul>{entries}</ul>"

    windows = "".join(
        f"<li><code>{escape(w.name)}</code> {escape(w.start)} &rarr; {escape(w.end)}"
        f" &mdash; {escape(w.rationale)}</li>"
        for w in plan.windows
    )

    return (
        "<h2>Research plan</h2>"
        '<div class="card ai">'
        '<p class="ai-badge">The question as the planner understood it. '
        "Reject a definition here rather than inherit it below.</p>"
        f'<p class="ai-headline">{escape(plan.restated_question)}</p>'
        f"{section('Ambiguous terms resolved', list(plan.interpretation))}"
        f"{section('Assumed (not specified in the question)', list(plan.assumptions))}"
        f"<p class='ai-label'>Windows</p><ul>{windows}</ul>"
        f"<p class='ai-label'>Spec fingerprint</p>"
        f"<p class='mono'>{escape(spec.fingerprint)}</p>"
        "</div>"
    )


def serve(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(
        "backtool.web.app:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
