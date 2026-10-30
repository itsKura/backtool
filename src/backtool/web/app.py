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

from backtool.ai.thesis_planner import PlannedThesis
from backtool.config import Settings
from backtool.core.time import UTC
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events import load_events
from backtool.events.models import MarketEvent
from backtool.events.resolution import DayAnchor, ResolvedEventConcept
from backtool.reporting.html import render_body
from backtool.research.results import StudyResult
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.thesis import (
    MAX_REMEMBERED_EXCHANGES,
    Exchange,
    Thesis,
)
from backtool.research.windows import DEFAULT_WINDOWS, WindowSpec, window_title
from backtool.web.calendar_page import render_calendar
from backtool.web.event_page import (
    CANDLE_CHART_JS,
    CHART_BLOCK_JS,
    CHART_SPAN,
    LIGHTWEIGHT_CHARTS_CDN,
    ai_summary_script,
    chart_span_for,
    chart_window_bands,
    history_load_script,
    price_chart_script,
    render_ai_summary,
    render_asset_selector,
    render_event_explainer,
    render_event_header,
    render_expectations,
    render_price_chart,
    render_scenarios_slot,
    render_similar_setups,
    render_upcoming_notice,
    scenarios_load_script,
    setups_load_script,
    thin_for_display,
    timeframe_by_label,
)
from backtool.web.history_panel import HISTORY_SCRIPT, render_history
from backtool.web.page import render_error, render_page, shell
from backtool.web.prepare_page import (
    build_preparation,
    comparable_releases,
    prepare_chart_script,
    render_preparation,
)
from backtool.web.research_page import (
    RESEARCH_SCRIPT,
    render_clarification,
    render_finding,
    render_research_page,
    render_spec_card,
    render_unresolved_event,
    render_unsupported,
)
from backtool.web.scenario_panel import (
    SCENARIO_SCRIPT,
    build_scenarios,
    render_scenarios_section,
)
from backtool.web.setups_panel import build_setup_match, render_setups_section

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


def _show_our_own_logs() -> None:
    """Make this package's INFO records reach the console.

    Uvicorn configures handlers for its own loggers and leaves the root
    untouched, so without this every `logger.info` in `backtool` goes nowhere.
    That was not a cosmetic gap: the per-request cost accounting is logged, so
    the figures existed and nothing could see them -- which is how a number
    nobody can look up gets recalled wrongly instead.

    Done here rather than in a library module because this is the application
    entry point; importing a library should not reconfigure anybody's logging.
    Idempotent, so an autoreloading server does not stack handlers.
    """
    package = logging.getLogger("backtool")
    package.setLevel(logging.INFO)
    if any(getattr(h, "name", "") == "backtool-console" for h in package.handlers):
        return
    handler = logging.StreamHandler()
    handler.name = "backtool-console"
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    package.addHandler(handler)


_show_our_own_logs()


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
    upcoming = event.timestamp_utc > now

    # The order is the reader's order: what is this, what is expected, what has
    # happened before, what stands out -- then, for anyone still reading, the
    # individual observations and the method. The three history slots are filled
    # asynchronously because a cold cache means fetching candles for twenty
    # releases, and the explanation should not wait on that.
    body = (
        render_event_header(event, now=now)
        + (render_upcoming_notice(event) if upcoming else "")
        + render_expectations(event, now=now)
        + render_asset_selector(event, SYMBOLS, chosen)
        + render_price_chart(event, chosen, available=not upcoming)
        + '<section class="reaction" id="reaction-summary">'
        + '<div id="slot-summary"><p class="status"><span class="spinner"></span>'
        + "Measuring the last "
        + str(DEFAULT_HISTORY_EVENTS)
        + " releases&hellip;</p></div></section>"
        + render_ai_summary(available=_ai_available(), has_history=True)
        + render_scenarios_slot()
        + render_similar_setups()
        + '<div id="slot-evidence"></div>'
        + render_event_explainer(event)
        + '<div id="slot-methodology"></div>'
    )

    script = (
        history_load_script(event.event_id, chosen)
        + setups_load_script(event.event_id, chosen)
        + scenarios_load_script(event.event_id, chosen)
        + SCENARIO_SCRIPT
        + HISTORY_SCRIPT
        + (
            ""
            if upcoming
            # The chart body, then the four lines that mount it.
            else CANDLE_CHART_JS + price_chart_script(event.event_id, chosen)
        )
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


@app.get("/event/{event_id}/prepare", response_class=HTMLResponse)
async def event_prepare(event_id: str, symbol: str = "BTCUSDT") -> HTMLResponse:
    """Everything a reader needs before one CPI release.

    Rendered in full rather than filled in asynchronously, the way the event
    page is. The difference is what the work is: the event page loads three
    independent panels that each need their own fetch, where this is a single
    study viewed five ways, and splitting one study across four requests would
    run it four times to make the first paint arrive sooner.

    Costs nothing to open. CPI is a curated calendar with a hand-verified
    values file, so no planner, resolver or interpreter is involved.
    """
    event = _find_event(event_id)
    if event.curated_type is not EventType.CPI:
        # Deliberately narrow. The page reads a CPI values file and a CPI
        # scenario rule, and rendering it for an event type that has neither
        # would produce a convincing-looking page with nothing behind it.
        raise HTTPException(
            status_code=404,
            detail="Event preparation is available for CPI releases.",
        )

    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]
    now = dt.datetime.now(tz=UTC)
    prep = await asyncio.to_thread(
        build_preparation, event, symbol=chosen, now=now
    )

    body = render_preparation(prep, SYMBOLS)
    # The page draws a chart only when there are comparable releases to draw,
    # and the library is 200 kB: asked for by the same list the markup is built
    # from, so the two cannot disagree about whether there is a chart.
    charted = bool(comparable_releases(prep))
    return HTMLResponse(
        shell(
            f"backtool - {event.event_id} preparation",
            body,
            active="calendar",
            script=(
                CANDLE_CHART_JS + CHART_BLOCK_JS + prepare_chart_script(chosen)
                if charted
                else ""
            ),
            head_scripts=(LIGHTWEIGHT_CHARTS_CDN,) if charted else (),
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
    from backtool.web.event_page import render_key_findings

    now = dt.datetime.now(tz=UTC)
    spec = ResearchSpec(
        symbol=symbol,
        event_type=event.curated_type,
        event_count=DEFAULT_HISTORY_EVENTS,
        interval=Interval(interval.lower()),
        windows=DEFAULT_WINDOWS,
        as_of=min(event.timestamp_utc, now),
    )
    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(spec.event_type), now=now)

    question = (
        f"Summarise how {symbol} behaved around the last "
        f"{DEFAULT_HISTORY_EVENTS} {event.curated_type.value} releases, for a "
        f"reader looking at the {event.local_date.isoformat()} release."
    )
    return render_key_findings(interpret_study(study, question=question))


@app.get("/api/event/{event_id}/candles")
async def event_candles(
    event_id: str, symbol: str = "BTCUSDT", timeframe: str = "5m"
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
        candles, bands, grain = await asyncio.to_thread(
            _event_candles, event=event, symbol=chosen, timeframe=timeframe
        )
    except Exception as exc:  # noqa: BLE001 - the chart shows the reason
        logger.exception("Candle fetch failed")
        return {"symbol": chosen, "candles": [], "message": f"{type(exc).__name__}: {exc}"}

    picked = timeframe_by_label(timeframe)
    return {
        "symbol": chosen,
        "interval": picked.fetch.value,
        "timeframe": picked.label,
        # What the candles on screen actually are. For 4h that is coarser than
        # the hourly series behind it, and saying so matters: a reader counting
        # hours off aggregated candles would be reading them wrong with nothing
        # on the page to correct them.
        "drawn_interval": grain or picked.fetch.value,
        "event_time": int(event.timestamp_utc.timestamp()),
        "candles": candles,
        # Derived from the same windows the statistics are measured over, and
        # priced by the same function, so a shaded box on the chart cannot
        # describe a different span -- or a different high -- than the research
        # layer would report for it.
        "windows": bands,
    }


def _event_candles(
    *, event: MarketEvent, symbol: str, timeframe: str
) -> tuple[list[dict[str, float | int]], list[dict[str, Any]], str | None]:
    """Fetch and shape the candles surrounding an event, plus the window boxes.

    Both come from one fetch. Pricing the bands from a second, separately
    fetched frame would let the boxes and the candles under them drift apart if
    the cache changed between calls.

    Times are UNIX seconds, which is what Lightweight Charts wants for
    intraday data. Converted via ``timestamp()`` rather than integer epoch
    arithmetic so the result does not depend on the frame's backing
    resolution -- the mistake that cost a debugging session earlier.
    """
    chosen = timeframe_by_label(timeframe)
    resolved = chosen.fetch
    # The widest window this page shades decides the minimum reach, so the
    # day-long band stays whole at every candle size.
    span = max(chosen.reach, CHART_SPAN)
    start = event.timestamp_utc - span
    end = event.timestamp_utc + span

    with MarketDataService.from_settings(Settings.from_env()) as service:
        frame = service.get_candles(symbol, resolved, start, end)

    # Priced from the frame as fetched, before it is thinned for drawing, so
    # a 4h chart still reports the high of the hourly series.
    bands = chart_window_bands(event.timestamp_utc, frame, resolved, span=span)
    drawn, grain = thin_for_display(frame, resolved, force=chosen.grain)
    return _shaped_candles(drawn), bands, grain


@app.get("/api/event/{event_id}/setups", response_class=HTMLResponse)
async def event_setups(
    event_id: str, symbol: str = "BTCUSDT", interval: str = "5m"
) -> HTMLResponse:
    """Past releases that arrived under today's market conditions.

    "Today" is the current instant for an upcoming release and the release's own
    instant for a past one, so a historical page shows what the market looked
    like going into *that* release rather than what it looks like now.
    """
    event = _find_event(event_id)
    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]

    try:
        html = await asyncio.to_thread(
            _event_setups, event=event, symbol=chosen, interval=interval
        )
    except Exception as exc:  # noqa: BLE001 - the panel must show the reason
        logger.exception("Setup matching failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=500)

    return HTMLResponse(html)


def _event_setups(*, event: MarketEvent, symbol: str, interval: str) -> str:
    """Build and render the setup match. Runs off the event loop."""
    now = dt.datetime.now(tz=UTC)
    as_of = min(event.timestamp_utc, now)

    match, measurable = build_setup_match(
        event_type=event.curated_type,
        symbol=symbol,
        as_of=as_of,
        interval=Interval(interval.lower()),
    )
    return render_setups_section(
        match,
        measurable=measurable,
        symbol=symbol,
        event_time=event.timestamp_utc,
    )


@app.get("/api/event/{event_id}/scenarios", response_class=HTMLResponse)
async def event_scenarios(
    event_id: str, symbol: str = "BTCUSDT", interval: str = "5m"
) -> HTMLResponse:
    """Past releases of this type, split by what the release itself said."""
    event = _find_event(event_id)
    chosen = symbol.upper() if symbol.upper() in SYMBOLS else SYMBOLS[0]

    try:
        html = await asyncio.to_thread(
            _event_scenarios, event=event, symbol=chosen, interval=interval
        )
    except Exception as exc:  # noqa: BLE001 - the panel must show the reason
        logger.exception("Scenario split failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=500)

    return HTMLResponse(html)


def _event_scenarios(*, event: MarketEvent, symbol: str, interval: str) -> str:
    """Build and render the scenario split. Runs off the event loop."""
    now = dt.datetime.now(tz=UTC)
    split = build_scenarios(
        event_type=event.curated_type,
        symbol=symbol,
        as_of=min(event.timestamp_utc, now),
        interval=Interval(interval.lower()),
    )
    return render_scenarios_section(
        split, event_type=event.curated_type, symbol=symbol
    )


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
        event_type=event.curated_type,
        event_count=DEFAULT_HISTORY_EVENTS,
        interval=Interval(interval.lower()),
        windows=DEFAULT_WINDOWS,
        as_of=as_of,
    )
    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(spec.event_type), now=now)

    return render_history(study)


@app.get("/research", response_class=HTMLResponse)
async def research(q: str = "") -> HTMLResponse:
    """The conversational research workspace.

    The emerging primary surface. The calendar and event pages remain as
    discovery and deep-dive tools; this is where a question gets asked.

    ``q`` arrives from a preparation page handing a question over. It fills the
    box and nothing more -- the reader still presses the button, so following
    a suggestion costs nothing until they decide it is worth it.
    """
    return HTMLResponse(
        shell(
            "backtool - research",
            render_research_page(q),
            active="research",
            # The chart body first, then the page script that mounts it on
            # whatever answers the thread gains.
            script=CANDLE_CHART_JS + CHART_BLOCK_JS + RESEARCH_SCRIPT,
            head_scripts=(LIGHTWEIGHT_CHARTS_CDN,),
        )
    )


@app.post("/api/research/interpret", response_class=HTMLResponse)
async def research_interpret(payload: dict[str, Any]) -> HTMLResponse:
    """Read a question into a thesis, or say why it cannot be read.

    Returns the specification for confirmation rather than running it. The
    reader has to be able to see what was understood before trusting what
    comes back.
    """
    question = str(payload.get("question", "")).strip()
    if not question:
        return HTMLResponse(render_error("Enter a question first."), status_code=400)

    if not _ai_available():
        return HTMLResponse(
            render_error(
                "ANTHROPIC_API_KEY is not set, so plain-language questions "
                "cannot be interpreted. The calendar and event pages work "
                "without it."
            ),
            status_code=503,
        )

    previous = _previous_thesis(payload.get("thesis"))
    conversation = _conversation(payload.get("conversation"))

    try:
        html = await asyncio.to_thread(
            _interpret_question, question, previous, conversation
        )
    except Exception as exc:  # noqa: BLE001 - the thread must show the reason
        logger.exception("Thesis planning failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=502)

    return HTMLResponse(html)


def _interpret_question(
    question: str,
    previous: Thesis | None,
    conversation: tuple[Exchange, ...] = (),
) -> str:
    """Plan, look up any public event, validate, and render. Off the event loop.

    The lookup sits between planning and conversion because of what each step
    can know. The planner can tell that a question is about the Ethereum Merge;
    it cannot tell when the Merge happened, and must not guess. The engine needs
    dated events and nothing else. So the concept is established from published
    sources in between, and the reader sees those sources on the same card as
    the specification -- for a looked-up event the dates *are* the research, and
    confirming a specification without them would be confirming the uninteresting
    half.
    """
    from backtool.ai.thesis_planner import plan_thesis

    planned = plan_thesis(question, previous=previous, conversation=conversation)

    # Every card below that ends in a question is handed the question that
    # prompted it, so it can record the exchange. Those are exactly the cards
    # with no thesis to carry, which is why a reader could answer the same
    # question four times and get asked it again each time.
    if planned.unsupported:
        return render_unsupported(planned, question=question)
    if planned.clarifications:
        return render_clarification(planned, question=question)

    resolution: ResolvedEventConcept | None = None
    if planned.needs_resolution:
        resolution = _resolution_for(planned.concept, previous)
        if not resolution.is_runnable:
            # Not an error. Several different outcomes land here -- a name that
            # matches more than one event, sources that disagree, nothing found
            # -- and each is something to tell the reader rather than something
            # to resolve by picking.
            return render_unresolved_event(resolution, question=question)

    # Conversion is where the vocabulary is enforced. A field or value the
    # planner invented fails here rather than reaching the engine. The anchor
    # comes from the question if it named one, and is absent otherwise -- this
    # layer never supplies a default.
    try:
        thesis = planned.to_thesis(
            question, resolution=resolution, anchor=_anchor_from(planned, previous)
        )
    except ValueError as exc:
        return render_error(f"That reading does not fit what Backtool measures: {exc}")

    return render_spec_card(thesis, planned)


def _anchor_from(planned: PlannedThesis, previous: Thesis | None) -> DayAnchor | None:
    """The instant the reader chose, from this question or the last one.

    Carried forward so a choice survives a follow-up: a reader who said "from
    the close" once should not have to restate it every time anything else
    about the question changes. The new answer wins when there is one, so they
    can change their mind.

    ``None`` when neither question named an instant, which is the ordinary
    case. The default is applied in
    :func:`~backtool.ai.thesis_planner.default_anchor`, at the single point a
    thesis is built from a resolution -- not here, so that a reader's silence
    means the same thing on every path into the engine.
    """
    if planned.anchor is not None:
        return planned.anchor
    return previous.anchor if previous is not None else None


def _resolution_for(
    concept: str, previous: Thesis | None
) -> ResolvedEventConcept:
    """Establish the concept's dates, or reuse what is already on screen.

    A lookup is a paid web search, and a follow-up about the same event is the
    common case -- "and what about ETH?" must not buy the same dates twice. The
    comparison is on the concept as asked, so a genuinely different event still
    gets its own lookup.
    """
    if previous is not None and previous.resolution is not None:
        already = previous.resolution
        if already.query.strip().lower() == concept.strip().lower():
            logger.info("Reusing the resolution already on screen for %r", concept)
            return already

    from backtool.ai.event_resolver import resolve_event_concept

    logger.info("Resolving %r for a research question", concept)
    return resolve_event_concept(concept)


def _conversation(raw: object) -> tuple[Exchange, ...]:
    """The turns the page sent, validated and trimmed.

    An unreadable entry is dropped rather than fatal, for the same reason a
    stale thesis is: the worst case is one turn planned without its context,
    which the reader can see and correct, and that is a far better failure
    than a page that cannot ask anything at all.

    Trimmed here as well as in the browser. The browser's limit keeps the
    request small and this one is the one that holds, because what reaches the
    planner should not depend on what a page chose to send.
    """
    if not isinstance(raw, list):
        return ()
    turns: list[Exchange] = []
    for item in raw:
        try:
            turns.append(Exchange.model_validate(item))
        except ValueError:
            logger.info("Ignoring an unreadable entry in the conversation")
    return tuple(turns[-MAX_REMEMBERED_EXCHANGES:])


def _previous_thesis(raw: object) -> Thesis | None:
    """The thesis currently on screen, if the page sent one.

    A malformed or stale payload is treated as absent rather than fatal: the
    worst case is that a follow-up is read as a fresh question, which the
    reader can see and correct from the specification card.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return Thesis.model_validate_json(raw)
    except ValueError:
        logger.info("Ignoring unreadable thesis payload")
        return None


@app.post("/api/research/chart")
async def research_chart(payload: dict[str, Any]) -> dict[str, Any]:
    """Candles around one event of a thesis, and the window it measures.

    A POST because the event is not in any calendar. A resolved concept exists
    only inside the thesis that carries it -- nothing here stores one -- so the
    thesis is what identifies the event, and it is far too large for a query
    string.

    Everything comes from the same places the answer did: the calendar from
    :func:`thesis_calendar`, the candles from the cache the statistics were
    computed over, and the shaded window from the horizon the question asked
    for. A chart that fetched its own data would be a second account of the
    same run, and the one a reader looks at would be the one nobody checked.
    """
    thesis = _previous_thesis(payload.get("thesis"))
    if thesis is None:
        return {"candles": [], "message": "That specification could not be read."}

    try:
        data = await asyncio.to_thread(
            _thesis_candles,
            thesis=thesis,
            event_id=str(payload.get("event_id") or ""),
            timeframe=str(payload.get("timeframe") or ""),
        )
    except ValueError as exc:
        return {"candles": [], "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the chart shows the reason
        logger.exception("Research chart failed")
        return {"candles": [], "message": f"{type(exc).__name__}: {exc}"}
    return data


def _measured_window(thesis: Thesis) -> WindowSpec | None:
    """The window the answer's headline number comes from.

    Looked up by the horizon the question named, through the same mapping the
    engine uses, so the box cannot describe a different span than the figure
    beside it. ``None`` when the horizon has no window, which is a question
    this chart has nothing to shade rather than an error.
    """
    from backtool.research.vocabulary import reaction_windows, window_for

    horizon = thesis.outcome.horizon
    name = window_for(horizon)
    return next((w for w in reaction_windows(horizon) if w.name == name), None)


def _thesis_candles(
    *, thesis: Thesis, event_id: str, timeframe: str
) -> dict[str, Any]:
    """Fetch the candles around one of a thesis's events, and shade its window.

    Raises:
        ValueError: when the thesis has no measurable event, or the one named
            is not among them.
    """
    from backtool.research.experiment import thesis_calendar

    calendar, _identity = thesis_calendar(thesis)
    now = dt.datetime.now(tz=UTC)
    past = [event for event in calendar if event.timestamp_utc <= now]
    if not past:
        raise ValueError("None of these events has happened yet, so there are no candles.")

    if event_id:
        chosen = next((e for e in past if e.event_id == event_id), None)
        if chosen is None:
            raise ValueError(f"{event_id} is not one of the events this question measures.")
    else:
        chosen = max(past, key=lambda event: event.timestamp_utc)

    # The window decides which timeframes exist at all, so a blank or stale
    # label lands on this window's default rather than on a global one.
    window_for_timeframe = _measured_window(thesis)
    picked = (
        timeframe_by_label(timeframe, window_for_timeframe)
        if window_for_timeframe is not None
        else timeframe_by_label(timeframe)
    )
    resolved = picked.fetch
    # The chart reaches as far as the window it shades, not a fixed day: a
    # ninety-day box on a chart spanning a day either side is a rectangle wider
    # than the canvas with none of the window inside it.
    window = _measured_window(thesis)
    span = chart_span_for(window, picked) if window is not None else max(
        picked.reach, CHART_SPAN
    )
    start = chosen.timestamp_utc - span
    end = chosen.timestamp_utc + span

    with MarketDataService.from_settings(Settings.from_env()) as service:
        frame = service.get_candles(thesis.asset, resolved, start, end)

    # Priced from the frame as fetched. A two-year window is drawn in daily
    # candles, and a high taken off those would be the high of whichever day
    # straddled the boundary rather than the high of the window.
    bands = (
        chart_window_bands(
            chosen.timestamp_utc, frame, resolved, windows=(window,), span=span
        )
        if window is not None
        else []
    )
    drawn, grain = thin_for_display(frame, resolved, force=picked.grain)

    return {
        "symbol": thesis.asset,
        "interval": resolved.value,
        "timeframe": picked.label,
        "drawn_interval": grain or resolved.value,
        "event_id": chosen.event_id,
        "event_label": chosen.notes or chosen.event_id,
        "event_time": int(chosen.timestamp_utc.timestamp()),
        "candles": _shaped_candles(drawn),
        "windows": bands,
    }


def _shaped_candles(frame: Any) -> list[dict[str, float | int]]:
    """A candle frame in the shape Lightweight Charts expects.

    Times are UNIX seconds via ``timestamp()`` rather than integer epoch
    arithmetic, so the result does not depend on the frame's backing
    resolution -- the mistake that cost a debugging session earlier.
    """
    import pandas as pd

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


@app.post("/api/research/run", response_class=HTMLResponse)
async def research_run(payload: dict[str, Any]) -> HTMLResponse:
    """Measure a confirmed thesis against real data."""
    thesis = _previous_thesis(payload.get("thesis"))
    if thesis is None:
        return HTMLResponse(
            render_error("That specification could not be read. Ask again."),
            status_code=400,
        )

    try:
        html = await asyncio.to_thread(_run_thesis, thesis)
    except ValueError as exc:
        return HTMLResponse(render_error(str(exc)), status_code=400)
    except Exception as exc:  # noqa: BLE001 - the thread must show the reason
        logger.exception("Research run failed")
        return HTMLResponse(render_error(f"{type(exc).__name__}: {exc}"), status_code=500)

    return HTMLResponse(html)


def _run_thesis(thesis: Thesis) -> str:
    """Execute, explain, and render. Runs off the event loop.

    The deterministic summary is built first and is what renders unless the AI
    reading both succeeds and survives its checks. That ordering is the point:
    the explanation is an improvement on the prose, never a component of the
    report, so no API key, no budget, no network and no bad paragraph can cost
    the reader the answer.
    """
    from backtool.research.experiment import run_thesis
    from backtool.research.interpretation import (
        build_facts,
        deterministic_summary,
        interpretation_payload,
    )

    finding = run_thesis(thesis)
    facts = build_facts(finding)
    paragraphs = deterministic_summary(facts)

    if _ai_available():
        try:
            from backtool.ai.research_interpreter import explain_research

            explained = explain_research(interpretation_payload(facts))
            if explained:
                paragraphs = explained
        except Exception:  # noqa: BLE001 - the report is already complete
            # Deliberately silent in the UI. A reader who asked a research
            # question does not need to hear about an API; they need the
            # result, and they have it.
            logger.warning("Falling back to the deterministic summary", exc_info=True)

    return render_finding(finding, paragraphs)


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

    # The planner's window names are free text it invented, so they are shown as
    # the duration they actually mean. The offsets stay beside them: this block
    # exists so a reader can reject a definition, and "-90m to 0h" is the
    # definition in a way that any phrasing of it is not.
    # Paired with the spec's own windows rather than re-parsing the plan's: the
    # spec is what actually ran, and ``to_spec`` builds it from these in order.
    windows = "".join(
        f"<li>{escape(window_title(window))} "
        f"<span class='mono'>({escape(window.start)} &rarr; {escape(window.end)})"
        f"</span> &mdash; {escape(planned.rationale)}</li>"
        for window, planned in zip(spec.windows, plan.windows, strict=True)
    )

    return (
        "<h2>Research plan</h2>"
        '<div class="card ai">'
        '<p class="ai-badge">The question as the planner understood it. '
        "Reject a definition here rather than inherit it below.</p>"
        f'<p class="ai-headline">{escape(plan.restated_question)}</p>'
        f"{section('Ambiguous terms resolved', list(plan.interpretation))}"
        f"{section('Assumed (not specified in the question)', list(plan.assumptions))}"
        f"<p class='ai-label'>Windows measured</p><ul>{windows}</ul>"
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
