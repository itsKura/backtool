"""The CPI preparation page: one release, every way it could land.

The calendar says a release is coming. The event page says what the release is
and how it has behaved. Neither answers the question somebody actually has at
08:00 on a print day, which is *what does the evidence say about each of the
ways this can land, and how does today compare with the days those landed on.*

This page is that question, laid out in the order it gets asked: what is
coming, what to watch, what each outcome has historically meant, what the
market looks like going in, and which past releases most resemble today.

Assembly only
-------------
Every figure comes from :mod:`backtool.research`: the split is
:func:`~backtool.research.cohort.split_study`, the statistics are
:func:`~backtool.research.aggregate.aggregate_window`, and the run-up to now is
:func:`~backtool.research.metrics.compute_window_metrics`. This module fetches,
arranges and writes HTML. It computes one thing -- the difference between two
medians -- and does that by subtracting two numbers the research layer already
produced.

Free to open
------------
No planner, no resolver, no interpreter. CPI is a curated calendar with a
hand-verified values file, and resolving "CPI" against the web to render a page
about a release whose date was published a year ago would be paying a model to
tell us something already in the repository. The prose at the top is a template
over computed facts for the same reason: a sentence that costs four cents is a
sentence a reader refreshes once and then stops reading.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from urllib.parse import quote
from xml.sax.saxutils import escape

from backtool.config import Settings
from backtool.core.time import NEW_YORK, utc_to_local
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events import load_events
from backtool.events.meta import meta_for
from backtool.events.models import MarketEvent
from backtool.events.releases import (
    CPI_METRIC,
    CPI_METRIC_SHORT,
    SCENARIO_ORDER,
    SCENARIO_RULES,
    PublishedValue,
    ReleaseValues,
    load_releases,
    release_figures,
)
from backtool.research.cohort import Cohort, split_study
from backtool.research.metrics import WindowMetrics
from backtool.research.preparation import (
    CONTEXT_LABELS,
    CONTEXT_WINDOWS,
    ENTRY_WINDOW,
    MarketContext,
    baseline_cohort,
    build_market_context,
    cohort_for_entry,
    entry_cohorts,
)
from backtool.research.results import EventResult, StudyResult
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.calendar_page import SEPARATOR, format_delta
from backtool.web.cohort_view import render_cohort_table
from backtool.web.timefmt import local_stamp

#: How many past releases the page considers. The whole supplied dataset fits
#: well inside this; releases without candles are reported, never dropped.
PREPARE_CANDIDATES = 40

#: The horizons shown on a scenario card, as opposed to the full grid below it.
#:
#: Three, because a card is a summary and the five-row table is two clicks away.
#: These are the three a reader names out loud -- the immediate reaction, the
#: rest of the session, and the full day.
CARD_WINDOWS: tuple[str, ...] = ("post_event_1h", "post_event_4h", "post_event_24h")

#: The horizon the headline comparisons are quoted at.
HEADLINE_WINDOW = "post_event_24h"

#: Short labels for the reaction grid, for axes and card rows.
WINDOW_LABELS: dict[str, str] = {
    "post_event_5m": "5m",
    "post_event_15m": "15m",
    "post_event_1h": "1h",
    "post_event_4h": "4h",
    "post_event_24h": "24h",
}

#: Display names for the symbols this page offers, matching the event page.
SYMBOL_NAMES: dict[str, str] = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum"}

#: Which line is which on the horizon chart.
#:
#: Keyed off the cohort rather than the order it was drawn in, because hot and
#: cold is the dimension being split on and a palette running against it is one
#: a reader has to fight. Hotter is warm, cooler is cool, in line is neutral --
#: and it stays that way if SCENARIO_ORDER is ever rearranged.
SERIES_CLASS: dict[str, str] = {
    "hotter": "prep-series-hotter",
    "in_line": "prep-series-inline",
    "cooler": "prep-series-cooler",
}

#: What each scenario actually tested, stated as the comparison it is.
#:
#: Spelled out beside every group because "cooler" on its own is a mood. The
#: rule is a comparison a reader can redo against the two figures in the same
#: table, which is the whole reason the classification is arithmetic and not a
#: score. The thresholds stay abstract while no consensus is published for the
#: upcoming release -- naming one would be inventing it.
SCENARIO_RULE_TEXT: dict[str, str] = {
    "hotter": "actual above consensus",
    "in_line": "actual equals consensus",
    "cooler": "actual below consensus",
}

#: Reader-facing names for the scenario keys, so no raw enum reaches the page.
SCENARIO_LABELS: dict[str, str] = {
    "cooler": "Cooler than expected",
    "in_line": "In line with expectations",
    "hotter": "Hotter than expected",
}


@dataclass(frozen=True)
class Preparation:
    """Everything the page shows, computed before any of it is written.

    Assembled in one pass so the page cannot render a header from one study and
    a table from another, and so the expensive part -- one run over forty
    releases -- happens exactly once per view.
    """

    event: MarketEvent
    symbol: str
    now: dt.datetime
    released: bool
    values: ReleaseValues | None
    study: StudyResult | None
    scenarios: list[Cohort]
    baseline: Cohort | None
    context: MarketContext
    entry_match: Cohort | None
    all_values: dict[str, ReleaseValues]
    reaction: dict[str, object] | None = None

    @property
    def entry_return_pct(self) -> float | None:
        """How the asset went into *this* release, over the 24 hours before it.

        Before the print, the run-up to now: the window has not closed, and the
        hours so far are the only estimate of it there is.

        After the print, this release's own ``pre_event_24h`` -- the same window
        every historical member of the comparable cohort was selected by. Going
        on quoting the run-up to now would describe a day that mostly happened
        *after* the event while calling it the day before.
        """
        if self.released and self.study is not None:
            own = next(
                (
                    item
                    for item in self.study.events
                    if item.event_id == self.event.event_id
                ),
                None,
            )
            if own is not None:
                measured = own.windows.get(ENTRY_WINDOW.name)
                if measured is not None and measured.is_usable:
                    return measured.return_pct
            return None
        return self.context.entry_return_pct

    @property
    def entry_direction(self) -> str | None:
        value = self.entry_return_pct
        if value is None:
            return None
        return "down" if value < 0 else "up"

    @property
    def scenario_key(self) -> str | None:
        """Which scenario this release actually landed in, once it has.

        ``None`` before the print, and also after it when the release has no
        verified consensus -- an actual with nothing to compare it against has
        not landed in a scenario, it has merely happened.
        """
        if not self.released or self.values is None:
            return None
        rule = SCENARIO_RULES[EventType.CPI]
        classified = rule(self.values)
        return classified[0] if classified else None


def build_preparation(
    event: MarketEvent,
    *,
    symbol: str,
    now: dt.datetime,
) -> Preparation:
    """Run the one study this page needs and arrange everything around it.

    One study, five views of it: the scenario split, the baseline, the entry
    split, the per-release table and -- after the release -- that release's own
    reaction. Running it once is the whole of the performance story here; a
    section-per-study page would fetch the same candles four times.
    """
    values = load_releases()
    rule = SCENARIO_RULES[EventType.CPI]

    spec = ResearchSpec(
        symbol=symbol,
        event_type=EventType.CPI,
        event_count=PREPARE_CANDIDATES,
        interval=Interval.M5,
        # The reaction grid is what the page reports; the entry window is what
        # selects which releases it reports over.
        windows=(*REACTION_WINDOWS, ENTRY_WINDOW),
        as_of=now,
    )

    released = event.timestamp_utc <= now
    context_from = now - dt.timedelta(hours=30)

    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(EventType.CPI), now=now)
        frame = service.get_candles(symbol, Interval.M5, context_from, now)

    context = build_market_context(
        frame, symbol=symbol, now=now, interval=Interval.M5
    )

    scenarios = split_study(study, lambda item: rule(values.get(item.event_id)))
    rank = {key: index for index, key in enumerate(SCENARIO_ORDER)}
    scenarios.sort(key=lambda cohort: rank.get(cohort.key, len(rank)))

    prep = Preparation(
        event=event,
        symbol=symbol,
        now=now,
        released=released,
        values=values.get(event.event_id),
        study=study,
        scenarios=scenarios,
        baseline=baseline_cohort(study),
        context=context,
        entry_match=None,
        all_values=values,
        reaction=_reaction_for(study, event, now) if released else None,
    )
    # Selected after the rest is assembled, because which 24 hours count
    # depends on whether the release has happened.
    return replace(prep, entry_match=_match_for(study, prep))


def _match_for(study: StudyResult, prep: Preparation) -> Cohort | None:
    """The cohort this release belongs to, by how the asset went into it.

    Delegates to :func:`cohort_for_entry` before the print, where the run-up to
    now is the reading. Afterwards the release has its own measured entry, so
    the cohort is chosen by that instead -- and the page's own release is
    excluded from the group it is being compared against, because a set that
    contains the thing it is explaining is not a comparison.
    """
    direction = prep.entry_direction
    if direction is None:
        return None
    if not prep.released:
        return cohort_for_entry(study, prep.context)

    wanted = "entered_down" if direction == "down" else "entered_up"
    others = study.model_copy(
        update={
            "events": tuple(
                item for item in study.events if item.event_id != prep.event.event_id
            )
        }
    )
    return next(
        (cohort for cohort in entry_cohorts(others) if cohort.key == wanted), None
    )


def _reaction_for(
    study: StudyResult, event: MarketEvent, now: dt.datetime
) -> dict[str, object] | None:
    """This release's own developing reaction, horizon by horizon.

    Only the windows that have actually closed carry a figure. A 24-hour window
    on a release four hours old is not zero and not flat; it is a window that
    has not finished, and the research layer already refuses to price one. That
    refusal is passed through rather than smoothed over -- on the morning of a
    release it is the single most misleading number the page could print.
    """
    result = next(
        (item for item in study.events if item.event_id == event.event_id), None
    )
    if result is None:
        return None
    rows: list[dict[str, object]] = []
    for name in WINDOW_LABELS:
        window = result.windows.get(name)
        if window is None:
            continue
        rows.append(
            {
                "label": WINDOW_LABELS[name],
                "return_pct": window.return_pct if window.is_usable else None,
                "reason": None if window.is_usable else _why_missing(window, now),
            }
        )
    return {"rows": rows}


def _why_missing(window: WindowMetrics, now: dt.datetime) -> str:
    """Why a horizon carries no figure, decided by the clock.

    An earlier version read the engine's reason string looking for words like
    "future", which is parsing a sentence written for a person -- and it missed
    the commonest case on this page. The window knows when it ends, so whether
    it is still running is a comparison rather than a guess, and a reader gets
    told how long they are waiting instead of that something is unavailable.
    """
    if window.end_utc > now:
        return f"closes {format_delta(window.end_utc - now, True)}"
    return "incomplete candle data"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_preparation(prep: Preparation, symbols: tuple[str, ...]) -> str:
    """The page body, as a fragment.

    Ordered the way the question gets asked twenty minutes before a print:
    what is coming, what the market is doing right now, what each outcome has
    meant, the evidence behind that, and only then the method. An earlier
    version put the historical sections first and the current one below a
    chart, which is the order a report is written in rather than the order it
    is read in.
    """
    return (
        _header(prep)
        + _asset_row(prep, symbols)
        + render_reaction(prep)
        + _what_to_watch(prep)
        + _context_section(prep)
        + _scenario_section(prep)
        + _horizon_chart(prep)
        + _occurrences(prep)
        + _methodology(prep)
        + _followup(prep)
    )


def _header(prep: Preparation) -> str:
    """What is coming, which figure it is, when, and the three published values."""
    meta = meta_for(EventType.CPI)
    local = utc_to_local(prep.event.timestamp_utc, NEW_YORK)
    delta = prep.event.timestamp_utc - prep.now
    state = (
        '<span class="prep-state prep-state-released">Released</span>'
        if prep.released
        else '<span class="prep-state prep-state-upcoming">Upcoming</span>'
    )
    countdown = (
        ""
        if prep.released
        else f'<span class="prep-countdown">{escape(format_delta(delta, True))}</span>'
    )

    # The release instant on the reader's own clock. Dropped by the browser for
    # anyone already in the release's timezone, where it would print the same
    # time twice -- which on a US release page is a good share of readers.
    in_your_time = local_stamp(
        prep.event.timestamp_utc, lead=SEPARATOR, instead_of=f"{local:%H:%M}"
    )

    figures = "".join(
        _figure_cell(figure) for figure in release_figures(
            prep.values, released=prep.released
        )
    )

    landed = ""
    key = prep.scenario_key
    if key is not None:
        surprise = prep.values.surprise_pp if prep.values else None
        gap = (
            f" &middot; {surprise:+.1f} pp against consensus"
            if surprise is not None
            else ""
        )
        landed = (
            f'<p class="prep-landed">This release printed '
            f'<strong>{escape(SCENARIO_LABELS.get(key, key))}</strong>{gap}.</p>'
        )

    return (
        '<section class="prep-header">'
        '<p class="prep-kicker">Event preparation</p>'
        f"<h1>{escape(meta.label)}</h1>"
        # The measure, directly under the name. Without it a reader has no way
        # to tell whether 3.8% is headline or core, annual or monthly -- and
        # those move in different directions on the same morning.
        f'<p class="prep-measure">{escape(CPI_METRIC_SHORT)}</p>'
        f'<p class="prep-when">{local:%A %d %B %Y} &middot; '
        f"{local:%H:%M} ET{in_your_time}"
        f"{state}{countdown}</p>"
        f'<p class="prep-sub">{escape(meta.summary)}</p>'
        f'<div class="prep-figures prep-figures-headline">{figures}</div>'
        f"{landed}"
        "</section>"
    )


def _figure_cell(figure: PublishedValue) -> str:
    """One published figure, or the reason there is not one.

    The reason is the point. A dash alone collapses "nobody has forecast this
    yet", "we have not transcribed it" and "it cannot exist until Tuesday" into
    one shrug, and only one of those is a gap in this product.
    """
    classes = "prep-figure"
    if not figure.is_known:
        classes += " prep-figure-missing"
    value = _pct(figure.value) if figure.value is not None else "&mdash;"
    note = (
        f'<span class="prep-figure-note">{escape(figure.note)}</span>'
        if figure.note
        else ""
    )
    return (
        f'<div class="{classes}">'
        f'<span class="prep-figure-label">{escape(figure.label)}</span>'
        f'<span class="prep-figure-value">{value}</span>{note}'
        "</div>"
    )


def _pct(value: float | None) -> str:
    return "&mdash;" if value is None else f"{value:.1f}%"


def _what_to_watch(prep: Preparation) -> str:
    """A short deterministic summary, templated over computed facts.

    Every clause is a number from the study or the dataset. Nothing is
    predicted, nothing is recommended, and the verbs are historical: this
    section says what the evidence on the page shows, not what to do about it.
    """
    lines = what_to_watch_lines(prep)
    if not lines:
        return ""
    items = "".join(f"<li>{line}</li>" for line in lines)
    return (
        '<section class="prep-watch panel-card">'
        "<h2>What to watch</h2>"
        f"<ul>{items}</ul>"
        '<p class="muted">Historical frequencies from the sample on this page. '
        "Not a forecast, and not advice.</p>"
        "</section>"
    )


def what_to_watch_lines(prep: Preparation) -> list[str]:
    """The summary sentences, as HTML fragments.

    Public so the behaviour can be tested without parsing a page, and so a
    sentence the numbers cannot support is absent rather than hedged into
    meaning nothing.

    The first sentence ranks *every* group rather than comparing two. An
    earlier version set cooler against hotter and stopped, which in this
    dataset skipped the strongest of the three -- a true sentence that left the
    reader with a false impression, which is the worse kind.
    """
    lines: list[str] = []
    name = SYMBOL_NAMES.get(prep.symbol, prep.symbol)

    ranked = sorted(
        (
            (cohort, median)
            for cohort in prep.scenarios
            if (median := _median(cohort, HEADLINE_WINDOW)) is not None
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if ranked:
        parts = [
            f"{escape(SCENARIO_LABELS.get(cohort.key, cohort.label)).lower()} "
            f"({_signed(median)}, n={cohort.sample_size})"
            for cohort, median in ranked
        ]
        lines.append(
            f"In this sample, median 24-hour {escape(name)} outcomes ran "
            + ", then ".join(parts)
            + "."
        )
        # Named rather than left to the badge on a card further down. A median
        # over three releases and one over thirty read identically in a
        # sentence, and this is the sentence people quote.
        smallest = min(ranked, key=lambda pair: pair[0].sample_size)[0]
        if smallest.quality.needs_warning:
            lines.append(
                f"The {escape(SCENARIO_LABELS.get(smallest.key, smallest.label)).lower()} "
                f"group holds only <strong>{smallest.sample_size}</strong> "
                "releases, which is too few to describe a tendency."
            )

    entry = prep.entry_return_pct
    if entry is not None:
        direction = "down" if entry < 0 else "up"
        verb = "went into this release" if prep.released else "is entering this release"
        lines.append(
            f"{escape(name)} {verb} <strong>{direction} {abs(entry):.1f}%</strong> "
            "over the previous 24 hours."
        )

    match = prep.entry_match
    if match is not None:
        window = match.windows.get(HEADLINE_WINDOW)
        if window is not None and window.sample_size:
            lines.append(
                "Other releases the asset entered the same way: "
                f"<strong>n&nbsp;=&nbsp;{match.sample_size}</strong>, median 24-hour "
                f"outcome {_signed(window.median_return_pct)}, positive "
                f"{window.up_count} of {window.sample_size}."
            )

    if prep.released and prep.scenario_key:
        lines.append(
            "This release landed "
            f"<strong>{escape(SCENARIO_LABELS.get(prep.scenario_key, prep.scenario_key))}"
            "</strong>; the matching scenario is highlighted below."
        )
    return lines


def _median(cohort: Cohort | None, window: str) -> float | None:
    if cohort is None:
        return None
    aggregate = cohort.windows.get(window)
    if aggregate is None or not aggregate.sample_size:
        return None
    return aggregate.median_return_pct


def _signed(value: float) -> str:
    return f"{value:+.1f}%"


def _asset_row(prep: Preparation, symbols: tuple[str, ...]) -> str:
    """Which market the whole page is measured against."""
    tabs = "".join(
        f'<a class="asset-tab" '
        f'href="/event/{escape(prep.event.event_id)}/prepare?symbol={symbol}" '
        f'aria-selected="{str(symbol == prep.symbol).lower()}">'
        f"{escape(SYMBOL_NAMES.get(symbol, symbol))}</a>"
        for symbol in symbols
    )
    return (
        '<div class="asset-row">'
        f'<div class="asset-tabs">{tabs}</div>'
        f'<span class="asset-ticker mono">{escape(prep.symbol)}</span>'
        "</div>"
    )


def _scenario_section(prep: Preparation) -> str:
    """The playbook: a table to scan, then a card per outcome to read."""
    if not prep.scenarios:
        return (
            '<section class="prep-scenarios"><h2>Scenario playbook</h2>'
            '<div class="panel-card empty-state"><p>No release in the dataset '
            "has a verified consensus, so none can be sorted into an outcome. "
            "The rest of this page still works.</p></div></section>"
        )

    cards = "".join(_scenario_card(prep, cohort) for cohort in prep.scenarios)
    return (
        '<section class="prep-scenarios">'
        "<h2>Scenario playbook</h2>"
        f'<p class="section-sub">How '
        f"{escape(SYMBOL_NAMES.get(prep.symbol, prep.symbol))} behaved after past "
        f"releases, grouped by where the {escape(CPI_METRIC_SHORT)} figure "
        "landed against the forecast. Each figure is measured from the release "
        "price.</p>"
        f"{_scenario_summary(prep)}"
        f'<div class="prep-cards">{cards}</div>'
        "</section>"
    )


def _scenario_summary(prep: Preparation) -> str:
    """The whole playbook in one scannable table.

    Above the cards because the first question is comparative -- which outcome
    has meant what -- and three cards side by side answer that only after being
    read three times.
    """
    rows = ""
    for cohort in prep.scenarios:
        window = cohort.windows.get(HEADLINE_WINDOW)
        label = SCENARIO_LABELS.get(cohort.key, cohort.label)
        rule = SCENARIO_RULE_TEXT.get(cohort.key, "")
        occurred = ' class="prep-row-occurred"' if prep.scenario_key == cohort.key else ""
        if window is None or not window.sample_size:
            rows += (
                f"<tr{occurred}><th>{escape(label)}</th>"
                f'<td class="prep-rule">{escape(rule)}</td>'
                f"<td>{cohort.sample_size}</td>"
                '<td class="prep-none" colspan="2">not measurable</td></tr>'
            )
            continue
        rows += (
            f"<tr{occurred}><th>{escape(label)}</th>"
            f'<td class="prep-rule">{escape(rule)}</td>'
            f"<td>{cohort.sample_size}</td>"
            f'<td class="{_sign(window.median_return_pct)}">'
            f"{_signed(window.median_return_pct)}</td>"
            f"<td>{window.up_count} / {window.sample_size}</td></tr>"
        )

    baseline = ""
    if prep.baseline is not None:
        window = prep.baseline.windows.get(HEADLINE_WINDOW)
        if window is not None and window.sample_size:
            baseline = (
                '<tr class="prep-row-baseline"><th>All releases</th>'
                '<td class="prep-rule">every measured release, however it '
                "landed</td>"
                f"<td>{prep.baseline.sample_size}</td>"
                f'<td class="{_sign(window.median_return_pct)}">'
                f"{_signed(window.median_return_pct)}</td>"
                f"<td>{window.up_count} / {window.sample_size}</td></tr>"
            )

    return (
        '<div class="prep-scroll panel-card prep-summary">'
        '<table class="prep-table"><thead><tr>'
        "<th>Scenario</th><th>Rule</th><th>n</th>"
        "<th>Median 24h</th><th>Positive</th>"
        f"</tr></thead><tbody>{rows}{baseline}</tbody></table></div>"
    )


def _scenario_card(prep: Preparation, cohort: Cohort) -> str:
    """One outcome: the three horizons, the frequency, the excursions.

    The sample warning is a badge and at most one sentence. An earlier version
    repeated the band, a description and a caveat on every card, which is three
    ways of saying the same thing and made the numbers harder to find than the
    hedging around them.
    """
    label = SCENARIO_LABELS.get(cohort.key, cohort.label)
    occurred = prep.scenario_key == cohort.key
    rows = "".join(_card_row(cohort, name, prep.baseline) for name in CARD_WINDOWS)
    headline = cohort.windows.get(HEADLINE_WINDOW)

    frequency = (
        '<p class="prep-freq">Positive after 24 hours '
        f"<strong>{headline.up_count} of {headline.sample_size}</strong></p>"
        if headline is not None and headline.sample_size
        else '<p class="prep-freq muted">No release in this group has a '
        "complete 24-hour window.</p>"
    )
    path = (
        f'<p class="prep-path">Best move {_signed(headline.mean_mfe_pct)} '
        f"&middot; worst {_signed(headline.mean_mae_pct)}, on average, within "
        "the 24 hours</p>"
        if headline is not None and headline.sample_size
        else ""
    )

    return (
        f'<article class="prep-card{" prep-card-occurred" if occurred else ""}">'
        + ('<span class="prep-occurred">This is what happened</span>' if occurred else "")
        + f"<h3>{escape(label)}</h3>"
        + f'<p class="prep-rule-line">{escape(SCENARIO_RULE_TEXT.get(cohort.key, ""))}</p>'
        + _card_sample(cohort)
        + f'<table class="prep-rows"><tbody>{rows}</tbody></table>'
        + frequency
        + path
        + "<details><summary>Full reaction grid</summary>"
        + '<div class="prep-scroll">'
        + render_cohort_table(cohort, REACTION_WINDOWS)
        + "</div></details>"
        + "</article>"
    )


def _card_sample(cohort: Cohort) -> str:
    """The count, its band, and one sentence only when one is needed.

    The shared badge carries a description and a caveat as well, which is right
    on a page showing one cohort and repetitive on a page showing three side by
    side. The honesty is in the count and the band; the prose was decoration
    around them.
    """
    quality = cohort.quality
    band = (
        f'<span class="cohort-quality quality-{quality.name.lower()}">'
        f"{escape(quality.value)} sample</span>"
        if quality.needs_warning
        else ""
    )
    caveat = (
        '<p class="prep-caveat">Too few to describe a tendency. Read these as '
        "individual cases.</p>"
        if quality.needs_warning and not cohort.is_reliable
        else ""
    )
    return (
        f'<p class="prep-n"><span class="cohort-n">n={cohort.sample_size}</span>'
        f"{band}</p>{caveat}"
    )


def _card_row(cohort: Cohort, name: str, baseline: Cohort | None) -> str:
    """One horizon: this cohort's median, and how it sits against all releases.

    The difference is the only arithmetic this module performs, and it is a
    subtraction of two medians the research layer computed. Quoted in
    percentage points because the difference between two percentages is not
    itself a percentage.
    """
    aggregate = cohort.windows.get(name)
    label = WINDOW_LABELS.get(name, name)
    if aggregate is None or not aggregate.sample_size:
        return (
            f"<tr><th>{escape(label)}</th>"
            '<td class="prep-none" colspan="2">not measurable</td></tr>'
        )

    base = _median(baseline, name)
    against = ""
    if base is not None:
        gap = aggregate.median_return_pct - base
        against = f'<td class="prep-vs {_sign(gap)}">{gap:+.1f} pp vs all</td>'

    return (
        f"<tr><th>{escape(label)}</th>"
        f'<td class="prep-val {_sign(aggregate.median_return_pct)}">'
        f"{_signed(aggregate.median_return_pct)}</td>{against}</tr>"
    )


def _sign(value: float | None) -> str:
    if value is None:
        return ""
    return "up" if value > 0 else "down" if value < 0 else ""


def _horizon_chart(prep: Preparation) -> str:
    """Median return at each horizon, grouped by outcome.

    Bars rather than lines, and that is the whole point of the change. These
    five numbers are *separate measurements from the same anchor* -- the 24h
    median is not the 4h median plus something -- so a line joining them draws a
    journey nobody measured and invites a reader to see momentum in it. The
    reaction windows are meant to be read as a sequence of comparisons, and
    bars put the comparison where it belongs: between the groups at one
    horizon, not along one group across horizons.

    Drawn from the aggregates already on the page: five numbers per cohort is
    the whole series, so there is nothing to fetch and nothing to downsample.
    """
    series: list[tuple[str, list[float | None], str]] = []
    for cohort in prep.scenarios:
        values = [_median(cohort, name) for name in WINDOW_LABELS]
        if any(value is not None for value in values):
            series.append(
                (
                    SCENARIO_LABELS.get(cohort.key, cohort.label),
                    values,
                    SERIES_CLASS.get(cohort.key, "prep-series-other"),
                )
            )
    if prep.baseline is not None:
        base = [_median(prep.baseline, name) for name in WINDOW_LABELS]
        if any(value is not None for value in base):
            series.append(("All releases", base, "prep-series-base"))

    if len(series) < 2:
        return ""

    every = [v for _label, values, _klass in series for v in values if v is not None]
    low, high = min(every + [0.0]), max(every + [0.0])
    pad = max((high - low) * 0.12, 0.15)
    low, high = low - pad, high + pad

    width, height, left, bottom, top = 640, 260, 52, 30, 10
    names = list(WINDOW_LABELS)
    plot = width - left - 12
    group = plot / len(names)
    # A gutter between groups, then one bar per series inside each.
    bar = (group * 0.74) / len(series)

    def y(value: float) -> float:
        span = height - bottom - top
        return (height - bottom) - (value - low) / (high - low) * span

    zero = y(0.0)
    bars = ""
    for order, (_label, values, klass) in enumerate(series):
        for index, value in enumerate(values):
            if value is None:
                continue
            x = left + index * group + group * 0.13 + order * bar
            top_y, bottom_y = min(y(value), zero), max(y(value), zero)
            bars += (
                f'<rect class="prep-bar {klass}" x="{x:.1f}" y="{top_y:.1f}" '
                f'width="{max(bar - 1.5, 1):.1f}" height="{max(bottom_y - top_y, 0.8):.1f}">'
                f"<title>{escape(_label)} at {escape(list(WINDOW_LABELS.values())[index])}: "
                f"{_signed(value)} median</title></rect>"
            )

    axis = (
        f'<line class="prep-axis" x1="{left}" y1="{zero:.1f}" '
        f'x2="{width - 12}" y2="{zero:.1f}" />'
    )
    ticks = "".join(
        f'<text class="prep-tick" x="{left + index * group + group / 2:.1f}" '
        f'y="{height - 10}" text-anchor="middle">{escape(label)}</text>'
        for index, label in enumerate(WINDOW_LABELS.values())
    )
    scale = "".join(
        f'<text class="prep-tick" x="{left - 8}" y="{y(value) + 4:.1f}" '
        f'text-anchor="end">{value:+.1f}%</text>'
        for value in (high - pad / 2, 0.0, low + pad / 2)
        if low <= value <= high
    )
    legend = "".join(
        f'<span class="prep-legend-item"><i class="prep-key {klass}"></i>'
        f"{escape(label)}</span>"
        for label, _values, klass in series
    )

    return (
        '<section class="prep-chart panel-card">'
        "<h2>Median reaction by horizon</h2>"
        '<p class="section-sub">Each bar is the median across that group, '
        "measured from the release price. Five separate measurements from the "
        "same instant, not a path any release followed &mdash; the 24-hour "
        "figure is not the 4-hour figure plus something.</p>"
        f'<div class="prep-legend">{legend}</div>'
        # Scrolls rather than shrinks on a narrow screen. Scaling a 640-wide
        # drawing into 300 pixels takes the axis text with it, and a chart
        # whose labels cannot be read is a decoration.
        '<div class="prep-scroll">'
        f'<svg viewBox="0 0 {width} {height}" class="prep-svg" role="img" '
        'aria-label="Median return at each horizon, grouped by CPI outcome">'
        f"{scale}{axis}{ticks}{bars}</svg></div>"
        "</section>"
    )


def _context_section(prep: Preparation) -> str:
    """The market going into the release, and the releases it resembles."""
    name = SYMBOL_NAMES.get(prep.symbol, prep.symbol)
    context = prep.context

    if context.price is None:
        body = (
            '<div class="panel-card empty-state"><p>No recent candles for '
            f"{escape(prep.symbol)}, so the run-up cannot be measured. The "
            "historical sections above are unaffected.</p></div>"
        )
        return f'<section class="prep-context"><h2>Market going in</h2>{body}</section>'

    cells = "".join(
        _context_cell(CONTEXT_LABELS[spec.name], context.ret(spec.name))
        for spec in CONTEXT_WINDOWS
    )
    price = (
        '<div class="prep-figure"><span class="prep-figure-label">Price</span>'
        f'<span class="prep-figure-value mono">{context.price:,.2f}</span>'
        '<span class="prep-figure-note">as of '
        f"{local_stamp(context.as_of)}</span>"
        "</div>"
    )

    heading = "Market now" if prep.released else "Market going in"
    lead = (
        f"{escape(name)} as it stands, measured the same way every historical "
        "figure on this page was. The release has already happened, so the "
        "comparison below uses the 24 hours that ran into it."
        if prep.released
        else f"{escape(name)} into this release, measured the same way every "
        "historical figure on this page was."
    )
    return (
        '<section class="prep-context">'
        f"<h2>{heading}</h2>"
        f'<p class="section-sub">{lead}</p>'
        f'<div class="panel-card"><div class="prep-figures">{price}{cells}</div>'
        f"{_similar(prep)}</div>"
        "</section>"
    )


def _context_cell(label: str, value: float | None) -> str:
    """One return over the run-up.

    Its own markup rather than the published-figure cell: that one carries a
    status explaining why a number is absent from a dataset, and a return that
    cannot be measured is absent for an entirely different reason -- the
    candles are not there.
    """
    if value is None:
        return (
            '<div class="prep-figure prep-figure-missing">'
            f'<span class="prep-figure-label">{escape(label)}</span>'
            '<span class="prep-figure-value">&mdash;</span>'
            '<span class="prep-figure-note">no candles for this window</span>'
            "</div>"
        )
    return (
        '<div class="prep-figure">'
        f'<span class="prep-figure-label">{escape(label)}</span>'
        f'<span class="prep-figure-value {_sign(value)}">{_signed(value)}</span>'
        "</div>"
    )


def _similar(prep: Preparation) -> str:
    """How releases the asset entered the same way turned out.

    The one join between today and the history above. Selection is the 24 hours
    *before* each release, so a release is chosen by what was already knowable
    when the figure printed -- never by what it did afterwards.
    """
    match = prep.entry_match
    if match is None:
        measured = "run-up to this release" if prep.released else "run-up so far"
        return (
            f'<p class="prep-similar muted">The {measured} cannot be measured, '
            "so there is no comparable group to show.</p>"
        )

    window = match.windows.get(HEADLINE_WINDOW)
    if window is None or not window.sample_size:
        return (
            f'<p class="prep-similar muted">{escape(match.label)}: '
            f"n&nbsp;=&nbsp;{match.sample_size}, but none has a complete "
            "24-hour window.</p>"
        )

    direction = prep.entry_direction
    phrase = "fallen" if direction == "down" else "risen"
    base = _median(prep.baseline, HEADLINE_WINDOW)
    against = (
        f" That is {window.median_return_pct - base:+.1f} pp against all releases."
        if base is not None
        else ""
    )

    return (
        '<div class="prep-similar">'
        "<h3>Releases that began the same way</h3>"
        f"<p>Past releases the asset had also {phrase} into over the previous "
        f"24 hours: <strong>n&nbsp;=&nbsp;{match.sample_size}</strong>. "
        f"Median 24-hour outcome {_signed(window.median_return_pct)}, positive "
        f"{window.up_count} of {window.sample_size}.{escape(against)}</p>"
        f"{_card_sample(match)}"
        "</div>"
    )


def _occurrences(prep: Preparation) -> str:
    """The releases behind the statistics, with the figures that sorted them.

    Present for trust rather than for completeness. A reader who doubts that a
    release belongs in "cooler" can read the consensus, the actual and the gap,
    and check the arithmetic -- which is possible only because the rule is a
    comparison rather than a score.
    """
    if prep.study is None or not prep.scenarios:
        return ""

    by_id = {event.event_id: event for event in prep.study.events}
    blocks = ""
    for cohort in prep.scenarios:
        rows = "".join(
            _occurrence_row(event_id, prep.all_values.get(event_id), by_id.get(event_id))
            for event_id in cohort.event_ids
        )
        if not rows:
            continue
        label = SCENARIO_LABELS.get(cohort.key, cohort.label)
        blocks += (
            f"<details><summary>{escape(label)} "
            f'<span class="tab-n">{cohort.sample_size}</span></summary>'
            '<div class="prep-scroll"><table class="prep-table"><thead><tr>'
            "<th>Date</th><th>Previous</th><th>Consensus</th><th>Actual</th>"
            "<th>Surprise</th><th>1h</th><th>24h</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div></details>"
        )

    if not blocks:
        return ""
    return (
        '<section class="prep-occurrences">'
        "<h2>The releases behind these numbers</h2>"
        f'<div class="panel-card">{blocks}</div>'
        "</section>"
    )


def _occurrence_row(
    event_id: str, values: ReleaseValues | None, event: EventResult | None
) -> str:
    if values is None or event is None:
        return ""
    surprise = values.surprise_pp
    gap = (
        f'<td class="{_sign(surprise)}">{surprise:+.1f} pp</td>'
        if surprise is not None
        else '<td class="prep-none">&mdash;</td>'
    )
    return (
        f"<tr><td>{event.local_date:%d %b %Y}</td>"
        f"<td>{_pct(values.previous_yoy)}</td>"
        f"<td>{_pct(values.consensus_yoy)}</td>"
        f"<td>{_pct(values.actual_yoy)}</td>"
        f"{gap}"
        f"{_cell(event, 'post_event_1h')}{_cell(event, 'post_event_24h')}</tr>"
    )


def _cell(event: EventResult, name: str) -> str:
    window = event.windows.get(name)
    if window is None or not window.is_usable or window.return_pct is None:
        return '<td class="prep-none">&mdash;</td>'
    return f'<td class="{_sign(window.return_pct)}">{_signed(window.return_pct)}</td>'


def _methodology(prep: Preparation) -> str:
    """Enough for a trader to audit the page, without a developer console."""
    study = prep.study
    counted = len(study.events) if study is not None else 0
    rows = (
        ("Event", "US CPI, from the curated release calendar"),
        ("Measure", CPI_METRIC),
        # Said as plainly as what is covered, because a reader who assumes core
        # or month-over-month is reading a different release's statistics off
        # this page and nothing else here would tell them.
        (
            "Not covered",
            "Core CPI and month-over-month are different series that can "
            "surprise in opposite directions on the same morning. Neither is "
            "in this dataset, and nothing on this page describes them.",
        ),
        ("Asset", f"{SYMBOL_NAMES.get(prep.symbol, prep.symbol)} ({prep.symbol})"),
        ("Price data", "Binance candles, 5-minute, from our own cache"),
        ("Measured from", "The release instant, 08:30 America/New_York"),
        (
            "Horizons",
            ", ".join(WINDOW_LABELS.values()) + " after the release",
        ),
        (
            "Scenario rule",
            "Actual against verified consensus, compared to one decimal place. "
            "A release with no verified consensus is left unclassified rather "
            "than called in line.",
        ),
        (
            "Similar setups",
            "The 24 hours before each release, which ends at the release. "
            "Nothing after the figure printed takes part in selecting them.",
        ),
        ("Releases examined", str(counted)),
    )
    body = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in rows
    )
    exclusions = _exclusions(prep)
    return (
        '<section class="prep-method">'
        "<details><summary>Methodology and data</summary>"
        f'<table class="prep-table prep-method-table"><tbody>{body}</tbody></table>'
        f"{exclusions}</details></section>"
    )


def _exclusions(prep: Preparation) -> str:
    """What is not in a cohort, counted by reason.

    A page reporting on twenty-two of twenty-seven releases without saying so
    would be claiming a completeness it does not have.
    """
    if prep.study is None:
        return ""
    tally: dict[str, int] = {}
    for event in prep.study.events:
        values = prep.all_values.get(event.event_id)
        reason = (
            values.exclusion_reason()
            if values is not None
            else "no published values recorded for this release"
        )
        if reason:
            tally[reason] = tally.get(reason, 0) + 1
    if not tally:
        return ""
    items = "".join(
        f"<li>{count} &mdash; {escape(reason)}</li>" for reason, count in tally.items()
    )
    return (
        '<div class="prep-exclusions"><h4>Not sorted into a scenario</h4>'
        f"<ul>{items}</ul></div>"
    )


def _followup(prep: Preparation) -> str:
    """A way into the conversational side, pre-loaded with a real question.

    Optional by construction. Everything above was computed without a model,
    and this is the one control on the page that spends anything -- so it is a
    link that hands the question over, not a box that quietly bills for a view.
    """
    name = SYMBOL_NAMES.get(prep.symbol, prep.symbol)
    suggestions = (
        f"How did {name} behave after cooler CPI releases "
        "when it was already down over the previous week?",
        f"Compare {name} after cooler CPI against all CPI releases.",
    )
    links = "".join(
        f'<a class="prep-ask" href="/research?q={quote(question)}">'
        f"{escape(question)}</a>"
        for question in suggestions
    )
    return (
        '<section class="prep-followup panel-card">'
        "<h2>Ask a follow-up</h2>"
        '<p class="section-sub">Everything above is computed from the dataset '
        "and the candle cache. A question that goes beyond it opens the "
        "research view, which does use a model.</p>"
        f'<div class="prep-asks">{links}</div>'
        "</section>"
    )


def render_reaction(prep: Preparation) -> str:
    """The developing reaction to a release that has already happened.

    Horizons that have not closed carry no figure. An unfinished 24-hour window
    is not zero, and showing one would be the single most misleading number the
    page could print on the day it matters most.
    """
    if prep.reaction is None:
        return ""
    rows = prep.reaction["rows"]
    if not isinstance(rows, list) or not rows:
        return ""
    cells = "".join(
        (
            f'<div class="prep-figure"><span class="prep-figure-label">'
            f'{escape(str(row["label"]))}</span>'
            + (
                f'<span class="prep-figure-value {_sign(float(row["return_pct"]))}">'
                f'{_signed(float(row["return_pct"]))}</span>'
                if row["return_pct"] is not None
                else '<span class="prep-figure-value">&mdash;</span>'
                f'<span class="prep-figure-note">{escape(str(row["reason"]))}</span>'
            )
            + "</div>"
        )
        for row in rows
    )
    return (
        '<section class="prep-reaction">'
        "<h2>What actually happened</h2>"
        '<p class="section-sub">Measured from this release, at each horizon '
        "that has closed.</p>"
        f'<div class="panel-card"><div class="prep-figures">{cells}</div></div>'
        "</section>"
    )


PREPARE_STYLE = """
.prep-header { margin: 0 0 22px; }
.prep-kicker { margin: 0 0 6px; font-size: 12px; letter-spacing: 0.08em;
               text-transform: uppercase; color: var(--text-muted); }
.prep-header h1 { font-size: 30px; margin: 0 0 4px; letter-spacing: -0.02em; }
/* The exact series, directly under the name. Quiet but never absent: a reader
   who assumes core or month-over-month is reading different statistics. */
.prep-measure { margin: 0 0 10px; font-size: 13px; font-weight: 500;
                color: var(--text-secondary); letter-spacing: 0.01em; }
.prep-when { margin: 0 0 6px; font-size: 13px; color: var(--text-secondary);
             display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.prep-sub { margin: 0 0 16px; color: var(--text-muted); font-size: 14px;
            max-width: 72ch; }
.prep-state { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
              padding: 3px 8px; border-radius: 999px; font-weight: 600; }
.prep-state-upcoming { background: var(--surface-2); color: var(--text-primary); }
.prep-state-released { background: var(--surface-2); color: var(--text-muted); }
.prep-countdown { color: var(--text-muted); }

.prep-figures { display: flex; flex-wrap: wrap; gap: 10px; }
.prep-figure { flex: 1 1 130px; background: var(--surface-1);
               border: 1px solid var(--grid); border-radius: 10px;
               padding: 12px 14px; display: flex; flex-direction: column; gap: 3px; }
.prep-figure-label { font-size: 11px; letter-spacing: 0.06em;
                     text-transform: uppercase; color: var(--text-muted); }
.prep-figure-value { font-size: 21px; font-weight: 600;
                     font-variant-numeric: tabular-nums; }
.prep-figure-note { font-size: 11px; color: var(--text-muted);
                     line-height: 1.35; }
/* A figure we do not have reads as quieter than one we do, rather than as an
   error. The note beside it says which kind of absence it is. */
.prep-figure-missing .prep-figure-value { color: var(--text-muted); }
.prep-figures-headline .prep-figure { min-height: 76px; }
.prep-landed { margin: 14px 0 0; font-size: 14px; }

.prep-watch { margin: 0 0 22px; }
.prep-watch h2 { font-size: 15px; margin: 0 0 10px; }
.prep-watch ul { margin: 0 0 10px; padding-left: 18px; }
.prep-watch li { margin: 0 0 6px; font-size: 14px; line-height: 1.5;
                 max-width: 78ch; }

.prep-scenarios, .prep-context, .prep-occurrences, .prep-method,
.prep-chart, .prep-followup, .prep-reaction { margin: 0 0 26px; }
.prep-scenarios h2, .prep-context h2, .prep-occurrences h2, .prep-chart h2,
.prep-followup h2, .prep-reaction h2 { font-size: 17px; margin: 0 0 4px; }
.section-sub { margin: 0 0 14px; color: var(--text-muted); font-size: 13px;
               max-width: 76ch; }

.prep-summary { padding: 4px 16px 10px; margin: 0 0 16px; }
.prep-summary .prep-table th:first-child { width: 26%; }
.prep-summary .prep-table th:not(:first-child),
.prep-summary .prep-table td:not(:first-child):not(.prep-rule) {
  text-align: right; }
.prep-summary tbody th { font-weight: 600; color: var(--text-primary);
                         font-size: 13px; }
.prep-rule { color: var(--text-muted); font-size: 12px; }
.prep-row-occurred th, .prep-row-occurred td { background: var(--surface-2); }
.prep-row-baseline th, .prep-row-baseline td { color: var(--text-muted); }
.prep-row-baseline th { font-weight: 500; }

.prep-cards { display: grid; gap: 14px;
              grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.prep-card { background: var(--surface-1); border: 1px solid var(--grid);
             border-radius: 12px; padding: 16px 18px; position: relative; }
.prep-card h3 { font-size: 15px; margin: 0 0 8px; }
.prep-card-occurred { border-color: var(--accent); }
.prep-occurred { position: absolute; top: -9px; right: 14px; font-size: 10px;
                 letter-spacing: 0.06em; text-transform: uppercase;
                 background: var(--accent); color: #fff; padding: 2px 8px;
                 border-radius: 999px; font-weight: 600; }
.prep-rule-line { margin: 0 0 10px; font-size: 12px; color: var(--text-muted); }
.prep-n { margin: 0 0 10px; font-size: 12px; display: flex; flex-wrap: wrap;
          gap: 6px; align-items: baseline; }
/* One sentence, and only for a sample too small to describe a tendency. The
   band beside the count already carries the warning for everything else. */
.prep-caveat { margin: -4px 0 10px; font-size: 12px; line-height: 1.45;
               color: var(--warn, #c9a227); }
.prep-rows { width: 100%; border-collapse: collapse; margin: 0 0 12px; }
.prep-rows th { text-align: left; font-weight: 500; color: var(--text-muted);
                font-size: 12px; padding: 4px 0; width: 48px; }
.prep-rows td { padding: 5px 0; font-variant-numeric: tabular-nums;
                font-size: 15px; }
.prep-val { font-weight: 600; }
.prep-vs { text-align: right; font-size: 12px; color: var(--text-muted); }
.prep-none { color: var(--text-muted); }
.prep-freq { margin: 0 0 4px; font-size: 13px; }
.prep-path { margin: 0 0 10px; font-size: 12px; color: var(--text-muted); }
.prep-card details summary { font-size: 12px; color: var(--text-muted);
                             cursor: pointer; }

.prep-svg { width: 100%; height: auto; min-width: 460px; }
.prep-axis { stroke: var(--grid); stroke-width: 1; }
.prep-tick { fill: var(--text-muted); font-size: 10px; }
/* Bars, not a line. These are five separate measurements from one anchor,
   and a line between them draws a journey nobody measured. */
.prep-bar { stroke: none; }
.prep-bar.prep-series-hotter { fill: #d9644a; }
.prep-bar.prep-series-inline { fill: #c9a227; }
.prep-bar.prep-series-cooler { fill: #4d9ef6; }
.prep-bar.prep-series-other { fill: #8a7fd4; }
.prep-bar.prep-series-base { fill: var(--text-muted); opacity: 0.55; }
.prep-legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 10px;
               font-size: 12px; color: var(--text-secondary); }
.prep-legend-item { display: inline-flex; align-items: center; gap: 6px; }
.prep-key { width: 11px; height: 11px; display: inline-block;
             border-radius: 2px; }
.prep-key.prep-series-hotter { background: #d9644a; }
.prep-key.prep-series-inline { background: #c9a227; }
.prep-key.prep-series-cooler { background: #4d9ef6; }
.prep-key.prep-series-other { background: #8a7fd4; }
.prep-key.prep-series-base { background: var(--text-muted); opacity: 0.55; }

.prep-similar { margin: 16px 0 0; padding: 14px 0 0;
                border-top: 1px solid var(--grid); }
.prep-similar h3 { font-size: 14px; margin: 0 0 6px; }
.prep-similar p { margin: 0 0 6px; font-size: 14px; max-width: 76ch; }

.prep-table { width: 100%; border-collapse: collapse; font-size: 13px;
              margin: 10px 0 0; }
.prep-table th { text-align: left; font-weight: 500; color: var(--text-muted);
                 font-size: 12px; padding: 6px 10px 6px 0;
                 border-bottom: 1px solid var(--grid); }
.prep-table td { padding: 6px 10px 6px 0; font-variant-numeric: tabular-nums;
                 border-bottom: 1px solid var(--grid); }
/* Figures right, labels left: a column of returns is read by comparing
   magnitudes down it, which only works when the decimal points line up. */
.prep-occurrences .prep-table th:not(:first-child),
.prep-occurrences .prep-table td:not(:first-child) { text-align: right; }
.prep-occurrences .prep-table th:first-child,
.prep-occurrences .prep-table td:first-child { width: 22%; }
.prep-method-table th { width: 170px; vertical-align: top; }
.prep-method-table td { color: var(--text-secondary); }
.prep-occurrences details, .prep-method details { margin: 0 0 8px; }
.prep-occurrences summary, .prep-method summary { cursor: pointer;
    font-size: 13px; padding: 6px 0; }
.prep-exclusions { margin: 14px 0 0; font-size: 13px; }
.prep-exclusions h4 { margin: 0 0 6px; font-size: 13px; }
.prep-exclusions ul { margin: 0; padding-left: 18px; color: var(--text-muted); }

.prep-asks { display: flex; flex-direction: column; gap: 8px; }
.prep-ask { display: block; padding: 10px 12px; border: 1px solid var(--grid);
            border-radius: 8px; font-size: 13px; color: var(--text-secondary);
            text-decoration: none; background: var(--surface-0); }
.prep-ask:hover { color: var(--text-primary); border-color: var(--text-muted); }

.prep-prepare { font-size: 12px; font-weight: 600; padding: 5px 11px;
                border-radius: 7px; background: var(--surface-2);
                color: var(--text-primary); text-decoration: none;
                border: 1px solid var(--grid); white-space: nowrap; }
.prep-prepare:hover { border-color: var(--accent); }

/* A table too wide for the screen scrolls inside its own box. The page
   itself never scrolls sideways, and a column that will not fit is reachable
   rather than cut off at the card edge. */
.prep-scroll { overflow-x: auto; }
.prep-scroll table { min-width: 460px; }

@media (max-width: 640px) {
  .prep-header h1 { font-size: 24px; }
  .prep-cards { grid-template-columns: 1fr; }
  .prep-method-table th { width: auto; }
}
"""
