"""The Research workspace: ask a question, confirm the reading, see the answer.

A research surface rather than a chat window. The conversation exists to build
and refine one specification; what the reader actually works with is the
specification card and the finding, and the prose around them is the smaller
part.

Conversation state lives in the page, not on the server: the current thesis
travels with each request as JSON. A follow-up therefore modifies the thesis on
screen rather than being parsed from nothing, with no session store, no
expiry, and no question of which conversation a stale request belonged to.

Every number rendered here comes from
:class:`~backtool.research.experiment.ResearchFinding`. Nothing in this module
computes, and the headline verdict is the engine's own deterministic sentence,
not something a model wrote.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from backtool.ai.thesis_planner import PlannedThesis
from backtool.events.models import SourceAuthority, SourceRef
from backtool.events.resolution import (
    CandidateEvent,
    DayAnchor,
    ResolutionStatus,
    ResolvedEventConcept,
)
from backtool.reporting.charts import BarItem, distribution_chart, diverging_bar_chart
from backtool.research.experiment import ResearchFinding
from backtool.research.thesis import Thesis
from backtool.research.vocabulary import (
    OUTCOME_HORIZONS,
    Aggregation,
    ResultShape,
    asset_label,
)
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.cohort_view import render_cohort_table, render_sample_badge

#: Starting points, chosen to show the three shapes a question can take: a
#: cohort comparison, a plain event study, and a conditional follow-up.
EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Does BTC perform better after cooler-than-expected CPI?",
    "What usually happens to BTC after FOMC?",
    "Does a bearish 7-day BTC trend change the reaction to CPI?",
)


def render_research_page() -> str:
    """The empty workspace."""
    examples = "".join(
        f'<button type="button" class="example" data-q="{escape(q)}">'
        f"{escape(q)}</button>"
        for q in EXAMPLE_QUESTIONS
    )
    return (
        '<div class="research">'
        '<div class="research-intro">'
        "<h1>What market idea do you want to test?</h1>"
        '<p class="research-sub">Describe a thesis in your own words. Backtool '
        "turns it into an explicit specification, shows you that specification "
        "before running anything, and measures it against real exchange data.</p>"
        "</div>"
        '<form id="research-form" class="research-form">'
        '<textarea id="question" name="question" rows="2" '
        'placeholder="Does BTC perform better after cooler-than-expected CPI?"'
        "></textarea>"
        '<button type="submit" class="btn" id="ask">Interpret</button>'
        "</form>"
        f'<div class="examples">{examples}</div>'
        '<div id="thread"></div>'
        "</div>"
    )


def render_question(question: str) -> str:
    """The reader's own words, echoed into the thread."""
    return f'<div class="turn turn-ask"><p>{escape(question)}</p></div>'


def render_spec_card(thesis: Thesis, planned: PlannedThesis) -> str:
    """The specification, for confirmation before anything runs.

    This is the transparency step the whole product rests on. A reader who
    cannot see what was understood has no way to tell a right answer to the
    wrong question from a right answer to theirs.
    """
    rows = "".join(
        f'<div class="spec-row"><span class="spec-label">{escape(label)}</span>'
        f'<span class="spec-value">{escape(value)}</span></div>'
        for label, value in thesis.describe_items()
    )
    assumptions = (
        '<p class="spec-assume"><strong>Assumed:</strong> '
        + "; ".join(escape(a) for a in planned.assumptions)
        + ". Edit the question if any of these are wrong.</p>"
        if planned.assumptions
        else ""
    )
    return (
        '<div class="turn turn-spec">'
        f'<p class="spec-head">{escape(planned.restated_question)}</p>'
        f'<div class="spec-card">{rows}</div>'
        f"{assumptions}"
        f"{render_event_provenance(thesis)}"
        '<div class="spec-actions">'
        '<button type="button" class="btn" data-run="1">Run research</button>'
        '<button type="button" class="btn-quiet" data-edit="1">Edit question</button>'
        "</div>"
        f'<script type="application/json" class="thesis">{thesis.model_dump_json()}'
        "</script></div>"
    )


def render_event_provenance(thesis: Thesis) -> str:
    """Where a resolved event's dates came from, beside the dates themselves.

    A curated family needs none of this: its calendar is in the repository and
    its methodology page explains it. A concept looked up from the web is
    different in kind -- the dates are the research, and a reader who cannot
    see which page established them is being asked to take the interesting part
    on trust.

    Returns an empty string for a curated thesis, so the caller can include it
    unconditionally.
    """
    resolution = thesis.resolution
    if resolution is None:
        return ""

    rows = "".join(
        f'<div class="spec-row"><span class="spec-label">{escape(event.label)}</span>'
        f'<span class="spec-value">{escape(_timing(event, thesis.anchor))}</span>'
        "</div>"
        for event in resolution.primary_events
    )

    sources = "".join(
        f'<li><a href="{escape(source.url)}" target="_blank" rel="noreferrer">'
        f"{escape(source.title or source.url)}</a> "
        f'<span class="muted">({escape(source.authority.value)}'
        + (f", {escape(source.publisher)}" if source.publisher else "")
        + ")</span></li>"
        for source in _cited_sources(resolution)
    )

    related = resolution.related_events
    context = (
        '<p class="muted">Also found, and deliberately not measured: '
        + ", ".join(
            f"{escape(event.label)} ({event.local_date.isoformat()})"
            for event in related
        )
        + ". These are milestones around the event, not occurrences of it.</p>"
        if related
        else ""
    )

    open_questions = (
        "<p><strong>Worth knowing:</strong></p>"
        + '<ul class="findings-list">'
        + "".join(
            f"<li>{escape(item.question)}"
            + (f' <span class="muted">{escape(item.why_it_matters)}</span>'
               if item.why_it_matters else "")
            + "</li>"
            for item in resolution.informational_ambiguities
        )
        + "</ul>"
        if resolution.informational_ambiguities
        else ""
    )

    caveats = (
        '<p class="muted">'
        + " ".join(escape(note) for note in resolution.missing_information)
        + "</p>"
        if resolution.missing_information
        else ""
    )

    return (
        '<div class="spec-provenance">'
        f'<p class="spec-head">Dates established from published sources</p>'
        f'<div class="spec-card">{rows}</div>'
        f"{context}"
        f"{open_questions}"
        "<p><strong>Sources</strong></p>"
        f'<ul class="findings-list">{sources}</ul>'
        f"{caveats}"
        "</div>"
    )


def _timing(event: CandidateEvent, anchor: DayAnchor | None) -> str:
    """One occurrence's instant, saying whether it was published or chosen.

    The distinction a reader needs most: a sourced timestamp is a fact, while a
    date measured from the start of its day is a decision somebody made, and
    the two must not look alike.
    """
    if event.local_time is not None:
        return (
            f"{event.local_date.isoformat()} "
            f"{event.local_time.strftime('%H:%M')} {event.timezone}, as published"
        )
    where = anchor.describe(event.timezone) if anchor is not None else "no anchor set"
    return f"{event.local_date.isoformat()} ({event.timezone}), measured from {where}"


def _cited_sources(resolution: ResolvedEventConcept) -> list[SourceRef]:
    """The pages behind the occurrences being measured, best first.

    Only the ones an occurrence actually cites. The search returns plenty the
    answer never relied on, and listing those beside the evidence would make
    the provenance look stronger than it is.
    """
    seen: dict[str, SourceRef] = {}
    for event in resolution.primary_events:
        for source in event.sources:
            seen.setdefault(source.url, source)
    order = {
        SourceAuthority.OFFICIAL: 0,
        SourceAuthority.PRIMARY: 1,
        SourceAuthority.SECONDARY: 2,
    }
    return sorted(seen.values(), key=lambda s: order.get(s.authority, 3))


def render_unresolved_event(resolution: ResolvedEventConcept) -> str:
    """A lookup that did not settle, as a question rather than a failure.

    Four different outcomes arrive here and they are not the same thing: a name
    that matches several events, sources that disagree, nothing found at all,
    and something real with no published date. Each needs a different sentence,
    and none of them is an error the reader caused.
    """
    questions = "".join(
        f"<li>{escape(item.question)}"
        + (f' <span class="muted">{escape(item.why_it_matters)}</span>'
           if item.why_it_matters else "")
        + "</li>"
        for item in resolution.blocking_ambiguities
    )
    found = (
        '<p class="muted">What was found: '
        + ", ".join(
            f"{escape(event.label)} ({event.local_date.isoformat()})"
            for event in resolution.candidate_events
        )
        + ".</p>"
        if resolution.candidate_events
        else ""
    )
    notes = (
        '<p class="muted">'
        + " ".join(escape(note) for note in resolution.missing_information)
        + "</p>"
        if resolution.missing_information
        else ""
    )
    return (
        '<div class="turn turn-clarify">'
        f'<p class="spec-head">{escape(resolution.canonical_name or resolution.query)}</p>'
        f"<p>{escape(_status_sentence(resolution))}</p>"
        + (f'<ul class="findings-list">{questions}</ul>' if questions else "")
        + f"{found}{notes}"
        '<p class="muted">Answer in a follow-up and I will measure it.</p>'
        "</div>"
    )


def _status_sentence(resolution: ResolvedEventConcept) -> str:
    """Why this could not be measured, in the reader's terms."""
    if resolution.status is ResolutionStatus.AMBIGUOUS:
        return (
            "That name matches more than one event, and which one is meant "
            "changes the answer."
        )
    if resolution.status is ResolutionStatus.CONFLICTING:
        return (
            "The sources disagree about when this happened, and choosing "
            "between them would decide the result."
        )
    if resolution.status is ResolutionStatus.NOT_FOUND:
        return (
            "I could not establish that this happened from any source I would "
            "trust with a date."
        )
    if resolution.status is ResolutionStatus.NOT_RESOLVABLE:
        return (
            "That is real but has no single published instant -- it happened "
            "over a period rather than at a moment."
        )
    if not resolution.primary_events:
        return "Nothing was found that is an occurrence of this, as opposed to context."
    return "One thing has to be settled before this can be measured."


def render_anchor_needed(resolution: ResolvedEventConcept) -> str:
    """Dates established, but no instant to measure from.

    The sources settled which day something happened and published no time,
    which is ordinary -- an election day has no announcement instant, and a
    ruling often has no stated hour. What is not ordinary is choosing one
    anyway. Midnight local is not a neutral reading: it places the whole
    trading day after the event, so a 1h or 24h window measured from it answers
    a different question than one measured from the close.

    So the study stops here. This is the same refusal
    :func:`~backtool.events.resolution.compile_event_set` makes, surfaced
    before the reader presses a button rather than after.

    What it must not do is invite a follow-up. :class:`DayAnchor` exists and
    the thesis carries one, but nothing reads an anchor out of a question --
    the planner has no field for it -- so "tell me which instant and I will run
    it" would be a promise this cannot keep, and the reader would find that out
    by being refused twice. Saying plainly that the case is not supported yet
    is worth more than an invitation into a loop.
    """
    days = "".join(
        f'<div class="spec-row"><span class="spec-label">{escape(event.label)}</span>'
        f'<span class="spec-value">{escape(event.local_date.isoformat())} '
        f"({escape(event.timezone)}), no published time</span></div>"
        for event in resolution.primary_events
        if event.is_date_only
    )
    return (
        '<div class="turn turn-clarify">'
        f'<p class="spec-head">{escape(resolution.canonical_name or resolution.query)}</p>'
        "<p>I established the dates, but no source gives a time of day, so "
        "there is no instant to measure from:</p>"
        f'<div class="spec-card">{days}</div>'
        "<p>Measuring would need one more decision, and it changes the answer: "
        "a window starting at midnight puts the whole trading day after the "
        "event, while one starting at the close puts it before. Picking for "
        "you would decide the result, so I will not.</p>"
        '<p class="muted">Events dated to the day but not the hour cannot be '
        "measured yet -- choosing that instant is not something this version "
        "supports. Events with a published time, such as a scheduled release "
        "or a timestamped upgrade, work normally.</p></div>"
    )


def render_clarification(planned: PlannedThesis) -> str:
    """Questions that must be answered before the test means anything."""
    items = "".join(f"<li>{escape(q)}</li>" for q in planned.clarifications)
    return (
        '<div class="turn turn-clarify">'
        f'<p class="spec-head">{escape(planned.restated_question)}</p>'
        "<p>I can test that, but one thing would change the answer and the "
        "question does not settle it:</p>"
        f'<ul class="findings-list">{items}</ul>'
        '<p class="muted">Reply with what you meant and I will rebuild the '
        "specification.</p></div>"
    )


def render_unsupported(planned: PlannedThesis) -> str:
    """An honest refusal, naming what is missing."""
    items = "".join(f"<li>{escape(item)}</li>" for item in planned.unsupported)
    return (
        '<div class="turn turn-unsupported">'
        "<p><strong>I cannot test that yet.</strong> It needs data that is not "
        "connected:</p>"
        f'<ul class="findings-list">{items}</ul>'
        '<p class="muted">Backtool measures BTC and ETH around CPI and FOMC '
        "releases, and around public events whose dates it can establish from "
        "published sources. Everything else would be a guess dressed as a "
        "result.</p></div>"
    )


def render_finding(
    finding: ResearchFinding, interpretation: tuple[str, ...] = ()
) -> str:
    """The answer, in the order a reader needs it.

    Headline, then what it suggests, then the comparison, then the picture,
    then the evidence, then everything. The detail is all still here -- it has
    moved below the point rather than in front of it, because a reader who has
    to assemble the meaning from six figures and two charts will usually
    assemble it wrong.

    Args:
        finding: The measured result. The only source of every number here.
        interpretation: Paragraphs explaining it. The deterministic summary
            when the AI reading was unavailable or rejected; the caller decides
            which, and this function cannot tell them apart.
    """
    if finding.sample_size == 0:
        return (
            '<div class="turn turn-result"><div class="panel-card">'
            f'<p class="finding-headline">{escape(finding.verdict())}</p>'
            f"{_interpretation(interpretation)}"
            f"{_excluded(finding)}</div></div>"
        )

    warnings = "".join(
        f'<p class="note">{escape(w)}</p>' for w in finding.warnings
    )

    return (
        '<div class="turn turn-result">'
        '<div class="panel-card">'
        f'<p class="finding-headline">{escape(finding.verdict())}</p>'
        f"{_interpretation(interpretation)}"
        f"{_hero(finding)}{warnings}"
        f"{_distribution(finding)}"
        f"{_observations(finding)}"
        f"{_full_statistics(finding)}"
        f"{_methodology(finding)}"
        "</div></div>"
    )


def _hero(finding: ResearchFinding) -> str:
    """The block that answers the question, chosen by the result's shape.

    Three layouts, selected from
    :class:`~backtool.research.vocabulary.ResultShape` -- which is derived from
    the query, not carried as a question type. A new kind of question reaches
    whichever of these fits its shape without adding a fourth branch here.

    Everything below this block is shared: the distribution, the releases, the
    full statistics and the methodology look the same whatever was asked.
    """
    if finding.shape is ResultShape.RANKED:
        return _ranked_rows(finding)
    if finding.shape is ResultShape.SCALAR:
        return _scalar_figure(finding)
    return _comparison(finding)


def _comparison(finding: ResearchFinding) -> str:
    """Two groups: three tiles and a two-bar chart.

    Symmetrical. Each tile carries its own group's name and its own sample
    size, because in an A-versus-B question neither side is the other's
    background.
    """
    cohort, other = finding.cohort_metrics, finding.comparison_metrics
    if finding.value is None or finding.comparison_value is None:
        return ""

    # The measure, said once, the same way the specification card says it.
    # Building it from the aggregation and the metric produced "frequency 24h
    # return", which reads as a kind of return rather than a count of them.
    label = finding.thesis.outcome.describe()
    tiles = (
        _tile(
            finding.cohort.label,
            finding.format_value(finding.value),
            finding.sample_size,
            _tile_detail(finding, cohort, finding.matched_count, finding.sample_size),
            label,
        )
        + _tile(
            _group_name(finding),
            finding.format_value(finding.comparison_value),
            finding.comparison_sample_size,
            _tile_detail(
                finding,
                other,
                finding.comparison_matched_count,
                finding.comparison_sample_size,
            ),
            label,
        )
        + _difference_tile(finding)
    )
    return f'<div class="compare">{tiles}</div>{_comparison_chart(finding)}'


def _overlap_note(finding: ResearchFinding) -> str:
    """Whether the two groups can share releases, said in their own terms.

    A baseline contains the cohort by construction and the reader has to know
    that before reading the gap. Two named groups usually do not overlap, but
    nothing guarantees it, so the possibility is stated rather than assumed
    away -- and neither is described as containing the other.
    """
    versus = finding.thesis.versus
    if versus is not None and versus.is_baseline:
        return "The comparison group includes the releases selected here."
    return "The two groups are selected independently."


def _group_name(finding: ResearchFinding) -> str:
    """What to call the right-hand group on its tile.

    "A typical release" is only honest for a group that is every release. When
    the reader named two groups, each tile carries its own name -- calling one
    of them typical would present it as the standard the other is judged
    against, which is not what was asked.
    """
    versus = finding.thesis.versus
    if versus is not None and versus.is_baseline:
        return "A typical release"
    return finding.comparison.label if finding.comparison else "The other group"


def _scalar_figure(finding: ResearchFinding) -> str:
    """One number, standing on its own, with what it was measured over.

    No difference tile and no baseline column: there is nothing to compare
    against, and an empty second tile would read as a measurement that failed.
    """
    if finding.value is None:
        return ""
    outcome = finding.thesis.outcome
    if finding.matched_count is not None:
        meta = (
            f"{finding.matched_count} of {finding.sample_size} releases "
            f"&middot; {escape(outcome.describe())}"
        )
    else:
        meta = (
            f"across {finding.sample_size} releases &middot; "
            f"{escape(outcome.describe())}"
        )
    return (
        '<div class="compare compare-single">'
        '<div class="compare-tile">'
        f'<span class="compare-label">{escape(finding.cohort.label)}</span>'
        f'<span class="compare-value">{escape(finding.format_value(finding.value))}</span>'
        f'<span class="compare-meta">{meta}</span>'
        "</div></div>"
    )


def _ranked_rows(finding: ResearchFinding) -> str:
    """The releases themselves, in the order asked for.

    The ranked table is the answer here rather than a supporting detail, so it
    sits where the comparison tiles would otherwise be. Each row links to the
    release, because "the largest range was 7.09%" invites the next question,
    which is "what happened that day?".
    """
    rows = finding.rows
    if not rows:
        return ""
    body = "".join(
        f"<tr><td class=\"rank\">{index}</td>"
        f'<td><a href="/event/{escape(row.event_id)}">'
        f"{row.event_date:%d %b %Y}</a></td>"
        f'<td class="rank-value">'
        f"{escape(finding.metric.format(row.value or 0.0))}</td>"
        f"{_return_cell(row.returns.get(finding.thesis.outcome.horizon))}</tr>"
        for index, row in enumerate(rows, start=1)
    )
    horizon = finding.thesis.outcome.horizon
    return (
        '<table class="cohort-table ranked">'
        f"<thead><tr><th>#</th><th>Release</th>"
        f"<th>{escape(horizon)} {escape(finding.metric.label)}</th>"
        f"<th>{escape(horizon)} return</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f'<p class="chart-note">Ranked across {finding.sample_size} releases '
        "with usable data at this horizon.</p>"
    )


def _interpretation(paragraphs: tuple[str, ...]) -> str:
    """What the result suggests, in prose, directly under the headline.

    Unlabelled as to its origin. Whether these sentences were written by a
    template or a model, they describe the same facts and are held to the same
    rules, and a badge saying which would only invite the reader to trust one
    of them less than the numbers warrant.
    """
    if not paragraphs:
        return ""
    body = "".join(f"<p>{escape(text)}</p>" for text in paragraphs)
    return (
        '<section class="interpretation">'
        '<h3 class="finding-sub">What this suggests</h3>'
        f"{body}</section>"
    )


def _tile(label: str, value: str, n: int, detail: str, measure: str) -> str:
    """One figure with what it was measured over."""
    css = "pos" if value.startswith("+") else "neg" if value.startswith("-") else ""
    extra = f" &middot; {escape(detail)}" if detail else ""
    return (
        '<div class="compare-tile">'
        f'<span class="compare-label">{escape(label)}</span>'
        f'<span class="compare-value {css}">{escape(value)}</span>'
        f'<span class="compare-meta">{escape(measure)} &middot; '
        f"n={n}{extra}</span></div>"
    )


def _tile_detail(
    finding: ResearchFinding, metrics: object, matched: int | None, size: int
) -> str:
    """The count beneath a tile's figure, chosen by what the figure is.

    A share gets the fraction it came from, because 75% of eight and 75% of
    eighty are the same number and different evidence. A median return gets the
    hit rate, which is a genuinely separate fact about the same releases. A
    median range gets neither: it is never negative, so "31/31 positive" would
    be true, useless, and read as a finding.
    """
    if matched is not None:
        return finding.describe_count(matched, size)
    if finding.result_in_metric_units and finding.metric.key == "return_pct":
        up = getattr(metrics, "up_count", None)
        if up is not None:
            return f"{up}/{size} positive"
    return ""


def _difference_tile(finding: ResearchFinding) -> str:
    """The gap, with the subtraction spelled out in the groups' own names.

    "cohort minus baseline" is only true when one side is every release. With
    two named groups it would reintroduce exactly the framing the labels above
    it were written to avoid, and a reader could not tell which way the
    subtraction ran.
    """
    difference = finding.difference_pp
    if difference is None:
        return ""
    value, css = _signed(difference, suffix=" pp")
    left = _truncate(finding.cohort.label, 24)
    right = _truncate(_group_name(finding), 24)
    return (
        '<div class="compare-tile compare-diff">'
        '<span class="compare-label">Difference</span>'
        f'<span class="compare-value {css}">{value}</span>'
        f'<span class="compare-meta">{escape(left)} minus '
        f"{escape(right)}</span>"
        "</div>"
    )


def _comparison_chart(finding: ResearchFinding) -> str:
    """Two bars: the cohort against a typical release.

    The single most important picture here, because the cohort figure alone
    means nothing. Reuses the report's diverging bar chart rather than drawing
    a second kind of bar.
    """
    if finding.value is None or finding.comparison_value is None:
        return ""
    items = [
        BarItem(
            label=_truncate(finding.cohort.label),
            value=finding.value,
            tooltip=f"{finding.cohort.label}: n={finding.sample_size}",
        ),
        BarItem(
            label=_truncate(_group_name(finding)),
            value=finding.comparison_value,
            tooltip=(
                f"{_group_name(finding)}: n={finding.comparison_sample_size}"
            ),
        ),
    ]
    outcome = finding.thesis.outcome
    return (
        '<h3 class="finding-sub">'
        f"{escape(_truncate(finding.cohort.label, 40))} against "
        f"{escape(_truncate(_group_name(finding), 40))}</h3>"
        f"{diverging_bar_chart(items, quantity=finding.result_quantity)}"
        f'<p class="chart-note">{escape(outcome.describe().capitalize())} '
        f"for each group. {escape(_overlap_note(finding))}</p>"
    )


def _distribution(finding: ResearchFinding) -> str:
    """How the qualifying releases were spread, not just where their middle was.

    Plots the queried metric, so a question about trading range shows the
    spread of ranges rather than the spread of returns.

    Two things it must not do, both of which it used to. It must not mark a
    result that is not a value of the metric: a frequency of 75 is a share of
    releases, and drawing it on an axis of returns presents it as a 75% move.
    And it is skipped entirely for a frequency, because the figure the reader
    came for is not on this chart at all -- the honest version marks the
    *threshold* across the distribution, which is a different chart and not
    this iteration's work.
    """
    if finding.thesis.outcome.aggregation is Aggregation.FREQUENCY:
        return ""

    clean = [
        observation.value
        for observation in finding.observations
        if observation.value is not None
    ]
    if len(clean) < 2:
        return ""

    # Marked only when the result is a value of this metric. A ranked query has
    # no central figure, and a count is a number of releases.
    marker = (
        finding.value
        if finding.shape is not ResultShape.RANKED and finding.result_in_metric_units
        else None
    )
    return (
        f'<h3 class="finding-sub">Distribution of {escape(finding.metric.label)} '
        "across the qualifying releases</h3>"
        f"{distribution_chart(clean, median=marker, quantity=finding.metric.label)}"
    )


def _observations(finding: ResearchFinding) -> str:
    """Every release that qualified, collapsed."""
    if not finding.observations:
        return ""

    horizons = list(OUTCOME_HORIZONS)
    heads = "".join(f"<th>{escape(h)}</th>" for h in horizons)
    rows = []
    for observation in finding.observations:
        cells = "".join(
            _return_cell(observation.returns.get(h)) for h in horizons
        )
        why = " &middot; ".join(
            f"{escape(k)}: {escape(v)}" for k, v in observation.conditions.items()
        )
        rows.append(
            f"<tr><td>{observation.event_date:%d %b %Y}</td>{cells}"
            f'<td class="why">{why}</td></tr>'
        )

    return (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">The releases behind this'
        f'</span><span class="disclosure-hint">{len(rows)} qualifying</span>'
        "</summary>"
        '<div class="disclosure-body"><table class="cohort-table">'
        f"<thead><tr><th>Release</th>{heads}<th>Why it qualified</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
    )


def _full_statistics(finding: ResearchFinding) -> str:
    """Everything the engine measured, for a reader who wants it."""
    return (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">View full statistics</span>'
        '<span class="disclosure-hint">every horizon, for each group'
        "</span></summary>"
        '<div class="disclosure-body">'
        f"{render_sample_badge(finding.cohort)}"
        f"{render_cohort_table(finding.cohort, REACTION_WINDOWS)}"
        f"{_comparison_statistics(finding)}"
        "</div></details>"
    )


def _comparison_statistics(finding: ResearchFinding) -> str:
    """The other group's full statistics, when the question had one."""
    if finding.comparison is None:
        return ""
    return (
        f'<h3 class="finding-sub">{escape(_group_name(finding))}</h3>'
        f"{render_sample_badge(finding.comparison)}"
        f"{render_cohort_table(finding.comparison, REACTION_WINDOWS)}"
    )


def _methodology(finding: ResearchFinding) -> str:
    """How this was tested, in research terms rather than implementation ones."""
    thesis = finding.thesis
    rows = [
        ("Asset", f"{asset_label(thesis.asset)} ({thesis.asset})"),
        ("Event", thesis.event_phrase),
        (
            "Condition",
            " and ".join(c.describe() for c in thesis.conditions) or "none",
        ),
        ("Measure", f"{thesis.outcome.describe()}, from the release price"),
    ]
    if thesis.versus is not None:
        rows.append(("Compared with", thesis.versus.describe(thesis.event_phrase)))
    rows += [
        (
            "Anchoring",
            "the last candle that closed at or before each boundary, so no "
            "measurement contains information from after the instant it "
            "describes",
        ),
        ("Candles", f"{finding.spec.interval.value} from Binance"),
        (
            "Releases used",
            f"{finding.sample_size} with a usable reading at this horizon",
        ),
        # Stated because it is the thing that makes "show me the top five
        # instead" free: the same measurement answers median, rank and
        # frequency, and the fingerprint is how a reader can tell.
        ("Specification", finding.spec.fingerprint),
    ]
    items = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in rows
    )
    excluded = _excluded(finding)
    return (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">How was this tested?</span>'
        '<span class="disclosure-hint">method, sample and exclusions</span>'
        "</summary>"
        f'<div class="disclosure-body"><dl class="assumptions mono">{items}</dl>'
        f"{excluded}</div></details>"
    )


def _excluded(finding: ResearchFinding) -> str:
    if not finding.excluded:
        return ""
    items = "".join(
        f"<li>{count} &mdash; {escape(reason)}</li>"
        for reason, count in sorted(finding.excluded.items(), key=lambda kv: -kv[1])
    )
    return (
        "<p><strong>Releases not in the cohort.</strong> Each for a different "
        "reason, counted rather than hidden:</p>"
        f'<ul class="findings-list">{items}</ul>'
    )


def _return_cell(value: float | None) -> str:
    if value is None:
        return '<td class="na">&mdash;</td>'
    text, css = _signed(value)
    return f'<td class="{css}">{text}</td>'


def _signed(value: float, *, suffix: str = "%") -> tuple[str, str]:
    rounded = round(value, 2)
    if rounded == 0:
        return f"0.00{suffix}", ""
    return f"{rounded:+.2f}{suffix}", "pos" if rounded > 0 else "neg"


def _truncate(text: str, limit: int = 28) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


RESEARCH_SCRIPT = """
(function () {
  const form = document.getElementById('research-form');
  if (!form) return;
  const box = document.getElementById('question');
  const thread = document.getElementById('thread');
  const ask = document.getElementById('ask');

  // The thesis on screen travels with the next question, so a follow-up
  // modifies it instead of being parsed from nothing.
  function currentThesis() {
    const cards = thread.querySelectorAll('script.thesis');
    return cards.length ? cards[cards.length - 1].textContent : '';
  }

  function scrollToEnd() {
    thread.lastElementChild?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function post(url, body) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return response.text();
  }

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const question = box.value.trim();
    if (!question) return;
    ask.disabled = true;
    const pending = document.createElement('div');
    pending.className = 'turn turn-status';
    pending.innerHTML = '<span class="spinner"></span>Reading the question\\u2026';
    thread.insertAdjacentHTML('beforeend',
      '<div class="turn turn-ask"><p></p></div>');
    thread.lastElementChild.querySelector('p').textContent = question;
    thread.appendChild(pending);
    box.value = '';
    scrollToEnd();
    try {
      const html = await post('/api/research/interpret',
        { question: question, thesis: currentThesis() });
      pending.outerHTML = html;
    } catch (e) {
      pending.outerHTML = '<div class="error"><h3>Could not reach the server</h3><p>'
        + String(e) + '</p></div>';
    } finally {
      ask.disabled = false;
      scrollToEnd();
    }
  });

  thread.addEventListener('click', async (ev) => {
    const edit = ev.target.closest('[data-edit]');
    if (edit) {
      const card = edit.closest('.turn');
      const asked = card.previousElementSibling?.querySelector('p');
      box.value = asked ? asked.textContent : '';
      box.focus();
      return;
    }

    const run = ev.target.closest('[data-run]');
    if (!run) return;
    const card = run.closest('.turn');
    const thesis = card.querySelector('script.thesis')?.textContent;
    if (!thesis) return;
    run.disabled = true;
    const pending = document.createElement('div');
    pending.className = 'turn turn-status';
    pending.innerHTML = '<span class="spinner"></span>Measuring against real candles\\u2026';
    card.insertAdjacentElement('afterend', pending);
    scrollToEnd();
    try {
      pending.outerHTML = await post('/api/research/run', { thesis: thesis });
    } catch (e) {
      pending.outerHTML = '<div class="error"><h3>Could not reach the server</h3><p>'
        + String(e) + '</p></div>';
    } finally {
      run.disabled = false;
      scrollToEnd();
    }
  });

  for (const example of document.querySelectorAll('.example')) {
    example.addEventListener('click', () => {
      box.value = example.dataset.q;
      form.requestSubmit();
    });
  }
})();
"""


RESEARCH_STYLE = """
.research { max-width: 760px; margin: 0 auto; }
.research-intro { margin: 24px 0 20px; }
.research-intro h1 { font-size: 26px; margin: 0 0 8px; letter-spacing: -0.02em; }
.research-sub { color: var(--text-secondary); font-size: 14px; margin: 0;
                max-width: 62ch; line-height: 1.6; }
.research-form { display: flex; gap: 10px; align-items: flex-end; margin-bottom: 12px; }
.research-form textarea {
  flex: 1; padding: 11px 13px; border: 1px solid var(--grid); border-radius: 9px;
  background: var(--surface-1); color: var(--text-primary);
  font: inherit; font-size: 15px; resize: vertical; min-height: 46px;
}
.research-form textarea:focus { outline: 2px solid var(--accent);
                                outline-offset: -1px; border-color: transparent; }
.examples { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 28px; }
.example { padding: 6px 12px; border: 1px solid var(--grid); border-radius: 99px;
           background: transparent; color: var(--text-secondary);
           font: inherit; font-size: 12.5px; cursor: pointer; }
.example:hover { border-color: var(--accent); color: var(--text-primary); }

.turn { margin-bottom: 18px; }
.turn-ask p { margin: 0; padding: 10px 14px; background: var(--surface-2);
              border-radius: 10px; font-size: 15px; display: inline-block;
              max-width: 100%; }
.turn-status { color: var(--text-muted); font-size: 13px; }
.spec-head { font-size: 16px; font-weight: 600; margin: 0 0 12px;
             letter-spacing: -0.01em; }
.spec-card { border: 1px solid var(--grid); border-radius: 10px; overflow: hidden; }
.spec-row { display: flex; justify-content: space-between; gap: 18px;
            padding: 10px 16px; border-bottom: 1px solid var(--grid); font-size: 14px; }
.spec-row:last-child { border-bottom: none; }
.spec-label { color: var(--text-muted); }
.spec-value { color: var(--text-primary); text-align: right; }
.spec-assume { font-size: 12.5px; color: var(--text-muted); margin: 10px 0 0; }
.spec-actions { display: flex; gap: 10px; align-items: center; margin-top: 14px; }
.btn-quiet { background: transparent; border: 1px solid var(--grid);
             border-radius: 7px; padding: 9px 16px; color: var(--text-secondary);
             font: inherit; font-size: 14px; cursor: pointer; }
.btn-quiet:hover { color: var(--text-primary); border-color: var(--accent); }
.turn-unsupported, .turn-clarify { border-left: 3px solid var(--accent);
                                   padding-left: 14px; }

.compare { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
           border: 1px solid var(--grid); border-radius: 10px; overflow: hidden;
           margin: 14px 0; }
.compare-tile { padding: 14px 16px; border-right: 1px solid var(--grid); }
.compare-tile:last-child { border-right: none; }
.compare-diff { background: var(--surface-2); }
.compare-label { display: block; font-size: 12px; color: var(--text-muted); }
.compare-value { display: block; font-size: 26px; font-weight: 600; margin: 3px 0;
                 letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.compare-value.pos { color: var(--up); }
.compare-value.neg { color: var(--down); }
.compare-meta { display: block; font-size: 11.5px; color: var(--text-muted); }
.finding-sub { font-size: 13px; color: var(--text-muted); font-weight: 500;
               margin: 20px 0 6px; }
.cohort-table td.why { text-align: left; color: var(--text-muted); font-size: 12px; }
/* The reading sits between the headline and the figures, so it is set as
   prose rather than as a panel: slightly wider leading, a measure short enough
   to read in one pass, and no border competing with the comparison below. */
.interpretation { margin: 4px 0 20px; }
.interpretation .finding-sub { margin-top: 14px; }
.interpretation p { margin: 0 0 10px; max-width: 64ch; font-size: 14.5px;
                    line-height: 1.62; color: var(--text); }
.interpretation p:last-child { margin-bottom: 0; }
/* A single figure has no second column to balance against, so it is left
   aligned with the prose above rather than stretched across the card. */
.compare-single { grid-template-columns: minmax(0, 320px); }
.cohort-table.ranked td.rank { color: var(--text-muted); width: 2.5em; }
.cohort-table.ranked td.rank-value { font-weight: 600;
                                     font-variant-numeric: tabular-nums; }
.cohort-table.ranked a { color: inherit; text-decoration: none;
                         border-bottom: 1px solid var(--grid); }
.cohort-table.ranked a:hover { border-bottom-color: var(--text-muted); }

@media (max-width: 680px) {
  .compare { grid-template-columns: 1fr; }
  .compare-tile { border-right: none; border-bottom: 1px solid var(--grid); }
  .compare-tile:last-child { border-bottom: none; }
  .research-form { flex-direction: column; align-items: stretch; }
}
"""


__all__ = [
    "render_anchor_needed",
    "render_event_provenance",
    "render_unresolved_event",
    "EXAMPLE_QUESTIONS",
    "RESEARCH_SCRIPT",
    "RESEARCH_STYLE",
    "render_clarification",
    "render_finding",
    "render_question",
    "render_research_page",
    "render_spec_card",
    "render_unsupported",
]
