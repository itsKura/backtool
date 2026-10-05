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
from backtool.reporting.charts import BarItem, distribution_chart, diverging_bar_chart
from backtool.research.experiment import ResearchFinding
from backtool.research.thesis import Thesis
from backtool.research.vocabulary import OUTCOME_HORIZONS, ResultShape, asset_label
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
        '<div class="spec-actions">'
        '<button type="button" class="btn" data-run="1">Run research</button>'
        '<button type="button" class="btn-quiet" data-edit="1">Edit question</button>'
        "</div>"
        f'<script type="application/json" class="thesis">{thesis.model_dump_json()}'
        "</script></div>"
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
        "releases today. Everything else would be a guess dressed as a result."
        "</p></div>"
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

    horizon = finding.thesis.outcome.horizon
    label = f"{finding.thesis.outcome.aggregation.value} {horizon} {finding.metric.label}"
    tiles = (
        _tile(
            finding.cohort.label,
            finding.format_value(finding.value),
            finding.sample_size,
            cohort.up_count if cohort and finding.metric.key == "return_pct" else None,
            label,
        )
        + _tile(
            _group_name(finding),
            finding.format_value(finding.comparison_value),
            finding.comparison_sample_size,
            other.up_count
            if other and finding.metric.key == "return_pct"
            else None,
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


def _tile(label: str, value: str, n: int, up: int | None, measure: str) -> str:
    """One figure with what it was measured over.

    The hit rate appears only when the metric is a signed return. A trading
    range is never negative, so "31/31 positive" beside one would be true,
    useless, and read as a finding.
    """
    css = "pos" if value.startswith("+") else "neg" if value.startswith("-") else ""
    hits = f" &middot; {up}/{n} positive" if up is not None else ""
    return (
        '<div class="compare-tile">'
        f'<span class="compare-label">{escape(label)}</span>'
        f'<span class="compare-value {css}">{escape(value)}</span>'
        f'<span class="compare-meta">{escape(measure)} &middot; '
        f"n={n}{hits}</span></div>"
    )


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
        f"{diverging_bar_chart(items)}"
        f'<p class="chart-note">{escape(outcome.aggregation.value.capitalize())} '
        f"{escape(outcome.horizon)} {escape(finding.metric.label)} across each "
        f"group. {escape(_overlap_note(finding))}</p>"
    )


def _distribution(finding: ResearchFinding) -> str:
    """How the qualifying releases were spread, not just where their middle was.

    Plots the queried metric, so a question about trading range shows the
    spread of ranges rather than the spread of returns. The marker is the
    scalar result where there is one -- on a ranked query there is no central
    figure, and drawing a median the reader did not ask for would imply the
    answer was about the middle.
    """
    clean = [
        observation.value
        for observation in finding.observations
        if observation.value is not None
    ]
    if len(clean) < 2:
        return ""
    marker = finding.value if finding.shape is not ResultShape.RANKED else None
    return (
        f'<h3 class="finding-sub">Distribution of {escape(finding.metric.label)} '
        "across the qualifying releases</h3>"
        f"{distribution_chart(clean, median=marker)}"
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
        ("Event", f"{thesis.event_type.value} releases"),
        (
            "Condition",
            " and ".join(c.describe() for c in thesis.conditions) or "none",
        ),
        ("Measure", f"{thesis.outcome.describe()}, from the release price"),
    ]
    if thesis.versus is not None:
        rows.append(("Compared with", thesis.versus.describe(thesis.event_type)))
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
