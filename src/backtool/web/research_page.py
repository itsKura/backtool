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
from backtool.research.vocabulary import OUTCOME_HORIZONS, asset_label
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
    cohort = finding.cohort_metrics
    baseline = finding.baseline_metrics

    if cohort is None or cohort.sample_size == 0:
        return (
            '<div class="turn turn-result"><div class="panel-card">'
            f'<p class="finding-headline">{escape(finding.verdict())}</p>'
            f"{_interpretation(interpretation)}"
            f"{_excluded(finding)}</div></div>"
        )

    assert baseline is not None
    horizon = finding.thesis.outcome.horizon
    difference = finding.difference_pp

    headline = (
        f'<p class="finding-headline">{escape(finding.verdict())}</p>'
    )
    cohort_tile = _tile(
        finding.cohort.label,
        cohort.median_return_pct,
        cohort.sample_size,
        cohort.up_count,
        horizon,
    )
    baseline_tile = _tile(
        "A typical release",
        baseline.median_return_pct,
        baseline.sample_size,
        baseline.up_count,
        horizon,
    )
    numbers = (
        '<div class="compare">'
        f"{cohort_tile}{baseline_tile}{_difference_tile(difference)}"
        "</div>"
    )

    warnings = "".join(
        f'<p class="note">{escape(w)}</p>' for w in finding.warnings
    )

    return (
        '<div class="turn turn-result">'
        '<div class="panel-card">'
        f"{headline}"
        f"{_interpretation(interpretation)}"
        f"{numbers}{warnings}"
        f"{_comparison_chart(finding)}"
        f"{_distribution(finding)}"
        f"{_observations(finding)}"
        f"{_full_statistics(finding)}"
        f"{_methodology(finding)}"
        "</div></div>"
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


def _tile(label: str, median: float, n: int, up: int, horizon: str) -> str:
    value, css = _signed(median)
    return (
        '<div class="compare-tile">'
        f'<span class="compare-label">{escape(label)}</span>'
        f'<span class="compare-value {css}">{value}</span>'
        f'<span class="compare-meta">median {escape(horizon)} return &middot; '
        f"n={n} &middot; {up}/{n} positive</span></div>"
    )


def _difference_tile(difference: float | None) -> str:
    if difference is None:
        return ""
    value, css = _signed(difference, suffix=" pp")
    return (
        '<div class="compare-tile compare-diff">'
        '<span class="compare-label">Difference</span>'
        f'<span class="compare-value {css}">{value}</span>'
        '<span class="compare-meta">cohort median minus baseline median</span>'
        "</div>"
    )


def _comparison_chart(finding: ResearchFinding) -> str:
    """Two bars: the cohort against a typical release.

    The single most important picture here, because the cohort figure alone
    means nothing. Reuses the report's diverging bar chart rather than drawing
    a second kind of bar.
    """
    cohort, baseline = finding.cohort_metrics, finding.baseline_metrics
    if cohort is None or baseline is None:
        return ""
    items = [
        BarItem(
            label=_truncate(finding.cohort.label),
            value=cohort.median_return_pct,
            tooltip=f"{finding.cohort.label}: n={cohort.sample_size}",
        ),
        BarItem(
            label="Typical release",
            value=baseline.median_return_pct,
            tooltip=f"All usable releases: n={baseline.sample_size}",
        ),
    ]
    return (
        '<h3 class="finding-sub">Cohort against baseline</h3>'
        f"{diverging_bar_chart(items)}"
        '<p class="chart-note">Median return at the chosen horizon. The '
        "baseline includes the cohort.</p>"
    )


def _distribution(finding: ResearchFinding) -> str:
    """How the qualifying releases were spread, not just where their middle was."""
    values = [
        observation.returns.get(finding.thesis.outcome.horizon)
        for observation in finding.observations
    ]
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return ""
    cohort = finding.cohort_metrics
    median = cohort.median_return_pct if cohort else None
    return (
        '<h3 class="finding-sub">Distribution of the qualifying releases</h3>'
        f"{distribution_chart(clean, median=median)}"
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
        '<span class="disclosure-hint">every horizon, cohort and baseline'
        "</span></summary>"
        '<div class="disclosure-body">'
        f"{render_sample_badge(finding.cohort)}"
        f"{render_cohort_table(finding.cohort, REACTION_WINDOWS)}"
        f'<h3 class="finding-sub">Baseline</h3>'
        f"{render_sample_badge(finding.baseline)}"
        f"{render_cohort_table(finding.baseline, REACTION_WINDOWS)}"
        "</div></details>"
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
        ("Outcome", f"{thesis.outcome.horizon} return from the release price"),
        ("Baseline", thesis.baseline.describe(thesis.event_type)),
        (
            "Anchoring",
            "the last candle that closed at or before each boundary, so no "
            "measurement contains information from after the instant it "
            "describes",
        ),
        ("Candles", f"{finding.spec.interval.value} from Binance"),
        ("Releases used", f"{finding.cohort.sample_size} of {finding.baseline.sample_size} usable"),
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
