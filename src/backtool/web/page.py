"""The single page shell: controls, and a slot for results.

Server-rendered HTML with a small amount of vanilla JavaScript -- no build step,
no framework, no package manager. The page has one form and one results area;
React would add a toolchain and a second language for that.

Report styling is imported from :mod:`backtool.reporting.html` rather than
duplicated, so a result rendered in the browser is byte-identical to one saved
to a file.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from backtool.core.types import Interval
from backtool.reporting.html import _STYLE as REPORT_STYLE

_PAGE_STYLE = """
.masthead { margin-bottom: 8px; }
.masthead h1 { font-size: 22px; margin: 0 0 2px; }
.masthead p { margin: 0; color: var(--text-secondary); font-size: 14px; }
.tabs { display: flex; gap: 4px; margin: 20px 0 0; }
.tab {
  padding: 8px 16px; border: 1px solid var(--grid); border-bottom: none;
  border-radius: 8px 8px 0 0; background: var(--surface-0);
  color: var(--text-secondary); cursor: pointer; font-size: 14px;
  font-family: inherit;
}
.tab[aria-selected="true"] {
  background: var(--surface-1); color: var(--text-primary); font-weight: 500;
}
.tab:disabled { opacity: 0.45; cursor: not-allowed; }
.panel { background: var(--surface-1); border: 1px solid var(--grid);
         border-radius: 0 10px 10px 10px; padding: 20px; }
.panel[hidden] { display: none; }
label { display: block; font-size: 13px; color: var(--text-muted); margin-bottom: 5px; }
input[type=text], textarea, select {
  width: 100%; padding: 9px 11px; border: 1px solid var(--grid); border-radius: 7px;
  background: var(--surface-0); color: var(--text-primary);
  font: inherit; font-size: 14px;
}
textarea { font-family: ui-monospace, Consolas, monospace; font-size: 13px; resize: vertical; }
input:focus, textarea:focus, select:focus {
  outline: 2px solid var(--accent); outline-offset: -1px; border-color: transparent;
}
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
       gap: 14px; margin-bottom: 14px; }
.field { margin-bottom: 14px; }
.hint { font-size: 12px; color: var(--text-muted); margin-top: 5px; }
.actions { display: flex; align-items: center; gap: 14px; margin-top: 18px; }
button.run {
  background: var(--accent); color: #fff; border: none; border-radius: 7px;
  padding: 10px 22px; font: inherit; font-size: 14px; font-weight: 500; cursor: pointer;
}
button.run:hover { filter: brightness(1.08); }
button.run:disabled { opacity: 0.55; cursor: progress; }
.checkbox { display: flex; align-items: center; gap: 7px; font-size: 13px;
            color: var(--text-secondary); }
.checkbox input { width: auto; }
.notice { background: var(--surface-2); border-radius: 8px; padding: 11px 14px;
          font-size: 13px; color: var(--text-secondary); margin-top: 14px; }
.status { color: var(--text-muted); font-size: 13px; }
#results { margin-top: 32px; }
#results:empty { display: none; }
.error { border-left: 3px solid var(--down); background: var(--surface-2);
         border-radius: 0 8px 8px 0; padding: 14px 18px; margin-top: 24px; }
.error h3 { margin: 0 0 6px; font-size: 15px; color: var(--down); }
.error p { margin: 0; color: var(--text-secondary); font-size: 14px; }
.spinner {
  width: 13px; height: 13px; border: 2px solid var(--grid);
  border-top-color: var(--accent); border-radius: 50%;
  display: inline-block; vertical-align: -2px; margin-right: 7px;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
"""

_SCRIPT = """
const form = document.getElementById('study-form');
const results = document.getElementById('results');
const button = document.getElementById('run');
const status = document.getElementById('status');
const modeInput = document.getElementById('mode');

for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    if (tab.disabled) return;
    const mode = tab.dataset.mode;
    modeInput.value = mode;
    for (const other of document.querySelectorAll('.tab')) {
      other.setAttribute('aria-selected', String(other === tab));
    }
    for (const panel of document.querySelectorAll('.panel')) {
      panel.hidden = panel.dataset.mode !== mode;
    }
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Fetching candles and computing\\u2026';
  results.innerHTML = '';

  try {
    const response = await fetch('/api/analyse', {
      method: 'POST',
      body: new FormData(form),
    });
    results.innerHTML = await response.text();
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    results.innerHTML =
      '<div class="error"><h3>Could not reach the server</h3><p>' +
      String(error) + '</p></div>';
  } finally {
    button.disabled = false;
    status.textContent = '';
  }
});
"""


def render_page(*, ai_available: bool) -> str:
    """Render the full page.

    Args:
        ai_available: Whether an API key is configured. When false the Ask tab
            is disabled with an explanation rather than hidden -- a missing
            feature the user can see and fix beats one that silently is not
            there.
    """
    intervals = "".join(
        f'<option value="{interval.value}"'
        f'{" selected" if interval is Interval.M5 else ""}>{interval.value}</option>'
        for interval in Interval
    )

    # Whichever tab starts selected must also start visible. Deriving both from
    # one value stops them disagreeing -- the panels were previously hardcoded
    # hidden, so with no API key the right tab was selected and showed nothing.
    initial_mode = "ask" if ai_available else "build"
    ask_panel = _ask_panel(ai_available, visible=initial_mode == "ask")
    build_panel = _build_panel(intervals, visible=initial_mode == "build")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>backtool &mdash; event study research</title>
<style>{REPORT_STYLE}{_PAGE_STYLE}</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <h1>backtool</h1>
    <p>Deterministic event-study research. Every number is computed from
       exchange candles; the AI plans and explains, and calculates nothing.</p>
  </div>

  <form id="study-form">
    <input type="hidden" name="mode" id="mode" value="{initial_mode}">
    <div class="tabs" role="tablist">
      <button type="button" class="tab" data-mode="ask" role="tab"
              aria-selected="{str(ai_available).lower()}"
              {"" if ai_available else "disabled"}>Ask a question</button>
      <button type="button" class="tab" data-mode="build" role="tab"
              aria-selected="{str(not ai_available).lower()}">Build a study</button>
    </div>
    {ask_panel}
    {build_panel}

    <div class="actions">
      <button type="submit" class="run" id="run">Run study</button>
      <span class="status" id="status"></span>
    </div>
  </form>

  <div id="results"></div>
</div>
<script>{_SCRIPT}</script>
</body>
</html>
"""


def _ask_panel(ai_available: bool, *, visible: bool) -> str:
    hidden = "" if visible else " hidden"
    disabled_notice = (
        ""
        if ai_available
        else (
            '<div class="notice"><strong>Not configured.</strong> Set '
            "<code>ANTHROPIC_API_KEY</code> in <code>.env</code> and restart to "
            "enable plain-language questions. Everything under <em>Build a "
            "study</em> works without it.</div>"
        )
    )
    return f"""
    <div class="panel" data-mode="ask"{hidden}>
      <div class="field">
        <label for="question">Research question</label>
        <input type="text" id="question" name="question"
               placeholder="How does BTC behave in the hour after FOMC?"
               {"" if ai_available else "disabled"}>
        <p class="hint">The planner turns this into an explicit specification and
           shows you every definition it chose, before running.</p>
      </div>
      {disabled_notice}
    </div>"""


def _build_panel(intervals: str, *, visible: bool) -> str:
    hidden = "" if visible else " hidden"
    return f"""
    <div class="panel" data-mode="build"{hidden}>
      <div class="row">
        <div>
          <label for="event_type">Event type</label>
          <select id="event_type" name="event_type">
            <option value="FOMC" selected>FOMC</option>
          </select>
        </div>
        <div>
          <label for="count">Events</label>
          <input type="text" id="count" name="count" value="20" inputmode="numeric">
        </div>
        <div>
          <label for="interval">Interval</label>
          <select id="interval" name="interval">{intervals}</select>
        </div>
      </div>
      <div class="field">
        <label for="windows">Windows &mdash; one per line, <code>name:start:end</code></label>
        <textarea id="windows" name="windows" rows="4"
          placeholder="pre_event_24h:-24h:0h&#10;post_event_1h:0h:1h"></textarea>
        <p class="hint">Leave empty for the defaults: 24h and 1h either side of
           the announcement. Offsets take a unit &mdash; m, h, or d.</p>
      </div>
      <label class="checkbox">
        <input type="checkbox" name="explain" value="1"> Explain the results with AI
      </label>
    </div>"""


def render_error(message: str) -> str:
    """Render a failure as markup, so the client never branches on content type."""
    return (
        '<div class="error"><h3>Could not run that study</h3>'
        f"<p>{escape(message)}</p></div>"
    )
