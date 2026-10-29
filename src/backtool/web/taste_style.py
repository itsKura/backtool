"""An experiment: the `data-terminal` style from tasteskill, applied to Backtool.

Dealt from `library/app.json` in the taste-skill package, which describes seven
app-screen styles distilled from studied interfaces. ``data-terminal`` is the
one for *markets and monitoring, read in long sessions* -- which is what this
is -- and its stated move is:

    Every selection is the same 1px accent line; signs, not colour, on numbers.

Kept as one appended sheet rather than edits spread through six modules, for
two reasons. It is an experiment, so it has to be removable in one line. And
the existing stylesheets already express almost everything through custom
properties, so re-pointing the tokens does most of the work on its own -- what
is left is the style's signature, which genuinely is an override.

What this deliberately does not do
----------------------------------
The style's shell is an icon rail down the left. Backtool's top nav works, and
swapping the shell would touch the markup of every page for a change nobody
asked for. The style's own instruction is to redraw only for a reason you can
state; this is the reverse, and the reason is that the experiment is meant to
be reversible.
"""

from __future__ import annotations

#: Fonts the style names. Public Sans for text, a monospace for every figure --
#: the style's first identity move is "every price, count and time in a
#: monospace, digits that stack". Loaded from Google Fonts, which is the one
#: network dependency this sheet adds; the fallbacks are the stack the app
#: already used, so the page is readable before they arrive.
TASTE_FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Public+Sans:wght@400;500;600;700"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&display=swap"
)

TASTE_STYLE = """
/* ---------------------------------------------------------------------------
   tasteskill `data-terminal` -- experiment layer.
   Delete this sheet and its <link> to return to the original look.
   --------------------------------------------------------------------------- */

:root {
  /* Tokens as dealt. The names are Backtool's existing ones, so the 286
     var() references across the app re-point without being touched. */
  --surface-0: #171719;   /* ground   */
  --surface-1: #131217;   /* surface  -- cards sit darker than the ground */
  --surface-2: #212023;   /* raised   */
  --text-primary: #ffffff;
  --text-secondary: #bdbdc0;
  --text-muted: #8a8c92;
  --grid: #222224;        /* border   */
  --rule: #1b1a1f;
  --accent: #4a8ad6;
  --control: #3a3a3e;

  /* The style deals green/red. Backtool used blue/red on purpose -- a comment
     in the original sheet explains that blue and red stay separable under the
     common red-green deficiencies, which green and red do not. The dealt
     values are here so the experiment is honest about what the style says;
     --up is the one place this sheet keeps the original decision. */
  --up: #4a8ad6;
  --down: #e5566f;
  --taste-positive: #44a282;

  --radius-card: 6px;
  --radius-control: 999px;
}

body, button, input, textarea, select {
  font-family: "Public Sans", Inter, system-ui, -apple-system, sans-serif;
}

/* Every figure in a monospace, digits that stack. The style calls this its
   first identity move, and it is the change you notice before any colour. */
.mono, .prep-figure-value, .prep-val, .prep-vs, .prep-rows td, .prep-table td,
.cohort-table td, .cohort-table th, .event-time, .spec-value, .stat-value,
.prep-tick, .expect-value, time[data-local] {
  font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}

/* Type scale: title 26, section 20 in muted grey, body 13, small 12. */
h1 { font-size: 26px; letter-spacing: -0.02em; font-weight: 600; }
h2 { font-size: 20px; font-weight: 400; color: var(--text-secondary); }
h3 { font-size: 16px; font-weight: 600; }
body { font-size: 13px; }
.muted, .hint, .section-sub, .prep-figure-note { font-size: 12px; }

/* --- the signature -------------------------------------------------------
   Every selection is the same 1px accent line, exactly as wide as its label,
   on the nearest existing edge. No filled pills, no coloured segments. */

.asset-tab[aria-selected="true"],
.cohort-tab[aria-selected="true"],
.chart-range button[aria-selected="true"],
.window-tab[aria-selected="true"],
.chart-switch button[aria-selected="true"],
.tab[aria-selected="true"] {
  background: transparent;
  color: var(--text-primary);
  border-radius: 0;
  box-shadow: inset 0 -1px 0 0 var(--accent);
}

.nav a[aria-current="page"] {
  box-shadow: inset 0 -1px 0 0 var(--accent);
  padding-bottom: 3px;
}

.asset-tabs, .cohort-tabs, .chart-range {
  background: transparent;
  border: none;
  padding: 0;
  gap: 18px;
}

/* The one filled control on the page was a blue button. In this style the
   accent is never more than a line, so the button becomes an outline and the
   accent marks it on the same edge as every other selection. */
.btn {
  background: var(--surface-2);
  color: var(--text-primary);
  border: 1px solid var(--grid);
  border-radius: var(--radius-control);
  font-weight: 500;
}
.btn:hover { border-color: var(--control); }

/* --- signs, not colour, on numbers ---------------------------------------
   The style's rule: colour on every number stops meaning anything. Returns
   keep their sign in the text and lose their hue, except where a column's
   whole job is which way a thing moved -- the scenario medians, the reaction
   row, and a headline that printed negative. */
.prep-vs, .prep-rows .prep-vs, .cohort-table td.pos, .cohort-table td.neg,
.cohort-table span.pos, .cohort-table span.neg {
  color: var(--text-secondary);
}
.prep-val.up, .prep-val.down,
.prep-figure-value.up, .prep-figure-value.down,
.prep-table td.up, .prep-table td.down {
  color: var(--text-primary);
}
.prep-summary .prep-table td.up, .prep-summary .prep-table td.down,
.prep-similar-scenarios .prep-table td.up,
.prep-similar-scenarios .prep-table td.down {
  color: var(--text-primary);
}
/* The exceptions the style allows. */
.prep-reaction .prep-figure-value.up { color: var(--up); }
.prep-reaction .prep-figure-value.down { color: var(--down); }
.prep-context .prep-figure-value.up { color: var(--up); }
.prep-context .prep-figure-value.down { color: var(--down); }

/* --- motion -------------------------------------------------------------
   Feedback only. Nothing performs an entrance: this page is opened twenty
   times on a release morning, and an entrance is a toll paid every time.

   Durations from the skill's table -- press 140ms, hover colour 180ms, panels
   150ms -- and ease-out everywhere something enters or leaves, because
   ease-in spends its first frames barely moving and reads as lag.

   Everything here lives inside the no-preference query, so the reduced page
   is the finished page rather than a degraded one. */
@media (prefers-reduced-motion: no-preference) {
  .btn, .asset-tab, .cohort-tab, .chart-range button, .prep-prepare,
  .prep-ask, .example, .nav a {
    transition: color 180ms ease, background-color 180ms ease,
                border-color 180ms ease, box-shadow 180ms ease;
  }

  /* Press gives, briefly. The only transform in the sheet, and it moves a
     control rather than any layout. */
  .btn:active, .prep-prepare:active, .example:active,
  .chart-range button:active {
    transform: translateY(1px);
    transition-duration: 140ms;
  }

  /* A panel swapped by the tabs crossfades instead of snapping. Seen many
     times in a session, so it is fast and small. */
  .cohort-panel, .prep-card details[open] > div {
    animation: taste-fade 150ms ease-out;
  }

  /* The spinner's replacement arrives rather than appearing. */
  .chart-canvas canvas, .prep-reaction, #slot-summary > * {
    animation: taste-fade 200ms ease-out;
  }

  /* A row arriving -- the one entrance the house rule allows, and only for
     the turn just appended, never the thread already on screen. */
  #thread > .turn:last-child {
    animation: taste-rise 200ms ease-out;
  }

  /* Bands are rebuilt on every zoom and the set genuinely changes, so this
     is a crossfade and not a resize. Driven by the class the chart toggles. */
  .chart-bands { transition: opacity 160ms ease-out; }
  .chart-bands.bands-settling { opacity: 0; }
}

@keyframes taste-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes taste-rise {
  from { opacity: 0; transform: translateY(3px); }
  to   { opacity: 1; transform: none; }
}

/* --- surfaces and density ------------------------------------------------ */

.panel-card, .prep-card, .event-card, .card, .panel, .chart-card,
.prep-figure, .research-form, .turn {
  border-radius: var(--radius-card);
  border-color: var(--grid);
}

.prep-cards { gap: 24px; }
.prep-figures { gap: 16px; }
.cohort-table td, .cohort-table th, .prep-table td, .prep-table th {
  padding-top: 10px;
  padding-bottom: 10px;
}

/* Rules between rows rather than borders around everything. */
.prep-table td, .prep-table th { border-bottom-color: var(--rule); }
.cohort-table tr + tr td { border-top: 1px solid var(--rule); }

/* Section titles sit quieter than the figures under them. */
.finding-sub, .section-sub { color: var(--text-muted); }

/* Status pills keep their meaning but lose the saturated fill. */
.badge, .prep-state, .cohort-quality {
  background: var(--surface-2);
  color: var(--text-secondary);
  border-radius: var(--radius-control);
  font-weight: 500;
}
.badge-high { background: #2c1a20; color: #e5566f; }
.badge-medium { background: #2a2416; color: #d9a441; }
.badge-low { background: #212023; color: #bdbdc0; }
"""
