"""The Historical Scenarios section: how price behaved under each outcome.

Same machinery as Similar Setups -- releases split into named cohorts, each
getting the identical stat block -- with a different splitting rule. Setups
split on what the market was doing beforehand; scenarios split on what the
release itself said.

No calculation happens here. Grouping is
:func:`~backtool.research.cohort.split_study` and the numbers are
:func:`~backtool.research.aggregate.aggregate_window`, which is the only place
in the project that produces a statistic.

What is *not* in a cohort is rendered as carefully as what is. Five of the 27
CPI releases cannot be classified -- four have no verified consensus, one has
sources that disagree -- and a page reporting on 22 without saying so would be
claiming a completeness it does not have.
"""

from __future__ import annotations

import datetime as dt
from xml.sax.saxutils import escape

from backtool.config import Settings
from backtool.core.time import UTC
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events import load_events
from backtool.events.releases import (
    SCENARIO_ORDER,
    SCENARIO_RULES,
    ConsensusMethod,
    ReleaseValues,
    load_releases,
    scenario_gap,
)
from backtool.research.cohort import Cohort, split_study
from backtool.research.results import EventResult, StudyResult
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.cohort_view import render_cohort_table, render_sample_badge

#: How many past releases the scenario split considers. Wide enough to cover
#: the whole supplied dataset; releases without values simply go uncounted.
SCENARIO_CANDIDATES = 40

#: Windows quoted in the per-release inspection table.
INSPECT_WINDOWS = ("post_event_1h", "post_event_24h")


class ScenarioSplit:
    """Cohorts plus everything that did not make it into one.

    Carried together because the page must show both. The exclusions are not
    an error state -- they are a property of a hand-verified dataset, and the
    reasons differ in kind.
    """

    def __init__(
        self,
        cohorts: list[Cohort],
        *,
        study: StudyResult | None,
        values: dict[str, ReleaseValues],
        examined: int,
    ) -> None:
        self.cohorts = cohorts
        self.study = study
        self.values = values
        self.examined = examined

    @property
    def classified(self) -> int:
        return sum(cohort.sample_size for cohort in self.cohorts)

    def exclusions(self) -> dict[str, int]:
        """Counted by reason, in the reader's terms."""
        if self.study is None:
            return {}
        tally: dict[str, int] = {}
        for event in self.study.events:
            values = self.values.get(event.event_id)
            reason = (
                values.exclusion_reason()
                if values is not None
                else "no published values recorded for this release"
            )
            if reason:
                tally[reason] = tally.get(reason, 0) + 1
        return tally


def build_scenarios(
    *,
    event_type: EventType,
    symbol: str,
    as_of: dt.datetime,
    interval: Interval = Interval.M5,
) -> ScenarioSplit:
    """Split past releases of this type into outcome cohorts."""
    now = dt.datetime.now(tz=UTC)
    values = load_releases()
    rule = SCENARIO_RULES[event_type]

    def classify(event: EventResult) -> tuple[str, str, str] | None:
        return rule(values.get(event.event_id))

    # Nothing classifiable means no cohorts, and running a forty-release study
    # to render an empty section would be forty candle fetches for nothing.
    if not any(rule(value) is not None for value in values.values()):
        return ScenarioSplit([], study=None, values=values, examined=0)

    spec = ResearchSpec(
        symbol=symbol,
        event_type=event_type,
        event_count=SCENARIO_CANDIDATES,
        interval=interval,
        windows=REACTION_WINDOWS,
        as_of=as_of,
    )
    with MarketDataService.from_settings(Settings.from_env()) as service:
        study = run_study(spec, service, load_events(event_type), now=now)

    cohorts = split_study(study, classify)
    # Fixed display order rather than first-seen, so the tabs do not reshuffle
    # when the dataset changes.
    rank = {key: i for i, key in enumerate(SCENARIO_ORDER)}
    cohorts.sort(key=lambda c: rank.get(c.key, len(rank)))

    return ScenarioSplit(
        cohorts, study=study, values=values, examined=len(study.events)
    )


def render_scenarios_section(
    split: ScenarioSplit,
    *,
    event_type: EventType,
    symbol: str,
) -> str:
    """The section body, as a fragment."""
    if not split.cohorts:
        return (
            '<div class="panel-card empty-state">'
            f"<p>{escape(scenario_gap(event_type))}</p>"
            '<p class="muted">The split runs on every page load. It produces '
            "nothing because there is nothing to split on, not because it is "
            "unimplemented.</p></div>"
        )

    tabs = "".join(
        f'<button type="button" class="cohort-tab" role="tab" '
        f'data-cohort="{index}" aria-selected="{str(index == 0).lower()}">'
        f'{escape(cohort.label)} <span class="tab-n">{cohort.sample_size}</span>'
        "</button>"
        for index, cohort in enumerate(split.cohorts)
    )

    panels = "".join(
        f'<div class="cohort-panel" data-cohort="{index}"'
        f'{"" if index == 0 else " hidden"}>'
        f"{render_sample_badge(cohort)}"
        f'<p class="chart-note">Reaction of <span class="mono">{escape(symbol)}'
        "</span> across these releases, measured from the release price at each "
        "horizon.</p>"
        f"{render_cohort_table(cohort, REACTION_WINDOWS)}"
        f"{_releases_table(cohort, split)}"
        "</div>"
        for index, cohort in enumerate(split.cohorts)
    )

    return (
        f'<div class="cohort-tabs" role="tablist">{tabs}</div>'
        f'<div class="panel-card">{panels}{_coverage(split)}</div>'
    )


def _releases_table(cohort: Cohort, split: ScenarioSplit) -> str:
    """Every release in this cohort, with the figures that put it there.

    Collapsed, but present for every cohort. A reader who doubts that a release
    belongs in "cooler than expected" can read the consensus, the actual and
    the gap, and check the arithmetic themselves -- which is the whole reason
    the classification rule is a comparison rather than a score.
    """
    if split.study is None:
        return ""

    by_id = {event.event_id: event for event in split.study.events}
    rows = []
    for event_id in cohort.event_ids:
        values = split.values.get(event_id)
        event = by_id.get(event_id)
        if values is None or event is None:
            continue

        surprise = values.surprise_pp
        gap = (
            f'<td class="{_sign(surprise)}">{surprise:+.1f} pp</td>'
            if surprise is not None
            else '<td class="na">&mdash;</td>'
        )
        reactions = "".join(
            _reaction_cell(event, window) for window in INSPECT_WINDOWS
        )
        flag = (
            ' <span class="derived-flag" title="No source printed this '
            'consensus directly; it was derived from a stated gap.">derived</span>'
            if values.consensus_method is ConsensusMethod.DERIVED
            else ""
        )
        date = values.release_date.strftime("%d %b %Y") if values.release_date else "&mdash;"
        rows.append(
            f"<tr><td>{escape(date)}</td>"
            f"<td>{_pct(values.previous_yoy)}</td>"
            f"<td>{_pct(values.consensus_yoy)}{flag}</td>"
            f"<td>{_pct(values.actual_yoy)}</td>"
            f"{gap}{reactions}</tr>"
        )

    if not rows:
        return ""

    return (
        '<details class="disclosure inline">'
        '<summary><span class="disclosure-title">The releases in this group</span>'
        f'<span class="disclosure-hint">{len(rows)} with the figures that '
        "classified them</span></summary>"
        '<div class="disclosure-body">'
        '<table class="cohort-table"><thead><tr><th>Release</th><th>Previous</th>'
        "<th>Expected</th><th>Actual</th><th>Surprise</th>"
        "<th>1h</th><th>24h</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
        "</div></details>"
    )


def _reaction_cell(event: EventResult, window: str) -> str:
    metrics = event.windows.get(window)
    if metrics is None or not metrics.is_usable or metrics.return_pct is None:
        return '<td class="na">&mdash;</td>'
    value = round(metrics.return_pct, 2)
    if value == 0:
        return "<td>0.00%</td>"
    return f'<td class="{"pos" if value > 0 else "neg"}">{value:+.2f}%</td>'


def _coverage(split: ScenarioSplit) -> str:
    """What the cohorts do not cover, by reason.

    Each reason is a different fact. "No consensus was published anywhere we
    could verify" and "two outlets published different consensus figures" are
    not the same gap, and one count covering both would hide that.
    """
    exclusions = split.exclusions()
    if not exclusions:
        return ""

    items = "".join(
        f"<li>{count} &mdash; {escape(reason)}</li>"
        for reason, count in sorted(exclusions.items(), key=lambda kv: -kv[1])
    )
    return (
        f'<p class="chart-note">{split.classified} of {split.examined} examined '
        "releases are in a group above. The rest are not, for these reasons:</p>"
        f'<ul class="findings-list">{items}</ul>'
        '<p class="chart-note">Excluded releases stay in the dataset and remain '
        "available for the historical reaction and similar-setup sections; they "
        "simply cannot be sorted by a surprise that is unknown or disputed.</p>"
    )


def _pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "&mdash;"


def _sign(value: float) -> str:
    rounded = round(value, 2)
    if rounded == 0:
        return ""
    return "pos" if rounded > 0 else "neg"


SCENARIO_SCRIPT = """
(function () {
  const root = document.getElementById('slot-scenarios');
  if (!root) return;
  root.addEventListener('click', (ev) => {
    const tab = ev.target.closest('.cohort-tab');
    if (!tab) return;
    const chosen = tab.dataset.cohort;
    for (const other of root.querySelectorAll('.cohort-tab')) {
      other.setAttribute('aria-selected', String(other.dataset.cohort === chosen));
    }
    for (const panel of root.querySelectorAll('.cohort-panel')) {
      panel.hidden = panel.dataset.cohort !== chosen;
    }
  });
})();
"""


__all__ = [
    "SCENARIO_CANDIDATES",
    "SCENARIO_SCRIPT",
    "ScenarioSplit",
    "build_scenarios",
    "render_scenarios_section",
]
