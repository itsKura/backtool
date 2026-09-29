"""Tests for sample-quality bands, relaxation priority, and setup timing.

These cover the three corrections that are about *how a result is presented*
rather than what it is. Each protects against a failure that leaves the numbers
correct and the reader misled: a 100% rate over five releases that reads like
certainty, a sample relaxed on its most important condition without saying so,
and a reading taken three weeks early presented as the state the release will
arrive into.
"""

from __future__ import annotations

import datetime as dt

import pytest
from tests.unit.test_setups_and_scenarios import NOW, _flat_then, _study

from backtool.research.aggregate import aggregate_window
from backtool.research.cohort import build_cohort, sample_quality
from backtool.research.regime import (
    MATCHING_METHODOLOGY,
    PRIMARY_DIMENSIONS,
    RELAXATION_ORDER,
    SECONDARY_DIMENSIONS,
    Trend,
    compute_setup,
)
from backtool.research.setups import match_setups
from backtool.research.snapshots import current_rules, load_snapshots, record_snapshot
from backtool.research.windows import REACTION_WINDOWS
from backtool.web.cohort_view import render_cohort_table, render_setup
from backtool.web.setups_panel import PRE_EVENT_HOURS, setup_framing


class TestSampleQualityBands:
    @pytest.mark.parametrize(
        ("count", "band"),
        [
            (1, "very limited"),
            (4, "very limited"),
            (5, "limited"),
            (9, "limited"),
            (10, "moderate"),
            (19, "moderate"),
            (20, "adequate"),
            (50, "adequate"),
        ],
    )
    def test_boundaries_are_the_specified_ones(self, count: int, band: str) -> None:
        assert sample_quality(count).value == band

    def test_only_an_adequate_sample_escapes_a_warning(self) -> None:
        assert not sample_quality(20).needs_warning
        assert sample_quality(19).needs_warning

    def test_no_confidence_score_is_invented(self) -> None:
        """A band is a word. A percentage here would be a statistic nobody
        computed, and the most authoritative-looking number on the page."""
        study = _study([1.0] * 5)
        cohort = build_cohort(
            study, key="k", label="L", description="d", members=list(study.events)
        )
        assert cohort.caveat is not None
        assert "%" not in cohort.caveat
        assert "confiden" not in cohort.caveat.lower()
        assert "probab" not in cohort.caveat.lower()


class TestCountsBesidePercentages:
    def test_a_small_sample_shows_the_fraction(self) -> None:
        study = _study([1.0, 2.0, 3.0, 4.0, 5.0])
        cohort = build_cohort(
            study, key="k", label="L", description="d", members=list(study.events)
        )
        assert "5/5 (100%)" in render_cohort_table(cohort, REACTION_WINDOWS)

    def test_counts_come_from_the_aggregate_not_from_the_rate(self) -> None:
        """Recovering a count by rounding a percentage produces the wrong
        integer often enough to matter over a small sample."""
        study = _study([1.0, -1.0, 1.0])
        agg = aggregate_window(study, "post_event_1h")
        assert agg.up_count == 2
        assert agg.down_count == 1
        assert agg.up_count + agg.down_count + agg.flat_count == agg.sample_size

    def test_threshold_counts_agree_with_their_rates(self) -> None:
        study = _study([0.2, -0.6, 1.4, -2.0, 0.1])
        agg = aggregate_window(study, "post_event_1h")
        assert agg.move_over_0_5_count == 3
        assert agg.move_over_1_count == 2
        assert agg.move_over_1_count / agg.sample_size * 100 == agg.move_over_1_pct


class TestRelaxationPriority:
    def test_secondary_conditions_drop_before_primary_ones(self) -> None:
        order = {dim: i for i, dim in enumerate(RELAXATION_ORDER)}
        assert max(order[d] for d in SECONDARY_DIMENSIONS) < min(
            order[d] for d in PRIMARY_DIMENSIONS
        )

    def test_the_tiers_are_the_specified_ones(self) -> None:
        assert set(PRIMARY_DIMENSIONS) == {"trend_7d", "vol_regime"}
        assert set(SECONDARY_DIMENSIONS) == {"trend_30d", "ma_position"}

    def test_the_note_names_what_still_matches(self) -> None:
        """A note carrying only what was dropped reads as a disclaimer; one
        carrying only what held hides the cost."""
        study = _study([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        base = compute_setup(_flat_then(0.0), NOW)
        history = {
            event.event_id: base.model_copy(
                update={
                    "trend_7d": Trend.BULLISH,
                    "trend_30d": Trend.BULLISH if i == 0 else Trend.BEARISH,
                }
            )
            for i, event in enumerate(study.events)
        }

        match = match_setups(
            study, history[study.events[0].event_id], history, target_sample=5
        )
        note = match.relaxation_note()
        assert note is not None
        assert "Still matching:" in note
        assert "BTC 7-day trend" in note
        assert "BTC 30-day trend" in note

    def test_methodology_is_published_for_the_disclosure(self) -> None:
        labels = {label for label, _ in MATCHING_METHODOLOGY}
        assert {"relaxation", "point-in-time", "match rule"} <= labels
        text = " ".join(value for _, value in MATCHING_METHODOLOGY)
        assert "secondary conditions dropped first" in text
        assert "no distance score" in text


class TestSetupFraming:
    def test_a_past_release_reads_as_settled_history(self) -> None:
        title, note = setup_framing(as_of=NOW, event_time=NOW - dt.timedelta(hours=1))
        assert title == "Market setup before the release"
        assert note == ""

    def test_within_two_days_it_is_the_pre_event_setup(self) -> None:
        title, note = setup_framing(as_of=NOW, event_time=NOW + dt.timedelta(hours=10))
        assert title == "Pre-event setup"
        assert "in 10 hours" in note

    def test_weeks_out_it_is_only_a_reading_of_today(self) -> None:
        """Calling a reading taken three weeks early a 'pre-event setup' dresses
        a current observation up as a prediction."""
        title, note = setup_framing(as_of=NOW, event_time=NOW + dt.timedelta(days=21))
        assert title == "Current market setup"
        assert "can change before then" in note

    def test_the_boundary_is_forty_eight_hours(self) -> None:
        assert PRE_EVENT_HOURS == 48
        inside, _ = setup_framing(
            as_of=NOW, event_time=NOW + dt.timedelta(hours=PRE_EVENT_HOURS)
        )
        outside, _ = setup_framing(
            as_of=NOW, event_time=NOW + dt.timedelta(hours=PRE_EVENT_HOURS + 1)
        )
        assert inside == "Pre-event setup"
        assert outside == "Current market setup"

    def test_the_reading_carries_the_instant_it_was_taken(self) -> None:
        html = render_setup(
            compute_setup(_flat_then(5.0), NOW), title="Current market setup", note="n"
        )
        assert "as of" in html
        assert f"{NOW:%d %b %Y}" in html


class TestSnapshots:
    def test_records_and_reloads_a_pre_release_reading(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        setup = compute_setup(_flat_then(5.0), NOW)
        stored = record_snapshot(
            tmp_path / "snap.db",
            event_id="CPI-2026-10-14",
            event_time=NOW + dt.timedelta(days=3),
            symbol="BTCUSDT",
            setup=setup,
            captured_at=NOW,
        )
        assert stored is not None
        assert stored.hours_before == pytest.approx(72.0)

        [loaded] = load_snapshots(tmp_path / "snap.db", "CPI-2026-10-14")
        assert loaded.setup == setup
        assert loaded.rules_match_current

    def test_refuses_to_record_after_the_release(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The point is a record of what was shown beforehand; a reading taken
        afterwards is not that, and storing it with a negative lead would let it
        pass as one."""
        assert (
            record_snapshot(
                tmp_path / "s.db",
                event_id="X",
                event_time=NOW - dt.timedelta(hours=1),
                symbol="BTCUSDT",
                setup=compute_setup(_flat_then(0.0), NOW),
                captured_at=NOW,
            )
            is None
        )

    def test_stores_the_thresholds_it_was_classified_under(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A recorded 'bearish' means nothing a year later without the number
        that made it bearish, and these thresholds are research choices."""
        stored = record_snapshot(
            tmp_path / "s.db",
            event_id="X",
            event_time=NOW + dt.timedelta(days=1),
            symbol="BTCUSDT",
            setup=compute_setup(_flat_then(0.0), NOW),
            captured_at=NOW,
        )
        assert stored is not None
        assert set(stored.rules) == set(current_rules())

    def test_no_snapshot_exists_for_an_unknown_release(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert load_snapshots(tmp_path / "missing.db", "NOPE") == []
