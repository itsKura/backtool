"""Tests for market-regime reading, setup matching, and scenario splitting.

The rules being protected here are the ones whose violation would be invisible
in the output: a volatility percentile that peeked at the future would look
exactly like one that did not, and a relaxed match that did not say so would
look exactly like an exact one.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backtool.core.types import EventType, Interval
from backtool.events import load_events
from backtool.events.releases import (
    ReleaseValues,
    classify_cpi,
    classify_fomc,
    load_releases,
)
from backtool.research.aggregate import aggregate_window
from backtool.research.cohort import build_cohort, split_study, unclassified_count
from backtool.research.metrics import Direction, WindowMetrics, WindowStatus
from backtool.research.regime import (
    MAPosition,
    Trend,
    compute_setup,
    daily_closes,
)
from backtool.research.results import EventResult, StudyResult
from backtool.research.setups import match_setups
from backtool.research.spec import ResearchSpec
from backtool.research.windows import REACTION_WINDOWS

NOW = dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.UTC)


def _hourly(closes: list[float], *, end: dt.datetime) -> pd.DataFrame:
    """An hourly frame ending at ``end``, one row per hour."""
    times = [end - dt.timedelta(hours=len(closes) - i) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "open_time": [t - dt.timedelta(hours=1) for t in times],
            "close_time": pd.to_datetime(times, utc=True),
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def _flat_then(gain_pct: float, *, days: int = 400) -> pd.DataFrame:
    """A long flat series that moves by ``gain_pct`` over its final week."""
    hours = days * 24
    closes = [100.0] * (hours - 168)
    closes += list(np.linspace(100.0, 100.0 * (1 + gain_pct / 100), 168))
    return _hourly(closes, end=NOW)


class TestDailyCloses:
    def test_one_close_per_day(self) -> None:
        frame = _hourly([100.0] * 72, end=NOW)
        assert len(daily_closes(frame)) in (3, 4)

    def test_a_gap_is_left_as_a_gap(self) -> None:
        """Forward-filling an outage invents a flat day, which deflates every
        volatility figure that spans it."""
        frame = _hourly([100.0] * 48, end=NOW)
        frame = frame.drop(frame.index[10:34])  # remove a day's worth
        closes = daily_closes(frame)
        assert not closes.isna().any()


class TestTrendBuckets:
    @pytest.mark.parametrize(
        ("gain", "expected"),
        [(10.0, Trend.BULLISH), (0.5, Trend.NEUTRAL), (-10.0, Trend.BEARISH)],
    )
    def test_seven_day_return_drives_the_bucket(
        self, gain: float, expected: Trend
    ) -> None:
        setup = compute_setup(_flat_then(gain), NOW)
        assert setup.trend_7d is expected

    def test_threshold_is_exclusive_at_the_boundary(self) -> None:
        """+3.0% exactly is neutral, not bullish: the rule is '> 3', and a
        boundary that drifts would silently reclassify releases."""
        setup = compute_setup(_flat_then(3.0), NOW)
        assert setup.trend_7d is Trend.NEUTRAL


class TestMovingAverage:
    def test_requires_two_hundred_days(self) -> None:
        """A 140-day average labelled as a 200-day one is a quiet
        redefinition, and invisible in the output."""
        setup = compute_setup(_flat_then(5.0, days=150), NOW)
        assert setup.ma_200d is None
        assert setup.ma_position is None

    def test_position_follows_price(self) -> None:
        setup = compute_setup(_flat_then(20.0, days=400), NOW)
        assert setup.ma_position is MAPosition.ABOVE


class TestPointInTime:
    def test_nothing_after_the_instant_is_used(self) -> None:
        """The single most important rule here. A setup read at T must be
        identical whether or not the frame contains data after T."""
        full = _flat_then(10.0, days=400)
        moment = NOW - dt.timedelta(days=30)
        truncated = full[full["close_time"] < pd.Timestamp(moment)]

        assert compute_setup(full, moment) == compute_setup(truncated, moment)

    def test_a_candle_closing_exactly_at_the_event_is_excluded(self) -> None:
        """That candle contains the release; pricing the setup with it would
        use the reaction the setup is supposed to precede."""
        frame = _hourly([100.0] * (400 * 24), end=NOW)
        setup = compute_setup(frame, NOW)
        last_used = frame["close_time"].iloc[-1]
        assert setup.close is not None
        assert last_used <= pd.Timestamp(NOW)


def _metrics(name: str, return_pct: float) -> WindowMetrics:
    end = 100.0 * (1 + return_pct / 100)
    return WindowMetrics(
        name=name,
        start_utc=NOW,
        end_utc=NOW + dt.timedelta(hours=1),
        status=WindowStatus.OK,
        start_price=100.0,
        end_price=end,
        return_pct=return_pct,
        direction=(
            Direction.UP
            if return_pct > 0
            else Direction.DOWN
            if return_pct < 0
            else Direction.FLAT
        ),
        high=max(100.0, end),
        low=min(100.0, end),
        abs_move_pct=abs(return_pct),
        mfe_pct=abs(return_pct),
        mae_pct=-abs(return_pct),
        range_pct=abs(return_pct) * 2,
    )


def _study(returns: list[float]) -> StudyResult:
    spec = ResearchSpec(
        event_count=len(returns), windows=REACTION_WINDOWS, as_of=NOW
    )
    events = tuple(
        EventResult(
            event_id=f"CPI-2026-01-{i + 1:02d}",
            event_time_utc=NOW - dt.timedelta(days=30 * i),
            local_date=dt.date(2026, 1, i + 1),
            is_scheduled=True,
            windows={w.name: _metrics(w.name, r) for w in REACTION_WINDOWS},
        )
        for i, r in enumerate(returns)
    )
    return StudyResult(spec=spec, generated_at=NOW, events=events)


class TestThresholdFrequencies:
    def test_counts_absolute_moves_over_each_threshold(self) -> None:
        study = _study([0.2, -0.6, 1.4, -2.0, 0.1])
        agg = aggregate_window(study, "post_event_1h")
        assert agg.move_over_0_5_pct == 60.0  # 0.6, 1.4, 2.0
        assert agg.move_over_1_pct == 40.0  # 1.4, 2.0

    def test_median_range_is_not_the_mean(self) -> None:
        study = _study([0.1, 0.1, 0.1, 0.1, 10.0])
        agg = aggregate_window(study, "post_event_1h")
        assert agg.median_range_pct < agg.mean_range_pct


class TestCohorts:
    def test_sample_size_is_the_member_count(self) -> None:
        study = _study([1.0, -1.0, 2.0])
        cohort = build_cohort(
            study, key="k", label="L", description="d", members=list(study.events[:2])
        )
        assert cohort.sample_size == 2
        assert cohort.windows["post_event_1h"].sample_size == 2

    def test_a_tiny_cohort_is_reported_not_suppressed(self) -> None:
        """n=3 is shown as n=3. Hiding it leaves the reader assuming the group
        did not exist."""
        study = _study([1.0, 2.0, 3.0, 4.0])
        cohort = build_cohort(
            study, key="k", label="L", description="d", members=list(study.events[:3])
        )
        assert cohort.sample_size == 3
        assert cohort.caveat is not None
        assert "3" in cohort.caveat

    def test_declined_releases_are_counted_not_bucketed(self) -> None:
        """A release with no consensus is unknown, not 'in line'."""
        study = _study([1.0, 2.0, 3.0, 4.0])
        keep = {study.events[0].event_id, study.events[1].event_id}

        def classify(event: EventResult) -> tuple[str, str, str] | None:
            return ("k", "L", "d") if event.event_id in keep else None

        cohorts = split_study(study, classify)
        assert len(cohorts) == 1
        assert cohorts[0].sample_size == 2
        assert unclassified_count(study, classify) == 2


class TestSetupMatching:
    def _history(self, buckets: list[dict[str, str]], study: StudyResult) -> dict:
        out = {}
        for event, wanted in zip(study.events, buckets, strict=True):
            out[event.event_id] = compute_setup(_flat_then(0.0), NOW).model_copy(
                update={
                    "trend_7d": Trend(wanted["trend_7d"]),
                    "trend_30d": Trend(wanted["trend_30d"]),
                }
            )
        return out

    def test_relaxation_is_recorded_not_silent(self) -> None:
        study = _study([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        history = self._history(
            [
                {"trend_7d": "bullish", "trend_30d": "bullish"},
                {"trend_7d": "bullish", "trend_30d": "bearish"},
                {"trend_7d": "bullish", "trend_30d": "bearish"},
                {"trend_7d": "bullish", "trend_30d": "bearish"},
                {"trend_7d": "bullish", "trend_30d": "bearish"},
                {"trend_7d": "bearish", "trend_30d": "bullish"},
            ],
            study,
        )
        current = history[study.events[0].event_id]

        match = match_setups(study, current, history, target_sample=5)
        assert match.exact_count == 1
        assert match.was_relaxed
        assert match.relaxation_note() is not None
        assert "trend_30d" in match.relaxed
        # Every rung recorded, not just the used one.
        assert len(match.ladder) >= 2
        assert sum(1 for rung in match.ladder if rung.used) == 1

    def test_an_exact_match_is_never_labelled_relaxed(self) -> None:
        study = _study([1.0, 2.0, 3.0, 4.0, 5.0])
        history = self._history(
            [{"trend_7d": "bullish", "trend_30d": "bullish"}] * 5, study
        )
        match = match_setups(study, history[study.events[0].event_id], history)
        assert not match.was_relaxed
        assert match.relaxation_note() is None
        assert match.sample_size == 5

    def test_a_release_missing_a_dimension_does_not_match_on_it(self) -> None:
        """Treating an unmeasurable reading as 'whatever today is' would pad the
        sample with releases selected for having less data."""
        study = _study([1.0, 2.0])
        base = compute_setup(_flat_then(0.0), NOW)
        history = {
            study.events[0].event_id: base.model_copy(
                update={"trend_7d": Trend.BULLISH, "trend_30d": Trend.BULLISH}
            ),
            study.events[1].event_id: base.model_copy(
                update={"trend_7d": Trend.BULLISH, "trend_30d": None}
            ),
        }
        current = history[study.events[0].event_id]
        match = match_setups(study, current, history, target_sample=99)
        assert match.exact_count == 1


class TestScenarioClassification:
    """V1 uses a strict three-way comparison with no tolerance band."""

    def _values(self, actual: float | None, consensus: float | None) -> ReleaseValues:
        return ReleaseValues(
            event_id="CPI-2025-01-15",
            actual_yoy=actual,
            consensus_yoy=consensus,
            consensus_verified=consensus is not None,
        )

    def test_the_dataset_loads(self) -> None:
        """What the count has to be is "enough to split", not a literal.

        The file used to be typed by hand and 27 was a fact about it. It is
        now extended by `backtool refresh-cpi` as releases happen, so a literal
        count is a test that fails every month for the one reason that is not a
        regression -- and gets edited to the new number each time until nobody
        reads it.
        """
        values = load_releases()
        assert len(values) >= 27
        assert "CPI-2026-09-11" in values

    def test_every_row_is_well_formed(self) -> None:
        """What the count was standing in for. A row that parsed into nothing
        would pass a length check and break every cohort."""
        for event_id, entry in load_releases().items():
            assert entry.event_id == event_id
            assert entry.release_date is not None, event_id
            assert entry.reference_month, event_id
            assert entry.actual_yoy is not None, event_id

    def test_enough_releases_are_classifiable_to_split(self) -> None:
        """The property the dataset exists for. Rows without a verified
        consensus are expected -- the refresh adds them, since no public source
        publishes a forecast -- but the classified set must stay large enough
        to produce the three cohorts the page is built on."""
        values = load_releases()
        classified = [v for v in values.values() if classify_cpi(v) is not None]
        assert len(classified) >= 20
        assert {classify_cpi(v)[0] for v in classified} == {  # type: ignore[index]
            "hotter",
            "in_line",
            "cooler",
        }

    def test_strict_three_way_comparison(self) -> None:
        assert classify_cpi(self._values(3.0, 2.9))[0] == "hotter"  # type: ignore[index]
        assert classify_cpi(self._values(2.9, 2.9))[0] == "in_line"  # type: ignore[index]
        assert classify_cpi(self._values(2.8, 2.9))[0] == "cooler"  # type: ignore[index]

    def test_a_one_tenth_miss_is_not_in_line(self) -> None:
        """The old rule had a 0.1pp tolerance band. V1 does not: a reader
        looking at '3.0% vs 2.9% expected' reads that as a miss, and the
        classification has to agree with them."""
        assert classify_cpi(self._values(3.0, 2.9))[0] == "hotter"  # type: ignore[index]
        assert classify_cpi(self._values(2.8, 2.9))[0] == "cooler"  # type: ignore[index]

    def test_equal_magnitudes_classify_symmetrically(self) -> None:
        """3.0-2.9 and 2.8-2.9 are not equal and opposite in binary floating
        point. Comparing integer hundredths removes the question."""
        up = classify_cpi(self._values(3.0, 2.9))
        down = classify_cpi(self._values(2.8, 2.9))
        assert up is not None and down is not None
        assert up[0] == "hotter"
        assert down[0] == "cooler"

    def test_a_missing_consensus_is_unknown_not_in_line(self) -> None:
        assert classify_cpi(self._values(3.0, None)) is None
        assert classify_cpi(self._values(None, 2.9)) is None
        assert classify_cpi(None) is None

    def test_a_conflicting_consensus_is_excluded_by_default(self) -> None:
        """Reuters said 2.5% and CNBC said 2.4% for the same release. Those
        classify it differently, so including it would be choosing a source
        silently."""
        disputed = ReleaseValues(
            event_id="CPI-2025-06-11",
            actual_yoy=2.4,
            consensus_yoy=2.4,
            consensus_verified=True,
            consensus_conflict=True,
        )
        assert classify_cpi(disputed) is None
        assert classify_cpi(disputed, include_conflicts=True) is not None
        assert "disagree" in (disputed.exclusion_reason() or "")

    def test_unverified_consensus_does_not_classify(self) -> None:
        unverified = ReleaseValues(
            event_id="X", actual_yoy=3.0, consensus_yoy=2.9, consensus_verified=False
        )
        assert classify_cpi(unverified) is None

    def test_surprise_is_reported_in_percentage_points(self) -> None:
        assert self._values(3.0, 2.9).surprise_pp == pytest.approx(0.1)
        assert self._values(2.8, 2.9).surprise_pp == pytest.approx(-0.1)
        assert self._values(2.9, 2.9).surprise_pp == 0.0
        assert self._values(3.0, None).surprise_pp is None

    def test_fomc_does_not_borrow_the_cpi_rule(self) -> None:
        """A rate decision is a choice among three options, not a number that
        missed a forecast."""
        assert classify_fomc(self._values(5.5, 5.25)) is None


class TestScenarioSection:
    def _empty(self, event_type: EventType):  # type: ignore[no-untyped-def]
        from backtool.web.scenario_panel import ScenarioSplit

        return ScenarioSplit([], study=None, values={}, examined=0)

    def test_states_what_is_missing_per_event_type(self) -> None:
        from backtool.web.scenario_panel import render_scenarios_section

        fomc = render_scenarios_section(
            self._empty(EventType.FOMC), event_type=EventType.FOMC, symbol="BTCUSDT"
        )
        cpi = render_scenarios_section(
            self._empty(EventType.CPI), event_type=EventType.CPI, symbol="BTCUSDT"
        )
        assert "target rate" in fomc
        assert "consensus" in cpi
        assert cpi != fomc

    def test_the_gap_is_not_described_as_unimplemented(self) -> None:
        from backtool.web.scenario_panel import render_scenarios_section

        html = render_scenarios_section(
            self._empty(EventType.CPI), event_type=EventType.CPI, symbol="BTCUSDT"
        )
        for phrase in ("not wired up", "TODO", "coming soon"):
            assert phrase not in html


class TestScenarioDataset:
    """The supplied CPI dataset, checked as data rather than as code."""

    def test_every_row_maps_onto_a_calendar_release(self) -> None:
        """The dataset and the calendar were built independently and spell ids
        differently. A mismatch would silently drop rows from every cohort."""
        calendar = {e.event_id for e in load_events(EventType.CPI)}
        assert set(load_releases()) <= calendar

    def test_previous_chains_to_the_prior_release_actual(self) -> None:
        """A strong integrity signal: last month's published figure is this
        month's 'previous', so a transcription slip breaks the chain."""
        rows = sorted(
            load_releases().values(), key=lambda v: v.release_date or dt.date.min
        )
        for earlier, later in zip(rows, rows[1:], strict=False):
            if earlier.actual_yoy is None or later.previous_yoy is None:
                continue
            assert round(earlier.actual_yoy * 100) == round(later.previous_yoy * 100), (
                f"{earlier.event_id} -> {later.event_id}"
            )

    def test_classification_and_exclusion_account_for_every_row(self) -> None:
        values = load_releases()
        classified = sum(1 for v in values.values() if classify_cpi(v) is not None)
        excluded = sum(1 for v in values.values() if v.exclusion_reason() is not None)
        assert classified + excluded == len(values)

    def test_the_four_unverified_rows_are_never_classified(self) -> None:
        values = load_releases()
        for event_id in (
            "CPI-2026-08-12",
            "CPI-2024-12-11",
            "CPI-2024-08-14",
            "CPI-2024-06-12",
        ):
            assert classify_cpi(values[event_id]) is None

    def test_the_conflict_row_is_excluded_but_retained(self) -> None:
        values = load_releases()
        disputed = values["CPI-2025-06-11"]
        assert disputed.consensus_conflict
        assert classify_cpi(disputed) is None
        assert classify_cpi(disputed, include_conflicts=True) is not None
        assert "Reuters" in disputed.notes

    def test_the_derived_row_is_marked(self) -> None:
        from backtool.events.releases import ConsensusMethod

        values = load_releases()
        assert values["CPI-2024-10-10"].consensus_method is ConsensusMethod.DERIVED
        explicit = values["CPI-2025-02-12"].consensus_method
        assert explicit is ConsensusMethod.EXPLICIT

    def test_every_classified_row_has_a_source_url(self) -> None:
        for values in load_releases().values():
            if classify_cpi(values) is not None:
                assert values.consensus_source.startswith("http")
                assert values.actual_source.startswith("http")

    def test_all_actuals_are_first_release_verified(self) -> None:
        assert all(v.first_release_verified for v in load_releases().values())


class TestReactionWindows:
    def test_covers_the_specified_horizons(self) -> None:
        ends = {w.end_offset for w in REACTION_WINDOWS}
        assert ends == {
            dt.timedelta(minutes=5),
            dt.timedelta(minutes=15),
            dt.timedelta(hours=1),
            dt.timedelta(hours=4),
            dt.timedelta(hours=24),
        }

    def test_every_window_starts_at_the_release(self) -> None:
        """Each is measured from the release price, not chained off the last."""
        assert all(w.start_offset == dt.timedelta(0) for w in REACTION_WINDOWS)

    def test_the_shortest_window_fits_the_finest_interval(self) -> None:
        assert REACTION_WINDOWS[0].end_offset >= Interval.M5.duration
