"""Horizons past a day.

A question like "how did BTC do in the two years after a president took
office" asks for a window 730 times longer than anything the engine measured
before, and the obstacle was never the grammar -- ``parse_offset`` has handled
days all along. It was that a study carries *one* candle interval, and no
interval serves both ends of that range: two years at five minutes is 210,240
candles per event, and a five-minute window measured hourly has nothing inside
it.

So horizons come in two families, each at the interval its spans deserve, and a
thesis runs against the family its own horizon belongs to. What the single
family was for survives inside each: whichever horizon was asked about, the
neighbours are measured too.

The honest part matters as much as the arithmetic. Crypto is young, so a
two-year window is usually not measurable at all -- and that answer has to
arrive as a refusal rather than as however much of two years happens to exist.
That guard lives in :mod:`tests.unit.test_window_reach`; these tests are about
getting the question as far as it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.core.types import Interval
from backtool.research.vocabulary import (
    DEFAULT_HORIZON,
    LONG_HORIZON_INTERVAL,
    LONG_HORIZONS,
    OUTCOME_HORIZONS,
    SHORT_HORIZON_INTERVAL,
    interval_for,
    is_long_horizon,
    reaction_windows,
    window_for,
)
from backtool.research.windows import (
    LONG_WINDOWS,
    REACTION_WINDOWS,
    compact_duration,
    humanise_duration,
    parse_offset,
    window_title,
)
from backtool.web.event_page import (
    CHART_TIMEFRAMES,
    MAX_RAW_CANDLES,
    chart_default_timeframe,
    chart_span_for,
    chart_timeframes_for,
    timeframe_by_label,
)

LONG = ("7d", "30d", "90d", "180d", "1y", "2y")
SHORT = ("5m", "15m", "1h", "4h", "24h")


def spec_for(horizon: str):  # type: ignore[no-untyped-def]
    name = window_for(horizon)
    return next(w for w in reaction_windows(horizon) if w.name == name)


class TestTheVocabularyReachesTwoYears:
    @pytest.mark.parametrize("horizon", LONG)
    def test_each_long_horizon_has_a_window(self, horizon: str) -> None:
        assert horizon in OUTCOME_HORIZONS
        assert spec_for(horizon) in LONG_WINDOWS

    @pytest.mark.parametrize("horizon", SHORT)
    def test_the_short_horizons_are_untouched(self, horizon: str) -> None:
        assert spec_for(horizon) in REACTION_WINDOWS

    def test_the_default_is_still_a_day(self) -> None:
        """Changing it would silently re-scope every question that names no
        horizon."""
        assert DEFAULT_HORIZON == "24h"
        assert not is_long_horizon(DEFAULT_HORIZON)

    def test_two_years_is_seven_hundred_and_thirty_days(self) -> None:
        assert spec_for("2y").end_offset == dt.timedelta(days=730)

    def test_the_offsets_were_always_expressible(self) -> None:
        """The grammar already took days -- what was missing was everything
        downstream of it."""
        assert parse_offset("730d") == dt.timedelta(days=730)


class TestOneIntervalPerStudy:
    """The reason the families are separate rather than one long list."""

    @pytest.mark.parametrize("horizon", LONG)
    def test_long_horizons_run_hourly(self, horizon: str) -> None:
        assert interval_for(horizon) is LONG_HORIZON_INTERVAL is Interval.H1

    @pytest.mark.parametrize("horizon", SHORT)
    def test_short_horizons_run_at_five_minutes(self, horizon: str) -> None:
        assert interval_for(horizon) is SHORT_HORIZON_INTERVAL is Interval.M5

    def test_the_families_do_not_overlap(self) -> None:
        assert not {w.name for w in LONG_WINDOWS} & {w.name for w in REACTION_WINDOWS}

    def test_every_horizon_belongs_to_exactly_one_family(self) -> None:
        for horizon in OUTCOME_HORIZONS:
            assert (spec_for(horizon) in LONG_WINDOWS) is is_long_horizon(horizon)

    def test_two_years_of_five_minute_candles_is_what_this_avoids(self) -> None:
        """Not a style preference. The number is the argument."""
        span = spec_for("2y").end_offset
        at_5m = span / Interval.M5.duration
        at_1h = span / Interval.H1.duration
        assert at_5m > 200_000
        assert at_1h < 20_000

    def test_the_neighbours_are_measured_too(self) -> None:
        """What the single family was for, kept inside each: a reader looking
        at a two-year figure wants to know where it was at ninety days."""
        assert len(reaction_windows("2y")) == len(LONG_WINDOWS) > 1
        assert spec_for("90d") in reaction_windows("2y")


class TestTheLabelsRead:
    @pytest.mark.parametrize(
        ("horizon", "title"),
        [
            ("7d", "7 days after"),
            ("90d", "90 days after"),
            ("1y", "1 year after"),
            ("2y", "2 years after"),
        ],
    )
    def test_a_window_says_what_it_is(self, horizon: str, title: str) -> None:
        assert window_title(spec_for(horizon)) == title

    def test_a_year_is_not_reported_as_days(self) -> None:
        assert humanise_duration(dt.timedelta(days=365)) == "1 year"
        assert compact_duration(dt.timedelta(days=730)) == "2y"

    def test_a_day_is_still_a_day(self) -> None:
        """Adding years to the unit ladder must not re-render what was there."""
        assert humanise_duration(dt.timedelta(hours=24)) == "24 hours"
        assert compact_duration(dt.timedelta(hours=24)) == "24h"


class TestTheChartFollowsTheWindow:
    """A ninety-day box on a chart spanning a day either side is a rectangle
    wider than the canvas with none of the window inside it.

    The control used to be zoom -- three widths of the same five-minute
    candles. It is now the candle size, which is what a trading chart's
    toolbar changes. The invariant that mattered survives the swap: whatever
    is offered, the window being shaded fits inside the view.
    """

    def test_the_view_always_contains_the_window(self) -> None:
        """The one carried over from the zoom ladder, and the reason that
        ladder was tested at all."""
        for horizon in OUTCOME_HORIZONS:
            window = spec_for(horizon)
            measured = window.end_offset - window.start_offset
            for timeframe in chart_timeframes_for(window):
                assert chart_span_for(window, timeframe) >= measured, (
                    horizon,
                    timeframe.label,
                )

    def test_a_short_window_offers_every_timeframe(self) -> None:
        offered = [t.label for t in chart_timeframes_for(spec_for("24h"))]
        assert offered == ["5m", "15m", "1h", "4h"]

    def test_a_long_window_drops_the_fine_ones(self) -> None:
        """Five minutes across two years is 420,000 candles: not a slow
        request, an unanswerable one."""
        offered = [t.label for t in chart_timeframes_for(spec_for("2y"))]
        assert "5m" not in offered
        assert "15m" not in offered
        assert offered == ["1h", "4h"]

    def test_something_is_always_offered(self) -> None:
        for horizon in OUTCOME_HORIZONS:
            assert chart_timeframes_for(spec_for(horizon))

    def test_no_offered_timeframe_asks_for_an_absurd_fetch(self) -> None:
        for horizon in OUTCOME_HORIZONS:
            window = spec_for(horizon)
            for timeframe in chart_timeframes_for(window):
                span = chart_span_for(window, timeframe)
                raw = 2 * span.total_seconds() / timeframe.fetch.duration.total_seconds()
                assert raw <= MAX_RAW_CANDLES, (horizon, timeframe.label, raw)

    def test_a_short_window_opens_on_the_finest(self) -> None:
        assert chart_default_timeframe(spec_for("24h")).label == "5m"

    def test_a_long_window_opens_on_a_coarse_one(self) -> None:
        """Opening a two-year study on five-minute candles would be a wall of
        hairlines even if it were affordable."""
        assert chart_default_timeframe(spec_for("2y")).label == "4h"

    def test_the_default_is_always_one_of_the_offered(self) -> None:
        for horizon in OUTCOME_HORIZONS:
            window = spec_for(horizon)
            assert chart_default_timeframe(window) in chart_timeframes_for(window)

    def test_every_timeframe_fetches_an_interval_the_data_layer_has(self) -> None:
        """4h is not fetchable and must not pretend to be. It is drawn from
        hourly candles, which is a display decision, and the statistics beside
        it stay hourly."""
        for timeframe in CHART_TIMEFRAMES:
            assert timeframe.fetch in set(Interval)
        coarse = next(t for t in CHART_TIMEFRAMES if t.label == "4h")
        assert coarse.fetch is Interval.H1
        assert coarse.grain == "4h"

    def test_an_unknown_label_falls_back_rather_than_failing(self) -> None:
        """A stale link or a hand-edited query string takes the chart's
        default, because a picture beside the numbers is not worth a 500."""
        window = spec_for("24h")
        assert timeframe_by_label("nonsense", window) is chart_default_timeframe(window)
        assert timeframe_by_label("", window) is chart_default_timeframe(window)

    def test_a_label_the_window_withdrew_falls_back_too(self) -> None:
        window = spec_for("2y")
        assert timeframe_by_label("5m", window) is chart_default_timeframe(window)


class TestAStudyRunsOnTheRightFamily:
    """The seam: the thesis decides, not a constant."""

    def test_the_spec_takes_its_interval_and_windows_from_the_horizon(self) -> None:
        from backtool.core.types import EventType
        from backtool.research.spec import ResearchSpec
        from backtool.research.thesis import Outcome, Thesis

        for horizon in ("24h", "90d"):
            thesis = Thesis(
                question="q",
                asset="BTCUSDT",
                event_type=EventType.CPI,
                outcome=Outcome(horizon=horizon),
            )
            spec = ResearchSpec(
                symbol=thesis.asset,
                event_type=thesis.event_type,
                interval=interval_for(thesis.outcome.horizon),
                windows=reaction_windows(thesis.outcome.horizon),
            )
            assert spec.interval is interval_for(horizon)
            assert window_for(horizon) in {w.name for w in spec.windows}

    def test_the_long_family_starts_at_the_event(self) -> None:
        """So the fetch needs a candle before it to anchor against -- which
        ``get_candles`` supplies as warmup. Pinned because a family that
        started after the event would price its own first boundary off a
        candle that had not closed."""
        assert all(w.start_offset == dt.timedelta(0) for w in LONG_WINDOWS)

    def test_the_longest_window_sets_the_fetch_span(self) -> None:
        from backtool.research.spec import ResearchSpec

        spec = ResearchSpec(symbol="BTCUSDT", windows=reaction_windows("2y"))
        assert spec.latest_offset == dt.timedelta(days=730)


def test_the_long_horizons_are_exactly_the_ones_declared() -> None:
    """One list, not two that can drift."""
    assert set(LONG) == set(LONG_HORIZONS)
    assert {w.name for w in LONG_WINDOWS} == {OUTCOME_HORIZONS[h] for h in LONG}


class TestASampleOfOneReadsAsOne:
    """"across 1 releases", on a study of one presidential inauguration.

    Two faults in one sentence, and long horizons are what made them ordinary.
    The plural was never conditioned on the count, because a sample of one was
    a degenerate case nobody saw until a two-year window made it the usual
    result; and the noun stayed "release" on a concept resolved from the web,
    which the event-noun work had fixed everywhere except the sentence a reader
    reads first.
    """

    def finding(self, *, dynamic: bool, size: int):  # type: ignore[no-untyped-def]
        from tests.unit.test_research_chart import cpi_thesis, finding_for

        thesis = cpi_thesis()
        if dynamic:
            from tests.fixtures.dynamic_events import eth_merge

            thesis = thesis.model_copy(
                update={"event_type": None, "resolution": eth_merge()}
            )
        dates = tuple(dt.date(2024, 1 + n, 11) for n in range(size))
        return finding_for(thesis, dates=dates).model_copy(
            update={"thesis": thesis, "sample_size": size}
        )

    def test_one_is_singular(self) -> None:
        assert "across 1 release." in self.finding(dynamic=False, size=1).verdict()

    def test_more_than_one_is_plural(self) -> None:
        assert "across 3 releases." in self.finding(dynamic=False, size=3).verdict()

    def test_a_resolved_concept_is_not_a_release(self) -> None:
        verdict = self.finding(dynamic=True, size=1).verdict()
        assert "across 1 event." in verdict
        assert "release" not in verdict

    def test_a_curated_family_still_says_release(self) -> None:
        """The noun follows the thesis, so curated output is untouched."""
        assert "release" in self.finding(dynamic=False, size=2).verdict()

    def test_the_counted_helper_agrees_with_the_verdict(self) -> None:
        finding = self.finding(dynamic=True, size=1)
        assert finding.counted(1) == "1 event"
        assert finding.counted(2) == "2 events"
