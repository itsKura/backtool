"""Tests for the study runner and cross-event aggregation.

Aggregation is where a data problem becomes a false claim: a mean reported over
"20 events" that quietly used 17 reads exactly like an honest one. These tests
pin the accounting as hard as the arithmetic.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import build_candles

from backtool.core.time import UTC
from backtool.core.types import EventType, Interval
from backtool.data.candles import empty_candles
from backtool.data.service import MarketDataService
from backtool.data.storage.sqlite import SQLiteCandleRepository
from backtool.events.models import MarketEvent
from backtool.research.aggregate import MIN_RELIABLE_SAMPLE, aggregate_study, aggregate_window
from backtool.research.runner import run_study
from backtool.research.spec import ResearchSpec
from backtool.research.windows import WindowSpec

POST_1H = WindowSpec(name="post_event_1h", start="0h", end="1h")
AS_OF = dt.datetime(2024, 6, 1, tzinfo=UTC)


def make_event(day: int) -> MarketEvent:
    return MarketEvent(
        event_id=f"FOMC-2024-01-{day:02d}",
        event_type=EventType.FOMC,
        local_date=dt.date(2024, 1, day),
        local_time=dt.time(14, 0),
        timezone="America/New_York",
    )


class ScriptedSource:
    """Serves a chosen post-event return per event, or nothing at all.

    Prices are derived from each candle's close time relative to the event, not
    from its index, so the result does not depend on how much warmup the service
    happens to request.

    A candle closing at or before the event is priced 100; everything after it
    is priced ``100 * (1 + target/100)``. The anchor rule therefore reads 100 at
    the event and the stepped price at any later boundary, making the post-event
    return exactly ``target``.
    """

    BASE_PRICE = 100.0

    def __init__(self, returns_by_day: dict[int, float | None]) -> None:
        self.returns_by_day = returns_by_day
        self.calls = 0

    def fetch_klines(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        *,
        now: dt.datetime | None = None,
    ) -> pd.DataFrame:
        self.calls += 1

        target: float | None = None
        for day, value in self.returns_by_day.items():
            # 14:00 America/New_York in January is 19:00 UTC.
            event_utc = dt.datetime(2024, 1, day, 19, 0, tzinfo=UTC)
            if start <= event_utc < end:
                target = value
                break

        if target is None:
            return empty_candles(interval)

        event_utc = next(
            dt.datetime(2024, 1, day, 19, 0, tzinfo=UTC)
            for day, value in self.returns_by_day.items()
            if value is target and start <= dt.datetime(2024, 1, day, 19, 0, tzinfo=UTC) < end
        )
        stepped = self.BASE_PRICE * (1 + target / 100)

        count = int((end - start) / interval.duration)
        closes = [
            self.BASE_PRICE
            if start + (i + 1) * interval.duration <= event_utc
            else stepped
            for i in range(count)
        ]
        return build_candles(start, closes, interval)

    def close(self) -> None:
        pass


@pytest.fixture
def service(tmp_path: Path) -> MarketDataService:
    return MarketDataService(
        ScriptedSource({}), SQLiteCandleRepository(tmp_path / "candles.db")
    )


def study_with(returns: dict[int, float | None], tmp_path: Path):  # type: ignore[no-untyped-def]
    source = ScriptedSource(returns)
    service = MarketDataService(source, SQLiteCandleRepository(tmp_path / "candles.db"))
    spec = ResearchSpec(event_count=len(returns), windows=(POST_1H,), as_of=AS_OF)
    calendar = [make_event(day) for day in sorted(returns)]
    return run_study(spec, service, calendar, now=AS_OF), source


class TestRunner:
    def test_runs_every_selected_event(self, tmp_path: Path) -> None:
        study, _ = study_with({3: 1.0, 4: 2.0, 5: 3.0}, tmp_path)
        assert len(study.events) == 3
        assert [e.event_id for e in study.events] == [
            "FOMC-2024-01-03",
            "FOMC-2024-01-04",
            "FOMC-2024-01-05",
        ]

    def test_results_carry_the_spec(self, tmp_path: Path) -> None:
        """Numbers must never be separable from the assumptions behind them."""
        study, _ = study_with({3: 1.0}, tmp_path)
        assert study.spec.fingerprint
        assert study.spec.windows[0].name == "post_event_1h"

    def test_computes_the_expected_return(self, tmp_path: Path) -> None:
        study, _ = study_with({3: 2.5}, tmp_path)
        metrics = study.events[0].window("post_event_1h")
        assert metrics.is_usable
        assert metrics.return_pct == pytest.approx(2.5, abs=0.01)

    def test_events_without_data_are_reported_not_dropped(self, tmp_path: Path) -> None:
        """The whole point of the coverage accounting: a missing event stays
        visible in the output."""
        study, _ = study_with({3: 1.0, 4: None, 5: 3.0}, tmp_path)

        assert len(study.events) == 3, "an unusable event disappeared from the result"
        assert len(study.complete_events) == 2
        assert not study.events[1].is_complete
        assert study.events[1].window("post_event_1h").reason is not None

    def test_coverage_note_states_the_shortfall(self, tmp_path: Path) -> None:
        study, _ = study_with({3: 1.0, 4: None, 5: 3.0}, tmp_path)
        note = study.coverage_note()
        assert "3 event(s) selected of 3 requested" in note
        assert "2 with all windows usable" in note

    def test_future_events_are_never_selected(self, tmp_path: Path) -> None:
        source = ScriptedSource({3: 1.0})
        service = MarketDataService(
            source, SQLiteCandleRepository(tmp_path / "candles.db")
        )
        spec = ResearchSpec(event_count=10, windows=(POST_1H,), as_of=AS_OF)
        calendar = [make_event(3), make_event(4)]
        calendar.append(
            MarketEvent(
                event_id="FOMC-2030-01-01",
                event_type=EventType.FOMC,
                local_date=dt.date(2030, 1, 1),
                local_time=dt.time(14, 0),
                timezone="America/New_York",
            )
        )

        study = run_study(spec, service, calendar, now=AS_OF)
        assert "FOMC-2030-01-01" not in [e.event_id for e in study.events]


class TestAggregate:
    def test_mean_and_median_over_known_returns(self, tmp_path: Path) -> None:
        """Returns of 1, 2, 3, 4: mean 2.5, median 2.5."""
        study, _ = study_with({3: 1.0, 4: 2.0, 5: 3.0, 8: 4.0}, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert agg.sample_size == 4
        assert agg.mean_return_pct == pytest.approx(2.5, abs=0.02)
        assert agg.median_return_pct == pytest.approx(2.5, abs=0.02)
        assert agg.min_return_pct == pytest.approx(1.0, abs=0.02)
        assert agg.max_return_pct == pytest.approx(4.0, abs=0.02)

    def test_up_and_down_rates(self, tmp_path: Path) -> None:
        study, _ = study_with({3: 1.0, 4: 2.0, 5: -3.0, 8: -4.0}, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert agg.up_rate_pct == pytest.approx(50.0)
        assert agg.down_rate_pct == pytest.approx(50.0)

    def test_unusable_events_are_excluded_but_counted(self, tmp_path: Path) -> None:
        """The critical accounting test. The mean is over 2 observations, and
        the aggregate says so rather than implying 3."""
        study, _ = study_with({3: 2.0, 4: None, 5: 4.0}, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert agg.sample_size == 2
        assert agg.excluded == 1
        assert agg.mean_return_pct == pytest.approx(3.0, abs=0.02)

    def test_stdev_needs_two_observations(self, tmp_path: Path) -> None:
        study, _ = study_with({3: 2.0}, tmp_path)
        assert aggregate_window(study, "post_event_1h").stdev_return_pct is None

    def test_small_sample_is_flagged(self, tmp_path: Path) -> None:
        study, _ = study_with({3: 1.0, 4: 2.0}, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert agg.is_reliable is False
        assert agg.caveat is not None
        assert "indicative" in agg.caveat

    def test_sample_at_the_threshold_is_not_flagged(self, tmp_path: Path) -> None:
        returns = {day: 1.0 for day in range(3, 3 + MIN_RELIABLE_SAMPLE)}
        study, _ = study_with(returns, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert agg.sample_size == MIN_RELIABLE_SAMPLE
        assert agg.is_reliable is True
        assert agg.caveat is None

    def test_window_with_no_usable_data_still_appears(self, tmp_path: Path) -> None:
        """Silently omitting the window would hide that it was attempted."""
        study, _ = study_with({3: None, 4: None}, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert agg.sample_size == 0
        assert agg.excluded == 2
        assert agg.caveat == "no usable observations"

    def test_aggregate_study_covers_every_window(self, tmp_path: Path) -> None:
        source = ScriptedSource({3: 1.0, 4: 2.0})
        service = MarketDataService(
            source, SQLiteCandleRepository(tmp_path / "candles.db")
        )
        spec = ResearchSpec(
            event_count=2,
            windows=(
                POST_1H,
                WindowSpec(name="pre_event_1h", start="-1h", end="0h"),
            ),
            as_of=AS_OF,
        )
        study = run_study(spec, service, [make_event(3), make_event(4)], now=AS_OF)

        aggregates = aggregate_study(study)
        assert set(aggregates) == {"post_event_1h", "pre_event_1h"}

    def test_percentiles_are_ordered(self, tmp_path: Path) -> None:
        returns = {day: float(day) for day in range(3, 15)}
        study, _ = study_with(returns, tmp_path)
        agg = aggregate_window(study, "post_event_1h")

        assert (
            agg.min_return_pct
            <= agg.p10_return_pct
            <= agg.p25_return_pct
            <= agg.median_return_pct
            <= agg.p75_return_pct
            <= agg.p90_return_pct
            <= agg.max_return_pct
        )
