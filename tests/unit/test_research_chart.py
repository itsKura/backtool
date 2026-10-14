"""The price chart on a research answer.

An event page already charts a release this way, from our own candle cache
rather than TradingView's feed, so the picture and the statistics are one
thing. What is different here is which box gets shaded.

A release page reports four windows and shades four. A research answer reports
*one* number, from the horizon the question named, and a box describing a
statistic that is not on the page is a box the reader has to decode. So the
zone follows the question -- and because it is looked up through the same
mapping the engine measures with, the caption, the box and the figure cannot
end up describing different spans.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_interpretation import aggregate

from backtool.core.types import EventType, Interval
from backtool.research.cohort import Cohort
from backtool.research.experiment import Observation, ResearchFinding
from backtool.research.spec import ResearchSpec
from backtool.research.thesis import Outcome, Thesis
from backtool.research.vocabulary import reaction_windows
from backtool.research.windows import DEFAULT_WINDOWS, WindowSpec
from backtool.web.app import _measured_window, app
from backtool.web.event_page import (
    CANDLE_CHART_JS,
    CHART_SPAN,
    DISPLAY_BUDGET,
    EVENT_STYLE,
    chart_span_for,
    chart_window_bands,
    display_grain,
    thin_for_display,
)
from backtool.web.research_page import RESEARCH_SCRIPT, render_finding

NOW = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def cpi_thesis(horizon: str = "24h") -> Thesis:
    return Thesis(
        question="What happens to BTC after CPI?",
        asset="BTCUSDT",
        event_type=EventType.CPI,
        outcome=Outcome(horizon=horizon),
    )


def finding_for(
    thesis: Thesis, *, dates: tuple[dt.date, ...] = (dt.date(2024, 7, 11),)
) -> ResearchFinding:
    """A finished answer with observations, built without touching the engine."""
    window = _measured_window(thesis)
    assert window is not None
    return ResearchFinding(
        thesis=thesis,
        spec=ResearchSpec(
            symbol=thesis.asset,
            event_type=thesis.event_type,
            event_count=40,
            interval=Interval.M5,
            windows=reaction_windows(),
            as_of=NOW,
        ),
        generated_at=NOW,
        cohort=Cohort(
            key="left",
            label="All usable CPI releases",
            description="Every CPI release with usable price data",
            event_ids=tuple(f"CPI-{d.isoformat()}" for d in dates),
            # Borrowed rather than rebuilt: this file tests the chart, and a
            # change to the statistics should not surface here as a failure of
            # it.
            windows={
                window.name: aggregate(
                    n=len(dates), median=0.4, up=len(dates), window=window.name
                )
            },
        ),
        observations=tuple(
            Observation(
                event_id=f"CPI-{d.isoformat()}",
                event_date=d,
                conditions={},
                returns={thesis.outcome.horizon: 0.4},
                value=0.4,
            )
            for d in dates
        ),
        value=0.4,
        sample_size=len(dates),
    )


def chart_block(html: str) -> str:
    assert "data-chart" in html, "no chart on this answer"
    return html.split("<div class=\"chart-block\" data-chart>", 1)[1].split(
        "</div></div>", 1
    )[0]


# --------------------------------------------------------------------------


class TestTheZoneIsTheQuestionsOwnWindow:
    """The claim the whole feature rests on.

    "Selected zones based on what the user asked for" is only true if changing
    the question changes the box. Checked by changing it.
    """

    @pytest.mark.parametrize(
        ("horizon", "expected"),
        [
            ("5m", "post_event_5m"),
            ("15m", "post_event_15m"),
            ("1h", "post_event_1h"),
            ("4h", "post_event_4h"),
            ("24h", "post_event_24h"),
        ],
    )
    def test_each_horizon_selects_its_own_window(
        self, horizon: str, expected: str
    ) -> None:
        window = _measured_window(cpi_thesis(horizon))
        assert window is not None
        assert window.name == expected

    def test_the_window_comes_from_the_engines_own_mapping(self) -> None:
        """Not a second table. A box and a figure that disagreed about which
        span they describe would be worse than no box."""
        for window in reaction_windows():
            assert window in reaction_windows()
        assert _measured_window(cpi_thesis("24h")) in reaction_windows()

    def test_the_caption_names_the_same_window(self) -> None:
        html = render_finding(finding_for(cpi_thesis("1h")))
        assert "shaded: 1 hour after" in html

    def test_a_different_horizon_captions_differently(self) -> None:
        html = render_finding(finding_for(cpi_thesis("24h")))
        assert "shaded: 24 hours after" in html
        assert "1 hour after" not in html

    def test_only_the_measured_window_is_shaded(self) -> None:
        """An event page shades four. An answer reports one number."""
        event_time = dt.datetime(2024, 7, 11, 12, 30, tzinfo=dt.UTC)
        window = _measured_window(cpi_thesis("1h"))
        assert window is not None
        bands = chart_window_bands(event_time, windows=(window,))
        assert len(bands) == 1
        assert bands[0]["title"] == "1 hour after"


class TestTheEventPageIsUnchanged:
    """The chart was lifted out of a working page, so the working page has to
    keep working. Both of the things that moved are defaulted."""

    def test_the_default_windows_are_still_the_four(self) -> None:
        event_time = dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC)
        bands = chart_window_bands(event_time)
        assert [band["title"] for band in bands] == [
            "24 hours before",
            "1 hour before",
            "1 hour after",
            "24 hours after",
        ]

    def test_the_default_span_is_still_the_charted_span(self) -> None:
        event_time = dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC)
        bands = chart_window_bands(event_time)
        earliest = min(int(band["start"]) for band in bands)  # type: ignore[call-overload]
        assert earliest == int((event_time - CHART_SPAN).timestamp())

    def test_a_window_reaching_past_the_span_is_clipped(self) -> None:
        """Otherwise the box shades empty canvas."""
        event_time = dt.datetime(2026, 9, 16, 18, 0, tzinfo=dt.UTC)
        far = WindowSpec(name="post_event_long", start="0h", end="100h")
        bands = chart_window_bands(event_time, windows=(far,))
        assert int(bands[0]["end"]) == int(  # type: ignore[call-overload]
            (event_time + CHART_SPAN).timestamp()
        )

    def test_the_event_page_still_mounts_the_chart(self, client: TestClient) -> None:
        html = client.get("/event/FOMC-2026-09-16").text
        assert "lightweight-charts" in html
        assert 'id="price-chart"' in html
        assert "backtoolCandleChart" in html

    def test_the_chart_body_is_defined_once(self) -> None:
        """Two copies would drift, and the one a reader looked at would be
        whichever page they opened."""
        assert "window.backtoolCandleChart = function" in CANDLE_CHART_JS
        assert CANDLE_CHART_JS.count("window.backtoolCandleChart = function") == 1


class TestTheAnswerCarriesAChart:
    def test_an_answer_with_observations_gets_one(self) -> None:
        assert "data-chart" in render_finding(finding_for(cpi_thesis()))

    def test_an_empty_answer_does_not(self) -> None:
        """Nothing measured, nothing to look at."""
        thesis = cpi_thesis()
        empty = finding_for(thesis).model_copy(
            update={"observations": (), "sample_size": 0}
        )
        assert "data-chart" not in render_finding(empty)

    def test_an_event_that_has_not_happened_is_not_offered(self) -> None:
        """A future release has no candles around it, so it cannot be charted
        and must not appear in the list as though it could."""
        ahead = dt.date.today() + dt.timedelta(days=30)
        finding = finding_for(cpi_thesis(), dates=(ahead,))
        assert "data-chart" not in render_finding(finding)

    def test_the_chart_carries_the_thesis(self) -> None:
        """What identifies the event. A resolved concept exists only inside the
        thesis that carries it -- nothing here stores one -- so the chart
        cannot ask for its candles by id alone."""
        html = render_finding(finding_for(cpi_thesis()))
        body = html.split('class="thesis">', 1)[1].split("</script>", 1)[0]
        assert Thesis.model_validate_json(body.replace(chr(92) + "u003c", "<"))

    def test_one_event_needs_no_picker(self) -> None:
        html = render_finding(finding_for(cpi_thesis()))
        assert "chart-event" not in html

    def test_several_events_get_one_newest_first(self) -> None:
        """Forty releases have forty charts in them and no useful way to draw
        them at once, so the most recent is shown and the rest are a list."""
        dates = (dt.date(2024, 7, 11), dt.date(2025, 7, 11), dt.date(2026, 1, 13))
        html = render_finding(finding_for(cpi_thesis(), dates=dates))
        block = chart_block(html)
        assert block.count("<option") == 3
        assert block.index("2026-01-13") < block.index("2024-07-11")


class TestTheChartIsMountedDespiteInnerHtml:
    """A script element inserted with innerHTML never runs.

    The thread is inserted exactly that way, so the card carries data and no
    behaviour, and the page script picks it up. This is also what lets several
    answers in one thread each hold a chart, which shared ids could not.
    """

    def test_the_card_contains_no_script_of_its_own(self) -> None:
        block = chart_block(render_finding(finding_for(cpi_thesis())))
        assert "backtoolCandleChart" not in block

    def test_the_page_script_mounts_whatever_arrives(self) -> None:
        assert "function mountCharts" in RESEARCH_SCRIPT
        assert "[data-chart]" in RESEARCH_SCRIPT

    def test_both_paths_that_insert_a_card_mount_it(self) -> None:
        """A chart appears after an answer is run and after a question is
        interpreted, and forgetting either leaves a permanent spinner."""
        assert RESEARCH_SCRIPT.count("mountCharts(thread)") == 2

    def test_a_card_is_mounted_once(self) -> None:
        assert "if (block.dataset.mounted) continue" in RESEARCH_SCRIPT

    def test_the_research_page_loads_the_library(self, client: TestClient) -> None:
        html = client.get("/research").text
        assert "lightweight-charts" in html
        assert "backtoolCandleChart" in html


class TestTheCandleEndpoint:
    """Everything it serves comes from where the answer came from."""

    def body(self, thesis: Thesis, **extra: Any) -> dict[str, Any]:
        return {"thesis": thesis.model_dump_json(), **extra}

    def test_an_unreadable_thesis_is_a_message_not_a_crash(
        self, client: TestClient
    ) -> None:
        """The chart shows the reason. A 500 here would blank the panel under
        a finished answer."""
        data = client.post("/api/research/chart", json={"thesis": "{]"}).json()
        assert data["candles"] == []
        assert "could not be read" in data["message"]

    def test_it_serves_candles_and_the_measured_window(
        self, client: TestClient
    ) -> None:
        data = client.post(
            "/api/research/chart", json=self.body(cpi_thesis("24h"))
        ).json()
        assert data.get("message") is None
        assert len(data["candles"]) > 100
        assert [band["title"] for band in data["windows"]] == ["24 hours after"]

    def test_the_window_follows_the_horizon(self, client: TestClient) -> None:
        data = client.post(
            "/api/research/chart", json=self.body(cpi_thesis("1h"))
        ).json()
        assert [band["title"] for band in data["windows"]] == ["1 hour after"]

    def test_the_band_is_priced_by_the_research_layer(
        self, client: TestClient
    ) -> None:
        """A high is a statistic with a rule about which candles fall inside
        the window. Recomputing it in the browser would put a second
        definition of it in the codebase.

        Pinned to a release a day of data has long since closed over. Asking
        for the most recent one made this fail on release days: a 24-hour
        window from a release two hours old ends tomorrow, and refusing to
        price it is the engine being right. That case is the next test.
        """
        data = client.post(
            "/api/research/chart",
            json=self.body(cpi_thesis("24h"), event_id="CPI-2024-07-11"),
        ).json()
        band = data["windows"][0]
        assert band["high"] is not None
        assert band["low"] is not None
        assert band["high"] >= band["low"]

    def test_a_window_the_data_does_not_reach_is_left_unpriced(
        self, client: TestClient
    ) -> None:
        """The box is still drawn; the figures are withheld.

        The span is a fact about the question, so it is shaded whether or not
        the days it covers have happened. The high is a fact about the data,
        and a high taken over the few hours that have closed so far is not the
        high of a 24-hour window -- it is a shorter window wearing a longer
        one's caption.

        Stated as the rule rather than as a date, because the thing being
        tested only occurs on release days and a test that cannot run today
        is not a test of today's behaviour.
        """
        data = client.post(
            "/api/research/chart", json=self.body(cpi_thesis("24h"))
        ).json()
        event = dt.datetime.fromtimestamp(data["event_time"], tz=dt.UTC)
        last = dt.datetime.fromtimestamp(data["candles"][-1]["time"], tz=dt.UTC)
        band = data["windows"][0]

        assert band["title"] == "24 hours after"
        assert band["end"] - band["start"] == 24 * 60 * 60

        a_whole_day_has_closed = last - event >= dt.timedelta(hours=24)
        assert (band["high"] is not None) is a_whole_day_has_closed
        assert (band["low"] is not None) is a_whole_day_has_closed

    def test_it_defaults_to_the_most_recent_event(self, client: TestClient) -> None:
        data = client.post(
            "/api/research/chart", json=self.body(cpi_thesis())
        ).json()
        assert data["event_time"] <= dt.datetime.now(tz=dt.UTC).timestamp()

    def test_a_named_event_is_honoured(self, client: TestClient) -> None:
        first = client.post(
            "/api/research/chart", json=self.body(cpi_thesis())
        ).json()
        other = client.post(
            "/api/research/chart",
            json=self.body(cpi_thesis(), event_id="CPI-2024-07-11"),
        ).json()
        assert other["event_id"] == "CPI-2024-07-11"
        assert other["event_time"] != first["event_time"]

    def test_an_event_this_question_does_not_measure_is_refused(
        self, client: TestClient
    ) -> None:
        data = client.post(
            "/api/research/chart",
            json=self.body(cpi_thesis(), event_id="FOMC-2026-09-16"),
        ).json()
        assert data["candles"] == []
        assert "not one of the events" in data["message"]

    def test_the_candles_are_the_shape_the_library_wants(
        self, client: TestClient
    ) -> None:
        data = client.post(
            "/api/research/chart", json=self.body(cpi_thesis())
        ).json()
        candle = data["candles"][0]
        assert sorted(candle) == ["close", "high", "low", "open", "time"]
        assert isinstance(candle["time"], int)

    def test_it_charts_the_thesis_own_asset(self, client: TestClient) -> None:
        data = client.post(
            "/api/research/chart", json=self.body(cpi_thesis())
        ).json()
        assert data["symbol"] == "BTCUSDT"


class TestTheChartAndTheStatisticsAreOneThing:
    """The reason the chart is fed from our own cache rather than embedded.

    TradingView's widget renders TradingView's aggregated feed, which can
    disagree with Binance candle by candle. A reader seeing +2.59% in a table
    beside a differently shaped candle has no way to tell which is right.
    """

    def test_the_bands_use_the_same_window_specs_as_the_study(self) -> None:
        thesis = cpi_thesis("4h")
        spec = ResearchSpec(
            symbol=thesis.asset,
            event_type=thesis.event_type,
            interval=Interval.M5,
            windows=reaction_windows(),
            as_of=NOW,
        )
        measured = _measured_window(thesis)
        assert measured in spec.windows

    def test_the_default_windows_are_not_the_reaction_windows(self) -> None:
        """If they were the same list, none of the above would be evidence of
        anything."""
        assert set(DEFAULT_WINDOWS) != set(reaction_windows())

    def test_a_horizon_with_no_window_shades_nothing_rather_than_guessing(
        self,
    ) -> None:
        """Nothing to shade is a chart with no box, not a box over a guessed
        span. The horizon is validated long before here, so this is the
        belt rather than the braces."""
        unmeasurable = cpi_thesis().model_copy(
            update={"outcome": Outcome.model_construct(horizon="nonsense")}
        )
        assert _measured_window(unmeasurable) is None


class TestALongChartIsDrawnInCandlesAChartCanDraw:
    """Two years of hourly candles is 35,042 rows and 2.9 MiB of JSON, sent to
    a canvas about 1,600 pixels wide -- some 39 candles per pixel. The reader
    waits for data that cannot be drawn, and the browser scans all of it for
    every band edge.

    The reduction has to be an aggregation. Keeping every n-th row would throw
    away the extremes in between: over two years of real candles, sampling
    loses the high by about $900, so the chart's tallest wick would sit below
    the high printed on the shaded box beside it. Bucketing keeps the max and
    the min, which is what a daily candle already is.
    """

    def frame(self, hours: int) -> Any:
        """A frame with a spike in a known place, so losing it is detectable."""
        import pandas as pd

        start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        frame = pd.DataFrame(
            {
                "open_time": [start + dt.timedelta(hours=i) for i in range(hours)],
                "open": [100.0 + i for i in range(hours)],
                "high": [101.0 + i for i in range(hours)],
                "low": [99.0 + i for i in range(hours)],
                "close": [100.5 + i for i in range(hours)],
            }
        )
        # One hour wide, and on a prime-numbered row so that no plausible
        # sampling stride lands on it by luck.
        if hours > 1459:
            frame.loc[1459, "high"] = 999_999.0
            frame.loc[1459, "low"] = -999.0
        return frame

    def test_a_short_chart_is_left_alone(self) -> None:
        """Most questions. A day either side at five minutes is 579 candles and
        wants no help; re-graining it would cost resolution for nothing."""
        frame = self.frame(500)
        drawn, grain = thin_for_display(frame, Interval.H1)
        assert grain is None
        assert drawn is frame

    def test_a_long_chart_is_aggregated(self) -> None:
        drawn, grain = thin_for_display(self.frame(17_523), Interval.H1)
        assert grain == "12h"
        assert len(drawn) <= DISPLAY_BUDGET
        assert list(drawn.columns) == ["open_time", "open", "high", "low", "close"]

    def test_no_extreme_is_lost(self) -> None:
        """The property that makes this safe rather than merely smaller."""
        frame = self.frame(17_523)
        drawn, _grain = thin_for_display(frame, Interval.H1)
        assert drawn["high"].max() == frame["high"].max() == 999_999.0
        assert drawn["low"].min() == frame["low"].min() == -999.0

    def test_sampling_would_have_lost_it(self) -> None:
        """Not a test of our code -- a test that the one above is worth having.
        If a stride kept the extremes too, aggregating would be needless
        complexity."""
        frame = self.frame(17_523)
        sampled = frame.iloc[::12]
        assert sampled["high"].max() < frame["high"].max()
        assert sampled["low"].min() > frame["low"].min()

    def test_the_ends_of_the_series_survive(self) -> None:
        frame = self.frame(17_523)
        drawn, _grain = thin_for_display(frame, Interval.H1)
        assert drawn["open"].iloc[0] == frame["open"].iloc[0]
        assert drawn["close"].iloc[-1] == frame["close"].iloc[-1]

    def test_every_drawn_candle_is_a_candle(self) -> None:
        drawn, _grain = thin_for_display(self.frame(17_523), Interval.H1)
        assert (drawn["high"] >= drawn["low"]).all()
        assert drawn["open_time"].is_monotonic_increasing

    @pytest.mark.parametrize(
        ("interval", "forbidden"),
        [(Interval.H1, {"15m", "30m"}), (Interval.M15, {"5m"})],
    )
    def test_a_grain_is_always_whole_source_candles(
        self, interval: Interval, forbidden: set[str]
    ) -> None:
        """A bucket that does not divide evenly would make one candle per group
        a different length from its neighbours while looking identical, and a
        bucket finer than the source cannot be built at all."""
        for rows in (3_000, 17_523, 40_000):
            grain = display_grain(rows, interval)
            assert grain is not None
            assert grain[0] not in forbidden
            assert grain[1] % interval.duration == dt.timedelta(0)
            assert grain[1] > interval.duration

    def test_the_coarsest_grain_covers_the_widest_question(self) -> None:
        """Two years, fetched as four years of hours, is the most the engine
        accepts. If that needed something past the ladder, the label on the
        chart would stop matching the candles under it."""
        four_years = int(dt.timedelta(days=1461).total_seconds() // 3600)
        grain = display_grain(four_years, Interval.H1)
        assert grain is not None
        assert grain[0] == "1d"

    def test_an_empty_frame_is_not_an_error(self) -> None:
        import pandas as pd

        empty = pd.DataFrame(
            {c: [] for c in ("open_time", "open", "high", "low", "close")}
        )
        drawn, grain = thin_for_display(empty, Interval.H1)
        assert grain is None
        assert drawn is empty
        assert thin_for_display(None, Interval.H1) == (None, None)


class TestTheDrawnCandleSizeIsReported:
    """A reader counting hours off a chart of daily candles is reading it
    wrong, and nothing on the page would say so. So the size the candles are
    drawn at crosses the wire with them.
    """

    def body(self, thesis: Thesis, **extra: Any) -> dict[str, Any]:
        return {"thesis": thesis.model_dump_json(), **extra}

    def test_a_short_question_reports_the_interval_it_measured(
        self, client: TestClient
    ) -> None:
        data = client.post(
            "/api/research/chart",
            json=self.body(cpi_thesis("24h"), event_id="CPI-2024-07-11"),
        ).json()
        assert data["interval"] == "5m"
        assert data["drawn_interval"] == "5m"

    def test_a_two_year_question_says_it_is_drawing_days(
        self, client: TestClient
    ) -> None:
        data = client.post(
            "/api/research/chart",
            json=self.body(cpi_thesis("2y"), event_id="CPI-2024-07-11"),
        ).json()
        assert data["interval"] == "1h"
        assert data["drawn_interval"] == "1d"
        assert len(data["candles"]) <= DISPLAY_BUDGET

    def test_the_figures_on_the_band_do_not_move_when_the_picture_coarsens(
        self, client: TestClient
    ) -> None:
        """The whole point. Bands are priced from the frame as fetched, so a
        two-year chart drawn in daily candles still reports the high of the
        window rather than the high of whichever day straddled its edge."""
        from backtool.config import Settings
        from backtool.data.service import MarketDataService
        from backtool.research.metrics import compute_window_metrics

        thesis = cpi_thesis("2y")
        data = client.post(
            "/api/research/chart",
            json=self.body(thesis, event_id="CPI-2024-07-11"),
        ).json()
        assert data["drawn_interval"] != data["interval"]

        window = _measured_window(thesis)
        assert window is not None
        event = dt.datetime.fromtimestamp(data["event_time"], tz=dt.UTC)
        span = chart_span_for(window)
        with MarketDataService.from_settings(Settings.from_env()) as service:
            frame = service.get_candles(
                thesis.asset, Interval.H1, event - span, event + span
            )
        metrics = compute_window_metrics(frame, window.resolve(event), Interval.H1)

        band = data["windows"][0]
        assert band["high"] == metrics.high
        assert band["low"] == metrics.low


class TestTheCaptionSaysWhatTheCandlesAre:
    """The card renders somewhere to write and the shared script writes there.

    Two halves that have to meet, because a chart on this page is data on a
    card plus behaviour in one script: a slot nothing fills and a script with
    nowhere to write both look fine on their own.
    """

    def test_the_card_renders_an_empty_slot(self) -> None:
        """Empty, not absent. The chart cannot know the size until the candles
        arrive, and a slot built in JavaScript would have to build the
        separator too."""
        html = chart_block(render_finding(finding_for(cpi_thesis("2y"))))
        assert '<span class="chart-grain"></span>' in html

    def test_the_script_is_given_the_slot(self) -> None:
        assert "grain: block.querySelector('.chart-grain')" in RESEARCH_SCRIPT

    def test_the_chart_writes_the_size_it_drew(self) -> None:
        assert "data.drawn_interval + ' candles'" in CANDLE_CHART_JS

    def test_it_only_writes_when_that_is_news(self) -> None:
        """A caption reading "5m candles" over a chart of 5m candles is noise
        standing where news should be, and every short question is that case."""
        assert "data.drawn_interval !== data.interval" in CANDLE_CHART_JS

    def test_a_missing_slot_is_not_a_crash(self) -> None:
        """The event page's chart is a fixed day at five minutes and never
        thinned, so its slot stays empty -- and any other caller may not have
        one at all."""
        assert "options.grain &&" in CANDLE_CHART_JS

    def test_the_separator_only_shows_on_a_filled_slot(self) -> None:
        """Otherwise every short chart carries a trailing middot pointing at
        nothing, which is the reason this is a CSS rule rather than markup."""
        assert ".chart-grain:not(:empty)::before" in EVENT_STYLE

    def test_the_note_survives_a_zoom(self) -> None:
        """The zoom handler rewrites .chart-span on every change. If the size
        shared that element it would vanish the first time anybody touched the
        view -- which is why it has its own."""
        assert ".chart-span" in RESEARCH_SCRIPT
        assert "span.textContent = picked.dataset.label" in RESEARCH_SCRIPT
        assert "grain.textContent" not in RESEARCH_SCRIPT


def test_the_shaded_zone_is_json_the_browser_can_read(client: TestClient) -> None:
    """The bands cross the wire as data, so a band that serialises badly is a
    chart with no boxes and no error."""
    response = client.post(
        "/api/research/chart",
        json={"thesis": cpi_thesis().model_dump_json()},
    )
    assert response.status_code == 200
    assert json.loads(response.text)["windows"]


class TestAResolvedConceptChartsToo:
    """The case this was asked for.

    An event looked up from the web is not in any calendar, which is exactly
    why the chart posts the whole thesis: the dates exist only inside the
    resolution it carries. The compiled calendar is the engine's own, so the
    instant the chart centres on is the instant the study measured from.
    """

    def dynamic(self, resolution: object) -> Thesis:
        return Thesis(
            question="What happened to ETH after the Merge?",
            asset="ETHUSDT",
            event_type=None,
            resolution=resolution,  # type: ignore[arg-type]
            outcome=Outcome(horizon="1h"),
        )

    def test_the_calendar_is_compiled_from_the_resolution(self) -> None:
        from tests.fixtures.dynamic_events import eth_merge

        from backtool.research.experiment import thesis_calendar

        calendar, identity = thesis_calendar(self.dynamic(eth_merge()))
        assert [event.local_date.isoformat() for event in calendar] == ["2022-09-15"]
        assert identity

    def test_the_chart_centres_on_the_instant_the_study_measured(self) -> None:
        """Not on the date. A resolved concept can be dated to the day, and
        which instant inside it was a decision the reader made -- so the chart
        has to inherit it rather than re-derive one."""
        import datetime as dt

        from tests.fixtures.dynamic_events import midterms

        from backtool.events.resolution import AnchorPolicy, DayAnchor
        from backtool.research.experiment import thesis_calendar

        thesis = self.dynamic(midterms()).model_copy(
            update={"anchor": DayAnchor(policy=AnchorPolicy.DAY_END)}
        )
        calendar, _identity = thesis_calendar(thesis)
        assert calendar
        assert all(event.local_time == dt.time(23, 59, 59) for event in calendar)

    def test_an_event_still_ahead_is_said_so_rather_than_charted(
        self, client: TestClient
    ) -> None:
        """A resolved concept can be scheduled rather than past -- the next
        election is a date with no candles around it."""
        import datetime as dt

        from tests.fixtures.dynamic_events import midterms

        ahead = dt.date.today() + dt.timedelta(days=200)
        resolution = midterms()
        future = resolution.model_copy(
            update={
                "candidate_events": tuple(
                    event.model_copy(update={"local_date": ahead})
                    for event in resolution.candidate_events
                )
            }
        )
        from backtool.events.resolution import AnchorPolicy, DayAnchor

        thesis = self.dynamic(future).model_copy(
            update={"anchor": DayAnchor(policy=AnchorPolicy.DAY_END)}
        )
        data = client.post(
            "/api/research/chart", json={"thesis": thesis.model_dump_json()}
        ).json()
        assert data["candles"] == []
        assert "has happened yet" in data["message"]


class TestZoomingInDoesNotDeleteTheZone:
    """Reported from the product: 24h, 4h and 1h looked like three charts.

    Two separate mechanisms, and only one of them was a defect.

    The band had to be *entirely* inside the view or it was dropped, so that a
    box labelled "24 hours before" could never be drawn covering one hour of
    it. Right for a release page, which shades four windows of different
    lengths and where zooming in is how the short ones become legible --
    backwards for an answer that shades exactly one, where it meant zooming in
    to look more closely at nothing.
    """

    def source(self) -> str:
        return CANDLE_CHART_JS

    def test_a_band_overlapping_the_view_is_kept(self) -> None:
        assert "b.end > low && b.start < high" in self.source()

    def test_the_old_rule_that_dropped_it_is_gone(self) -> None:
        assert "b.start >= low && b.end <= high" not in self.source()

    def test_it_is_clipped_to_the_view(self) -> None:
        """Kept is not enough: drawn at its full span it would run outside the
        canvas and the box would stop meaning anything."""
        assert "Math.max(b.start, low)" in self.source()
        assert "Math.min(b.end, high)" in self.source()

    def test_a_clipped_band_says_it_continues(self) -> None:
        """The arrow and the open border are the whole of the claim. Without
        them the box reads as ending where the view does, which is the
        mislabelling the old rule existed to prevent."""
        source = self.source()
        assert "clippedStart" in source
        assert "clippedEnd" in source
        assert chr(92) + "u2190" in source
        assert chr(92) + "u2192" in source

    def test_the_open_side_loses_its_border(self) -> None:
        assert "band-open-start" in EVENT_STYLE
        assert "band-open-end" in EVENT_STYLE
        assert "border-right-color: transparent" in EVENT_STYLE

    def test_the_marker_classes_reach_the_box(self) -> None:
        assert "band-open-start" in self.source()
        assert "band-open-end" in self.source()

    def test_the_arrows_are_escapes_not_characters(self) -> None:
        """So they do not depend on how the response is encoded. Written as
        the escape in the Python source too, since a plain triple-quoted
        string would otherwise resolve it before the browser ever saw it."""
        source = self.source()
        assert chr(92) + "u2190" in source
        assert "←" not in source


class TestTheCaptionDescribesTheView:
    """The other half, and not a defect: the zoom narrows the series.

    `fitContent`, `setVisibleRange` and `setVisibleLogicalRange` are accepted
    and ignored by this build, so narrowing the data is the only lever that
    moves the view. The price axis then rescales to what is shown and the
    last-price line follows the last visible candle -- three endpoints of one
    series, which reads as three different prices unless the caption says what
    is being shown.
    """

    def test_the_span_is_its_own_element(self) -> None:
        html = render_finding(finding_for(cpi_thesis()))
        assert 'class="chart-span"' in html

    def test_it_starts_at_the_opening_zoom(self) -> None:
        html = render_finding(finding_for(cpi_thesis()))
        opening = int(CHART_SPAN.total_seconds() // 3600)
        assert f">{opening}h either side</span>" in html

    def test_the_chart_rewrites_it_on_every_zoom(self) -> None:
        """The label comes off the pressed button rather than being built in
        JavaScript, because formatting a duration belongs with the ladder that
        defines the rungs -- "17520h" is not a label a reader can use."""
        assert "onZoom" in RESEARCH_SCRIPT
        assert "dataset.label" in RESEARCH_SCRIPT
        assert "either side" in RESEARCH_SCRIPT

    def test_the_chart_offers_the_hook_rather_than_requiring_it(self) -> None:
        """The event page passes no callback, and must keep working."""
        assert "typeof options.onZoom === 'function'" in CANDLE_CHART_JS
