"""A window that runs past the data is not that window.

The anchoring rule takes the last candle that closed at or before a boundary
(ADR 002). That is exactly right for avoiding look-ahead, and it has a quiet
second consequence: when history stops short, the "last candle at or before the
end" is simply the last candle there is. The window prices, returns a number,
and reports ``ok``.

Harmless while the longest window is 24 hours, because a day of candles either
side of an event is always there. Not harmless once horizons are measured in
years, which is what these tests were written for: "how did BTC do in the two
years after the inauguration" must not answer with twenty-one months of the
2025 president and call it two years.

The number would be plausible, the status would be ``ok``, and the only trace
would be a lag field nothing reads. That is the worst shape a wrong answer can
take in this system, so it is refused instead.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backtool.core.types import Interval
from backtool.research.metrics import WindowStatus, _roughly, compute_window_metrics
from backtool.research.windows import WindowSpec

UTC = dt.UTC
EVENT = dt.datetime(2025, 1, 20, 12, 0, tzinfo=UTC)


def hourly(hours: int, *, start: dt.datetime | None = None) -> pd.DataFrame:
    """A gap-free hourly frame beginning an hour before the event."""
    first = start if start is not None else EVENT - dt.timedelta(hours=1)
    rows = []
    for i in range(hours):
        opened = first + dt.timedelta(hours=i)
        price = 100.0 + i * 0.01
        rows.append(
            {
                "open_time": opened,
                "close_time": opened + dt.timedelta(hours=1),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
        )
    frame = pd.DataFrame(rows)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    return frame


def measure(frame: pd.DataFrame, end: str, *, start: str = "0h"):  # type: ignore[no-untyped-def]
    window = WindowSpec(name="w", start=start, end=end).resolve(EVENT)
    return compute_window_metrics(frame, window, Interval.H1)


class TestAWindowBeyondTheDataIsRefused:
    """The case this exists for."""

    def test_two_years_over_one_year_of_candles(self) -> None:
        """Before the guard this returned ``ok`` with an 87.59% "two-year"
        return that was really one year."""
        result = measure(hourly(24 * 365), "730d")
        assert result.status is WindowStatus.INSUFFICIENT_DATA
        assert result.return_pct is None

    def test_the_reason_names_the_boundary_and_the_shortfall(self) -> None:
        result = measure(hourly(24 * 365), "730d")
        assert result.reason is not None
        assert "window end" in result.reason
        assert "2027-01-20" in result.reason
        assert "1 year beyond" in result.reason

    def test_the_anchors_are_still_reported(self) -> None:
        """Refusing is not forgetting. A reader asking why wants to see how far
        the data actually reached."""
        result = measure(hourly(24 * 365), "730d")
        assert result.end_anchor_utc is not None
        assert result.end_anchor_lag_s is not None
        assert result.end_anchor_lag_s > 86_400

    def test_a_missing_start_is_caught_too(self) -> None:
        """Symmetric: an anchor a year early prices the wrong instant whichever
        end of the window it sits at."""
        late = hourly(48, start=EVENT + dt.timedelta(days=365))
        result = measure(late, "730d")
        assert result.status is WindowStatus.INSUFFICIENT_DATA


class TestACoveredWindowStillMeasures:
    """The guard must not refuse what it can answer."""

    def test_a_fully_covered_window_is_ok(self) -> None:
        result = measure(hourly(24 * 365), "364d")
        assert result.status is WindowStatus.OK
        assert result.return_pct is not None

    def test_a_window_ending_exactly_at_the_last_candle_is_ok(self) -> None:
        """The boundary the data reaches precisely. Off-by-one here would
        refuse every window that happens to end on the last candle."""
        frame = hourly(24 * 10)
        last = frame["close_time"].max().to_pydatetime()
        hours = int((last - EVENT).total_seconds() // 3600)
        result = measure(frame, f"{hours}h")
        assert result.status is WindowStatus.OK

    def test_one_missing_candle_at_the_boundary_is_tolerated(self) -> None:
        """An exchange outage of a single candle is ordinary, and is what
        ``coverage_pct`` is for. Refusing it would turn a note into a wall."""
        frame = hourly(24 * 10)
        trimmed = frame.iloc[:-1]
        last = frame["close_time"].max().to_pydatetime()
        hours = int((last - EVENT).total_seconds() // 3600)
        result = measure(trimmed, f"{hours}h")
        assert result.status is WindowStatus.OK

    def test_two_missing_candles_are_not(self) -> None:
        """One interval of slack, not a licence. Past that the anchor is
        pricing an instant the window did not ask about."""
        frame = hourly(24 * 10)
        trimmed = frame.iloc[:-3]
        last = frame["close_time"].max().to_pydatetime()
        hours = int((last - EVENT).total_seconds() // 3600)
        result = measure(trimmed, f"{hours}h")
        assert result.status is WindowStatus.INSUFFICIENT_DATA

    def test_the_short_windows_the_product_already_runs_are_unaffected(self) -> None:
        """The whole existing suite rides on this: 5m through 24h over data
        that covers them must behave exactly as before."""
        frame = hourly(24 * 3)
        for end in ("1h", "4h", "24h"):
            assert measure(frame, end).status is WindowStatus.OK


class TestTheShortfallReadsAsASentence:
    """Observed live: "is 8761 hours beyond the last candle available"."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (31_536_000, "1 year"),
            (63_072_000, "2 years"),
            (86_400 * 365.04, "1 year"),
            (259_200, "3 days"),
            (90_000, "1 day"),
            (7_200, "2 hours"),
            (300, "5 minutes"),
        ],
    )
    def test_it_rounds_to_the_largest_sensible_unit(
        self, seconds: float, expected: str
    ) -> None:
        assert _roughly(seconds) == expected

    def test_it_does_not_report_a_year_as_thousands_of_hours(self) -> None:
        assert "hours" not in _roughly(31_536_000)
