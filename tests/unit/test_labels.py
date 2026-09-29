"""Tests for the user-facing naming layer.

The rule these enforce is narrow and load-bearing: a label is derived from a
window's *offsets*, never from its name. A planner-invented window called
``run_up_before_the_thing`` must still render as a duration, because the name is
free text the model chose and the offsets are the structured truth.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.research.windows import (
    DEFAULT_WINDOWS,
    WindowSpec,
    compact_duration,
    humanise_duration,
    window_label,
    window_sentence,
    window_title,
)


class TestDurations:
    @pytest.mark.parametrize(
        ("hours", "expected"),
        [
            (1, "1 hour"),
            (4, "4 hours"),
            (24, "24 hours"),
            (48, "2 days"),
            (72, "3 days"),
        ],
    )
    def test_reads_as_words(self, hours: int, expected: str) -> None:
        assert humanise_duration(dt.timedelta(hours=hours)) == expected

    def test_twenty_four_hours_is_not_promoted_to_a_day(self) -> None:
        """"1 day" beside "1 hour" in a tab strip hides that one span is
        twenty-four times the other."""
        assert humanise_duration(dt.timedelta(hours=24)) == "24 hours"
        assert compact_duration(dt.timedelta(hours=24)) == "24h"

    def test_sub_hour_offsets_read_as_minutes(self) -> None:
        assert humanise_duration(dt.timedelta(minutes=30)) == "30 minutes"
        assert compact_duration(dt.timedelta(minutes=15)) == "15m"

    def test_sign_is_dropped(self) -> None:
        """Direction is carried by the word 'before', not by a minus sign."""
        assert humanise_duration(dt.timedelta(hours=-24)) == "24 hours"

    def test_an_odd_offset_still_renders(self) -> None:
        assert humanise_duration(dt.timedelta(seconds=90)) == "1.5 minutes"


class TestWindowLabels:
    @pytest.mark.parametrize(
        ("start", "end", "title", "label"),
        [
            ("-24h", "0h", "24 hours before", "24h before"),
            ("-1h", "0h", "1 hour before", "1h before"),
            ("0h", "1h", "1 hour after", "1h after"),
            ("0h", "24h", "24 hours after", "24h after"),
            ("-1h", "1h", "1 hour before to 1 hour after", "1h before to 1h after"),
            ("-24h", "-1h", "24 hours to 1 hour before", "24h to 1h before"),
            ("1h", "4h", "1 hour to 4 hours after", "1h to 4h after"),
        ],
    )
    def test_covers_every_window_shape(
        self, start: str, end: str, title: str, label: str
    ) -> None:
        window = WindowSpec(name="irrelevant", start=start, end=end)
        assert window_title(window) == title
        assert window_label(window) == label

    def test_the_name_is_never_consulted(self) -> None:
        """The planner may invent any name it likes; none of it reaches the
        page."""
        window = WindowSpec(name="run_up_before_the_thing", start="-1h", end="0h")
        rendered = window_title(window) + window_label(window) + window_sentence(window)
        assert "run_up" not in rendered
        assert "_" not in rendered

    def test_no_default_window_leaks_its_name(self) -> None:
        for window in DEFAULT_WINDOWS:
            assert "_" not in window_title(window)
            assert "_" not in window_label(window)
            assert window.name not in window_title(window)

    def test_sentence_form_reads_inside_prose(self) -> None:
        window = WindowSpec(name="x", start="0h", end="1h")
        assert window_sentence(window) == "the 1 hour after the release"
