"""Tests for offset parsing and event-relative window resolution."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from backtool.core.time import NEW_YORK, UTC
from backtool.research.windows import (
    DEFAULT_WINDOWS,
    WindowSpec,
    format_offset,
    parse_offset,
)


class TestParseOffset:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0h", dt.timedelta(0)),
            ("0m", dt.timedelta(0)),
            ("1h", dt.timedelta(hours=1)),
            ("+1h", dt.timedelta(hours=1)),
            ("-24h", dt.timedelta(hours=-24)),
            ("30m", dt.timedelta(minutes=30)),
            ("-90m", dt.timedelta(minutes=-90)),
            ("3d", dt.timedelta(days=3)),
            ("-1.5h", dt.timedelta(minutes=-90)),
            (" -24h ", dt.timedelta(hours=-24)),
        ],
    )
    def test_valid_offsets(self, text: str, expected: dt.timedelta) -> None:
        assert parse_offset(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "24",  # no unit: is that hours or days?
            "",
            "h",
            "24 h",
            "-24hours",
            "abc",
            "1w",  # weeks deliberately unsupported
            "--24h",
            "1h30m",  # compound offsets not supported
        ],
    )
    def test_rejects_malformed_offsets(self, text: str) -> None:
        """A silently misparsed offset would shift an entire study with no
        other symptom, so parsing is strict."""
        with pytest.raises(ValueError, match="Invalid offset"):
            parse_offset(text)


class TestFormatOffset:
    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            # timedelta normalises on construction: timedelta(hours=-24) *is*
            # timedelta(days=-1), so "-24h" cannot be recovered. Only the
            # round-trip property is meaningful.
            (dt.timedelta(hours=-24), "-1d"),
            (dt.timedelta(hours=1), "1h"),
            (dt.timedelta(minutes=30), "30m"),
            (dt.timedelta(days=3), "3d"),
            (dt.timedelta(0), "0m"),
        ],
    )
    def test_round_trips(self, delta: dt.timedelta, expected: str) -> None:
        assert format_offset(delta) == expected
        assert parse_offset(format_offset(delta)) == delta


class TestWindowSpec:
    def test_resolves_to_absolute_utc_instants(self) -> None:
        spec = WindowSpec(name="pre_event_24h", start="-24h", end="0h")
        window = spec.resolve(dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC))

        assert window.name == "pre_event_24h"
        assert window.start == dt.datetime(2024, 1, 30, 19, 0, tzinfo=UTC)
        assert window.end == dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
        assert window.duration == dt.timedelta(hours=24)

    def test_resolution_normalises_event_time_to_utc(self) -> None:
        spec = WindowSpec(name="post_event_1h", start="0h", end="1h")
        from_utc = spec.resolve(dt.datetime(2024, 1, 31, 19, 0, tzinfo=UTC))
        from_local = spec.resolve(dt.datetime(2024, 1, 31, 14, 0, tzinfo=NEW_YORK))
        assert from_utc == from_local

    def test_rejects_naive_event_time(self) -> None:
        spec = WindowSpec(name="w", start="0h", end="1h")
        with pytest.raises(ValueError, match="Naive datetime"):
            spec.resolve(dt.datetime(2024, 1, 31, 19, 0))

    def test_rejects_backwards_window(self) -> None:
        with pytest.raises(ValidationError, match="must span forward"):
            WindowSpec(name="backwards", start="1h", end="-1h")

    def test_rejects_zero_length_window(self) -> None:
        with pytest.raises(ValidationError, match="must span forward"):
            WindowSpec(name="empty", start="0h", end="0h")

    def test_rejects_malformed_offset(self) -> None:
        with pytest.raises(ValidationError, match="Invalid offset"):
            WindowSpec(name="bad", start="-24", end="0h")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            WindowSpec(name="   ", start="0h", end="1h")

    def test_is_immutable(self) -> None:
        spec = WindowSpec(name="w", start="0h", end="1h")
        with pytest.raises(ValidationError):
            spec.name = "renamed"  # type: ignore[misc]

    def test_serialises_for_the_ai_layer(self) -> None:
        """The same model becomes the LLM's structured-output schema, so it
        must round-trip through plain JSON-compatible data."""
        spec = WindowSpec(name="post_event_24h", start="0h", end="24h")
        assert spec.model_dump() == {
            "name": "post_event_24h",
            "start": "0h",
            "end": "24h",
        }
        assert WindowSpec.model_validate(spec.model_dump()) == spec


class TestDefaultWindows:
    def test_all_defaults_are_valid_and_named_uniquely(self) -> None:
        names = [window.name for window in DEFAULT_WINDOWS]
        assert len(names) == len(set(names))

    def test_defaults_bracket_the_event(self) -> None:
        """Two windows end at the event, two begin at it."""
        ending_at_event = [w for w in DEFAULT_WINDOWS if w.end_offset == dt.timedelta(0)]
        starting_at_event = [w for w in DEFAULT_WINDOWS if w.start_offset == dt.timedelta(0)]
        assert len(ending_at_event) == 2
        assert len(starting_at_event) == 2
