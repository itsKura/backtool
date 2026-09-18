"""Tests for the research specification."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from backtool.core.time import UTC
from backtool.core.types import EventType, Interval
from backtool.research.spec import ResearchSpec
from backtool.research.windows import WindowSpec


class TestValidation:
    def test_defaults_are_usable(self) -> None:
        spec = ResearchSpec()
        assert spec.symbol == "BTCUSDT"
        assert spec.event_type is EventType.FOMC
        assert spec.interval is Interval.M5
        assert len(spec.windows) == 4

    def test_symbol_is_normalised(self) -> None:
        assert ResearchSpec(symbol="  btcusdt ").symbol == "BTCUSDT"

    def test_rejects_non_alphanumeric_symbol(self) -> None:
        with pytest.raises(ValidationError, match="alphanumeric"):
            ResearchSpec(symbol="../../etc/passwd")

    def test_rejects_empty_window_set(self) -> None:
        with pytest.raises(ValidationError, match="at least one window"):
            ResearchSpec(windows=())

    def test_rejects_duplicate_window_names(self) -> None:
        """Windows are keyed by name in results; duplicates would silently
        overwrite each other."""
        with pytest.raises(ValidationError, match="Duplicate window name"):
            ResearchSpec(
                windows=(
                    WindowSpec(name="w", start="-1h", end="0h"),
                    WindowSpec(name="w", start="0h", end="1h"),
                )
            )

    def test_rejects_non_positive_event_count(self) -> None:
        with pytest.raises(ValidationError):
            ResearchSpec(event_count=0)

    def test_is_immutable(self) -> None:
        spec = ResearchSpec()
        with pytest.raises(ValidationError):
            spec.event_count = 5  # type: ignore[misc]


class TestOffsets:
    def test_earliest_and_latest_span_all_windows(self) -> None:
        spec = ResearchSpec(
            windows=(
                WindowSpec(name="a", start="-48h", end="-24h"),
                WindowSpec(name="b", start="0h", end="6h"),
            )
        )
        assert spec.earliest_offset == dt.timedelta(hours=-48)
        assert spec.latest_offset == dt.timedelta(hours=6)

    def test_defaults_span_24h_either_side(self) -> None:
        spec = ResearchSpec()
        assert spec.earliest_offset == dt.timedelta(hours=-24)
        assert spec.latest_offset == dt.timedelta(hours=24)


class TestFingerprint:
    def test_identical_specs_share_a_fingerprint(self) -> None:
        assert ResearchSpec().fingerprint == ResearchSpec().fingerprint

    @pytest.mark.parametrize(
        "changes",
        [
            {"event_count": 21},
            {"interval": Interval.H1},
            {"symbol": "ETHUSDT"},
            {"as_of": dt.datetime(2026, 1, 1, tzinfo=UTC)},
        ],
    )
    def test_any_change_changes_the_fingerprint(self, changes: dict[str, object]) -> None:
        assert ResearchSpec(**changes).fingerprint != ResearchSpec().fingerprint

    def test_a_single_window_boundary_changes_the_fingerprint(self) -> None:
        """Results are only reproducible if the fingerprint covers every input
        that moves a number, down to one boundary."""
        base = ResearchSpec(windows=(WindowSpec(name="w", start="-24h", end="0h"),))
        tweaked = ResearchSpec(windows=(WindowSpec(name="w", start="-23h", end="0h"),))
        assert base.fingerprint != tweaked.fingerprint

    def test_fingerprint_is_short_and_hex(self) -> None:
        fingerprint = ResearchSpec().fingerprint
        assert len(fingerprint) == 12
        assert all(character in "0123456789abcdef" for character in fingerprint)


class TestDescribe:
    def test_lists_every_window(self) -> None:
        text = "\n".join(ResearchSpec().describe())
        for name in ("pre_event_24h", "pre_event_1h", "post_event_1h", "post_event_24h"):
            assert name in text

    def test_states_the_anchor_rule(self) -> None:
        """The product promise is that no definition is applied silently."""
        text = "\n".join(ResearchSpec().describe())
        assert "no look-ahead" in text
        assert "MFE / MAE" in text

    def test_warns_when_the_cutoff_is_not_pinned(self) -> None:
        assert "not reproducible" in "\n".join(ResearchSpec().describe())

    def test_shows_a_pinned_cutoff(self) -> None:
        spec = ResearchSpec(as_of=dt.datetime(2026, 1, 1, tzinfo=UTC))
        text = "\n".join(spec.describe())
        assert "2026-01-01" in text
        assert "not reproducible" not in text


class TestSerialisation:
    def test_round_trips_through_json(self) -> None:
        """The same model becomes the LLM planner's output schema, so it must
        survive a plain JSON round trip."""
        spec = ResearchSpec(event_count=12, interval=Interval.M15)
        assert ResearchSpec.model_validate_json(spec.model_dump_json()) == spec

    def test_json_schema_is_generatable(self) -> None:
        schema = ResearchSpec.model_json_schema()
        assert "symbol" in schema["properties"]
        assert "windows" in schema["properties"]
