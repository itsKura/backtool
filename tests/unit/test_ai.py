"""Tests for the planning and interpretation layer, against a mocked API.

No network is touched. These assert the contract around the model -- what gets
sent, what is accepted back, and what is rejected -- rather than the model's
judgement, which is not a testable property.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from backtool.ai.client import (
    AIClient,
    AIError,
    AINotConfiguredError,
    is_configured,
    strict_json_schema,
)
from backtool.ai.models import Interpretation, PlannedStudy, PlannedWindow
from backtool.ai.planner import MAX_WINDOWS, plan_study, render_plan
from backtool.core.types import Interval
from backtool.research.spec import ResearchSpec

AS_OF = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def valid_plan_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "restated_question": "How does BTC behave in the hour after FOMC?",
        "symbol": "BTCUSDT",
        "event_type": "FOMC",
        "event_count": 20,
        "interval": "5m",
        "windows": [
            {
                "name": "post_event_1h",
                "start": "0h",
                "end": "1h",
                "rationale": "The immediate reaction to the announcement.",
            }
        ],
        "interpretation": ["immediate reaction = 0h to +1h after the announcement"],
        "assumptions": ["interval not specified; 5m chosen for a one-hour window"],
    }
    payload.update(overrides)
    return payload


def mock_client(
    payload: dict[str, Any] | str,
    *,
    status: int = 200,
    stop_reason: str = "end_turn",
    capture: list[httpx.Request] | None = None,
) -> AIClient:
    """An AIClient wired to a fake transport returning ``payload`` as the model's JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "nope"}})
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
                "stop_details": {"explanation": "policy"} if stop_reason == "refusal" else None,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

    return AIClient(
        api_key="test-key",
        max_retries=2,
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://mock.invalid"
        ),
    )


class TestStrictSchema:
    def test_inlines_nested_model_references(self) -> None:
        """Structured outputs cannot follow $ref, so nested models must be
        expanded in place."""
        schema = strict_json_schema(PlannedStudy)
        assert "$defs" not in json.dumps(schema)
        assert "$ref" not in json.dumps(schema)

    def test_requires_every_property(self) -> None:
        schema = strict_json_schema(PlannedStudy)
        assert set(schema["required"]) == set(schema["properties"])

    def test_forbids_extra_properties_at_every_level(self) -> None:
        schema = strict_json_schema(PlannedStudy)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["windows"]["items"]["additionalProperties"] is False

    def test_strips_defaults(self) -> None:
        """With every field required, a default is meaningless and some
        validators reject the combination."""

        class WithDefault(BaseModel):
            a: int = 3
            b: str = "x"

        assert "default" not in json.dumps(strict_json_schema(WithDefault))


class TestConfiguration:
    def test_missing_key_raises_a_distinct_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its own type so the CLI can fall back to the deterministic engine
        rather than failing the whole run."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert is_configured() is False
        with pytest.raises(AINotConfiguredError, match="ANTHROPIC_API_KEY"):
            AIClient()

    def test_is_configured_reflects_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert is_configured() is True


class TestRequestShape:
    def test_sends_the_schema_and_prompts(self) -> None:
        captured: list[httpx.Request] = []
        client = mock_client(valid_plan_payload(), capture=captured)
        client.structured(
            system="SYSTEM TEXT", user="USER TEXT", schema=PlannedStudy
        )

        body = json.loads(captured[0].content)
        assert body["system"] == "SYSTEM TEXT"
        assert body["messages"] == [{"role": "user", "content": "USER TEXT"}]
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["thinking"] == {"type": "adaptive"}

    def test_sends_authentication_headers(self) -> None:
        captured: list[httpx.Request] = []
        client = mock_client(valid_plan_payload(), capture=captured)
        client.structured(system="s", user="u", schema=PlannedStudy)

        assert captured[0].headers["x-api-key"] == "test-key"
        assert captured[0].headers["anthropic-version"]


class TestResponseHandling:
    def test_validates_into_the_requested_model(self) -> None:
        client = mock_client(valid_plan_payload())
        result = client.structured(system="s", user="u", schema=PlannedStudy)

        assert isinstance(result, PlannedStudy)
        assert result.symbol == "BTCUSDT"
        assert result.windows[0].name == "post_event_1h"

    def test_refusal_raises(self) -> None:
        client = mock_client(valid_plan_payload(), stop_reason="refusal")
        with pytest.raises(AIError, match="declined"):
            client.structured(system="s", user="u", schema=PlannedStudy)

    def test_truncation_raises_rather_than_returning_partial_output(self) -> None:
        client = mock_client(valid_plan_payload(), stop_reason="max_tokens")
        with pytest.raises(AIError, match="cut off"):
            client.structured(system="s", user="u", schema=PlannedStudy)

    def test_malformed_json_raises(self) -> None:
        client = mock_client("{not json at all")
        with pytest.raises(AIError, match="not valid JSON"):
            client.structured(system="s", user="u", schema=PlannedStudy)

    def test_wrong_shape_raises(self) -> None:
        client = mock_client({"unexpected": "shape"})
        with pytest.raises(AIError, match="did not match"):
            client.structured(system="s", user="u", schema=PlannedStudy)

    def test_bad_key_raises_not_configured(self) -> None:
        client = mock_client(valid_plan_payload(), status=401)
        with pytest.raises(AINotConfiguredError, match="rejected the API key"):
            client.structured(system="s", user="u", schema=PlannedStudy)

    def test_server_error_retries_then_gives_up(self) -> None:
        client = mock_client(valid_plan_payload(), status=503)
        with pytest.raises(AIError, match="after 2 attempts"):
            client.structured(system="s", user="u", schema=PlannedStudy)


class TestPlanner:
    def test_produces_a_usable_spec(self) -> None:
        plan, spec = plan_study(
            "What happens to BTC in the hour after FOMC?",
            client=mock_client(valid_plan_payload()),
            as_of=AS_OF,
        )

        assert isinstance(spec, ResearchSpec)
        assert spec.symbol == "BTCUSDT"
        assert spec.interval is Interval.M5
        assert spec.as_of == AS_OF
        assert [w.name for w in spec.windows] == ["post_event_1h"]
        assert plan.interpretation

    def test_the_cutoff_comes_from_the_caller_not_the_model(self) -> None:
        """Reproducibility is not the model's decision to make."""
        _, spec = plan_study(
            "anything", client=mock_client(valid_plan_payload()), as_of=AS_OF
        )
        assert spec.as_of == AS_OF

    def test_rejects_an_empty_question(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            plan_study("   ", client=mock_client(valid_plan_payload()))

    def test_rejects_an_unsupported_symbol(self) -> None:
        """Structured outputs guarantee shape, not sense -- the model can still
        emit a symbol the engine cannot run."""
        client = mock_client(valid_plan_payload(symbol="DOGEUSDT"))
        with pytest.raises(ValueError, match="outside V1 scope"):
            plan_study("anything", client=client)

    def test_rejects_an_unsupported_interval(self) -> None:
        client = mock_client(valid_plan_payload(interval="1w"))
        with pytest.raises(ValueError):
            plan_study("anything", client=client)

    def test_rejects_too_many_windows(self) -> None:
        """Each extra window is another comparison; the cap is a guard against
        the model being thorough into a false positive."""
        windows = [
            {"name": f"w{i}", "start": "0h", "end": "1h", "rationale": "x"}
            for i in range(MAX_WINDOWS + 1)
        ]
        client = mock_client(valid_plan_payload(windows=windows))
        with pytest.raises(ValueError, match="maximum is"):
            plan_study("anything", client=client)

    def test_rejects_duplicate_window_names(self) -> None:
        windows = [
            {"name": "same", "start": "0h", "end": "1h", "rationale": "x"},
            {"name": "same", "start": "-1h", "end": "0h", "rationale": "y"},
        ]
        client = mock_client(valid_plan_payload(windows=windows))
        with pytest.raises(ValueError, match="reused window name"):
            plan_study("anything", client=client)

    def test_rejects_a_backwards_window(self) -> None:
        windows = [{"name": "w", "start": "1h", "end": "0h", "rationale": "x"}]
        client = mock_client(valid_plan_payload(windows=windows))
        with pytest.raises(ValueError):
            plan_study("anything", client=client)


class TestRenderPlan:
    def test_shows_the_resolved_ambiguities(self) -> None:
        """The product promise: the reader sees the definitions before the
        numbers, and can reject one."""
        plan, spec = plan_study(
            "anything", client=mock_client(valid_plan_payload()), as_of=AS_OF
        )
        text = "\n".join(render_plan(plan, spec))

        assert "immediate reaction = 0h to +1h" in text
        assert "Assumed" in text
        assert spec.fingerprint in text


class TestInterpretationModel:
    def test_round_trips_through_json(self) -> None:
        interpretation = Interpretation(
            headline="BTC fell in 13 of 20 meetings.",
            findings=["Median post-event return was -0.208%."],
            caveats=["20 observations is a small sample."],
            suggested_followups=["Split by rate-hike versus rate-cut decisions."],
        )
        assert Interpretation.model_validate_json(interpretation.model_dump_json()) == (
            interpretation
        )


class TestPlannedStudyConversion:
    def test_normalises_case(self) -> None:
        plan = PlannedStudy(
            restated_question="q",
            symbol="btcusdt",
            event_type="fomc",
            event_count=5,
            interval="1H",
            windows=[PlannedWindow(name="w", start="0h", end="1h", rationale="r")],
            interpretation=[],
            assumptions=[],
        )
        spec = plan.to_spec()
        assert spec.symbol == "BTCUSDT"
        assert spec.interval is Interval.H1
