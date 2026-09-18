"""Anthropic Messages API client, over plain HTTP.

Why not the official SDK
------------------------
The ``anthropic`` package depends on ``jiter``, a Rust JSON parser shipped as a
compiled extension. Windows Application Control blocks that DLL on the target
machine, so ``import anthropic`` fails outright -- the SDK is not usable here at
any version. The same policy has already blocked pyarrow, Git's bundled libcurl,
and Git's bundled ssh; the pattern is unsigned native binaries.

``httpx`` is pure Python, already a dependency, and speaks the same endpoint.
See ``docs/decisions/005-anthropic-over-http.md``.

The tradeoff is that structured outputs must be assembled by hand rather than
via ``client.messages.parse()``: :func:`strict_json_schema` does what the SDK's
helper does, and the response is validated with the same Pydantic model.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Default model. Overridable via BACKTOOL_AI_MODEL.
DEFAULT_MODEL = "claude-opus-5"

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

#: Generous enough that a plan or interpretation is never truncated mid-object,
#: while staying well inside the request timeout.
MAX_TOKENS = 16000

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class AIError(RuntimeError):
    """The AI layer could not produce a usable result."""


class AINotConfiguredError(AIError):
    """No API credentials are available.

    Its own type so callers can fall back to the deterministic engine, which
    works perfectly well with no AI at all.
    """


def is_configured() -> bool:
    """Whether an API key is present, without attempting a request."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a JSON schema the structured-outputs API will accept.

    Pydantic emits ``$ref``/``$defs`` for nested models, marks only
    non-defaulted fields as required, and permits extra properties. Structured
    outputs need the opposite on all three counts, so this inlines every
    reference, requires every property, and forbids extras.

    Args:
        model: The Pydantic model the response must conform to.

    Returns:
        A self-contained, strict JSON schema.
    """
    raw = model.model_json_schema()
    definitions = raw.pop("$defs", {})
    hardened: dict[str, Any] = _harden(_inline(raw, definitions))
    return hardened


def _inline(node: Any, definitions: dict[str, Any]) -> Any:
    """Replace every ``$ref`` with the definition it points at."""
    if isinstance(node, dict):
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = definitions.get(reference.removeprefix("#/$defs/"), {})
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return _inline(merged, definitions)
        return {key: _inline(value, definitions) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline(item, definitions) for item in node]
    return node


def _harden(node: Any) -> Any:
    """Require every property and forbid extras, recursively."""
    if isinstance(node, dict):
        result = {
            key: _harden(value)
            for key, value in node.items()
            # `default` is meaningless once every field is required, and some
            # schema validators reject the combination.
            if key != "default"
        }
        if result.get("type") == "object" and "properties" in result:
            result["additionalProperties"] = False
            result["required"] = list(result["properties"])
        return result
    if isinstance(node, list):
        return [_harden(item) for item in node]
    return node


class AIClient:
    """Issues structured-output requests and validates them into Pydantic models."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise AINotConfiguredError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or run without --ask to use the deterministic engine "
                "on its own."
            )

        self._model: str = model or os.getenv("BACKTOOL_AI_MODEL") or DEFAULT_MODEL
        self._max_retries = max_retries
        self._owns_client = http_client is None
        # Sent per-request rather than baked into the client, so an injected
        # client (tests, a proxy, a custom transport) is still authenticated.
        self._headers = {
            "x-api-key": key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        self._http = http_client or httpx.Client(
            base_url=os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL),
            timeout=httpx.Timeout(300.0, connect=15.0),
        )

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> AIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "high",
    ) -> T:
        """Request a response constrained to ``schema`` and return it validated.

        Args:
            system: System prompt defining the role.
            user: The request content.
            schema: Pydantic model the response must conform to.
            effort: Thinking depth -- ``low`` through ``max``.

        Returns:
            A validated instance of ``schema``.

        Raises:
            AINotConfiguredError: the key was rejected.
            AIError: on refusal, truncation, transport failure, or a response
                that does not validate against ``schema``.
        """
        body = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
            },
            "messages": [{"role": "user", "content": user}],
        }

        payload = self._post(body)
        return self._parse(payload, schema)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_error: str | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http.post("/v1/messages", json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                logger.warning("Request failed (%s), attempt %d", exc, attempt + 1)
                continue

            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            if response.status_code in {401, 403}:
                raise AINotConfiguredError(
                    f"Anthropic rejected the API key (HTTP {response.status_code}). "
                    "Check ANTHROPIC_API_KEY in your .env."
                )
            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    "Anthropic returned %s (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    self._max_retries,
                )
                continue

            raise AIError(
                f"Anthropic API error {response.status_code}: {response.text[:300]}"
            )

        raise AIError(f"Anthropic API unreachable after {self._max_retries} attempts: {last_error}")

    def _parse(self, payload: dict[str, Any], schema: type[T]) -> T:
        stop_reason = payload.get("stop_reason")
        if stop_reason == "refusal":
            detail = (payload.get("stop_details") or {}).get("explanation") or ""
            raise AIError(f"The model declined this request. {detail}".strip())
        if stop_reason == "max_tokens":
            raise AIError(
                "The model's response was cut off before it was complete. "
                "Try a simpler question or fewer windows."
            )

        text = next(
            (
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            ),
            None,
        )
        if not text:
            raise AIError("The model returned no structured output.")

        usage = payload.get("usage", {})
        logger.debug(
            "%s: %s in / %s out tokens",
            self._model,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )

        try:
            return schema.model_validate(json.loads(text))
        except json.JSONDecodeError as exc:
            raise AIError(f"The model's response was not valid JSON: {exc}") from exc
        except ValidationError as exc:
            raise AIError(f"The model's response did not match the expected shape: {exc}") from exc
