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
from pydantic import BaseModel, ConfigDict, ValidationError

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


#: Most continuations the client will follow when the API pauses a long server
#: tool turn. The API returns ``stop_reason: "pause_turn"`` mid-search and
#: expects the assistant message back unchanged to carry on; that is finishing
#: one request, not running an agent, so the cap is deliberately small and
#: exhausting it is an error rather than a result.
MAX_TOOL_CONTINUATIONS = 3


#: Separator between several passages quoted from the same page.
#:
#: A page can be cited more than once in one turn, and each citation quotes a
#: different sentence. Both are genuinely from that page, so both are kept, and
#: this is the seam that lets :attr:`Citation.passages` recover them -- which is
#: what makes merging repeatable rather than something that compounds.
PASSAGE_SEPARATOR = "\n...\n"


class Citation(BaseModel):
    """A source the model cited, as the API reported it.

    Carried verbatim. The point of a citation is that a reader can check it,
    which it cannot be if the application paraphrased it on the way past.

    One page can arrive several times in one response, in blocks of different
    richness: a search result carries a url and a title, and the text block
    that later quotes that same page carries the sentence. They are records of
    one source, so :meth:`merged_with` combines them.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    title: str = ""
    #: Every passage quoted from this page, joined by :data:`PASSAGE_SEPARATOR`.
    #: Empty when the page was listed but never quoted.
    cited_text: str = ""

    @property
    def passages(self) -> tuple[str, ...]:
        """The distinct passages quoted from this page."""
        return tuple(
            passage.strip()
            for passage in self.cited_text.split(PASSAGE_SEPARATOR)
            if passage.strip()
        )

    def merged_with(self, other: Citation) -> Citation:
        """One record of this page, keeping the richest form of every part.

        Defined so the outcome cannot depend on arrival order: titles resolve by
        a rule rather than by precedence, and passages are unioned and sorted.
        That matters because the API reports the weaker record *first* -- a
        search result lists the page, and only the later text block quotes it --
        so any rule where the earlier record wins silently discards the only
        text a date could be checked against.

        Raises:
            ValueError: if the records describe different pages.
        """
        if other.url != self.url:
            raise ValueError(
                f"Refusing to merge records of different pages: {self.url!r} "
                f"and {other.url!r}."
            )
        # Longest wins, with the text itself breaking a tie. A search result and
        # a citation usually title a page identically; where they differ, the
        # fuller one is the more useful, and either way the answer is the same
        # whichever arrived first.
        title = max(self.title, other.title, key=lambda text: (len(text), text))
        passages = sorted({*self.passages, *other.passages})
        return Citation(
            url=self.url,
            title=title,
            cited_text=PASSAGE_SEPARATOR.join(passages),
        )


class ServerToolOutcome(BaseModel):
    """What came back from a request that used Anthropic-run tools.

    Deliberately plain: text, citations, counts, errors. This client knows how
    to *call* a server tool and how to read the envelope safely; it does not
    know what any particular tool is for, and nothing here is specific to
    search. The caller supplies the tool configuration and interprets the text.
    """

    model_config = ConfigDict(frozen=True)

    #: Everything the model wrote, concatenated. Untrusted -- it contains
    #: whatever the fetched pages said.
    text: str
    citations: tuple[Citation, ...] = ()
    #: Queries the model issued, in order, for showing its working.
    queries: tuple[str, ...] = ()
    #: ``usage.server_tool_use.web_search_requests``. Billed per search on top
    #: of tokens, so a caller that cares about cost can read it.
    search_requests: int = 0
    #: Error codes reported *inside* a 200 response. A tool failure is not an
    #: HTTP failure, so without this a rate-limited search is indistinguishable
    #: from a search that found nothing.
    tool_errors: tuple[str, ...] = ()
    usage: dict[str, int] = {}

    @property
    def had_tool_error(self) -> bool:
        return bool(self.tool_errors)


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


#: JSON Schema keywords the structured-outputs validator rejects.
#:
#: `default` is meaningless once every field is required. The numeric range
#: keywords are refused outright -- a schema carrying them returns
#: "For 'integer' type, properties maximum, minimum are not supported".
#: Pydantic emits them from `Field(ge=..., le=...)`, so they have to be
#: stripped rather than avoided at the model.
#:
#: Dropping them does not drop the constraint: the response is still validated
#: with `Model.model_validate()`, which enforces the original bounds. The schema
#: stops *describing* the limit, so each bounded field states it in its
#: description instead -- otherwise the model has no way to know.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)


def _harden(node: Any) -> Any:
    """Require every property, forbid extras, and drop rejected keywords."""
    if isinstance(node, dict):
        result = {
            key: _harden(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_KEYWORDS
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
        self._last_usage: dict[str, int] = {}
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

    @property
    def last_usage(self) -> dict[str, int]:
        """Token counts from the most recent successful request.

        Kept so the cost of a call can be reported rather than estimated. Empty
        before the first request, and not cleared by a failure -- a retry that
        eventually succeeds overwrites it, and nothing else reads it.
        """
        return dict(self._last_usage)

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
        effort: str | None = "medium",
        cache_system: bool = True,
        thinking: bool = True,
    ) -> T:
        """Request a response constrained to ``schema`` and return it validated.

        Args:
            system: System prompt defining the role.
            user: The request content.
            schema: Pydantic model the response must conform to.
            effort: Reasoning depth -- ``low`` through ``max``. Defaults to
                ``medium``: this client only ever asks for a structured object
                against a fixed schema, and the deep-reasoning setting bills
                for thinking that translation does not need. ``None`` omits the
                setting, which is right when ``thinking`` is off and there is no
                reasoning to budget for.
            cache_system: Whether to mark the system prompt cacheable. Worth it
                whenever the same prompt is used more than once in a few
                minutes, which is every batch run -- the prompt is identical
                across calls and only the question changes, so paying full
                price for it each time is pure waste.
            thinking: Whether to request extended thinking. Off for work that
                is transcription rather than reasoning -- copying established
                facts into fields earns nothing from deliberation, and asking
                for it bills for tokens the task does not use. Also what makes
                the smallest models usable for such a step: several support
                neither extended thinking nor a reasoning effort, and a job
                that needs neither should not be demanding both.

        Returns:
            A validated instance of ``schema``.

        Raises:
            AINotConfiguredError: the key was rejected.
            AIError: on refusal, truncation, transport failure, or a response
                that does not validate against ``schema``.
        """
        # Sent as a block list rather than a string so the cache breakpoint can
        # be attached. A cache read is billed at a fraction of a fresh read,
        # and this prompt carries the whole vocabulary.
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cache_system:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": strict_json_schema(schema)}
        }
        if effort is not None:
            output_config["effort"] = effort

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": system_blocks,
            "output_config": output_config,
            "messages": [{"role": "user", "content": user}],
        }
        if thinking:
            body["thinking"] = {"type": "adaptive"}

        payload = self._post(body)
        return self._parse(payload, schema)

    def with_server_tools(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        effort: str = "medium",
        max_tokens: int | None = None,
    ) -> ServerToolOutcome:
        """Run a request that may use Anthropic-hosted tools, and read it back.

        Server tools run on Anthropic's infrastructure and their results arrive
        as content blocks in the same response, so there is no execution loop
        here and none is wanted. The one exception is ``pause_turn``: the API
        may pause a long tool turn and expects the assistant message returned
        unchanged to continue. That is bounded by
        :data:`MAX_TOOL_CONTINUATIONS`.

        Deliberately not :meth:`structured`. A tool-using turn produces prose
        and citations, and asking for a schema at the same time would make the
        thing reading untrusted pages also the thing filling in the fields the
        rest of the system trusts. Extraction is a separate, tool-free call --
        see :mod:`backtool.ai.event_resolver`.

        Args:
            system: System prompt.
            user: The request.
            tools: Tool definitions, passed through verbatim. This client does
                not know or validate what they are: the caller owns that
                choice, including domain restrictions and usage caps.
            effort: Thinking depth.
            max_tokens: Override the default ceiling.

        Returns:
            A :class:`ServerToolOutcome`. Tool errors are reported on it rather
            than raised -- a failed search is a fact about the result, and a
            partial answer is usually still worth reading.

        Raises:
            AIError: on transport failure, refusal, truncation, or a turn that
                never stopped pausing.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        blocks: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        searches = 0

        for _ in range(MAX_TOOL_CONTINUATIONS + 1):
            payload = self._post(
                {
                    "model": self._model,
                    "max_tokens": max_tokens or MAX_TOKENS,
                    "system": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": effort},
                    "tools": tools,
                    "messages": messages,
                }
            )
            self._refuse_if_declined(payload)

            content = payload.get("content") or []
            blocks.extend(block for block in content if isinstance(block, dict))
            searches += _search_requests(payload)
            for key, value in (payload.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value

            if payload.get("stop_reason") != "pause_turn":
                # Recorded here as well as returned, so `last_usage` means the
                # same thing on both paths. It did not, and the tool-using
                # path -- the expensive one -- was the half that went
                # unrecorded.
                self._last_usage = dict(usage)
                return _read_blocks(blocks, searches, usage)

            # Paused mid-tool. The assistant turn goes back exactly as it came,
            # including each result's encrypted_content, or the API rejects it.
            messages.append({"role": "assistant", "content": content})

        raise AIError(
            f"The request was still paused after {MAX_TOOL_CONTINUATIONS} "
            "continuations. Narrow the question, or lower the tool's max_uses."
        )

    def _refuse_if_declined(self, payload: dict[str, Any]) -> None:
        """Stop-reason handling, shared by both request shapes."""
        stop_reason = payload.get("stop_reason")
        if stop_reason == "refusal":
            detail = (payload.get("stop_details") or {}).get("explanation") or ""
            raise AIError(f"The model declined this request. {detail}".strip())
        if stop_reason == "max_tokens":
            raise AIError(
                "The model's response was cut off before it was complete. "
                "Try a simpler question or fewer windows."
            )

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
        self._refuse_if_declined(payload)

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
        self._last_usage = {
            key: int(value)
            for key, value in usage.items()
            if isinstance(value, int | float)
        }
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


def _search_requests(payload: dict[str, object]) -> int:
    """How many searches this response was billed for.

    Reported under ``usage.server_tool_use``. Each search counts once whatever
    it returned, and a search that errored is not billed at all.
    """
    usage = payload.get("usage")
    server = usage.get("server_tool_use") if isinstance(usage, dict) else None
    count = server.get("web_search_requests", 0) if isinstance(server, dict) else 0
    return count if isinstance(count, int) else 0


def _read_blocks(
    blocks: list[dict[str, Any]], searches: int, usage: dict[str, int]
) -> ServerToolOutcome:
    """Read a tool-using response without trusting its shape.

    Every field is probed rather than assumed. This is the one place in the
    project that parses a payload whose contents were shaped by whatever a web
    page contained, and a missing key, a string where a list belongs, or a
    result with no url should not be an unhandled exception that a page author
    can trigger.

    The branch that matters is the tool result: on success its ``content`` is a
    *list* of results, on failure a single error *object*. Indexing without
    checking would turn a rate-limited search into a crash.

    A page can appear in several blocks of one response, and the order is fixed
    against us: the search result that merely *lists* it always precedes the
    text block that *quotes* it. Records are therefore merged per url rather
    than deduplicated, because dropping the later one throws away the only text
    a claim could be checked against -- which is exactly what it did.
    """
    text_parts: list[str] = []
    queries: list[str] = []
    errors: list[str] = []
    # Keyed by url, so a page reported twice becomes one record rather than a
    # race between its two forms. Insertion order is kept, so the citation list
    # still reads in the order the response mentioned each page.
    found: dict[str, Citation] = {}

    def remember(citation: Citation) -> None:
        existing = found.get(citation.url)
        found[citation.url] = (
            existing.merged_with(citation) if existing else citation
        )

    for block in blocks:
        kind = block.get("type")

        if kind == "text":
            if isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            for raw in block.get("citations") or []:
                if not isinstance(raw, dict):
                    continue
                url = raw.get("url")
                if not isinstance(url, str) or not url:
                    continue
                remember(
                    Citation(
                        url=url,
                        title=str(raw.get("title") or ""),
                        cited_text=str(raw.get("cited_text") or ""),
                    )
                )

        elif kind == "server_tool_use":
            raw_input = block.get("input")
            query = raw_input.get("query") if isinstance(raw_input, dict) else None
            if isinstance(query, str):
                queries.append(query)

        elif isinstance(kind, str) and kind.endswith("_tool_result"):
            content = block.get("content")
            if isinstance(content, dict):
                code = content.get("error_code")
                errors.append(str(code) if code else "unknown_tool_error")
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    url = item.get("url")
                    if isinstance(url, str) and url:
                        remember(
                            Citation(url=url, title=str(item.get("title") or ""))
                        )

    return ServerToolOutcome(
        text="".join(text_parts),
        citations=tuple(found.values()),
        queries=tuple(queries),
        search_requests=searches,
        tool_errors=tuple(errors),
        usage=usage,
    )
