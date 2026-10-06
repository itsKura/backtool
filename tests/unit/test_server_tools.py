"""Tests for server-tool support in AIClient.

Two things to prove, and the second matters more.

**The new path reads the envelope safely.** This is the only place in the
project that parses a payload whose contents were shaped by whatever a web page
contained, so a missing key, a wrong type, or an error where a list belongs
must not be an unhandled exception a page author can trigger.

**The old paths are untouched.** The planner and the interpreter are working,
recorded against, and expensive to re-verify. A request without tools must come
out byte-identical to what it was before server tools existed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from backtool.ai.client import (
    MAX_TOOL_CONTINUATIONS,
    PASSAGE_SEPARATOR,
    AIClient,
    AIError,
    Citation,
    ServerToolOutcome,
)
from backtool.ai.models import PlannedStudy

SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


def client_for(
    responses: list[dict[str, Any]], capture: list[dict[str, Any]] | None = None
) -> AIClient:
    """A client that returns each response in turn."""
    remaining = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(json.loads(request.content))
        return httpx.Response(200, json=remaining.pop(0) if remaining else responses[-1])

    return AIClient(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://example.test"
        ),
        max_retries=1,
    )


def run(
    content: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None = None,
    stop_reason: str = "end_turn",
) -> ServerToolOutcome:
    client = client_for(
        [
            {
                "content": content,
                "stop_reason": stop_reason,
                "usage": usage or {"input_tokens": 10, "output_tokens": 5},
            }
        ]
    )
    return client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])


class TestOrdinaryCallsAreUnchanged:
    """The planner and interpreter must not notice this existed."""

    def test_a_structured_call_sends_no_tools(self) -> None:
        captured: list[dict[str, Any]] = []
        client = client_for(
            [
                {
                    "content": [{"type": "text", "text": json.dumps(_plan())}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ],
            capture=captured,
        )
        client.structured(system="s", user="u", schema=PlannedStudy)

        assert "tools" not in captured[0]
        assert captured[0]["output_config"]["format"]["type"] == "json_schema"

    def test_a_structured_call_keeps_its_shape(self) -> None:
        """Pinned field by field: the request a working planner sends."""
        captured: list[dict[str, Any]] = []
        client = client_for(
            [
                {
                    "content": [{"type": "text", "text": json.dumps(_plan())}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ],
            capture=captured,
        )
        client.structured(system="SYS", user="USR", schema=PlannedStudy)

        body = captured[0]
        assert set(body) == {
            "model",
            "max_tokens",
            "system",
            "thinking",
            "output_config",
            "messages",
        }
        assert body["system"][0]["text"] == "SYS"
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert body["messages"] == [{"role": "user", "content": "USR"}]
        assert body["thinking"] == {"type": "adaptive"}

    def test_refusal_handling_is_shared_not_duplicated(self) -> None:
        """Both paths route through one stop-reason check, so they cannot
        drift into disagreeing about what a refusal is."""
        for call in ("structured", "tools"):
            client = client_for(
                [
                    {
                        "content": [],
                        "stop_reason": "refusal",
                        "stop_details": {"explanation": "policy"},
                        "usage": {},
                    }
                ]
            )
            with pytest.raises(AIError, match="declined this request"):
                if call == "structured":
                    client.structured(system="s", user="u", schema=PlannedStudy)
                else:
                    client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])

    def test_truncation_is_reported_on_both_paths(self) -> None:
        for call in ("structured", "tools"):
            client = client_for(
                [{"content": [], "stop_reason": "max_tokens", "usage": {}}]
            )
            with pytest.raises(AIError, match="cut off"):
                if call == "structured":
                    client.structured(system="s", user="u", schema=PlannedStudy)
                else:
                    client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])


class TestToolRequestShape:
    """What a tool-using request sends."""

    def test_the_tools_are_passed_through_verbatim(self) -> None:
        """The client does not know what a tool is for. The caller owns the
        configuration, including domain limits and usage caps."""
        captured: list[dict[str, Any]] = []
        client = client_for(
            [{"content": [], "stop_reason": "end_turn", "usage": {}}], capture=captured
        )
        odd = {"type": "something_else_20991231", "name": "x", "custom": [1, 2]}
        client.with_server_tools(system="s", user="u", tools=[odd])
        assert captured[0]["tools"] == [odd]

    def test_no_output_format_is_requested(self) -> None:
        """A schema here would make the stage reading untrusted pages the same
        stage filling in trusted fields."""
        captured: list[dict[str, Any]] = []
        client = client_for(
            [{"content": [], "stop_reason": "end_turn", "usage": {}}], capture=captured
        )
        client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])
        assert "format" not in captured[0]["output_config"]

    def test_the_system_prompt_is_still_cached(self) -> None:
        captured: list[dict[str, Any]] = []
        client = client_for(
            [{"content": [], "stop_reason": "end_turn", "usage": {}}], capture=captured
        )
        client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])
        assert captured[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


class TestReadingResults:
    """The documented response shape, and everything adjacent to it."""

    def test_text_and_citations_are_collected(self) -> None:
        outcome = run(
            [
                {
                    "type": "text",
                    "text": "Shannon was born in 1916",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                            "title": "Claude Shannon",
                            "cited_text": "born April 30, 1916",
                        }
                    ],
                }
            ]
        )
        assert outcome.text == "Shannon was born in 1916"
        assert outcome.citations[0].url.endswith("Claude_Shannon")
        assert outcome.citations[0].cited_text == "born April 30, 1916"

    def test_queries_are_recorded(self) -> None:
        outcome = run(
            [
                {
                    "type": "server_tool_use",
                    "id": "s1",
                    "name": "web_search",
                    "input": {"query": "election dates"},
                }
            ]
        )
        assert outcome.queries == ("election dates",)

    def test_result_urls_become_citations(self) -> None:
        outcome = run(
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "s1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://fec.gov/",
                            "title": "FEC",
                        }
                    ],
                }
            ]
        )
        assert outcome.citations[0].url == "https://fec.gov/"

    def test_duplicate_urls_appear_once_and_keep_their_content(self) -> None:
        """One record per page -- and the record is the richest one.

        The original version of this test asserted only the count, which is
        exactly how the live defect stayed invisible: the url survived, the
        quoted text did not, and nothing here noticed.
        """
        outcome = run(
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "s1",
                    "content": [
                        {"type": "web_search_result", "url": "https://a/", "title": "A"},
                        {"type": "web_search_result", "url": "https://a/", "title": "A"},
                    ],
                },
                {
                    "type": "text",
                    "text": "x",
                    "citations": [
                        {"url": "https://a/", "title": "A", "cited_text": "the quote"}
                    ],
                },
            ]
        )
        assert len(outcome.citations) == 1
        assert outcome.citations[0].cited_text == "the quote"

    def test_the_search_count_comes_from_usage(self) -> None:
        outcome = run(
            [{"type": "text", "text": "x"}],
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "server_tool_use": {"web_search_requests": 3},
            },
        )
        assert outcome.search_requests == 3
        assert outcome.usage["input_tokens"] == 100

    def test_no_server_tool_usage_means_no_searches(self) -> None:
        assert run([{"type": "text", "text": "x"}]).search_requests == 0


class TestToolErrorsInsideA200:
    """A failed search is not a failed request."""

    def test_an_error_object_is_detected(self) -> None:
        outcome = run(
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "s1",
                    "content": {
                        "type": "web_search_tool_result_error",
                        "error_code": "max_uses_exceeded",
                    },
                }
            ]
        )
        assert outcome.had_tool_error
        assert outcome.tool_errors == ("max_uses_exceeded",)

    def test_an_error_does_not_raise(self) -> None:
        """A capped search still leaves whatever was found before it."""
        outcome = run(
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "s1",
                    "content": {"type": "web_search_tool_result_error",
                                "error_code": "too_many_requests"},
                },
                {"type": "text", "text": "partial answer"},
            ]
        )
        assert outcome.text == "partial answer"

    def test_an_error_with_no_code_is_still_an_error(self) -> None:
        outcome = run(
            [{"type": "web_search_tool_result", "tool_use_id": "s1", "content": {}}]
        )
        assert outcome.tool_errors == ("unknown_tool_error",)

    def test_an_empty_result_list_is_not_an_error(self) -> None:
        """A search that matched nothing is a different fact from one that
        failed."""
        outcome = run(
            [{"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}]
        )
        assert not outcome.had_tool_error


class TestMalformedPayloads:
    """Nothing a page can shape should reach an unhandled exception."""

    @pytest.mark.parametrize(
        "content",
        [
            [],
            [{"type": "text"}],
            [{"type": "text", "text": None}],
            [{"type": "text", "text": "x", "citations": "not a list"}],
            [{"type": "text", "text": "x", "citations": [None, 42]}],
            [{"type": "text", "text": "x", "citations": [{"title": "no url"}]}],
            [{"type": "server_tool_use"}],
            [{"type": "server_tool_use", "input": "not a dict"}],
            [{"type": "web_search_tool_result"}],
            [{"type": "web_search_tool_result", "content": "a string"}],
            [{"type": "web_search_tool_result", "content": [None, 7, "x"]}],
            [{"type": "web_search_tool_result", "content": [{"no": "url"}]}],
            [{"type": "unknown_future_block", "whatever": True}],
            ["not a block at all"],
        ],
    )
    def test_it_survives(self, content: list[Any]) -> None:
        outcome = run(content)
        assert isinstance(outcome, ServerToolOutcome)

    def test_missing_usage_is_not_fatal(self) -> None:
        client = client_for([{"content": [{"type": "text", "text": "x"}]}])
        outcome = client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])
        assert outcome.search_requests == 0

    def test_a_non_integer_search_count_is_ignored(self) -> None:
        outcome = run(
            [{"type": "text", "text": "x"}],
            usage={"server_tool_use": {"web_search_requests": "many"}},
        )
        assert outcome.search_requests == 0


class TestPausedTurns:
    """The API may pause a long tool turn. Finishing it is not an agent loop."""

    def test_a_paused_turn_is_continued(self) -> None:
        captured: list[dict[str, Any]] = []
        client = client_for(
            [
                {
                    "content": [{"type": "text", "text": "first "}],
                    "stop_reason": "pause_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                {
                    "content": [{"type": "text", "text": "second"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 20, "output_tokens": 7},
                },
            ],
            capture=captured,
        )
        outcome = client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])

        assert outcome.text == "first second"
        assert len(captured) == 2

    def test_the_assistant_turn_goes_back_unchanged(self) -> None:
        """The API rejects a modified one -- each result's encrypted_content
        has to survive the round trip exactly."""
        captured: list[dict[str, Any]] = []
        block = {
            "type": "web_search_tool_result",
            "tool_use_id": "s1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://a/",
                    "title": "A",
                    "encrypted_content": "EqgfCioIARgBIiQ3",
                }
            ],
        }
        client = client_for(
            [
                {"content": [block], "stop_reason": "pause_turn", "usage": {}},
                {"content": [{"type": "text", "text": "done"}],
                 "stop_reason": "end_turn", "usage": {}},
            ],
            capture=captured,
        )
        client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])

        resumed = captured[1]["messages"][1]
        assert resumed["role"] == "assistant"
        assert resumed["content"] == [block]

    def test_usage_accumulates_across_continuations(self) -> None:
        client = client_for(
            [
                {
                    "content": [],
                    "stop_reason": "pause_turn",
                    "usage": {"input_tokens": 10,
                              "server_tool_use": {"web_search_requests": 2}},
                },
                {
                    "content": [],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 15,
                              "server_tool_use": {"web_search_requests": 1}},
                },
            ]
        )
        outcome = client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])
        assert outcome.search_requests == 3
        assert outcome.usage["input_tokens"] == 25

    def test_endless_pausing_is_bounded(self) -> None:
        """A cap, not a retry policy. This is finishing one request."""
        client = client_for(
            [{"content": [], "stop_reason": "pause_turn", "usage": {}}]
        )
        with pytest.raises(AIError, match="still paused"):
            client.with_server_tools(system="s", user="u", tools=[SEARCH_TOOL])

    def test_the_cap_is_small(self) -> None:
        assert MAX_TOOL_CONTINUATIONS <= 5


LISTED = {
    "type": "web_search_tool_result",
    "tool_use_id": "s1",
    "content": [
        {
            "type": "web_search_result",
            "url": "https://ethereum.org/merge/",
            "title": "The Merge | ethereum.org",
        }
    ],
}

QUOTED = {
    "type": "text",
    "text": "The Merge completed in September 2022.",
    "citations": [
        {
            "type": "web_search_result_location",
            "url": "https://ethereum.org/merge/",
            "title": "The Merge",
            "cited_text": "The Merge was completed on September 15, 2022 at 06:42 UTC.",
        }
    ],
}


class TestRepeatedSourcesAreMerged:
    """One page, reported twice, in blocks of different richness.

    This is the defect a live run exposed and no fixture had. The API reports
    the same page in two places: a search result that *lists* it, carrying a url
    and a title, and a text block that *quotes* it, carrying the sentence. The
    listing always comes first, so a reader that dedupes on url and keeps the
    first record throws away every excerpt -- which left the grounding layer
    with nothing to check any date against on both live resolutions.
    """

    def test_listed_then_quoted_keeps_the_quote(self) -> None:
        """The live order, exactly."""
        outcome = run([LISTED, QUOTED])
        assert len(outcome.citations) == 1
        assert "September 15, 2022 at 06:42 UTC" in outcome.citations[0].cited_text

    def test_quoted_then_listed_keeps_the_quote(self) -> None:
        """The reverse order, for the same reason: a later, weaker record must
        not be able to overwrite what was already established."""
        outcome = run([QUOTED, LISTED])
        assert len(outcome.citations) == 1
        assert "September 15, 2022 at 06:42 UTC" in outcome.citations[0].cited_text

    def test_both_orders_produce_the_same_evidence(self) -> None:
        """Stated as a property rather than as two assertions, because order
        independence is the actual requirement."""
        forward = run([LISTED, QUOTED]).citations
        reverse = run([QUOTED, LISTED]).citations
        assert forward == reverse

    def test_the_title_survives_from_whichever_block_had_it(self) -> None:
        bare = {
            "type": "text",
            "text": "x",
            "citations": [{"url": "https://a/", "title": "", "cited_text": "quote"}],
        }
        titled = {
            "type": "web_search_tool_result",
            "tool_use_id": "s1",
            "content": [
                {"type": "web_search_result", "url": "https://a/", "title": "A Page"}
            ],
        }
        for blocks in ([bare, titled], [titled, bare]):
            outcome = run(blocks)
            assert outcome.citations[0].title == "A Page"
            assert outcome.citations[0].cited_text == "quote"

    def test_the_same_excerpt_twice_is_not_two_excerpts(self) -> None:
        outcome = run([QUOTED, dict(QUOTED)])
        assert len(outcome.citations) == 1
        assert outcome.citations[0].passages == (
            "The Merge was completed on September 15, 2022 at 06:42 UTC.",
        )

    def test_two_different_excerpts_are_both_kept(self) -> None:
        """A page quoted twice usually quoted two different sentences, and both
        are genuinely from it. Keeping only one would discard evidence -- and
        the one discarded could easily be the sentence carrying the time.
        """
        second = {
            "type": "text",
            "text": "y",
            "citations": [
                {
                    "url": "https://ethereum.org/merge/",
                    "title": "The Merge",
                    "cited_text": "Block 15537393 was the last proof-of-work block.",
                }
            ],
        }
        outcome = run([LISTED, QUOTED, second])
        assert len(outcome.citations) == 1
        passages = outcome.citations[0].passages
        assert len(passages) == 2
        assert any("06:42 UTC" in passage for passage in passages)
        assert any("15537393" in passage for passage in passages)

    def test_merging_is_repeatable(self) -> None:
        """Merging a record into itself changes nothing, which is what keeps
        several rounds of search from compounding the same quote."""
        one = Citation(url="https://a/", title="A", cited_text="first")
        two = Citation(url="https://a/", title="A", cited_text="second")
        both = one.merged_with(two)
        assert both.merged_with(two) == both
        assert both.merged_with(one) == both
        assert two.merged_with(one) == both

    def test_separate_pages_are_not_merged(self) -> None:
        outcome = run(
            [
                LISTED,
                {
                    "type": "text",
                    "text": "x",
                    "citations": [{"url": "https://other/", "cited_text": "elsewhere"}],
                },
            ]
        )
        assert {c.url for c in outcome.citations} == {
            "https://ethereum.org/merge/",
            "https://other/",
        }

    def test_merging_different_pages_is_refused(self) -> None:
        """A programming error, not a data one. Silently picking one url would
        attach a quote to a page it did not come from."""
        with pytest.raises(ValueError, match="different pages"):
            Citation(url="https://a/").merged_with(Citation(url="https://b/"))

    def test_passages_round_trip_through_the_separator(self) -> None:
        joined = PASSAGE_SEPARATOR.join(["one", "two"])
        assert Citation(url="https://a/", cited_text=joined).passages == ("one", "two")

    def test_a_citation_with_no_url_is_skipped(self) -> None:
        outcome = run([{"type": "text", "text": "x", "citations": [{"url": ""}]}])
        assert outcome.citations == ()

    def test_a_request_without_tools_is_unaffected(self) -> None:
        """The merge lives on the tool-using path. A planner call has no
        citations to merge and must behave exactly as it did before."""
        captured: list[dict[str, Any]] = []
        client = client_for(
            [
                {
                    "content": [{"type": "text", "text": json.dumps(_plan())}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ],
            capture=captured,
        )
        plan = client.structured(system="s", user="u", schema=PlannedStudy)
        assert plan.symbol == "BTCUSDT"
        assert "tools" not in captured[0]


def _plan() -> dict[str, Any]:
    return {
        "restated_question": "q",
        "symbol": "BTCUSDT",
        "event_type": "FOMC",
        "event_count": 20,
        "interval": "5m",
        "windows": [{"name": "post_event_1h", "start": "0h", "end": "1h",
                     "rationale": "r"}],
        "interpretation": [],
        "assumptions": [],
    }
