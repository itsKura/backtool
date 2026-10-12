"""What the page remembers, and how it asks.

Both of these are one bug seen from two ends, and the bug was reported by
someone using the product rather than found by a test -- which is worth saying
plainly, because the suite was green while it happened.

They asked about Bitcoin around a presidential inauguration. They were asked
which inauguration and where in the day to measure from; they answered; they
were asked the same two things again. Four rounds, no measurement. The page
carried the thesis forward and nothing else, and a turn that *ends in a
question* has no thesis -- so every answer arrived as a brand new question with
no record of what it was answering, and the only honest thing left to do with
it was ask again.

The second half is the same failure earlier: a question that can only be
answered by typing asks two things at once, the real one and "which words will
be understood". The answers now come from the step that will read them back.
"""

from __future__ import annotations

import json
from html import unescape
from typing import Any

import httpx
import pytest
from tests.fixtures.dynamic_events import eth_merge, unresolvable

from backtool.ai.client import AIClient
from backtool.ai.thesis_planner import (
    ANCHOR_CHOICES,
    THESIS_PLANNER_SYSTEM,
    PlannedThesis,
    plan_thesis,
)
from backtool.events.resolution import AnchorPolicy
from backtool.research.thesis import (
    MAX_REMEMBERED_EXCHANGES,
    Exchange,
    Outcome,
    Thesis,
)
from backtool.web.app import _conversation
from backtool.web.research_page import (
    RESEARCH_SCRIPT,
    render_clarification,
    render_spec_card,
    render_unresolved_event,
    render_unsupported,
)

# The exchange that actually happened, trimmed to what matters.
ASKED_WHICH = (
    "Which inauguration do you mean -- the first term beginning 20 January "
    "2017, or the second beginning 20 January 2025?"
)
ASKED_WHERE = "Where in the day should the measurement start?"


def planned(**overrides: Any) -> PlannedThesis:
    base: dict[str, Any] = {
        "restated_question": "How did BTC perform after the inauguration?",
        "asset": "BTCUSDT",
        "event_type": "",
        "event_concept": "presidential inauguration",
        "event_anchor": "",
        "conditions": [],
        "outcome_metric": "return_pct",
        "outcome_horizon": "24h",
        "aggregation": "median",
        "order": "",
        "limit": "",
        "outcome_threshold": "",
        "outcome_threshold_operator": "",
        "comparison_conditions": [],
        "compare_to_baseline": False,
        "assumptions": [],
        "clarifications": [],
        "unsupported": [],
    }
    base.update(overrides)
    return PlannedThesis.model_validate(base)


def asking(*questions: str | dict[str, Any]) -> PlannedThesis:
    return planned(clarifications=list(questions))


def dynamic_thesis(**overrides: Any) -> Thesis:
    base: dict[str, Any] = {
        "question": "What happened to BTC after the inauguration?",
        "asset": "BTCUSDT",
        "event_type": None,
        "resolution": eth_merge(),
        "outcome": Outcome(),
    }
    base.update(overrides)
    return Thesis(**base)


def remembered(html: str) -> list[dict[str, Any]]:
    """Every exchange the rendered card would hand to the next turn.

    Parsed the way the browser parses it, so a block that is valid HTML and
    invalid JSON fails here rather than silently emptying the page's memory.
    """
    found: list[dict[str, Any]] = []
    marker = '<script type="application/json" class="exchange">'
    rest = html
    while marker in rest:
        rest = rest.split(marker, 1)[1]
        body, rest = rest.split("</script>", 1)
        found.append(json.loads(body))
    return found


class Recorder:
    """A planner client that answers blandly and keeps what it was asked.

    What is under test here is the *payload* -- whether the thread reached the
    planner and in what shape -- so the answer is a constant and the question
    is the evidence. No network, no key, no cost.
    """

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": planned().model_dump_json()}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

    @property
    def client(self) -> AIClient:
        return AIClient(
            api_key="test-key",
            http_client=httpx.Client(
                transport=httpx.MockTransport(self),
                base_url="https://example.test",
            ),
            max_retries=1,
        )

    @property
    def payload(self) -> dict[str, Any]:
        """What the planner was given, as it read it."""
        return json.loads(self.bodies[-1]["messages"][0]["content"])


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


# --------------------------------------------------------------------------


class TestATurnThatAsksSomethingRemembersAsking:
    """The regression, card by card.

    Each of these ends in a question and has no thesis to carry. Before this
    they rendered the question and kept nothing, so the answer came back to a
    planner that had never heard of it.
    """

    def test_a_clarification_records_what_it_asked(self) -> None:
        html = render_clarification(
            asking(ASKED_WHICH, ASKED_WHERE), question="btc on Trump's first day?"
        )
        turn = remembered(html)[0]
        assert turn["question"] == "btc on Trump's first day?"
        assert turn["asked"] == [ASKED_WHICH, ASKED_WHERE]

    def test_it_records_how_the_question_was_read(self) -> None:
        html = render_clarification(asking(ASKED_WHICH), question="q")
        assert remembered(html)[0]["restated"] == (
            "How did BTC perform after the inauguration?"
        )

    def test_a_refusal_remembers_the_reading_it_refused(self) -> None:
        """A reader rephrases after a refusal, and the rephrasing should be
        read against what was refused rather than from nothing."""
        html = render_unsupported(
            planned(unsupported=["Funding rates are not connected."]), question="q"
        )
        assert remembered(html)[0]["restated"]

    def test_an_unsettled_lookup_records_its_questions(self) -> None:
        html = render_unresolved_event(unresolvable(), question="which crisis?")
        turn = remembered(html)[0]
        assert turn["question"] == "which crisis?"
        assert turn["asked"]

    def test_a_card_with_no_question_is_not_remembered_as_a_blank(self) -> None:
        """A turn that asked nothing and restated nothing spends tokens saying
        that something happened and not what."""
        assert not Exchange(question="q").is_worth_remembering
        assert Exchange(question="q", asked=("why?",)).is_worth_remembering
        assert Exchange(question="q", restated="read as this").is_worth_remembering


class TestTheThreadReachesThePlanner:
    """Carried, trimmed, and labelled as what it is."""

    def test_the_turns_travel_oldest_first(self, recorder: Any) -> None:
        plan_thesis(
            "the second term",
            client=recorder.client,
            conversation=[
                Exchange(question="first", asked=("a?",)),
                Exchange(question="second", asked=("b?",)),
            ],
        )
        said = [turn["reader_said"] for turn in recorder.payload["conversation"]]
        assert said == ["first", "second"]

    def test_each_turn_carries_what_was_asked_back(self, recorder: Any) -> None:
        plan_thesis(
            "the second term",
            client=recorder.client,
            conversation=[Exchange(question="q", restated="r", asked=(ASKED_WHICH,))],
        )
        turn = recorder.payload["conversation"][0]
        assert turn["i_asked"] == [ASKED_WHICH]
        assert turn["read_as"] == "r"

    def test_the_planner_is_told_the_message_is_probably_an_answer(
        self, recorder: Any
    ) -> None:
        """The specific failure: without this the reply is read as a request,
        restated as best it can be, and the same question comes back."""
        plan_thesis(
            "from the close of that day",
            client=recorder.client,
            conversation=[Exchange(question="q", asked=(ASKED_WHERE,))],
        )
        assert "answer to what you last asked" in recorder.payload["instruction"]

    def test_a_thesis_still_wins_the_instruction(self, recorder: Any) -> None:
        """When there is a thesis the question modifies it, which is a stronger
        statement than "this is probably an answer" -- and the conversation
        still travels alongside."""
        plan_thesis(
            "what about ETH?",
            client=recorder.client,
            previous=dynamic_thesis(),
            conversation=[Exchange(question="q", asked=("a?",))],
        )
        assert "Start from current_thesis" in recorder.payload["instruction"]
        assert recorder.payload["conversation"]

    def test_nothing_is_sent_when_there_is_nothing_to_remember(
        self, recorder: Any
    ) -> None:
        plan_thesis("a question", client=recorder.client, conversation=[Exchange()])
        assert "conversation" not in recorder.payload

    def test_only_the_last_turns_are_sent(self, recorder: Any) -> None:
        plan_thesis(
            "a question",
            client=recorder.client,
            conversation=[
                Exchange(question=str(n), asked=("?",))
                for n in range(MAX_REMEMBERED_EXCHANGES + 4)
            ],
        )
        sent = recorder.payload["conversation"]
        assert len(sent) == MAX_REMEMBERED_EXCHANGES
        assert sent[-1]["reader_said"] == str(MAX_REMEMBERED_EXCHANGES + 3)

    def test_the_prompt_forbids_asking_twice(self) -> None:
        assert "Never ask again what the conversation has already answered" in (
            THESIS_PLANNER_SYSTEM
        )
        assert "usually an answer, not a request" in THESIS_PLANNER_SYSTEM


class TestWhatThePageSends:
    """The boundary. Everything here arrives from a browser."""

    def test_a_posted_thread_is_read(self) -> None:
        turns = _conversation([{"question": "q", "asked": ["a?"]}])
        assert turns[0].question == "q"
        assert turns[0].asked == ("a?",)

    @pytest.mark.parametrize("raw", [None, "", {}, 7, "not a list"])
    def test_anything_that_is_not_a_thread_is_no_thread(self, raw: Any) -> None:
        assert _conversation(raw) == ()

    def test_an_unreadable_turn_is_dropped_not_fatal(self) -> None:
        """One turn planned without its context is a failure the reader can
        see and correct. A page that cannot ask anything is not."""
        turns = _conversation([{"question": "fine"}, {"asked": "not a list"}, 12])
        assert len(turns) == 1

    def test_the_server_trims_whatever_it_is_sent(self) -> None:
        """The browser trims too. This is the one that holds, because what
        reaches the planner must not depend on what a page chose to send."""
        turns = _conversation(
            [{"question": str(n)} for n in range(MAX_REMEMBERED_EXCHANGES * 3)]
        )
        assert len(turns) == MAX_REMEMBERED_EXCHANGES

    def test_the_browser_trims_to_the_same_number(self) -> None:
        """Two limits that disagreed would be invisible until a conversation
        got long enough to notice."""
        assert f"slice(-{MAX_REMEMBERED_EXCHANGES})" in RESEARCH_SCRIPT

    def test_the_browser_sends_the_thread(self) -> None:
        assert "script.exchange" in RESEARCH_SCRIPT
        assert "conversation: conversation()" in RESEARCH_SCRIPT


class TestAnswersCanBePointedAt:
    """A question answerable only by typing asks two things at once."""

    def test_the_options_become_buttons(self) -> None:
        html = render_clarification(
            asking({"question": ASKED_WHICH, "options": ["the first term", "the second term"]})
        )
        assert 'data-choice="the first term"' in html
        assert 'data-choice="the second term"' in html

    def test_something_else_is_always_offered(self) -> None:
        """An enumerated list is a guess about what a reader wants, and a
        reader who wants something else must not be stuck."""
        html = render_clarification(
            asking({"question": ASKED_WHICH, "options": ["the first term"]})
        )
        assert "data-other" in html
        assert "Something else" in html

    def test_an_open_question_offers_nothing_to_point_at(self) -> None:
        """No options is a real answer -- a date or a number cannot be
        enumerated -- and an empty button row would be worse than none."""
        html = render_clarification(asking({"question": "From which date?", "options": []}))
        assert "From which date?" in html
        assert "data-choice" not in html
        assert "data-other" not in html

    def test_a_question_recorded_before_options_existed_still_renders(self) -> None:
        """The planner cassette is full of bare strings. They are stale
        evidence about phrasing, which is a different problem from being
        unreadable, and conflating the two would throw the suite away."""
        html = render_clarification(asking(ASKED_WHICH))
        assert ASKED_WHICH in html
        assert "data-choice" not in html

    @pytest.mark.parametrize(
        "option",
        [
            'the "first" term',
            "the 'first' term",
            "it's the \"first\" term",
            "a & b",
        ],
    )
    def test_an_option_cannot_end_the_attribute_it_travels_in(
        self, option: str
    ) -> None:
        """An option is the reader's next message, so it is arbitrary text in
        an HTML attribute. Checked by reading the attribute back rather than
        by matching an escaping scheme -- which quoting the escaper picks is
        its business, and the value surviving is not."""
        html = render_clarification(asking({"question": "Which?", "options": [option]}))
        after = html.split("data-choice=", 1)[1]
        delimiter, body = after[0], after[1:]
        assert delimiter in "\"'"
        assert unescape(body.split(delimiter, 1)[0]) == option

    def test_a_blank_option_is_not_a_choice(self) -> None:
        html = render_clarification(
            asking({"question": "Which?", "options": ["   ", "the second term"]})
        )
        assert html.count("data-choice") == 1

    def test_picking_fills_the_box_rather_than_sending(self) -> None:
        """A card can ask two things, so the reader has to see the whole reply
        before it goes -- and can edit one word of a suggestion rather than
        accepting it or retyping it."""
        assert "box.value = picked.join" in RESEARCH_SCRIPT
        assert "aria-pressed" in RESEARCH_SCRIPT

    def test_the_card_says_how_to_answer(self) -> None:
        html = render_clarification(
            asking({"question": ASKED_WHICH, "options": ["the first term"]})
        )
        assert "Pick an answer or type your own" in html

    def test_several_questions_read_as_several(self) -> None:
        one = render_clarification(asking(ASKED_WHICH))
        two = render_clarification(asking(ASKED_WHICH, ASKED_WHERE))
        assert "one thing would change the answer" in one
        assert "some things would change the answer" in two

    def test_the_prompt_asks_for_the_readers_own_words(self) -> None:
        assert "Give the reader something to point at" in THESIS_PLANNER_SYSTEM
        assert "as the reader would say it" in THESIS_PLANNER_SYSTEM


class TestTheAnchorQuestionOffersWordsThePlannerReads:
    """The one clarification whose answers are known in advance.

    Decided in Python rather than by the planner, because the policies are
    this system's own. Which makes drift possible in a way it is not for a
    model-written option: a button whose wording stopped being understood
    would send the reader round the same question again, silently.
    """

    @pytest.mark.parametrize(("spoken", "policy"), ANCHOR_CHOICES)
    def test_every_offered_answer_is_read_back_as_intended(
        self, spoken: str, policy: AnchorPolicy
    ) -> None:
        anchor = planned(event_anchor=spoken).anchor
        assert anchor is not None, spoken
        assert anchor.policy is policy

    def test_a_clarification_can_still_offer_them(self) -> None:
        """The card that asked is gone -- a date-only event is measured from
        the start of its day now -- but the words outlive it. A planner
        clarification about which instant to use still has to offer spellings
        it will recognise coming back."""
        html = render_clarification(
            asking(
                {
                    "question": "Where in the day should the window start?",
                    "options": [text for text, _policy in ANCHOR_CHOICES],
                }
            )
        )
        for spoken, _policy in ANCHOR_CHOICES:
            assert f'data-choice="{spoken}"' in html


class TestStateCannotEscapeItsBlock:
    """The reader's own question travels inside a script element.

    A question containing a closing script tag ended the element, and whatever
    followed was parsed as HTML. Found while giving the exchange its own
    block -- the thesis block had the same hole, and had had it longer.
    """

    HOSTILE = "</script><img src=x onerror=alert(1)>"

    def test_a_closing_tag_in_a_question_does_not_end_the_thesis_block(self) -> None:
        html = render_spec_card(dynamic_thesis(question=self.HOSTILE), planned())
        assert "</script><img" not in html
        assert "u003c/script" in html

    def test_it_does_not_end_the_exchange_block_either(self) -> None:
        html = render_clarification(asking(ASKED_WHICH), question=self.HOSTILE)
        assert "</script><img" not in html

    def test_the_value_survives_the_escaping(self) -> None:
        """Escaped for the parser, identical once parsed -- otherwise this
        would fix an injection by corrupting the state."""
        html = render_clarification(asking(ASKED_WHICH), question=self.HOSTILE)
        assert remembered(html)[0]["question"] == self.HOSTILE

    def test_a_hostile_question_still_round_trips_as_a_thesis(self) -> None:
        html = render_spec_card(dynamic_thesis(question=self.HOSTILE), planned())
        body = html.split('class="thesis">', 1)[1].split("</script>", 1)[0]
        assert Thesis.model_validate_json(body).question == self.HOSTILE
class TestTheHalvesFitTogether:
    """Drive the conversation the reader actually had.

    The planner is stubbed, so this spends nothing and tests the seam rather
    than the judgement: what the card wrote down, what the browser would send
    back, and what the planner was handed. The bug lived exactly here -- both
    halves were correct and there was nothing between them.
    """

    def exchanges_from(self, html: str) -> list[dict[str, Any]]:
        """What the browser's `conversation()` would collect from this card."""
        return remembered(html)

    def test_an_answer_arrives_knowing_what_it_answers(
        self, recorder: Recorder
    ) -> None:
        # Turn one: the reader asks, and is asked two things back.
        card = render_clarification(
            asking(
                {"question": ASKED_WHICH, "options": ["the second term, January 2025"]},
                {"question": ASKED_WHERE, "options": ["from the start of the day"]},
            ),
            question="can you tell me how btc performed during Trump's first day?",
        )

        # Turn two: the reader picks both, and the page sends the thread.
        turns = _conversation(self.exchanges_from(card))
        plan_thesis(
            "the second term, January 2025; from the start of the day",
            client=recorder.client,
            conversation=turns,
        )

        carried = recorder.payload["conversation"]
        assert len(carried) == 1
        assert carried[0]["i_asked"] == [ASKED_WHICH, ASKED_WHERE]
        assert "first day" in carried[0]["reader_said"]

    def test_a_chain_of_clarifications_accumulates(
        self, recorder: Recorder
    ) -> None:
        """Four rounds is what happened. Each one has to add to the thread
        rather than replace it, or the fourth answer is as contextless as the
        first."""
        thread: list[dict[str, Any]] = []
        for n in range(4):
            card = render_clarification(
                asking(f"question {n}?"), question=f"answer {n}"
            )
            thread.extend(self.exchanges_from(card))

        plan_thesis("and finally this", client=recorder.client,
                    conversation=_conversation(thread))
        carried = recorder.payload["conversation"]
        assert [turn["i_asked"][0] for turn in carried] == [
            f"question {n}?" for n in range(4)
        ]

    def test_an_unanswerable_lookup_hands_its_questions_on(
        self, recorder: Recorder
    ) -> None:
        card = render_unresolved_event(unresolvable(), question="which crisis?")
        plan_thesis(
            "the 2008 one",
            client=recorder.client,
            conversation=_conversation(self.exchanges_from(card)),
        )
        assert recorder.payload["conversation"][0]["i_asked"]

