"""What a request cost, and why that is computed rather than recalled.

This module exists because of a mistake worth keeping a test suite honest
about. The published rates lived nowhere in the repository, so every cost
figure in every report came from recollection -- and one set was quoted at
three times the real price, because the rates recalled belonged to a previous
model generation. The arithmetic was fine. The inputs were invented.

So what is tested here is not really multiplication. It is that a rate nobody
wrote down cannot become a total that looks measured.
"""

from __future__ import annotations

import logging

import pytest

from backtool.ai.usage import (
    PER_SEARCH,
    RATES,
    RATES_READ_ON,
    Cost,
    UnpricedModelError,
    log_cost,
    price,
)

OPUS = "claude-opus-5"
SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-4-5-20251001"


class TestAnUnpricedModelIsAGapNotAZero:
    """The failure mode that matters. Everything else is arithmetic."""

    def test_an_unknown_model_refuses(self) -> None:
        with pytest.raises(UnpricedModelError):
            price("claude-something-nobody-recorded", {"input_tokens": 1000})

    def test_the_refusal_says_what_to_do(self) -> None:
        with pytest.raises(UnpricedModelError, match="rather than estimating"):
            price("claude-unknown", {})

    def test_a_near_miss_is_not_a_rate(self) -> None:
        """Opus 4.1 and Opus 5 differ threefold. Matching on a prefix would
        have hidden the exact error that prompted this module."""
        with pytest.raises(UnpricedModelError):
            price("claude-opus-4-1", {"input_tokens": 1_000_000})

    def test_logging_an_unpriced_call_warns_rather_than_inventing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reported as unknown. A silent zero would read as 'free'."""
        with caplog.at_level(logging.WARNING):
            cost = log_cost("somewhere", "claude-unknown", {"output_tokens": 500})
        assert cost.total_usd == 0.0
        assert "cost unknown" in caplog.text

    def test_a_search_still_counts_when_the_tokens_cannot_be_priced(self) -> None:
        """The search fee does not depend on the model, so it is not lost with
        the rest."""
        cost = log_cost("x", "claude-unknown", {}, searches=3)
        assert cost.searches == 3


class TestTheRatesAreWrittenDown:
    """Not recalled. That is the whole point of the file."""

    def test_every_rate_is_a_complete_set_of_four(self) -> None:
        for model, rates in RATES.items():
            assert len(rates) == 4, model
            assert all(rate > 0 for rate in rates), model

    def test_output_costs_more_than_input_everywhere(self) -> None:
        """True of every published model, and the reason output dominates a
        bill. A table where it were not would be a transcription error."""
        for model, (inp, _write, _read, out) in RATES.items():
            assert out > inp, model

    def test_a_cache_write_costs_more_than_a_fresh_read(self) -> None:
        """Which is why re-reading a growing context pays a quiet premium -- the
        thing that made a resolver call cost what it did."""
        for model, (inp, write, read, _out) in RATES.items():
            assert write > inp > read, model

    def test_the_models_the_product_actually_uses_are_priced(self) -> None:
        """A rate missing for one of these would mean a real question could
        not be priced."""
        from backtool.ai.client import DEFAULT_MODEL
        from backtool.ai.event_resolver import (
            DEFAULT_EXTRACTION_MODEL,
            DEFAULT_SEARCH_MODEL,
        )

        for model in (DEFAULT_MODEL, DEFAULT_SEARCH_MODEL, DEFAULT_EXTRACTION_MODEL):
            assert model in RATES, model

    def test_when_the_rates_were_read_is_recorded(self) -> None:
        """So a stale figure is visible as stale rather than as fact."""
        assert RATES_READ_ON

    def test_a_search_is_a_hundredth_of_a_dollar(self) -> None:
        assert pytest.approx(0.01) == PER_SEARCH


class TestPricing:
    """The arithmetic, pinned against the published figures."""

    def test_a_million_input_tokens_costs_the_input_rate(self) -> None:
        cost = price(OPUS, {"input_tokens": 1_000_000})
        assert cost.input_usd == pytest.approx(5.00)
        assert cost.total_usd == pytest.approx(5.00)

    def test_each_part_is_priced_on_its_own_rate(self) -> None:
        cost = price(
            OPUS,
            {
                "input_tokens": 1_000_000,
                "cache_creation_input_tokens": 1_000_000,
                "cache_read_input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
            },
        )
        assert cost.input_usd == pytest.approx(5.00)
        assert cost.cache_write_usd == pytest.approx(6.25)
        assert cost.cache_read_usd == pytest.approx(0.50)
        assert cost.output_usd == pytest.approx(25.00)
        assert cost.total_usd == pytest.approx(36.75)

    def test_missing_cache_fields_count_as_zero(self) -> None:
        """A response without them did not use the cache, which is not the
        same as a malformed response."""
        cost = price(SONNET, {"input_tokens": 100, "output_tokens": 100})
        assert cost.cache_write_usd == 0.0
        assert cost.cache_read_usd == 0.0

    def test_searches_are_billed_on_top_of_tokens(self) -> None:
        cost = price(SONNET, {"input_tokens": 0}, searches=4)
        assert cost.search_usd == pytest.approx(0.04)
        assert cost.searches == 4

    def test_tokens_are_totalled_across_every_kind(self) -> None:
        cost = price(
            HAIKU,
            {
                "input_tokens": 1,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 4,
                "output_tokens": 8,
            },
        )
        assert cost.tokens == 15

    def test_the_cheap_pair_really_is_cheaper(self) -> None:
        """The claim the resolver's model choice rests on, checked rather than
        assumed: the same work on Haiku costs a fifth of Opus."""
        usage = {"input_tokens": 10_000, "output_tokens": 2_000}
        assert price(HAIKU, usage).total_usd < price(OPUS, usage).total_usd / 4

    def test_costs_add_up(self) -> None:
        """Two stages on two models are one bill."""
        first = price(SONNET, {"input_tokens": 1000}, searches=1)
        second = price(HAIKU, {"output_tokens": 1000})
        both = first + second
        assert both.total_usd == pytest.approx(first.total_usd + second.total_usd)
        assert both.searches == 1
        assert both.tokens == 2000

    def test_an_empty_cost_is_free(self) -> None:
        assert Cost().total_usd == 0.0


class TestDescribingACost:
    """One line, for a log or a report footer."""

    def test_it_states_dollars_and_tokens(self) -> None:
        line = price(OPUS, {"input_tokens": 12_345}).describe()
        assert "$" in line
        assert "12,345 tokens" in line

    def test_one_search_is_singular(self) -> None:
        assert "1 search)" in price(OPUS, {}, searches=1).describe()

    def test_several_searches_are_plural(self) -> None:
        assert "2 searches)" in price(OPUS, {}, searches=2).describe()

    def test_no_searches_are_not_mentioned(self) -> None:
        assert "search" not in price(OPUS, {"input_tokens": 1}).describe()

    def test_it_is_logged_where_it_happened(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The figure has to survive the request that produced it, because
        "what did that cost?" is always asked afterwards."""
        with caplog.at_level(logging.INFO):
            log_cost("planner", OPUS, {"input_tokens": 1000, "output_tokens": 500})
        assert "planner" in caplog.text
        assert OPUS in caplog.text
        assert "$" in caplog.text
