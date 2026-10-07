"""What a request cost, computed rather than estimated.

This module exists because of a specific mistake. The published rates lived
nowhere in the repository, so every cost figure came from recollection -- and
one set of figures was quoted at three times the real price for weeks, because
the rates recalled were a previous model generation's. A number nobody can
recompute is not a measurement.

So the rates are written down, per model, with the date they were read; pricing
is a pure function of a usage block; and a model nobody recorded a rate for
raises rather than being guessed at. An unpriced call should be a loud gap in a
report, not a plausible-looking total.

Nothing here bills anything or talks to an API. It turns the ``usage`` block the
Messages API already returns into dollars, so a run can say what it cost
instead of someone asserting it afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: When the rates below were last read from the published price list.
#: Quoted in any report built from them, so a stale figure is visible as stale
#: rather than as fact.
RATES_READ_ON = "2026-10-05"

#: USD per million tokens, from platform.claude.com/docs/en/about-claude/pricing
#: -- input, 5-minute cache write, cache read, output.
#:
#: Keyed by the exact model id a request carries, because that is what the
#: billing is keyed by. A near-miss is not a rate: Opus 4.1 and Opus 5 differ
#: threefold, and matching on a prefix would have hidden exactly the error that
#: prompted this file.
RATES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5": (5.00, 6.25, 0.50, 25.00),
    "claude-opus-5-5": (4.00, 5.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 2.50, 0.20, 10.00),
    "claude-sonnet-5": (2.00, 2.50, 0.20, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 1.25, 0.10, 5.00),
}

#: USD per web search performed by the server-side tool. Charged per search
#: whatever it returned, and not charged at all for one that errored.
PER_SEARCH = 10.00 / 1000


class UnpricedModelError(KeyError):
    """No published rate is recorded for this model.

    Its own type so a caller can report the gap. Guessing would produce a
    total that looks like a measurement and is not one.
    """


@dataclass(frozen=True)
class Cost:
    """What one or more requests cost, broken out by what drove it.

    Separated because the parts behave differently and the difference is where
    the savings are: output is five times input, and a cache write is 1.25x an
    ordinary read, so a request that re-reads a growing context pays a quiet
    premium that a single total hides.
    """

    input_usd: float = 0.0
    cache_write_usd: float = 0.0
    cache_read_usd: float = 0.0
    output_usd: float = 0.0
    search_usd: float = 0.0
    searches: int = 0
    tokens: int = 0

    @property
    def total_usd(self) -> float:
        return (
            self.input_usd
            + self.cache_write_usd
            + self.cache_read_usd
            + self.output_usd
            + self.search_usd
        )

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            input_usd=self.input_usd + other.input_usd,
            cache_write_usd=self.cache_write_usd + other.cache_write_usd,
            cache_read_usd=self.cache_read_usd + other.cache_read_usd,
            output_usd=self.output_usd + other.output_usd,
            search_usd=self.search_usd + other.search_usd,
            searches=self.searches + other.searches,
            tokens=self.tokens + other.tokens,
        )

    def describe(self) -> str:
        """One line, for a log or a report footer."""
        return (
            f"${self.total_usd:.4f} "
            f"({self.tokens:,} tokens"
            + (f", {self.searches} search" if self.searches == 1 else "")
            + (f", {self.searches} searches" if self.searches > 1 else "")
            + ")"
        )


def price(model: str, usage: dict[str, int], *, searches: int = 0) -> Cost:
    """What a request on ``model`` with this usage block cost.

    Args:
        model: The exact model id the request carried.
        usage: The API's ``usage`` block. Missing keys count as zero, which is
            right -- a response without cache fields did not use the cache.
        searches: Server-side searches performed, billed per search on top of
            tokens.

    Returns:
        A :class:`Cost` broken out by driver.

    Raises:
        UnpricedModelError: if no rate is recorded for ``model``.
    """
    try:
        per_input, per_write, per_read, per_output = RATES[model]
    except KeyError as exc:
        raise UnpricedModelError(
            f"No published rate recorded for {model!r}. Add it to "
            "backtool.ai.usage.RATES rather than estimating: a guessed rate "
            "produces a total that looks measured and is not."
        ) from exc

    fresh = int(usage.get("input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)

    return Cost(
        input_usd=fresh / 1e6 * per_input,
        cache_write_usd=written / 1e6 * per_write,
        cache_read_usd=read / 1e6 * per_read,
        output_usd=out / 1e6 * per_output,
        search_usd=searches * PER_SEARCH,
        searches=searches,
        tokens=fresh + written + read + out,
    )


def log_cost(what: str, model: str, usage: dict[str, int], *, searches: int = 0) -> Cost:
    """Price a request and record it, returning the cost.

    Logging rather than returning only: the figure has to survive the request
    that produced it, because the question "what did that cost?" is always
    asked afterwards. An unpriced model is reported as such rather than
    silently omitted, so a missing rate shows up as a gap instead of as zero.
    """
    try:
        cost = price(model, usage, searches=searches)
    except UnpricedModelError as exc:
        logger.warning("%s on %s: cost unknown -- %s", what, model, exc)
        return Cost(searches=searches)
    logger.info("%s on %s: %s", what, model, cost.describe())
    return cost


__all__ = [
    "PER_SEARCH",
    "RATES",
    "RATES_READ_ON",
    "Cost",
    "UnpricedModelError",
    "log_cost",
    "price",
]
