"""Tracking which time ranges have already been fetched.

Why this exists separately from the candles themselves
-----------------------------------------------------
The obvious way to decide what to download is "compare the candles on disk
against the expected interval grid, and fetch whatever is missing". That breaks
on real market data, because some candles will *never* exist: exchange
outages, maintenance windows, and the period before a symbol was listed all
leave permanent holes in the grid.

If absence of a candle meant "not yet fetched", every research run would
re-request those holes forever, hitting the API for data that cannot be
returned.

So coverage is recorded explicitly: we remember the ranges we have *asked*
about, independently of what came back. A gap inside a covered range is a real
market gap, and we stop asking.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

from backtool.core.time import ensure_utc, from_utc_ms, utc_ms


@dataclass(frozen=True, order=True)
class TimeRange:
    """A half-open interval ``[start, end)`` in UTC."""

    start: dt.datetime
    end: dt.datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"TimeRange start {self.start} must precede end {self.end}")

    @property
    def duration(self) -> dt.timedelta:
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start:%Y-%m-%d %H:%M} .. {self.end:%Y-%m-%d %H:%M})"


class CoverageIndex:
    """An immutable, normalised set of fetched time ranges.

    Ranges are always stored sorted and merged, so overlapping or touching
    ranges collapse into one. That keeps :meth:`missing` simple and keeps the
    on-disk manifest from growing without bound as runs accumulate.
    """

    __slots__ = ("_ranges",)

    def __init__(self, ranges: tuple[TimeRange, ...] = ()) -> None:
        self._ranges = _merge(ranges)

    @property
    def ranges(self) -> tuple[TimeRange, ...]:
        return self._ranges

    def __len__(self) -> int:
        return len(self._ranges)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoverageIndex):
            return NotImplemented
        return self._ranges == other._ranges

    def __repr__(self) -> str:
        return f"CoverageIndex({[str(r) for r in self._ranges]})"

    def covers(self, start: dt.datetime, end: dt.datetime) -> bool:
        """True if ``[start, end)`` is fully contained in recorded coverage."""
        return not self.missing(start, end)

    def with_range(self, start: dt.datetime, end: dt.datetime) -> CoverageIndex:
        """Return a new index that additionally covers ``[start, end)``."""
        return CoverageIndex((*self._ranges, TimeRange(ensure_utc(start), ensure_utc(end))))

    def missing(self, start: dt.datetime, end: dt.datetime) -> list[TimeRange]:
        """Return the sub-ranges of ``[start, end)`` not yet covered.

        Args:
            start: Inclusive lower bound, timezone-aware.
            end: Exclusive upper bound, timezone-aware.

        Returns:
            Zero or more disjoint ranges, in chronological order. Empty means
            everything requested has already been fetched.
        """
        cursor = ensure_utc(start)
        limit = ensure_utc(end)
        if cursor >= limit:
            return []

        gaps: list[TimeRange] = []
        for covered in self._ranges:
            if covered.end <= cursor:
                continue
            if covered.start >= limit:
                break
            if covered.start > cursor:
                gaps.append(TimeRange(cursor, min(covered.start, limit)))
            cursor = max(cursor, covered.end)
            if cursor >= limit:
                break

        if cursor < limit:
            gaps.append(TimeRange(cursor, limit))
        return gaps

    # --- persistence ------------------------------------------------------

    def to_json(self) -> str:
        """Serialise as epoch milliseconds, matching the exchange's own unit."""
        payload = {
            "version": 1,
            "ranges": [[utc_ms(r.start), utc_ms(r.end)] for r in self._ranges],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> CoverageIndex:
        """Parse a manifest. An unreadable manifest yields empty coverage.

        Treating corruption as "nothing fetched" is safe: the worst outcome is
        re-downloading data we already have, whereas trusting a damaged
        manifest would silently skip ranges and produce incomplete research.
        """
        try:
            payload: Any = json.loads(text)
            raw = payload["ranges"]
            return cls(tuple(TimeRange(from_utc_ms(int(a)), from_utc_ms(int(b))) for a, b in raw))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return cls()


def _merge(ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
    """Sort and coalesce overlapping or adjacent ranges."""
    if not ranges:
        return ()

    merged: list[TimeRange] = []
    for candidate in sorted(ranges):
        if merged and candidate.start <= merged[-1].end:
            previous = merged[-1]
            if candidate.end > previous.end:
                merged[-1] = TimeRange(previous.start, candidate.end)
        else:
            merged.append(candidate)
    return tuple(merged)
