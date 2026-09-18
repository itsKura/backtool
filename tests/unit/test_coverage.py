"""Tests for fetch-coverage tracking.

Coverage decides what gets downloaded. A bug that over-reports coverage causes
silent data loss -- research runs on ranges that were never fetched and nobody
is told. A bug that under-reports merely wastes bandwidth. These tests lean
hard on the first case.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtool.core.time import UTC
from backtool.data.storage.coverage import CoverageIndex, TimeRange


def at(hour: int, day: int = 1) -> dt.datetime:
    return dt.datetime(2024, 1, day, hour, tzinfo=UTC)


def rng(start_hour: int, end_hour: int, day: int = 1) -> TimeRange:
    return TimeRange(at(start_hour, day), at(end_hour, day))


class TestTimeRange:
    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="must precede"):
            TimeRange(at(5), at(3))

    def test_rejects_empty_range(self) -> None:
        with pytest.raises(ValueError, match="must precede"):
            TimeRange(at(5), at(5))

    def test_duration(self) -> None:
        assert rng(3, 7).duration == dt.timedelta(hours=4)


class TestMerging:
    def test_overlapping_ranges_collapse(self) -> None:
        index = CoverageIndex((rng(0, 5), rng(3, 8)))
        assert index.ranges == (rng(0, 8),)

    def test_adjacent_ranges_collapse(self) -> None:
        """Touching ranges are contiguous coverage, not two separate spans."""
        index = CoverageIndex((rng(0, 5), rng(5, 9)))
        assert index.ranges == (rng(0, 9),)

    def test_disjoint_ranges_are_kept_apart(self) -> None:
        index = CoverageIndex((rng(0, 5), rng(7, 9)))
        assert index.ranges == (rng(0, 5), rng(7, 9))

    def test_contained_range_is_absorbed(self) -> None:
        index = CoverageIndex((rng(0, 10), rng(3, 6)))
        assert index.ranges == (rng(0, 10),)

    def test_input_order_does_not_matter(self) -> None:
        assert CoverageIndex((rng(7, 9), rng(0, 5), rng(4, 8))) == CoverageIndex(
            (rng(0, 5), rng(4, 8), rng(7, 9))
        )

    def test_repeated_adds_do_not_grow_the_index(self) -> None:
        """Manifest size must not scale with the number of runs."""
        index = CoverageIndex()
        for _ in range(50):
            index = index.with_range(at(0), at(5))
        assert len(index) == 1


class TestMissing:
    def test_everything_missing_when_nothing_covered(self) -> None:
        assert CoverageIndex().missing(at(0), at(5)) == [rng(0, 5)]

    def test_nothing_missing_when_fully_covered(self) -> None:
        index = CoverageIndex((rng(0, 10),))
        assert index.missing(at(2), at(8)) == []
        assert index.covers(at(2), at(8))

    def test_exact_match_is_fully_covered(self) -> None:
        assert CoverageIndex((rng(0, 5),)).missing(at(0), at(5)) == []

    def test_gap_before_coverage(self) -> None:
        index = CoverageIndex((rng(5, 10),))
        assert index.missing(at(2), at(8)) == [rng(2, 5)]

    def test_gap_after_coverage(self) -> None:
        index = CoverageIndex((rng(0, 5),))
        assert index.missing(at(2), at(8)) == [rng(5, 8)]

    def test_gap_on_both_sides(self) -> None:
        index = CoverageIndex((rng(4, 6),))
        assert index.missing(at(2), at(9)) == [rng(2, 4), rng(6, 9)]

    def test_multiple_interior_gaps(self) -> None:
        index = CoverageIndex((rng(2, 4), rng(6, 8), rng(10, 12)))
        assert index.missing(at(0), at(14)) == [
            rng(0, 2),
            rng(4, 6),
            rng(8, 10),
            rng(12, 14),
        ]

    def test_request_entirely_inside_a_gap(self) -> None:
        index = CoverageIndex((rng(0, 2), rng(8, 10)))
        assert index.missing(at(4), at(6)) == [rng(4, 6)]

    def test_request_entirely_before_all_coverage(self) -> None:
        index = CoverageIndex((rng(10, 12),))
        assert index.missing(at(2), at(5)) == [rng(2, 5)]

    def test_request_entirely_after_all_coverage(self) -> None:
        index = CoverageIndex((rng(0, 2),))
        assert index.missing(at(10), at(14)) == [rng(10, 14)]

    def test_empty_request_yields_nothing(self) -> None:
        assert CoverageIndex().missing(at(5), at(5)) == []

    def test_inverted_request_yields_nothing(self) -> None:
        assert CoverageIndex().missing(at(8), at(3)) == []

    def test_missing_ranges_are_chronological_and_disjoint(self) -> None:
        index = CoverageIndex((rng(2, 4), rng(6, 8)))
        gaps = index.missing(at(0), at(12))
        for earlier, later in zip(gaps, gaps[1:], strict=False):
            assert earlier.end <= later.start

    def test_accepts_non_utc_bounds(self) -> None:
        from backtool.core.time import NEW_YORK

        index = CoverageIndex((rng(0, 10),))
        # 2024-01-01 03:00 UTC expressed as New York wall clock.
        ny = dt.datetime(2023, 12, 31, 22, 0, tzinfo=NEW_YORK)
        assert index.missing(ny, at(8)) == []

    def test_rejects_naive_bounds(self) -> None:
        with pytest.raises(ValueError, match="Naive datetime"):
            CoverageIndex().missing(dt.datetime(2024, 1, 1), at(5))


class TestPersistence:
    def test_round_trips_through_json(self) -> None:
        index = CoverageIndex((rng(0, 5), rng(8, 12)))
        assert CoverageIndex.from_json(index.to_json()) == index

    def test_empty_index_round_trips(self) -> None:
        assert CoverageIndex.from_json(CoverageIndex().to_json()) == CoverageIndex()

    @pytest.mark.parametrize(
        "text",
        ["", "not json", "{}", '{"ranges": "nope"}', '{"ranges": [[1]]}', "null"],
    )
    def test_corrupt_manifest_reads_as_no_coverage(self, text: str) -> None:
        """Safe direction to fail: re-downloading data we already have costs
        bandwidth, whereas trusting a damaged manifest would silently skip
        ranges and produce incomplete research."""
        assert CoverageIndex.from_json(text) == CoverageIndex()
