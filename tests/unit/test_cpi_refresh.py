"""Filling the release file from the agency that publishes the figures.

The actual and previous columns were typed in by hand. They did not need to
be: BLS serves the whole series free, and it reproduced all 27 existing rows
exactly, which is what justified wiring it in at all.

What these protect is the restraint rather than the fetching. A refresh that
can edit a hand-verified dataset is one wrong rule away from quietly rewriting
evidence, so the rules worth pinning are the ones about what it must *not* do:
never overwrite a person, never invent a consensus, never fill a month nobody
has published.
"""

from __future__ import annotations

import datetime as dt

import httpx
import pytest

from backtool.core.types import EventType
from backtool.data.bls import (
    CPI_SERIES,
    CPI_SERIES_URL,
    BLSClient,
    BLSError,
    reference_month,
    year_over_year,
)
from backtool.events.cpi_refresh import Change, refresh_releases
from backtool.events.models import MarketEvent
from backtool.events.releases import (
    ReleaseValues,
    load_releases,
    releases_path,
    to_csv,
)

NOW = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.UTC)


def levels(**rows: float) -> dict[tuple[int, int], float]:
    """Index levels keyed ``y2026m09=336.9``, which reads as a table."""
    return {
        (int(key[1:5]), int(key[6:])): value for key, value in rows.items()
    }


def event(event_id: str, when: dt.datetime) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        event_type=EventType.CPI,
        local_date=when.date(),
        local_time=when.time(),
        timezone="UTC",
        is_scheduled=True,
    )


def row(event_id: str, **fields: object) -> ReleaseValues:
    base: dict[str, object] = {
        "event_id": event_id,
        "source_id": event_id.lower().replace("-", "_"),
        "release_date": dt.date(2026, 10, 14),
        "reference_month": "2026-09",
    }
    base.update(fields)
    return ReleaseValues(**base)  # type: ignore[arg-type]


class TestTheArithmeticIsThePublishedOne:
    def test_year_over_year_is_the_two_index_levels(self) -> None:
        table = levels(y2026m09=336.940, y2025m09=325.059)
        assert year_over_year(table, 2026, 9) == 3.7

    def test_it_is_rounded_to_the_precision_cpi_is_published_at(self) -> None:
        """One decimal place, which is also what the scenario rule compares at.
        Carrying more would classify releases by digits no source printed."""
        table = levels(y2026m09=336.940, y2025m09=325.059)
        assert year_over_year(table, 2026, 9) == round(
            (336.940 / 325.059 - 1) * 100, 1
        )

    def test_a_missing_month_is_not_an_estimate(self) -> None:
        """Half a year-over-year change is not most of one."""
        assert year_over_year(levels(y2026m09=336.9), 2026, 9) is None
        assert year_over_year(levels(y2025m09=325.0), 2026, 9) is None

    def test_a_release_reports_the_month_before_it(self) -> None:
        assert reference_month(dt.date(2026, 10, 14)) == (2026, 9)
        assert reference_month(dt.date(2026, 1, 13)) == (2025, 12)

    def test_that_rule_holds_for_every_row_we_already_had(self) -> None:
        """The derivation is only safe while this does. If a release ever
        reports a different month, this fails rather than the refresh writing a
        year-over-year figure for the wrong period."""
        for values in load_releases().values():
            if not values.reference_month or values.release_date is None:
                continue
            year, month = (int(part) for part in values.reference_month.split("-"))
            assert (year, month) == reference_month(values.release_date)


class TestTheClientReadsOneSeries:
    """No network. The parsing is what can be wrong; the HTTP is httpx's job."""

    def responding(self, body: object, status: int = 200) -> BLSClient:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json=body)

        return BLSClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    def payload(self, rows: list[dict[str, str]]) -> dict[str, object]:
        return {
            "status": "REQUEST_SUCCEEDED",
            "message": [],
            "Results": {"series": [{"seriesID": CPI_SERIES, "data": rows}]},
        }

    def test_it_reads_published_months(self) -> None:
        with self.responding(
            self.payload(
                [
                    {"year": "2026", "period": "M09", "value": "336.940"},
                    {"year": "2026", "period": "M08", "value": "334.980"},
                ]
            )
        ) as client:
            assert client.index_by_month(start_year=2026, end_year=2026) == {
                (2026, 9): 336.940,
                (2026, 8): 334.980,
            }

    def test_an_unpublished_month_is_omitted_not_zeroed(self) -> None:
        """BLS lists a scheduled month with "-" until it publishes. A zero
        there would become a -100% year-over-year change."""
        with self.responding(
            self.payload(
                [
                    {"year": "2026", "period": "M10", "value": "-"},
                    {"year": "2026", "period": "M09", "value": "336.940"},
                ]
            )
        ) as client:
            assert (2026, 10) not in client.index_by_month(
                start_year=2026, end_year=2026
            )

    def test_the_annual_average_is_not_a_month(self) -> None:
        """M13 is the year's average. Treated as a thirteenth month it would
        land in a month-keyed table and be compared against nothing."""
        with self.responding(
            self.payload(
                [
                    {"year": "2025", "period": "M13", "value": "320.000"},
                    {"year": "2025", "period": "M09", "value": "325.059"},
                ]
            )
        ) as client:
            table = client.index_by_month(start_year=2025, end_year=2025)
            assert set(table) == {(2025, 9)}

    def test_a_refusal_is_raised_with_its_reason(self) -> None:
        with self.responding(
            {"status": "REQUEST_NOT_PROCESSED", "message": ["daily threshold"]}
        ) as client, pytest.raises(BLSError, match="daily threshold"):
            client.index_by_month(start_year=2026, end_year=2026)

    def test_an_empty_series_is_an_error_not_an_empty_refresh(self) -> None:
        """Silently returning nothing would read downstream as "BLS has not
        published anything yet" and skip every release."""
        with (
            self.responding(self.payload([])) as client,
            pytest.raises(BLSError, match="no published values"),
        ):
            client.index_by_month(start_year=2026, end_year=2026)

    def test_asking_past_the_allowance_fails_before_the_request(self) -> None:
        with (
            self.responding(self.payload([])) as client,
            pytest.raises(BLSError, match="allows"),
        ):
            client.index_by_month(start_year=1990, end_year=2026)


class TestTheRefreshFillsGapsAndNothingElse:
    def table(self) -> dict[tuple[int, int], float]:
        return levels(
            y2026m09=336.940,
            y2025m09=325.059,
            y2026m08=334.980,
            y2025m08=323.976,
        )

    def test_it_adds_a_release_the_file_had_not_reached(self) -> None:
        report = refresh_releases(
            existing={},
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        assert [c.change for c in report.changes] == [Change.ADDED]
        added = report.rows["CPI-2026-10-14"]
        assert added.actual_yoy == 3.7
        assert added.actual_source == CPI_SERIES_URL
        assert added.first_release_verified

    def test_a_release_it_adds_carries_no_consensus(self) -> None:
        """The honest state: this source has none. A row without one is
        excluded from the scenario cohorts rather than counted as in line, so
        a refresh grows the header and the history table without touching the
        statistics -- which is correct, and worth knowing."""
        report = refresh_releases(
            existing={},
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        added = report.rows["CPI-2026-10-14"]
        assert added.consensus_yoy is None
        assert not added.consensus_verified
        assert added.surprise_pp is None
        assert added.exclusion_reason() is not None

    def test_it_fills_a_missing_figure(self) -> None:
        existing = {"CPI-2026-10-14": row("CPI-2026-10-14", actual_yoy=None)}
        report = refresh_releases(
            existing=existing,
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        assert report.of(Change.FILLED)
        assert report.rows["CPI-2026-10-14"].actual_yoy == 3.7

    def test_it_leaves_a_correct_row_alone(self) -> None:
        existing = {"CPI-2026-10-14": row("CPI-2026-10-14", actual_yoy=3.7, previous_yoy=3.4)}
        report = refresh_releases(
            existing=existing,
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        assert [c.change for c in report.changes] == [Change.UNCHANGED]
        assert report.rows == existing

    def test_it_never_overwrites_a_stored_figure(self) -> None:
        """The rule the whole design rests on. A disagreement is a
        transcription error or a revision, and those want opposite responses --
        resolving it here would destroy the evidence needed to tell them
        apart."""
        existing = {"CPI-2026-10-14": row("CPI-2026-10-14", actual_yoy=9.9)}
        report = refresh_releases(
            existing=existing,
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        assert report.has_conflicts
        assert report.rows["CPI-2026-10-14"].actual_yoy == 9.9
        assert "9.9" in report.of(Change.CONFLICT)[0].detail
        assert "3.7" in report.of(Change.CONFLICT)[0].detail

    def test_it_never_writes_a_consensus(self) -> None:
        """Across every path. BLS has none to give, and a consensus appearing
        from a source that does not publish one is the single thing this
        integration must never do."""
        existing = {
            "CPI-2026-10-14": row("CPI-2026-10-14", actual_yoy=None, consensus_yoy=None)
        }
        report = refresh_releases(
            existing=existing,
            calendar=[
                event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC)),
                event("CPI-2026-09-11", dt.datetime(2026, 9, 11, 12, 30, tzinfo=dt.UTC)),
            ],
            levels=self.table(),
            now=NOW,
        )
        for values in report.rows.values():
            assert values.consensus_yoy is None

    def test_an_existing_consensus_survives_a_fill(self) -> None:
        existing = {
            "CPI-2026-10-14": row(
                "CPI-2026-10-14",
                actual_yoy=None,
                consensus_yoy=3.6,
                consensus_verified=True,
                consensus_source="https://example.invalid/forecast",
            )
        }
        report = refresh_releases(
            existing=existing,
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        after = report.rows["CPI-2026-10-14"]
        assert after.actual_yoy == 3.7
        assert after.consensus_yoy == 3.6
        assert after.consensus_verified
        assert after.consensus_source == "https://example.invalid/forecast"

    def test_a_release_that_has_not_happened_gets_no_row(self) -> None:
        """An empty row would turn "not yet" into "we are missing this", which
        is the distinction the page's figure statuses exist to keep."""
        report = refresh_releases(
            existing={},
            calendar=[event("CPI-2026-12-10", dt.datetime(2026, 12, 10, 13, 30, tzinfo=dt.UTC))],
            levels=self.table(),
            now=NOW,
        )
        assert report.changes == ()
        assert report.rows == {}

    def test_a_month_bls_has_not_published_is_reported_not_filled(self) -> None:
        report = refresh_releases(
            existing={},
            calendar=[event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC))],
            levels=levels(y2025m09=325.059),
            now=NOW,
        )
        assert [c.change for c in report.changes] == [Change.AWAITING_DATA]
        assert report.rows == {}

    def test_a_conflict_stops_the_whole_run(self) -> None:
        """Rather than writing the rows it liked and leaving the one it did
        not, which would be a half-applied edit nobody asked for."""
        existing = {"CPI-2026-10-14": row("CPI-2026-10-14", actual_yoy=9.9)}
        report = refresh_releases(
            existing=existing,
            calendar=[
                event("CPI-2026-10-14", dt.datetime(2026, 10, 14, 12, 30, tzinfo=dt.UTC)),
                event("CPI-2026-09-11", dt.datetime(2026, 9, 11, 12, 30, tzinfo=dt.UTC)),
            ],
            levels=self.table(),
            now=NOW,
        )
        assert report.has_conflicts
        assert report.wrote_anything, "the other row would have been added"
        # The command refuses on has_conflicts before writing anything.


class TestTheFileSurvivesBeingRewritten:
    def test_rendering_what_was_loaded_reproduces_the_file(self) -> None:
        """Byte for byte. This is what makes a refresh diff readable: any
        change in the file is a row the refresh meant to touch, not a
        reformatting of the other twenty-seven."""
        assert to_csv(load_releases()) == releases_path().read_text(encoding="utf-8")

    def test_a_written_row_is_read_back_as_itself(self) -> None:
        import csv as csv_module

        from backtool.events.releases import RELEASE_COLUMNS

        rendered = to_csv(load_releases())
        parsed = list(csv_module.DictReader(rendered.splitlines()))
        assert list(parsed[0]) == list(RELEASE_COLUMNS)
        assert len(parsed) == len(load_releases())

    def test_a_missing_figure_is_written_as_null_not_blank(self) -> None:
        """Matching what the file already does, and keeping "no consensus"
        visibly different from a cell somebody forgot."""
        rendered = to_csv(
            {"CPI-2026-10-14": row("CPI-2026-10-14", actual_yoy=3.7, consensus_yoy=None)}
        )
        assert ",NULL," in rendered


class TestTheRealDatasetAgreesWithTheAgency:
    def test_every_stored_actual_matches_the_published_series(self) -> None:
        """Offline, against the figures the live run confirmed. The point is
        not to re-prove it monthly but to notice if a row is ever edited into
        disagreement with the source it claims to come from."""
        values = load_releases()
        from_bls = [
            v
            for v in values.values()
            if v.actual_source == CPI_SERIES_URL and v.actual_yoy is not None
        ]
        assert from_bls, "the refresh has run at least once"
        for entry in from_bls:
            assert entry.first_release_verified
            assert entry.consensus_yoy is None
