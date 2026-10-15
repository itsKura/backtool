"""Filling the release file from BLS, without ever overwriting a person.

The actual and previous columns can be derived from the published series. The
consensus column cannot: it is a survey somebody sells, and nothing here
touches it. So this is not a replacement for the curated file -- it is the
typing, done by a machine, into the same file with the same provenance rules.

What it is allowed to do
------------------------
Fill a gap, and add a row for a release the file has not reached yet. That is
all.

What it is not allowed to do
----------------------------
Change a figure a person verified. If BLS disagrees with a stored verified
value the disagreement is **reported and the file left alone**, because the two
possibilities are a transcription error and a revision, and picking one
silently would destroy the evidence needed to tell them apart. A refresh that
quietly rewrote history would also make the 27-for-27 agreement that justified
this integration unrepeatable.

Pure on purpose: it takes figures and rows and returns rows and a report, so
the rules above are testable without a network, a key, or a file.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from enum import StrEnum

from backtool.data.bls import CPI_SERIES_URL, reference_month, year_over_year
from backtool.events.models import MarketEvent
from backtool.events.releases import ReleaseValues, to_event_id


class Change(StrEnum):
    """What a refresh did, or declined to do, to one release."""

    ADDED = "added"
    FILLED = "filled"
    UNCHANGED = "unchanged"
    #: BLS has not published the reference month yet.
    AWAITING_DATA = "awaiting_data"
    #: Stored and published figures differ. Reported, never resolved here.
    CONFLICT = "conflict"


@dataclass(frozen=True)
class RowChange:
    """One release, what happened to it, and why."""

    event_id: str
    change: Change
    detail: str = ""

    @property
    def is_notable(self) -> bool:
        """Whether a reader needs to see this line."""
        return self.change is not Change.UNCHANGED


@dataclass(frozen=True)
class RefreshReport:
    """Everything a refresh would do, before anything is written."""

    rows: dict[str, ReleaseValues]
    changes: tuple[RowChange, ...]

    def of(self, kind: Change) -> tuple[RowChange, ...]:
        return tuple(change for change in self.changes if change.change is kind)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.of(Change.CONFLICT))

    @property
    def wrote_anything(self) -> bool:
        return bool(self.of(Change.ADDED) or self.of(Change.FILLED))


def refresh_releases(
    *,
    existing: dict[str, ReleaseValues],
    calendar: list[MarketEvent],
    levels: dict[tuple[int, int], float],
    now: dt.datetime,
) -> RefreshReport:
    """Merge published figures into the stored rows.

    Args:
        existing: Rows as the file holds them, keyed by event id.
        calendar: The releases that exist, which is what decides whether a row
            is missing. A file with no row for next month is not incomplete
            until the calendar says that release happened.
        levels: Published index levels by ``(year, month)``.
        now: The current instant, which decides whether an actual can exist
            yet. It does not decide whether a row can: the *previous* figure
            for a release still weeks away is simply the last one published,
            and it is one of the three numbers somebody opens the page to see.

    Returns:
        The rows as they would be, and a line per release explaining itself.
    """
    rows = dict(existing)
    changes: list[RowChange] = []

    for event in sorted(calendar, key=lambda item: item.timestamp_utc):
        release_date = event.timestamp_utc.date()
        year, month = reference_month(release_date)
        previous = year_over_year(levels, *_month_before(year, month))
        # An actual cannot exist before the release, whatever BLS holds.
        actual = (
            year_over_year(levels, year, month)
            if event.timestamp_utc <= now
            else None
        )

        if actual is None and previous is None:
            changes.append(
                RowChange(
                    event.event_id,
                    Change.AWAITING_DATA,
                    f"BLS has published neither {year}-{month:02d} nor the "
                    "month before it",
                )
            )
            continue

        current = rows.get(event.event_id)
        if current is None:
            rows[event.event_id] = _new_row(
                event.event_id, release_date, year, month, actual, previous
            )
            figures = [f"actual {actual:.1f}%"] if actual is not None else []
            if previous is not None:
                figures.append(f"previous {previous:.1f}%")
            changes.append(
                RowChange(event.event_id, Change.ADDED, ", ".join(figures))
            )
            continue

        changes.append(_reconcile(current, actual, previous, rows))

    return RefreshReport(rows=rows, changes=tuple(changes))


def _reconcile(
    current: ReleaseValues,
    actual: float | None,
    previous: float | None,
    rows: dict[str, ReleaseValues],
) -> RowChange:
    """Compare a stored row against the published figures.

    A disagreement is reported rather than resolved. Two things produce one --
    a transcription error and a revision -- and they want opposite responses,
    so the refresh stops and says what it found.

    ``actual`` is ``None`` for a release that has not happened, where there is
    nothing to compare and nothing to fill. Its *previous* can still arrive, so
    the row is still worth visiting.
    """
    disagreements: list[str] = []
    if (
        actual is not None
        and current.actual_yoy is not None
        and not _same(current.actual_yoy, actual)
    ):
        disagreements.append(f"actual stored {current.actual_yoy:.1f}%, BLS {actual:.1f}%")
    if (
        previous is not None
        and current.previous_yoy is not None
        and not _same(current.previous_yoy, previous)
    ):
        disagreements.append(
            f"previous stored {current.previous_yoy:.1f}%, BLS {previous:.1f}%"
        )
    if disagreements:
        return RowChange(current.event_id, Change.CONFLICT, "; ".join(disagreements))

    filled: list[str] = []
    updated = current
    if actual is not None and current.actual_yoy is None:
        updated = replace(
            updated,
            actual_yoy=actual,
            actual_source=CPI_SERIES_URL,
            first_release_verified=True,
        )
        filled.append(f"actual {actual:.1f}%")
    if current.previous_yoy is None and previous is not None:
        updated = replace(updated, previous_yoy=previous)
        filled.append(f"previous {previous:.1f}%")

    if not filled:
        return RowChange(current.event_id, Change.UNCHANGED)

    rows[current.event_id] = updated
    return RowChange(current.event_id, Change.FILLED, ", ".join(filled))


def _new_row(
    event_id: str,
    release_date: dt.date,
    year: int,
    month: int,
    actual: float | None,
    previous: float | None,
) -> ReleaseValues:
    """A row for a release the file had not reached.

    Consensus is left empty, which is the honest state: this source has none,
    and a row without one is excluded from the scenario cohorts rather than
    counted as in line. So a refresh adds releases to the page's header and to
    its history table without adding them to the statistics -- which is
    correct, and worth knowing before anybody wonders why the counts held
    still.

    An upcoming release gets a row too, carrying its previous figure and no
    actual. The two absences are different and the page already renders them
    differently: a missing actual before the print is "not published until the
    release", not a gap in this dataset.
    """
    carries_actual = actual is not None
    return ReleaseValues(
        event_id=event_id,
        source_id=event_id.lower().replace("-", "_"),
        release_date=release_date,
        reference_month=f"{year}-{month:02d}",
        previous_yoy=previous,
        consensus_yoy=None,
        actual_yoy=actual,
        # Provenance belongs to a figure, not to an intention. A row still
        # waiting for its actual claims no source for it.
        actual_source=CPI_SERIES_URL if carries_actual else "",
        first_release_verified=carries_actual,
        consensus_verified=False,
        notes=(
            f"{dt.date(year, month, 1):%B %Y} reference month. "
            + (
                "Actual and previous from the BLS published series"
                if carries_actual
                else "Previous from the BLS published series; actual is not "
                "published until the release"
            )
            + "; no consensus recorded, so this release is not sorted into a "
            "scenario."
        ),
    )


def _same(stored: float, published: float) -> bool:
    """Equal at the precision CPI is published to.

    Compared as integer tenths rather than through a tolerance on a
    subtraction, for the reason the scenario rule is: two figures published to
    one decimal place either are the same number or they are not.
    """
    return round(stored * 10) == round(published * 10)


def _month_before(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def to_source_id(event_id: str) -> str:
    """``CPI-2026-09-11`` -> ``cpi_2026_09_11``, the file's own spelling.

    The inverse of :func:`~backtool.events.releases.to_event_id`, so a row this
    module writes is read back by the loader as the release it describes.
    """
    return event_id.lower().replace("-", "_")


__all__ = [
    "Change",
    "RefreshReport",
    "RowChange",
    "refresh_releases",
    "to_event_id",
    "to_source_id",
]
