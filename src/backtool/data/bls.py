"""CPI figures from the agency that publishes them.

The actual and previous columns of the release file were typed in by hand from
contemporaneous reporting. They did not need to be: CPI's source of record is
the Bureau of Labor Statistics, which serves the whole series through a free
public API designed for exactly this.

One series, deliberately
------------------------
:data:`CPI_SERIES` is ``CUUR0000SA0`` -- CPI-U, all items, U.S. city average,
**not seasonally adjusted**, which is word for word the measure
:data:`~backtool.events.releases.CPI_METRIC` names. Core, month-over-month and
the seasonally adjusted variants are different series with different ids, and
mixing them would produce a column that looks identical and means something
else.

What this does not supply
-------------------------
Consensus. It is not a published statistic -- it is a survey of economists that
Reuters, Bloomberg and Dow Jones each run and sell -- so no agency has it and
no free endpoint serves it. The consensus column stays hand-transcribed, and
nothing here touches it.

Offline, not on the page
------------------------
Nothing in the web layer calls this. A page that fetched from BLS while
rendering would trade a guaranteed, free, instant read of a local file for a
network round trip with its own rate limit and failure modes, to produce
figures that change twelve times a year. The refresh is a command; the page
stays a file read.
"""

from __future__ import annotations

import datetime as dt
import os
from types import TracebackType

import httpx

#: CPI-U, all items, U.S. city average, not seasonally adjusted.
CPI_SERIES = "CUUR0000SA0"

#: The published landing page for the series, recorded as provenance on every
#: row this fills. A bare API endpoint is not something a reader can check.
CPI_SERIES_URL = "https://data.bls.gov/timeseries/CUUR0000SA0"

#: Version 2 of the public timeseries API.
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

#: How far back a single unregistered request may reach.
#:
#: The v2 API serves 10 years without a key and 20 with one. Requesting more
#: than the allowance returns an error rather than a truncated series, so the
#: span is clamped to what the current credentials actually permit.
YEARS_WITHOUT_KEY = 10
YEARS_WITH_KEY = 20


class BLSError(RuntimeError):
    """The API answered, and the answer was not usable."""


class BLSClient:
    """A reader for one published series.

    Mirrors :class:`~backtool.data.binance.client.BinanceClient`: a context
    manager around one ``httpx.Client``, its own error type, and no retry
    cleverness, because this is called by hand a dozen times a year rather than
    in a loop.

    The API key is optional. Without one the endpoint still answers, with a
    lower daily quota and a shorter history, which is ample for a refresh that
    runs monthly.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._key = api_key if api_key is not None else os.getenv("BLS_API_KEY") or None
        self._http = http_client or httpx.Client(
            timeout=timeout_s, headers={"Content-Type": "application/json"}
        )

    def __enter__(self) -> BLSClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    @property
    def max_years(self) -> int:
        return YEARS_WITH_KEY if self._key else YEARS_WITHOUT_KEY

    def index_by_month(
        self, *, start_year: int, end_year: int
    ) -> dict[tuple[int, int], float]:
        """Published index levels, keyed by ``(year, month)``.

        Months BLS lists but has not published yet come back as ``"-"`` and are
        omitted rather than coerced: an unpublished month is not a zero, and a
        gap here becomes a figure this refresh declines to fill rather than one
        it invents.
        """
        span = end_year - start_year + 1
        if span > self.max_years:
            raise BLSError(
                f"{span} years requested; this key allows {self.max_years}. "
                "Set BLS_API_KEY for the longer history."
            )

        payload: dict[str, object] = {
            "seriesid": [CPI_SERIES],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self._key:
            payload["registrationkey"] = self._key

        try:
            response = self._http.post(BLS_API_URL, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise BLSError(f"could not reach the BLS API: {exc}") from exc
        except ValueError as exc:
            raise BLSError(f"the BLS API returned something that is not JSON: {exc}") from exc

        if body.get("status") != "REQUEST_SUCCEEDED":
            messages = "; ".join(body.get("message") or []) or "no reason given"
            raise BLSError(f"BLS refused the request: {messages}")

        levels: dict[tuple[int, int], float] = {}
        for series in body.get("Results", {}).get("series", []):
            if series.get("seriesID") != CPI_SERIES:
                continue
            for row in series.get("data", []):
                period = str(row.get("period", ""))
                value = str(row.get("value", ""))
                # M13 is the annual average, not a month.
                if not period.startswith("M") or period == "M13":
                    continue
                if value in ("", "-"):
                    continue
                try:
                    levels[(int(row["year"]), int(period[1:]))] = float(value)
                except (KeyError, ValueError):
                    continue

        if not levels:
            raise BLSError(f"no published values for {CPI_SERIES} in that range")
        return levels


def year_over_year(
    levels: dict[tuple[int, int], float], year: int, month: int
) -> float | None:
    """The published year-over-year change for one reference month.

    Rounded to one decimal place, which is the precision CPI is reported and
    forecast at -- and therefore the precision the scenario rule compares at. A
    figure carried to more places here would classify releases by digits no
    source ever published.

    ``None`` when either month is missing, because a year-over-year change
    needs both and half of one is not an estimate of it.
    """
    now = levels.get((year, month))
    then = levels.get((year - 1, month))
    if now is None or then is None or then == 0:
        return None
    return round((now / then - 1) * 100, 1)


def reference_month(release_date: dt.date) -> tuple[int, int]:
    """The month a release reports on: the one before it.

    True of every row in the shipped dataset, and of the schedule generally --
    CPI for a month is published partway through the next one.
    """
    if release_date.month == 1:
        return release_date.year - 1, 12
    return release_date.year, release_date.month - 1
