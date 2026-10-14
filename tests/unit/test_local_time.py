"""Timestamps shown on the reader's own clock.

"as of 23:48 UTC" asks somebody checking whether a price is fresh to do
arithmetic in their head against an offset they have to remember. The server
cannot know their timezone -- it is a property of the browser -- so it writes
UTC and the browser rewrites it.

What these protect is mostly the failure direction. A page that cannot run the
script must still state a true time, not an empty element or a wrong one, and
the markup has to carry enough for the rewrite to be exact rather than
approximate.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest
from fastapi.testclient import TestClient

from backtool.core.types import EventType
from backtool.events import load_events
from backtool.web.app import app
from backtool.web.timefmt import LOCAL_TIME_SCRIPT, local_stamp

NOON = dt.datetime(2026, 11, 10, 13, 30, tzinfo=dt.UTC)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestTheStampCarriesWhatTheBrowserNeeds:
    def test_it_states_a_true_time_without_any_script(self) -> None:
        """The failure that matters. Scripting off, a slow parse, a stripped
        script tag -- the reader still sees a correct instant rather than an
        empty element where a time should be."""
        assert ">13:30 UTC</time>" in local_stamp(NOON)

    def test_it_carries_the_instant_a_machine_can_read(self) -> None:
        html = local_stamp(NOON)
        found = re.search(r'datetime="([^"]+)"', html)
        assert found is not None
        assert dt.datetime.fromisoformat(found.group(1)) == NOON

    def test_the_date_is_optional(self) -> None:
        """A stamp that could be days old needs one; "21:26" alone invites the
        reader to assume today."""
        assert "10 Nov 2026" in local_stamp(NOON, with_date=True)
        assert "10 Nov 2026" not in local_stamp(NOON)

    def test_the_separator_travels_inside_the_element(self) -> None:
        """So a stamp the browser drops as redundant takes its separator with
        it, rather than leaving one pointing at nothing."""
        html = local_stamp(NOON, lead=" · ")
        assert html.index("·") > html.index("<time")
        assert html.index("·") < html.index("</time>")

    def test_a_duplicate_is_marked_for_removal(self) -> None:
        assert 'data-instead-of="08:30"' in local_stamp(NOON, instead_of="08:30")

    def test_nothing_is_marked_when_there_is_nothing_to_duplicate(self) -> None:
        assert "data-instead-of" not in local_stamp(NOON)

    def test_the_markup_survives_an_odd_value(self) -> None:
        """The attribute is quoted rather than concatenated, so a value cannot
        add an attribute of its own. Everything passed here comes from our own
        formatting; this is so that stays true of a caller who forgets.

        Asserted by parsing rather than by searching the text: the value is
        quoted correctly whether the quoting is done with entities or by
        switching the delimiter, and a string search would call one of those
        two correct answers a failure.
        """
        from html.parser import HTMLParser

        seen: list[dict[str, str | None]] = []

        class Reader(HTMLParser):
            def handle_starttag(
                self, tag: str, attrs: list[tuple[str, str | None]]
            ) -> None:
                if tag == "time":
                    seen.append(dict(attrs))

        hostile = '08:30" onload="alert(1)'
        Reader().feed(local_stamp(NOON, instead_of=hostile))

        assert len(seen) == 1
        assert seen[0]["data-instead-of"] == hostile
        assert "onload" not in seen[0]


class TestTheRewriteIsCarefulAboutFailing:
    """Assertions on the script as text, the way this project tests the chart.

    Not a substitute for running it -- that happened in a browser -- but these
    are the branches that only execute on a reader's machine, in a locale or a
    state no test here reproduces.
    """

    def test_an_unparseable_instant_is_left_alone(self) -> None:
        assert "isNaN(when.getTime())" in LOCAL_TIME_SCRIPT
        assert "continue" in LOCAL_TIME_SCRIPT

    def test_a_locale_that_throws_leaves_the_server_text(self) -> None:
        assert "catch (err)" in LOCAL_TIME_SCRIPT

    def test_the_utc_reading_is_kept_on_hover(self) -> None:
        """Nothing is actually lost by the rewrite -- UTC is still the anchor
        every figure on the page was measured against."""
        assert "el.title = text.trim()" in LOCAL_TIME_SCRIPT

    def test_the_clock_is_twenty_four_hour(self) -> None:
        """Every other time on these pages is. "08:30 ET" beside
        "07:48 PM EDT" reads as two different products."""
        assert "hour12: false" in LOCAL_TIME_SCRIPT

    def test_the_duplicate_check_compares_like_with_like(self) -> None:
        """The value it is compared against is a 24-hour HH:MM written by
        Python, so the reading taken for the comparison has to be too --
        whatever the reader's locale would have chosen."""
        comparison = LOCAL_TIME_SCRIPT.split("data-instead-of")[1]
        assert "hour12: false" in comparison


class TestEveryPageGetsIt:
    def test_the_calendar_stamps_its_events(self, client: TestClient) -> None:
        html = client.get("/").text
        assert "<time datetime=" in html
        assert "time[data-local]" in html

    def test_the_preparation_page_stamps_its_price(self, client: TestClient) -> None:
        event = next(
            item
            for item in load_events(EventType.CPI)
            if item.timestamp_utc > dt.datetime.now(tz=dt.UTC)
        )
        html = client.get(f"/event/{event.event_id}/prepare").text
        stamps = re.findall(r"<time[^>]*>[^<]*</time>", html)
        assert len(stamps) >= 2, "the release time and the price time"
        assert any("data-instead-of" in stamp for stamp in stamps)
        assert any("data-instead-of" not in stamp for stamp in stamps)

    def test_the_release_time_is_marked_against_its_own_zone(
        self, client: TestClient
    ) -> None:
        """CPI prints at 08:30 New York. A reader there should see that once,
        not twice in two spellings."""
        event = next(
            item
            for item in load_events(EventType.CPI)
            if item.timestamp_utc > dt.datetime.now(tz=dt.UTC)
        )
        html = client.get(f"/event/{event.event_id}/prepare").text
        assert 'data-instead-of="08:30"' in html

    def test_a_page_with_no_script_of_its_own_still_rewrites(
        self, client: TestClient
    ) -> None:
        """The preparation page carries no page script. If the rewrite lived
        with the pages that happened to want it, this is the one it would have
        been forgotten on."""
        event = next(
            item
            for item in load_events(EventType.CPI)
            if item.timestamp_utc > dt.datetime.now(tz=dt.UTC)
        )
        assert "time[data-local]" in client.get(
            f"/event/{event.event_id}/prepare"
        ).text

    def test_the_research_page_gets_it_too(self, client: TestClient) -> None:
        assert "time[data-local]" in client.get("/research").text
