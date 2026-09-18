"""Tests for the web interface.

The server is a thin layer over the engine, so these cover the layer: request
parsing, error surfacing, and the invariant that the page is usable with no AI
configured. They do not re-test the research maths.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backtool.web.app import MAX_EVENTS, _parse_window, app
from backtool.web.page import render_page


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestPage:
    def test_serves_the_page(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "backtool" in response.text
        assert 'id="study-form"' in response.text

    def test_exactly_one_panel_is_visible_with_ai(self) -> None:
        html = render_page(ai_available=True)
        assert 'data-mode="ask" hidden' not in html
        assert 'data-mode="build" hidden' in html

    def test_exactly_one_panel_is_visible_without_ai(self) -> None:
        """Regression: both panels were hardcoded hidden, so with no API key
        the correct tab was selected and showed an empty form area."""
        html = render_page(ai_available=False)
        assert 'data-mode="ask" hidden' in html
        assert 'data-mode="build" hidden' not in html

    def test_ask_is_disabled_rather_than_removed_without_a_key(self) -> None:
        """A missing feature the user can see and fix beats one that silently
        is not there."""
        html = render_page(ai_available=False)
        assert "ANTHROPIC_API_KEY" in html
        assert "disabled" in html

    def test_report_styling_is_shared_not_duplicated(self) -> None:
        """A served result must look identical to a saved one."""
        from backtool.reporting.html import _STYLE

        assert _STYLE in render_page(ai_available=False)

    def test_page_is_self_contained(self) -> None:
        html = render_page(ai_available=True)
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "cdn." not in html


class TestHealth:
    def test_reports_capabilities(self, client: TestClient) -> None:
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert payload["max_events"] == MAX_EVENTS
        assert "5m" in payload["intervals"]
        assert isinstance(payload["ai_available"], bool)


class TestWindowParsing:
    def test_parses_a_valid_line(self) -> None:
        window = _parse_window("post_1h:0h:1h")
        assert (window.name, window.start, window.end) == ("post_1h", "0h", "1h")

    def test_tolerates_surrounding_whitespace(self) -> None:
        assert _parse_window("  post_1h : 0h : 1h  ").name == "post_1h"

    @pytest.mark.parametrize("line", ["nonsense", "a:b", "a:b:c:d"])
    def test_rejects_malformed_lines(self, line: str) -> None:
        with pytest.raises(ValueError, match="Invalid window"):
            _parse_window(line)


class TestAnalyseValidation:
    """Bad input must come back as a readable message, not a stack trace."""

    def test_rejects_too_many_events(self, client: TestClient) -> None:
        response = client.post(
            "/api/analyse", data={"mode": "build", "count": str(MAX_EVENTS + 1)}
        )
        assert response.status_code == 400
        assert "between 1 and" in response.text

    def test_rejects_zero_events(self, client: TestClient) -> None:
        response = client.post("/api/analyse", data={"mode": "build", "count": "0"})
        assert response.status_code == 400

    def test_rejects_a_malformed_window(self, client: TestClient) -> None:
        response = client.post(
            "/api/analyse",
            data={"mode": "build", "count": "5", "windows": "broken-line"},
        )
        assert response.status_code == 400
        assert "Invalid window" in response.text

    def test_rejects_a_backwards_window(self, client: TestClient) -> None:
        response = client.post(
            "/api/analyse",
            data={"mode": "build", "count": "5", "windows": "w:1h:0h"},
        )
        assert response.status_code == 400

    def test_ask_mode_requires_a_question(self, client: TestClient) -> None:
        response = client.post("/api/analyse", data={"mode": "ask", "question": "   "})
        assert response.status_code == 400
        assert "Enter a question" in response.text

    def test_errors_are_returned_as_markup(self, client: TestClient) -> None:
        """The client injects the response directly, so it must never have to
        branch on content type."""
        response = client.post("/api/analyse", data={"mode": "build", "count": "0"})
        assert response.headers["content-type"].startswith("text/html")
        assert '<div class="error">' in response.text


class TestAnalyseSuccess:
    """A study runs against a stubbed engine -- the maths is tested elsewhere."""

    def test_returns_report_markup(self, client: TestClient) -> None:
        from tests.unit.test_reporting import TestHtmlReport  # noqa: F401

        with patch("backtool.web.app.run_study") as run, patch(
            "backtool.web.app.render_body", return_value="<h2>Assumptions</h2>"
        ):
            run.return_value = object()
            response = client.post(
                "/api/analyse", data={"mode": "build", "count": "3", "interval": "1h"}
            )

        assert response.status_code == 200
        assert "<h2>Assumptions</h2>" in response.text

    def test_pins_the_as_of_cutoff(self, client: TestClient) -> None:
        """Without a pinned cutoff the same request would select different
        events tomorrow, and the fingerprint would stop meaning anything."""
        captured = {}

        def capture(spec, service, calendar, *, now=None):  # type: ignore[no-untyped-def]
            captured["as_of"] = spec.as_of
            return object()

        with patch("backtool.web.app.run_study", side_effect=capture), patch(
            "backtool.web.app.render_body", return_value=""
        ):
            client.post("/api/analyse", data={"mode": "build", "count": "3"})

        assert isinstance(captured["as_of"], dt.datetime)
        assert captured["as_of"].tzinfo is not None

    def test_interpretation_failure_does_not_lose_the_study(
        self, client: TestClient
    ) -> None:
        """The numbers are the product and are already computed by that point."""
        from backtool.ai import AIError

        with patch("backtool.web.app.run_study", return_value=object()), patch(
            "backtool.web.app.render_body", return_value="<h2>Results</h2>"
        ), patch(
            "backtool.ai.interpret_study", side_effect=AIError("model down")
        ), patch("backtool.ai.is_configured", return_value=True):
            response = client.post(
                "/api/analyse",
                data={"mode": "build", "count": "3", "explain": "1"},
            )

        assert response.status_code == 200
        assert "<h2>Results</h2>" in response.text
