"""The architectural constraint that makes this system trustworthy.

`ai/` may never reach the market-data layer. The model emits specifications and
consumes computed results; it cannot touch a candle, so it cannot fabricate a
number that looks like one.

That is the whole product claim, and a comment cannot enforce it. Without this
test, someone would eventually write "just pass the last 500 candles to the
model and ask what the average move was" -- which would work, produce confident
output, and be wrong with nothing to flag it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

AI_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "backtool" / "ai"

#: Modules the AI layer must not reach, and why.
FORBIDDEN = {
    "backtool.data": (
        "the AI layer must not access market data; it consumes computed results"
    ),
}


def ai_modules() -> list[Path]:
    return sorted(AI_PACKAGE.glob("*.py"))


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by a file, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_the_ai_package_exists_and_has_modules() -> None:
    """Guards the guard: if the package moved, the test below would pass
    vacuously by finding nothing to check."""
    modules = ai_modules()
    assert modules, f"no modules found under {AI_PACKAGE}"
    assert {p.name for p in modules} >= {
        "client.py",
        "planner.py",
        "interpreter.py",
        "models.py",
    }


@pytest.mark.parametrize("path", ai_modules(), ids=lambda p: p.name)
def test_ai_module_does_not_import_forbidden_layers(path: Path) -> None:
    for module in imported_modules(path):
        for forbidden, reason in FORBIDDEN.items():
            assert not (module == forbidden or module.startswith(f"{forbidden}.")), (
                f"{path.name} imports {module!r}: {reason}"
            )


def test_ai_package_imports_without_pulling_in_the_data_layer() -> None:
    """A transitive import would defeat the AST check above."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import backtool.ai, sys;"
            "leaked = [m for m in sys.modules if m.startswith('backtool.data')];"
            "print(','.join(sorted(leaked)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = result.stdout.strip()
    assert not leaked, f"importing backtool.ai transitively loaded: {leaked}"


def test_interpreter_payload_contains_no_price_series() -> None:
    """Belt and braces: even the data handed to the model must be statistics,
    not a price path the model could 'analyse' itself."""
    import datetime as dt

    from backtool.ai.interpreter import build_payload
    from backtool.research.results import EventResult, StudyResult
    from backtool.research.spec import ResearchSpec
    from backtool.research.windows import WindowSpec

    spec = ResearchSpec(
        event_count=1,
        windows=(WindowSpec(name="post_event_1h", start="0h", end="1h"),),
        as_of=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
    )
    from backtool.research.metrics import Direction, WindowMetrics, WindowStatus

    metrics = WindowMetrics(
        name="post_event_1h",
        start_utc=dt.datetime(2024, 1, 31, 19, 0, tzinfo=dt.UTC),
        end_utc=dt.datetime(2024, 1, 31, 20, 0, tzinfo=dt.UTC),
        status=WindowStatus.OK,
        start_price=43000.0,
        end_price=43500.0,
        return_pct=1.1627,
        abs_move_pct=1.1627,
        direction=Direction.UP,
        high=43600.0,
        low=42950.0,
        mfe_pct=1.3953,
        mae_pct=-0.1163,
        range_pct=1.5116,
        realized_vol_pct=0.42,
        candle_count=12,
        expected_candles=12,
        coverage_pct=100.0,
    )
    study = StudyResult(
        spec=spec,
        generated_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
        events=(
            EventResult(
                event_id="FOMC-2024-01-31",
                event_time_utc=dt.datetime(2024, 1, 31, 19, 0, tzinfo=dt.UTC),
                local_date=dt.date(2024, 1, 31),
                is_scheduled=True,
                windows={"post_event_1h": metrics},
            ),
        ),
    )

    payload = build_payload(study)

    assert set(payload) == {"question", "assumptions", "coverage", "aggregates", "per_event"}
    for event in payload["per_event"]:
        for window in event["windows"].values():
            assert "open" not in window
            assert "high" not in window
            assert "low" not in window
            assert "close" not in window
            assert "volume" not in window
