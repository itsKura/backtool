"""Command-line entry point.

Deliberately thin: this parses arguments and formats output. All research logic
lives in ``backtool.research``, all I/O in ``backtool.data``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import webbrowser
from pathlib import Path

from backtool.config import Settings
from backtool.core.time import NEW_YORK, utc_to_local
from backtool.core.types import EventType, Interval
from backtool.data.service import MarketDataService
from backtool.events import load_fomc_events, select_last_n
from backtool.reporting.html import write_report
from backtool.research.aggregate import aggregate_study
from backtool.research.runner import StudyResult, run_study
from backtool.research.spec import ResearchSpec
from backtool.research.windows import DEFAULT_WINDOWS, WindowSpec


def _configure_logging(verbose: bool, settings: Settings) -> None:
    logging.basicConfig(
        level="DEBUG" if verbose else settings.log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def _cmd_events(args: argparse.Namespace) -> int:
    """List calendar events for eyeball verification."""
    events = load_fomc_events()
    selected = events if args.all else select_last_n(events, args.limit)

    print(f"{'event_id':<18} {'weekday':<10} {'local (ET)':<18} {'UTC':<18} kind")
    print("-" * 78)
    for event in selected:
        local = utc_to_local(event.timestamp_utc, NEW_YORK)
        kind = "scheduled" if event.is_scheduled else "UNSCHEDULED"
        print(
            f"{event.event_id:<18} "
            f"{event.local_date.strftime('%A'):<10} "
            f"{local:%Y-%m-%d %H:%M}   "
            f"{event.timestamp_utc:%Y-%m-%d %H:%M}   "
            f"{kind}"
        )
        if event.notes:
            print(f"{'':<18} note: {event.notes}")

    print(f"\n{len(selected)} event(s) of {len(events)} in calendar.")
    return 0


# --------------------------------------------------------------------------
# analyse
# --------------------------------------------------------------------------


def _parse_as_of(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    parsed = dt.datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _build_spec(args: argparse.Namespace, as_of: dt.datetime | None) -> ResearchSpec:
    windows = DEFAULT_WINDOWS
    if args.window:
        windows = tuple(_parse_window(text) for text in args.window)

    return ResearchSpec(
        symbol=args.symbol,
        event_type=EventType.FOMC,
        event_count=args.count,
        interval=Interval(args.interval),
        windows=windows,
        as_of=as_of,
    )


def _parse_window(text: str) -> WindowSpec:
    """Parse ``name:start:end``, e.g. ``pre_event_2h:-2h:0h``."""
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid window {text!r}. Expected name:start:end, e.g. 'pre_2h:-2h:0h'"
        )
    return WindowSpec(name=parts[0], start=parts[1], end=parts[2])


def _cmd_analyse(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    as_of = _parse_as_of(args.as_of)
    plan = None

    if args.ask:
        from backtool.ai import AIError, plan_study, render_plan

        try:
            plan, spec = plan_study(args.ask, as_of=as_of)
        except AIError as exc:
            print(f"Planning failed: {exc}", file=sys.stderr)
            return 2
        print("\nRESEARCH PLAN")
        print("-" * 78)
        for line in render_plan(plan, spec):
            print(f"  {line}")
    else:
        spec = _build_spec(args, as_of)

    with MarketDataService.from_settings(settings) as service:
        study = run_study(spec, service, load_fomc_events())

    _print_assumptions(spec)
    _print_per_event(study, args.window_focus)
    _print_aggregates(study)

    interpretation = None
    if args.ask or args.explain:
        from backtool.ai import AIError, interpret_study, render_interpretation

        try:
            interpretation = interpret_study(study, question=args.ask)
        except AIError as exc:
            print(f"\nInterpretation unavailable: {exc}", file=sys.stderr)
        else:
            print("\nINTERPRETATION")
            print("-" * 78)
            for line in render_interpretation(interpretation):
                print(f"  {line}")

    if args.html:
        path = Path(write_report(study, args.html, interpretation=interpretation))
        print(f"\nHTML report written to {path}")
        if args.open:
            webbrowser.open(path.as_uri())
    return 0


def _print_assumptions(spec: ResearchSpec) -> None:
    print("\nASSUMPTIONS")
    print("-" * 78)
    for line in spec.describe():
        print(f"  {line}")


def _print_per_event(study: StudyResult, focus: str | None) -> None:
    window_names = [w.name for w in study.spec.windows]
    target = focus or window_names[0]
    if target not in window_names:
        print(f"\nUnknown window {target!r}; known: {', '.join(window_names)}")
        return

    print(f"\nPER-EVENT RETURNS (%)   [{study.coverage_note()}]")
    print("-" * 78)
    header = f"{'event':<18} " + " ".join(f"{name[:12]:>12}" for name in window_names)
    print(header)
    print("-" * len(header))

    for event in study.events:
        cells = []
        for name in window_names:
            metrics = event.windows[name]
            cells.append(
                f"{metrics.return_pct:>12.3f}" if metrics.is_usable else f"{'n/a':>12}"
            )
        print(f"{event.event_id:<18} " + " ".join(cells))

    incomplete = [e for e in study.events if not e.is_complete]
    if incomplete:
        print("\n  excluded observations:")
        for event in incomplete:
            for name, metrics in event.windows.items():
                if not metrics.is_usable:
                    print(f"    {event.event_id}  {name}: {metrics.reason}")


def _print_aggregates(study: StudyResult) -> None:
    aggregates = aggregate_study(study)

    print("\nAGGREGATE")
    print("-" * 78)
    header = (
        f"{'window':<16} {'n':>3} {'mean':>8} {'median':>8} {'stdev':>8} "
        f"{'up%':>6} {'p10':>8} {'p90':>8} {'range':>7}"
    )
    print(header)
    print("-" * len(header))

    for name, agg in aggregates.items():
        if agg.sample_size == 0:
            print(f"{name:<16} {'0':>3}   no usable observations")
            continue
        stdev = f"{agg.stdev_return_pct:>8.3f}" if agg.stdev_return_pct is not None else f"{'-':>8}"
        print(
            f"{name:<16} {agg.sample_size:>3} {agg.mean_return_pct:>8.3f} "
            f"{agg.median_return_pct:>8.3f} {stdev} {agg.up_rate_pct:>6.1f} "
            f"{agg.p10_return_pct:>8.3f} {agg.p90_return_pct:>8.3f} "
            f"{agg.mean_range_pct:>7.3f}"
        )

    caveats = [(name, agg.caveat) for name, agg in aggregates.items() if agg.caveat]
    if caveats:
        print()
        for name, caveat in caveats:
            print(f"  ! {name}: {caveat}")

    print(
        "\n  Exploratory statistics over a small sample. Not a trading signal;"
        "\n  no correction has been applied for multiple comparisons."
    )


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def _cmd_cache(args: argparse.Namespace) -> int:
    from backtool.data.storage.sqlite import SQLiteCandleRepository

    settings = Settings.from_env()
    repo = SQLiteCandleRepository(settings.candle_db)

    pairs = repo.symbols()
    if not pairs:
        print(f"Cache is empty ({repo.path}).")
        return 0

    size_mb = repo.path.stat().st_size / 1_048_576
    print(f"{repo.path}  ({size_mb:.1f} MB)\n")
    print(f"{'symbol':<12} {'interval':<10} {'candles':>10}  coverage")
    print("-" * 70)
    for symbol, interval_value in pairs:
        interval = Interval(interval_value)
        coverage = repo.coverage(symbol, interval)
        span = (
            f"{coverage.ranges[0].start:%Y-%m-%d} .. {coverage.ranges[-1].end:%Y-%m-%d}"
            f"  ({len(coverage)} range(s))"
            if coverage.ranges
            else "none"
        )
        print(
            f"{symbol:<12} {interval_value:<10} "
            f"{repo.candle_count(symbol, interval):>10,}  {span}"
        )
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtool",
        description="Deterministic event-study research engine for crypto markets.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    events = subparsers.add_parser("events", help="list events from the calendar")
    events.add_argument("-n", "--limit", type=int, default=20, help="how many to show")
    events.add_argument("--all", action="store_true", help="include future meetings")
    events.set_defaults(func=_cmd_events)

    analyse = subparsers.add_parser("analyse", help="run an event study")
    analyse.add_argument("-n", "--count", type=int, default=20, help="events to analyse")
    analyse.add_argument("--symbol", default="BTCUSDT")
    analyse.add_argument(
        "--interval", default="5m", choices=[i.value for i in Interval]
    )
    analyse.add_argument(
        "--window",
        action="append",
        metavar="NAME:START:END",
        help="override the default windows, e.g. --window pre_2h:-2h:0h (repeatable)",
    )
    analyse.add_argument(
        "--window-focus", help="window to highlight in the per-event table"
    )
    analyse.add_argument(
        "--as-of",
        metavar="ISO8601",
        help="analysis cutoff, e.g. 2026-01-01. Pin this for a reproducible run.",
    )
    analyse.add_argument(
        "--ask",
        metavar="QUESTION",
        help=(
            "describe the study in plain language; the AI planner builds the "
            "spec and explains the results. Needs ANTHROPIC_API_KEY."
        ),
    )
    analyse.add_argument(
        "--explain",
        action="store_true",
        help="interpret the results with AI without using it to plan",
    )
    analyse.add_argument(
        "--html",
        metavar="PATH",
        help="also write a self-contained HTML report with charts",
    )
    analyse.add_argument(
        "--open", action="store_true", help="open the HTML report in a browser"
    )
    analyse.set_defaults(func=_cmd_analyse)

    cache = subparsers.add_parser("cache", help="show what is stored locally")
    cache.set_defaults(func=_cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", False), Settings.from_env())
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
