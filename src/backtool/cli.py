"""Command-line entry point.

Deliberately thin. The CLI parses arguments and prints; it contains no research
logic. As milestones land, subcommands are added here and the work happens in
``backtool.research``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from backtool.core.time import NEW_YORK, utc_to_local
from backtool.events import load_fomc_events, select_last_n


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("BACKTOOL_LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _cmd_events(args: argparse.Namespace) -> int:
    """List FOMC events, newest last, for eyeball verification."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtool",
        description="Deterministic event-study research engine for crypto markets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    events = subparsers.add_parser("events", help="list events from the calendar")
    events.add_argument(
        "-n", "--limit", type=int, default=20, help="how many recent events to show (default: 20)"
    )
    events.add_argument(
        "--all", action="store_true", help="show the entire calendar, including future meetings"
    )
    events.set_defaults(func=_cmd_events)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
