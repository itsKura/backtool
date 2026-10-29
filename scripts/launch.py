"""Start Backtool and open it, from a desktop shortcut.

Double-clicking an icon should do the obvious thing: if the server is already
up, open the browser at it; if it is not, start it, wait until it actually
answers, then open the browser.

Waiting for the real answer is the point. Opening the browser the instant
uvicorn is told to start lands on a connection error about half the time,
because the app loads two calendars and a candle cache before it binds. So
this polls the health endpoint and opens only once something replies.

The server then runs in the foreground, so the console window is the log and
Ctrl-C stops it. Starting it detached would leave an invisible process holding
the port, which is worse than no shortcut at all.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

HOST = "127.0.0.1"
PORT = 8010
#: Where to land. The calendar is the front door.
LANDING = "/"

#: How long to wait for the server to answer before opening anyway. A cold
#: candle cache makes the first request slow; giving up early and showing a
#: connection error would be the one failure this script exists to prevent.
STARTUP_TIMEOUT_S = 90.0


def already_listening(host: str, port: int) -> bool:
    """Whether something already holds the port.

    Checked before starting anything: a second uvicorn on a taken port exits
    with a stack trace, and the honest response to "it is already running" is
    to open the browser, not to report an error.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.6)
        return probe.connect_ex((host, port)) == 0


def answers(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bool(response.status < 500)
    except (urllib.error.URLError, OSError):
        return False


def open_when_ready(url: str, health: str) -> None:
    """Poll until the app replies, then open the browser once."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if answers(health):
            webbrowser.open(url)
            return
        time.sleep(0.4)
    print(
        f"\nThe server did not answer within {STARTUP_TIMEOUT_S:.0f}s. "
        f"It may still be starting -- try {url} in a browser.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Backtool and open it.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--no-browser", action="store_true", help="start the server only"
    )
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}{LANDING}"
    health = f"http://{args.host}:{args.port}/api/health"

    if already_listening(args.host, args.port):
        print(f"Backtool is already running. Opening {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    print(f"Starting Backtool on {url}")
    print("Close this window, or press Ctrl-C, to stop it.\n")

    if not args.no_browser:
        threading.Thread(
            target=open_when_ready, args=(url, health), daemon=True
        ).start()

    import uvicorn

    try:
        uvicorn.run("backtool.web.app:app", host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
