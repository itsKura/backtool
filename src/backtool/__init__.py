"""backtool: deterministic event-study research engine for crypto markets.

Loading ``.env`` happens here, at package import, rather than inside any one
module. It previously lived in :mod:`backtool.config`, which meant anything
reading an environment variable without importing ``Settings`` saw an unloaded
environment -- ``backtool.ai`` reported "no API key" with a valid key sitting in
``.env``, and only because the CLI and web app happen to import ``Settings``
first did it ever work at all. Importing any submodule runs this file, so there
is now no import order that skips it.
"""

from dotenv import load_dotenv

# Existing environment variables win, so a shell export or CI secret is never
# silently overridden by a stale local file.
load_dotenv(override=False)

__version__ = "0.1.0"
