"""Runtime configuration, sourced from environment variables.

Nothing in the codebase reads ``os.environ`` directly; everything goes through
:class:`Settings`. That keeps configuration greppable and makes tests able to
construct a Settings object without touching the process environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

#: Binance's public market-data mirror.
#:
#: ``api.binance.com`` returns HTTP 451 (Unavailable For Legal Reasons) from a
#: number of jurisdictions. ``data-api.binance.vision`` serves the identical
#: ``/api/v3/klines`` contract, requires no authentication, and is not
#: geo-restricted, so it is the better default for a market-data-only client.
#: Override with ``BINANCE_BASE_URL`` if you need the main API or a testnet.
DEFAULT_BINANCE_BASE_URL = "https://data-api.binance.vision"

#: Binance caps a single klines response at this many candles.
BINANCE_MAX_KLINES_PER_REQUEST = 1000


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    binance_base_url: str = DEFAULT_BINANCE_BASE_URL
    data_dir: Path = Path("data")
    request_timeout_s: float = 20.0
    max_retries: int = 4
    log_level: str = "INFO"

    @property
    def candle_db(self) -> Path:
        """SQLite file holding the candle cache and its coverage metadata."""
        return self.data_dir / "candles.db"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment, falling back to defaults."""
        return cls(
            binance_base_url=os.getenv("BINANCE_BASE_URL", DEFAULT_BINANCE_BASE_URL).rstrip("/"),
            data_dir=Path(os.getenv("BACKTOOL_DATA_DIR", "data")),
            request_timeout_s=float(os.getenv("BACKTOOL_REQUEST_TIMEOUT_S", "20")),
            max_retries=int(os.getenv("BACKTOOL_MAX_RETRIES", "4")),
            log_level=os.getenv("BACKTOOL_LOG_LEVEL", "INFO").upper(),
        )
