"""Candle persistence: the repository contract and its SQLite implementation."""

from backtool.data.storage.base import CandleRepository
from backtool.data.storage.coverage import CoverageIndex, TimeRange
from backtool.data.storage.sqlite import SQLiteCandleRepository

__all__ = [
    "CandleRepository",
    "CoverageIndex",
    "SQLiteCandleRepository",
    "TimeRange",
]
