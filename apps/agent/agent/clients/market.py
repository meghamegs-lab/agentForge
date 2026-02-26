"""
Market data client using yfinance.
Returns structured dicts with timestamps for freshness checking.

Implementation note
-------------------
yfinance is a synchronous library.  To avoid blocking the asyncio event loop,
all network calls are dispatched to a thread-pool executor via asyncio.to_thread().
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import yfinance as yf


class MarketDataClient:
    """Async-safe market data client backed by yfinance (run in thread pool)."""

    # ── internal sync helpers (run in thread pool) ────────────────────────────

    def _fetch_quote_sync(self, symbol: str) -> dict[str, Any]:
        """Synchronous yfinance fetch — called from asyncio.to_thread()."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            hist = ticker.history(period="1d")

            if hist.empty:
                return {
                    "status": "price_unavailable",
                    "symbol": symbol,
                    "error": "No market data returned — market may be closed or symbol invalid",
                    "instruction": (
                        "Tell the user the price is currently unavailable. "
                        "Do NOT guess or use training data."
                    ),
                }

            current_price = float(hist["Close"].iloc[-1])

            return {
                "status": "ok",
                "symbol": symbol.upper(),
                "current_price": current_price,
                "currency": getattr(info, "currency", "USD"),
                "fifty_two_week_high": getattr(info, "year_high", None),
                "fifty_two_week_low": getattr(info, "year_low", None),
                "market_cap": getattr(info, "market_cap", None),
                "volume": getattr(info, "three_month_average_volume", None),
                "data_timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "status": "price_unavailable",
                "symbol": symbol,
                "error": str(e),
                "instruction": (
                    "Tell the user the price is currently unavailable due to a data "
                    "provider error. Do NOT guess or use training data."
                ),
            }

    # ── public async API ──────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Async get_quote — runs yfinance in thread pool to avoid blocking."""
        return await asyncio.to_thread(self._fetch_quote_sync, symbol)

    async def get_batch_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """Async batch — fetches all symbols concurrently in the thread pool."""
        results = await asyncio.gather(*[self.get_quote(s) for s in symbols])
        return {
            "status": "ok",
            "quotes": {s: r for s, r in zip(symbols, results)},
            "data_timestamp": datetime.now(UTC).isoformat(),
        }
