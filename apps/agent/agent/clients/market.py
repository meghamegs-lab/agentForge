"""
Market data client using yfinance.
Returns structured dicts with timestamps for freshness checking.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import yfinance as yf


class MarketDataClient:
    """Fetches real-time and historical market data via yfinance."""

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get current price and key stats for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            hist = ticker.history(period="1d")

            if hist.empty:
                return {
                    "status": "error",
                    "error": "symbol_not_found",
                    "symbol": symbol,
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
                "status": "error",
                "error": str(e),
                "symbol": symbol,
            }

    def get_batch_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """Get quotes for multiple symbols at once."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_quote(symbol)
        return {
            "status": "ok",
            "quotes": results,
            "data_timestamp": datetime.now(UTC).isoformat(),
        }
