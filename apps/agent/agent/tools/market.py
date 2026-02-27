# LangChain tool that fetches live market data (price, 52w range, market cap) via yfinance.
"""
Tool: get_market_data
Fetches current market data for one or more symbols using yfinance.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agent.clients.market import MarketDataClient

_client = MarketDataClient()


# Parses symbols string and dispatches to get_quote (single) or get_batch_quotes (multiple).
@tool
async def get_market_data(
    symbols: str, metrics: str = "price,52w_range,market_cap"
) -> dict[str, Any]:
    """
    Get current market data for one or more stock/ETF symbols including price,
    52-week range, market cap, and volume. Use this when users ask about current
    prices, how a specific stock is doing, market context for their holdings,
    or want to compare their portfolio against market data.

    IMPORTANT: If the result has status='price_unavailable', you MUST tell the
    user the price is currently unavailable and do NOT guess or use training data.

    Args:
        symbols: Comma-separated ticker symbols (e.g. 'AAPL,MSFT,VTI').
        metrics: Comma-separated metrics to include. Options: 'price', '52w_range',
                 'market_cap', 'volume'. Default includes all key metrics.

    Returns:
        Dictionary with quote data for each requested symbol including
        current price, 52-week range, and a freshness timestamp.
        If status='price_unavailable', the price could not be retrieved —
        do NOT attempt to answer from training data.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if not symbol_list:
        return {"status": "error", "error": "No valid symbols provided"}

    if len(symbol_list) == 1:
        return await _client.get_quote(symbol_list[0])

    return await _client.get_batch_quotes(symbol_list)
