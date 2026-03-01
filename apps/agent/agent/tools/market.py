# LangChain tool that fetches live market data (price, 52w range, market cap) via yfinance.
"""
Tool: get_market_data
Fetches current market data for one or more symbols using yfinance.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agent.clients.market import get_shared_market_client

_client = get_shared_market_client()


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

    For general market overview questions ("how's the market today?", "how are
    markets doing?", "what's happening in the market?") — use SPY, QQQ, and ^DJI
    as the default benchmark symbols: get_market_data("SPY,QQQ,^DJI").
    These cover the S&P 500, Nasdaq 100, and Dow Jones Industrial Average.

    IMPORTANT: If the result has status='price_unavailable', you MUST tell the
    user the price is currently unavailable and do NOT guess or use training data.

    Args:
        symbols: Comma-separated ticker symbols (e.g. 'AAPL,MSFT,VTI').
                 For general market overview, use 'SPY,QQQ,^DJI'.
        metrics: Comma-separated metrics to include. Options: 'price', '52w_range',
                 'market_cap', 'volume'. Default includes all key metrics.

    Returns:
        Dictionary with quote data for each requested symbol including
        current price, 52-week range, and a freshness timestamp.
        If status='price_unavailable', the price could not be retrieved —
        do NOT attempt to answer from training data.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbol_list:
            return {"status": "error", "error": "No valid symbols provided"}

        if len(symbol_list) == 1:
            return await _client.get_quote(symbol_list[0])

        return await _client.get_batch_quotes(symbol_list)
    except Exception as e:
        return {"status": "error", "error": str(e)}
