# LangChain tool that retrieves portfolio performance metrics for a requested time period from Ghostfolio.
"""
Tool: get_performance
Returns portfolio performance metrics across multiple time periods.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client

# ── Implementation (importable in unit tests without @tool overhead) ──────────


# Core logic: fetches and reshapes Ghostfolio performance data for the requested date range.
async def _get_performance(date_range: str = "ytd") -> dict[str, Any]:
    """
    Core logic for get_performance.
    Separated from the @tool wrapper so unit tests can call it directly.
    """
    valid_ranges = {"1d", "wtd", "mtd", "ytd", "1y", "5y", "max"}
    if date_range not in valid_ranges:
        date_range = "ytd"

    try:
        client = get_shared_client()
        data = await client.get_portfolio_performance(date_range)

        # v2 API returns a FLAT performance object for the requested range.
        # The ?range= param controls the calculation window; there are no
        # nested period keys (ytd/1y/etc.) in the response.
        perf = data.get("performance", {})

        net_perf_pct = perf.get("netPerformancePercentage", 0) or 0  # decimal: 0.123 = 12.3%
        net_performance = perf.get("netPerformance", 0) or 0  # absolute gain/loss in base currency
        current_value = perf.get("currentValueInBaseCurrency", 0) or 0
        total_investment = perf.get("totalInvestment", 0) or 0
        net_worth = perf.get("currentNetWorth", 0) or 0

        return {
            "status": "ok",
            "requested_period": date_range,
            "performance": {
                "relative_change_pct": round(net_perf_pct * 100, 2),  # convert fraction → %
                "absolute_change": round(net_performance, 2),
                "current_value": round(current_value, 2),
                "total_investment": round(total_investment, 2),
                "net_worth": round(net_worth, 2),
            },
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Ghostfolio",
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message, "error_code": e.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── LangChain Tool (used by the graph) ────────────────────────────────────────


# LangChain @tool wrapper — delegates to _get_performance; used by the agent graph.
@tool
async def get_performance(date_range: str = "ytd") -> dict[str, Any]:
    """
    Get portfolio performance metrics including return on average investment (ROAI),
    absolute gains/losses, and percentage changes across time periods.
    Use this when users ask about returns, performance, gains, losses, or how well
    their portfolio has done over any time period.

    Args:
        date_range: Time period for performance. Options: '1d', 'wtd', 'mtd',
                    'ytd', '1y', '5y', 'max'. Defaults to 'ytd'.

    Returns:
        Dictionary with performance metrics, absolute and relative changes.
    """
    return await _get_performance(date_range)
