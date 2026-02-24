"""
Tool: get_performance
Returns portfolio performance metrics across multiple time periods.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client


# ── Implementation (importable in unit tests without @tool overhead) ──────────

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

        perf = data.get("performance", {})

        # Extract all available periods
        periods = {}
        for period_key in ["1d", "wtd", "mtd", "ytd", "1y", "5y", "max"]:
            period_data = perf.get(period_key, {})
            if period_data:
                periods[period_key] = {
                    "relative_change": period_data.get("relativeChange", 0),
                    "absolute_change": period_data.get("absoluteChange", 0),
                    "current_value": period_data.get("currentValue", 0),
                    "net_worth": period_data.get("netWorth", 0),
                }

        requested = periods.get(date_range, {})

        return {
            "status": "ok",
            "requested_period": date_range,
            "performance": {
                "relative_change_pct": round(
                    requested.get("relative_change", 0) * 100, 2
                ),
                "absolute_change": round(requested.get("absolute_change", 0), 2),
                "current_value": round(requested.get("current_value", 0), 2),
            },
            "all_periods": periods,
            "data_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message, "error_code": e.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── LangChain Tool (used by the graph) ────────────────────────────────────────

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
