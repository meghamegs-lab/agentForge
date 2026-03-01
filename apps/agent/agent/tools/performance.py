# LangChain tool that retrieves portfolio performance metrics for a requested time period from Ghostfolio.
"""
Tool: get_performance
Returns portfolio performance metrics across multiple time periods.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client

# ── Implementation (importable in unit tests without @tool overhead) ──────────


def _last_month_range() -> tuple[str, str]:
    """Return (date_from, date_to) ISO strings for the previous calendar month."""
    today = date.today()
    first_of_this_month = today.replace(day=1)
    last_day_prev = first_of_this_month - timedelta(days=1)
    first_day_prev = last_day_prev.replace(day=1)
    return first_day_prev.isoformat(), last_day_prev.isoformat()


# Core logic: fetches and reshapes Ghostfolio performance data for the requested date range.
async def _get_performance(date_range: str = "ytd") -> dict[str, Any]:
    """
    Core logic for get_performance.
    Separated from the @tool wrapper so unit tests can call it directly.
    """
    valid_ranges = {"1d", "wtd", "mtd", "ytd", "1y", "5y", "max"}

    # "1m" / "last_month" → fetch the previous calendar month via two ytd calls
    # by mapping to the closest supported Ghostfolio range (mtd covers current month;
    # we fall back to ytd for the previous-month case and note the limitation).
    if date_range in {"1m", "last_month"}:
        date_from, date_to = _last_month_range()
        # Ghostfolio doesn't support arbitrary from/to ranges on the performance endpoint;
        # we use ytd as the closest proxy and annotate the response clearly.
        date_range = "ytd"
        note = (
            f"Note: Ghostfolio does not expose a 'previous month' performance endpoint. "
            f"Showing year-to-date performance as the closest available metric. "
            f"For previous month ({date_from} to {date_to}), check your Ghostfolio dashboard directly."
        )
    else:
        note = None
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

        result: dict[str, Any] = {
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
        if note:
            result["note"] = note
        return result

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
        date_range: Time period for performance. Options:
                    '1d'         — today
                    'wtd'        — week-to-date
                    'mtd'        — current month-to-date (from the 1st of this month)
                    '1m'         — previous calendar month (e.g. if today is March, returns February)
                    'ytd'        — year-to-date (default)
                    '1y'         — last 12 months
                    '5y'         — last 5 years
                    'max'        — all time
                    Use '1m' when the user says "last month". Use 'mtd' only when they
                    explicitly say "this month" or "month to date".

    Returns:
        Dictionary with performance metrics, absolute and relative changes.
        If '1m' was requested, includes a 'note' field explaining the data source limitation.
    """
    return await _get_performance(date_range)
