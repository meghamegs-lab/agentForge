# LangChain tool that fetches current holdings, allocation percentages, and total value from Ghostfolio.
"""
Tool: get_portfolio_summary
Returns current holdings, allocation percentages, and total portfolio value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client

# ── Implementation (importable in unit tests without @tool overhead) ──────────


# Core logic: normalises the Ghostfolio holdings payload and computes per-position allocation %.
async def _get_portfolio_summary(account_id: str = "") -> dict[str, Any]:
    """
    Core logic for get_portfolio_summary.
    Separated from the @tool wrapper so unit tests can call it directly.
    """
    try:
        client = get_shared_client()
        data = await client.get_portfolio_holdings()

        raw = data.get("holdings", [])

        # Ghostfolio can return holdings as either a list or a dict keyed by symbol
        holdings_list = list(raw.values()) if isinstance(raw, dict) else raw

        if not holdings_list:
            return {
                "status": "empty",
                "message": "No holdings found. Add transactions in Ghostfolio to see your portfolio.",
                "total_value": 0.0,
                "currency": "USD",
                "holdings": [],
                "data_timestamp": datetime.now(UTC).isoformat(),
            }

        processed = []
        total_value = 0.0

        for holding in holdings_list:
            symbol = holding.get("symbol", "UNKNOWN")
            value = holding.get("valueInBaseCurrency", holding.get("value", 0)) or 0
            total_value += value
            processed.append(
                {
                    "symbol": symbol,
                    "name": holding.get("name", symbol),
                    "quantity": holding.get("quantity", 0),
                    "current_value": value,
                    "currency": holding.get("currency", "USD"),
                    "asset_class": holding.get("assetClass", "EQUITY"),
                    "asset_sub_class": holding.get("assetSubClass", ""),
                    "sectors": holding.get("sectors", []),
                    "countries": holding.get("countries", []),
                }
            )

        # Calculate allocation percentages
        for h in processed:
            h["allocation_percent"] = (
                round(h["current_value"] / total_value * 100, 2) if total_value > 0 else 0.0
            )

        # Sort by value descending
        processed.sort(key=lambda x: x["current_value"], reverse=True)

        return {
            "status": "ok",
            "total_value": round(total_value, 2),
            "currency": "USD",
            "position_count": len(processed),
            "holdings": processed,
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Ghostfolio",
        }

    except GhostfolioError as e:
        return {
            "status": "error",
            "error": e.message,
            "error_code": e.status_code,
            "holdings": [],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "holdings": [],
        }


# ── LangChain Tool (used by the graph) ────────────────────────────────────────


# LangChain @tool wrapper — delegates to _get_portfolio_summary; used by the agent graph.
@tool
async def get_portfolio_summary(account_id: str = "") -> dict[str, Any]:
    """
    Retrieve a summary of the user's investment portfolio including all holdings,
    their current values, allocation percentages, and total portfolio value.
    Use this tool when the user asks about their portfolio composition, what they own,
    their current positions, or their overall portfolio value.

    Args:
        account_id: Optional account ID to filter by specific account. Leave empty for all accounts.

    Returns:
        Dictionary with holdings list, total value, currency, and data timestamp.
    """
    return await _get_portfolio_summary(account_id)
