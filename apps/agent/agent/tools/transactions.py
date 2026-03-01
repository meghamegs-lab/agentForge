# LangChain tool that retrieves transaction history with fee analysis and type categorisation.
"""
Tool: get_transactions
Returns transaction history with fee analysis and categorization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client

# ── Implementation (importable in unit tests without @tool overhead) ──────────


# Core logic: fetches Ghostfolio orders, applies optional filters, and summarises total fees by type.
async def _get_transactions(
    account_id: str = "",
    date_from: str = "",
    date_to: str = "",
    transaction_type: str = "",
) -> dict[str, Any]:
    """
    Core logic for get_transactions.
    Separated from the @tool wrapper so unit tests can call it directly.
    """
    try:
        client = get_shared_client()
        data = await client.get_orders(
            account_id=account_id or None,
            date_from=date_from or None,
            date_to=date_to or None,
        )

        # Guard against None/empty response from Ghostfolio (e.g. 200 with null body)
        if not data:
            return {
                "status": "empty",
                "message": "No transaction data returned from Ghostfolio.",
                "transactions": [],
                "summary": {},
                "data_timestamp": datetime.now(UTC).isoformat(),
            }

        activities = data.get("activities", [])

        if not activities:
            return {
                "status": "empty",
                "message": "No transactions found for the specified filters.",
                "transactions": [],
                "summary": {},
                "data_timestamp": datetime.now(UTC).isoformat(),
            }

        # Filter by type if requested
        if transaction_type:
            activities = [
                a for a in activities if a.get("type", "").upper() == transaction_type.upper()
            ]

        transactions = []
        total_fees = 0.0
        type_counts: dict[str, int] = {}
        type_values: dict[str, float] = {}

        for activity in activities:
            tx_type = activity.get("type", "UNKNOWN")
            fee = activity.get("fee", 0) or 0
            quantity = activity.get("quantity", 0) or 0
            unit_price = activity.get("unitPrice", 0) or 0
            total_value = quantity * unit_price

            total_fees += fee
            type_counts[tx_type] = type_counts.get(tx_type, 0) + 1
            type_values[tx_type] = type_values.get(tx_type, 0) + total_value

            symbol_profile = activity.get("SymbolProfile") or {}
            account_info = activity.get("account") or {}
            transactions.append(
                {
                    "id": activity.get("id", ""),
                    "date": activity.get("date", ""),
                    "type": tx_type,
                    "symbol": symbol_profile.get("symbol", ""),
                    "name": symbol_profile.get("name", ""),
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "total_value": round(total_value, 2),
                    "fee": fee,
                    "currency": activity.get("currency", "USD"),
                    "account": account_info.get("name", ""),
                }
            )

        # Sort by date descending (newest first)
        transactions.sort(key=lambda x: x["date"], reverse=True)

        return {
            "status": "ok",
            "transaction_count": len(transactions),
            "transactions": transactions,
            "summary": {
                "total_fees_paid": round(total_fees, 2),
                "by_type": {
                    t: {"count": type_counts[t], "total_value": round(type_values[t], 2)}
                    for t in type_counts
                },
            },
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Ghostfolio",
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message, "error_code": e.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── LangChain Tool (used by the graph) ────────────────────────────────────────


# LangChain @tool wrapper — delegates to _get_transactions; used by the agent graph.
@tool
async def get_transactions(
    account_id: str = "",
    date_from: str = "",
    date_to: str = "",
    transaction_type: str = "",
) -> dict[str, Any]:
    """
    Retrieve transaction history including buys, sells, dividends, and fees.
    Use this when users ask about their trading history, past transactions,
    fees paid, dividends received, or want to analyze their trading patterns.

    Args:
        account_id: Optional account ID filter. Leave empty for all accounts.
        date_from: Optional start date filter in ISO format (e.g. '2024-01-01').
        date_to: Optional end date filter in ISO format (e.g. '2024-12-31').
        transaction_type: Optional type filter: 'BUY', 'SELL', 'DIVIDEND', 'FEE', 'INTEREST'.

    Returns:
        Dictionary with transactions list, fee summary, and type breakdown.
    """
    return await _get_transactions(account_id, date_from, date_to, transaction_type)
