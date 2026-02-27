# LangChain tool that calculates how much trading fees have cost as a percentage of total gross returns.
"""
Tool: get_fee_drag_analysis
Multi-step: transactions (all fees) + performance (gross returns) → computes fee drag %.
Standout: Expresses fees as % of total returns — no retail tool surfaces this framing.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client


@tool
async def get_fee_drag_analysis(date_range: str = "max") -> dict[str, Any]:
    """
    Analyse how much trading fees have cost you as a percentage of total portfolio returns.
    Fee drag = total fees paid / total absolute gain — a metric no standard portfolio
    tracker surfaces. Also breaks fees down by symbol to identify most expensive holdings.

    Use when users ask:
    - 'How much are fees costing me?'
    - 'Are my fees worth it?'
    - 'What percentage of my returns went to fees?'
    - 'Which positions are most expensive to hold?'

    Args:
        date_range: 'ytd', '1y', '5y', or 'max' (default 'max' for lifetime view)

    Returns:
        total_fees_paid, gross_return, net_return, fee_drag_pct,
        fee_drag_explanation, fee_by_symbol[], annualised_fee_rate
    """
    return await _fee_drag(date_range)


# Core logic: aggregates all transaction fees and computes fee drag % relative to gross returns.
async def _fee_drag(date_range: str = "max") -> dict[str, Any]:
    try:
        client = get_shared_client()
        orders_data = await client.get_orders()
        perf_data = await client.get_portfolio_performance(date_range)
        holdings_data = await client.get_portfolio_holdings()

        activities = orders_data.get("activities", [])
        # Normalise: Ghostfolio can return holdings as a list OR a dict keyed by symbol
        raw_holdings = holdings_data.get("holdings", {})
        if isinstance(raw_holdings, list):
            raw_holdings = {h.get("symbol", f"pos_{i}"): h for i, h in enumerate(raw_holdings)}
        total_value = sum(
            h.get("value", 0) or 0
            for h in raw_holdings.values()
        )

        # ── Fee aggregation ────────────────────────────────────────
        total_fees = 0.0
        fee_by_symbol: dict[str, float] = {}
        fee_by_year: dict[str, float] = {}

        for act in activities:
            fee = act.get("fee", 0) or 0
            sym = act.get("SymbolProfile", {}).get("symbol", "UNKNOWN")
            date_str = act.get("date", "")
            year = date_str[:4] if date_str else "Unknown"

            total_fees += fee
            fee_by_symbol[sym] = fee_by_symbol.get(sym, 0) + fee
            fee_by_year[year] = fee_by_year.get(year, 0) + fee

        # ── Performance data ───────────────────────────────────────
        perf = perf_data.get("performance", {}).get(date_range, {})
        abs_gain = perf.get("absoluteChange", 0) or 0
        rel_change = perf.get("relativeChange", 0) or 0
        current_val = perf.get("currentValue", total_value) or total_value

        # ── Derived metrics ────────────────────────────────────────
        gross_gain = abs_gain + total_fees        # what gain would be without fees
        fee_drag_pct = (total_fees / gross_gain * 100) if gross_gain > 0 else 0

        # Annualised fee rate (fees / avg portfolio value)
        years_of_data = max(len(fee_by_year), 1)
        avg_portfolio = current_val  # simplified; good enough for estimate
        annual_fee_rate = (
            (total_fees / years_of_data / avg_portfolio * 100)
            if avg_portfolio > 0 else 0
        )

        # Fee by symbol sorted
        fee_symbols = [
            {
                "symbol": sym,
                "total_fees": round(v, 2),
                "pct_of_all_fees": round(v / total_fees * 100, 2) if total_fees > 0 else 0,
            }
            for sym, v in sorted(fee_by_symbol.items(), key=lambda x: x[1], reverse=True)
            if v > 0
        ]

        # Human-readable verdict
        if fee_drag_pct < 1:
            verdict = "Low fee drag — your cost structure is lean."
        elif fee_drag_pct < 5:
            verdict = (
                "Moderate fee drag — comparable to an active fund. "
                "Consider low-cost ETFs for new positions."
            )
        elif fee_drag_pct < 15:
            verdict = (
                "High fee drag — fees are meaningfully reducing your returns. "
                "Review your broker's fee structure."
            )
        else:
            verdict = (
                "Very high fee drag — fees have consumed a large share of your returns. "
                "Immediate review recommended."
            )

        return {
            "status": "ok",
            "date_range": date_range,
            "total_fees_paid": round(total_fees, 2),
            "gross_gain_without_fees": round(gross_gain, 2),
            "net_gain_after_fees": round(abs_gain, 2),
            "fee_drag_pct": round(fee_drag_pct, 2),
            "fee_drag_explanation": (
                f"You paid ${total_fees:,.2f} in fees. That represents "
                f"{fee_drag_pct:.1f}% of your ${gross_gain:,.2f} gross gains."
            ),
            "annual_fee_rate_pct": round(annual_fee_rate, 3),
            "net_return_pct": round(rel_change * 100, 2),
            "gross_return_pct": (
                round((gross_gain / (current_val - gross_gain)) * 100, 2)
                if (current_val - gross_gain) > 0 else 0
            ),
            "verdict": verdict,
            "fee_by_symbol": fee_symbols[:10],
            "fee_by_year": {k: round(v, 2) for k, v in sorted(fee_by_year.items())},
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Ghostfolio",
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}
