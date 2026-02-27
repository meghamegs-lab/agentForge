# LangChain tool that computes sector/geography/asset-class breakdown and flags concentration risk.
"""
Tool: analyze_diversification
Computes sector, geography, asset class breakdown and risk flags.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client
from agent.config import settings

# ── Implementation (importable in unit tests without @tool overhead) ──────────

# Core logic: aggregates sector/geography/asset-class weights and computes a 0-100 diversification score.
async def _analyze_diversification() -> dict[str, Any]:
    """
    Core logic for analyze_diversification.
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
                "message": "No holdings to analyze.",
                "data_timestamp": datetime.now(UTC).isoformat(),
            }

        total_value = sum(h.get("value", 0) or 0 for h in holdings_list)
        if total_value == 0:
            return {
                "status": "empty",
                "message": "Portfolio has no value.",
                "data_timestamp": datetime.now(UTC).isoformat(),
            }

        sectors: dict[str, float] = {}
        countries: dict[str, float] = {}
        asset_classes: dict[str, float] = {}
        concentration_flags = []

        for holding in holdings_list:
            symbol = holding.get("symbol", "UNKNOWN")
            value = holding.get("value", 0) or 0
            allocation = value / total_value

            # Flag individual concentration
            if allocation >= settings.portfolio_concentration_threshold:
                concentration_flags.append({
                    "symbol": symbol,
                    "name": holding.get("name", symbol),
                    "allocation_percent": round(allocation * 100, 2),
                    "severity": "HIGH" if allocation >= 0.35 else "MEDIUM",
                    "message": (
                        f"{symbol} represents {round(allocation * 100, 1)}% of portfolio "
                        "— consider diversifying"
                    ),
                })

            # Aggregate sectors
            for sector_obj in holding.get("sectors", []):
                sector_name = sector_obj.get("name", "Unknown")
                sector_weight = sector_obj.get("weight", 1.0)
                sectors[sector_name] = sectors.get(sector_name, 0) + (value * sector_weight)

            # Aggregate countries
            for country_obj in holding.get("countries", []):
                country_name = country_obj.get("name", "Unknown")
                country_weight = country_obj.get("weight", 1.0)
                countries[country_name] = countries.get(country_name, 0) + (value * country_weight)

            # Aggregate asset classes
            asset_class = holding.get("assetClass", "EQUITY")
            asset_classes[asset_class] = asset_classes.get(asset_class, 0) + value

        def to_pct_list(d: dict[str, float]) -> list[dict]:
            items = [
                {"name": k, "value": round(v, 2), "percent": round(v / total_value * 100, 2)}
                for k, v in d.items()
            ]
            return sorted(items, key=lambda x: x["percent"], reverse=True)

        # Diversification score (0-100): penalise concentration and few positions
        max_position_pct = max(
            (h.get("value", 0) or 0) / total_value * 100
            for h in holdings_list
        )
        num_positions = len(holdings_list)
        score = max(0, min(100, int(
            100
            - max(0, max_position_pct - 10) * 2     # penalise >10% positions
            - max(0, 10 - num_positions) * 3          # penalise fewer than 10 positions
        )))

        return {
            "status": "ok",
            "total_positions": num_positions,
            "total_value": round(total_value, 2),
            "diversification_score": score,
            "diversification_grade": (
                "A" if score >= 80 else
                "B" if score >= 65 else
                "C" if score >= 50 else "D"
            ),
            "sector_breakdown": to_pct_list(sectors),
            "geographic_breakdown": to_pct_list(countries),
            "asset_class_breakdown": to_pct_list(asset_classes),
            "concentration_flags": concentration_flags,
            "risk_summary": {
                "flag_count": len(concentration_flags),
                "highest_concentration": round(max_position_pct, 2),
                "needs_rebalancing": len(concentration_flags) > 0,
            },
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Ghostfolio",
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── LangChain Tool (used by the graph) ────────────────────────────────────────

# LangChain @tool wrapper — delegates to _analyze_diversification; used by the agent graph.
@tool
async def analyze_diversification() -> dict[str, Any]:
    """
    Analyze portfolio diversification across sectors, geographies, and asset classes.
    Automatically flags concentration risk when any single position exceeds the
    threshold (default 20%). Use this when users ask about diversification,
    concentration risk, sector exposure, geographic allocation, or rebalancing needs.

    Returns:
        Dictionary with sector breakdown, geographic breakdown, asset class breakdown,
        risk flags, and a diversification score.
    """
    return await _analyze_diversification()
