"""
Tool: get_portfolio_health_scorecard
Multi-step: calls holdings + performance + diversification, then synthesises an A-D grade.
Standout: First tool to produce a graded scorecard with specific named action items.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client


@tool
async def get_portfolio_health_scorecard() -> dict[str, Any]:
    """
    Produce a comprehensive A-D health scorecard for the user's entire portfolio.
    Combines holdings, performance, and diversification data into a single graded
    report with prioritised, specific action items.

    Use this when the user asks:
    - 'How healthy is my portfolio?'
    - 'Give me an overall assessment'
    - 'What grade would you give my portfolio?'
    - 'What should I fix first?'

    Returns:
        grade (A-D), score (0-100), performance_summary, diversification_summary,
        risk_flags[], action_items[] (prioritised), data_timestamp
    """
    return await _scorecard()


async def _scorecard() -> dict[str, Any]:
    try:
        client = get_shared_client()
        holdings_data = await client.get_portfolio_holdings()
        perf_data = await client.get_portfolio_performance("ytd")
        perf_1y = await client.get_portfolio_performance("1y")

        holdings = holdings_data.get("holdings", {})
        if not holdings:
            return {
                "status": "empty",
                "message": "No holdings found. Add transactions in Ghostfolio first.",
                "data_timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # ── Compute metrics ────────────────────────────────────────
        total_value = sum(h.get("value", 0) or 0 for h in holdings.values())
        num_positions = len(holdings)

        # Concentration: max single-position %
        max_alloc = (
            max(
                (h.get("value", 0) or 0) / total_value * 100
                for h in holdings.values()
            )
            if total_value > 0 else 0
        )

        # Sector & asset class breakdown
        sectors: dict[str, float] = {}
        asset_classes: dict[str, float] = {}
        for h in holdings.values():
            val = h.get("value", 0) or 0
            for s in h.get("sectors", []):
                sectors[s["name"]] = sectors.get(s["name"], 0) + val * s.get("weight", 1)
            ac = h.get("assetClass", "EQUITY")
            asset_classes[ac] = asset_classes.get(ac, 0) + val

        max_sector_pct = max((v / total_value * 100 for v in sectors.values()), default=0)

        ytd_return = (
            perf_data.get("performance", {}).get("ytd", {}).get("relativeChange", 0) * 100
        )
        one_y_return = (
            perf_1y.get("performance", {}).get("1y", {}).get("relativeChange", 0) * 100
        )

        # ── Scoring (0-100) ────────────────────────────────────────
        score = 100

        if num_positions < 5:
            score -= 25
        elif num_positions < 10:
            score -= 10

        if max_alloc > 40:
            score -= 20
        elif max_alloc > 25:
            score -= 10
        elif max_alloc > 20:
            score -= 5

        if max_sector_pct > 50:
            score -= 15
        elif max_sector_pct > 35:
            score -= 8

        # Only-equity penalty
        equity_pct = (
            asset_classes.get("EQUITY", 0) / total_value * 100 if total_value else 0
        )
        if equity_pct > 95:
            score -= 10

        score = max(0, min(100, score))
        grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"

        # ── Risk flags ─────────────────────────────────────────────
        flags = []
        if max_alloc > 20:
            top_sym = max(holdings.items(), key=lambda x: x[1].get("value", 0) or 0)[0]
            flags.append({
                "type": "CONCENTRATION",
                "severity": "HIGH" if max_alloc > 35 else "MEDIUM",
                "message": f"{top_sym} represents {max_alloc:.1f}% of portfolio",
            })
        if num_positions < 5:
            flags.append({
                "type": "UNDIVERSIFIED",
                "severity": "HIGH",
                "message": f"Only {num_positions} positions — very undiversified",
            })
        if max_sector_pct > 40:
            top_sector = max(sectors.items(), key=lambda x: x[1])[0]
            flags.append({
                "type": "SECTOR_CONCENTRATION",
                "severity": "MEDIUM",
                "message": f"{top_sector} sector = {max_sector_pct:.1f}% of portfolio",
            })
        if equity_pct > 95:
            flags.append({
                "type": "NO_BONDS",
                "severity": "LOW",
                "message": "100% equity — consider adding bonds for volatility buffer",
            })

        # ── Action items (prioritised) ─────────────────────────────
        actions = []
        if max_alloc > 20:
            actions.append({
                "priority": 1,
                "action": "Reduce largest position",
                "detail": "Trim the position above 20% threshold to reduce concentration risk",
            })
        if num_positions < 10:
            actions.append({
                "priority": 2,
                "action": "Add more positions",
                "detail": (
                    f"Increase from {num_positions} to at least 10 holdings "
                    "for meaningful diversification"
                ),
            })
        if max_sector_pct > 35:
            top_sector_name = max(sectors, key=lambda k: sectors[k])
            actions.append({
                "priority": 3,
                "action": "Rebalance sector exposure",
                "detail": f"Reduce {top_sector_name} sector below 30%",
            })
        if equity_pct > 95:
            actions.append({
                "priority": 4,
                "action": "Add bond allocation",
                "detail": "Consider 10-20% bonds/fixed income to reduce portfolio volatility",
            })
        if not actions:
            actions.append({
                "priority": 1,
                "action": "Maintain current allocation",
                "detail": "Portfolio is well-structured. Review quarterly.",
            })

        return {
            "status": "ok",
            "grade": grade,
            "score": score,
            "total_value": round(total_value, 2),
            "position_count": num_positions,
            "performance": {
                "ytd_pct": round(ytd_return, 2),
                "one_year_pct": round(one_y_return, 2),
            },
            "diversification": {
                "max_single_position_pct": round(max_alloc, 2),
                "max_sector_pct": round(max_sector_pct, 2),
                "top_sector": max(sectors, key=lambda k: sectors[k]) if sectors else "Unknown",
                "equity_pct": round(equity_pct, 2),
            },
            "risk_flags": flags,
            "action_items": sorted(actions, key=lambda x: x["priority"]),
            "data_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}
