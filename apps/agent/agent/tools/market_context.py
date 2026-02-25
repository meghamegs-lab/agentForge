"""
Tool: get_market_context_overlay
Multi-step: current holdings + benchmark regime + sector market data →
maps YOUR specific holdings to macro themes.
Standout: Generic macro insight applied to your actual portfolio — not generic advice.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client

# Sector sensitivity map: how each sector behaves in macro regimes
SECTOR_MACRO_SENSITIVITY: dict[str, dict[str, str]] = {
    "Technology": {
        "rising_rates": "NEGATIVE — growth stocks hurt by higher discount rates",
        "recession":    "MIXED — cloud/SaaS defensive; hardware/semis cyclical",
        "inflation":    "NEGATIVE — high valuations compressed by inflation",
        "bull_market":  "POSITIVE — leads in risk-on environments",
    },
    "Financial Services": {
        "rising_rates": "POSITIVE — net interest margin expands",
        "recession":    "NEGATIVE — loan defaults rise",
        "inflation":    "MIXED — can pass through but credit risk rises",
        "bull_market":  "POSITIVE — benefits from economic expansion",
    },
    "Healthcare": {
        "rising_rates": "NEUTRAL — defensive, less rate-sensitive",
        "recession":    "POSITIVE — defensive sector, demand inelastic",
        "inflation":    "MIXED — pricing power but cost pressures",
        "bull_market":  "NEUTRAL — lags in strong risk-on markets",
    },
    "Consumer Defensive": {
        "rising_rates": "NEUTRAL — dividend yield less attractive vs bonds",
        "recession":    "POSITIVE — consumer staples are recession-resistant",
        "inflation":    "MIXED — some pricing power, margin pressure",
        "bull_market":  "NEUTRAL/NEGATIVE — lags in bull markets",
    },
    "Energy": {
        "rising_rates": "NEUTRAL — commodity prices dominate",
        "recession":    "NEGATIVE — energy demand falls",
        "inflation":    "POSITIVE — energy IS inflation; commodity hedge",
        "bull_market":  "POSITIVE — demand rises with economic growth",
    },
    "Real Estate": {
        "rising_rates": "NEGATIVE — REITs hit hard by rate rises",
        "recession":    "NEGATIVE — commercial RE suffers",
        "inflation":    "POSITIVE — real assets hedge inflation long-term",
        "bull_market":  "NEUTRAL — depends on sub-sector",
    },
    "Industrials": {
        "rising_rates": "NEUTRAL — moderately sensitive",
        "recession":    "NEGATIVE — capital spending falls",
        "inflation":    "MIXED — can pass through costs if demand holds",
        "bull_market":  "POSITIVE — benefits from expansion",
    },
    "Communication Services": {
        "rising_rates": "NEGATIVE — high-duration growth names suffer",
        "recession":    "MIXED — advertising cyclical; telecom defensive",
        "inflation":    "NEUTRAL",
        "bull_market":  "POSITIVE",
    },
    "Consumer Cyclical": {
        "rising_rates": "NEGATIVE — consumer borrowing costs rise",
        "recession":    "NEGATIVE — discretionary spending falls first",
        "inflation":    "NEGATIVE — purchasing power erosion",
        "bull_market":  "POSITIVE — outperforms in strong economy",
    },
    "Basic Materials": {
        "rising_rates": "NEUTRAL",
        "recession":    "NEGATIVE — industrial demand falls",
        "inflation":    "POSITIVE — commodity prices rise",
        "bull_market":  "POSITIVE",
    },
    "Utilities": {
        "rising_rates": "NEGATIVE — bond proxy, hurt by rate rises",
        "recession":    "POSITIVE — defensive, regulated revenue",
        "inflation":    "MIXED — regulated pricing limits pass-through",
        "bull_market":  "NEUTRAL/NEGATIVE — lags risk-on",
    },
}

_VALID_THEMES = frozenset({"rising_rates", "recession", "inflation", "bull_market"})

_HEDGES_MAP: dict[str, list[str]] = {
    "rising_rates": ["SCHD (dividend value ETF)", "BRK.B (financials/value)", "SHV (short-term treasuries)"],
    "recession":    ["VHT (healthcare ETF)", "XLP (consumer staples)", "GLD (gold)"],
    "inflation":    ["TIPS (inflation-protected bonds)", "GLD (gold)", "VNQ (REITs)", "DJP (commodities)"],
    "bull_market":  ["QQQ (tech growth)", "IWM (small caps)", "VWO (emerging markets)"],
}

_SENTIMENT_SCORE: dict[str, int] = {"POSITIVE": 1, "NEUTRAL": 0, "MIXED": 0, "NEGATIVE": -1}


@tool
async def get_market_context_overlay(macro_theme: str = "rising_rates") -> dict[str, Any]:
    """
    Analyse how YOUR specific holdings would perform in a given macro environment.
    Maps each sector in your portfolio to its known sensitivity to the chosen theme.
    Gives portfolio-level vulnerability score, not generic advice.

    Use when users ask:
    - 'How is my portfolio positioned if rates keep rising?'
    - 'What happens to my holdings in a recession?'
    - 'Am I hedged against inflation?'
    - 'Is my portfolio set up for a bull/bear market?'

    Args:
        macro_theme: One of 'rising_rates', 'recession', 'inflation', 'bull_market'

    Returns:
        theme_sensitivity_score, holding_by_holding analysis, portfolio_vulnerability,
        most_exposed_positions[], best_positioned_positions[], suggested_hedges[]
    """
    return await _market_context(macro_theme)


async def _market_context(macro_theme: str) -> dict[str, Any]:
    if macro_theme not in _VALID_THEMES:
        macro_theme = "rising_rates"

    try:
        client = get_shared_client()
        data = await client.get_portfolio_holdings()

        holdings = data.get("holdings", {})
        if not holdings:
            return {"status": "empty", "message": "No holdings to analyse."}

        total_value = sum(h.get("value", 0) or 0 for h in holdings.values())

        # ── Analyse each position ──────────────────────────────────
        position_analyses = []
        sector_exposure: dict[str, float] = {}
        weighted_score = 0.0

        for sym, h in holdings.items():
            val = h.get("value", 0) or 0
            alloc = val / total_value * 100 if total_value > 0 else 0
            sectors = h.get("sectors", [])

            position_sentiment = []
            for s in sectors:
                sname = s.get("name", "Unknown")
                weight = s.get("weight", 1.0)
                macro_map = SECTOR_MACRO_SENSITIVITY.get(sname, {})
                sensitivity = macro_map.get(macro_theme, "NEUTRAL — sector not mapped")
                sentiment = sensitivity.split(" — ")[0].split("/")[0].strip()
                position_sentiment.append({
                    "sector": sname,
                    "sector_weight_in_holding": round(weight * 100, 1),
                    "sensitivity": sensitivity,
                    "sentiment": sentiment,
                })
                sector_exposure[sname] = sector_exposure.get(sname, 0) + val * weight

            # Dominant sentiment for this position
            if position_sentiment:
                sentiments = [ps["sentiment"] for ps in position_sentiment]
                neg_count = sentiments.count("NEGATIVE")
                pos_count = sentiments.count("POSITIVE")
                dominant = "NEGATIVE" if neg_count > pos_count else (
                    "POSITIVE" if pos_count > neg_count else "NEUTRAL"
                )
            else:
                dominant = "NEUTRAL"

            weighted_score += _SENTIMENT_SCORE.get(dominant, 0) * (alloc / 100)

            position_analyses.append({
                "symbol": sym,
                "name": h.get("name", sym),
                "allocation_pct": round(alloc, 2),
                "value": round(val, 2),
                "dominant_sentiment": dominant,
                "sector_analysis": position_sentiment,
            })

        # ── Portfolio-level score ──────────────────────────────────
        portfolio_score = round(weighted_score * 100, 1)  # -100 to +100

        if portfolio_score > 20:
            vulnerability = "WELL_POSITIONED"
            summary = f"Your portfolio is well-positioned for {macro_theme.replace('_', ' ')}."
        elif portfolio_score > -10:
            vulnerability = "NEUTRAL"
            summary = f"Your portfolio has mixed exposure to {macro_theme.replace('_', ' ')}."
        else:
            vulnerability = "VULNERABLE"
            summary = (
                f"Your portfolio has significant exposure to "
                f"{macro_theme.replace('_', ' ')} headwinds."
            )

        most_exposed = [p for p in position_analyses if p["dominant_sentiment"] == "NEGATIVE"]
        best_positioned = [p for p in position_analyses if p["dominant_sentiment"] == "POSITIVE"]

        return {
            "status": "ok",
            "macro_theme": macro_theme,
            "portfolio_score": portfolio_score,
            "vulnerability": vulnerability,
            "summary": summary,
            "position_analyses": sorted(
                position_analyses,
                key=lambda x: (x["dominant_sentiment"] == "NEGATIVE", -x["allocation_pct"]),
                reverse=True,
            ),
            "most_exposed_positions": [p["symbol"] for p in most_exposed[:3]],
            "best_positioned_positions": [p["symbol"] for p in best_positioned[:3]],
            "suggested_hedges": _HEDGES_MAP.get(macro_theme, []),
            "sector_exposure": {
                k: round(v / total_value * 100, 2)
                for k, v in sorted(
                    sector_exposure.items(), key=lambda x: x[1], reverse=True
                )
            },
            "low_confidence_note": (
                "This analysis uses historical sector patterns. "
                "Macro outcomes are uncertain — treat as directional guidance only."
            ),
            "data_timestamp": datetime.now(UTC).isoformat(),
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}
