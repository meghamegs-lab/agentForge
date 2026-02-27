# LangChain tool that computes exact buy/sell dollar amounts needed to reach target asset allocations.
"""
Tool: get_rebalancing_plan
Multi-step: holdings + market prices → computes exact buy/sell dollar amounts per position.
Standout: Returns specific dollar amounts to buy/sell — not just "you're overweight in X".
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client
from agent.clients.market import MarketDataClient

_market = MarketDataClient()


@tool
async def get_rebalancing_plan(
    target_us_equity_pct: float = 55.0,
    target_intl_equity_pct: float = 25.0,
    target_bonds_pct: float = 15.0,
    target_cash_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Calculate an exact rebalancing plan showing the specific dollar amount to buy
    or sell for each position to reach target allocations.

    Use when users ask:
    - 'How do I rebalance my portfolio?'
    - 'What exactly should I buy or sell to rebalance?'
    - 'I want 60/40 stocks/bonds — what do I need to change?'
    - 'How much of X should I sell?'

    Args:
        target_us_equity_pct:   Target % for US equity holdings (default 55%)
        target_intl_equity_pct: Target % for international equity (default 25%)
        target_bonds_pct:       Target % for bonds/fixed income (default 15%)
        target_cash_pct:        Target % for cash/alternatives (default 5%)

    Returns:
        Current vs target allocation, rebalancing trades[] with exact dollar amounts,
        estimated total trades value, tax notes.
    """
    return await _rebalancing_plan(
        target_us_equity_pct / 100,
        target_intl_equity_pct / 100,
        target_bonds_pct / 100,
        target_cash_pct / 100,
    )


# Core logic: classifies holdings into asset buckets, computes deltas, and generates per-position trades.
async def _rebalancing_plan(
    tgt_us: float,
    tgt_intl: float,
    tgt_bonds: float,
    tgt_cash: float,
) -> dict[str, Any]:
    try:
        client = get_shared_client()
        data = await client.get_portfolio_holdings()

        # Normalise: Ghostfolio can return holdings as a list OR a dict keyed by symbol
        raw = data.get("holdings", {})
        if isinstance(raw, list):
            holdings = {h.get("symbol", f"pos_{i}"): h for i, h in enumerate(raw)}
        else:
            holdings = raw or {}

        if not holdings:
            return {"status": "empty", "message": "No holdings to rebalance."}

        total_value = sum(h.get("value", 0) or 0 for h in holdings.values())
        if total_value == 0:
            return {"status": "empty", "message": "Portfolio has no value."}

        # ── Classify each holding ──────────────────────────────────
        # Maps a single holding to US_EQUITY, INTL_EQUITY, BONDS, or CASH based on asset class and country.
        def classify(h: dict) -> str:
            ac = h.get("assetClass", "EQUITY")
            countries = [c.get("name", "") for c in h.get("countries", [])]
            if ac in ("BOND", "FIXED_INCOME"):
                return "BONDS"
            if ac == "CASH":
                return "CASH"
            if any(c == "United States" for c in countries):
                return "US_EQUITY"
            return "INTL_EQUITY"

        # Current bucket values
        buckets: dict[str, float] = {"US_EQUITY": 0, "INTL_EQUITY": 0, "BONDS": 0, "CASH": 0}
        position_buckets: dict[str, str] = {}

        for sym, h in holdings.items():
            bucket = classify(h)
            val = h.get("value", 0) or 0
            buckets[bucket] = buckets.get(bucket, 0) + val
            position_buckets[sym] = bucket

        # ── Target vs current ──────────────────────────────────────
        targets = {
            "US_EQUITY":   tgt_us * total_value,
            "INTL_EQUITY": tgt_intl * total_value,
            "BONDS":       tgt_bonds * total_value,
            "CASH":        tgt_cash * total_value,
        }
        deltas = {b: targets[b] - buckets.get(b, 0) for b in targets}

        # ── Per-position trades ────────────────────────────────────
        trades = []
        for sym, h in sorted(
            holdings.items(),
            key=lambda x: x[1].get("value", 0) or 0,
            reverse=True,
        ):
            val = h.get("value", 0) or 0
            bucket = position_buckets[sym]
            alloc_pct = val / total_value * 100
            delta = deltas.get(bucket, 0)

            # Each position contributes proportionally to its bucket's delta
            bucket_total = buckets.get(bucket, 0)
            trade_amount = (delta * val / bucket_total) if bucket_total > 0 else 0

            if abs(trade_amount) < 50:   # ignore tiny trades
                continue

            # Fetch current price for share count estimate
            price_data = await _market.get_quote(sym)
            price = price_data.get("current_price") if price_data.get("status") == "ok" else None
            shares_estimate = round(abs(trade_amount) / price, 2) if price else None

            trades.append({
                "symbol":          sym,
                "name":            h.get("name", sym),
                "bucket":          bucket,
                "current_value":   round(val, 2),
                "current_pct":     round(alloc_pct, 2),
                "action":          "SELL" if trade_amount < 0 else "BUY",
                "dollar_amount":   round(abs(trade_amount), 2),
                "shares_estimate": shares_estimate,
                "current_price":   price,
                "note": (
                    "Consider tax-loss harvest if at a loss"
                    if trade_amount < 0
                    else "Add via new contribution if possible to avoid capital gains"
                ),
            })

        trades.sort(key=lambda x: x["dollar_amount"], reverse=True)
        total_trade_value = sum(t["dollar_amount"] for t in trades)

        return {
            "status": "ok",
            "total_value": round(total_value, 2),
            "current_allocation": {
                b: {"value": round(v, 2), "pct": round(v / total_value * 100, 2)}
                for b, v in buckets.items()
            },
            "target_allocation": {
                "US_EQUITY":   round(tgt_us * 100, 1),
                "INTL_EQUITY": round(tgt_intl * 100, 1),
                "BONDS":       round(tgt_bonds * 100, 1),
                "CASH":        round(tgt_cash * 100, 1),
            },
            "trades": trades,
            "summary": {
                "total_sells": round(
                    sum(t["dollar_amount"] for t in trades if t["action"] == "SELL"), 2
                ),
                "total_buys": round(
                    sum(t["dollar_amount"] for t in trades if t["action"] == "BUY"), 2
                ),
                "total_trade_value": round(total_trade_value, 2),
                "trade_count": len(trades),
            },
            "tax_note": (
                "Selling appreciated positions triggers capital gains tax. "
                "Consider rebalancing by directing new contributions to underweight buckets first."
            ),
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Ghostfolio (holdings) + Yahoo Finance (via yfinance) for current prices",
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}
