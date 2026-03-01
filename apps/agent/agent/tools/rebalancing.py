# LangChain tool that computes exact buy/sell dollar amounts needed to reach target asset allocations.
"""
Tool: get_rebalancing_plan
Multi-step: holdings + market prices → computes exact buy/sell dollar amounts per position.
Standout: Returns specific dollar amounts to buy/sell — not just "you're overweight in X".
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client, normalize_holdings
from agent.clients.market import get_shared_market_client

_market = get_shared_market_client()


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
        target_us_equity_pct:   Target % for US equity holdings (0–100, default 55)
        target_intl_equity_pct: Target % for international equity (0–100, default 25)
        target_bonds_pct:       Target % for bonds/fixed income (0–100, default 15)
        target_cash_pct:        Target % for cash/alternatives (0–100, default 5)

    Returns:
        Current vs target allocation, rebalancing trades[] with exact dollar amounts,
        estimated total trades value, tax notes.
    """
    return await _rebalancing_plan(
        target_us_equity_pct,
        target_intl_equity_pct,
        target_bonds_pct,
        target_cash_pct,
    )


# Core logic: classifies holdings into asset buckets, computes deltas, and generates per-position trades.
# All tgt_* parameters are on the 0–100 percentage scale; division by 100 happens internally.
async def _rebalancing_plan(
    tgt_us: float,
    tgt_intl: float,
    tgt_bonds: float,
    tgt_cash: float,
) -> dict[str, Any]:
    try:
        client = get_shared_client()
        data = await client.get_portfolio_holdings()

        holdings = normalize_holdings(data)

        if not holdings:
            return {"status": "empty", "message": "No holdings to rebalance."}

        total_value = sum(
            h.get("valueInBaseCurrency", h.get("value", 0)) or 0 for h in holdings.values()
        )
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
            val = h.get("valueInBaseCurrency", h.get("value", 0)) or 0
            buckets[bucket] = buckets.get(bucket, 0) + val
            position_buckets[sym] = bucket

        # ── Target vs current ──────────────────────────────────────
        # tgt_* are on the 0–100 scale — divide by 100 once here.
        targets = {
            "US_EQUITY": (tgt_us / 100) * total_value,
            "INTL_EQUITY": (tgt_intl / 100) * total_value,
            "BONDS": (tgt_bonds / 100) * total_value,
            "CASH": (tgt_cash / 100) * total_value,
        }
        deltas = {b: targets[b] - buckets.get(b, 0) for b in targets}

        # ── Pre-compute trade amounts to decide which symbols need prices ─────
        candidate_trades: list[dict[str, Any]] = []
        for sym, h in sorted(
            holdings.items(),
            key=lambda x: x[1].get("valueInBaseCurrency", x[1].get("value", 0)) or 0,
            reverse=True,
        ):
            val = h.get("valueInBaseCurrency", h.get("value", 0)) or 0
            bucket = position_buckets[sym]
            alloc_pct = val / total_value * 100
            delta = deltas.get(bucket, 0)
            bucket_total = buckets.get(bucket, 0)
            trade_amount = (delta * val / bucket_total) if bucket_total > 0 else 0

            if abs(trade_amount) < 50:  # ignore tiny trades
                continue

            candidate_trades.append(
                {
                    "symbol": sym,
                    "name": h.get("name", sym),
                    "bucket": bucket,
                    "current_value": round(val, 2),
                    "current_pct": round(alloc_pct, 2),
                    "trade_amount": trade_amount,
                }
            )

        # ── Batch-fetch all prices in parallel (N× speedup vs sequential loop) ─
        trade_symbols = [t["symbol"] for t in candidate_trades]
        prices: dict[str, float | None] = {}
        if trade_symbols:
            batch_result = await _market.get_batch_quotes(trade_symbols)
            if batch_result.get("status") == "ok":
                # get_batch_quotes returns {"status": "ok", "quotes": {sym: quote_dict, ...}}
                quote_data = batch_result.get("quotes", {})
                for sym in trade_symbols:
                    q = quote_data.get(sym, {})
                    prices[sym] = q.get("current_price") if q.get("status") == "ok" else None
            else:
                # Fallback: fetch individually in parallel if batch fails
                individual_results = await asyncio.gather(
                    *[_market.get_quote(sym) for sym in trade_symbols],
                    return_exceptions=True,
                )
                for sym, res in zip(trade_symbols, individual_results, strict=False):
                    if isinstance(res, Exception) or not isinstance(res, dict):
                        prices[sym] = None
                    else:
                        prices[sym] = (
                            res.get("current_price") if res.get("status") == "ok" else None
                        )

        # ── Build trades list using pre-fetched prices ─────────────────────────
        trades: list[dict[str, Any]] = []
        for t in candidate_trades:
            sym = t["symbol"]
            trade_amount = t["trade_amount"]
            price = prices.get(sym)
            shares_estimate = round(abs(trade_amount) / price, 2) if price else None

            trades.append(
                {
                    "symbol": sym,
                    "name": t["name"],
                    "bucket": t["bucket"],
                    "current_value": t["current_value"],
                    "current_pct": t["current_pct"],
                    "action": "SELL" if trade_amount < 0 else "BUY",
                    "dollar_amount": round(abs(trade_amount), 2),
                    "shares_estimate": shares_estimate,
                    "current_price": price,
                    "note": (
                        "Consider tax-loss harvest if at a loss"
                        if trade_amount < 0
                        else "Add via new contribution if possible to avoid capital gains"
                    ),
                }
            )

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
                # tgt_* already in 0–100 scale — display directly
                "US_EQUITY": round(tgt_us, 1),
                "INTL_EQUITY": round(tgt_intl, 1),
                "BONDS": round(tgt_bonds, 1),
                "CASH": round(tgt_cash, 1),
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
