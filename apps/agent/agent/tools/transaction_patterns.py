"""
Tool: get_transaction_pattern_intelligence
Multi-step: all transactions + market data for each symbol →
detects behavioural patterns like performance-chasing, panic selling, DCA consistency.
Standout: Behavioural coaching from your OWN trade history — no other tool does this.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client
from agent.clients.market import MarketDataClient

_market = MarketDataClient()


@tool
async def get_transaction_pattern_intelligence() -> dict[str, Any]:
    """
    Analyse YOUR trading history to identify behavioural patterns that help or hurt returns.
    Detects: performance-chasing, panic selling, DCA consistency, holding period tendencies,
    fee-heavy trading, and best/worst trade analysis.

    Use when users ask:
    - 'Am I a good investor?'
    - 'What are my trading patterns?'
    - 'Do I tend to buy high and sell low?'
    - 'What were my best and worst trades?'
    - 'Am I a buy-and-hold investor or do I trade too much?'

    Returns:
        trading_patterns[], best_trades[], worst_trades[], behavioural_insights[],
        dca_score, churn_rate, coaching_summary
    """
    return await _transaction_patterns()


async def _transaction_patterns() -> dict[str, Any]:
    try:
        client = get_shared_client()
        orders_data = await client.get_orders()
        holdings_data = await client.get_portfolio_holdings()

        activities = orders_data.get("activities", [])
        if not activities:
            return {"status": "empty", "message": "No transactions to analyse."}

        holdings = holdings_data.get("holdings", {})

        buys = [a for a in activities if a.get("type") == "BUY"]
        sells = [a for a in activities if a.get("type") == "SELL"]

        # ── Churn rate ─────────────────────────────────────────────
        months_active = max(1, len({a.get("date", "")[:7] for a in activities}))
        trades_per_month = len(buys + sells) / months_active

        if trades_per_month > 4:
            churn_label = (
                "HIGH — you trade very frequently. "
                "High turnover increases fees and tax drag."
            )
        elif trades_per_month > 1:
            churn_label = "MODERATE — you make regular adjustments."
        else:
            churn_label = (
                "LOW — you are a buy-and-hold investor. "
                "This generally benefits long-term returns."
            )

        # ── DCA consistency ────────────────────────────────────────
        buy_months = sorted({b.get("date", "")[:7] for b in buys if b.get("date")})
        if len(buy_months) >= 2:
            first = buy_months[0]
            last = buy_months[-1]
            from_year, from_month = int(first[:4]), int(first[5:7])
            to_year, to_month = int(last[:4]), int(last[5:7])
            possible_months = (to_year - from_year) * 12 + (to_month - from_month) + 1
            dca_score = (
                round(len(buy_months) / possible_months * 100, 1)
                if possible_months > 0 else 0
            )
            dca_label = (
                "Excellent DCA discipline" if dca_score >= 75
                else "Good DCA consistency" if dca_score >= 50
                else "Irregular contributions — consider a fixed monthly schedule"
            )
        else:
            dca_score = 0.0
            dca_label = "Insufficient data for DCA analysis"

        # ── Per-symbol P&L analysis ────────────────────────────────
        symbol_trades: dict[str, list[dict]] = {}
        for act in activities:
            sym = act.get("SymbolProfile", {}).get("symbol", "UNKNOWN")
            symbol_trades.setdefault(sym, []).append(act)

        trade_results = []
        for sym, trades in symbol_trades.items():
            sym_buys = [t for t in trades if t.get("type") == "BUY"]
            if not sym_buys:
                continue

            total_qty = sum(t.get("quantity", 0) for t in sym_buys)
            avg_buy_price = (
                sum(t.get("unitPrice", 0) * t.get("quantity", 0) for t in sym_buys) / total_qty
                if total_qty > 0 else 0
            )

            price_data = _market.get_quote(sym)
            current_price = (
                price_data.get("current_price")
                if price_data.get("status") == "ok" else None
            )

            unrealized_pct = (
                (current_price - avg_buy_price) / avg_buy_price * 100
                if (current_price and avg_buy_price) else None
            )

            total_fees = sum(t.get("fee", 0) or 0 for t in trades)
            total_invested = sum(
                t.get("unitPrice", 0) * t.get("quantity", 0) for t in sym_buys
            )

            trade_results.append({
                "symbol": sym,
                "name": sym_buys[0].get("SymbolProfile", {}).get("name", sym),
                "avg_buy_price": round(avg_buy_price, 2),
                "current_price": current_price,
                "unrealized_gain_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
                "total_invested": round(total_invested, 2),
                "total_fees": round(total_fees, 2),
                "trade_count": len(trades),
                "still_holding": sym in holdings,
                "first_buy_date": min(t.get("date", "") for t in sym_buys) if sym_buys else None,
            })

        trade_results.sort(
            key=lambda x: x.get("unrealized_gain_pct") or 0, reverse=True
        )

        best_trades = [t for t in trade_results if (t.get("unrealized_gain_pct") or 0) > 0][:3]
        worst_trades = [t for t in trade_results if (t.get("unrealized_gain_pct") or 0) < 0][:3]

        # ── Behavioural patterns ───────────────────────────────────
        patterns = []

        if len(sells) >= 2:
            patterns.append({
                "pattern": "SELL_FREQUENCY",
                "observation": f"You have made {len(sells)} sell transactions vs {len(buys)} buys.",
                "insight": (
                    "Frequent selling often underperforms buy-and-hold strategies due to "
                    "transaction costs and market timing difficulty."
                    if len(sells) > len(buys) * 0.5
                    else "Low sell frequency suggests a buy-and-hold approach."
                ),
            })

        if dca_score < 50 and len(buys) >= 4:
            patterns.append({
                "pattern": "IRREGULAR_CONTRIBUTIONS",
                "observation": f"DCA score: {dca_score}% — contributions are irregular.",
                "insight": (
                    "Irregular investing often means buying more after good market periods "
                    "(performance-chasing). A fixed monthly schedule removes this bias."
                ),
            })

        high_fee_positions = [t for t in trade_results if t["total_fees"] > 20]
        if high_fee_positions:
            patterns.append({
                "pattern": "FEE_DRAG",
                "observation": f"{len(high_fee_positions)} positions with >$20 total fees.",
                "insight": "High-fee positions may indicate active trading. Consider low-cost ETFs.",
            })

        multi_buy_symbols = [
            s for s, t in symbol_trades.items()
            if len([x for x in t if x.get("type") == "BUY"]) >= 3
        ]
        if multi_buy_symbols:
            patterns.append({
                "pattern": "AVERAGING_DOWN_OR_DCA",
                "observation": f"3+ buy transactions in: {', '.join(multi_buy_symbols[:3])}",
                "insight": (
                    "Multiple buys in the same symbol can be disciplined DCA or averaging down "
                    "into a loser — check performance."
                ),
            })

        # ── Coaching summary ───────────────────────────────────────
        if dca_score >= 75 and trades_per_month <= 1 and len(worst_trades) < len(best_trades):
            coaching = (
                "Strong investor behaviour. "
                "Consistent contributions, low churn, more winners than losers."
            )
        elif trades_per_month > 3:
            coaching = (
                "High trading frequency is your biggest risk. "
                "Studies show most active traders underperform index funds."
            )
        elif dca_score < 40:
            coaching = (
                "Irregular contribution timing is a risk. "
                "Set a fixed monthly investment schedule to remove timing bias."
            )
        else:
            coaching = (
                "Solid fundamentals. "
                "Focus on contribution consistency and minimising transaction fees."
            )

        return {
            "status": "ok",
            "total_buy_transactions": len(buys),
            "total_sell_transactions": len(sells),
            "trades_per_month": round(trades_per_month, 2),
            "churn_assessment": churn_label,
            "dca_score": dca_score,
            "dca_label": dca_label,
            "best_trades": best_trades,
            "worst_trades": worst_trades,
            "all_positions_analysed": len(trade_results),
            "behavioural_patterns": patterns,
            "coaching_summary": coaching,
            "data_timestamp": datetime.now(UTC).isoformat(),
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}
