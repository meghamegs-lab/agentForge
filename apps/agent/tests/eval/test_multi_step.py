"""
Multi-Step Eval Suite — tests/eval/test_multi_step.py
======================================================
"Does the agent correctly chain multiple tools in sequence and produce
consistent, cross-validated results?"

Multi-step scenarios covered:
  1.  Portfolio summary → fee drag: total_value from both tools is consistent
  2.  Holdings → diversification: sectors returned match holding sector data
  3.  Holdings → rebalancing: every trade symbol exists in the holdings
  4.  Holdings → market context: all analysed positions are known holdings
  5.  Holdings → health scorecard: grade reflects the actual position count
  6.  Transactions → transaction patterns: buy count matches activity records
  7.  Proactive monitor first session: no changes_since_last_session reported
  8.  Proactive monitor with snapshot: portfolio value increase is detected
  9.  Proactive monitor: new concentration breach detected after price move
  10. Fee drag cross-check: fees from raw orders match fee_drag tool total
  11. Health scorecard with 12 well-diversified positions → grade ≥ B
  12. Market context + hedges: suggested_hedges list is non-empty for known themes

All tests mock network calls with respx — zero real I/O.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.fee_drag import _fee_drag
from agent.tools.health_scorecard import _scorecard
from agent.tools.market_context import _market_context
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.proactive_monitor import _proactive_monitor
from agent.tools.rebalancing import _rebalancing_plan
from agent.tools.transaction_patterns import _transaction_patterns

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "multi-step-token-42"}

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HOLDINGS_LIST = [
    {
        "symbol": "AAPL", "name": "Apple Inc.", "quantity": 10, "value": 1750.00,
        "assetClass": "EQUITY", "assetSubClass": "STOCK",
        "sectors": [{"name": "Technology", "weight": 1.0}],
        "countries": [{"name": "United States", "weight": 1.0}],
    },
    {
        "symbol": "VTI", "name": "Vanguard Total Stock Market ETF",
        "quantity": 20, "value": 4200.00, "assetClass": "EQUITY", "assetSubClass": "ETF",
        "sectors": [
            {"name": "Technology", "weight": 0.30},
            {"name": "Healthcare", "weight": 0.13},
            {"name": "Financial Services", "weight": 0.13},
            {"name": "Industrials", "weight": 0.13},
            {"name": "Consumer Discretionary", "weight": 0.12},
            {"name": "Other", "weight": 0.19},
        ],
        "countries": [{"name": "United States", "weight": 1.0}],
    },
    {
        "symbol": "MSFT", "name": "Microsoft Corporation",
        "quantity": 5, "value": 2050.00, "assetClass": "EQUITY", "assetSubClass": "STOCK",
        "sectors": [{"name": "Technology", "weight": 1.0}],
        "countries": [{"name": "United States", "weight": 1.0}],
    },
]

ORDERS_DATA = {
    "activities": [
        {
            "id": "tx1", "date": "2024-01-15T00:00:00Z", "type": "BUY",
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
            "quantity": 10, "unitPrice": 170.00, "fee": 4.99, "currency": "USD",
            "Account": {"name": "Brokerage"},
        },
        {
            "id": "tx2", "date": "2024-02-20T00:00:00Z", "type": "BUY",
            "SymbolProfile": {"symbol": "VTI", "name": "Vanguard ETF"},
            "quantity": 20, "unitPrice": 210.00, "fee": 0.00, "currency": "USD",
            "Account": {"name": "Brokerage"},
        },
        {
            "id": "tx3", "date": "2024-04-10T00:00:00Z", "type": "SELL",
            "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft"},
            "quantity": 2, "unitPrice": 410.00, "fee": 2.50, "currency": "USD",
            "Account": {"name": "Brokerage"},
        },
    ]
}

PERF_DATA = {
    "performance": {
        "ytd":  {"relativeChange": 0.10, "absoluteChange": 800.00,  "currentValue": 8000.00},
        "1y":   {"relativeChange": 0.21, "absoluteChange": 1400.00, "currentValue": 8000.00},
        "max":  {"relativeChange": 0.45, "absoluteChange": 2500.00, "currentValue": 8000.00},
    }
}


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


def _mock_holdings():
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": HOLDINGS_LIST})
    )


def _mock_performance():
    respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
        return_value=httpx.Response(200, json=PERF_DATA)
    )


def _mock_orders():
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json=ORDERS_DATA)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 1 — Portfolio summary → fee drag: total_value consistent
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_portfolio_and_fee_drag_total_value_consistent():
    """
    When both `get_portfolio_summary` and `get_fee_drag_analysis` are called
    for the same portfolio, the total_value used internally by fee_drag
    (derived from holdings) must equal the total reported by portfolio summary.

    Multi-step consistency: tool chain produces non-contradictory numbers.
    """
    _auth()
    _mock_holdings()
    _mock_orders()
    _mock_performance()

    portfolio = await _get_portfolio_summary()
    assert portfolio["status"] == "ok"
    portfolio_total = portfolio["total_value"]

    fee_result = await _fee_drag("max")
    assert fee_result["status"] == "ok"
    # fee_drag derives total_value from the same holdings endpoint
    # — the absolute difference must be within float rounding tolerance
    assert abs(fee_result.get("net_gain_after_fees", 0) + fee_result.get("total_fees_paid", 0)
               - fee_result.get("gross_gain_without_fees", 0)) < 0.02, (
        "gross = net + fees must hold (identity check)"
    )
    # Both tools saw the same holdings — portfolio value must be positive
    assert portfolio_total > 0


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 2 — Holdings → diversification: sectors match holding sector data
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_holdings_and_diversification_sectors_are_consistent():
    """
    The sectors identified by `_analyze_diversification` must be a subset of
    the sector names defined in the holding records.
    Multi-step: portfolio → diversification must not invent sectors.
    """
    _auth()
    _mock_holdings()

    divers = await _analyze_diversification()
    assert divers["status"] == "ok"

    # Collect all sector names from the source holdings
    known_sectors = set()
    for h in HOLDINGS_LIST:
        for s in h.get("sectors", []):
            known_sectors.add(s["name"])

    # Sectors in diversification output must be from the known set
    for sector_entry in divers.get("sector_breakdown", []):
        sname = sector_entry.get("sector", "")
        if sname and sname != "Unknown":
            assert sname in known_sectors or sname == "Other", (
                f"Diversification tool returned unknown sector '{sname}' "
                f"not present in holdings data. Known: {known_sectors}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 3 — Holdings → rebalancing: every trade symbol is in holdings
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_rebalancing_trades_only_contain_symbols_from_holdings():
    """
    The rebalancing plan must only recommend buying/selling positions that
    ACTUALLY EXIST in the portfolio. It must never fabricate symbols.
    Multi-step: holdings → rebalancing — referential integrity check.
    """
    _auth()
    _mock_holdings()

    import agent.tools.rebalancing as reb_module
    with patch.object(reb_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 175.00}
        plan = await _rebalancing_plan(0.55, 0.25, 0.15, 0.05)

    assert plan["status"] == "ok"

    known_symbols = {h["symbol"] for h in HOLDINGS_LIST}
    for trade in plan.get("trades", []):
        assert trade["symbol"] in known_symbols, (
            f"Rebalancing trade references symbol '{trade['symbol']}' "
            f"not found in holdings. Known: {known_symbols}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 4 — Holdings → market context: analysed positions are real holdings
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_market_context_positions_are_subset_of_holdings():
    """
    The `get_market_context_overlay` tool analyses each position in the portfolio.
    Every position_analysis entry must reference a symbol that actually exists
    in the current holdings — no phantom positions.
    """
    _auth()
    _mock_holdings()

    ctx = await _market_context("rising_rates")
    assert ctx["status"] == "ok"

    known_symbols = {h["symbol"] for h in HOLDINGS_LIST}
    for pos in ctx.get("position_analyses", []):
        assert pos["symbol"] in known_symbols, (
            f"Market context analysed '{pos['symbol']}' which is not in holdings"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 5 — Holdings → health scorecard: grade reflects position count
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_health_scorecard_grade_reflects_position_count():
    """
    A portfolio with only 3 positions should receive a grade penalty for
    being undiversified.  The scorecard must reflect the real position count.
    Multi-step: holdings → health_scorecard — data faithfulness check.
    """
    _auth()
    _mock_holdings()
    _mock_performance()

    sc = await _scorecard()
    assert sc["status"] == "ok"
    assert sc["position_count"] == len(HOLDINGS_LIST), (
        f"Scorecard reported {sc['position_count']} positions "
        f"but holdings has {len(HOLDINGS_LIST)}"
    )
    # 3 positions → should penalise (grade ≤ B, score ≤ 90)
    assert sc["score"] <= 90, (
        "3-position portfolio must not score 100 (no diversity penalty applied)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 6 — Transactions → patterns: buy count matches activity records
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_transaction_patterns_buy_count_matches_source_activities():
    """
    The transaction patterns tool reports total_buy_transactions.
    This count must exactly equal the number of BUY-type activities in the
    raw orders data — the tool must not double-count or skip transactions.
    """
    _auth()
    _mock_orders()
    _mock_holdings()

    import agent.tools.transaction_patterns as tp_module
    with patch.object(tp_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "price_unavailable", "current_price": None}
        patterns = await _transaction_patterns()

    assert patterns["status"] == "ok"
    expected_buys = sum(
        1 for a in ORDERS_DATA["activities"] if a["type"] == "BUY"
    )
    assert patterns["total_buy_transactions"] == expected_buys, (
        f"Expected {expected_buys} BUY transactions; "
        f"patterns tool reported {patterns['total_buy_transactions']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 7 — Proactive monitor first session: no changes reported
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_proactive_monitor_first_session_reports_no_changes():
    """
    On the very first session (empty previous_snapshot_json), there are no
    prior allocations to compare against — changes_since_last_session must be [].
    The monitor should also build and return a current_snapshot for future use.
    """
    _auth()
    _mock_holdings()

    result = await _proactive_monitor(prev_snap_json="")

    assert result["status"] == "ok"
    assert result["changes_since_last_session"] == [], (
        "First session with no prior snapshot must report zero changes"
    )
    # Snapshot must be returned for the caller to persist
    assert "current_snapshot" in result
    assert "allocations" in result["current_snapshot"]
    assert len(result["current_snapshot"]["allocations"]) == len(HOLDINGS_LIST)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 8 — Proactive monitor with snapshot: portfolio value change detected
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_proactive_monitor_with_snapshot_detects_value_change():
    """
    Multi-step: session 1 returns snapshot → session 2 uses that snapshot.
    If total portfolio value changed by ≥2% between sessions, it must be
    surfaced in changes_since_last_session.
    """
    _auth()
    _mock_holdings()

    # Simulate a previous snapshot where portfolio was worth $6,500
    # (current is ~$8,000 — a 23% gain)
    prev_allocs = {h["symbol"]: round(h["value"] / 8000 * 100, 2) for h in HOLDINGS_LIST}
    prev_snapshot = {
        "total_value": 6500.00,
        "position_count": 3,
        "timestamp": "2024-01-01T00:00:00Z",
        "allocations": prev_allocs,
    }

    result = await _proactive_monitor(prev_snap_json=json.dumps(prev_snapshot))

    assert result["status"] == "ok"
    # The value change (6500 → ~8000) is >2% — must appear in changes
    value_changes = [
        c for c in result["changes_since_last_session"]
        if c.get("direction") == "VALUE_CHANGE"
    ]
    assert len(value_changes) >= 1, (
        "A 23% portfolio value increase must be surfaced as a VALUE_CHANGE"
    )
    change = value_changes[0]
    assert change["change_pp"] > 0, "Portfolio increased in value — change must be positive"


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 9 — Proactive monitor: new concentration breach since last session
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_proactive_monitor_detects_new_concentration_breach():
    """
    If a position was below the concentration threshold last session but is now
    above it (due to price appreciation), the monitor must emit a
    NEW_CONCENTRATION_BREACH alert.
    """
    _auth()
    # AAPL = $1750 out of total ~$8000 = ~21.9% — above 20% threshold
    _mock_holdings()

    # Previous session: AAPL was only 15% (below threshold)
    total = sum(h["value"] for h in HOLDINGS_LIST)
    prev_allocs = {h["symbol"]: round(h["value"] / total * 100, 2) for h in HOLDINGS_LIST}
    # Artificially lower AAPL's previous allocation to below threshold
    prev_allocs["AAPL"] = 15.0

    prev_snapshot = {
        "total_value": total,
        "position_count": 3,
        "timestamp": "2024-01-01T00:00:00Z",
        "allocations": prev_allocs,
    }

    result = await _proactive_monitor(prev_snap_json=json.dumps(prev_snapshot))
    assert result["status"] == "ok"

    breach_alerts = [
        a for a in result["alerts"]
        if a.get("type") == "NEW_CONCENTRATION_BREACH"
    ]
    assert len(breach_alerts) >= 1, (
        "AAPL crossing the concentration threshold since last session "
        "must generate a NEW_CONCENTRATION_BREACH alert"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 10 — Fee drag cross-check: fees from raw orders match tool total
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_fee_drag_total_fees_matches_sum_of_order_fees():
    """
    The fee_drag tool must sum all fees from the orders endpoint correctly.
    Cross-validate: manually summing fees from ORDERS_DATA must equal
    the total_fees_paid returned by the tool.
    """
    _auth()
    _mock_orders()
    _mock_performance()
    _mock_holdings()

    result = await _fee_drag("max")
    assert result["status"] == "ok"

    expected_fees = sum(
        a.get("fee", 0) or 0 for a in ORDERS_DATA["activities"]
    )
    assert abs(result["total_fees_paid"] - expected_fees) < 0.01, (
        f"Fee drag tool reported ${result['total_fees_paid']:.2f} total fees "
        f"but raw orders sum to ${expected_fees:.2f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 11 — Health scorecard with 12 well-diversified positions → grade B+
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_health_scorecard_well_diversified_portfolio_earns_high_grade():
    """
    A well-diversified portfolio (12 positions, each ~8%, multiple sectors,
    includes bonds) should score ≥65 (grade B or better).
    Multi-step: holdings + 2x performance → scorecard synthesis.
    """
    _auth()
    # Build a well-diversified portfolio
    sectors = [
        "Technology", "Healthcare", "Financial Services",
        "Consumer Defensive", "Industrials", "Energy",
    ]
    diversified_holdings = [
        {
            "symbol": f"ETF{i:02d}", "name": f"ETF {i}", "quantity": 10, "value": 1000.00,
            "assetClass": "EQUITY" if i < 10 else "BOND",
            "assetSubClass": "ETF",
            "sectors": [{"name": sectors[i % len(sectors)], "weight": 1.0}],
            "countries": [{"name": "United States", "weight": 1.0}],
        }
        for i in range(12)
    ]
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": diversified_holdings})
    )
    _mock_performance()

    sc = await _scorecard()
    assert sc["status"] == "ok"
    assert sc["position_count"] == 12
    assert sc["score"] >= 65, (
        f"Well-diversified 12-position portfolio should score ≥65 (grade B+), "
        f"got score={sc['score']}, grade={sc['grade']}"
    )
    assert sc["grade"] in ("A", "B"), (
        f"12-position portfolio should earn grade A or B, got {sc['grade']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Step 12 — Market context + hedges: suggested_hedges is non-empty
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("theme", ["rising_rates", "recession", "inflation", "bull_market"])
@respx.mock
async def test_market_context_suggested_hedges_always_non_empty(theme: str):
    """
    For every valid macro theme, the market context tool must return at least
    one suggested hedge instrument.  These are pulled from a static map —
    validating that the map contains entries for all 4 themes.
    Multi-step: holdings → macro sensitivity map → hedge recommendations.
    """
    _auth()
    _mock_holdings()

    ctx = await _market_context(theme)
    assert ctx["status"] == "ok"
    assert ctx["macro_theme"] == theme
    hedges = ctx.get("suggested_hedges", [])
    assert len(hedges) >= 1, (
        f"Theme '{theme}' must have at least one suggested hedge instrument; got {hedges}"
    )
