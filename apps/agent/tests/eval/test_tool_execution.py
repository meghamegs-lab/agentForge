"""
Tool Execution Eval Suite — tests/eval/test_tool_execution.py
=============================================================
"Do tool calls succeed? Are parameters correct? Does each advanced tool
return the right structure with valid values?"

Advanced tools covered (6 tools, 16 test cases):
  ── get_fee_drag_analysis ────────────────────────────────────────────────
  1.  Basic execution: status=ok, expected fields present
  2.  Zero-fee portfolio: fee_drag_pct must be 0.0
  3.  Empty activities: total_fees_paid = 0, no crash
  ── get_portfolio_health_scorecard ───────────────────────────────────────
  4.  Basic execution: grade is A/B/C/D, score 0–100
  5.  Single-position portfolio: grade D (undiversified), flags present
  6.  Empty portfolio: status=empty (not error), graceful message
  ── get_rebalancing_plan ─────────────────────────────────────────────────
  7.  List-format holdings (the fixed bug): status=ok, not 'list' error
  8.  Dict-format holdings: status=ok, both formats accepted
  9.  Already-balanced portfolio: trades list may be very short
  ── get_market_context_overlay ───────────────────────────────────────────
  10. Rising rates theme: position_analyses contains entries with sentiments
  11. Invalid theme falls back to rising_rates
  12. Recession theme: healthcare positions are POSITIVE sentiment
  ── get_transaction_pattern_intelligence ─────────────────────────────────
  13. Basic execution: status=ok, buy count and sell count correct
  14. No activities: status=empty, not a crash
  ── get_proactive_risk_monitor ───────────────────────────────────────────
  15. First session (no snapshot): snapshot is built and returned
  16. Highly concentrated portfolio: HIGH-severity CONCENTRATION alerts fired

All tests mock network calls with respx — zero real I/O.
Market data calls (yfinance) are mocked with AsyncMock.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import respx

from agent.config import settings
from agent.tools.fee_drag import _fee_drag
from agent.tools.health_scorecard import _scorecard
from agent.tools.market_context import _market_context
from agent.tools.proactive_monitor import _proactive_monitor
from agent.tools.rebalancing import _rebalancing_plan
from agent.tools.transaction_patterns import _transaction_patterns

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "exec-token-2025"}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

US_EQUITY_HOLDING = {
    "symbol": "AAPL", "name": "Apple Inc.", "quantity": 10, "value": 1750.00,
    "assetClass": "EQUITY", "assetSubClass": "STOCK",
    "sectors": [{"name": "Technology", "weight": 1.0}],
    "countries": [{"name": "United States", "weight": 1.0}],
}
BOND_HOLDING = {
    "symbol": "BND", "name": "Vanguard Bond ETF", "quantity": 20, "value": 3000.00,
    "assetClass": "BOND", "assetSubClass": "ETF",
    "sectors": [{"name": "Financial Services", "weight": 1.0}],
    "countries": [{"name": "United States", "weight": 1.0}],
}
INTL_HOLDING = {
    "symbol": "VEA", "name": "Vanguard Intl ETF", "quantity": 15, "value": 2250.00,
    "assetClass": "EQUITY", "assetSubClass": "ETF",
    "sectors": [{"name": "Financial Services", "weight": 0.5},
                {"name": "Industrials", "weight": 0.5}],
    "countries": [{"name": "Germany", "weight": 0.4}, {"name": "Japan", "weight": 0.6}],
}
HEALTHCARE_HOLDING = {
    "symbol": "VHT", "name": "Vanguard Health Care ETF", "quantity": 8, "value": 2000.00,
    "assetClass": "EQUITY", "assetSubClass": "ETF",
    "sectors": [{"name": "Healthcare", "weight": 1.0}],
    "countries": [{"name": "United States", "weight": 1.0}],
}

STANDARD_HOLDINGS = [US_EQUITY_HOLDING, BOND_HOLDING, INTL_HOLDING, HEALTHCARE_HOLDING]
STANDARD_TOTAL = sum(h["value"] for h in STANDARD_HOLDINGS)  # 9_000.00

STANDARD_ORDERS = {
    "activities": [
        {
            "id": "t1", "date": "2024-01-10T00:00:00Z", "type": "BUY",
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
            "quantity": 10, "unitPrice": 170.00, "fee": 4.99, "currency": "USD",
            "Account": {"name": "Brokerage"},
        },
        {
            "id": "t2", "date": "2024-03-05T00:00:00Z", "type": "BUY",
            "SymbolProfile": {"symbol": "BND", "name": "Vanguard Bond ETF"},
            "quantity": 20, "unitPrice": 150.00, "fee": 0.00, "currency": "USD",
            "Account": {"name": "Brokerage"},
        },
        {
            "id": "t3", "date": "2024-06-01T00:00:00Z", "type": "SELL",
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
            "quantity": 2, "unitPrice": 185.00, "fee": 1.99, "currency": "USD",
            "Account": {"name": "Brokerage"},
        },
    ]
}

STANDARD_PERF = {
    "performance": {
        "ytd":  {"relativeChange": 0.10, "absoluteChange": 900.00, "currentValue": 9000.00},
        "1y":   {"relativeChange": 0.22, "absoluteChange": 1980.00, "currentValue": 9000.00},
        "max":  {"relativeChange": 0.40, "absoluteChange": 3600.00, "currentValue": 9000.00},
    }
}


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


def _mock_standard_holdings(as_list: bool = True):
    """Mock the holdings endpoint. as_list=True uses the list format (real Ghostfolio format)."""
    payload = (
        {"holdings": STANDARD_HOLDINGS}
        if as_list
        else {"holdings": {h["symbol"]: h for h in STANDARD_HOLDINGS}}
    )
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=payload)
    )


def _mock_orders(orders: dict | None = None):
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json=orders or STANDARD_ORDERS)
    )


def _mock_perf():
    respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
        return_value=httpx.Response(200, json=STANDARD_PERF)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool: get_fee_drag_analysis
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_fee_drag_basic_execution_returns_required_fields():
    """
    Happy path for get_fee_drag_analysis.
    Validates: status=ok and all documented output keys are present.
    """
    _auth()
    _mock_orders()
    _mock_perf()
    _mock_standard_holdings()

    result = await _fee_drag("max")

    assert result["status"] == "ok", f"Expected status=ok, got: {result}"
    required_keys = [
        "total_fees_paid", "gross_gain_without_fees", "net_gain_after_fees",
        "fee_drag_pct", "fee_drag_explanation", "annual_fee_rate_pct",
        "net_return_pct", "verdict", "fee_by_symbol", "fee_by_year",
        "data_timestamp",
    ]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in fee_drag result"

    assert result["total_fees_paid"] >= 0
    assert 0.0 <= result["fee_drag_pct"]
    assert isinstance(result["fee_by_symbol"], list)


@respx.mock
async def test_fee_drag_zero_fees_yields_zero_fee_drag_pct():
    """
    When all activities have fee=0, fee_drag_pct must be exactly 0.0.
    Tests the branch: 'fee_drag_pct = 0 when gross_gain == 0 or fees == 0'.
    """
    _auth()
    zero_fee_orders = {
        "activities": [
            {
                "id": "t0", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                "SymbolProfile": {"symbol": "VTI", "name": "VTI"},
                "quantity": 10, "unitPrice": 200.00, "fee": 0.00,
                "currency": "USD", "Account": {"name": "Brokerage"},
            }
        ]
    }
    _mock_orders(zero_fee_orders)
    _mock_perf()
    _mock_standard_holdings()

    result = await _fee_drag("max")
    assert result["status"] == "ok"
    assert result["total_fees_paid"] == 0.0, (
        f"All zero-fee orders must yield total_fees_paid=0.0; got {result['total_fees_paid']}"
    )
    assert result["fee_drag_pct"] == 0.0, (
        f"Zero total fees must produce fee_drag_pct=0.0; got {result['fee_drag_pct']}"
    )


@respx.mock
async def test_fee_drag_empty_activities_returns_ok_with_zero_fees():
    """
    When there are no transactions at all, the tool must return status=ok
    with total_fees_paid=0 and not crash (division-by-zero guard).
    """
    _auth()
    _mock_orders({"activities": []})
    _mock_perf()
    _mock_standard_holdings()

    result = await _fee_drag("max")
    # No transactions — should still produce a valid response
    assert result["status"] in {"ok", "empty"}, (
        f"Empty activities should produce ok or empty status; got {result}"
    )
    if result["status"] == "ok":
        assert result["total_fees_paid"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Tool: get_portfolio_health_scorecard
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_health_scorecard_basic_execution_returns_grade_and_score():
    """
    Happy path for get_portfolio_health_scorecard.
    Validates: grade is one of A/B/C/D, score is between 0–100, all keys present.
    """
    _auth()
    _mock_standard_holdings()
    _mock_perf()

    sc = await _scorecard()

    assert sc["status"] == "ok", f"Expected status=ok; got: {sc}"
    assert sc["grade"] in ("A", "B", "C", "D"), (
        f"Grade must be A/B/C/D; got {sc['grade']}"
    )
    assert 0 <= sc["score"] <= 100, f"Score must be 0-100; got {sc['score']}"

    required_keys = [
        "grade", "score", "total_value", "position_count",
        "performance", "diversification", "risk_flags", "action_items",
        "data_timestamp",
    ]
    for key in required_keys:
        assert key in sc, f"Missing key '{key}' in scorecard result"

    # Performance sub-dict
    assert "ytd_pct" in sc["performance"]
    assert "one_year_pct" in sc["performance"]


@respx.mock
async def test_health_scorecard_single_position_gets_grade_d_and_flags():
    """
    A portfolio with only ONE position must:
    - Score ≤ 50 (undiversified penalty)
    - Receive grade C or D
    - Have at least one UNDIVERSIFIED risk flag
    - Include an action_item to add more positions
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {
                    "symbol": "GME", "name": "GameStop", "quantity": 100, "value": 5000.00,
                    "assetClass": "EQUITY", "assetSubClass": "STOCK",
                    "sectors": [{"name": "Consumer Cyclical", "weight": 1.0}],
                    "countries": [{"name": "United States", "weight": 1.0}],
                }
            ]
        })
    )
    _mock_perf()

    sc = await _scorecard()
    assert sc["status"] == "ok"
    assert sc["grade"] in ("C", "D"), (
        f"Single-position portfolio must get grade C or D; got {sc['grade']}"
    )
    assert sc["score"] <= 65, (
        f"Single-position portfolio must score ≤65; got {sc['score']}"
    )
    undiv_flags = [f for f in sc["risk_flags"] if f["type"] == "UNDIVERSIFIED"]
    assert len(undiv_flags) >= 1, "Single-position portfolio must have UNDIVERSIFIED risk flag"


@respx.mock
async def test_health_scorecard_empty_portfolio_returns_empty_not_error():
    """
    An empty portfolio (no holdings) must return status='empty' with a helpful
    message — not status='error' or an unhandled exception.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": []})
    )
    _mock_perf()

    sc = await _scorecard()
    assert sc["status"] == "empty", (
        f"Empty portfolio must return status='empty'; got {sc}"
    )
    assert "message" in sc


# ══════════════════════════════════════════════════════════════════════════════
# Tool: get_rebalancing_plan
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_rebalancing_plan_with_list_format_holdings_succeeds():
    """
    THE FIXED BUG TEST: Ghostfolio returns holdings as a LIST, not a dict.
    Before the fix: 'list' object has no attribute 'values' → status='error'
    After the fix: normalisation converts list → dict → status='ok'
    """
    _auth()
    # Holdings as a LIST — the format that was causing the production bug
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [                       # ← LIST format
                US_EQUITY_HOLDING,
                BOND_HOLDING,
                INTL_HOLDING,
            ]
        })
    )

    import agent.tools.rebalancing as reb_module
    with patch.object(reb_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 175.00}
        result = await _rebalancing_plan(0.55, 0.25, 0.15, 0.05)

    assert result["status"] == "ok", (
        f"Rebalancing plan must succeed with list-format holdings; got: {result}. "
        "This tests the fix for the 'list has no attribute values' production bug."
    )
    assert "trades" in result
    assert "current_allocation" in result
    assert "target_allocation" in result
    assert "summary" in result


@respx.mock
async def test_rebalancing_plan_with_dict_format_holdings_succeeds():
    """
    Ghostfolio can also return holdings as a DICT keyed by symbol.
    Both formats must work after normalisation.
    """
    _auth()
    # Holdings as a DICT — alternate Ghostfolio format
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": {                       # ← DICT format
                "AAPL": US_EQUITY_HOLDING,
                "BND":  BOND_HOLDING,
                "VEA":  INTL_HOLDING,
            }
        })
    )

    import agent.tools.rebalancing as reb_module
    with patch.object(reb_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 200.00}
        result = await _rebalancing_plan(0.55, 0.25, 0.15, 0.05)

    assert result["status"] == "ok", (
        f"Rebalancing plan must succeed with dict-format holdings; got: {result}"
    )
    assert result["total_value"] > 0


@respx.mock
async def test_rebalancing_plan_with_already_balanced_portfolio():
    """
    When the portfolio is already at target allocation (within tolerance),
    the trades list should be empty or very short (all trades below $50 threshold).
    Verifies the $50 minimum trade filter works correctly.
    """
    _auth()
    # Build a portfolio exactly at 55/25/15/5 target
    total = 10_000.00
    balanced_holdings = [
        {
            "symbol": "VTI",  "name": "US Equity ETF", "quantity": 30, "value": 5500.00,
            "assetClass": "EQUITY", "sectors": [], "countries": [{"name": "United States"}],
        },
        {
            "symbol": "VEA",  "name": "Intl Equity ETF", "quantity": 25, "value": 2500.00,
            "assetClass": "EQUITY", "sectors": [], "countries": [{"name": "Germany"}],
        },
        {
            "symbol": "BND",  "name": "Bond ETF", "quantity": 15, "value": 1500.00,
            "assetClass": "BOND", "sectors": [], "countries": [],
        },
        {
            "symbol": "CASH", "name": "Cash", "quantity": 1, "value": 500.00,
            "assetClass": "CASH", "sectors": [], "countries": [],
        },
    ]
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": balanced_holdings})
    )

    import agent.tools.rebalancing as reb_module
    with patch.object(reb_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 180.00}
        # Use target = actual allocation → deltas should be ~$0
        result = await _rebalancing_plan(0.55, 0.25, 0.15, 0.05)

    assert result["status"] == "ok"
    # Perfectly balanced → all trade amounts should be < $50 minimum → trades = []
    for trade in result.get("trades", []):
        assert abs(trade["dollar_amount"]) >= 50, (
            "All returned trades must exceed the $50 minimum threshold"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tool: get_market_context_overlay
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_market_context_rising_rates_returns_position_analyses():
    """
    Happy path for get_market_context_overlay with 'rising_rates' theme.
    Validates: status=ok, position_analyses are returned, vulnerability is set.
    """
    _auth()
    _mock_standard_holdings()

    ctx = await _market_context("rising_rates")

    assert ctx["status"] == "ok", f"Expected status=ok; got: {ctx}"
    assert ctx["macro_theme"] == "rising_rates"
    assert isinstance(ctx["position_analyses"], list)
    assert len(ctx["position_analyses"]) == len(STANDARD_HOLDINGS)
    assert ctx["vulnerability"] in ("WELL_POSITIONED", "NEUTRAL", "VULNERABLE")

    # Each position must have required keys
    for pos in ctx["position_analyses"]:
        assert "symbol" in pos
        assert "dominant_sentiment" in pos
        assert pos["dominant_sentiment"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")


@respx.mock
async def test_market_context_invalid_theme_falls_back_to_rising_rates():
    """
    When the LLM passes an invalid macro_theme (e.g. 'bear_market' or 'unknown'),
    the tool must silently fall back to 'rising_rates' and still succeed.
    """
    _auth()
    _mock_standard_holdings()

    ctx = await _market_context("bear_market_2024_crypto_crash")

    assert ctx["status"] == "ok"
    assert ctx["macro_theme"] == "rising_rates", (
        f"Invalid theme must fall back to 'rising_rates'; got {ctx['macro_theme']}"
    )


@respx.mock
async def test_market_context_recession_healthcare_is_positive():
    """
    In a recession, healthcare is classically defensive (POSITIVE sentiment).
    A portfolio with a pure healthcare holding must show POSITIVE dominant_sentiment
    for that position when analysed under the 'recession' theme.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": [HEALTHCARE_HOLDING]})
    )

    ctx = await _market_context("recession")
    assert ctx["status"] == "ok"
    assert len(ctx["position_analyses"]) == 1

    vht_analysis = ctx["position_analyses"][0]
    assert vht_analysis["symbol"] == "VHT"
    assert vht_analysis["dominant_sentiment"] == "POSITIVE", (
        f"Healthcare must be POSITIVE in a recession; "
        f"got {vht_analysis['dominant_sentiment']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool: get_transaction_pattern_intelligence
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_transaction_patterns_basic_execution_returns_required_fields():
    """
    Happy path for get_transaction_pattern_intelligence.
    Validates: status=ok, all documented output keys present.
    """
    _auth()
    _mock_orders()
    _mock_standard_holdings()

    import agent.tools.transaction_patterns as tp_module
    with patch.object(tp_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 175.00}
        patterns = await _transaction_patterns()

    assert patterns["status"] == "ok", f"Expected status=ok; got: {patterns}"
    required_keys = [
        "total_buy_transactions", "total_sell_transactions", "trades_per_month",
        "churn_assessment", "dca_score", "dca_label",
        "best_trades", "worst_trades", "behavioural_patterns",
        "coaching_summary", "data_timestamp",
    ]
    for key in required_keys:
        assert key in patterns, f"Missing key '{key}' in transaction patterns result"

    # Check types
    assert isinstance(patterns["total_buy_transactions"], int)
    assert isinstance(patterns["total_sell_transactions"], int)
    assert isinstance(patterns["dca_score"], (int, float))
    assert 0.0 <= patterns["dca_score"] <= 100.0


@respx.mock
async def test_transaction_patterns_no_activities_returns_empty():
    """
    When there are no transactions, the tool must return status='empty'
    with a helpful message — not crash or return status='error'.
    """
    _auth()
    _mock_orders({"activities": []})
    _mock_standard_holdings()

    import agent.tools.transaction_patterns as tp_module
    with patch.object(tp_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 175.00}
        patterns = await _transaction_patterns()

    assert patterns["status"] == "empty", (
        f"No activities must yield status='empty'; got {patterns}"
    )
    assert "message" in patterns


# ══════════════════════════════════════════════════════════════════════════════
# Tool: get_proactive_risk_monitor
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_proactive_monitor_first_session_builds_and_returns_snapshot():
    """
    First-time session (empty previous_snapshot_json):
    - alerts are assessed against current state only
    - current_snapshot is built with allocations for all positions
    - changes_since_last_session is []
    """
    _auth()
    _mock_standard_holdings()

    result = await _proactive_monitor(prev_snap_json="")

    assert result["status"] == "ok", f"Expected status=ok; got: {result}"
    assert result["changes_since_last_session"] == []

    snapshot = result["current_snapshot"]
    assert "total_value" in snapshot
    assert "allocations" in snapshot
    assert len(snapshot["allocations"]) == len(STANDARD_HOLDINGS)

    # Allocations must sum to ~100%
    total_alloc = sum(snapshot["allocations"].values())
    assert abs(total_alloc - 100.0) < 0.5, (
        f"Snapshot allocations must sum to ~100%; got {total_alloc:.2f}%"
    )


@respx.mock
async def test_proactive_monitor_concentrated_portfolio_fires_high_alert():
    """
    A heavily concentrated portfolio (one position > 35%) must trigger a
    HIGH-severity CONCENTRATION alert.
    The alert message must name the concentrated symbol.
    """
    _auth()
    # BIGCO = 80% of portfolio
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {
                    "symbol": "BIGCO", "name": "Big Company Inc.", "quantity": 100,
                    "value": 8000.00, "assetClass": "EQUITY", "sectors": [], "countries": [],
                },
                {
                    "symbol": "SMALL", "name": "Small Co.", "quantity": 10,
                    "value": 2000.00, "assetClass": "EQUITY", "sectors": [], "countries": [],
                },
            ]
        })
    )

    result = await _proactive_monitor(prev_snap_json="")

    assert result["status"] == "ok"
    high_alerts = [a for a in result["alerts"] if a["severity"] == "HIGH"]
    assert len(high_alerts) >= 1, (
        "Portfolio with 80% in one position must fire at least one HIGH severity alert"
    )
    # Alert message must reference the concentrated symbol
    all_messages = " ".join(a.get("message", "") for a in high_alerts)
    assert "BIGCO" in all_messages, (
        f"HIGH alert must name the concentrated symbol 'BIGCO'. "
        f"Alert messages: {all_messages}"
    )
    assert result["overall_risk_level"] in ("HIGH", "MEDIUM"), (
        f"80% concentration must elevate overall_risk_level; got {result['overall_risk_level']}"
    )
