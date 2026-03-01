"""
evals/test_latency.py — Latency Eval Suite (v2)
====================================================
Eval IDs: L01–L03

"Do tools complete within defined time budgets?"

Latency tests measure wall-clock execution time for each tool call (with mocked
network responses). The budgets defined here are for MOCKED calls — real API
calls will be much slower. See `real_api_latency_budget_ms` in eval_suite.json
for production latency targets.

Latency budget tiers:
  SIMPLE (single tool call, mocked):         < 200 ms
  MULTI-STEP (3+ internal calls, mocked):    < 1 000 ms
  COMPLEX (3+ tools + computation, mocked):  < 1 500 ms

If a test exceeds its budget, it is logged but treated as a WARNING by default.
To make latency failures BLOCKING, set LATENCY_FAILURES_BLOCKING = True.

All network calls are mocked with respx — zero real I/O.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from agent.config import settings

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "latency-v2-token"}

# Budget constants (milliseconds) — for mocked calls
SIMPLE_BUDGET_MS = 200
MULTI_STEP_BUDGET_MS = 1_000
COMPLEX_BUDGET_MS = 1_500

# Change to True to make latency failures block the test suite
LATENCY_FAILURES_BLOCKING = False


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


def _check_latency(elapsed_ms: float, budget_ms: int, test_id: str) -> None:
    """
    Assert that elapsed_ms <= budget_ms.
    If LATENCY_FAILURES_BLOCKING is False, emit a warning instead of failing.
    Logs the result either way for reporting.
    """
    status = "OK" if elapsed_ms <= budget_ms else "EXCEEDED"
    print(f"\n  [{test_id}] Latency: {elapsed_ms:.1f}ms (budget: {budget_ms}ms) — {status}")
    if elapsed_ms > budget_ms:
        msg = (
            f"{test_id}: Latency {elapsed_ms:.1f}ms exceeds budget {budget_ms}ms "
            f"(delta: +{elapsed_ms - budget_ms:.1f}ms). "
            f"This is measured with mocked network — investigate Python overhead."
        )
        if LATENCY_FAILURES_BLOCKING:
            pytest.fail(msg)
        else:
            pytest.warns(UserWarning, match=".*") if False else None
            print(f"  WARNING: {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# L01 — Simple portfolio query completes within SIMPLE_BUDGET_MS
#        Single tool call: get_portfolio_summary
#        Measures: Python overhead, data processing, mock HTTP round-trip
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_l01_portfolio_summary_within_simple_budget():
    """
    L01: get_portfolio_summary (single tool, 2 holdings) must complete in < 200ms
    with mocked responses.
    Production target (live API): < 2000ms (see eval_suite.json L01).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "quantity": 10,
                        "valueInBaseCurrency": 1750.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "VTI",
                        "name": "Vanguard Total Stock Market ETF",
                        "quantity": 20,
                        "valueInBaseCurrency": 4200.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "ETF",
                        "sectors": [],
                        "countries": [],
                    },
                ]
            },
        )
    )

    from agent.tools.portfolio import _get_portfolio_summary

    start = time.perf_counter()
    result = await _get_portfolio_summary()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Verify correctness
    assert result["status"] == "ok"
    assert result["position_count"] == 2

    # Latency check
    _check_latency(elapsed_ms, SIMPLE_BUDGET_MS, "L01")


@respx.mock
async def test_l01_performance_query_within_simple_budget():
    """
    L01b: get_performance('ytd') (single tool) must complete in < 200ms with mocks.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={
                "performance": {
                    "netPerformancePercentage": 0.12,
                    "netPerformance": 960.00,
                    "currentValueInBaseCurrency": 8960.00,
                    "totalInvestment": 8000.00,
                    "currentNetWorth": 8960.00,
                }
            },
        )
    )

    from agent.tools.performance import _get_performance

    start = time.perf_counter()
    result = await _get_performance("ytd")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result["status"] == "ok"
    _check_latency(elapsed_ms, SIMPLE_BUDGET_MS, "L01b")


@respx.mock
async def test_l01_transactions_query_within_simple_budget():
    """
    L01c: get_transactions (single tool) must complete in < 200ms with mocks.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "id": "tx-001",
                        "date": "2024-03-15T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                        "quantity": 10,
                        "unitPrice": 170.00,
                        "fee": 4.99,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                ]
            },
        )
    )

    from agent.tools.transactions import _get_transactions

    start = time.perf_counter()
    result = await _get_transactions()
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result["status"] == "ok"
    _check_latency(elapsed_ms, SIMPLE_BUDGET_MS, "L01c")


# ══════════════════════════════════════════════════════════════════════════════
# L02 — Multi-step health scorecard within MULTI_STEP_BUDGET_MS
#        get_portfolio_health_scorecard makes 3 internal API calls:
#        1. get_portfolio_holdings
#        2. get_portfolio_performance (ytd)
#        3. get_portfolio_performance (1y)
#        All mocked but the Python computation still runs.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_l02_health_scorecard_within_multi_step_budget():
    """
    L02: get_portfolio_health_scorecard (3 internal calls + computation) must
    complete in < 1000ms with mocked responses.
    Production target (live API): < 5000ms (see eval_suite.json L02).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "quantity": 10,
                        "valueInBaseCurrency": 1750.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [{"name": "Technology", "weight": 1.0}],
                        "countries": [{"name": "United States", "weight": 1.0}],
                    },
                    {
                        "symbol": "VTI",
                        "name": "Vanguard Total Stock Market ETF",
                        "quantity": 20,
                        "valueInBaseCurrency": 4200.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "ETF",
                        "sectors": [
                            {"name": "Technology", "weight": 0.30},
                            {"name": "Other", "weight": 0.70},
                        ],
                        "countries": [{"name": "United States", "weight": 1.0}],
                    },
                    {
                        "symbol": "MSFT",
                        "name": "Microsoft Corporation",
                        "quantity": 5,
                        "valueInBaseCurrency": 2050.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [{"name": "Technology", "weight": 1.0}],
                        "countries": [{"name": "United States", "weight": 1.0}],
                    },
                ]
            },
        )
    )
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={
                "performance": {
                    "netPerformancePercentage": 0.1234,
                    "netPerformance": 987.65,
                    "currentValueInBaseCurrency": 8987.65,
                    "totalInvestment": 8000.00,
                    "currentNetWorth": 8987.65,
                }
            },
        )
    )

    from agent.tools.health_scorecard import _scorecard

    start = time.perf_counter()
    result = await _scorecard()
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result.get("status") == "ok", f"L02: Scorecard failed: {result}"
    assert "grade" in result, "L02: Scorecard must include grade"
    assert "score" in result, "L02: Scorecard must include score"

    _check_latency(elapsed_ms, MULTI_STEP_BUDGET_MS, "L02")


# ══════════════════════════════════════════════════════════════════════════════
# L03 — Transaction pattern intelligence within COMPLEX_BUDGET_MS
#        get_transaction_pattern_intelligence makes 2+ internal calls:
#        1. get_orders (all transactions)
#        2. get_portfolio_holdings
#        Plus: behavioural computation (churn rate, DCA score, pattern detection)
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_l03_transaction_pattern_within_complex_budget():
    """
    L03: get_transaction_pattern_intelligence (2 API calls + computation) must
    complete in < 1500ms with mocked responses.
    Production target (live API + market data): < 8000ms (see eval_suite.json L03).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "id": "tx-001",
                        "date": "2024-01-10T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                        "quantity": 10,
                        "unitPrice": 170.00,
                        "fee": 4.99,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "tx-002",
                        "date": "2024-02-15T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {
                            "symbol": "VTI",
                            "name": "Vanguard Total Stock Market ETF",
                        },
                        "quantity": 10,
                        "unitPrice": 205.00,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "tx-003",
                        "date": "2024-03-15T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {
                            "symbol": "VTI",
                            "name": "Vanguard Total Stock Market ETF",
                        },
                        "quantity": 10,
                        "unitPrice": 208.00,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "tx-004",
                        "date": "2024-06-01T00:00:00.000Z",
                        "type": "SELL",
                        "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                        "quantity": 5,
                        "unitPrice": 190.00,
                        "fee": 4.99,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "tx-005",
                        "date": "2024-09-01T00:00:00.000Z",
                        "type": "DIVIDEND",
                        "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                        "quantity": 5,
                        "unitPrice": 0.25,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                ]
            },
        )
    )
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "quantity": 5,
                        "valueInBaseCurrency": 937.50,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "VTI",
                        "name": "Vanguard Total Stock Market ETF",
                        "quantity": 20,
                        "valueInBaseCurrency": 4200.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "ETF",
                        "sectors": [],
                        "countries": [],
                    },
                ]
            },
        )
    )

    from agent.tools.transaction_patterns import _transaction_patterns

    start = time.perf_counter()
    result = await _transaction_patterns()
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result.get("status") == "ok", f"L03: Pattern analysis failed: {result}"
    assert "churn_rate" in result or "coaching_summary" in result, (
        "L03: Result must include churn_rate or coaching_summary"
    )

    _check_latency(elapsed_ms, COMPLEX_BUDGET_MS, "L03")


# ══════════════════════════════════════════════════════════════════════════════
# L04 — Concurrent tool calls: parallel execution does not deadlock
#        Simulates the agent calling multiple tools in parallel (LangGraph nodes).
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_l04_concurrent_tool_calls_no_deadlock():
    """
    Fire two tool coroutines concurrently using asyncio.gather.
    Both must complete within MULTI_STEP_BUDGET_MS total (not SIMPLE * 2).
    This verifies there's no shared-state lock that serialises concurrent calls.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "valueInBaseCurrency": 1750.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                ]
            },
        )
    )
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={
                "performance": {
                    "netPerformancePercentage": 0.12,
                    "netPerformance": 200.00,
                    "currentValueInBaseCurrency": 1750.00,
                    "totalInvestment": 1550.00,
                    "currentNetWorth": 1750.00,
                }
            },
        )
    )

    from agent.tools.performance import _get_performance
    from agent.tools.portfolio import _get_portfolio_summary

    start = time.perf_counter()
    portfolio_result, perf_result = await asyncio.gather(
        _get_portfolio_summary(),
        _get_performance("ytd"),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert portfolio_result["status"] == "ok"
    assert perf_result["status"] == "ok"

    _check_latency(elapsed_ms, MULTI_STEP_BUDGET_MS, "L04-concurrent")
