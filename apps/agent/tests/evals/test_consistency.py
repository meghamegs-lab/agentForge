"""
evals/test_consistency.py — Consistency Eval Suite (v2)
=============================================================
Eval IDs: CON01–CON04

"Does the same input always produce the same output structure?"

Tests in this file:
  CON01 — Identical mock input produces identical output (idempotency)
  CON02 — Holdings are always sorted by value descending (stable sort)
  CON03 — Transactions are always sorted newest-first (stable sort)
  CON04 — Performance response includes all required fields even when values are 0

Consistency tests ensure the agent's outputs are deterministic and
structurally stable, which is required for:
  - Reliable downstream processing
  - Reproducible evaluation runs
  - Predictable user experience

All network calls are mocked with respx — zero real I/O.
"""

from __future__ import annotations

import httpx
import respx

from agent.config import settings
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.transactions import _get_transactions

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "consistency-v2-token"}

DOLLAR_TOLERANCE = 0.01
PCT_TOLERANCE = 0.01


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


# ══════════════════════════════════════════════════════════════════════════════
# CON01 — Identical mock input produces identical output (idempotency)
#          Run the tool twice with the same mocked API response.
#          Both results must have identical structure and values.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_con01_portfolio_summary_is_idempotent():
    """
    Two calls with identical mocked responses must produce identical results:
    same total_value, same position_count, same holdings order, same allocations.
    """
    holdings_mock = {
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
    }

    # First run
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_mock)
    )
    result_1 = await _get_portfolio_summary()

    # Second run — re-mock the same responses
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_mock)
    )
    result_2 = await _get_portfolio_summary()

    assert result_1["status"] == result_2["status"] == "ok"
    assert abs(result_1["total_value"] - result_2["total_value"]) < DOLLAR_TOLERANCE, (
        f"CON01: total_value differs between runs: {result_1['total_value']} vs {result_2['total_value']}"
    )
    assert result_1["position_count"] == result_2["position_count"], (
        "CON01: position_count differs between runs"
    )
    # Holdings order must be identical
    symbols_1 = [h["symbol"] for h in result_1["holdings"]]
    symbols_2 = [h["symbol"] for h in result_2["holdings"]]
    assert symbols_1 == symbols_2, (
        f"CON01: Holdings order differs between runs: {symbols_1} vs {symbols_2}"
    )
    # Allocation percentages must be identical
    allocs_1 = [h["allocation_percent"] for h in result_1["holdings"]]
    allocs_2 = [h["allocation_percent"] for h in result_2["holdings"]]
    assert allocs_1 == allocs_2, (
        f"CON01: Allocation percentages differ between runs: {allocs_1} vs {allocs_2}"
    )


@respx.mock
async def test_con01_performance_is_idempotent():
    """
    Two calls to _get_performance('ytd') with identical mocked responses
    must produce identical relative_change_pct and absolute_change.
    """
    perf_mock = {
        "performance": {
            "netPerformancePercentage": 0.1234,
            "netPerformance": 987.65,
            "currentValueInBaseCurrency": 8987.65,
            "totalInvestment": 8000.00,
            "currentNetWorth": 8987.65,
        }
    }

    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json=perf_mock)
    )
    r1 = await _get_performance("ytd")

    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json=perf_mock)
    )
    r2 = await _get_performance("ytd")

    assert r1["performance"]["relative_change_pct"] == r2["performance"]["relative_change_pct"], (
        "CON01: relative_change_pct must be identical across runs"
    )
    assert r1["performance"]["absolute_change"] == r2["performance"]["absolute_change"]
    assert r1["requested_period"] == r2["requested_period"] == "ytd"


# ══════════════════════════════════════════════════════════════════════════════
# CON02 — Holdings are always sorted by value descending (stable sort)
#          Regardless of the order Ghostfolio returns holdings, the tool output
#          must always be sorted from highest to lowest value.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_con02_holdings_sorted_by_value_descending():
    """
    Ghostfolio returns holdings in arbitrary order (AMZN first, then AAPL etc).
    The tool must sort them descending: VTI($4200) > MSFT($2050) > AAPL($1750) > AMZN($520).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    # Deliberately in wrong order to test sorting
                    {
                        "symbol": "AMZN",
                        "name": "Amazon",
                        "quantity": 3,
                        "valueInBaseCurrency": 520.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
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
                    {
                        "symbol": "MSFT",
                        "name": "Microsoft Corporation",
                        "quantity": 5,
                        "valueInBaseCurrency": 2050.00,
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
    result = await _get_portfolio_summary()
    holdings = result["holdings"]

    expected_order = ["VTI", "MSFT", "AAPL", "AMZN"]
    actual_order = [h["symbol"] for h in holdings]
    assert actual_order == expected_order, (
        f"CON02: Holdings not sorted by value descending. "
        f"Expected {expected_order}, got {actual_order}"
    )

    # Verify strictly descending
    for i in range(len(holdings) - 1):
        assert holdings[i]["current_value"] >= holdings[i + 1]["current_value"], (
            f"CON02: Sort order violation at index {i}: "
            f"{holdings[i]['symbol']}={holdings[i]['current_value']} < "
            f"{holdings[i + 1]['symbol']}={holdings[i + 1]['current_value']}"
        )


@respx.mock
async def test_con02_sort_stable_for_equal_values():
    """
    When two holdings have identical values, the sort must be stable
    (preserve relative order from the API response).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    {
                        "symbol": "A1",
                        "name": "Asset One",
                        "quantity": 5,
                        "valueInBaseCurrency": 1000.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "A2",
                        "name": "Asset Two",
                        "quantity": 5,
                        "valueInBaseCurrency": 1000.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "A3",
                        "name": "Asset Three",
                        "quantity": 5,
                        "valueInBaseCurrency": 2000.00,
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
    result = await _get_portfolio_summary()
    holdings = result["holdings"]

    # A3 must be first (highest value)
    assert holdings[0]["symbol"] == "A3", (
        f"CON02: A3 ($2000) must be first, got {holdings[0]['symbol']}"
    )
    # A1 and A2 tied at $1000 — both must be present in positions 1 and 2
    remaining = {h["symbol"] for h in holdings[1:]}
    assert "A1" in remaining and "A2" in remaining


# ══════════════════════════════════════════════════════════════════════════════
# CON03 — Transactions sorted newest-first (descending date)
#          The tool must sort transactions by date descending so the most
#          recent transaction is always at index 0.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_con03_transactions_sorted_newest_first():
    """
    Three transactions with dates 2024-01, 2024-06, 2024-09.
    Must be returned newest→oldest: tx-003, tx-002, tx-001.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(
            200,
            json={
                "activities": [
                    # Intentionally oldest first to test the sort
                    {
                        "id": "tx-001",
                        "date": "2024-01-10T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft"},
                        "quantity": 5,
                        "unitPrice": 400.00,
                        "fee": 4.99,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "tx-002",
                        "date": "2024-06-01T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {
                            "symbol": "VTI",
                            "name": "Vanguard Total Stock Market ETF",
                        },
                        "quantity": 20,
                        "unitPrice": 210.00,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "tx-003",
                        "date": "2024-09-01T00:00:00.000Z",
                        "type": "DIVIDEND",
                        "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                        "quantity": 10,
                        "unitPrice": 0.25,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                ]
            },
        )
    )
    result = await _get_transactions()

    assert result["status"] == "ok"
    transactions = result["transactions"]
    assert len(transactions) == 3

    assert transactions[0]["id"] == "tx-003", (
        f"CON03: Newest transaction (2024-09-01) must be first. Got: {transactions[0]['id']}"
    )
    assert transactions[1]["id"] == "tx-002", (
        f"CON03: Second newest (2024-06-01) must be second. Got: {transactions[1]['id']}"
    )
    assert transactions[2]["id"] == "tx-001", (
        f"CON03: Oldest transaction (2024-01-10) must be last. Got: {transactions[2]['id']}"
    )


@respx.mock
async def test_con03_sort_correct_with_same_day_transactions():
    """
    Two transactions on the same day — sort must be stable (preserve API order).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(
            200,
            json={
                "activities": [
                    {
                        "id": "same-day-a",
                        "date": "2024-07-04T10:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {"symbol": "AAPL"},
                        "quantity": 5,
                        "unitPrice": 180.00,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "same-day-b",
                        "date": "2024-07-04T14:00:00.000Z",
                        "type": "SELL",
                        "SymbolProfile": {"symbol": "AAPL"},
                        "quantity": 3,
                        "unitPrice": 181.00,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                    {
                        "id": "older-tx",
                        "date": "2024-01-15T00:00:00.000Z",
                        "type": "BUY",
                        "SymbolProfile": {"symbol": "VTI"},
                        "quantity": 10,
                        "unitPrice": 210.00,
                        "fee": 0.00,
                        "currency": "USD",
                        "account": {"name": "Brokerage"},
                    },
                ]
            },
        )
    )
    result = await _get_transactions()
    transactions = result["transactions"]

    # The July 4 transactions must come before the January 15 one
    july_ids = {"same-day-a", "same-day-b"}
    assert transactions[0]["id"] in july_ids, "CON03: July transactions must precede January"
    assert transactions[1]["id"] in july_ids, "CON03: Both July transactions before January"
    assert transactions[2]["id"] == "older-tx", "CON03: January transaction must be last"


# ══════════════════════════════════════════════════════════════════════════════
# CON04 — Performance response includes ALL required fields even when values are 0
#          A flat day (0% return, $0 gain) must still return all fields populated
#          with 0.0 — never omit fields because the value happens to be falsy.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_con04_required_fields_present_when_performance_is_zero():
    """
    Flat performance (0% gain, $0 absolute) must still return all required fields.
    Zero values must NOT be omitted — they are valid data.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={
                "performance": {
                    "netPerformancePercentage": 0.0,
                    "netPerformance": 0.0,
                    "currentValueInBaseCurrency": 8000.00,
                    "totalInvestment": 8000.00,
                    "currentNetWorth": 8000.00,
                }
            },
        )
    )
    result = await _get_performance("1d")

    # Required top-level keys
    required_top = ["status", "requested_period", "performance", "data_timestamp", "source"]
    for key in required_top:
        assert key in result, (
            f"CON04: Required top-level key '{key}' missing when performance is zero"
        )

    # Required performance sub-keys — all must exist even if value is 0.0
    required_perf = [
        "relative_change_pct",
        "absolute_change",
        "current_value",
        "total_investment",
        "net_worth",
    ]
    for key in required_perf:
        assert key in result["performance"], (
            f"CON04: Required performance key '{key}' missing when value is 0.0"
        )

    # Zero values must be preserved, not omitted
    assert result["performance"]["relative_change_pct"] == 0.0, (
        "CON04: relative_change_pct must be 0.0 (not omitted or None)"
    )
    assert result["performance"]["absolute_change"] == 0.0, (
        "CON04: absolute_change must be 0.0 (not omitted or None)"
    )
    assert result["status"] == "ok", "CON04: Zero performance must still return status='ok'"
