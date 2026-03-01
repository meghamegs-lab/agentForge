"""
evalsNew/test_tool_execution_v2.py — Tool Execution Eval Suite (v2)
====================================================================
Eval IDs: TE01–TE07

"Do tool calls succeed with valid params and fail gracefully with invalid ones?"

Tests in this file:
  TE01 — Valid date_range 'ytd' succeeds with correct status and fields
  TE02 — Invalid date_range '3y' silently falls back to 'ytd' (no crash)
  TE03 — Invalid date_from format returns graceful error dict, not Python traceback
  TE04 — Empty account_id returns all-accounts data (no filter applied)
  TE05 — Transaction type filter 'buy' (lowercase) matches BUY records (case-insensitive)
  TE06 — Empty symbols list to get_market_data returns structured error, not crash
  TE07 — Ghostfolio 401 Unauthorized → structured error dict, agent told to check token

All network calls are mocked with respx — zero real I/O.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from agent.config import settings
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.transactions import _get_transactions

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "tool-exec-v2-token"}


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


# ══════════════════════════════════════════════════════════════════════════════
# TE01 — Valid date_range 'ytd' succeeds
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_te01_valid_date_range_ytd_succeeds():
    """
    Calling _get_performance('ytd') with a valid, mocked Ghostfolio response
    must return status='ok' with all required fields populated.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json={
            "performance": {
                "netPerformancePercentage": 0.0821,
                "netPerformance": 610.00,
                "currentValueInBaseCurrency": 8610.00,
                "totalInvestment": 8000.00,
                "currentNetWorth": 8610.00,
            },
            "hasErrors": False,
        })
    )
    result = await _get_performance("ytd")

    assert result["status"] == "ok", f"TE01: Expected ok, got: {result}"
    assert result["requested_period"] == "ytd"

    required_perf_keys = ["relative_change_pct", "absolute_change", "current_value",
                          "total_investment", "net_worth"]
    for key in required_perf_keys:
        assert key in result["performance"], (
            f"TE01: Required performance key '{key}' missing from result"
        )

    required_top_keys = ["status", "requested_period", "performance", "data_timestamp", "source"]
    for key in required_top_keys:
        assert key in result, f"TE01: Required top-level key '{key}' missing"

    assert abs(result["performance"]["relative_change_pct"] - 8.21) <= 0.01


# ══════════════════════════════════════════════════════════════════════════════
# TE02 — Invalid date_range '3y' silently falls back to 'ytd'
#         The valid set is: '1d', 'wtd', 'mtd', 'ytd', '1y', '5y', 'max'.
#         Any unrecognised value must be silently coerced to 'ytd'.
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_te02_invalid_date_range_falls_back_to_ytd():
    """
    '3y' is not in the valid set.
    The tool must silently substitute 'ytd' and continue — no crash, no error status.
    """
    _auth()
    # The tool will call with ?range=ytd after the fallback
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json={
            "performance": {
                "netPerformancePercentage": 0.12,
                "netPerformance": 960.00,
                "currentValueInBaseCurrency": 8960.00,
                "totalInvestment": 8000.00,
                "currentNetWorth": 8960.00,
            }
        })
    )
    result = await _get_performance("3y")

    # Must not raise an exception — result must be a dict with status
    assert isinstance(result, dict), "TE02: Result must be a dict, not an exception"
    assert result.get("status") == "ok", (
        f"TE02: Invalid date_range '3y' must fall back to 'ytd' and return ok, got: {result}"
    )
    # The fallback period is what should be recorded
    assert result.get("requested_period") == "ytd", (
        f"TE02: requested_period must be 'ytd' (fallback), got: {result.get('requested_period')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TE03 — Invalid date_from format returns graceful error dict
#         'January 2024' is NOT ISO 8601. The tool must return {"status": "error"}
#         or {"status": "empty"} — NOT raise an unhandled exception.
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_te03_invalid_date_format_returns_graceful_error():
    """
    The Ghostfolio API rejects non-ISO dates and returns 400.
    The tool must catch this and return a structured error dict, not a traceback.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(400, json={"message": "Bad Request: invalid date format"})
    )
    result = await _get_transactions(date_from="January 2024")

    assert isinstance(result, dict), "TE03: Must return a dict on error, not raise exception"
    assert result.get("status") in ("error", "empty"), (
        f"TE03: Expected error or empty status for invalid date format, got: {result}"
    )
    # Must not expose Python tracebacks
    result_str = str(result)
    assert "Traceback" not in result_str, "TE03: Python traceback must not appear in result dict"
    assert "File " not in result_str, "TE03: File path traceback lines must not appear in result"


# ══════════════════════════════════════════════════════════════════════════════
# TE04 — Empty account_id returns data from ALL accounts
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_te04_empty_account_id_returns_all_accounts():
    """
    When account_id is '' (empty), the tool must pass no account filter to Ghostfolio,
    resulting in transactions from all accounts being returned.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "tx-001", "date": "2024-03-15T00:00:00.000Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                 "quantity": 10, "unitPrice": 170.00, "fee": 4.99,
                 "currency": "USD", "account": {"name": "Brokerage A"}},
                {"id": "tx-002", "date": "2024-06-01T00:00:00.000Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF"},
                 "quantity": 20, "unitPrice": 210.00, "fee": 0.00,
                 "currency": "USD", "account": {"name": "Brokerage B"}},
            ]
        })
    )
    result = await _get_transactions(account_id="")

    assert result["status"] == "ok"
    assert result["transaction_count"] == 2, (
        f"TE04: Expected 2 transactions (all accounts), got {result['transaction_count']}"
    )
    # Both accounts should be represented
    account_names = {t["account"] for t in result["transactions"]}
    assert "Brokerage A" in account_names and "Brokerage B" in account_names, (
        f"TE04: Expected both 'Brokerage A' and 'Brokerage B', got: {account_names}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TE05 — Transaction type filter 'buy' (lowercase) matches BUY records
#         The filter must be case-insensitive: 'buy' == 'BUY' == 'Buy'.
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_te05_transaction_type_filter_case_insensitive():
    """
    transaction_type='buy' must match records where type='BUY'.
    DIVIDEND records must be excluded.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "tx-001", "date": "2024-03-15T00:00:00.000Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                 "quantity": 10, "unitPrice": 170.00, "fee": 4.99,
                 "currency": "USD", "account": {"name": "Brokerage"}},
                {"id": "tx-003", "date": "2024-09-01T00:00:00.000Z", "type": "DIVIDEND",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                 "quantity": 10, "unitPrice": 0.25, "fee": 0.00,
                 "currency": "USD", "account": {"name": "Brokerage"}},
            ]
        })
    )
    result = await _get_transactions(transaction_type="buy")  # lowercase intentional

    assert result["status"] == "ok"
    assert result["transaction_count"] == 1, (
        f"TE05: Expected 1 BUY transaction, got {result['transaction_count']}. "
        f"Filter 'buy' must match 'BUY' case-insensitively."
    )
    assert result["transactions"][0]["type"] == "BUY"
    dividend_txs = [t for t in result["transactions"] if t["type"] == "DIVIDEND"]
    assert len(dividend_txs) == 0, "TE05: DIVIDEND transactions must be excluded by 'buy' filter"


@respx.mock
async def test_te05_transaction_type_filter_mixed_case():
    """Also verify 'Buy' (title case) works, not just 'buy' or 'BUY'."""
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "tx-s1", "date": "2024-04-01T00:00:00.000Z", "type": "SELL",
                 "SymbolProfile": {"symbol": "MSFT"}, "quantity": 3, "unitPrice": 400.00,
                 "fee": 4.99, "currency": "USD", "account": {"name": "Brokerage"}},
                {"id": "tx-b1", "date": "2024-05-01T00:00:00.000Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "VTI"}, "quantity": 5, "unitPrice": 210.00,
                 "fee": 0.00, "currency": "USD", "account": {"name": "Brokerage"}},
            ]
        })
    )
    result = await _get_transactions(transaction_type="Buy")  # title-case

    assert result["status"] == "ok"
    assert result["transaction_count"] == 1
    assert result["transactions"][0]["type"] == "BUY"


# ══════════════════════════════════════════════════════════════════════════════
# TE06 — Empty symbols list returns structured error, not crash
# ══════════════════════════════════════════════════════════════════════════════

async def test_te06_empty_symbols_returns_error():
    """
    get_market_data with symbols='' must return {"status": "error"}.
    No exception should propagate to the caller.
    """
    from agent.tools.market import get_market_data as _market_tool

    result = await _market_tool.ainvoke({"symbols": ""})

    assert isinstance(result, dict), "TE06: Must return dict, not raise exception"
    assert result.get("status") == "error", (
        f"TE06: Expected status='error' for empty symbols, got: {result}"
    )
    assert "error" in result or "message" in result, (
        "TE06: Error result must include an 'error' or 'message' field"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TE07 — Ghostfolio 401 Unauthorized returns structured error
#         The tool must catch HTTP 401 and return status='error' with error_code=401.
#         No stack trace, no re-raise.
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_te07_ghostfolio_401_returns_structured_error():
    """
    When Ghostfolio returns HTTP 401 (expired or invalid token), the tool
    must return {"status": "error", "error_code": 401} without crashing.
    The agent can then tell the user to check their Ghostfolio token.
    """
    # Mock the auth call to succeed (token exists but is invalid for the actual endpoint)
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )
    # Portfolio endpoint returns 401
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(401, json={"message": "Unauthorized"})
    )

    result = await _get_portfolio_summary()

    assert isinstance(result, dict), "TE07: Must return dict on 401"
    assert result.get("status") == "error", (
        f"TE07: Expected status='error' for 401 response, got: {result}"
    )
    # Must NOT contain fabricated holdings
    assert result.get("holdings", []) == [], (
        "TE07: On 401, holdings must be empty — not fabricated"
    )
    # Must NOT contain Python traceback text
    result_str = str(result)
    assert "Traceback" not in result_str
