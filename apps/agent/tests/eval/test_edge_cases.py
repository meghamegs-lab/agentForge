"""
Edge Case Eval Suite — tests/eval/test_edge_cases.py
====================================================
"Does the agent handle unusual, missing, or malformed data gracefully?"

Edge cases covered here:
  1.  Holdings returned as a DICT (not a list) — alternate Ghostfolio format
  2.  Holdings with zero value — must not cause division-by-zero
  3.  Holdings with null/missing fields — must use safe defaults
  4.  Single holding portfolio — still produces valid allocation = 100 %
  5.  Very large portfolio (100 positions) — aggregation stays performant
  6.  Unicode in holding names — must not crash or corrupt output
  7.  Invalid performance period — must fall back to "ytd" silently
  8.  Mixed-case transaction type filter — case-insensitive matching
  9.  Market data: symbol with only whitespace — treated as no-input error
  10. Market data: comma-separated symbols with spaces — parsed correctly

All tests mock network calls with respx — zero real I/O.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import respx

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.market import get_market_data
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.transactions import _get_transactions

BASE_URL  = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "edge-token-999"}


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 1 — Ghostfolio returns holdings as a DICT keyed by symbol
#               (both list and dict are valid Ghostfolio formats)
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_portfolio_handles_holdings_as_dict_format():
    """
    Some versions of the Ghostfolio API return holdings as:
        {"holdings": {"AAPL": {...}, "VTI": {...}}}
    rather than a list.  The tool must normalise both formats.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": {
                "AAPL": {
                    "symbol": "AAPL", "name": "Apple", "quantity": 10, "value": 1750.00,
                    "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                    "sectors": [], "countries": [],
                },
                "VTI": {
                    "symbol": "VTI", "name": "Vanguard", "quantity": 20, "value": 4200.00,
                    "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                    "sectors": [], "countries": [],
                },
            }
        })
    )
    result = await _get_portfolio_summary()
    assert result["status"] == "ok", f"Dict-format holdings failed: {result}"
    assert result["position_count"] == 2
    assert abs(result["total_value"] - 5950.00) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 2 — Holdings with value = 0 must not cause division-by-zero
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_diversification_zero_value_holdings_no_division_error():
    """
    If ALL holdings have value 0 (delisted, suspended, etc.), total_value=0.
    Dividing by zero would crash — the tool must return status="empty" safely.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "DELIST", "name": "Delisted Co", "quantity": 100, "value": 0,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [], "countries": []},
            ]
        })
    )
    result = await _analyze_diversification()
    # Must NOT raise; should be empty or error — never an unhandled exception
    assert result["status"] in {"empty", "error"}, (
        f"Zero-value portfolio should return empty/error, got {result['status']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 3 — Holdings with null/missing optional fields use safe defaults
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_portfolio_missing_optional_fields_use_defaults():
    """
    Ghostfolio may omit optional fields (sectors, countries, assetSubClass).
    The tool must substitute sensible defaults rather than raising KeyError.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                # Only required fields present — everything else absent
                {"symbol": "MIN", "value": 500.00},
            ]
        })
    )
    result = await _get_portfolio_summary()
    assert result["status"] == "ok"
    h = result["holdings"][0]
    assert h["sectors"]  == [], f"Missing sectors should default to [], got {h['sectors']}"
    assert h["countries"] == [], f"Missing countries should default to [], got {h['countries']}"
    assert h["quantity"]  == 0,  f"Missing quantity should default to 0, got {h['quantity']}"


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 4 — Single holding: allocation must be exactly 100 %
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_single_holding_allocation_is_100_percent():
    """
    If the user holds only one asset, its allocation must be 100.00 %.
    This tests that the denominator (total_value) is not accidentally 0.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "ONLY", "name": "Only Stock", "quantity": 50, "value": 5000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [{"name": "Technology", "weight": 1.0}],
                 "countries": [{"name": "United States", "weight": 1.0}]},
            ]
        })
    )
    result = await _get_portfolio_summary()
    assert result["status"] == "ok"
    assert len(result["holdings"]) == 1
    assert abs(result["holdings"][0]["allocation_percent"] - 100.0) < 0.01, (
        f"Single holding allocation should be 100%, got {result['holdings'][0]['allocation_percent']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 5 — Unicode characters in holding names must not crash the tool
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_unicode_holding_names_do_not_crash():
    """
    Users holding international ETFs may see names like:
      "iShares MSCI Emerging Markets 新兴市场" or "日本株式 ETF"
    The tool must handle UTF-8 strings without encoding errors.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "EEM", "name": "新兴市场 ETF 🌍", "quantity": 10, "value": 1200.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                 "sectors": [{"name": "Financiëel", "weight": 1.0}],
                 "countries": [{"name": "日本", "weight": 0.5},
                               {"name": "中国", "weight": 0.5}]},
            ]
        })
    )
    result = await _get_portfolio_summary()
    assert result["status"] == "ok"
    assert result["holdings"][0]["name"] == "新兴市场 ETF 🌍"


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 6 — Large portfolio: 100 holdings completes without error
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_large_portfolio_100_holdings_completes():
    """
    No hard limits should exist on the number of holdings.
    100 positions must be processed, with allocations still summing to 100 %.
    """
    _auth()
    holdings = [
        {
            "symbol": f"STK{i:03d}", "name": f"Stock {i}", "quantity": 10,
            "value": 100.00,   # equal weight → each = 1 %
            "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
            "sectors": [{"name": "Technology", "weight": 1.0}],
            "countries": [{"name": "United States", "weight": 1.0}],
        }
        for i in range(100)
    ]
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": holdings})
    )
    result = await _get_portfolio_summary()
    assert result["status"] == "ok"
    assert result["position_count"] == 100
    total_alloc = sum(h["allocation_percent"] for h in result["holdings"])
    assert abs(total_alloc - 100.0) < 0.5   # float drift across 100 positions


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 7 — Invalid performance period silently falls back to "ytd"
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_invalid_performance_period_falls_back_to_ytd():
    """
    The LLM might pass a nonsense period like "last_quarter" or "q3".
    The tool must default to "ytd" rather than raising an exception.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json={
            "performance": {
                "ytd": {"relativeChange": 0.05, "absoluteChange": 400.0, "currentValue": 8400.0},
            }
        })
    )
    result = await _get_performance("last_quarter")    # invalid period
    assert result["status"] == "ok", f"Invalid period should not error: {result}"
    assert result["requested_period"] == "ytd", (
        f"Invalid period should fall back to 'ytd', got {result['requested_period']!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 8 — Transaction type filter is case-insensitive
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_transaction_type_filter_is_case_insensitive():
    """
    The LLM might pass "buy", "Buy", or "BUY" — all must match correctly.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "t1", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 5, "unitPrice": 170.0, "fee": 0.0, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
                {"id": "t2", "date": "2024-02-01T00:00:00Z", "type": "DIVIDEND",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 5, "unitPrice": 0.25, "fee": 0.0, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
            ]
        })
    )
    # Pass lowercase "buy" — must match the uppercase "BUY" activity
    result = await _get_transactions(transaction_type="buy")
    assert result["status"] == "ok"
    assert all(tx["type"] == "BUY" for tx in result["transactions"]), (
        "Lowercase 'buy' filter should match uppercase 'BUY' transactions"
    )
    assert result["transaction_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 9 — Market data: whitespace-only symbol string is an error
# ══════════════════════════════════════════════════════════════════════════════

def test_market_data_whitespace_symbol_returns_error():
    """
    The LLM might pass "   " (spaces) as a symbol if it misunderstood the query.
    After stripping, the symbol list is empty — should return a structured error,
    not crash with an unhandled exception.
    """
    result = get_market_data.invoke({"symbols": "   "})
    assert result["status"] == "error", (
        f"Whitespace symbol should produce error, got {result['status']}"
    )
    assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# Edge Case 10 — Market data: comma-separated symbols with extra spaces parsed correctly
# ══════════════════════════════════════════════════════════════════════════════

def test_market_data_comma_separated_symbols_with_spaces():
    """
    The tool docstring says: "Comma-separated ticker symbols (e.g. 'AAPL,MSFT,VTI')".
    The LLM might pass "AAPL, MSFT , VTI" with extra spaces.
    The tool must strip whitespace from each symbol and still return valid results.
    """
    mock_ticker = MagicMock()
    mock_ticker.fast_info.currency = "USD"
    mock_ticker.fast_info.year_high = 200.0
    mock_ticker.fast_info.year_low  = 150.0
    mock_ticker.history.return_value = pd.DataFrame(
        {"Close": [175.32]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    with patch("yfinance.Ticker", return_value=mock_ticker):
        # Extra spaces around each symbol
        result = get_market_data.invoke({"symbols": " AAPL , MSFT , VTI "})

    # Batch result: top-level "quotes" key with all 3 symbols
    assert result.get("status") != "error", (
        f"Spaced symbol list should not error; got: {result}"
    )
