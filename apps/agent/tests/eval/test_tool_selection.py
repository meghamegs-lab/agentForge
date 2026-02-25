"""
Tool Selection Eval Suite — tests/eval/test_tool_selection.py
=============================================================
"Does the agent pick the RIGHT tool for each user query?"

Tool selection has two dimensions we can test without a live LLM:

  A. DESCRIPTOR COVERAGE — the tool docstring contains every trigger keyword
     the LLM might use to decide which tool to call. If a keyword is missing
     from the docstring, the LLM is unlikely to select the tool.

  B. DOMAIN BOUNDARY — when called with realistic inputs, each tool returns
     data in its own domain and does NOT bleed into another tool's domain.
     Example: get_portfolio_summary should NOT return performance metrics.

We also include PARAMETER MAPPING tests — given a natural-language intent,
the correct parameter values are passed (e.g. "this year" → date_range="ytd").

All Ghostfolio / yfinance network calls are mocked.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from agent.config import settings
from agent.tools.diversification import analyze_diversification
from agent.tools.market import get_market_data
from agent.tools.performance import _get_performance, get_performance
from agent.tools.portfolio import _get_portfolio_summary, get_portfolio_summary
from agent.tools.transactions import _get_transactions, get_transactions

BASE_URL  = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "select-token-xyz"}

# ── fixture helpers ────────────────────────────────────────────────────────────

def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )

_HOLDINGS_BODY = {
    "holdings": [
        {"symbol": "AAPL", "name": "Apple", "quantity": 10, "value": 1750.00,
         "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
         "sectors": [{"name": "Technology", "weight": 1.0}],
         "countries": [{"name": "United States", "weight": 1.0}]},
    ]
}

_PERF_BODY = {
    "performance": {
        "ytd": {"relativeChange": 0.12, "absoluteChange": 900.0, "currentValue": 8900.0},
        "1y":  {"relativeChange": 0.22, "absoluteChange": 1600.0, "currentValue": 8900.0},
    }
}

_ORDERS_BODY = {
    "activities": [
        {"id": "t1", "date": "2024-06-01T00:00:00Z", "type": "BUY",
         "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
         "quantity": 10, "unitPrice": 170.0, "fee": 4.99, "currency": "USD",
         "Account": {"name": "Brokerage"}},
    ]
}


# ══════════════════════════════════════════════════════════════════════════════
# A. DESCRIPTOR COVERAGE — trigger keywords present in each tool docstring
# ══════════════════════════════════════════════════════════════════════════════

class TestToolDescriptors:
    """
    The LLM reads each tool's docstring to decide which tool to call.
    If critical trigger phrases are absent, the LLM may pick the wrong tool.
    These tests act as a contract: change the docstring → test will catch it.
    """

    def _doc(self, tool_fn) -> str:
        # LangChain @tool wraps the function — description lives in .description,
        # falling back to __doc__ for plain functions.
        desc = getattr(tool_fn, "description", None) or tool_fn.__doc__ or ""
        return desc.lower()

    # Selection 1 ─ portfolio tool docstring covers composition queries
    def test_portfolio_tool_covers_composition_queries(self):
        doc = self._doc(get_portfolio_summary)
        triggers = ["holdings", "allocation", "portfolio", "positions", "value"]
        missing = [t for t in triggers if t not in doc]
        assert not missing, (
            f"get_portfolio_summary docstring is missing trigger keywords: {missing}"
        )

    # Selection 2 ─ performance tool docstring covers returns queries
    def test_performance_tool_covers_returns_queries(self):
        doc = self._doc(get_performance)
        triggers = ["performance", "return", "gain", "loss", "period"]
        missing = [t for t in triggers if t not in doc]
        assert not missing, (
            f"get_performance docstring is missing trigger keywords: {missing}"
        )

    # Selection 3 ─ transactions tool docstring covers history queries
    def test_transactions_tool_covers_history_queries(self):
        doc = self._doc(get_transactions)
        triggers = ["transaction", "history", "fee", "dividend", "buy", "sell"]
        missing = [t for t in triggers if t not in doc]
        assert not missing, (
            f"get_transactions docstring is missing trigger keywords: {missing}"
        )

    # Selection 4 ─ diversification tool docstring covers risk queries
    def test_diversification_tool_covers_risk_queries(self):
        doc = self._doc(analyze_diversification)
        triggers = ["diversification", "concentration", "sector", "geographic", "rebalancing"]
        missing = [t for t in triggers if t not in doc]
        assert not missing, (
            f"analyze_diversification docstring is missing trigger keywords: {missing}"
        )

    # Selection 5 ─ market data tool docstring covers price lookup queries
    def test_market_data_tool_covers_price_queries(self):
        doc = self._doc(get_market_data)
        triggers = ["price", "stock", "symbol", "market", "52"]
        missing = [t for t in triggers if t not in doc]
        assert not missing, (
            f"get_market_data docstring is missing trigger keywords: {missing}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# B. DOMAIN BOUNDARY — each tool returns only its own domain of data
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainBoundary:
    """
    When the user asks "what is my portfolio worth?", get_portfolio_summary
    must be the correct tool — and its output must contain holdings/value
    fields, NOT performance metrics.  Conversely, get_performance must NOT
    return holdings.
    """

    # Selection 6 ─ portfolio summary does NOT contain performance metrics
    @respx.mock
    async def test_portfolio_tool_does_not_return_performance_fields(self):
        _auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=_HOLDINGS_BODY)
        )
        result = await _get_portfolio_summary()
        performance_keys = {"relative_change_pct", "absolute_change", "requested_period"}
        overlap = performance_keys & set(result.keys())
        assert not overlap, (
            f"Portfolio tool leaked performance fields: {overlap}"
        )

    # Selection 7 ─ performance tool does NOT contain holdings list
    @respx.mock
    async def test_performance_tool_does_not_return_holdings(self):
        _auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
            return_value=httpx.Response(200, json=_PERF_BODY)
        )
        result = await _get_performance("ytd")
        assert "holdings" not in result, (
            "Performance tool must not return holdings — that belongs to portfolio tool"
        )

    # Selection 8 ─ transactions tool does NOT contain sector breakdown
    @respx.mock
    async def test_transactions_tool_does_not_return_sector_data(self):
        _auth()
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json=_ORDERS_BODY)
        )
        result = await _get_transactions()
        diversification_keys = {"sector_breakdown", "diversification_score", "concentration_flags"}
        overlap = diversification_keys & set(result.keys())
        assert not overlap, (
            f"Transactions tool leaked diversification fields: {overlap}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# C. PARAMETER MAPPING — natural-language intent → correct parameter values
# ══════════════════════════════════════════════════════════════════════════════

class TestParameterMapping:
    """
    These tests verify that parameter constants used by the LLM
    (e.g. date_range="ytd") are accepted and handled correctly.
    The LLM infers parameter values from the docstring; wrong values
    would cause silent failures or fall-throughs to defaults.
    """

    # Selection 9 ─ valid date range strings accepted without error
    @pytest.mark.parametrize("period,expected_period", [
        ("1d",  "1d"),
        ("wtd", "wtd"),
        ("mtd", "mtd"),
        ("ytd", "ytd"),
        ("1y",  "1y"),
        ("5y",  "5y"),
        ("max", "max"),
    ])
    @respx.mock
    async def test_all_valid_performance_periods_are_accepted(self, period, expected_period):
        """
        'ytd', '1y', '5y', 'max', etc. are all listed in the tool docstring.
        Passing any of them must succeed and echo back the period in the response.
        """
        _auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
            return_value=httpx.Response(200, json=_PERF_BODY)
        )
        result = await _get_performance(period)
        assert result["status"] == "ok"
        # The tool normalises unknown periods to "ytd"; known ones stay as-is
        assert result["requested_period"] == expected_period

    # Selection 10 ─ transaction_type filter maps to correct enum values
    @pytest.mark.parametrize("tx_type", ["BUY", "SELL", "DIVIDEND", "FEE"])
    @respx.mock
    async def test_transaction_type_filter_matches_docstring_values(self, tx_type):
        """
        The docstring lists 'BUY', 'SELL', 'DIVIDEND', 'FEE', 'INTEREST' as valid types.
        Passing each must not cause an error — even if the mock returns no matching rows.
        """
        _auth()
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json=_ORDERS_BODY)
        )
        result = await _get_transactions(transaction_type=tx_type)
        # Either "ok" (match found) or "empty" (no match) — never "error"
        assert result["status"] in {"ok", "empty"}, (
            f"Unexpected status for transaction_type={tx_type!r}: {result['status']}"
        )
