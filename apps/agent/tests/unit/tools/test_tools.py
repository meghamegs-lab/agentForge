"""
Unit tests for all 5 agent tools.
All HTTP calls are mocked with respx — no real network requests.

Imports the private _impl functions (not the @tool wrappers) so tests
run the business logic directly without LangChain overhead.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import respx

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.market import get_market_data
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.transactions import _get_transactions

# ── Helpers ───────────────────────────────────────────────────────────────────

AUTH_RESPONSE = {"authToken": "test-bearer-token-123"}
BASE_URL = settings.ghostfolio_base_url.rstrip("/")  # reads from .env / settings


def mock_auth(base_url: str = BASE_URL) -> None:
    """Register the auth endpoint mock (call inside @respx.mock blocks)."""
    respx.post(f"{base_url}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESPONSE)
    )


# ── Tool 1: Portfolio Summary ──────────────────────────────────────────────────


class TestGetPortfolioSummary:
    @respx.mock
    async def test_returns_holdings_with_allocation_percentages(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _get_portfolio_summary()
        assert result["status"] == "ok"
        assert len(result["holdings"]) == 3
        total_alloc = sum(h["allocation_percent"] for h in result["holdings"])
        assert abs(total_alloc - 100.0) < 0.1

    @respx.mock
    async def test_total_value_matches_sum_of_positions(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _get_portfolio_summary()
        summed = sum(h["current_value"] for h in result["holdings"])
        assert abs(result["total_value"] - summed) < 0.01

    @respx.mock
    async def test_empty_portfolio_returns_empty_state_not_error(self):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json={"holdings": []})
        )
        result = await _get_portfolio_summary()
        assert result["status"] == "empty"
        assert result["total_value"] == 0.0
        assert isinstance(result["holdings"], list)

    @respx.mock
    async def test_network_error_returns_structured_error(self):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        result = await _get_portfolio_summary()
        assert result["status"] == "error"
        assert "holdings" in result

    @respx.mock
    async def test_holdings_sorted_by_value_descending(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _get_portfolio_summary()
        values = [h["current_value"] for h in result["holdings"]]
        assert values == sorted(values, reverse=True)

    @respx.mock
    async def test_data_timestamp_present(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _get_portfolio_summary()
        assert "data_timestamp" in result


# ── Tool 2: Performance ────────────────────────────────────────────────────────


class TestGetPerformance:
    @respx.mock
    async def test_returns_ytd_performance_fields(self, sample_performance_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
            return_value=httpx.Response(200, json=sample_performance_response)
        )
        result = await _get_performance("ytd")
        assert result["status"] == "ok"
        assert result["requested_period"] == "ytd"
        perf = result["performance"]
        assert "relative_change_pct" in perf
        assert "absolute_change" in perf

    @respx.mock
    async def test_relative_change_is_percentage(self, sample_performance_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
            return_value=httpx.Response(200, json=sample_performance_response)
        )
        result = await _get_performance("ytd")
        # 0.1234 raw → 12.34%
        assert abs(result["performance"]["relative_change_pct"] - 12.34) < 0.01

    @respx.mock
    async def test_invalid_range_falls_back_to_ytd(self, sample_performance_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
            return_value=httpx.Response(200, json=sample_performance_response)
        )
        result = await _get_performance("invalid_range")
        assert result["status"] == "ok"
        assert result["requested_period"] == "ytd"


# ── Tool 3: Transactions ───────────────────────────────────────────────────────


class TestGetTransactions:
    @respx.mock
    async def test_returns_typed_transactions(self, sample_orders_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json=sample_orders_response)
        )
        result = await _get_transactions()
        assert result["status"] == "ok"
        assert result["transaction_count"] == 3
        types = {tx["type"] for tx in result["transactions"]}
        assert "BUY" in types
        assert "DIVIDEND" in types

    @respx.mock
    async def test_fee_sum_is_correct(self, sample_orders_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json=sample_orders_response)
        )
        result = await _get_transactions()
        assert abs(result["summary"]["total_fees_paid"] - 4.99) < 0.01

    @respx.mock
    async def test_type_filter_works(self, sample_orders_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json=sample_orders_response)
        )
        result = await _get_transactions(transaction_type="BUY")
        assert all(tx["type"] == "BUY" for tx in result["transactions"])

    @respx.mock
    async def test_empty_transactions_handled(self):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json={"activities": []})
        )
        result = await _get_transactions()
        assert result["status"] == "empty"
        assert isinstance(result["transactions"], list)


# ── Tool 4: Diversification ────────────────────────────────────────────────────


class TestAnalyzeDiversification:
    @respx.mock
    async def test_sector_weights_sum_to_100(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _analyze_diversification()
        total_pct = sum(s["percent"] for s in result["sector_breakdown"])
        assert abs(total_pct - 100.0) < 1.0  # allow minor float drift

    @respx.mock
    async def test_flags_concentration_above_threshold(self):
        # Holdings as a LIST — matching real Ghostfolio API format
        concentrated = {
            "holdings": [
                {
                    "symbol": "BIG",
                    "name": "Big Stock",
                    "quantity": 100,
                    "value": 8000.0,
                    "currency": "USD",
                    "assetClass": "EQUITY",
                    "assetSubClass": "STOCK",
                    "sectors": [{"name": "Technology", "weight": 1.0}],
                    "countries": [{"name": "United States", "weight": 1.0}],
                },
                {
                    "symbol": "SMALL",
                    "name": "Small Stock",
                    "quantity": 10,
                    "value": 2000.0,
                    "currency": "USD",
                    "assetClass": "EQUITY",
                    "assetSubClass": "STOCK",
                    "sectors": [{"name": "Healthcare", "weight": 1.0}],
                    "countries": [{"name": "United States", "weight": 1.0}],
                },
            ]
        }
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=concentrated)
        )
        result = await _analyze_diversification()
        assert len(result["concentration_flags"]) > 0
        assert result["risk_summary"]["needs_rebalancing"] is True

    @respx.mock
    async def test_diversification_score_between_0_and_100(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _analyze_diversification()
        assert 0 <= result["diversification_score"] <= 100

    @respx.mock
    async def test_grade_assigned(self, sample_holdings_response):
        mock_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=sample_holdings_response)
        )
        result = await _analyze_diversification()
        assert result["diversification_grade"] in {"A", "B", "C", "D"}


# ── Tool 5: Market Data ────────────────────────────────────────────────────────


class TestGetMarketData:
    async def test_returns_price_for_valid_symbol(self):
        mock_ticker = MagicMock()
        mock_ticker.fast_info.currency = "USD"
        mock_ticker.fast_info.year_high = 200.0
        mock_ticker.fast_info.year_low = 150.0
        import pandas as pd

        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [175.32]}, index=pd.date_range("2024-01-01", periods=1)
        )
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await get_market_data.ainvoke({"symbols": "AAPL"})
        assert result["status"] == "ok"
        assert abs(result["current_price"] - 175.32) < 0.01
        assert result["symbol"] == "AAPL"

    async def test_invalid_symbol_returns_error_not_exception(self):
        mock_ticker = MagicMock()
        import pandas as pd

        mock_ticker.history.return_value = pd.DataFrame()  # empty = not found
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await get_market_data.ainvoke({"symbols": "FAKEXYZ123"})
        # Market client returns price_unavailable (not error) when yfinance finds no data
        assert result["status"] == "price_unavailable"
        assert result.get("symbol") == "FAKEXYZ123"

    async def test_data_includes_timestamp(self):
        mock_ticker = MagicMock()
        import pandas as pd

        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1)
        )
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await get_market_data.ainvoke({"symbols": "AAPL"})
        assert "data_timestamp" in result

    async def test_empty_symbols_returns_error(self):
        result = await get_market_data.ainvoke({"symbols": ""})
        assert result["status"] == "error"
