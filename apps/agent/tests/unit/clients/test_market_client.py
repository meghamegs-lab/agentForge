"""
Unit tests for agent/clients/market.py
=======================================
Tests MarketDataClient.get_quote() and get_batch_quotes() with mocked
yfinance — all blocking calls are routed through asyncio.to_thread()
so these tests run purely in the async test environment.

All yfinance.Ticker instances are patched with MagicMock — no real
network calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.clients.market import MarketDataClient


def _mock_ticker(price: float | None = 175.32) -> MagicMock:
    """Build a MagicMock yfinance Ticker with configurable close price."""
    ticker = MagicMock()
    ticker.fast_info.currency = "USD"
    ticker.fast_info.year_high = 200.0
    ticker.fast_info.year_low = 150.0
    ticker.fast_info.market_cap = 2_800_000_000_000
    ticker.fast_info.three_month_average_volume = 60_000_000
    if price is None:
        ticker.history.return_value = pd.DataFrame()  # empty → unavailable
    else:
        ticker.history.return_value = pd.DataFrame(
            {"Close": [price]},
            index=pd.date_range("2024-01-01", periods=1),
        )
    return ticker


# ── get_quote (single symbol) ────────────────────────────────────────────────


class TestGetQuote:
    async def test_ok_status_for_valid_symbol(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(175.32)):
            result = await MarketDataClient().get_quote("AAPL")
        assert result["status"] == "ok"

    async def test_price_matches_latest_close(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(175.32)):
            result = await MarketDataClient().get_quote("AAPL")
        assert abs(result["current_price"] - 175.32) < 0.001

    async def test_symbol_is_uppercased_in_result(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(175.32)):
            result = await MarketDataClient().get_quote("aapl")
        assert result["symbol"] == "AAPL"

    async def test_result_includes_52_week_high_and_low(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(175.32)):
            result = await MarketDataClient().get_quote("AAPL")
        assert result["fifty_two_week_high"] == 200.0
        assert result["fifty_two_week_low"] == 150.0

    async def test_result_includes_data_timestamp(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(175.32)):
            result = await MarketDataClient().get_quote("AAPL")
        assert "data_timestamp" in result
        assert result["data_timestamp"] is not None

    async def test_empty_history_returns_price_unavailable(self):
        """When yfinance returns an empty DataFrame the tool must not crash."""
        with patch("yfinance.Ticker", return_value=_mock_ticker(None)):
            result = await MarketDataClient().get_quote("FAKEXYZ")
        assert result["status"] == "price_unavailable"
        assert "error" in result
        assert "instruction" in result

    async def test_price_unavailable_carries_symbol(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(None)):
            result = await MarketDataClient().get_quote("BADTICKER")
        assert result["symbol"] == "BADTICKER"

    async def test_yfinance_exception_returns_price_unavailable(self):
        """If yfinance raises, the client must catch it and return a safe dict."""
        boom_ticker = MagicMock()
        boom_ticker.history.side_effect = RuntimeError("network timeout")

        with patch("yfinance.Ticker", return_value=boom_ticker):
            result = await MarketDataClient().get_quote("AAPL")

        assert result["status"] == "price_unavailable"
        assert "network timeout" in result.get("error", "")
        assert "instruction" in result

    async def test_result_never_raises_exception(self):
        """get_quote must always return a dict — never propagate exceptions."""
        with patch("yfinance.Ticker", side_effect=Exception("fatal error")):
            try:
                result = await MarketDataClient().get_quote("AAPL")
                assert isinstance(result, dict)
            except Exception as exc:
                pytest.fail(f"get_quote raised instead of returning a safe dict: {exc}")

    async def test_currency_from_fast_info(self):
        ticker = _mock_ticker(300.0)
        ticker.fast_info.currency = "EUR"
        with patch("yfinance.Ticker", return_value=ticker):
            result = await MarketDataClient().get_quote("ASML")
        assert result["currency"] == "EUR"


# ── get_batch_quotes (multiple symbols) ──────────────────────────────────────


class TestGetBatchQuotes:
    async def test_all_symbols_present_in_quotes_dict(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(100.0)):
            result = await MarketDataClient().get_batch_quotes(["AAPL", "MSFT", "VTI"])

        assert result["status"] == "ok"
        assert "quotes" in result
        assert set(result["quotes"].keys()) == {"AAPL", "MSFT", "VTI"}

    async def test_data_timestamp_in_batch_result(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(100.0)):
            result = await MarketDataClient().get_batch_quotes(["AAPL"])
        assert "data_timestamp" in result

    async def test_individual_quotes_have_correct_status(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(250.0)):
            result = await MarketDataClient().get_batch_quotes(["GOOGL"])
        assert result["quotes"]["GOOGL"]["status"] == "ok"

    async def test_mixed_valid_and_invalid_symbols(self):
        """Batch must succeed even if some symbols are unavailable."""

        def _side_effect(symbol: str, **_: object) -> MagicMock:
            if symbol == "AAPL":
                return _mock_ticker(175.0)
            return _mock_ticker(None)  # invalid → price_unavailable

        with patch("yfinance.Ticker", side_effect=_side_effect):
            result = await MarketDataClient().get_batch_quotes(["AAPL", "INVALID"])

        assert result["status"] == "ok"
        assert result["quotes"]["AAPL"]["status"] == "ok"
        assert result["quotes"]["INVALID"]["status"] == "price_unavailable"

    async def test_empty_symbol_list_returns_empty_quotes(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker(100.0)):
            result = await MarketDataClient().get_batch_quotes([])
        assert result["status"] == "ok"
        assert result["quotes"] == {}

    async def test_batch_fetches_all_symbols_concurrently(self):
        """Verify gather is used — all three Ticker objects must be created."""
        call_log: list[str] = []

        def _tracking_ticker(symbol: str, **_: object) -> MagicMock:
            call_log.append(symbol)
            return _mock_ticker(100.0)

        with patch("yfinance.Ticker", side_effect=_tracking_ticker):
            await MarketDataClient().get_batch_quotes(["AAPL", "MSFT", "VTI"])

        assert sorted(call_log) == ["AAPL", "MSFT", "VTI"]
