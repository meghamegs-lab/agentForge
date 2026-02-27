"""Shared pytest fixtures for all test suites."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest


@pytest.fixture
def sample_holdings_response():
    """
    Mock Ghostfolio /portfolio/holdings response.
    Holdings are returned as a LIST (not a dict) — matches real Ghostfolio API format.
    """
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 10,
                "value": 1750.00,
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
                "value": 4200.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "ETF",
                "sectors": [
                    {"name": "Technology", "weight": 0.30},
                    {"name": "Healthcare", "weight": 0.13},
                    {"name": "Financial Services", "weight": 0.13},
                    {"name": "Industrials", "weight": 0.13},
                    {"name": "Consumer Discretionary", "weight": 0.12},
                    {"name": "Other", "weight": 0.19},  # weights sum to 1.0
                ],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "quantity": 5,
                "value": 2050.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
        ]
    }


@pytest.fixture
def sample_performance_response():
    # v2 API returns a flat performance object for the requested ?range= period.
    # There are no nested period keys (ytd/1y/etc.) — one request = one period.
    return {
        "performance": {
            "netPerformancePercentage": 0.1234,       # decimal: 0.1234 = 12.34%
            "netPerformance": 987.65,                  # absolute gain/loss in base currency
            "currentValueInBaseCurrency": 8000.00,
            "totalInvestment": 7012.35,
            "currentNetWorth": 8000.00,
            "netPerformancePercentageWithCurrencyEffect": 0.1234,
            "netPerformanceWithCurrencyEffect": 987.65,
            "totalInvestmentValueWithCurrencyEffect": 7012.35,
        },
        "firstOrderDate": "2023-01-15T00:00:00.000Z",
        "hasErrors": False,
        "chart": [],
    }


@pytest.fixture
def sample_orders_response():
    return {
        "activities": [
            {
                "id": "tx1",
                "date": "2024-03-15T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                "quantity": 10,
                "unitPrice": 170.00,
                "fee": 4.99,
                "currency": "USD",
                "Account": {"name": "Brokerage"},
            },
            {
                "id": "tx2",
                "date": "2024-06-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF"},
                "quantity": 20,
                "unitPrice": 210.00,
                "fee": 0.00,
                "currency": "USD",
                "Account": {"name": "Brokerage"},
            },
            {
                "id": "tx3",
                "date": "2024-09-01T00:00:00.000Z",
                "type": "DIVIDEND",
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                "quantity": 10,
                "unitPrice": 0.25,
                "fee": 0.00,
                "currency": "USD",
                "Account": {"name": "Brokerage"},
            },
        ]
    }


@pytest.fixture
def now_iso():
    return datetime.now(UTC).isoformat()
