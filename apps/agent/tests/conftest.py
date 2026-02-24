"""Shared pytest fixtures for all test suites."""
import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_holdings_response():
    """Mock Ghostfolio /portfolio/holdings response."""
    return {
        "holdings": {
            "AAPL": {
                "name": "Apple Inc.",
                "quantity": 10,
                "value": 1750.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            "VTI": {
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
                ],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            "MSFT": {
                "name": "Microsoft Corporation",
                "quantity": 5,
                "value": 2050.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
        }
    }


@pytest.fixture
def sample_performance_response():
    return {
        "performance": {
            "ytd": {
                "relativeChange": 0.1234,
                "absoluteChange": 987.65,
                "currentValue": 8000.00,
            },
            "1y": {
                "relativeChange": 0.2156,
                "absoluteChange": 1432.10,
                "currentValue": 8000.00,
            },
            "max": {
                "relativeChange": 0.4521,
                "absoluteChange": 2450.00,
                "currentValue": 8000.00,
            },
        }
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
    return datetime.now(timezone.utc).isoformat()
