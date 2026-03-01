"""
evalsNew/conftest.py — Fixtures for the Ghostfolio Agent v2 Eval Suite.

Provides realistic, varied mock payloads that mirror actual Ghostfolio API
responses. Every fixture is documented with the scenario it represents.

Module-level stubs: some optional dependencies (yfinance, curl_cffi) may not
be installed in all CI environments. We stub them so the eval suite can be
collected and the unit-level tests can run regardless. Tests that exercise the
market data client directly are skipped if the real packages are missing.
"""
from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

# ─── Optional-dependency stubs ────────────────────────────────────────────────
# These stubs let the test suite *collect* (import) even when optional packages
# are absent from the environment. Real integration tests that need live market
# data are skipped via the skip markers below.

def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "yfinance" not in sys.modules:
    _yf = _stub_module("yfinance")
    _yf.Ticker = MagicMock  # type: ignore[attr-defined]
    _yf.download = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

if "curl_cffi" not in sys.modules:
    _stub_module("curl_cffi")
    _stub_module("curl_cffi.requests")

if "langsmith" not in sys.modules:
    _stub_module("langsmith")

import pytest


# ─── Auth helpers ─────────────────────────────────────────────────────────────

AUTH_TOKEN = "evalsNew-token-ghost-2024"
AUTH_RESP = {"authToken": AUTH_TOKEN}


# ─── Holdings fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def holdings_standard():
    """
    Three-position USD portfolio: AAPL (stock), VTI (ETF), MSFT (stock).
    Total value: $8 000.00 exactly.
    Used for: correctness, allocation, sector aggregation tests.
    """
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 10,
                "value": 1750.00,
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
                "value": 4200.00,
                "valueInBaseCurrency": 4200.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "ETF",
                "sectors": [
                    {"name": "Technology", "weight": 0.30},
                    {"name": "Healthcare", "weight": 0.13},
                    {"name": "Financial Services", "weight": 0.13},
                    {"name": "Industrials", "weight": 0.13},
                    {"name": "Consumer Discretionary", "weight": 0.12},
                    {"name": "Other", "weight": 0.19},
                ],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "quantity": 5,
                "value": 2050.00,
                "valueInBaseCurrency": 2050.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
        ]
    }


@pytest.fixture
def holdings_multi_currency():
    """
    Multi-currency portfolio: AAPL (USD), ASML (EUR), VTI (USD).
    EUR holding is converted to USD base currency.
    Used for: FX exposure, multi-currency correctness tests.
    """
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 10,
                "value": 1750.00,
                "valueInBaseCurrency": 1750.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "ASML",
                "name": "ASML Holding N.V.",
                "quantity": 3,
                "value": 2190.00,
                "valueInBaseCurrency": 2400.00,  # EUR 2190 → USD 2400 at ~1.096 FX rate
                "currency": "EUR",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "Netherlands", "weight": 1.0}],
            },
            {
                "symbol": "VTI",
                "name": "Vanguard Total Stock Market ETF",
                "quantity": 20,
                "value": 4200.00,
                "valueInBaseCurrency": 4200.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "ETF",
                "sectors": [{"name": "Technology", "weight": 0.30}, {"name": "Other", "weight": 0.70}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
        ]
    }


@pytest.fixture
def holdings_empty():
    """Empty portfolio — no positions held."""
    return {"holdings": []}


@pytest.fixture
def holdings_single():
    """
    Single-position portfolio: 100% AAPL.
    Used for: edge case — single holding must produce allocation_percent = 100.
    """
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 50,
                "value": 8750.00,
                "valueInBaseCurrency": 8750.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            }
        ]
    }


@pytest.fixture
def holdings_with_duplicates():
    """
    Portfolio where the same symbol (AAPL) appears as two separate line items.
    Simulates a split-account or import artifact scenario.
    """
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 10,
                "value": 1750.00,
                "valueInBaseCurrency": 1750.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 5,
                "value": 875.00,
                "valueInBaseCurrency": 875.00,
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
                "valueInBaseCurrency": 4200.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "ETF",
                "sectors": [],
                "countries": [],
            },
        ]
    }


@pytest.fixture
def holdings_with_delisted():
    """
    Portfolio containing a delisted/price-unavailable asset (DELISTED_CO).
    Used for: edge case — tool must handle missing price history gracefully.
    """
    return {
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 10,
                "value": 1750.00,
                "valueInBaseCurrency": 1750.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "DLST",
                "name": "Delisted Corp",
                "quantity": 100,
                "value": 0.00,
                "valueInBaseCurrency": 0.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [],
                "countries": [],
            },
        ]
    }


@pytest.fixture
def holdings_ai_related():
    """
    Portfolio with AI-related holdings: NVDA, MSFT, GOOGL, and a non-AI ETF.
    Used for: 'Do I hold any AI stocks?' test.
    """
    return {
        "holdings": [
            {
                "symbol": "NVDA",
                "name": "NVIDIA Corporation",
                "quantity": 5,
                "value": 3000.00,
                "valueInBaseCurrency": 3000.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "quantity": 5,
                "value": 2050.00,
                "valueInBaseCurrency": 2050.00,
                "currency": "USD",
                "assetClass": "EQUITY",
                "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "BND",
                "name": "Vanguard Total Bond Market ETF",
                "quantity": 30,
                "value": 2300.00,
                "valueInBaseCurrency": 2300.00,
                "currency": "USD",
                "assetClass": "FIXED_INCOME",
                "assetSubClass": "ETF",
                "sectors": [],
                "countries": [],
            },
        ]
    }


# ─── Performance fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def performance_ytd():
    """
    YTD performance: +12.34% relative, +$987.65 absolute.
    Ghostfolio v2 flat format — no nested period keys.
    """
    return {
        "performance": {
            "netPerformancePercentage": 0.1234,
            "netPerformance": 987.65,
            "currentValueInBaseCurrency": 8987.65,
            "totalInvestment": 8000.00,
            "currentNetWorth": 8987.65,
            "netPerformancePercentageWithCurrencyEffect": 0.1234,
            "netPerformanceWithCurrencyEffect": 987.65,
        },
        "firstOrderDate": "2024-01-02T00:00:00.000Z",
        "hasErrors": False,
        "chart": [],
    }


@pytest.fixture
def performance_1y():
    """1-year performance: +22.10% relative, +$1 560.00 absolute."""
    return {
        "performance": {
            "netPerformancePercentage": 0.2210,
            "netPerformance": 1560.00,
            "currentValueInBaseCurrency": 8560.00,
            "totalInvestment": 7000.00,
            "currentNetWorth": 8560.00,
        },
        "hasErrors": False,
    }


@pytest.fixture
def performance_negative():
    """Down week: -3.50% relative, -$312.00 absolute. Used for 'why dropped' test."""
    return {
        "performance": {
            "netPerformancePercentage": -0.0350,
            "netPerformance": -312.00,
            "currentValueInBaseCurrency": 7688.00,
            "totalInvestment": 8000.00,
            "currentNetWorth": 7688.00,
        },
        "hasErrors": False,
    }


@pytest.fixture
def performance_zero():
    """Zero performance (flat) — edge case for no gain/loss calculations."""
    return {
        "performance": {
            "netPerformancePercentage": 0.0,
            "netPerformance": 0.0,
            "currentValueInBaseCurrency": 8000.00,
            "totalInvestment": 8000.00,
            "currentNetWorth": 8000.00,
        },
        "hasErrors": False,
    }


# ─── Transaction fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def transactions_standard():
    """Standard transaction history: 2 BUYs + 1 DIVIDEND."""
    return {
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
            {
                "id": "tx-002",
                "date": "2024-06-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF"},
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
    }


@pytest.fixture
def transactions_with_duplicates_data():
    """
    Transaction history containing an exact duplicate pair (tx-dup-1 / tx-dup-2).
    Same symbol + date + quantity + price on two different IDs.
    Used for: 'Find duplicate transactions' eval.
    """
    return {
        "activities": [
            {
                "id": "tx-dup-1",
                "date": "2024-05-10T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft Corporation"},
                "quantity": 5,
                "unitPrice": 420.00,
                "fee": 4.99,
                "currency": "USD",
                "account": {"name": "Brokerage"},
            },
            {
                "id": "tx-dup-2",
                "date": "2024-05-10T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft Corporation"},
                "quantity": 5,
                "unitPrice": 420.00,
                "fee": 4.99,
                "currency": "USD",
                "account": {"name": "Brokerage"},
            },
            {
                "id": "tx-003",
                "date": "2024-06-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                "quantity": 10,
                "unitPrice": 170.00,
                "fee": 4.99,
                "currency": "USD",
                "account": {"name": "Brokerage"},
            },
        ]
    }


@pytest.fixture
def transactions_empty():
    """Empty transaction history."""
    return {"activities": []}


@pytest.fixture
def transactions_fractional():
    """
    Fractional share transactions — tests that quantity < 1 is handled correctly.
    """
    return {
        "activities": [
            {
                "id": "tx-frac-1",
                "date": "2024-08-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "AMZN", "name": "Amazon.com Inc."},
                "quantity": 0.25,
                "unitPrice": 1840.00,
                "fee": 0.00,
                "currency": "USD",
                "account": {"name": "Fractional Brokerage"},
            },
            {
                "id": "tx-frac-2",
                "date": "2024-09-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "AMZN", "name": "Amazon.com Inc."},
                "quantity": 0.75,
                "unitPrice": 1900.00,
                "fee": 0.00,
                "currency": "USD",
                "account": {"name": "Fractional Brokerage"},
            },
        ]
    }


# ─── Market data fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def market_data_spy():
    """SPY benchmark market data for comparison tests."""
    return {
        "status": "ok",
        "symbol": "SPY",
        "current_price": 512.34,
        "currency": "USD",
        "52_week_high": 521.00,
        "52_week_low": 410.15,
        "market_cap": None,
        "data_timestamp": datetime.now(UTC).isoformat(),
        "source": "yfinance",
    }


@pytest.fixture
def market_data_aapl():
    """AAPL market data for individual stock price tests."""
    return {
        "status": "ok",
        "symbol": "AAPL",
        "current_price": 175.00,
        "currency": "USD",
        "52_week_high": 199.62,
        "52_week_low": 164.08,
        "market_cap": 2_700_000_000_000,
        "data_timestamp": datetime.now(UTC).isoformat(),
        "source": "yfinance",
    }


@pytest.fixture
def market_data_unavailable():
    """Price unavailable response — used for missing price history tests."""
    return {
        "status": "price_unavailable",
        "symbol": "DLST",
        "error": "No price data found for DLST — it may be delisted or have no history",
        "data_timestamp": datetime.now(UTC).isoformat(),
    }


# ─── Orders with fees (fee drag tests) ───────────────────────────────────────


@pytest.fixture
def orders_with_fees():
    """
    Orders with varying fees.
    Total fees: $14.97  (4.99 + 9.98 + 0.00)
    Used for: fee drag calculation accuracy test.
    """
    return {
        "activities": [
            {
                "id": "f-001",
                "date": "2024-01-15T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
                "quantity": 10,
                "unitPrice": 185.00,
                "fee": 4.99,
                "currency": "USD",
                "account": {"name": "Brokerage"},
            },
            {
                "id": "f-002",
                "date": "2024-03-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft Corporation"},
                "quantity": 5,
                "unitPrice": 410.00,
                "fee": 9.98,
                "currency": "USD",
                "account": {"name": "Brokerage"},
            },
            {
                "id": "f-003",
                "date": "2024-06-01T00:00:00.000Z",
                "type": "BUY",
                "SymbolProfile": {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF"},
                "quantity": 20,
                "unitPrice": 210.00,
                "fee": 0.00,
                "currency": "USD",
                "account": {"name": "Brokerage"},
            },
        ]
    }
