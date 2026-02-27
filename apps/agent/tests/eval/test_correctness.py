"""
Correctness Eval Suite — tests/eval/test_correctness.py
==========================================================
"Does the tool return numerically and structurally ACCURATE data?"

Each test defines a ground-truth fixture (what Ghostfolio / yfinance would
return) and then asserts the tool output matches expected values exactly.
We test:
  - Exact arithmetic    (fees, totals, percentages)
  - Correct conversions (raw fraction → percentage)
  - Output shape        (required fields present, correct types)
  - Sort order          (biggest holdings first, newest transactions first)
  - Currency handling   (USD labels, non-zero values retained)
  - Multi-holding aggregation (sector/geo weights rolled up correctly)

All network calls are mocked with respx — zero real I/O.
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

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "eval-token-abc"}


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 1 — Portfolio total value = arithmetic sum of individual holdings
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_portfolio_total_equals_sum_of_holdings():
    """
    Ground truth: AAPL=$1 750, VTI=$4 200, MSFT=$2 050  → total $8 000.
    The tool must sum them exactly — no rounding error > $0.01.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "AAPL", "name": "Apple",    "quantity": 10,  "value": 1750.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [], "countries": []},
                {"symbol": "VTI",  "name": "Vanguard", "quantity": 20,  "value": 4200.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                 "sectors": [], "countries": []},
                {"symbol": "MSFT", "name": "Microsoft","quantity": 5,   "value": 2050.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [], "countries": []},
            ]
        })
    )
    result = await _get_portfolio_summary()
    expected_total = 1750.00 + 4200.00 + 2050.00   # = 8 000.00
    assert abs(result["total_value"] - expected_total) < 0.01, (
        f"Total ${result['total_value']} ≠ expected ${expected_total}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 2 — Allocation percentages sum to exactly 100 %
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_allocation_percentages_sum_to_100(sample_holdings_response):
    """
    Allocations are derived by dividing each holding's value by the total.
    Their sum must always be 100 % (± 0.1 % for float rounding).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=sample_holdings_response)
    )
    result = await _get_portfolio_summary()
    total_alloc = sum(h["allocation_percent"] for h in result["holdings"])
    assert abs(total_alloc - 100.0) < 0.1, f"Allocations sum to {total_alloc}, not 100"


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 3 — Individual allocation values are correct to 2 decimal places
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_allocation_per_holding_is_correct():
    """
    With AAPL=$1 000 and VTI=$3 000 (total $4 000):
      AAPL should be 25.00 %
      VTI  should be 75.00 %
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "AAPL", "name": "Apple",    "quantity": 10, "value": 1000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [], "countries": []},
                {"symbol": "VTI",  "name": "Vanguard", "quantity": 30, "value": 3000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                 "sectors": [], "countries": []},
            ]
        })
    )
    result = await _get_portfolio_summary()
    by_symbol = {h["symbol"]: h["allocation_percent"] for h in result["holdings"]}
    assert abs(by_symbol["AAPL"] - 25.00) < 0.01
    assert abs(by_symbol["VTI"]  - 75.00) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 4 — Holdings are sorted largest-value first
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_holdings_sorted_largest_value_first():
    """
    Regardless of the order Ghostfolio returns holdings, the tool must
    sort them descending by current_value so the user sees biggest first.
    """
    _auth()
    # Deliberately shuffled: MSFT first, then AAPL (bigger), then VTI (biggest)
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "MSFT", "name": "Microsoft", "quantity": 5,  "value": 500.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [], "countries": []},
                {"symbol": "AAPL", "name": "Apple",     "quantity": 10, "value": 1750.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [], "countries": []},
                {"symbol": "VTI",  "name": "Vanguard",  "quantity": 20, "value": 4200.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                 "sectors": [], "countries": []},
            ]
        })
    )
    result = await _get_portfolio_summary()
    values = [h["current_value"] for h in result["holdings"]]
    assert values == sorted(values, reverse=True), f"Holdings not sorted: {values}"


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 5 — Performance: raw fraction is converted to percentage correctly
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_performance_raw_fraction_converted_to_percentage():
    """
    Ghostfolio returns relativeChange as a raw fraction (e.g. 0.1234 = 12.34 %).
    The tool must multiply by 100 and round to 2 decimal places.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json={
            "performance": {
                "ytd": {
                    "relativeChange": 0.1234,
                    "absoluteChange": 987.65,
                    "currentValue": 8987.65,
                }
            }
        })
    )
    result = await _get_performance("ytd")
    assert result["status"] == "ok"
    assert abs(result["performance"]["relative_change_pct"] - 12.34) < 0.01, (
        f"Expected 12.34%, got {result['performance']['relative_change_pct']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 6 — Performance: negative returns are preserved correctly
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_performance_negative_returns_preserved():
    """
    A loss of -8.5 % must be stored as -8.5, not 0 or positive.
    Absolute change should also be negative.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json={
            "performance": {
                "ytd": {
                    "relativeChange": -0.085,
                    "absoluteChange": -750.00,
                    "currentValue": 8150.00,
                }
            }
        })
    )
    result = await _get_performance("ytd")
    assert result["performance"]["relative_change_pct"] < 0, "Loss should be negative %"
    assert result["performance"]["absolute_change"] < 0, "Loss should be negative $"
    assert abs(result["performance"]["relative_change_pct"] - (-8.50)) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 7 — Transactions: fee total is the exact sum of individual fees
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_transaction_fee_sum_is_exact():
    """
    Three transactions with fees $4.99, $1.50, $0.00 → total_fees_paid = $6.49.
    No rounding or accumulation errors permitted.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "t1", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 5, "unitPrice": 170.00, "fee": 4.99, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
                {"id": "t2", "date": "2024-02-01T00:00:00Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft"},
                 "quantity": 2, "unitPrice": 400.00, "fee": 1.50, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
                {"id": "t3", "date": "2024-03-01T00:00:00Z", "type": "DIVIDEND",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 5, "unitPrice": 0.25, "fee": 0.00, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
            ]
        })
    )
    result = await _get_transactions()
    expected_fees = 4.99 + 1.50 + 0.00   # = 6.49
    assert abs(result["summary"]["total_fees_paid"] - expected_fees) < 0.001, (
        f"Fee total ${result['summary']['total_fees_paid']} ≠ ${expected_fees}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 8 — Transactions: total_value per line = quantity × unit_price
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_transaction_line_total_value_is_quantity_times_price():
    """
    10 shares × $170.00 = $1 700.00.  The tool must compute this exactly.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "t1", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 10, "unitPrice": 170.00, "fee": 0.00, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
            ]
        })
    )
    result = await _get_transactions()
    tx = result["transactions"][0]
    assert abs(tx["total_value"] - 1700.00) < 0.01, (
        f"10 × $170 should be $1700, got ${tx['total_value']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 9 — Transactions: sorted newest-first
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_transactions_sorted_newest_first():
    """
    Ghostfolio returns transactions oldest-first.  The tool must reverse-sort
    by date so the most recent activity appears at index 0.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "t1", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 1, "unitPrice": 170.00, "fee": 0, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
                {"id": "t3", "date": "2024-09-01T00:00:00Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft"},
                 "quantity": 1, "unitPrice": 400.00, "fee": 0, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
                {"id": "t2", "date": "2024-05-15T00:00:00Z", "type": "DIVIDEND",
                 "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                 "quantity": 1, "unitPrice": 0.24, "fee": 0, "currency": "USD",
                 "Account": {"name": "Brokerage"}},
            ]
        })
    )
    result = await _get_transactions()
    dates = [tx["date"] for tx in result["transactions"]]
    assert dates == sorted(dates, reverse=True), f"Not newest-first: {dates}"


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 10 — Diversification: sector weights are rolled up correctly
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_diversification_sector_rollup_is_correct():
    """
    AAPL ($2 000, 100% Technology) and VTI ($2 000, 50% Tech / 50% Healthcare):
      Technology total = $2 000 + $1 000 = $3 000 → 75 % of $4 000
      Healthcare total = $1 000              → 25 % of $4 000
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "AAPL", "name": "Apple", "quantity": 10, "value": 2000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [{"name": "Technology", "weight": 1.0}],
                 "countries": [{"name": "United States", "weight": 1.0}]},
                {"symbol": "VTI", "name": "Vanguard", "quantity": 10, "value": 2000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                 "sectors": [{"name": "Technology", "weight": 0.5},
                             {"name": "Healthcare",  "weight": 0.5}],
                 "countries": [{"name": "United States", "weight": 1.0}]},
            ]
        })
    )
    result = await _analyze_diversification()
    sectors = {s["name"]: s["percent"] for s in result["sector_breakdown"]}
    assert abs(sectors["Technology"] - 75.0) < 0.5, (
        f"Technology should be ~75%, got {sectors['Technology']}"
    )
    assert abs(sectors["Healthcare"] - 25.0) < 0.5, (
        f"Healthcare should be ~25%, got {sectors['Healthcare']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 11 — Diversification: score drops when single position > 10 %
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_diversification_score_penalises_concentration():
    """
    A single position at 80 % of the portfolio should produce a score
    significantly lower than 100. Formula: 100 - (80-10)*2 = 100-140 → clamped to 0.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "BIG", "name": "Big Co", "quantity": 100, "value": 8000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [{"name": "Technology", "weight": 1.0}],
                 "countries": [{"name": "United States", "weight": 1.0}]},
                {"symbol": "SML", "name": "Small Co", "quantity": 10, "value": 2000.00,
                 "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [{"name": "Healthcare", "weight": 1.0}],
                 "countries": [{"name": "United States", "weight": 1.0}]},
            ]
        })
    )
    result = await _analyze_diversification()
    assert result["diversification_score"] < 50, (
        f"80% concentration should score <50, got {result['diversification_score']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Correctness 12 — Market data: price retrieved to full precision
# ══════════════════════════════════════════════════════════════════════════════

def test_market_data_price_precision():
    """
    yfinance returns 175.32; the tool must store it exactly to 2 dp — not
    round to 175 or inflate to 175.999.
    """
    mock_ticker = MagicMock()
    mock_ticker.fast_info.currency = "USD"
    mock_ticker.fast_info.year_high = 199.62
    mock_ticker.fast_info.year_low  = 124.17
    mock_ticker.history.return_value = pd.DataFrame(
        {"Close": [175.32]},
        index=pd.date_range("2024-01-01", periods=1)
    )
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_market_data.invoke({"symbols": "AAPL"})
    assert result["status"] == "ok"
    assert abs(result["current_price"] - 175.32) < 0.001, (
        f"Price should be 175.32, got {result['current_price']}"
    )
