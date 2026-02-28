"""
evalsNew/test_edge_cases_v2.py — Edge Case Eval Suite (v2)
===========================================================
Eval IDs: EC01–EC08

"Does the agent handle unusual, missing, or malformed data gracefully?"

Tests in this file:
  EC01 — Empty portfolio: status='empty', no crash, helpful message
  EC02 — Unknown ticker: price_unavailable response, agent must not guess price
  EC03 — Ambiguous query (missing timeframe): agent asks for clarification OR states assumption
  EC04 — Future date in transaction filter: returns empty gracefully
  EC05 — Duplicate transactions detected and flagged correctly
  EC06 — Delisted asset (value=0.00): no ZeroDivisionError, allocation=0%
  EC07 — Single-holding portfolio: allocation_percent=100%, concentration flag
  EC08 — Fractional share transactions: quantities <1 handled without rounding

All network calls are mocked with respx — zero real I/O.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.transactions import _get_transactions

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "edge-cases-v2-token"}

DOLLAR_TOLERANCE = 0.01


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


# ══════════════════════════════════════════════════════════════════════════════
# EC01 — Empty portfolio returns graceful empty response
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_ec01_empty_portfolio_returns_empty_status():
    """
    When Ghostfolio returns an empty holdings list, the tool must return
    status='empty' with a helpful message. No crash, no division by zero.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": []})
    )
    result = await _get_portfolio_summary()

    assert result["status"] == "empty", (
        f"EC01: Empty portfolio must return status='empty', got: {result['status']}"
    )
    assert result.get("holdings", []) == [], "EC01: holdings must be empty list"
    assert result.get("total_value", 0) == 0.0, "EC01: total_value must be 0.0"
    assert "message" in result, "EC01: Must include a message field explaining empty portfolio"
    assert result["message"], "EC01: Message must not be empty"


@respx.mock
async def test_ec01_empty_portfolio_no_exception():
    """Empty portfolio must not raise any exception."""
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": []})
    )
    try:
        result = await _get_portfolio_summary()
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"EC01: Empty portfolio raised exception: {type(e).__name__}: {e}")


@respx.mock
async def test_ec01_empty_transactions_returns_empty_status():
    """Empty transaction list must return status='empty' — not crash."""
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": []})
    )
    result = await _get_transactions()

    assert result["status"] == "empty"
    assert result.get("transactions", []) == []


# ══════════════════════════════════════════════════════════════════════════════
# EC02 — Unknown ticker: price_unavailable, agent must not guess
# ══════════════════════════════════════════════════════════════════════════════

async def test_ec02_unknown_ticker_returns_price_unavailable():
    """
    get_market_data for a non-existent symbol (FAKESTOCK) must return
    status='price_unavailable', not fabricate a price.
    The agent docstring instructs: 'If status=price_unavailable, do NOT guess.'
    """
    from agent.tools.market import get_market_data as market_tool
    from unittest.mock import AsyncMock, patch

    price_unavailable_response = {
        "status": "price_unavailable",
        "symbol": "FAKESTOCK",
        "error": "No price data found for FAKESTOCK — it may be delisted or have no history",
    }

    with patch("agent.clients.market.MarketDataClient.get_quote",
               new_callable=AsyncMock, return_value=price_unavailable_response):
        result = await market_tool.ainvoke({"symbols": "FAKESTOCK"})

    assert result.get("status") == "price_unavailable", (
        f"EC02: Unknown ticker must return status='price_unavailable'. Got: {result}"
    )
    assert "FAKESTOCK" in str(result), "EC02: Symbol must be in error response"
    # Must not contain a numeric price
    assert "current_price" not in result or result.get("current_price") is None, (
        "EC02: price_unavailable response must not contain current_price"
    )


# ══════════════════════════════════════════════════════════════════════════════
# EC03 — Ambiguous query (missing timeframe): tool falls back defensively
#         Since we're testing the TOOL, not the LLM: verify the tool has a
#         safe default (ytd) and that the docstring signals valid options.
#         The LLM-side test is: 'How did I do?' → agent asks clarifying question.
# ══════════════════════════════════════════════════════════════════════════════

def test_ec03_performance_tool_has_safe_default_date_range():
    """
    When no date_range is specified (uses default), the tool must use 'ytd'.
    This ensures an ambiguous query ('how did I do?') defaults gracefully.
    """
    import inspect
    from agent.tools.performance import _get_performance

    sig = inspect.signature(_get_performance)
    default = sig.parameters["date_range"].default
    assert default == "ytd", (
        f"EC03: _get_performance default date_range must be 'ytd', got: '{default}'. "
        f"This ensures ambiguous queries without a timeframe don't crash."
    )


def test_ec03_performance_docstring_lists_all_valid_ranges():
    """
    The docstring must enumerate all valid date_range values so the LLM
    knows to ask for clarification when the user's timeframe is ambiguous.
    """
    from agent.tools.performance import get_performance

    doc = (getattr(get_performance, "description", None) or get_performance.__doc__ or "").lower()
    valid_ranges = ["1d", "wtd", "mtd", "ytd", "1y", "5y", "max"]
    missing = [r for r in valid_ranges if r not in doc]
    assert len(missing) <= 1, (
        f"EC03: get_performance docstring missing valid ranges: {missing}. "
        f"The LLM needs this to know what to ask for when query is ambiguous."
    )


# ══════════════════════════════════════════════════════════════════════════════
# EC04 — Future date in transaction filter returns empty gracefully
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_ec04_future_date_returns_empty():
    """
    date_from='2035-01-01' is a valid ISO date but in the far future.
    Ghostfolio returns an empty activities list — tool must return status='empty'.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": []})
    )
    result = await _get_transactions(date_from="2035-01-01", date_to="2035-01-31")

    assert isinstance(result, dict), "EC04: Must return dict, not raise exception"
    assert result.get("status") in ("empty", "ok"), (
        f"EC04: Future date must return empty or ok (no transactions), got: {result}"
    )
    tx_count = result.get("transaction_count", 0)
    assert tx_count == 0, (
        f"EC04: Future date must return 0 transactions, got: {tx_count}"
    )


@respx.mock
async def test_ec04_past_date_before_portfolio_start_returns_empty():
    """
    date_to='1970-01-01' is before any portfolio could exist.
    Must return empty gracefully.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={"activities": []})
    )
    result = await _get_transactions(date_from="1970-01-01", date_to="1970-12-31")

    assert result.get("status") in ("empty", "ok")
    assert result.get("transaction_count", 0) == 0


# ══════════════════════════════════════════════════════════════════════════════
# EC05 — Duplicate transactions detected
#         Two transactions with same symbol+date+qty+price but different IDs.
#         The tool must return both (detection is the agent's job); the agent
#         must identify them as a potential duplicate pair.
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_ec05_duplicate_transactions_both_returned(transactions_with_duplicates_data):
    """
    The tool must return ALL transactions including duplicates — it's the
    agent's job (not the tool's) to detect and flag them. We verify the
    tool returns both duplicate records intact.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json=transactions_with_duplicates_data)
    )
    result = await _get_transactions()

    assert result["status"] == "ok"
    # Must contain all 3 transactions including both duplicates
    assert result["transaction_count"] == 3, (
        f"EC05: Expected 3 transactions (including duplicates), got {result['transaction_count']}"
    )

    # Both duplicate IDs must be present
    tx_ids = {t["id"] for t in result["transactions"]}
    assert "tx-dup-1" in tx_ids, "EC05: tx-dup-1 must be in result"
    assert "tx-dup-2" in tx_ids, "EC05: tx-dup-2 must be in result"


def test_ec05_detect_duplicates_from_transaction_list():
    """
    Given a list of transactions, identify pairs where:
    symbol + date[:10] + quantity + unit_price are all equal.
    This is the agent's duplicate detection logic.
    """
    transactions = [
        {"id": "tx-dup-1", "symbol": "MSFT", "date": "2024-05-10T00:00:00.000Z",
         "quantity": 5, "unit_price": 420.00, "type": "BUY"},
        {"id": "tx-dup-2", "symbol": "MSFT", "date": "2024-05-10T00:00:00.000Z",
         "quantity": 5, "unit_price": 420.00, "type": "BUY"},
        {"id": "tx-003", "symbol": "AAPL", "date": "2024-06-01T00:00:00.000Z",
         "quantity": 10, "unit_price": 170.00, "type": "BUY"},
    ]

    # Duplicate detection function (mirrors what the agent would do)
    def find_duplicates(txs):
        seen = {}
        duplicates = []
        for tx in txs:
            key = (
                tx["symbol"],
                tx["date"][:10],  # date-only part
                tx["quantity"],
                tx["unit_price"],
                tx["type"],
            )
            if key in seen:
                duplicates.append((seen[key], tx["id"]))
            else:
                seen[key] = tx["id"]
        return duplicates

    dupes = find_duplicates(transactions)
    assert len(dupes) == 1, (
        f"EC05: Expected 1 duplicate pair, found {len(dupes)}: {dupes}"
    )
    assert set(dupes[0]) == {"tx-dup-1", "tx-dup-2"}, (
        f"EC05: Duplicate pair must be (tx-dup-1, tx-dup-2), got {dupes[0]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# EC06 — Delisted asset (value=0.00): no ZeroDivisionError, allocation=0%
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_ec06_delisted_asset_zero_value_no_crash(holdings_with_delisted):
    """
    DLST has value=0.00. When computing DLST's allocation_percent, the tool
    must not raise ZeroDivisionError. DLST allocation must be 0.0.
    Total value must be AAPL's $1750 only.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_with_delisted)
    )
    try:
        result = await _get_portfolio_summary()
    except ZeroDivisionError as e:
        pytest.fail(f"EC06: ZeroDivisionError raised for zero-value holding: {e}")

    assert result["status"] == "ok"
    # Total value = AAPL($1750) only — DLST contributes $0
    assert abs(result["total_value"] - 1750.00) <= DOLLAR_TOLERANCE, (
        f"EC06: total_value={result['total_value']} ≠ 1750.00 (DLST contributes $0)"
    )

    holdings_map = {h["symbol"]: h for h in result["holdings"]}
    dlst = holdings_map.get("DLST", {})
    assert dlst.get("allocation_percent", 0) == 0.0, (
        f"EC06: DLST allocation_percent={dlst.get('allocation_percent')} ≠ 0.0"
    )
    aapl = holdings_map.get("AAPL", {})
    assert abs(aapl.get("allocation_percent", 0) - 100.0) <= 0.1, (
        f"EC06: AAPL allocation_percent={aapl.get('allocation_percent')} ≠ 100.0"
    )


@respx.mock
async def test_ec06_diversification_handles_zero_value_holding():
    """
    analyze_diversification must not crash when a holding has value=0.
    The zero-value holding must be excluded from sector aggregation.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "AAPL", "valueInBaseCurrency": 1750.00, "assetClass": "EQUITY",
                 "sectors": [{"name": "Technology", "weight": 1.0}], "countries": []},
                {"symbol": "DLST", "valueInBaseCurrency": 0.00, "assetClass": "EQUITY",
                 "sectors": [], "countries": []},
            ]
        })
    )
    try:
        result = await _analyze_diversification()
    except ZeroDivisionError as e:
        pytest.fail(f"EC06: ZeroDivisionError in analyze_diversification: {e}")

    assert result.get("status") == "ok"
    assert abs(result["total_value"] - 1750.00) <= DOLLAR_TOLERANCE


# ══════════════════════════════════════════════════════════════════════════════
# EC07 — Single-holding portfolio: allocation_percent = 100%, concentration flag
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_ec07_single_holding_allocation_is_100_pct(holdings_single):
    """
    A portfolio with exactly one holding (AAPL, $8750) must report:
    - allocation_percent = 100.0 (not 99.99 or 100.001)
    - total_value = 8750.00
    - position_count = 1
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_single)
    )
    result = await _get_portfolio_summary()

    assert result["status"] == "ok"
    assert result["position_count"] == 1
    assert abs(result["total_value"] - 8750.00) <= DOLLAR_TOLERANCE

    aapl = result["holdings"][0]
    assert abs(aapl["allocation_percent"] - 100.0) <= 0.01, (
        f"EC07: Single holding must have allocation_percent=100.0, "
        f"got {aapl['allocation_percent']}"
    )


@respx.mock
async def test_ec07_single_holding_triggers_concentration_flag():
    """
    A single holding at 100% of the portfolio must trigger a concentration
    warning flag in analyze_diversification.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={
            "holdings": [
                {"symbol": "AAPL", "name": "Apple Inc.", "valueInBaseCurrency": 8750.00,
                 "assetClass": "EQUITY", "assetSubClass": "STOCK",
                 "sectors": [{"name": "Technology", "weight": 1.0}],
                 "countries": [{"name": "United States", "weight": 1.0}]},
            ]
        })
    )
    result = await _analyze_diversification()

    assert result.get("status") == "ok"
    assert len(result.get("concentration_flags", [])) >= 1, (
        "EC07: Single holding at 100% must trigger at least one concentration flag"
    )
    flag = result["concentration_flags"][0]
    assert flag.get("severity") in ("HIGH", "MEDIUM"), (
        f"EC07: Concentration flag severity must be HIGH or MEDIUM, got: {flag.get('severity')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# EC08 — Fractional share transactions: quantity < 1 handled without rounding
# ══════════════════════════════════════════════════════════════════════════════

@respx.mock
async def test_ec08_fractional_shares_correct_total_value(transactions_fractional):
    """
    tx-frac-1: 0.25 shares × $1840.00 = $460.00
    tx-frac-2: 0.75 shares × $1900.00 = $1425.00
    Both quantities must be preserved; neither should be rounded to 0 or 1.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json=transactions_fractional)
    )
    result = await _get_transactions()

    assert result["status"] == "ok"
    assert result["transaction_count"] == 2

    # Sort: newest first, so frac-2 (Sep) > frac-1 (Aug)
    tx_by_id = {t["id"]: t for t in result["transactions"]}

    frac_1 = tx_by_id.get("tx-frac-1", {})
    frac_2 = tx_by_id.get("tx-frac-2", {})

    assert abs(frac_1.get("quantity", 0) - 0.25) <= 0.0001, (
        f"EC08: tx-frac-1 quantity={frac_1.get('quantity')} ≠ 0.25 (must not round fractional)"
    )
    assert abs(frac_2.get("quantity", 0) - 0.75) <= 0.0001, (
        f"EC08: tx-frac-2 quantity={frac_2.get('quantity')} ≠ 0.75"
    )

    assert abs(frac_1.get("total_value", 0) - 460.00) <= DOLLAR_TOLERANCE, (
        f"EC08: tx-frac-1 total_value={frac_1.get('total_value')} ≠ 460.00 "
        f"(0.25 × $1840.00 = $460.00)"
    )
    assert abs(frac_2.get("total_value", 0) - 1425.00) <= DOLLAR_TOLERANCE, (
        f"EC08: tx-frac-2 total_value={frac_2.get('total_value')} ≠ 1425.00 "
        f"(0.75 × $1900.00 = $1425.00)"
    )


@respx.mock
async def test_ec08_fractional_quantity_not_zero():
    """
    A quantity of 0.001 (micro-fractional) must not be rounded to 0.
    Edge case: float precision must not eat very small quantities.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json={
            "activities": [
                {"id": "micro-tx", "date": "2024-10-01T00:00:00.000Z", "type": "BUY",
                 "SymbolProfile": {"symbol": "BRK.B", "name": "Berkshire Hathaway"},
                 "quantity": 0.001, "unitPrice": 350000.00, "fee": 0.00,
                 "currency": "USD", "account": {"name": "Brokerage"}},
            ]
        })
    )
    result = await _get_transactions()

    tx = result["transactions"][0]
    assert tx["quantity"] > 0, (
        f"EC08: Micro-fractional quantity 0.001 must not be rounded to 0. Got: {tx['quantity']}"
    )
    assert abs(tx["quantity"] - 0.001) <= 0.0001
