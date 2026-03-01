"""
evals/test_correctness.py — Correctness Eval Suite (v2)
==============================================================
Eval IDs: C01–C07

"Do the tools return numerically and structurally ACCURATE data?"

Tests in this file assert:
  C01 — YTD fraction-to-percentage conversion is exact (0.1234 → 12.34%)
  C02 — Top 3 contributors identified by value (descending sort)
  C03 — Sector allocation weights sum to ~100%
  C04 — Fee drag percentage arithmetic is correct
  C05 — Net P&L = current_value − total_investment (arithmetic identity)
  C06 — Asset class breakdown aggregation across mixed holdings
  C07 — Multi-currency portfolio: USD/EUR exposure splits use base-currency values

Tolerances:
  Percentages: ±0.01 percentage points (absolute)
  Dollar amounts: ±$0.01
  Sums to 100%: ±0.5 percentage points

All network calls are mocked with respx — zero real I/O.
"""

from __future__ import annotations

import httpx
import respx

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "correctness-v2-token"}

PCT_TOLERANCE = 0.01  # ±0.01 percentage points
DOLLAR_TOLERANCE = 0.01  # ±$0.01


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


# ══════════════════════════════════════════════════════════════════════════════
# C01 — YTD return fraction-to-percentage conversion is exact
#        Ghostfolio returns netPerformancePercentage as a DECIMAL (0.1234 = 12.34%).
#        The tool must multiply by 100 and round to 2dp before returning.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c01_ytd_fraction_to_pct_conversion():
    """
    Ground truth: netPerformancePercentage=0.1234 → relative_change_pct=12.34.
    Tolerance: ±0.01 percentage points.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={
                "performance": {
                    "netPerformancePercentage": 0.1234,
                    "netPerformance": 987.65,
                    "currentValueInBaseCurrency": 8987.65,
                    "totalInvestment": 8000.00,
                    "currentNetWorth": 8987.65,
                },
                "hasErrors": False,
            },
        )
    )
    result = await _get_performance("ytd")

    assert result["status"] == "ok", f"Expected ok, got: {result}"
    assert result["requested_period"] == "ytd"

    actual_pct = result["performance"]["relative_change_pct"]
    assert abs(actual_pct - 12.34) <= PCT_TOLERANCE, (
        f"C01: relative_change_pct={actual_pct} ≠ 12.34 "
        f"(tolerance ±{PCT_TOLERANCE}). "
        f"Check that 0.1234 is multiplied by 100 before rounding."
    )

    actual_abs = result["performance"]["absolute_change"]
    assert abs(actual_abs - 987.65) <= DOLLAR_TOLERANCE, (
        f"C01: absolute_change={actual_abs} ≠ 987.65 (tolerance ±{DOLLAR_TOLERANCE})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# C02 — Top 3 contributors identified by value (descending sort)
#        Holdings must be returned sorted by current_value descending.
#        The top-3 are: VTI($4200) > MSFT($2050) > AAPL($1750). AMZN($520) is 4th.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c02_top_3_contributors_by_value():
    """
    Four holdings. Sorted descending by value: VTI, MSFT, AAPL, AMZN.
    Top 3 must be VTI, MSFT, AAPL — in that order.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    # Intentionally unsorted to test the sort
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "quantity": 10,
                        "valueInBaseCurrency": 1750.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "AMZN",
                        "name": "Amazon.com Inc.",
                        "quantity": 3,
                        "valueInBaseCurrency": 520.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "VTI",
                        "name": "Vanguard Total Stock Market ETF",
                        "quantity": 20,
                        "valueInBaseCurrency": 4200.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "ETF",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "MSFT",
                        "name": "Microsoft Corporation",
                        "quantity": 5,
                        "valueInBaseCurrency": 2050.00,
                        "currency": "USD",
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                ]
            },
        )
    )
    result = await _get_portfolio_summary()

    assert result["status"] == "ok"
    holdings = result["holdings"]
    assert len(holdings) == 4

    top_3_symbols = [h["symbol"] for h in holdings[:3]]
    assert top_3_symbols == ["VTI", "MSFT", "AAPL"], (
        f"C02: Top 3 by value must be ['VTI','MSFT','AAPL'], got {top_3_symbols}"
    )
    assert holdings[3]["symbol"] == "AMZN", "C02: AMZN must be 4th (smallest value $520)"

    # Verify the values are in strictly descending order
    for i in range(len(holdings) - 1):
        assert holdings[i]["current_value"] >= holdings[i + 1]["current_value"], (
            f"C02: Holdings not sorted descending: "
            f"{holdings[i]['symbol']}={holdings[i]['current_value']} < "
            f"{holdings[i + 1]['symbol']}={holdings[i + 1]['current_value']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# C03 — Sector allocation weights sum to ~100%
#        analyze_diversification rolls up weighted sector exposure per holding.
#        The sum of sector percentages across all sectors must equal 100 ± 0.5%.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c03_sector_allocation_sums_to_100(holdings_standard):
    """
    Three holdings with overlapping sector weights.
    Sum of all sector allocation percentages must be 100.0 ± 0.5%.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_standard)
    )
    result = await _analyze_diversification()

    assert result["status"] == "ok", f"C03: Expected ok, got {result}"
    sector_breakdown = result["sector_breakdown"]
    assert len(sector_breakdown) > 0, "C03: No sectors returned"

    sector_pct_sum = sum(s["percent"] for s in sector_breakdown)
    assert abs(sector_pct_sum - 100.0) <= 0.5, (
        f"C03: Sector percentages sum to {sector_pct_sum:.4f}%, expected 100% ±0.5%"
    )

    # Technology must be the dominant sector (AAPL + MSFT fully + 30% of VTI)
    tech_sector = next((s for s in sector_breakdown if s["name"] == "Technology"), None)
    assert tech_sector is not None, "C03: Technology sector missing from breakdown"
    assert tech_sector["percent"] > 50.0, (
        f"C03: Technology should be >50% of portfolio, got {tech_sector['percent']}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# C04 — Fee drag percentage arithmetic is correct
#        total_fees = 4.99 + 9.98 + 0.00 = $14.97
#        gross_return = $987.65 (from performance mock)
#        fee_drag_pct = 14.97 / 987.65 * 100 = 1.515...% → ≈ 1.52%
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c04_fee_drag_arithmetic(orders_with_fees, performance_ytd, holdings_standard):
    """
    Fee drag = total_fees / gross_return.
    Ground truth: $14.97 / $987.65 = 1.515% ≈ 1.52% (tolerance ±0.1%).
    VTI must not appear in fee_by_symbol (fee=0.00).
    """
    from agent.tools.fee_drag import _fee_drag

    _auth()
    respx.get(f"{BASE_URL}/api/v1/order").mock(
        return_value=httpx.Response(200, json=orders_with_fees)
    )
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json=performance_ytd)
    )
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_standard)
    )
    result = await _fee_drag("ytd")

    assert result.get("status") == "ok", f"C04: Expected ok, got {result}"

    total_fees = result.get("total_fees_paid", 0)
    assert abs(total_fees - 14.97) <= DOLLAR_TOLERANCE, (
        f"C04: total_fees_paid={total_fees} ≠ 14.97 (±{DOLLAR_TOLERANCE})"
    )

    fee_drag_pct = result.get("fee_drag_pct", 0)
    assert abs(fee_drag_pct - 1.515) <= 0.1, (
        f"C04: fee_drag_pct={fee_drag_pct} ≠ ~1.52% (tolerance ±0.1%)"
    )

    # VTI must NOT appear in fee_by_symbol with non-zero value
    fee_by_symbol = result.get("fee_by_symbol", {})
    if "VTI" in fee_by_symbol:
        assert fee_by_symbol["VTI"] <= DOLLAR_TOLERANCE, (
            f"C04: VTI has fee={fee_by_symbol['VTI']} but should be 0.00"
        )


# ══════════════════════════════════════════════════════════════════════════════
# C05 — Net P&L = current_value − total_investment (arithmetic identity)
#        This is an invariant: the reported absolute_change must equal the
#        difference between current_value and total_investment in the tool output.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c05_net_pnl_arithmetic_identity():
    """
    For ANY valid performance response, the invariant must hold:
        absolute_change == current_value - total_investment
    Tolerance: ±$0.01 (floating point rounding).
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={
                "performance": {
                    "netPerformancePercentage": 0.1500,
                    "netPerformance": 1200.00,
                    "currentValueInBaseCurrency": 9200.00,
                    "totalInvestment": 8000.00,
                    "currentNetWorth": 9200.00,
                }
            },
        )
    )
    result = await _get_performance("max")

    assert result["status"] == "ok"
    perf = result["performance"]

    computed_pnl = perf["current_value"] - perf["total_investment"]
    reported_pnl = perf["absolute_change"]

    assert abs(reported_pnl - computed_pnl) <= DOLLAR_TOLERANCE, (
        f"C05: Arithmetic identity failed. "
        f"current_value({perf['current_value']}) - total_investment({perf['total_investment']}) "
        f"= {computed_pnl} ≠ absolute_change({reported_pnl})"
    )

    assert abs(reported_pnl - 1200.00) <= DOLLAR_TOLERANCE, (
        f"C05: absolute_change={reported_pnl} ≠ 1200.00"
    )
    assert abs(perf["relative_change_pct"] - 15.00) <= PCT_TOLERANCE, (
        f"C05: relative_change_pct={perf['relative_change_pct']} ≠ 15.00%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# C06 — Asset class breakdown aggregation across mixed holdings
#        Three EQUITY holdings → 100% EQUITY asset class.
#        Verify the tool rolls up asset_class correctly using valueInBaseCurrency.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c06_asset_class_breakdown_aggregation():
    """
    Three EQUITY holdings (AAPL $1750, MSFT $2050, VTI $4200) → 100% EQUITY.
    Total portfolio value: $8000.
    asset_class_breakdown must show EQUITY ≈ 100%.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(
            200,
            json={
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "valueInBaseCurrency": 1750.00,
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "MSFT",
                        "valueInBaseCurrency": 2050.00,
                        "assetClass": "EQUITY",
                        "assetSubClass": "STOCK",
                        "sectors": [],
                        "countries": [],
                    },
                    {
                        "symbol": "VTI",
                        "valueInBaseCurrency": 4200.00,
                        "assetClass": "EQUITY",
                        "assetSubClass": "ETF",
                        "sectors": [],
                        "countries": [],
                    },
                ]
            },
        )
    )
    result = await _analyze_diversification()

    assert result["status"] == "ok"
    ac_breakdown = result["asset_class_breakdown"]
    assert len(ac_breakdown) >= 1

    equity = next((a for a in ac_breakdown if a["name"] == "EQUITY"), None)
    assert equity is not None, "C06: EQUITY not found in asset_class_breakdown"
    assert abs(equity["percent"] - 100.0) <= 0.1, (
        f"C06: EQUITY percent={equity['percent']} ≠ 100.0% (tolerance ±0.1%)"
    )
    assert abs(result["total_value"] - 8000.00) <= DOLLAR_TOLERANCE, (
        f"C06: total_value={result['total_value']} ≠ 8000.00"
    )


# ══════════════════════════════════════════════════════════════════════════════
# C07 — Multi-currency: USD/EUR exposure splits use base-currency values
#        AAPL (USD, $1750) + ASML (EUR local, but $2400 base) + VTI (USD, $4200)
#        Total base: $8350. USD% ≈ 71.26%, EUR% ≈ 28.74%.
#        The tool must use valueInBaseCurrency, NOT the local-currency value.
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_c07_multicurrency_uses_base_currency_values(holdings_multi_currency):
    """
    ASML is EUR-denominated: local value=€2190, base value=$2400.
    The tool must use $2400 (valueInBaseCurrency) for allocation, not €2190.
    USD % = 5950/8350 ≈ 71.26%; EUR % = 2400/8350 ≈ 28.74%.
    """
    _auth()
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json=holdings_multi_currency)
    )
    result = await _get_portfolio_summary()

    assert result["status"] == "ok"
    # Total using base-currency values: 1750 + 2400 + 4200 = 8350
    assert abs(result["total_value"] - 8350.00) <= DOLLAR_TOLERANCE, (
        f"C07: total_value={result['total_value']} ≠ 8350.00 "
        f"(must use valueInBaseCurrency, not local currency value)"
    )

    holdings_map = {h["symbol"]: h for h in result["holdings"]}
    asml = holdings_map.get("ASML", {})
    assert abs(asml.get("current_value", 0) - 2400.00) <= DOLLAR_TOLERANCE, (
        f"C07: ASML current_value={asml.get('current_value')} ≠ $2400.00 "
        f"(must use valueInBaseCurrency=$2400, not local EUR value=$2190)"
    )

    # EUR allocation = 2400 / 8350 * 100 ≈ 28.74%
    expected_eur_pct = 2400.0 / 8350.0 * 100
    actual_eur_pct = asml.get("allocation_percent", 0)
    assert abs(actual_eur_pct - expected_eur_pct) <= 0.1, (
        f"C07: ASML allocation_percent={actual_eur_pct:.2f}% ≠ {expected_eur_pct:.2f}% "
        f"(tolerance ±0.1%)"
    )

    # USD total (AAPL + VTI) = 5950 / 8350 * 100 ≈ 71.26%
    usd_holdings = [h for h in result["holdings"] if h.get("currency") == "USD"]
    usd_total = sum(h["current_value"] for h in usd_holdings)
    assert abs(usd_total - 5950.00) <= DOLLAR_TOLERANCE, f"C07: USD total={usd_total} ≠ 5950.00"
