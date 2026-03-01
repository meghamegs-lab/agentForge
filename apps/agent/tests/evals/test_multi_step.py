"""
evals/test_multi_step.py — Multi-Step Reasoning Eval Suite (v2)
======================================================================
Eval IDs: MS01–MS12

"Does the agent correctly chain multiple tools in sequence and produce
consistent, cross-validated results?"

Multi-step scenarios covered:
  MS01 — Portfolio summary → fee drag: gross = net + fees identity
  MS02 — Holdings → diversification: sectors match holding sector data
  MS03 — Holdings → rebalancing: every trade symbol exists in holdings
  MS04 — Holdings → market context: analysed positions are real holdings
  MS05 — Holdings → health scorecard: grade reflects actual position count
  MS06 — Transactions → patterns: buy count matches raw activity records
  MS07 — Proactive monitor first session: no changes reported
  MS08 — Proactive monitor with snapshot: portfolio value change detected
  MS09 — Proactive monitor: new concentration breach detected
  MS10 — Fee drag cross-check: fees from raw orders match tool total
  MS11 — Health scorecard with 12 well-diversified positions → grade ≥ B
  MS12 — Market context + hedges: suggested_hedges non-empty (4 themes)

All tests mock network calls with respx — zero real I/O.
Ghostfolio performance API: v2 flat format (no nested period keys).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.fee_drag import _fee_drag
from agent.tools.health_scorecard import _scorecard
from agent.tools.market_context import _market_context
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.proactive_monitor import _proactive_monitor
from agent.tools.rebalancing import _rebalancing_plan
from agent.tools.transaction_patterns import _transaction_patterns

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_RESP = {"authToken": "multi-step-v2-token"}

# ---------------------------------------------------------------------------
# Shared fixtures — Ghostfolio v2 flat performance format
# ---------------------------------------------------------------------------

HOLDINGS_LIST = [
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

ORDERS_DATA = {
    "activities": [
        {
            "id": "tx1",
            "date": "2024-01-15T00:00:00Z",
            "type": "BUY",
            "SymbolProfile": {"symbol": "AAPL", "name": "Apple Inc."},
            "quantity": 10,
            "unitPrice": 170.00,
            "fee": 4.99,
            "currency": "USD",
            "account": {"name": "Brokerage"},
        },
        {
            "id": "tx2",
            "date": "2024-02-20T00:00:00Z",
            "type": "BUY",
            "SymbolProfile": {"symbol": "VTI", "name": "Vanguard ETF"},
            "quantity": 20,
            "unitPrice": 210.00,
            "fee": 0.00,
            "currency": "USD",
            "account": {"name": "Brokerage"},
        },
        {
            "id": "tx3",
            "date": "2024-04-10T00:00:00Z",
            "type": "SELL",
            "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft"},
            "quantity": 2,
            "unitPrice": 410.00,
            "fee": 2.50,
            "currency": "USD",
            "account": {"name": "Brokerage"},
        },
    ]
}

# Ghostfolio v2 flat format — no nested period keys
PERF_DATA = {
    "performance": {
        "netPerformancePercentage": 0.10,
        "netPerformance": 800.00,
        "currentValueInBaseCurrency": 8800.00,
        "totalInvestment": 8000.00,
        "currentNetWorth": 8800.00,
    },
    "hasErrors": False,
}


def _auth():
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=AUTH_RESP)
    )


def _mock_holdings():
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": HOLDINGS_LIST})
    )


def _mock_performance():
    respx.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
        return_value=httpx.Response(200, json=PERF_DATA)
    )


def _mock_orders():
    respx.get(f"{BASE_URL}/api/v1/order").mock(return_value=httpx.Response(200, json=ORDERS_DATA))


# ══════════════════════════════════════════════════════════════════════════════
# MS01 — Portfolio summary → fee drag: gross = net + fees identity
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms01_portfolio_and_fee_drag_identity():
    """
    When both `get_portfolio_summary` and `get_fee_drag_analysis` are called
    for the same portfolio, the accounting identity gross = net + fees must hold.

    Multi-step consistency: tool chain produces non-contradictory numbers.
    """
    _auth()
    _mock_holdings()
    _mock_orders()
    _mock_performance()

    portfolio = await _get_portfolio_summary()
    assert portfolio["status"] == "ok"
    portfolio_total = portfolio["total_value"]

    fee_result = await _fee_drag("max")
    assert fee_result["status"] == "ok"

    # Accounting identity: gross_gain = net_gain + fees
    gross = fee_result.get("gross_gain_without_fees", 0)
    net = fee_result.get("net_gain_after_fees", 0)
    fees = fee_result.get("total_fees_paid", 0)
    assert abs(gross - (net + fees)) < 0.02, (
        f"MS01: Accounting identity failed. gross={gross} ≠ net({net}) + fees({fees})"
    )
    assert portfolio_total > 0, "MS01: Portfolio total must be positive"


# ══════════════════════════════════════════════════════════════════════════════
# MS02 — Holdings → diversification: sectors match holding sector data
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms02_holdings_and_diversification_sectors_consistent():
    """
    The sectors identified by `_analyze_diversification` must be a subset of
    the sector names defined in the holding records.
    Multi-step: portfolio → diversification must not invent sectors.
    """
    _auth()
    _mock_holdings()

    divers = await _analyze_diversification()
    assert divers["status"] == "ok"

    known_sectors = set()
    for h in HOLDINGS_LIST:
        for s in h.get("sectors", []):
            known_sectors.add(s["name"])

    for sector_entry in divers.get("sector_breakdown", []):
        sname = sector_entry.get("name", sector_entry.get("sector", ""))
        if sname and sname not in ("Unknown", "Other"):
            assert sname in known_sectors, (
                f"MS02: Diversification returned unknown sector '{sname}' "
                f"not in holdings data. Known: {known_sectors}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MS03 — Holdings → rebalancing: every trade symbol exists in holdings
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms03_rebalancing_trades_only_contain_holdings_symbols():
    """
    The rebalancing plan must only recommend buying/selling positions that
    ACTUALLY EXIST in the portfolio — it must never fabricate symbols.
    Multi-step: holdings → rebalancing — referential integrity check.
    """
    _auth()
    _mock_holdings()

    import agent.tools.rebalancing as reb_module

    with patch.object(reb_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "ok", "current_price": 175.00}
        plan = await _rebalancing_plan(0.55, 0.25, 0.15, 0.05)

    assert plan["status"] == "ok"

    known_symbols = {h["symbol"] for h in HOLDINGS_LIST}
    for trade in plan.get("trades", []):
        assert trade["symbol"] in known_symbols, (
            f"MS03: Rebalancing referenced symbol '{trade['symbol']}' "
            f"not in holdings. Known: {known_symbols}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MS04 — Holdings → market context: analysed positions are real holdings
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms04_market_context_positions_are_subset_of_holdings():
    """
    The `get_market_context_overlay` tool analyses each position in the portfolio.
    Every position_analysis entry must reference a symbol that actually exists
    in the current holdings — no phantom positions.
    """
    _auth()
    _mock_holdings()

    ctx = await _market_context("rising_rates")
    assert ctx["status"] == "ok"

    known_symbols = {h["symbol"] for h in HOLDINGS_LIST}
    for pos in ctx.get("position_analyses", []):
        assert pos["symbol"] in known_symbols, (
            f"MS04: Market context analysed '{pos['symbol']}' not in holdings"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MS05 — Holdings → health scorecard: grade reflects position count
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms05_health_scorecard_grade_reflects_position_count():
    """
    A portfolio with only 3 positions should receive a grade penalty for
    being undiversified. The scorecard must reflect the real position count.
    Multi-step: holdings + performance → health_scorecard — faithfulness check.
    """
    _auth()
    _mock_holdings()
    _mock_performance()

    sc = await _scorecard()
    assert sc["status"] == "ok"
    assert sc["position_count"] == len(HOLDINGS_LIST), (
        f"MS05: Scorecard reported {sc['position_count']} positions "
        f"but holdings has {len(HOLDINGS_LIST)}"
    )
    # 3 positions → should penalise (grade ≤ B, score ≤ 90)
    assert sc["score"] <= 90, (
        "MS05: 3-position portfolio must not score 100 (no diversity penalty applied)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MS06 — Transactions → patterns: buy count matches raw activity records
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms06_transaction_patterns_buy_count_matches_source():
    """
    The transaction patterns tool reports total_buy_transactions.
    This count must exactly equal the number of BUY-type activities in the
    raw orders data — no double-counting or skipped transactions.
    """
    _auth()
    _mock_orders()
    _mock_holdings()

    import agent.tools.transaction_patterns as tp_module

    with patch.object(tp_module._market, "get_quote", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = {"status": "price_unavailable", "current_price": None}
        patterns = await _transaction_patterns()

    assert patterns["status"] == "ok"
    expected_buys = sum(1 for a in ORDERS_DATA["activities"] if a["type"] == "BUY")
    assert patterns["total_buy_transactions"] == expected_buys, (
        f"MS06: Expected {expected_buys} BUY transactions; "
        f"patterns tool reported {patterns['total_buy_transactions']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MS07 — Proactive monitor first session: no changes reported
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms07_proactive_monitor_first_session_no_changes():
    """
    On the very first session (empty previous_snapshot_json), there are no
    prior allocations to compare against — changes_since_last_session must be [].
    The monitor should also build and return a current_snapshot for future use.
    """
    _auth()
    _mock_holdings()

    result = await _proactive_monitor(prev_snap_json="")

    assert result["status"] == "ok"
    assert result["changes_since_last_session"] == [], (
        "MS07: First session with no prior snapshot must report zero changes"
    )
    assert "current_snapshot" in result
    assert "allocations" in result["current_snapshot"]
    assert len(result["current_snapshot"]["allocations"]) == len(HOLDINGS_LIST)


# ══════════════════════════════════════════════════════════════════════════════
# MS08 — Proactive monitor with snapshot: portfolio value change detected
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms08_proactive_monitor_detects_value_change():
    """
    Multi-step: session 1 returns snapshot → session 2 uses that snapshot.
    If total portfolio value changed by ≥2% between sessions, it must be
    surfaced in changes_since_last_session.
    """
    _auth()
    _mock_holdings()

    # Previous snapshot: portfolio was worth $6,500 (current is ~$8,000 — 23% gain)
    total = sum(h["value"] for h in HOLDINGS_LIST)
    prev_allocs = {h["symbol"]: round(h["value"] / total * 100, 2) for h in HOLDINGS_LIST}
    prev_snapshot = {
        "total_value": 6500.00,
        "position_count": 3,
        "timestamp": "2024-01-01T00:00:00Z",
        "allocations": prev_allocs,
    }

    result = await _proactive_monitor(prev_snap_json=json.dumps(prev_snapshot))

    assert result["status"] == "ok"
    value_changes = [
        c for c in result["changes_since_last_session"] if c.get("direction") == "VALUE_CHANGE"
    ]
    assert len(value_changes) >= 1, (
        "MS08: A 23% portfolio value increase must be surfaced as a VALUE_CHANGE"
    )
    change = value_changes[0]
    assert change["change_pp"] > 0, "MS08: Portfolio increased — change_pp must be positive"


# ══════════════════════════════════════════════════════════════════════════════
# MS09 — Proactive monitor: new concentration breach detected
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms09_proactive_monitor_detects_concentration_breach():
    """
    If a position was below the concentration threshold last session but is now
    above it (due to price appreciation), the monitor must emit a
    NEW_CONCENTRATION_BREACH alert.
    """
    _auth()
    # AAPL = $1750 out of ~$8000 = ~21.9% — above 20% threshold
    _mock_holdings()

    total = sum(h["value"] for h in HOLDINGS_LIST)
    prev_allocs = {h["symbol"]: round(h["value"] / total * 100, 2) for h in HOLDINGS_LIST}
    # Artificially lower AAPL's previous allocation to below threshold
    prev_allocs["AAPL"] = 15.0

    prev_snapshot = {
        "total_value": total,
        "position_count": 3,
        "timestamp": "2024-01-01T00:00:00Z",
        "allocations": prev_allocs,
    }

    result = await _proactive_monitor(prev_snap_json=json.dumps(prev_snapshot))
    assert result["status"] == "ok"

    breach_alerts = [a for a in result["alerts"] if a.get("type") == "NEW_CONCENTRATION_BREACH"]
    assert len(breach_alerts) >= 1, (
        "MS09: AAPL crossing the concentration threshold must generate "
        "a NEW_CONCENTRATION_BREACH alert"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MS10 — Fee drag cross-check: fees from raw orders match tool total
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms10_fee_drag_total_matches_sum_of_order_fees():
    """
    The fee_drag tool must sum all fees from the orders endpoint correctly.
    Cross-validate: manually summing fees from ORDERS_DATA must equal
    the total_fees_paid returned by the tool.
    """
    _auth()
    _mock_orders()
    _mock_performance()
    _mock_holdings()

    result = await _fee_drag("max")
    assert result["status"] == "ok"

    expected_fees = sum(a.get("fee", 0) or 0 for a in ORDERS_DATA["activities"])
    assert abs(result["total_fees_paid"] - expected_fees) < 0.01, (
        f"MS10: fee_drag reported ${result['total_fees_paid']:.2f} total fees "
        f"but raw orders sum to ${expected_fees:.2f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MS11 — Health scorecard with 12 well-diversified positions → grade B+
# ══════════════════════════════════════════════════════════════════════════════


@respx.mock
async def test_ms11_health_scorecard_well_diversified_earns_high_grade():
    """
    A well-diversified portfolio (12 positions, each ~8%, multiple sectors,
    includes bonds) should score ≥65 (grade B or better).
    Multi-step: holdings + 2x performance → scorecard synthesis.
    """
    _auth()
    sectors = [
        "Technology",
        "Healthcare",
        "Financial Services",
        "Consumer Defensive",
        "Industrials",
        "Energy",
    ]
    diversified_holdings = [
        {
            "symbol": f"ETF{i:02d}",
            "name": f"ETF {i}",
            "quantity": 10,
            "value": 1000.00,
            "valueInBaseCurrency": 1000.00,
            "currency": "USD",
            "assetClass": "EQUITY" if i < 10 else "BOND",
            "assetSubClass": "ETF",
            "sectors": [{"name": sectors[i % len(sectors)], "weight": 1.0}],
            "countries": [{"name": "United States", "weight": 1.0}],
        }
        for i in range(12)
    ]
    respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
        return_value=httpx.Response(200, json={"holdings": diversified_holdings})
    )
    _mock_performance()

    sc = await _scorecard()
    assert sc["status"] == "ok"
    assert sc["position_count"] == 12
    assert sc["score"] >= 65, (
        f"MS11: Well-diversified 12-position portfolio should score ≥65, "
        f"got score={sc['score']}, grade={sc['grade']}"
    )
    assert sc["grade"] in ("A", "B"), (
        f"MS11: 12-position portfolio should earn grade A or B, got {sc['grade']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MS12 — Market context + hedges: suggested_hedges non-empty (all 4 themes)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("theme", ["rising_rates", "recession", "inflation", "bull_market"])
@respx.mock
async def test_ms12_market_context_suggested_hedges_non_empty(theme: str):
    """
    For every valid macro theme, the market context tool must return at least
    one suggested hedge instrument. These are pulled from a static map —
    validating that the map contains entries for all 4 supported themes.
    Multi-step: holdings → macro sensitivity map → hedge recommendations.
    """
    _auth()
    _mock_holdings()

    ctx = await _market_context(theme)
    assert ctx["status"] == "ok"
    assert ctx["macro_theme"] == theme
    hedges = ctx.get("suggested_hedges", [])
    assert len(hedges) >= 1, (
        f"MS12: Theme '{theme}' must have at least one suggested hedge; got {hedges}"
    )
