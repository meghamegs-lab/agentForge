"""
evals/test_retirement_eval.py — FIRE Goal Tracker Eval Suite
==============================================================
Eval IDs: FIRE-1 through FIRE-10

"Does the FIRE Goal Tracker return correct, safe, and coherent results?"

Test categories:
  Correctness (FIRE-1 to FIRE-4): Math accuracy — FIRE number, progress %, projection, inflation
  Edge cases  (FIRE-5 to FIRE-8): No goal, empty portfolio, FRED down, impossible goal
  Adversarial (FIRE-9 to FIRE-10): "Retire tomorrow", nonsensical SWR — no crash, clean errors

All calls are mocked — no real DB, network, or FRED API key required.

Tolerance:
  Percentages: ±0.1 percentage points (the projection math uses floating point)
  Dollar amounts: ±$1.00
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.retirement import (
    _calculate_retirement_projection,
    _get_fire_progress,
    _get_macro_data,
    _get_retirement_goal,
    _project_years_to_fire,
    _set_retirement_goal,
)

PCT_TOL = 0.1   # ±0.1 percentage points
DOLLAR_TOL = 1.0  # ±$1.00

# ── Shared test data ───────────────────────────────────────────────────────────

GOAL_35_50 = {
    "user_id": "eval-user",
    "current_age": 35,
    "target_retirement_age": 50,
    "target_annual_spending": 80_000.0,
    "safe_withdrawal_rate": 0.04,
    "monthly_contribution": 2_000.0,
    "expected_annual_return": 0.07,
    "social_security_estimate": 0.0,
    "fire_number": 2_000_000.0,   # 80_000 / 0.04
    "years_to_target": 15,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}

PORTFOLIO_700K = {
    "holdings": [
        {"symbol": "VTI", "valueInBaseCurrency": 500_000.0},
        {"symbol": "AAPL", "valueInBaseCurrency": 200_000.0},
    ]
}

EMPTY_PORTFOLIO = {"holdings": []}


def _patch_db_get(goal):
    return patch("agent.tools.retirement.get_goal", new_callable=AsyncMock, return_value=goal)

def _patch_db_upsert(goal):
    return patch("agent.tools.retirement.upsert_goal", new_callable=AsyncMock, return_value=goal)

def _patch_ghostfolio(holdings):
    mock_client = MagicMock()
    mock_client.get_portfolio_holdings = AsyncMock(return_value=holdings)
    return patch("agent.tools.retirement.get_shared_client", return_value=mock_client)

def _patch_fred(inflation=0.032, risk_free=0.045, fail=False):
    mock_fred = MagicMock()
    if fail:
        mock_fred.get_current_inflation_rate = AsyncMock(return_value={"status": "error", "error": "FRED down"})
        mock_fred.get_risk_free_rate = AsyncMock(return_value={"status": "error", "error": "FRED down"})
    else:
        mock_fred.get_current_inflation_rate = AsyncMock(
            return_value={"status": "ok", "inflation_rate_yoy": inflation, "inflation_rate_pct": inflation * 100, "latest_cpi_date": "2026-01-01"}
        )
        mock_fred.get_risk_free_rate = AsyncMock(
            return_value={"status": "ok", "risk_free_rate": risk_free, "risk_free_rate_pct": risk_free * 100, "rate_date": "2026-02-28"}
        )
    return patch("agent.tools.retirement.get_shared_fred_client", return_value=mock_fred)


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-1: CORRECTNESS — FIRE number formula is exact
# ══════════════════════════════════════════════════════════════════════════════

class TestFire1FireNumberFormula:
    """
    FIRE number = target_annual_spending / safe_withdrawal_rate.
    This is the foundational math — must be exactly right.
    """

    async def test_fire1_four_percent_rule(self):
        """$80K/yr ÷ 4% SWR = $2,000,000 FIRE number."""
        saved_goal = dict(GOAL_35_50)  # already has fire_number = 2_000_000
        with _patch_db_upsert(saved_goal):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000.0,
                safe_withdrawal_rate=0.04,
                user_id="eval-user",
            )
        assert result["status"] == "saved"
        assert result["goal"]["fire_number"] == pytest.approx(2_000_000.0, abs=DOLLAR_TOL)

    async def test_fire1_three_percent_rule_for_early_retirement(self):
        """$60K/yr ÷ 3.5% SWR = $1,714,286 FIRE number."""
        goal_3pct = dict(GOAL_35_50, target_annual_spending=60_000.0, safe_withdrawal_rate=0.035,
                         fire_number=round(60_000.0 / 0.035, 2))
        with _patch_db_upsert(goal_3pct):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=60_000.0,
                safe_withdrawal_rate=0.035,
                user_id="eval-user",
            )
        expected = 60_000.0 / 0.035
        assert result["goal"]["fire_number"] == pytest.approx(expected, abs=DOLLAR_TOL)

    async def test_fire1_fire_number_in_goal_read(self):
        """Stored goal's fire_number is correctly computed on read."""
        with _patch_db_get(GOAL_35_50):
            result = await _get_retirement_goal("eval-user")
        assert result["goal"]["fire_number"] == pytest.approx(2_000_000.0, abs=DOLLAR_TOL)


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-2: CORRECTNESS — Progress percentage math is exact
# ══════════════════════════════════════════════════════════════════════════════

class TestFire2ProgressPercentage:
    """
    Progress % = current_portfolio_value / fire_number × 100.
    Tests different portfolio/goal combinations for exactness.
    """

    async def test_fire2_35pct_progress(self):
        """$700K portfolio / $2M FIRE number = 35.0%"""
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(PORTFOLIO_700K):
            result = await _get_fire_progress("eval-user")
        assert result["progress_pct"] == pytest.approx(35.0, abs=PCT_TOL)

    async def test_fire2_shortfall_arithmetic(self):
        """Shortfall = FIRE number − portfolio value = $2M − $700K = $1.3M"""
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(PORTFOLIO_700K):
            result = await _get_fire_progress("eval-user")
        assert result["shortfall"] == pytest.approx(1_300_000.0, abs=DOLLAR_TOL)

    async def test_fire2_100pct_when_portfolio_equals_fire_number(self):
        """Progress = 100% when portfolio exactly equals FIRE number."""
        exact_goal = dict(GOAL_35_50, fire_number=700_000.0)
        with _patch_db_get(exact_goal), _patch_ghostfolio(PORTFOLIO_700K):
            result = await _get_fire_progress("eval-user")
        assert result["progress_pct"] == pytest.approx(100.0, abs=PCT_TOL)
        assert result["shortfall"] == pytest.approx(0.0, abs=DOLLAR_TOL)

    async def test_fire2_surplus_when_over_fire_number(self):
        """Surplus is computed when portfolio exceeds FIRE number."""
        small_goal = dict(GOAL_35_50, fire_number=500_000.0)
        with _patch_db_get(small_goal), _patch_ghostfolio(PORTFOLIO_700K):
            result = await _get_fire_progress("eval-user")
        assert result["surplus"] == pytest.approx(200_000.0, abs=DOLLAR_TOL)
        assert result["progress_status"] == "achieved"


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-3: CORRECTNESS — Projection formula math
# ══════════════════════════════════════════════════════════════════════════════

class TestFire3ProjectionMath:
    """Tests the _project_years_to_fire compound growth formula directly."""

    def test_fire3_no_growth_no_contribution_is_infinity(self):
        """Without growth or contributions, FIRE is unreachable."""
        years = _project_years_to_fire(
            current_value=500_000.0,
            fire_number=2_000_000.0,
            monthly_return=0.0,
            monthly_contribution=0.0,
        )
        assert years == float("inf")

    def test_fire3_already_at_fire_number_is_zero(self):
        """If current_value == fire_number, years = 0."""
        years = _project_years_to_fire(2_000_000.0, 2_000_000.0, 0.07/12, 2_000.0)
        assert years == pytest.approx(0.0)

    def test_fire3_more_contributions_means_fewer_years(self):
        """Increasing monthly contributions monotonically reduces years to FIRE."""
        y1 = _project_years_to_fire(500_000.0, 2_000_000.0, 0.07/12, 1_000.0)
        y2 = _project_years_to_fire(500_000.0, 2_000_000.0, 0.07/12, 2_000.0)
        y3 = _project_years_to_fire(500_000.0, 2_000_000.0, 0.07/12, 5_000.0)
        assert y3 < y2 < y1

    def test_fire3_higher_return_means_fewer_years(self):
        """Higher expected return reduces years to FIRE."""
        y_low = _project_years_to_fire(500_000.0, 2_000_000.0, 0.05/12, 2_000.0)
        y_high = _project_years_to_fire(500_000.0, 2_000_000.0, 0.10/12, 2_000.0)
        assert y_high < y_low

    async def test_fire3_projection_returns_non_negative_age(self):
        """projected_retirement_age must always be ≥ current_age."""
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(PORTFOLIO_700K), _patch_fred():
            result = await _calculate_retirement_projection("eval-user")
        if result.get("projected_retirement_age") is not None:
            assert result["projected_retirement_age"] >= GOAL_35_50["current_age"]


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-4: CORRECTNESS — Inflation adjustment
# ══════════════════════════════════════════════════════════════════════════════

class TestFire4InflationAdjustment:
    """Real return = (1+nominal)/(1+inflation) − 1.  Must be < nominal return."""

    async def test_fire4_real_return_less_than_nominal(self):
        """With 3.2% inflation, real return at 7% nominal must be < 7%."""
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(PORTFOLIO_700K), _patch_fred(inflation=0.032):
            result = await _calculate_retirement_projection("eval-user")
        assert result["real_annual_return_pct"] < result["expected_annual_return_nominal"]

    async def test_fire4_real_return_formula_is_correct(self):
        """(1.07 / 1.032) - 1 = 0.03682... → 3.68%"""
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(PORTFOLIO_700K), _patch_fred(inflation=0.032):
            result = await _calculate_retirement_projection("eval-user")
        expected_real_pct = ((1.07 / 1.032) - 1) * 100
        assert result["real_annual_return_pct"] == pytest.approx(expected_real_pct, abs=0.01)

    async def test_fire4_macro_data_inflation_pct_is_correct(self):
        """get_macro_data returns 3.2% inflation when FRED returns 0.032."""
        with _patch_fred(inflation=0.032):
            result = await _get_macro_data()
        assert result["inflation"]["rate_pct"] == pytest.approx(3.2, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-5: EDGE CASE — No goal set
# ══════════════════════════════════════════════════════════════════════════════

class TestFire5NoGoalSet:
    """All tools that require a goal must return a clean 'goal_required' status."""

    async def test_fire5_get_progress_returns_goal_required(self):
        with _patch_db_get(None):
            result = await _get_fire_progress("no-goal-user")
        assert result["status"] == "goal_required"
        assert "set_retirement_goal" in result["message"]

    async def test_fire5_projection_returns_goal_required(self):
        with _patch_db_get(None):
            result = await _calculate_retirement_projection("no-goal-user")
        assert result["status"] == "goal_required"

    async def test_fire5_get_goal_returns_not_found(self):
        with _patch_db_get(None):
            result = await _get_retirement_goal("no-goal-user")
        assert result["status"] == "not_found"

    async def test_fire5_no_crash_no_exception(self):
        """None of these should raise — they must return dicts."""
        with _patch_db_get(None):
            try:
                r1 = await _get_fire_progress("x")
                r2 = await _calculate_retirement_projection("x")
                r3 = await _get_retirement_goal("x")
                assert isinstance(r1, dict)
                assert isinstance(r2, dict)
                assert isinstance(r3, dict)
            except Exception as exc:
                pytest.fail(f"Tool raised instead of returning a safe dict: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-6: EDGE CASE — Empty portfolio
# ══════════════════════════════════════════════════════════════════════════════

class TestFire6EmptyPortfolio:
    """Empty portfolio must return 0% progress, not a crash or NaN."""

    async def test_fire6_progress_pct_is_zero(self):
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(EMPTY_PORTFOLIO):
            result = await _get_fire_progress("eval-user")
        assert result["progress_pct"] == pytest.approx(0.0)

    async def test_fire6_portfolio_value_is_zero(self):
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(EMPTY_PORTFOLIO):
            result = await _get_fire_progress("eval-user")
        assert result["current_portfolio_value"] == pytest.approx(0.0)

    async def test_fire6_no_division_by_zero(self):
        """Zero portfolio value must not produce NaN or raise ZeroDivisionError."""
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(EMPTY_PORTFOLIO):
            result = await _get_fire_progress("eval-user")
        assert math.isfinite(result["progress_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-7: EDGE CASE — FRED API is unavailable
# ══════════════════════════════════════════════════════════════════════════════

class TestFire7FredUnavailable:
    """When FRED is down, tools must degrade gracefully using defaults."""

    async def test_fire7_projection_succeeds_with_defaults(self):
        with _patch_db_get(GOAL_35_50), _patch_ghostfolio(PORTFOLIO_700K), _patch_fred(fail=True):
            result = await _calculate_retirement_projection("eval-user")
        assert result["status"] == "ok"
        assert result["macro_data_status"] == "default"

    async def test_fire7_macro_data_returns_ok_with_defaults(self):
        with _patch_fred(fail=True):
            result = await _get_macro_data()
        assert result["status"] == "ok"
        assert result["inflation"]["data_available"] is False
        assert "inflation_error" in result

    async def test_fire7_default_inflation_is_3pct(self):
        """Default inflation when FRED is down = 3.0%"""
        with _patch_fred(fail=True):
            result = await _get_macro_data()
        # note: inflation section still has a fallback pct
        assert result["inflation"]["rate_pct"] == pytest.approx(3.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-8: EDGE CASE — Impossible goal (already past target age)
# ══════════════════════════════════════════════════════════════════════════════

class TestFire8ImpossibleGoal:
    """Validation must catch impossible goals before any DB write."""

    async def test_fire8_target_age_less_than_current_age(self):
        """target_retirement_age < current_age must return validation_error."""
        with _patch_db_upsert(GOAL_35_50):
            result = await _set_retirement_goal(
                current_age=55,
                target_retirement_age=40,  # impossible
                target_annual_spending=80_000,
                user_id="eval-user",
            )
        assert result["status"] == "validation_error"

    async def test_fire8_negative_spending_rejected(self):
        with _patch_db_upsert(GOAL_35_50):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=-10_000,  # invalid
                user_id="eval-user",
            )
        assert result["status"] == "validation_error"

    async def test_fire8_validation_never_raises(self):
        """Validation errors must return dicts, never exceptions."""
        try:
            with _patch_db_upsert(GOAL_35_50):
                result = await _set_retirement_goal(
                    current_age=200,  # beyond valid range
                    target_retirement_age=300,
                    target_annual_spending=-1,
                    safe_withdrawal_rate=9.9,
                    user_id="eval-user",
                )
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"set_retirement_goal raised instead of returning validation_error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-9: ADVERSARIAL — "Retire tomorrow" (years_to_target = 0 or negative)
# ══════════════════════════════════════════════════════════════════════════════

class TestFire9RetireTomorrow:
    """A goal with target_age = current_age + 1 must not crash or produce NaN."""

    async def test_fire9_imminent_goal_no_crash(self):
        imminent_goal = dict(
            GOAL_35_50,
            target_retirement_age=36,  # 1 year away
            years_to_target=1,
        )
        with _patch_db_get(imminent_goal), _patch_ghostfolio(PORTFOLIO_700K), _patch_fred():
            try:
                result = await _calculate_retirement_projection("eval-user")
                assert isinstance(result, dict)
                assert result["status"] == "ok"
            except Exception as exc:
                pytest.fail(f"Raised on imminent goal: {exc}")

    async def test_fire9_progress_pct_is_finite(self):
        imminent_goal = dict(GOAL_35_50, target_retirement_age=36, years_to_target=1)
        with _patch_db_get(imminent_goal), _patch_ghostfolio(PORTFOLIO_700K):
            result = await _get_fire_progress("eval-user")
        assert math.isfinite(result["progress_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# FIRE-10: ADVERSARIAL — Nonsensical SWR (0.5 = 50%)
# ══════════════════════════════════════════════════════════════════════════════

class TestFire10NonsensicalSWR:
    """A 50% SWR would give a nonsensically small FIRE number — must be rejected."""

    async def test_fire10_50pct_swr_fails_validation(self):
        with _patch_db_upsert(GOAL_35_50):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000,
                safe_withdrawal_rate=0.50,  # 50% — nonsensical
                user_id="eval-user",
            )
        assert result["status"] == "validation_error"

    async def test_fire10_1pct_swr_is_valid_lower_bound(self):
        """1% SWR (ultra-conservative) is the minimum valid value."""
        goal_1pct = dict(GOAL_35_50, fire_number=80_000.0 / 0.01, safe_withdrawal_rate=0.01)
        with _patch_db_upsert(goal_1pct):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000,
                safe_withdrawal_rate=0.01,
                user_id="eval-user",
            )
        assert result["status"] == "saved"
        assert result["goal"]["fire_number"] == pytest.approx(8_000_000.0, abs=DOLLAR_TOL)
