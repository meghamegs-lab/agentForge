"""
Unit tests for agent/tools/retirement.py — the 5 FIRE Goal Tracker tools.

Strategy:
  - All DB calls are patched via unittest.mock.patch on the agent.db.retirement module.
  - All Ghostfolio HTTP calls are patched on the GhostfolioClient.
  - All FRED calls are patched on FredClient.
  - No real network or database connections are made.
  - We test the private _impl functions directly (not the @tool wrappers) to
    avoid LangChain overhead — same pattern as the existing test_tools.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Fake DB URL used to bypass the `if not settings.database_url` guard in all tools.
_FAKE_DB_URL = "postgresql+asyncpg://test:test@localhost/testdb"

from agent.tools.retirement import (
    _calculate_retirement_projection,
    _get_fire_progress,
    _get_macro_data,
    _get_retirement_goal,
    _project_years_to_fire,
    _set_retirement_goal,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_GOAL = {
    "user_id": "test-user",
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

SAMPLE_HOLDINGS = {
    "holdings": [
        {"symbol": "VTI", "valueInBaseCurrency": 500_000.0},
        {"symbol": "AAPL", "valueInBaseCurrency": 200_000.0},
    ]
}


@contextmanager
def _patch_db_get(return_value):
    """Patch get_goal AND settings.database_url so the early-exit guard doesn't fire."""
    with (
        patch("agent.tools.retirement.settings") as mock_settings,
        patch(
            "agent.tools.retirement.get_goal",
            new_callable=AsyncMock,
            return_value=return_value,
        ),
    ):
        mock_settings.database_url = _FAKE_DB_URL
        yield


@contextmanager
def _patch_db_upsert(return_value):
    """Patch upsert_goal AND settings.database_url so the early-exit guard doesn't fire."""
    with (
        patch("agent.tools.retirement.settings") as mock_settings,
        patch(
            "agent.tools.retirement.upsert_goal",
            new_callable=AsyncMock,
            return_value=return_value,
        ),
    ):
        mock_settings.database_url = _FAKE_DB_URL
        yield


def _patch_ghostfolio(holdings=None):
    """Patch GhostfolioClient.get_portfolio_holdings."""
    holdings = holdings or SAMPLE_HOLDINGS
    mock_client = MagicMock()
    mock_client.get_portfolio_holdings = AsyncMock(return_value=holdings)
    return patch("agent.tools.retirement.get_shared_client", return_value=mock_client)


def _patch_fred(inflation=0.032, risk_free=0.045):
    """Patch FredClient to return fixed macro data."""
    mock_fred = MagicMock()
    mock_fred.get_current_inflation_rate = AsyncMock(
        return_value={
            "status": "ok",
            "inflation_rate_yoy": inflation,
            "inflation_rate_pct": inflation * 100,
            "latest_cpi_date": "2026-01-01",
        }
    )
    mock_fred.get_risk_free_rate = AsyncMock(
        return_value={
            "status": "ok",
            "risk_free_rate": risk_free,
            "risk_free_rate_pct": risk_free * 100,
            "rate_date": "2026-02-28",
        }
    )
    return patch("agent.tools.retirement.get_shared_fred_client", return_value=mock_fred)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: get_retirement_goal
# ══════════════════════════════════════════════════════════════════════════════


class TestGetRetirementGoal:
    async def test_returns_ok_when_goal_exists(self):
        with _patch_db_get(SAMPLE_GOAL):
            result = await _get_retirement_goal("test-user")
        assert result["status"] == "ok"
        assert "goal" in result

    async def test_goal_contains_fire_number(self):
        with _patch_db_get(SAMPLE_GOAL):
            result = await _get_retirement_goal("test-user")
        assert result["goal"]["fire_number"] == pytest.approx(2_000_000.0)

    async def test_not_found_when_no_goal(self):
        with _patch_db_get(None):
            result = await _get_retirement_goal("new-user")
        assert result["status"] == "not_found"
        assert "set_retirement_goal" in result["message"]

    async def test_result_includes_data_timestamp(self):
        with _patch_db_get(SAMPLE_GOAL):
            result = await _get_retirement_goal("test-user")
        assert "data_timestamp" in result

    async def test_missing_database_url_returns_error(self):
        with patch("agent.tools.retirement.settings") as mock_settings:
            mock_settings.database_url = ""
            result = await _get_retirement_goal("user")
        assert result["status"] == "error"
        assert "Database" in result["error"]

    async def test_db_exception_returns_structured_error(self):
        with (
            patch("agent.tools.retirement.settings") as mock_settings,
            patch(
                "agent.tools.retirement.get_goal",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            mock_settings.database_url = _FAKE_DB_URL
            result = await _get_retirement_goal("test-user")
        assert result["status"] == "error"
        assert "connection refused" in result["error"]


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: set_retirement_goal
# ══════════════════════════════════════════════════════════════════════════════


class TestSetRetirementGoal:
    async def test_returns_saved_status(self):
        with _patch_db_upsert(SAMPLE_GOAL):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000,
                user_id="test-user",
            )
        assert result["status"] == "saved"

    async def test_saved_goal_contains_fire_number(self):
        with _patch_db_upsert(SAMPLE_GOAL):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000,
                user_id="test-user",
            )
        assert result["goal"]["fire_number"] == pytest.approx(2_000_000.0)

    async def test_validation_error_when_age_invalid(self):
        """target_retirement_age must be greater than current_age."""
        with _patch_db_upsert(SAMPLE_GOAL):
            result = await _set_retirement_goal(
                current_age=50,
                target_retirement_age=40,  # ← invalid: target < current
                target_annual_spending=80_000,
                user_id="test-user",
            )
        assert result["status"] == "validation_error"
        assert len(result["errors"]) > 0

    async def test_validation_error_for_zero_spending(self):
        with _patch_db_upsert(SAMPLE_GOAL):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=0,  # ← invalid
                user_id="test-user",
            )
        assert result["status"] == "validation_error"

    async def test_validation_error_for_bad_swr(self):
        with _patch_db_upsert(SAMPLE_GOAL):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000,
                safe_withdrawal_rate=0.5,  # ← 50% is nonsensical
                user_id="test-user",
            )
        assert result["status"] == "validation_error"

    async def test_summary_message_included_in_result(self):
        with _patch_db_upsert(SAMPLE_GOAL):
            result = await _set_retirement_goal(
                current_age=35,
                target_retirement_age=50,
                target_annual_spending=80_000,
                user_id="test-user",
            )
        assert "summary" in result
        assert "$" in result["summary"]  # should include dollar amount


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: get_fire_progress
# ══════════════════════════════════════════════════════════════════════════════


class TestGetFireProgress:
    async def test_returns_ok_with_portfolio(self):
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio():
            result = await _get_fire_progress("test-user")
        assert result["status"] == "ok"

    async def test_progress_pct_is_correct(self):
        """Portfolio = $700K, FIRE number = $2M → 35%"""
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio():
            result = await _get_fire_progress("test-user")
        # $700K portfolio (500K VTI + 200K AAPL) / $2M FIRE number = 35%
        assert result["progress_pct"] == pytest.approx(35.0, abs=0.1)

    async def test_shortfall_is_correct(self):
        """Shortfall = $2M - $700K = $1.3M"""
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio():
            result = await _get_fire_progress("test-user")
        assert result["shortfall"] == pytest.approx(1_300_000.0, abs=1.0)

    async def test_goal_required_when_no_goal_set(self):
        with _patch_db_get(None):
            result = await _get_fire_progress("no-goal-user")
        assert result["status"] == "goal_required"
        assert "set_retirement_goal" in result["message"]

    async def test_empty_portfolio_gives_zero_progress(self):
        empty_holdings = {"holdings": []}
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(empty_holdings):
            result = await _get_fire_progress("test-user")
        assert result["progress_pct"] == pytest.approx(0.0)
        assert result["current_portfolio_value"] == pytest.approx(0.0)

    async def test_achieved_status_when_portfolio_exceeds_fire_number(self):
        rich_goal = dict(SAMPLE_GOAL, fire_number=500_000.0)  # FIRE number < portfolio
        with _patch_db_get(rich_goal), _patch_ghostfolio():
            result = await _get_fire_progress("rich-user")
        assert result["progress_status"] == "achieved"
        assert result["surplus"] > 0

    async def test_data_timestamp_in_result(self):
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio():
            result = await _get_fire_progress("test-user")
        assert "data_timestamp" in result


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4: calculate_retirement_projection  (math unit tests)
# ══════════════════════════════════════════════════════════════════════════════


class TestProjectYearsToFire:
    """Direct tests of the projection math function — no mocks needed."""

    def test_zero_years_when_already_at_fire_number(self):
        years = _project_years_to_fire(
            current_value=2_000_000.0,
            fire_number=2_000_000.0,
            monthly_return=0.07 / 12,
            monthly_contribution=0.0,
        )
        assert years == pytest.approx(0.0)

    def test_positive_years_when_below_fire_number(self):
        years = _project_years_to_fire(
            current_value=500_000.0,
            fire_number=2_000_000.0,
            monthly_return=0.07 / 12,
            monthly_contribution=2_000.0,
        )
        assert years > 0
        assert years < 50  # reasonable bound

    def test_infinite_years_with_no_growth_and_no_contribution(self):
        years = _project_years_to_fire(
            current_value=100_000.0,
            fire_number=2_000_000.0,
            monthly_return=0.0,
            monthly_contribution=0.0,
        )
        assert years == float("inf")

    def test_higher_contribution_gives_fewer_years(self):
        years_low = _project_years_to_fire(500_000.0, 2_000_000.0, 0.07 / 12, 1_000.0)
        years_high = _project_years_to_fire(500_000.0, 2_000_000.0, 0.07 / 12, 3_000.0)
        assert years_high < years_low


class TestCalculateRetirementProjection:
    async def test_returns_ok_status(self):
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), _patch_fred():
            result = await _calculate_retirement_projection("test-user")
        assert result["status"] == "ok"

    async def test_projected_age_is_positive(self):
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), _patch_fred():
            result = await _calculate_retirement_projection("test-user")
        assert result["projected_retirement_age"] is not None
        assert result["projected_retirement_age"] > 35

    async def test_what_if_scenario_changes_projection(self):
        """Doubling monthly contribution should bring projected age closer to target."""
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), _patch_fred():
            base = await _calculate_retirement_projection("test-user", monthly_contribution_override=-1.0)

        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), _patch_fred():
            boosted = await _calculate_retirement_projection("test-user", monthly_contribution_override=5_000.0)

        if base["projected_retirement_age"] and boosted["projected_retirement_age"]:
            assert boosted["projected_retirement_age"] <= base["projected_retirement_age"]

    async def test_goal_required_when_no_goal(self):
        with _patch_db_get(None):
            result = await _calculate_retirement_projection("no-goal-user")
        assert result["status"] == "goal_required"

    async def test_real_return_is_less_than_nominal(self):
        """Real return = (1+nominal)/(1+inflation) - 1 < nominal."""
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), _patch_fred(inflation=0.032):
            result = await _calculate_retirement_projection("test-user")
        assert result["real_annual_return_pct"] < result["expected_annual_return_nominal"]

    async def test_disclaimer_present(self):
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), _patch_fred():
            result = await _calculate_retirement_projection("test-user")
        assert "disclaimer" in result
        assert "⚠️" in result["disclaimer"]

    async def test_fred_fallback_on_error(self):
        """Even if FRED fails, projection should still succeed with defaults."""
        mock_fred = MagicMock()
        mock_fred.get_current_inflation_rate = AsyncMock(return_value={"status": "error", "error": "FRED down"})
        mock_fred.get_risk_free_rate = AsyncMock(return_value={"status": "error", "error": "FRED down"})
        with _patch_db_get(SAMPLE_GOAL), _patch_ghostfolio(), patch(
            "agent.tools.retirement.get_shared_fred_client", return_value=mock_fred
        ):
            result = await _calculate_retirement_projection("test-user")
        # Should succeed with default macro values
        assert result["status"] == "ok"
        assert result["macro_data_status"] == "default"


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5: get_macro_data
# ══════════════════════════════════════════════════════════════════════════════


class TestGetMacroData:
    async def test_returns_ok_status(self):
        with _patch_fred():
            result = await _get_macro_data()
        assert result["status"] == "ok"

    async def test_inflation_section_present(self):
        with _patch_fred(inflation=0.032):
            result = await _get_macro_data()
        assert "inflation" in result
        assert result["inflation"]["rate_pct"] == pytest.approx(3.2, abs=0.01)

    async def test_treasury_section_present(self):
        with _patch_fred(risk_free=0.045):
            result = await _get_macro_data()
        assert "treasury_10y" in result
        assert result["treasury_10y"]["yield_pct"] == pytest.approx(4.5, abs=0.01)

    async def test_derived_real_return_is_computed(self):
        """Real return at 7% nominal with 3.2% inflation: (1.07/1.032) - 1 ≈ 3.68%"""
        with _patch_fred(inflation=0.032):
            result = await _get_macro_data()
        expected_real = (1.07 / 1.032 - 1) * 100
        assert result["derived"]["real_return_at_7pct_nominal_pct"] == pytest.approx(
            expected_real, abs=0.1
        )

    async def test_fred_failure_still_returns_defaults(self):
        """If FRED is down, get_macro_data returns defaults — not an error."""
        mock_fred = MagicMock()
        mock_fred.get_current_inflation_rate = AsyncMock(return_value={"status": "error", "error": "timeout"})
        mock_fred.get_risk_free_rate = AsyncMock(return_value={"status": "error", "error": "timeout"})
        with patch("agent.tools.retirement.get_shared_fred_client", return_value=mock_fred):
            result = await _get_macro_data()
        assert result["status"] == "ok"  # graceful degradation
        assert "inflation_error" in result
        assert "treasury_error" in result

    async def test_data_timestamp_in_result(self):
        with _patch_fred():
            result = await _get_macro_data()
        assert "data_timestamp" in result

    async def test_source_is_fred(self):
        with _patch_fred():
            result = await _get_macro_data()
        assert "FRED" in result["source"]
