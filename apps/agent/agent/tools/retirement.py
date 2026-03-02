# LangChain tools for the FIRE (Financial Independence, Retire Early) Goal Tracker.
"""
FIRE Goal Tracker — LangChain Tools (5 tools)
==============================================
These tools let the agent answer retirement planning questions by combining:
  - The user's stored retirement goal (from Postgres via agent/db/retirement.py)
  - The user's live portfolio value (from Ghostfolio via get_portfolio_holdings)
  - Current macroeconomic data (from FRED API via agent/clients/fred.py)

Tool list:
  1. get_retirement_goal           — Read stored FIRE goal from DB
  2. set_retirement_goal           — Create or update FIRE goal in DB
  3. get_fire_progress             — Current progress toward FIRE (% + shortfall)
  4. calculate_retirement_projection — Full projection with inflation & contributions
  5. get_macro_data                — FRED macro: inflation rate + 10Y Treasury yield

Math reference:
  FIRE number    = target_annual_spending / safe_withdrawal_rate
                   e.g. $80,000 / 0.04 = $2,000,000
  Progress %     = current_portfolio / fire_number × 100
  FV (compound)  = PV × (1+r)^n + PMT × [(1+r)^n - 1] / r
  Solve for n    = log((FV×r + PMT) / (PV×r + PMT)) / log(1+r)
                   where r = monthly_return, PMT = monthly_contribution

All tools follow the project pattern:
  - Private _async_impl() contains the business logic
  - @tool wrapper delegates to the impl
  - Never raise — always return {"status": "error", "error": "..."}
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.tools import tool

from agent.clients.fred import get_shared_fred_client
from agent.clients.ghostfolio import GhostfolioError, get_shared_client
from agent.config import settings
from agent.db.retirement import (
    delete_goal,
    get_goal,
    upsert_goal,
    validate_goal_fields,
)

log = structlog.get_logger()

# Default macro assumptions used when FRED is unavailable.
_DEFAULT_INFLATION = 0.03  # 3.0% annual inflation
_DEFAULT_RISK_FREE = 0.045  # 4.5% 10-year Treasury


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: get_retirement_goal
# ══════════════════════════════════════════════════════════════════════════════


async def _get_retirement_goal(user_id: str = "anonymous") -> dict[str, Any]:
    """Core logic for get_retirement_goal — separated for unit-testability."""
    if not settings.database_url:
        return {
            "status": "error",
            "error": "Database not configured. Set DATABASE_URL in .env",
        }
    try:
        goal = await get_goal(settings.database_url, user_id)
        if goal is None:
            return {
                "status": "not_found",
                "user_id": user_id,
                "message": (
                    "No retirement goal found for this user. "
                    "Use set_retirement_goal to create one. "
                    "You will need: current_age, target_retirement_age, "
                    "and target_annual_spending (how much you want to spend per year in retirement)."
                ),
            }
        return {
            "status": "ok",
            "goal": goal,
            "data_timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        log.error("get_retirement_goal_error", user_id=user_id, error=str(e))
        return {"status": "error", "error": str(e)}


@tool
async def get_retirement_goal(user_id: str = "anonymous") -> dict[str, Any]:
    """
    Retrieve the user's stored FIRE (Financial Independence, Retire Early) goal.

    Use this tool when the user asks:
    - 'What is my retirement goal?'
    - 'Show me my FIRE settings'
    - 'What is my target retirement age?'
    - 'What is my FIRE number?'
    - 'How much do I need to retire?'

    Args:
        user_id: The user identifier (defaults to 'anonymous'). Use the user_id
                 from the conversation context if available.

    Returns:
        Dictionary with the stored goal including: current_age, target_retirement_age,
        target_annual_spending, safe_withdrawal_rate, monthly_contribution,
        expected_annual_return, social_security_estimate, fire_number (computed),
        years_to_target, created_at, updated_at.
        If no goal exists: status='not_found' with instructions to create one.
    """
    return await _get_retirement_goal(user_id)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: set_retirement_goal
# ══════════════════════════════════════════════════════════════════════════════


async def _set_retirement_goal(
    current_age: int,
    target_retirement_age: int,
    target_annual_spending: float,
    safe_withdrawal_rate: float = 0.04,
    monthly_contribution: float = 0.0,
    expected_annual_return: float = 0.07,
    social_security_estimate: float = 0.0,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Core logic for set_retirement_goal — separated for unit-testability."""
    if not settings.database_url:
        return {
            "status": "error",
            "error": "Database not configured. Set DATABASE_URL in .env",
        }

    # Validate inputs before touching the DB
    errors = validate_goal_fields(
        current_age, target_retirement_age, target_annual_spending, safe_withdrawal_rate
    )
    if errors:
        return {
            "status": "validation_error",
            "errors": errors,
            "message": "Please correct the following issues: " + "; ".join(errors),
        }

    try:
        saved_goal = await upsert_goal(
            database_url=settings.database_url,
            user_id=user_id,
            current_age=current_age,
            target_retirement_age=target_retirement_age,
            target_annual_spending=target_annual_spending,
            safe_withdrawal_rate=safe_withdrawal_rate,
            monthly_contribution=monthly_contribution,
            expected_annual_return=expected_annual_return,
            social_security_estimate=social_security_estimate,
        )
        log.info(
            "retirement_goal_saved", user_id=user_id, fire_number=saved_goal.get("fire_number")
        )
        return {
            "status": "saved",
            "goal": saved_goal,
            "summary": (
                f"✅ Goal saved! Your FIRE number is ${saved_goal['fire_number']:,.0f} "
                f"(${target_annual_spending:,.0f}/yr ÷ {safe_withdrawal_rate:.0%} SWR). "
                f"You have {saved_goal['years_to_target']} years until your target age of {target_retirement_age}."
            ),
            "data_timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        log.error("set_retirement_goal_error", user_id=user_id, error=str(e))
        return {"status": "error", "error": str(e)}


@tool
async def set_retirement_goal(
    current_age: int,
    target_retirement_age: int,
    target_annual_spending: float,
    safe_withdrawal_rate: float = 0.04,
    monthly_contribution: float = 0.0,
    expected_annual_return: float = 0.07,
    social_security_estimate: float = 0.0,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """
    Create or update the user's FIRE retirement goal in the database.
    This is an upsert — calling it again updates the existing goal.

    Use this tool when the user says:
    - 'I want to retire at 50'
    - 'Set my retirement goal'
    - 'I plan to spend $80,000 a year in retirement'
    - 'Update my monthly savings to $2,000'
    - 'My FIRE target is...'
    - 'I want to be financially independent by age 45'

    Args:
        current_age: User's current age in years (18–100)
        target_retirement_age: Age the user wants to retire (must be > current_age, ≤ 100)
        target_annual_spending: How much money the user wants to spend per year in retirement (USD)
        safe_withdrawal_rate: Annual withdrawal rate as a decimal (default 0.04 = 4%).
                              The 4% rule means a portfolio lasting 30+ years.
                              Use 0.035 (3.5%) for longer retirements (age < 45).
        monthly_contribution: How much the user adds to their portfolio each month (default 0)
        expected_annual_return: Expected annual portfolio return as a decimal (default 0.07 = 7%)
        social_security_estimate: Expected annual Social Security income in retirement (default 0)
        user_id: User identifier (defaults to 'anonymous')

    Returns:
        Saved goal with fire_number (total portfolio needed), years_to_target, and a summary message.
    """
    return await _set_retirement_goal(
        current_age=current_age,
        target_retirement_age=target_retirement_age,
        target_annual_spending=target_annual_spending,
        safe_withdrawal_rate=safe_withdrawal_rate,
        monthly_contribution=monthly_contribution,
        expected_annual_return=expected_annual_return,
        social_security_estimate=social_security_estimate,
        user_id=user_id,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: get_fire_progress
# ══════════════════════════════════════════════════════════════════════════════


async def _get_fire_progress(user_id: str = "anonymous") -> dict[str, Any]:
    """Core logic for get_fire_progress — separated for unit-testability."""
    if not settings.database_url:
        return {"status": "error", "error": "Database not configured. Set DATABASE_URL in .env"}

    # Step 1: Load retirement goal from DB
    try:
        goal = await get_goal(settings.database_url, user_id)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load retirement goal: {e}"}

    if goal is None:
        return {
            "status": "goal_required",
            "message": (
                "No retirement goal found. Please set one first using set_retirement_goal. "
                "You'll need: current_age, target_retirement_age, and target_annual_spending."
            ),
        }

    fire_number = goal["fire_number"]

    # Step 2: Fetch live portfolio value from Ghostfolio
    portfolio_value = 0.0
    portfolio_status = "ok"
    try:
        client = get_shared_client()
        holdings_data = await client.get_portfolio_holdings()
        raw = holdings_data.get("holdings", [])
        holdings_list = list(raw.values()) if isinstance(raw, dict) else raw
        portfolio_value = sum(
            h.get("valueInBaseCurrency", h.get("value", 0)) or 0 for h in holdings_list
        )
    except GhostfolioError as e:
        portfolio_status = "error"
        log.warning("fire_progress_portfolio_error", error=e.message)
    except Exception as e:
        portfolio_status = "error"
        log.warning("fire_progress_portfolio_error", error=str(e))

    # Step 3: Compute progress metrics
    progress_pct = (portfolio_value / fire_number * 100) if fire_number > 0 else 0.0
    shortfall = max(0.0, fire_number - portfolio_value)
    surplus = max(0.0, portfolio_value - fire_number)

    status_label = (
        "achieved"
        if portfolio_value >= fire_number
        else "on_track"
        if progress_pct >= 50
        else "early_stage"
    )

    return {
        "status": "ok",
        "user_id": user_id,
        "progress_status": status_label,
        "fire_number": round(fire_number, 2),
        "current_portfolio_value": round(portfolio_value, 2),
        "progress_pct": round(progress_pct, 1),
        "shortfall": round(shortfall, 2),
        "surplus": round(surplus, 2),
        "goal_summary": {
            "current_age": goal["current_age"],
            "target_retirement_age": goal["target_retirement_age"],
            "years_to_target": goal["years_to_target"],
            "target_annual_spending": goal["target_annual_spending"],
            "safe_withdrawal_rate": goal["safe_withdrawal_rate"],
        },
        "portfolio_data_status": portfolio_status,
        "data_timestamp": datetime.now(UTC).isoformat(),
        "source": "Ghostfolio + Fortio FIRE DB",
    }


@tool
async def get_fire_progress(user_id: str = "anonymous") -> dict[str, Any]:
    """
    Show the user's current progress toward their FIRE (Financial Independence) goal.
    Combines the stored retirement goal with the live portfolio value from Ghostfolio.

    Use this tool when the user asks:
    - 'Am I on track to retire?'
    - 'How close am I to FIRE?'
    - 'What percentage of my retirement goal have I reached?'
    - 'How much more do I need to save to retire?'
    - 'What is my shortfall to retirement?'
    - 'Am I financially independent yet?'

    Args:
        user_id: User identifier (defaults to 'anonymous')

    Returns:
        progress_pct (% toward FIRE number), fire_number (total needed), current_portfolio_value,
        shortfall (remaining amount needed), surplus (if already reached), progress_status
        ('achieved', 'on_track', or 'early_stage'), and goal_summary.
        If no goal is set: status='goal_required' with instructions.
    """
    return await _get_fire_progress(user_id)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4: calculate_retirement_projection
# ══════════════════════════════════════════════════════════════════════════════


def _project_years_to_fire(
    current_value: float,
    fire_number: float,
    monthly_return: float,
    monthly_contribution: float,
) -> float:
    """
    Solve for n (months) in the future value formula:
        FV = PV×(1+r)^n + PMT×[(1+r)^n - 1]/r

    Rearranges to:
        n = log((FV×r + PMT) / (PV×r + PMT)) / log(1+r)

    Returns number of YEARS (float).
    Returns float('inf') if the goal cannot be reached with current contributions.
    """
    if current_value >= fire_number:
        return 0.0

    r = monthly_return

    if r <= 0:
        # No growth → only contributions
        if monthly_contribution <= 0:
            return float("inf")
        months = (fire_number - current_value) / monthly_contribution
        return months / 12

    # Numerator: (FV×r + PMT), denominator: (PV×r + PMT)
    numerator = fire_number * r + monthly_contribution
    denominator = current_value * r + monthly_contribution

    if denominator <= 0 or numerator <= denominator:
        # Cannot converge — no solution
        return float("inf")

    months = math.log(numerator / denominator) / math.log(1 + r)
    return months / 12


async def _calculate_retirement_projection(
    user_id: str = "anonymous",
    monthly_contribution_override: float = -1.0,
) -> dict[str, Any]:
    """Core logic for calculate_retirement_projection — separated for unit-testability."""
    if not settings.database_url:
        return {"status": "error", "error": "Database not configured. Set DATABASE_URL in .env"}

    # Step 1: Load retirement goal from DB
    try:
        goal = await get_goal(settings.database_url, user_id)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load retirement goal: {e}"}

    if goal is None:
        return {
            "status": "goal_required",
            "message": (
                "No retirement goal found. Please use set_retirement_goal first. "
                "You will need: current_age, target_retirement_age, target_annual_spending."
            ),
        }

    # Step 2: Fetch live portfolio value from Ghostfolio
    portfolio_value = 0.0
    try:
        client = get_shared_client()
        holdings_data = await client.get_portfolio_holdings()
        raw = holdings_data.get("holdings", [])
        holdings_list = list(raw.values()) if isinstance(raw, dict) else raw
        portfolio_value = sum(
            h.get("valueInBaseCurrency", h.get("value", 0)) or 0 for h in holdings_list
        )
    except Exception as e:
        log.warning("projection_portfolio_error", error=str(e))

    # Step 3: Fetch macro data from FRED (inflation + risk-free rate)
    inflation_rate = _DEFAULT_INFLATION
    risk_free_rate = _DEFAULT_RISK_FREE
    macro_status = "default"
    macro_note = ""
    try:
        fred = get_shared_fred_client()
        inf_result = await fred.get_current_inflation_rate()
        if inf_result["status"] == "ok":
            inflation_rate = inf_result["inflation_rate_yoy"]
            macro_status = "live"
        rfr_result = await fred.get_risk_free_rate()
        if rfr_result["status"] == "ok":
            risk_free_rate = rfr_result["risk_free_rate"]
    except Exception as e:
        macro_note = f"FRED unavailable — using defaults (3% inflation, 4.5% risk-free). Error: {e}"
        log.warning("projection_fred_error", error=str(e))

    # Step 4: Compute projection
    monthly_contribution = (
        goal["monthly_contribution"]
        if monthly_contribution_override < 0
        else monthly_contribution_override
    )

    expected_return = goal["expected_annual_return"]
    # Real return (inflation-adjusted): (1+nominal)/(1+inflation) - 1
    real_annual_return = (1 + expected_return) / (1 + inflation_rate) - 1
    monthly_real_return = (1 + real_annual_return) ** (1 / 12) - 1

    # FIRE number adjusted for inflation (grows over time)
    years_to_target = goal["years_to_target"]
    fire_number_nominal = goal["fire_number"]
    fire_number_real = fire_number_nominal * ((1 + inflation_rate) ** years_to_target)

    # Years to reach FIRE_number_nominal (current dollars) with real returns
    years_to_fire = _project_years_to_fire(
        current_value=portfolio_value,
        fire_number=fire_number_nominal,
        monthly_return=monthly_real_return,
        monthly_contribution=monthly_contribution,
    )

    projected_retirement_age = (
        goal["current_age"] + years_to_fire if years_to_fire != float("inf") else None
    )

    # How much do they need to save monthly to hit target exactly on time?
    # Solve PMT from FV formula: PMT = (FV - PV×(1+r)^n) × r / [(1+r)^n - 1]
    required_monthly = None
    if years_to_target > 0 and monthly_real_return > 0:
        n_months = years_to_target * 12
        factor = (1 + monthly_real_return) ** n_months
        pv_grown = portfolio_value * factor
        if fire_number_nominal > pv_grown:
            required_monthly = (fire_number_nominal - pv_grown) * monthly_real_return / (factor - 1)
        else:
            required_monthly = 0.0  # Already have enough even without contributions

    # Scenario label
    if years_to_fire == float("inf"):
        scenario = "impossible_without_contributions"
    elif (
        projected_retirement_age is not None
        and projected_retirement_age <= goal["target_retirement_age"]
    ):
        scenario = "ahead_of_schedule"
    elif (
        projected_retirement_age is not None
        and projected_retirement_age <= goal["target_retirement_age"] + 5
    ):
        scenario = "slightly_behind"
    else:
        scenario = "significantly_behind"

    result: dict[str, Any] = {
        "status": "ok",
        "user_id": user_id,
        "scenario": scenario,
        "current_portfolio_value": round(portfolio_value, 2),
        "fire_number_nominal": round(fire_number_nominal, 2),
        "fire_number_inflation_adjusted": round(fire_number_real, 2),
        "years_to_fire": round(years_to_fire, 1) if years_to_fire != float("inf") else None,
        "projected_retirement_age": (
            round(projected_retirement_age, 1) if projected_retirement_age is not None else None
        ),
        "target_retirement_age": goal["target_retirement_age"],
        "current_age": goal["current_age"],
        "monthly_contribution_used": round(monthly_contribution, 2),
        "monthly_contribution_override_applied": monthly_contribution_override >= 0,
        "expected_annual_return_nominal": round(expected_return * 100, 2),
        "inflation_rate_used_pct": round(inflation_rate * 100, 2),
        "real_annual_return_pct": round(real_annual_return * 100, 2),
        "risk_free_rate_pct": round(risk_free_rate * 100, 2),
        "required_monthly_to_hit_target": (
            round(required_monthly, 2) if required_monthly is not None else None
        ),
        "macro_data_status": macro_status,
        "data_timestamp": datetime.now(UTC).isoformat(),
        "source": "Ghostfolio + FRED + Fortio FIRE DB",
        "disclaimer": (
            "⚠️ This is a projection based on assumptions, not a guarantee. "
            "Actual returns vary. Consult a financial advisor for personalised advice."
        ),
    }
    if macro_note:
        result["macro_note"] = macro_note

    return result


@tool
async def calculate_retirement_projection(
    user_id: str = "anonymous",
    monthly_contribution_override: float = -1.0,
) -> dict[str, Any]:
    """
    Project the user's retirement date based on their current portfolio,
    stored savings goal, and live macroeconomic data (inflation from FRED).

    Use this tool when the user asks:
    - 'When can I retire?'
    - 'What is my projected retirement date?'
    - 'How many years until I reach FIRE?'
    - 'What if I save $3,000 a month? When can I retire then?'
    - 'How does inflation affect my retirement timeline?'
    - 'What return do I need to retire by 50?'
    - 'How much do I need to save per month to retire on time?'

    Args:
        user_id: User identifier (defaults to 'anonymous')
        monthly_contribution_override: Override the stored monthly contribution for
            what-if scenarios. Pass -1.0 (default) to use the stored value.
            Example: pass 3000.0 to simulate saving $3,000/month.

    Returns:
        projected_retirement_age, years_to_fire, scenario ('ahead_of_schedule',
        'slightly_behind', 'significantly_behind', or 'impossible_without_contributions'),
        required_monthly_to_hit_target, inflation_rate_used_pct, real_annual_return_pct,
        fire_number_nominal, fire_number_inflation_adjusted.
        Includes a mandatory disclaimer (⚠️ not financial advice).
    """
    return await _calculate_retirement_projection(user_id, monthly_contribution_override)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5: get_macro_data
# ══════════════════════════════════════════════════════════════════════════════


async def _get_macro_data() -> dict[str, Any]:
    """Core logic for get_macro_data — separated for unit-testability."""
    fred = get_shared_fred_client()

    # Two FRED calls in parallel
    import asyncio as _asyncio  # noqa: PLC0415 (local import OK — avoids circular at module top)

    inf_result, rfr_result = await _asyncio.gather(
        fred.get_current_inflation_rate(),
        fred.get_risk_free_rate(),
    )

    inflation_ok = inf_result.get("status") == "ok"
    rfr_ok = rfr_result.get("status") == "ok"

    inflation_rate = (
        inf_result.get("inflation_rate_yoy", _DEFAULT_INFLATION)
        if inflation_ok
        else _DEFAULT_INFLATION
    )
    risk_free_rate = (
        rfr_result.get("risk_free_rate", _DEFAULT_RISK_FREE) if rfr_ok else _DEFAULT_RISK_FREE
    )

    # Real return at 7% nominal: (1.07/1+inflation) - 1
    nominal_return = 0.07
    real_return = (1 + nominal_return) / (1 + inflation_rate) - 1

    result: dict[str, Any] = {
        "status": "ok",
        "inflation": {
            "rate_yoy": round(inflation_rate, 4),
            "rate_pct": round(inflation_rate * 100, 2),
            "data_available": inflation_ok,
            "latest_cpi_date": inf_result.get("latest_cpi_date") if inflation_ok else None,
        },
        "treasury_10y": {
            "yield_decimal": round(risk_free_rate, 4),
            "yield_pct": round(risk_free_rate * 100, 2),
            "data_available": rfr_ok,
            "rate_date": rfr_result.get("rate_date") if rfr_ok else None,
        },
        "derived": {
            "real_return_at_7pct_nominal_pct": round(real_return * 100, 2),
            "swr_vs_risk_free": (
                "risk-free rate EXCEEDS 4% SWR — bonds may be competitive with equities"
                if risk_free_rate > 0.04
                else "risk-free rate below 4% SWR — equities still favoured for long-term growth"
            ),
        },
        "data_timestamp": datetime.now(UTC).isoformat(),
        "source": "FRED / Federal Reserve Bank of St. Louis",
        "note": "CPI = CPIAUCSL (seasonally adjusted). 10Y rate = DGS10.",
    }

    # Add error details if either call failed
    if not inflation_ok:
        result["inflation_error"] = inf_result.get("error", "Unknown error")
        result["inflation"]["rate_pct"] = round(_DEFAULT_INFLATION * 100, 2)
        result["inflation"]["note"] = (
            f"Using default ({_DEFAULT_INFLATION * 100:.1f}%) — FRED unavailable"
        )

    if not rfr_ok:
        result["treasury_error"] = rfr_result.get("error", "Unknown error")
        result["treasury_10y"]["yield_pct"] = round(_DEFAULT_RISK_FREE * 100, 2)
        result["treasury_10y"]["note"] = (
            f"Using default ({_DEFAULT_RISK_FREE * 100:.1f}%) — FRED unavailable"
        )

    return result


@tool
async def get_macro_data() -> dict[str, Any]:
    """
    Fetch current macroeconomic data from the FRED API (Federal Reserve Bank of St. Louis).
    Returns year-over-year CPI inflation rate and the 10-Year US Treasury yield.

    Use this tool when the user asks:
    - 'What is inflation right now?'
    - 'What is the current interest rate?'
    - 'What is the risk-free rate?'
    - 'How does today's inflation affect my retirement plan?'
    - 'Is the 4% rule still valid given current rates?'
    - 'What is the 10-year Treasury yield?'

    Returns:
        inflation (rate_yoy, rate_pct, latest_cpi_date),
        treasury_10y (yield_decimal, yield_pct, rate_date),
        derived (real_return_at_7pct_nominal_pct, swr_vs_risk_free context),
        data_timestamp, source ("FRED / Federal Reserve Bank of St. Louis").
        If FRED is unavailable, returns best-available defaults with a note.
    """
    return await _get_macro_data()


# ══════════════════════════════════════════════════════════════════════════════
# Helper: delete_retirement_goal (used by API route, not exposed as agent tool)
# ══════════════════════════════════════════════════════════════════════════════


async def delete_retirement_goal_for_user(user_id: str) -> dict[str, Any]:
    """
    Delete the retirement goal for a given user. Used by the DELETE API route.
    Not exposed as an agent tool (the agent should not delete user data without clear intent).

    Returns: {"status": "deleted"} or {"status": "not_found"} or {"status": "error"}
    """
    if not settings.database_url:
        return {"status": "error", "error": "Database not configured"}
    try:
        was_deleted = await delete_goal(settings.database_url, user_id)
        if was_deleted:
            log.info("retirement_goal_deleted", user_id=user_id)
            return {"status": "deleted", "user_id": user_id}
        return {"status": "not_found", "user_id": user_id}
    except Exception as e:
        log.error("delete_retirement_goal_error", user_id=user_id, error=str(e))
        return {"status": "error", "error": str(e)}
