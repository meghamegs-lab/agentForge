# Pydantic request and response models for the /health and /api/chat FastAPI endpoints.
"""
Pydantic schemas for the Fortio Agent API.

All request/response models are defined here and imported by main.py.
This keeps main.py focused on routing logic only.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response shape for the GET /health endpoint."""

    status: str  # always "ok" when the service is up
    service: str  # service identifier, e.g. "fortio-agent"


class ChatRequest(BaseModel):
    """Incoming chat message from the Angular frontend."""

    message: str
    conversation_id: str = ""  # Empty = new conversation; reuse to continue one
    user_id: str = "anonymous"


class ToolCallInfo(BaseModel):
    """
    Summary of a single tool invocation during one agent reasoning turn.
    Populated from LangGraph ToolMessage objects and surfaced in ChatResponse
    so callers can see which tools the agent used and whether they succeeded.
    """

    tool_name: str  # e.g. get_portfolio_summary, get_market_data
    status: str  # "ok", "error", or "empty"
    error: str | None = None  # error message when status == "error"


class VerificationFlag(BaseModel):
    type: str  # e.g. DISCLAIMER_ADDED, POTENTIAL_HALLUCINATION
    severity: str  # HIGH, MEDIUM, LOW, INFO
    message: str


class ChatResponse(BaseModel):
    """Response sent back to the Angular frontend."""

    answer: str
    confidence: str  # HIGH, MEDIUM, or LOW
    flags: list[VerificationFlag]  # Verification warnings (empty = clean)
    tool_calls: list[ToolCallInfo] = []  # Tools invoked during this turn
    conversation_id: str  # Echo back so Angular can continue the thread
    turn_number: int = 1  # Which turn of the conversation this is (1-indexed)
    context_entities: dict[
        str, list[str]
    ] = {}  # Entities tracked across turns {"tickers": [...], "sectors": [...]}


# ── FIRE Goal Tracker — CRUD Schemas ──────────────────────────────────────────
# Used by GET/POST/PUT/DELETE /api/goals/retirement endpoints.


class RetirementGoalRequest(BaseModel):
    """
    Request body for POST /api/goals/retirement and PUT /api/goals/retirement/{user_id}.

    All monetary values are in USD. Rates are decimals (e.g. 0.04 = 4%).
    """

    current_age: int
    """User's current age (18–100)."""

    target_retirement_age: int
    """Age the user wants to retire (must be > current_age)."""

    target_annual_spending: float
    """How much the user wants to spend per year in retirement (USD > 0)."""

    safe_withdrawal_rate: float = 0.04
    """Annual withdrawal rate as a decimal. Default 0.04 = 4% (the 'Four Percent Rule')."""

    monthly_contribution: float = 0.0
    """How much the user adds to their investment portfolio each month (USD ≥ 0)."""

    expected_annual_return: float = 0.07
    """Expected annual portfolio return as a decimal. Default 0.07 = 7% (historical S&P avg)."""

    social_security_estimate: float = 0.0
    """Expected annual Social Security income in retirement (USD ≥ 0)."""


class RetirementGoalResponse(BaseModel):
    """
    Response body for all /api/goals/retirement endpoints.
    The fire_number (= annual_spending / SWR) is computed server-side.
    """

    status: str
    """'ok', 'not_found', 'saved', 'deleted', 'validation_error', or 'error'."""

    user_id: str = ""
    goal: dict = {}
    """The full retirement goal record, including computed fire_number and years_to_target."""

    message: str = ""
    """Human-readable summary or error description."""

    errors: list[str] = []
    """Validation errors (only present when status='validation_error')."""
