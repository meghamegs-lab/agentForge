"""
Pydantic schemas for the Fortio Agent API.

All request/response models are defined here and imported by main.py.
This keeps main.py focused on routing logic only.
"""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response shape for the GET /health endpoint."""
    status: str     # always "ok" when the service is up
    service: str    # service identifier, e.g. "fortio-agent"


class ChatRequest(BaseModel):
    """Incoming chat message from the Angular frontend."""
    message: str
    conversation_id: str = ""    # Empty = new conversation; reuse to continue one
    user_id: str = "anonymous"


class ToolCallInfo(BaseModel):
    """
    Summary of a single tool invocation during one agent reasoning turn.
    Populated from LangGraph ToolMessage objects and surfaced in ChatResponse
    so callers can see which tools the agent used and whether they succeeded.
    """
    tool_name: str          # e.g. get_portfolio_summary, get_market_data
    status: str             # "ok", "error", or "empty"
    error: str | None = None  # error message when status == "error"


class VerificationFlag(BaseModel):
    type: str       # e.g. DISCLAIMER_ADDED, POTENTIAL_HALLUCINATION
    severity: str   # HIGH, MEDIUM, LOW, INFO
    message: str


class ChatResponse(BaseModel):
    """Response sent back to the Angular frontend."""
    answer: str
    confidence: str                      # HIGH, MEDIUM, or LOW
    flags: list[VerificationFlag]        # Verification warnings (empty = clean)
    tool_calls: list[ToolCallInfo] = []  # Tools invoked during this turn (empty = LLM answered directly)
    conversation_id: str                 # Echo back so Angular can continue the thread
