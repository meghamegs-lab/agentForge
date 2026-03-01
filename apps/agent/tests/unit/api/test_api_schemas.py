"""
Unit tests for agent/api/schemas.py
====================================
Validates request/response Pydantic model defaults, optional fields,
and serialisation shapes — no network calls, no LLM, no graph.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ToolCallInfo,
    VerificationFlag,
)

# ── HealthResponse ──────────────────────────────────────────────────────────


class TestHealthResponse:
    def test_valid_response(self):
        r = HealthResponse(status="ok", service="fortio-agent")
        assert r.status == "ok"
        assert r.service == "fortio-agent"

    def test_fields_required(self):
        with pytest.raises(ValidationError):
            HealthResponse()  # type: ignore[call-arg]


# ── ChatRequest ─────────────────────────────────────────────────────────────


class TestChatRequest:
    def test_message_required(self):
        with pytest.raises(ValidationError):
            ChatRequest()  # type: ignore[call-arg]

    def test_default_conversation_id_is_empty_string(self):
        req = ChatRequest(message="hello")
        assert req.conversation_id == ""

    def test_default_user_id_is_anonymous(self):
        req = ChatRequest(message="hello")
        assert req.user_id == "anonymous"

    def test_explicit_values_accepted(self):
        req = ChatRequest(
            message="What is my portfolio worth?",
            conversation_id="conv-123",
            user_id="user-456",
        )
        assert req.message == "What is my portfolio worth?"
        assert req.conversation_id == "conv-123"
        assert req.user_id == "user-456"

    def test_empty_message_accepted(self):
        # Pydantic does not reject empty strings by default
        req = ChatRequest(message="")
        assert req.message == ""


# ── ToolCallInfo ─────────────────────────────────────────────────────────────


class TestToolCallInfo:
    def test_required_fields(self):
        t = ToolCallInfo(tool_name="get_portfolio_summary", status="ok")
        assert t.tool_name == "get_portfolio_summary"
        assert t.status == "ok"
        assert t.error is None

    def test_error_field_optional(self):
        t = ToolCallInfo(
            tool_name="get_market_data",
            status="error",
            error="symbol_not_found",
        )
        assert t.error == "symbol_not_found"

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            ToolCallInfo()  # type: ignore[call-arg]


# ── VerificationFlag ─────────────────────────────────────────────────────────


class TestVerificationFlag:
    def test_all_fields_required(self):
        flag = VerificationFlag(
            type="DISCLAIMER_ADDED",
            severity="INFO",
            message="Investment language detected",
        )
        assert flag.type == "DISCLAIMER_ADDED"
        assert flag.severity == "INFO"
        assert flag.message == "Investment language detected"

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            VerificationFlag(type="X", severity="HIGH")  # type: ignore[call-arg]


# ── ChatResponse ─────────────────────────────────────────────────────────────


class TestChatResponse:
    def test_minimal_required_fields(self):
        resp = ChatResponse(
            answer="Your portfolio is worth $10,000.",
            confidence="HIGH",
            flags=[],
            conversation_id="conv-abc",
        )
        assert resp.answer == "Your portfolio is worth $10,000."
        assert resp.confidence == "HIGH"
        assert resp.flags == []
        assert resp.conversation_id == "conv-abc"

    def test_tool_calls_default_empty_list(self):
        resp = ChatResponse(
            answer="ok",
            confidence="MEDIUM",
            flags=[],
            conversation_id="c",
        )
        assert resp.tool_calls == []

    def test_context_entities_default_empty_dict(self):
        resp = ChatResponse(
            answer="ok",
            confidence="MEDIUM",
            flags=[],
            conversation_id="c",
        )
        assert resp.context_entities == {}

    def test_turn_number_default_is_1(self):
        resp = ChatResponse(
            answer="ok",
            confidence="HIGH",
            flags=[],
            conversation_id="c",
        )
        assert resp.turn_number == 1

    def test_flags_with_verification_flags(self):
        flags = [
            VerificationFlag(type="DISCLAIMER_ADDED", severity="INFO", message="advice detected"),
            VerificationFlag(type="STALE_DATA", severity="MEDIUM", message="data is old"),
        ]
        resp = ChatResponse(
            answer="Rebalance your portfolio.",
            confidence="LOW",
            flags=flags,
            conversation_id="conv-xyz",
        )
        assert len(resp.flags) == 2
        assert resp.flags[0].type == "DISCLAIMER_ADDED"
        assert resp.flags[1].severity == "MEDIUM"

    def test_tool_calls_populated(self):
        tool_calls = [
            ToolCallInfo(tool_name="get_portfolio_summary", status="ok"),
            ToolCallInfo(tool_name="get_market_data", status="error", error="not found"),
        ]
        resp = ChatResponse(
            answer="ok",
            confidence="HIGH",
            flags=[],
            tool_calls=tool_calls,
            conversation_id="c",
        )
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[1].error == "not found"

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            ChatResponse()  # type: ignore[call-arg]

    def test_json_serialisation_round_trip(self):
        resp = ChatResponse(
            answer="test",
            confidence="HIGH",
            flags=[VerificationFlag(type="X", severity="INFO", message="m")],
            conversation_id="c",
        )
        data = resp.model_dump()
        assert data["answer"] == "test"
        assert data["flags"][0]["type"] == "X"
        restored = ChatResponse(**data)
        assert restored.answer == "test"
