"""
Integration tests for agent/api/main.py FastAPI endpoints.
===========================================================
Tests the HTTP layer of the Fortio agent API:
  - GET  /health  → HealthResponse
  - POST /api/chat → ChatResponse (happy path, error path, conversation id)

The LangGraph agent graph is mocked at the app.state level so no real
LLM or Ghostfolio network calls are made. The FastAPI ASGI app is driven
through httpx.AsyncClient with ASGITransport, exercising the full request/
response cycle including middleware and serialisation.

Tests covered:
  1.  GET /health returns 200 with correct JSON body
  2.  GET /health content-type is application/json
  3.  POST /api/chat returns 200 with ChatResponse schema
  4.  POST /api/chat populates answer from graph final_response
  5.  POST /api/chat echoes conversation_id back to caller
  6.  POST /api/chat generates a UUID when conversation_id is empty
  7.  POST /api/chat returns LOW confidence + AGENT_ERROR when graph raises
  8.  POST /api/chat includes verification flags in response
  9.  POST /api/chat tool_calls populated from ToolMessage history
  10. POST /api/chat CORS header present for allowed origin
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.api.main import app


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_graph_state(
    answer: str = "Your portfolio summary.",
    confidence: str = "HIGH",
    flags: list | None = None,
    tool_results: list | None = None,
    messages_extra: list | None = None,
    turn_number: int = 1,
) -> dict:
    """Build the dict that the mocked graph.ainvoke returns."""
    messages: list = [AIMessage(content=answer)]
    if messages_extra:
        messages.extend(messages_extra)
    return {
        "messages": messages,
        "final_response": answer,
        "verification_flags": flags or [],
        "confidence": confidence,
        "tool_results": tool_results or [],
        "should_escalate": False,
        "reasoning_steps": 1,
        "turn_number": turn_number,
        "context_entities": {},
        "conversation_id": "test-conv",
        "user_id": "anonymous",
    }


@pytest.fixture
async def api_client():
    """
    FastAPI test client with the agent graph mocked.

    httpx.ASGITransport does NOT trigger the ASGI lifespan, so we inject
    the mock graph directly onto app.state before issuing requests.
    """
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = _make_graph_state()

    # Inject the mock directly — no lifespan required
    app.state.agent_graph = mock_graph

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client, mock_graph

    # Clean up so other tests get a fresh state
    try:
        del app.state.agent_graph
    except AttributeError:
        pass


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    async def test_returns_200(self, api_client):
        client, _ = api_client
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_body_has_status_ok(self, api_client):
        client, _ = api_client
        data = (await client.get("/health")).json()
        assert data["status"] == "ok"

    async def test_body_has_service_name(self, api_client):
        client, _ = api_client
        data = (await client.get("/health")).json()
        assert data["service"] == "fortio-agent"

    async def test_content_type_is_json(self, api_client):
        client, _ = api_client
        response = await client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ── POST /api/chat — happy path ───────────────────────────────────────────────

class TestChatEndpointHappyPath:
    async def test_returns_200(self, api_client):
        client, _ = api_client
        response = await client.post("/api/chat", json={"message": "Show me my portfolio"})
        assert response.status_code == 200

    async def test_response_contains_answer(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.return_value = _make_graph_state(
            answer="Your portfolio is worth $8,000."
        )
        data = (
            await client.post("/api/chat", json={"message": "portfolio value"})
        ).json()
        assert data["answer"] == "Your portfolio is worth $8,000."

    async def test_response_contains_confidence(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.return_value = _make_graph_state(confidence="MEDIUM")
        data = (
            await client.post("/api/chat", json={"message": "diversification?"})
        ).json()
        assert data["confidence"] == "MEDIUM"

    async def test_conversation_id_echoed_back(self, api_client):
        client, _ = api_client
        conv_id = "my-conv-abc123"
        data = (
            await client.post(
                "/api/chat",
                json={"message": "hello", "conversation_id": conv_id},
            )
        ).json()
        assert data["conversation_id"] == conv_id

    async def test_new_conversation_id_generated_when_empty(self, api_client):
        client, _ = api_client
        data = (
            await client.post("/api/chat", json={"message": "hello", "conversation_id": ""})
        ).json()
        conv_id = data["conversation_id"]
        assert conv_id != ""
        # Should be a valid UUID
        uuid.UUID(conv_id)  # raises ValueError if invalid

    async def test_flags_propagated_in_response(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.return_value = _make_graph_state(
            answer="I recommend rebalancing.",
            flags=[
                {"type": "DISCLAIMER_ADDED", "severity": "INFO", "message": "advice detected"},
            ],
        )
        data = (
            await client.post("/api/chat", json={"message": "should I rebalance?"})
        ).json()
        assert len(data["flags"]) == 1
        assert data["flags"][0]["type"] == "DISCLAIMER_ADDED"

    async def test_empty_flags_when_no_verification_issues(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.return_value = _make_graph_state(flags=[])
        data = (
            await client.post("/api/chat", json={"message": "what time is it?"})
        ).json()
        assert data["flags"] == []

    async def test_tool_calls_populated_from_tool_messages(self, api_client):
        """ToolMessage objects in the message history must appear in tool_calls."""
        client, mock_graph = api_client
        tool_result = json.dumps({"status": "ok", "total_value": 8000.0})
        mock_graph.ainvoke.return_value = _make_graph_state(
            messages_extra=[
                ToolMessage(
                    content=tool_result,
                    name="get_portfolio_summary",
                    tool_call_id="call-1",
                )
            ]
        )
        data = (
            await client.post("/api/chat", json={"message": "portfolio?"})
        ).json()
        tool_calls = data.get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "get_portfolio_summary"
        assert tool_calls[0]["status"] == "ok"

    async def test_turn_number_in_response(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.return_value = _make_graph_state(turn_number=3)
        data = (
            await client.post("/api/chat", json={"message": "test"})
        ).json()
        assert data["turn_number"] == 3

    async def test_graph_invoked_with_correct_thread_id(self, api_client):
        """The graph must be invoked with the conversation_id as thread_id."""
        client, mock_graph = api_client
        await client.post(
            "/api/chat",
            json={"message": "hello", "conversation_id": "thread-xyz"},
        )
        call_kwargs = mock_graph.ainvoke.call_args
        config = call_kwargs[1]["config"] if "config" in call_kwargs[1] else call_kwargs[0][1]
        assert config["configurable"]["thread_id"] == "thread-xyz"


# ── POST /api/chat — error handling ──────────────────────────────────────────

class TestChatEndpointErrorHandling:
    async def test_graph_exception_returns_200_not_500(self, api_client):
        """
        When the graph raises an exception the endpoint must NOT return 500.
        It must catch it and return a graceful ChatResponse with LOW confidence.
        """
        client, mock_graph = api_client
        mock_graph.ainvoke.side_effect = RuntimeError("LLM timeout")
        response = await client.post("/api/chat", json={"message": "portfolio"})
        assert response.status_code == 200

    async def test_graph_exception_returns_low_confidence(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.side_effect = RuntimeError("LLM timeout")
        data = (
            await client.post("/api/chat", json={"message": "portfolio"})
        ).json()
        assert data["confidence"] == "LOW"

    async def test_graph_exception_returns_agent_error_flag(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.side_effect = RuntimeError("connection refused")
        data = (
            await client.post("/api/chat", json={"message": "portfolio"})
        ).json()
        error_flags = [f for f in data.get("flags", []) if f["type"] == "AGENT_ERROR"]
        assert len(error_flags) == 1

    async def test_graph_exception_error_message_included(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.side_effect = ValueError("token expired")
        data = (
            await client.post("/api/chat", json={"message": "portfolio"})
        ).json()
        flags = data.get("flags", [])
        assert any("token expired" in f.get("message", "") for f in flags)

    async def test_graph_exception_conversation_id_still_returned(self, api_client):
        client, mock_graph = api_client
        mock_graph.ainvoke.side_effect = Exception("crash")
        data = (
            await client.post(
                "/api/chat",
                json={"message": "hi", "conversation_id": "conv-err"},
            )
        ).json()
        assert data["conversation_id"] == "conv-err"

    async def test_missing_final_response_falls_back_to_last_message(self, api_client):
        """
        When final_response is empty, the endpoint should read the last message's
        content as the answer.
        """
        client, mock_graph = api_client
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Fallback answer from AIMessage.")],
            "final_response": "",   # empty → triggers fallback
            "verification_flags": [],
            "confidence": "HIGH",
            "tool_results": [],
            "should_escalate": False,
            "reasoning_steps": 1,
            "turn_number": 1,
            "context_entities": {},
        }
        data = (
            await client.post("/api/chat", json={"message": "test"})
        ).json()
        assert "Fallback answer from AIMessage." in data["answer"]


# ── CORS ─────────────────────────────────────────────────────────────────────

class TestCORSMiddleware:
    async def test_allowed_origin_gets_cors_header(self, api_client):
        client, _ = api_client
        response = await client.get(
            "/health",
            headers={"Origin": "http://localhost:4200"},
        )
        assert "access-control-allow-origin" in response.headers

    async def test_options_preflight_returns_200(self, api_client):
        client, _ = api_client
        response = await client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:4200",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code in (200, 204)
