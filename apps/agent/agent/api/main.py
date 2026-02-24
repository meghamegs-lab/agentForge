"""
FastAPI REST endpoint for the Fortio AI agent.
Called by Ghostfolio's Angular frontend chat component.

Run with:
    uvicorn agent.api.main:app --host 0.0.0.0 --port 8001 --reload
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.config import settings
from agent.graph.graph import agent_graph
from agent.graph.state import AgentState

# ── Push settings into os.environ so LangChain tracing picks them up ──────────
# pydantic-settings reads .env into the Settings object but does NOT set
# os.environ — LangChain reads os.environ directly for LANGCHAIN_* vars.
if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
if settings.langchain_project:
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

log = structlog.get_logger()

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fortio Agent API",
    description="REST API for the Fortio Ghostfolio Finance AI Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allows the Angular frontend (port 4200) and Ghostfolio UI (port 3333)
# to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",   # Angular dev server
        "http://localhost:3333",   # Ghostfolio UI
        "http://localhost:8000",   # Chainlit (for cross-testing)
        "*",                       # Allow all in dev — restrict in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from the Angular frontend."""
    message: str
    conversation_id: str = ""    # Empty = new conversation; reuse to continue one
    user_id: str = "anonymous"


class VerificationFlag(BaseModel):
    type: str       # e.g. DISCLAIMER_ADDED, POTENTIAL_HALLUCINATION
    severity: str   # HIGH, MEDIUM, LOW, INFO
    message: str


class ChatResponse(BaseModel):
    """Response sent back to the Angular frontend."""
    answer: str
    confidence: str                      # HIGH, MEDIUM, or LOW
    flags: list[VerificationFlag]        # Verification warnings (empty = clean)
    conversation_id: str                 # Echo back so Angular can continue the thread


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check — used by Railway / Docker healthchecks."""
    return {"status": "ok", "service": "fortio-agent"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Angular sends:  { message, conversation_id?, user_id? }
    We return:      { answer, confidence, flags, conversation_id }

    The agent graph handles:
      reasoning → tools → collect_results → verify → response
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())

    log.info(
        "chat_request",
        conversation_id=conversation_id,
        user_id=request.user_id,
        message_preview=request.message[:80],
    )

    # Build the initial LangGraph state
    state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "tool_results": [],
        "verification_flags": [],
        "confidence": "HIGH",
        "reasoning_steps": 0,
        "conversation_id": conversation_id,
        "user_id": request.user_id,
        "final_response": "",
        "should_escalate": False,
    }

    try:
        # Run the LangGraph agent — this is where the magic happens:
        # reasoning → tools → collect_results → reasoning → verify → END
        final_state = await agent_graph.ainvoke(state)

        # Get the verified final answer
        answer = final_state.get("final_response", "")
        if not answer:
            # Fallback: read directly from last message
            last_msg = final_state["messages"][-1]
            content = getattr(last_msg, "content", "")
            answer = content if isinstance(content, str) else str(content)

        flags = [
            VerificationFlag(
                type=f.get("type", "UNKNOWN"),
                severity=f.get("severity", "INFO"),
                message=f.get("message", ""),
            )
            for f in final_state.get("verification_flags", [])
        ]

        log.info(
            "chat_response",
            conversation_id=conversation_id,
            confidence=final_state.get("confidence", "MEDIUM"),
            flag_count=len(flags),
        )

        return ChatResponse(
            answer=answer,
            confidence=final_state.get("confidence", "MEDIUM"),
            flags=flags,
            conversation_id=conversation_id,
        )

    except Exception as e:
        log.error("chat_error", error=str(e), conversation_id=conversation_id)
        return ChatResponse(
            answer=(
                "I encountered an error processing your request. "
                "Please try again or rephrase your question."
            ),
            confidence="LOW",
            flags=[
                VerificationFlag(
                    type="AGENT_ERROR",
                    severity="HIGH",
                    message=str(e),
                )
            ],
            conversation_id=conversation_id,
        )
