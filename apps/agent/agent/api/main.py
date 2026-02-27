# FastAPI application that serves the Fortio agent over HTTP to the Angular frontend and external clients.
"""
FastAPI REST endpoint for the Fortio AI agent.
Called by Ghostfolio's Angular frontend chat component.

Checkpointing:
  Uses AsyncPostgresSaver (via DATABASE_URL) to persist full conversation
  history in Postgres. Each conversation is identified by `conversation_id`,
  which becomes the LangGraph `thread_id`. On every turn:
    1. Angular sends { message, conversation_id }
    2. LangGraph loads the previous state for that thread_id from Postgres
    3. The new message is appended via the add_messages reducer
    4. The graph runs and the updated state is saved back to Postgres
  Result: the LLM sees the full conversation history on every turn.

Run with:
    uvicorn agent.api.main:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ToolCallInfo,
    VerificationFlag,
)
from agent.config import settings
from agent.graph.graph import build_graph
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


# ── Lifespan: set up the checkpointer once at startup ─────────────────────────


# Initialises the Postgres checkpointer at startup and tears it down cleanly on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    - Tries to open an AsyncPostgresSaver connection pool at startup.
    - If Postgres is unavailable (wrong creds, network issue), falls back to
      MemorySaver so the app still starts and /health responds successfully.
    - Runs `saver.setup()` to create the checkpoints table if it doesn't exist.
    - Builds the agent graph with the checkpointer wired in.
    - Stores the graph on app.state so route handlers can access it.
    - Cleans up the connection pool on shutdown.
    """
    saver_ctx = None

    if settings.checkpoint_backend == "postgres" and settings.database_url:
        log.info("checkpointer_startup", backend="postgres")
        try:
            # Manually manage the async context manager so we can fall back
            # to MemorySaver if the Postgres connection fails — instead of
            # crashing the entire app and blocking the health check endpoint.
            saver_ctx = AsyncPostgresSaver.from_conn_string(settings.database_url)
            saver = await saver_ctx.__aenter__()
            # Creates the langgraph_checkpoints table if it doesn't exist yet.
            # Safe to call on every startup — it's idempotent.
            await saver.setup()
            app.state.agent_graph = build_graph(checkpointer=saver)
            log.info("checkpointer_ready", backend="postgres")
        except Exception as exc:
            # Postgres is unreachable or credentials are wrong.
            # Fall back to in-memory checkpointing so the service stays healthy.
            log.error(
                "checkpointer_postgres_failed",
                error=str(exc),
                fallback="memory",
            )
            saver_ctx = None
            app.state.agent_graph = build_graph(checkpointer=MemorySaver())
    else:
        # Fallback: in-memory checkpointing (history lost on restart)
        log.warning(
            "checkpointer_fallback",
            reason="DATABASE_URL not set or checkpoint_backend != postgres",
            backend="memory",
        )
        app.state.agent_graph = build_graph(checkpointer=MemorySaver())

    try:
        yield  # App is running — /health and /api/chat are now served
    finally:
        # Cleanly close the Postgres connection pool on shutdown
        if saver_ctx is not None:
            try:
                await saver_ctx.__aexit__(None, None, None)
            except Exception as exc:
                log.warning("checkpointer_cleanup_error", error=str(exc))


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fortio Agent API",
    description="REST API for the Fortio Ghostfolio Finance AI Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allows the Angular frontend and Ghostfolio UI to call this API.
# NOTE: "allow_credentials=True" + "*" wildcard is an invalid CORS combination
# that browsers reject silently. Always list origins explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # configured via CORS_ORIGINS in .env
    allow_credentials=False,  # Angular HttpClient does not send cookies — no credentials needed
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────


# Returns {"status": "ok"} — used by Railway / Docker health checks to confirm the service is up.
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — used by Railway / Docker healthchecks."""
    return HealthResponse(status="ok", service="fortio-agent")


# Receives a user message, runs the LangGraph agent, and returns the verified answer with metadata.
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    Main chat endpoint — supports multi-turn conversation history.

    Angular sends:  { message, conversation_id?, user_id? }
    We return:      { answer, confidence, flags, conversation_id }

    How history works:
      - `conversation_id` maps to LangGraph's `thread_id`.
      - On every turn, LangGraph loads the full AgentState for that thread
        from Postgres, appends the new HumanMessage (via add_messages),
        runs the graph, and saves the updated state back to Postgres.
      - The LLM therefore sees the complete conversation history every turn.
      - Angular only needs to send the NEW message + the same conversation_id.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())

    log.info(
        "chat_request",
        conversation_id=conversation_id,
        user_id=request.user_id,
        message_preview=request.message[:80],
    )

    # LangGraph config — thread_id is the key that identifies this conversation
    # in the Postgres checkpoints table.
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": request.user_id,
        }
    }

    # Initial state — only the NEW message is needed here.
    # LangGraph merges it with the checkpointed history via add_messages.
    # The other fields (tool_results, verification_flags) reset each turn
    # because they have no reducer — they're overwritten by these defaults.
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
        # Multi-turn context — reasoning_node increments turn_number and
        # populates context_entities; the checkpointer persists both so
        # values from prior turns are available in the next turn.
        "turn_number": 0,
        "context_entities": {},
    }

    # Get the graph with checkpointer from app.state (set during lifespan)
    graph = http_request.app.state.agent_graph

    try:
        # Run the LangGraph agent — history is loaded from Postgres automatically
        final_state = await graph.ainvoke(state, config=config)

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

        # Build ToolCallInfo from ToolMessage objects in the message history.
        # Each ToolMessage records the name of the tool and its JSON result,
        # which always contains a "status" key ("success" / "error" / "empty").
        tool_calls: list[ToolCallInfo] = []
        for msg in final_state.get("messages", []):
            if isinstance(msg, ToolMessage):
                try:
                    result: dict = json.loads(msg.content) if isinstance(msg.content, str) else {}
                except (json.JSONDecodeError, TypeError):
                    result = {}
                tool_calls.append(
                    ToolCallInfo(
                        tool_name=msg.name or "unknown",
                        status=result.get("status", "success"),
                        error=result.get("error"),
                    )
                )

        log.info(
            "chat_response",
            conversation_id=conversation_id,
            confidence=final_state.get("confidence", "MEDIUM"),
            flag_count=len(flags),
            tool_call_count=len(tool_calls),
        )

        return ChatResponse(
            answer=answer,
            confidence=final_state.get("confidence", "MEDIUM"),
            flags=flags,
            tool_calls=tool_calls,
            conversation_id=conversation_id,
            turn_number=final_state.get("turn_number", 1),
            context_entities=final_state.get("context_entities", {}),
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
