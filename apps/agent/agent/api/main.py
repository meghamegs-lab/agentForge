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
from fastapi.responses import StreamingResponse
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
from agent.cache.query_cache import get_cached_response, set_cached_response
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


# Lists all 11 available agent tools — useful for graders and demo videos.
@app.get("/api/tools")
async def list_tools() -> dict:
    """
    List all available agent tools with name and description.

    Returns the complete catalogue of tools the Fortio agent can call.
    Useful for verifying that all tools are registered and available without
    running a full conversation turn.
    """
    from agent.tools import ALL_TOOLS  # local import to avoid circular at startup

    return {
        "tool_count": len(ALL_TOOLS),
        "tools": [
            {
                "name": t.name,
                "description": (t.description or "").split("\n")[0].strip(),
            }
            for t in ALL_TOOLS
        ],
    }


# Streams the agent response as Server-Sent Events so the UI can display tokens as they arrive.
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """
    Streaming chat endpoint — emits SSE events as the agent thinks and responds.

    Event types emitted (each as `data: <json>\\n\\n`):
      {"type": "tool_start", "name": "<tool_name>"}     — a tool began executing
      {"type": "tool_done",  "name": "<tool_name>"}     — a tool finished
      {"type": "token",      "content": "<text>"}       — one LLM text chunk
      {"type": "done",       "conversation_id": "...",
                             "answer": "...", "confidence": "...",
                             "flags": [...], "tool_calls": [...],
                             "turn_number": N, "from_cache": bool}
      {"type": "error",      "message": "<msg>"}        — unrecoverable error

    Angular should open this as a fetch() + ReadableStream; EventSource is not
    suitable here because it only supports GET requests.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # ── Semantic cache fast-path ───────────────────────────────────────────────
    is_fresh_session = not request.conversation_id
    if is_fresh_session:
        cached = await get_cached_response(request.user_id, request.message)
        if cached:
            log.info("stream_cache_hit", conversation_id=conversation_id)

            async def _cached_stream():
                payload = json.dumps(
                    {
                        "type": "done",
                        "conversation_id": conversation_id,
                        "answer": cached["answer"],
                        "confidence": cached["confidence"],
                        "flags": cached.get("flags", []),
                        "tool_calls": cached.get("tool_calls", []),
                        "turn_number": cached.get("turn_number", 1),
                        "context_entities": cached.get("context_entities", {}),
                        "from_cache": True,
                    }
                )
                yield f"data: {payload}\n\n"

            return StreamingResponse(
                _cached_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    graph = http_request.app.state.agent_graph

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
        "turn_number": 0,
        "context_entities": {},
    }
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": request.user_id,
        }
    }

    async def _event_stream():
        final_state: dict[str, Any] = {}

        try:
            async for event in graph.astream_events(state, config=config, version="v2"):
                kind: str = event["event"]
                name: str = event.get("name", "")

                if kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': name})}\n\n"

                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_done', 'name': name})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content

                    # Extract text — Claude returns a list of typed blocks;
                    # OpenAI returns a plain string.
                    text = ""
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text += block.get("text", "")

                    if text:
                        yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event["data"].get("output", {})

            # ── Build the `done` event from the final state ────────────────────
            answer = final_state.get("final_response", "")
            if not answer:
                msgs = final_state.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    raw = getattr(last, "content", "")
                    answer = raw if isinstance(raw, str) else str(raw)

            flags = [
                {
                    "type": f.get("type", "UNKNOWN"),
                    "severity": f.get("severity", "INFO"),
                    "message": f.get("message", ""),
                }
                for f in final_state.get("verification_flags", [])
            ]

            # Only include tool calls from the current turn
            all_messages = final_state.get("messages", [])
            last_human_idx = 0
            for i, msg in enumerate(all_messages):
                if isinstance(msg, HumanMessage):
                    last_human_idx = i
            current_turn_msgs = all_messages[last_human_idx:]

            tool_calls: list[dict[str, Any]] = []
            for msg in current_turn_msgs:
                if isinstance(msg, ToolMessage):
                    try:
                        result: dict = (
                            json.loads(msg.content) if isinstance(msg.content, str) else {}
                        )
                    except (json.JSONDecodeError, TypeError):
                        result = {}
                    tool_calls.append(
                        {
                            "tool_name": msg.name or "unknown",
                            "status": result.get("status", "ok"),
                            "error": result.get("error"),
                        }
                    )

            confidence = final_state.get("confidence", "MEDIUM")
            done_payload = json.dumps(
                {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "answer": answer,
                    "confidence": confidence,
                    "flags": flags,
                    "tool_calls": tool_calls,
                    "turn_number": final_state.get("turn_number", 1),
                    "context_entities": final_state.get("context_entities", {}),
                    "from_cache": False,
                }
            )
            yield f"data: {done_payload}\n\n"

            log.info(
                "stream_done",
                conversation_id=conversation_id,
                confidence=confidence,
                tool_count=len(tool_calls),
                flag_count=len(flags),
            )

            # ── Cache fresh-session HIGH/MEDIUM answers ────────────────────────
            if is_fresh_session and confidence != "LOW":
                await set_cached_response(
                    request.user_id,
                    request.message,
                    {
                        "answer": answer,
                        "confidence": confidence,
                        "flags": flags,
                        "tool_calls": tool_calls,
                        "turn_number": final_state.get("turn_number", 1),
                        "context_entities": final_state.get("context_entities", {}),
                    },
                )

        except Exception as exc:
            log.error("stream_error", error=str(exc), conversation_id=conversation_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

    # ── Semantic cache check ───────────────────────────────────────────────────
    # Only cache single-turn queries (no conversation_id means a fresh session).
    # Multi-turn queries depend on conversation history and must NOT be cached.
    is_fresh_session = not request.conversation_id
    if is_fresh_session:
        cached = await get_cached_response(request.user_id, request.message)
        if cached:
            log.info("cache_response_served", conversation_id=conversation_id)
            return ChatResponse(
                answer=cached["answer"],
                confidence=cached["confidence"],
                flags=[VerificationFlag(**f) for f in cached.get("flags", [])],
                tool_calls=[ToolCallInfo(**tc) for tc in cached.get("tool_calls", [])],
                conversation_id=conversation_id,
                turn_number=cached.get("turn_number", 1),
                context_entities=cached.get("context_entities", {}),
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

        # Build ToolCallInfo — only from the CURRENT turn's tool calls.
        # final_state["messages"] contains the full conversation history (all turns)
        # because of the add_messages reducer + Postgres checkpointer. Without this
        # slice, tool calls from ALL previous turns bleed into the current response,
        # making the widget show stale tools as if they ran this turn.
        # Fix: find the last HumanMessage (= start of current turn) and only scan
        # ToolMessages that follow it.
        all_messages = final_state.get("messages", [])
        last_human_idx = 0
        for i, msg in enumerate(all_messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = i
        current_turn_messages = all_messages[last_human_idx:]

        tool_calls: list[ToolCallInfo] = []
        for msg in current_turn_messages:
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

        # ── Cache the response for fresh-session queries ───────────────────────
        # Only cache HIGH/MEDIUM confidence answers to avoid caching error states.
        confidence = final_state.get("confidence", "MEDIUM")
        if is_fresh_session and confidence != "LOW":
            await set_cached_response(
                request.user_id,
                request.message,
                {
                    "answer": answer,
                    "confidence": confidence,
                    "flags": [
                        {"type": f.type, "severity": f.severity, "message": f.message}
                        for f in flags
                    ],
                    "tool_calls": [
                        {"tool_name": tc.tool_name, "status": tc.status, "error": tc.error}
                        for tc in tool_calls
                    ],
                    "turn_number": final_state.get("turn_number", 1),
                    "context_entities": final_state.get("context_entities", {}),
                },
            )

        return ChatResponse(
            answer=answer,
            confidence=confidence,
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
