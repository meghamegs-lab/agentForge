# Defines AgentState — the shared TypedDict that flows through every LangGraph node.
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State that flows through the LangGraph agent nodes."""

    messages: Annotated[list, add_messages]  # Full conversation history
    tool_results: list[dict[str, Any]]  # Accumulated tool call results
    verification_flags: list[dict[str, str]]  # Flags from verification layer
    confidence: str  # HIGH / MEDIUM / LOW
    reasoning_steps: int  # How many tool chains used
    conversation_id: str  # For Postgres checkpointing
    user_id: str  # For multi-user isolation
    final_response: str  # Verified final response text
    should_escalate: bool  # Human-in-the-loop trigger
    # ── Multi-turn context awareness ──────────────────────────────────────────
    turn_number: int  # 1-indexed; incremented each reasoning step; persisted by checkpointer
    context_entities: dict[
        str, list[str]
    ]  # Entities tracked across turns: {"tickers": [...], "sectors": [...]}
