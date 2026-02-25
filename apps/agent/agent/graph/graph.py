"""
LangGraph agent graph for Fortio, the Ghostfolio Finance Agent.
Nodes: reasoning → tool_execution → verification → output

Checkpointing:
  The graph is compiled with an optional checkpointer.
  When a checkpointer is supplied, LangGraph automatically saves and restores
  AgentState between turns using `thread_id` from the run config:

      config = {"configurable": {"thread_id": conversation_id}}
      await graph.ainvoke(state, config=config)

  - FastAPI uses AsyncPostgresSaver (persistent across restarts)
  - Chainlit uses MemorySaver (in-memory, sufficient for dev sessions)
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import settings
from agent.graph.state import AgentState
from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS
from agent.verification import run_verification_pipeline

# ── LLM Singleton ─────────────────────────────────────────────────────────────
# Built ONCE at module import time — never rebuilt per-request.
# Rebuilding ChatAnthropic on every reasoning step adds overhead and
# prevents connection pooling inside the SDK.

def _build_llm():
    """Build the LLM with all tools bound. Called exactly once."""
    if settings.anthropic_api_key:
        return ChatAnthropic(
            model=settings.primary_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1024,   # cap output — portfolio answers are concise
        ).bind_tools(ALL_TOOLS)
    return ChatOpenAI(
        model=settings.fallback_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=1024,
    ).bind_tools(ALL_TOOLS)


_llm = _build_llm()   # ← module-level singleton
_log = structlog.get_logger()


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def reasoning_node(state: AgentState) -> dict[str, Any]:
    """
    Call the LLM with conversation history.
    The LLM decides which tools to call (or responds directly).
    Uses the module-level LLM singleton — no re-instantiation per request.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await _llm.ainvoke(messages)
    return {
        "messages": [response],
        "reasoning_steps": state.get("reasoning_steps", 0) + 1,
    }


async def verification_node(state: AgentState) -> dict[str, Any]:
    """
    Run all 5 verification checks on the final LLM response.
    Collect all tool results from the message history.
    """
    # Extract final text response
    last_message = state["messages"][-1]
    response_text = ""
    if hasattr(last_message, "content"):
        content = last_message.content
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            response_text = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

    # Collect tool results from message history
    tool_results = state.get("tool_results", [])

    # Run verification pipeline
    result = run_verification_pipeline(
        response=response_text,
        tool_results=tool_results,
        reasoning_steps=state.get("reasoning_steps", 1),
    )

    # Check for escalation trigger
    should_escalate = result["has_high_severity"] and any(
        f["type"] == "POTENTIAL_HALLUCINATION" for f in result["verification_flags"]
    )

    return {
        "final_response": result["response"],
        "verification_flags": result["verification_flags"],
        "confidence": result["confidence"],
        "should_escalate": should_escalate,
    }


def tool_result_collector_node(state: AgentState) -> dict[str, Any]:
    """
    After tools execute, collect their results into state for the verification layer.
    """
    tool_results = list(state.get("tool_results", []))
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content)
                if isinstance(result, dict):
                    tool_results.append(result)
            except (json.JSONDecodeError, TypeError):
                pass
    return {"tool_results": tool_results}


# ── Routing ───────────────────────────────────────────────────────────────────

def should_use_tools(state: AgentState) -> str:
    """Route to tools if the LLM requested tool calls, else go to verification."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "verify"


def should_escalate(state: AgentState) -> str:
    """Route to escalation node if high-severity hallucination detected."""
    if state.get("should_escalate"):
        return "escalate"
    return END


# ── Nodes (continued) ─────────────────────────────────────────────────────────

async def escalation_node(state: AgentState) -> dict[str, Any]:
    """
    Human-in-the-loop escalation handler.
    Triggered when verification detects a high-severity POTENTIAL_HALLUCINATION.
    Replaces the agent's response with a safe fallback and logs the incident.

    In production this would: page an on-call engineer, open a support ticket,
    or route to a human advisor. For now it logs and returns a safe message.
    """
    _log.warning(
        "escalation_triggered",
        flags=state.get("verification_flags", []),
        confidence=state.get("confidence"),
    )
    safe_response = (
        "⚠️ I detected a potential issue with the accuracy of my previous response. "
        "For safety I'm withholding it.\n\n"
        "Please try rephrasing your question, or consult your Ghostfolio dashboard "
        "directly for exact figures. If this keeps happening, contact support."
    )
    return {
        "final_response": safe_response,
        "messages": [AIMessage(content=safe_response)],
    }


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph agent graph.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. AsyncPostgresSaver,
                      MemorySaver). When supplied, conversation state is
                      automatically persisted between turns using thread_id.
                      Pass None for a stateless graph (testing only).
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("collect_results", tool_result_collector_node)
    graph.add_node("verify", verification_node)
    graph.add_node("escalate", escalation_node)

    graph.set_entry_point("reasoning")

    graph.add_conditional_edges(
        "reasoning",
        should_use_tools,
        {"tools": "tools", "verify": "verify"},
    )
    graph.add_edge("tools", "collect_results")
    graph.add_edge("collect_results", "reasoning")   # loop back for synthesis

    # After verification: either end cleanly or escalate on hallucination
    graph.add_conditional_edges(
        "verify",
        should_escalate,
        {"escalate": "escalate", END: END},
    )
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)


# ── Default singleton (Chainlit dev UI) ───────────────────────────────────────
# Uses MemorySaver — persists for the lifetime of the process.
# FastAPI creates its own graph instance via lifespan with AsyncPostgresSaver.
agent_graph = build_graph(checkpointer=MemorySaver())
