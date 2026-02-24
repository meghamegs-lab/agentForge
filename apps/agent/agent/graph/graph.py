"""
LangGraph agent graph for Fortio, the Ghostfolio Finance Agent.
Nodes: reasoning → tool_execution → verification → output
"""
from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.config import settings
from agent.graph.state import AgentState
from agent.tools import ALL_TOOLS
from agent.verification import run_verification_pipeline
from agent.prompts import SYSTEM_PROMPT


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
    """Route to escalation node if high-severity verification flags detected."""
    if state.get("should_escalate"):
        return "escalate"
    return END


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("collect_results", tool_result_collector_node)
    graph.add_node("verify", verification_node)

    graph.set_entry_point("reasoning")

    graph.add_conditional_edges(
        "reasoning",
        should_use_tools,
        {"tools": "tools", "verify": "verify"},
    )
    graph.add_edge("tools", "collect_results")
    graph.add_edge("collect_results", "reasoning")  # Loop back for synthesis
    graph.add_edge("verify", END)

    return graph.compile()


# Singleton graph instance
agent_graph = build_graph()
