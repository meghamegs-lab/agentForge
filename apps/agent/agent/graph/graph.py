# Assembles the LangGraph reasoning loop: reasoning → tools → collect_results → verify → end.
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
  - CLI/dev uses MemorySaver (in-memory, sufficient for a single process)
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

# Builds the primary LLM (Claude or GPT-4o fallback) with all tools bound — called once at module import.
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


# ── Context entity extraction ──────────────────────────────────────────────────

# Words that look like ticker symbols (all-caps, 1-5 chars) but are NOT tickers.
# Kept narrow — only the most common false-positives from tool JSON + LLM output.
_TICKER_STOPWORDS: frozenset[str] = frozenset({
    "HIGH", "LOW", "ETF", "USD", "THE", "FOR", "AND", "YOU", "YOUR", "NOT",
    "ALL", "ARE", "MEDIUM", "INFO", "ROAI", "YTD", "YES", "NO", "TOP",
    "BUY", "SELL", "FEE", "DATA", "API", "NONE", "TRUE", "NULL", "GOOD",
    "BAD", "RISK", "FEES", "URL", "N/A", "OK", "MAX",
})


# Scans ToolMessage results in conversation history to extract ticker symbols, sectors, and time periods.
def _extract_context_entities(messages: list) -> dict[str, list[str]]:
    """
    Scan ToolMessage results in the conversation history to extract structured
    entities: tickers, sectors, and time periods.

    This is called once per reasoning step and the result is:
      1. Stored in AgentState.context_entities (persisted by checkpointer)
      2. Injected into the system message as an "Active Conversation Context"
         block so the LLM can resolve pronouns like "those" or "them" without
         re-calling get_portfolio_summary unnecessarily.

    Only ToolMessages are scanned (structured JSON) to avoid false positives
    from LLM prose.
    """
    tickers: set[str] = set()
    sectors: set[str] = set()
    periods: set[str] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else {}
        except (json.JSONDecodeError, TypeError):
            continue

        # Holdings list: [{"symbol": "AAPL", "allocation": 0.15, ...}]
        for holding in data.get("holdings", []):
            if sym := holding.get("symbol"):
                tickers.add(str(sym).upper())

        # Single-symbol market data response: {"symbol": "NVDA", "price": ...}
        if sym := data.get("symbol"):
            tickers.add(str(sym).upper())

        # Batch market data: {"data": {"AAPL": {...}, "MSFT": {...}}}
        for sym in data.get("data", {}):
            if isinstance(sym, str) and 1 <= len(sym) <= 5 and sym.replace(".", "").isalpha():
                tickers.add(sym.upper())

        # Sector breakdown (analyze_diversification tool)
        for entry in data.get("sectors", []) + data.get("sector_breakdown", []):
            name = entry.get("name") or entry.get("sector")
            if name and isinstance(name, str):
                sectors.add(name)

        # Time periods present in performance data
        for period in ("1d", "ytd", "1y", "5y", "max"):
            if period in data:
                periods.add(period)

    return {
        "tickers": sorted(tickers - _TICKER_STOPWORDS),
        "sectors": sorted(sectors),
        "periods": sorted(periods),
    }


# ── Nodes ─────────────────────────────────────────────────────────────────────

# Invokes the LLM with the full conversation history and a dynamically injected context block for pronoun resolution.
async def reasoning_node(state: AgentState) -> dict[str, Any]:
    """
    Call the LLM with conversation history + dynamic context injection.

    On every reasoning step:
    1. Extract tickers/sectors/periods from all ToolMessages in history
    2. If turn > 1 and entities exist, append an "Active Conversation Context"
       block to the system prompt — this is what lets Claude resolve "THOSE"
       correctly without asking the user to repeat the stock names
    3. Increment turn_number and persist context_entities back to state
       (the checkpointer saves both for the next turn)
    """
    # Increment turn counter (0 on first ever turn → 1 after this node runs)
    turn = state.get("turn_number", 0) + 1

    # Extract structured entities from prior tool results
    entities = _extract_context_entities(state["messages"])

    # Build dynamic context block (only injected from turn 2 onwards when
    # there are entities to share — avoids cluttering the first-turn prompt)
    context_lines: list[str] = []
    if entities["tickers"]:
        symbols = ", ".join(entities["tickers"])
        context_lines.append(
            f"Securities discussed in this conversation: {symbols}. "
            "When the user says 'those', 'them', 'it', or 'that stock', "
            f"resolve to these symbols — do NOT ask for clarification."
        )
    if entities["sectors"]:
        context_lines.append(
            f"Sectors discussed: {', '.join(entities['sectors'])}."
        )
    if entities["periods"]:
        context_lines.append(
            f"Time periods already established: {', '.join(entities['periods'])}. "
            "Re-use these when the user says 'last year', 'that period', etc."
        )

    system_content = SYSTEM_PROMPT
    if context_lines and turn > 1:
        system_content += (
            f"\n\n## Active Conversation Context (Turn {turn})\n"
            + "\n".join(context_lines)
        )

    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = await _llm.ainvoke(messages)

    return {
        "messages": [response],
        "reasoning_steps": state.get("reasoning_steps", 0) + 1,
        "turn_number": turn,
        "context_entities": entities,
    }


# Runs the 5-stage verification pipeline on the final LLM response before it is returned to the user.
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


# Harvests ToolMessage JSON payloads from the message list and accumulates them in state.tool_results.
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

# Returns "tools" if the last LLM message contains tool_calls, otherwise routes to "verify".
def should_use_tools(state: AgentState) -> str:
    """Route to tools if the LLM requested tool calls, else go to verification."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "verify"


# Returns "escalate" if a HIGH-severity hallucination flag is present, otherwise ends the graph normally.
def should_escalate(state: AgentState) -> str:
    """Route to escalation node if high-severity hallucination detected."""
    if state.get("should_escalate"):
        return "escalate"
    return END


# ── Nodes (continued) ─────────────────────────────────────────────────────────

# Triggered on confirmed hallucination — replaces the agent response with a safe fallback and logs the event.
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

# Wires all nodes and edges into a compiled LangGraph StateGraph with an optional checkpointer.
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


# ── Default singleton (CLI / dev) ─────────────────────────────────────────────
# Uses MemorySaver — persists for the lifetime of the process.
# FastAPI creates its own graph instance via lifespan with AsyncPostgresSaver.
agent_graph = build_graph(checkpointer=MemorySaver())
