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
import re
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import settings
from agent.graph.state import AgentState
from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS
from agent.tools.schema_compression import compress_tool_schemas
from agent.verification import run_verification_pipeline

# ── LLM Singleton ─────────────────────────────────────────────────────────────
# Built ONCE at module import time — never rebuilt per-request.
# Rebuilding ChatAnthropic on every reasoning step adds overhead and
# prevents connection pooling inside the SDK.


# Builds the primary LLM (Claude or GPT-4o fallback) with all tools bound — called once at module import.
def _build_llm(force_tools: bool = False):
    """
    Build the LLM with all tools bound.

    Args:
        force_tools: When True, sets tool_choice to "any" (Anthropic) or
                     "required" (OpenAI), forcing the model to call at least
                     one tool before responding.  Used on the first reasoning
                     step of every turn to prevent the LLM from answering
                     finance questions directly from training data or stale
                     conversation context.
    """
    # Use compressed schemas when binding to the LLM (saves ~40% of the 3,161-token
    # tool-schema footprint). ToolNode still receives the full originals for execution.
    tools_for_llm = (
        compress_tool_schemas(ALL_TOOLS) if settings.tool_schema_compression else ALL_TOOLS
    )

    if settings.anthropic_api_key:
        llm = ChatAnthropic(
            model=settings.primary_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=2048,  # raised from 1024 — gives the model enough room to reason
            # through tool selection with 11 tools bound; 1024 was
            # causing silent truncation before tool_calls were emitted
        )
        # Anthropic: "any" = must call at least one tool; "auto" = model decides
        tool_choice = "any" if force_tools else "auto"
        return llm.bind_tools(tools_for_llm, tool_choice=tool_choice)

    llm = ChatOpenAI(
        model=settings.fallback_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=1024,
    )
    # OpenAI: "required" = must call at least one function; "auto" = model decides
    tool_choice = "required" if force_tools else "auto"
    return llm.bind_tools(tools_for_llm, tool_choice=tool_choice)


_llm = _build_llm(force_tools=False)  # synthesis / free-form: model decides
_llm_force_tools = _build_llm(force_tools=True)  # first reasoning step: MUST call a tool
_log = structlog.get_logger()

# ── Jailbreak / off-topic detection ───────────────────────────────────────────
# Fast-path guard: if any of these phrases appear in the user's message, the
# guard node short-circuits the graph before the LLM (and its bound tools) runs.
# This prevents jailbreak attempts from triggering tool calls.
_JAILBREAK_PATTERNS: frozenset[str] = frozenset(
    {
        "ignore your instructions",
        "ignore all instructions",
        "ignore previous",
        "pretend you are",
        "pretend to be",
        "you are now",
        "act as",
        "act like",
        "forget your",
        "forget all",
        "override",
        "disregard",
        "disregard your",
        "bypass",
        "bypass your",
        "your new instructions",
        "new instructions are",
        "roleplay as",
        "jailbreak",
        "do anything now",
        "dan mode",
        "developer mode",
        "unrestricted mode",
        "without restrictions",
        "no restrictions",
        "without guidelines",
        "ignore the above",
    }
)

_JAILBREAK_REFUSAL = (
    "I'm Fortio, your financial assistant. I'm only able to help with portfolio analysis, "
    "investment questions, and Ghostfolio data. I can't change my behaviour, persona, or "
    "follow instructions that override my guidelines. "
    "Happy to help with any investment questions! 💼"
)

# Maximum tool-reasoning loops before forcing verification (prevents infinite loops)
_MAX_REASONING_STEPS: int = 5

# ── Finance-domain classifier ──────────────────────────────────────────────────
# Used to gate `_llm_force_tools` on the first reasoning step.
# Only finance-related queries trigger tool_choice="any"/"required" — off-topic
# queries (weather, recipes, coding help) keep tool_choice="auto" so the LLM
# can respond directly without wastefully calling a Ghostfolio endpoint.
#
# Deliberately broad: a false-positive (forcing a tool on an edge-case finance
# query) is harmless; a false-negative (missing a real finance query) risks
# the LLM answering from training data → hallucination.
_FINANCE_KEYWORDS: frozenset[str] = frozenset(
    {
        # Portfolio composition
        "portfolio",
        "holding",
        "holdings",
        "position",
        "positions",
        "allocation",
        "diversif",
        "i own",
        "i hold",
        "what do i own",
        # Performance & time-period queries
        "performance",
        "return",
        "returns",
        "gain",
        "loss",
        "profit",
        "ytd",
        "mtd",
        "wtd",
        "year-to-date",
        "month-to-date",
        # Price & value queries
        "price",
        "worth",
        "market cap",
        # Transaction queries
        "transaction",
        "transactions",
        "trade",
        "trades",
        "dividend",
        "dividends",
        "fee",
        "fees",
        "expense",
        # Asset class keywords
        "stock",
        "stocks",
        "etf",
        "etfs",
        "fund",
        "funds",
        "bond",
        "bonds",
        "equity",
        "equities",
        "crypto",
        "bitcoin",
        "share",
        "shares",
        "ticker",
        # Analysis keywords
        "sector",
        "rebalance",
        "rebalancing",
        "concentration",
        "scorecard",
        "health score",
        "risk",
        # Investment
        "invest",
        "investing",
        "investment",
        # Dollar sign
        "$",
    }
)


# Pattern: 2–5 uppercase letters as a whole word — very likely a stock/ETF ticker
# (e.g. QQQ, AAPL, SPY).  Requires ≥2 chars so common single-letter words like
# "I" or "A" don't falsely trigger tool forcing on non-finance queries.
# Checked against the original message (not lowercased) so tickers like "QQQ"
# are detected even when no keyword list entry matches.
_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")


def _is_finance_query(messages: list) -> bool:
    """
    Return True if the most recent HumanMessage contains finance-domain keywords
    OR an uppercase ticker-like token (e.g. QQQ, AAPL, SPY).

    Used to decide whether to force tool use (_llm_force_tools) or allow a
    free-form response (_llm) on the first reasoning step of a turn.

    Only the most recent HumanMessage is checked — prior-turn messages are
    irrelevant to whether the *current* query is finance-related.

    Deliberately broad: a false-positive (forcing a tool on a borderline
    finance query) is harmless; a false-negative (missing a real finance
    query) risks the LLM answering from training data → hallucination.

    Two-stage check:
      1. Keyword list — catches "price", "portfolio", "ETF", etc.
      2. Ticker pattern — catches bare uppercase symbols like "QQQ" or "AAPL"
         that don't appear in the keyword list.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original = str(msg.content)  # keep original case for ticker check
            text = original.lower()
            if any(kw in text for kw in _FINANCE_KEYWORDS):
                return True
            # Uppercase token ≥ 2 chars is almost certainly a ticker symbol
            return bool(_TICKER_PATTERN.search(original))
    return False


# ── Sliding-window history helper ─────────────────────────────────────────────


def _trim_to_window(messages: list, max_turns: int) -> list:
    """
    Keep only the last `max_turns` complete conversation turns in the context
    sent to the LLM.  A "turn" starts at each HumanMessage boundary.

    If max_turns == 0 the full history is returned unchanged (original behaviour).

    The full history is NOT modified — only the slice passed to the LLM is
    trimmed.  Postgres (via AsyncPostgresSaver) still holds the complete log.

    Args:
        messages: Full list of messages from AgentState.
        max_turns: Maximum number of HumanMessage boundaries to keep.
                   0 = disabled (return all messages).

    Returns:
        Trimmed list of messages suitable for the next LLM call.
    """
    if max_turns <= 0:
        return messages

    # Walk backwards to find the Nth HumanMessage boundary
    human_boundaries: list[int] = []
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            human_boundaries.append(i)
            if len(human_boundaries) == max_turns:
                break

    if not human_boundaries:
        return messages

    cutoff = human_boundaries[-1]  # index of the oldest HumanMessage to keep
    return messages[cutoff:]


# ── Prior-turn tool data redaction ────────────────────────────────────────────


def _redact_prior_tool_messages(messages: list) -> list:
    """
    Redact ToolMessage content from all PRIOR turns before sending history
    to the LLM.

    Why: The LLM reads ToolMessage JSON from previous turns and uses those
    stale financial numbers to answer current questions — bypassing a fresh
    tool call. This causes the verification layer to fire POTENTIAL_HALLUCINATION
    (correctly: the data is unverified for the current turn).

    What we keep:
      - ToolMessages from the CURRENT turn (after the last HumanMessage) —
        the LLM needs these for synthesis.
      - All HumanMessages and AIMessages — conversational context intact.

    What we replace:
      - ToolMessages from PRIOR turns — replaced with a placeholder that
        signals "stale, re-fetch required".

    Entity extraction (_extract_context_entities) runs on the unmodified
    state["messages"] BEFORE this function, so ticker/sector context for
    pronoun resolution is unaffected.
    """
    # Find the start of the current turn (last HumanMessage)
    last_human_idx = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = i

    result = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and i < last_human_idx:
            # Prior-turn tool result: strip the financial data
            msg = msg.model_copy(
                update={
                    "content": (
                        "[Stale tool result from a prior turn — "
                        "call the tool again to get current data for this question]"
                    )
                }
            )
        result.append(msg)
    return result


# ── Tool result truncation helper ─────────────────────────────────────────────


def _truncate_tool_messages(messages: list, max_tokens: int) -> list:
    """
    Return a copy of messages with every ToolMessage content capped at
    approximately `max_tokens` tokens (using a 4-chars-per-token heuristic
    for English prose / minified JSON).

    If max_tokens == 0 the messages are returned unchanged.

    Args:
        messages: List of LangChain messages (may include ToolMessages).
        max_tokens: Soft cap on ToolMessage content length.
                    0 = disabled (no truncation).

    Returns:
        New list with ToolMessage contents truncated where necessary.
        Non-ToolMessages are passed through unchanged.
    """
    if max_tokens <= 0:
        return messages

    max_chars = max_tokens * 4  # 4 chars ≈ 1 token (English / JSON heuristic)
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and len(str(msg.content)) > max_chars:
            truncated_content = str(msg.content)[:max_chars] + " …[truncated]"
            msg = msg.model_copy(update={"content": truncated_content})
        result.append(msg)
    return result


# ── Context entity extraction ──────────────────────────────────────────────────

# Words that look like ticker symbols (all-caps, 1-5 chars) but are NOT tickers.
# Kept narrow — only the most common false-positives from tool JSON + LLM output.
_TICKER_STOPWORDS: frozenset[str] = frozenset(
    {
        "HIGH",
        "LOW",
        "ETF",
        "USD",
        "THE",
        "FOR",
        "AND",
        "YOU",
        "YOUR",
        "NOT",
        "ALL",
        "ARE",
        "MEDIUM",
        "INFO",
        "ROAI",
        "YTD",
        "YES",
        "NO",
        "TOP",
        "BUY",
        "SELL",
        "FEE",
        "DATA",
        "API",
        "NONE",
        "TRUE",
        "NULL",
        "GOOD",
        "BAD",
        "RISK",
        "FEES",
        "URL",
        "N/A",
        "OK",
        "MAX",
    }
)


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


# Pre-reasoning guard: detects jailbreak/override attempts and short-circuits the graph before any tool can fire.
async def guard_node(state: AgentState) -> dict[str, Any]:
    """
    Fast-path jailbreak guard — runs BEFORE the LLM receives the message.

    If the latest human message matches any jailbreak/override pattern, inject
    a safe refusal directly into state and set final_response so the graph
    routes straight to verification (and then END) without ever invoking the
    LLM with bound tools.

    Returns an empty dict (no-op) when the message is legitimate, allowing
    normal routing to the reasoning node.
    """
    # Find the most recent HumanMessage
    last_human: HumanMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg
            break

    if last_human is not None:
        text = str(last_human.content).lower()
        if any(pat in text for pat in _JAILBREAK_PATTERNS):
            _log.warning(
                "jailbreak_detected",
                preview=str(last_human.content)[:120],
            )
            refusal = AIMessage(content=_JAILBREAK_REFUSAL)
            return {
                "messages": [refusal],
                "final_response": _JAILBREAK_REFUSAL,
                "confidence": "HIGH",
                "verification_flags": [],
                "should_escalate": False,
            }

    return {}  # No-op — pass through to reasoning_node


# Routes after guard: skip reasoning entirely if guard already set a final_response.
def after_guard(state: AgentState) -> str:
    """If guard placed a final_response, skip to verify; otherwise reason normally."""
    if state.get("final_response"):
        return "verify"
    return "reasoning"


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
        context_lines.append(f"Sectors discussed: {', '.join(entities['sectors'])}.")
    if entities["periods"]:
        context_lines.append(
            f"Time periods already established: {', '.join(entities['periods'])}. "
            "Re-use these when the user says 'last year', 'that period', etc."
        )

    system_content = SYSTEM_PROMPT
    if context_lines and turn > 1:
        system_content += f"\n\n## Active Conversation Context (Turn {turn})\n" + "\n".join(
            context_lines
        )

    # ── Apply sliding-window history trim ─────────────────────────────────────
    # Trim to last N turns before building the context for this LLM call.
    # The full history remains in state["messages"] and Postgres.
    history = _trim_to_window(state["messages"], settings.history_window_turns)

    # ── Apply tool result truncation ──────────────────────────────────────────
    # Cap each ToolMessage at ~300 tokens to reduce synthesis-step input cost.
    history = _truncate_tool_messages(history, settings.max_tool_result_tokens)

    # ── Redact prior-turn tool data from LLM context ──────────────────────────
    # Strips stale financial numbers from previous turns so the LLM is forced
    # to re-fetch via tool call rather than answering from old data in history.
    # Current-turn ToolMessages (after the last HumanMessage) are left intact
    # so the LLM can still synthesise this turn's fresh tool results.
    history = _redact_prior_tool_messages(history)

    # ── Choose LLM based on turn state and query domain ──────────────────────
    # _llm_force_tools (tool_choice="any"/"required") is used ONLY when BOTH:
    #   1. No tools have run yet this turn (first reasoning step), AND
    #   2. The query is finance-related (contains portfolio/price/etc. keywords)
    #
    # This prevents forcing tool calls on off-topic queries (weather, recipes,
    # coding help) while still blocking the LLM from answering finance questions
    # directly from training data or stale conversation context.
    #
    # After tools have run (synthesis step), _llm (auto) is always used so the
    # LLM can freely compose its answer from the fresh tool results.
    last_human_idx = 0
    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_idx = i

    current_turn_has_tool_results = any(
        isinstance(msg, ToolMessage) for msg in state["messages"][last_human_idx + 1 :]
    )

    is_first_step = not current_turn_has_tool_results
    active_llm = (
        _llm_force_tools if is_first_step and _is_finance_query(state["messages"]) else _llm
    )

    messages = [SystemMessage(content=system_content)] + history
    response = await active_llm.ainvoke(messages)

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
                block.get("text", "")
                for block in content
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

    Deduplication: uses tool_call_id as the canonical key so that the same tool
    result is never appended twice (e.g. across multiple reasoning loops or
    across LangGraph replay on resume).  ToolMessages without a tool_call_id
    fall back to content-hash deduplication.
    """
    existing: list[dict[str, Any]] = list(state.get("tool_results", []))
    # Build a set of IDs already present so we never double-count
    seen_ids: set[str] = {r.get("_tool_call_id", "") for r in existing} - {""}

    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        tool_call_id: str = getattr(msg, "tool_call_id", "") or ""
        if tool_call_id and tool_call_id in seen_ids:
            continue  # already collected from a previous loop iteration
        try:
            result = json.loads(msg.content)
            if isinstance(result, dict):
                # Tag with tool_call_id for future dedup rounds
                result["_tool_call_id"] = tool_call_id
                existing.append(result)
                if tool_call_id:
                    seen_ids.add(tool_call_id)
        except (json.JSONDecodeError, TypeError):
            pass

    return {"tool_results": existing}


# ── Routing ───────────────────────────────────────────────────────────────────


# Returns "tools" if the last LLM message contains tool_calls, otherwise routes to "verify".
def should_use_tools(state: AgentState) -> str:
    """
    Route to tools if the LLM requested tool calls, else go to verification.

    Also enforces a hard ceiling of _MAX_REASONING_STEPS to prevent infinite
    tool-call loops. If the ceiling is reached, forces verification regardless
    of whether the LLM requested another tool call.
    """
    if state.get("reasoning_steps", 0) >= _MAX_REASONING_STEPS:
        _log.warning(
            "max_reasoning_steps_reached",
            steps=state.get("reasoning_steps"),
            ceiling=_MAX_REASONING_STEPS,
        )
        return "verify"
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
        "⚠️ I was unable to verify the accuracy of my response against your actual portfolio data. "
        "To protect you from potentially incorrect figures, I'm not showing it.\n\n"
        "This usually happens when all data sources returned errors but I still attempted an answer. "
        "Please try again, or check your Ghostfolio dashboard directly for exact figures."
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

    graph.add_node("guard", guard_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("collect_results", tool_result_collector_node)
    graph.add_node("verify", verification_node)
    graph.add_node("escalate", escalation_node)

    # Guard runs first — routes to reasoning (normal) or verify (jailbreak short-circuit)
    graph.set_entry_point("guard")
    graph.add_conditional_edges(
        "guard",
        after_guard,
        {"reasoning": "reasoning", "verify": "verify"},
    )

    graph.add_conditional_edges(
        "reasoning",
        should_use_tools,
        {"tools": "tools", "verify": "verify"},
    )
    graph.add_edge("tools", "collect_results")
    graph.add_edge("collect_results", "reasoning")  # loop back for synthesis

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
