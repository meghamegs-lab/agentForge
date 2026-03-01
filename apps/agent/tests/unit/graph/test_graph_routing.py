"""
Unit tests for agent/graph/graph.py routing functions and helpers.
==================================================================
Tests the pure-Python routing conditions, context-entity extractor,
and the graph compilation itself — without any real LLM or network calls.

What's tested:
  - should_use_tools()    — "tools" when tool_calls present, "verify" otherwise
  - should_escalate()     — "escalate" when flag set, END otherwise
  - _extract_context_entities() — ticker/sector extraction from ToolMessages
  - build_graph()         — graph compiles without error
  - tool_result_collector_node() — accumulates ToolMessage payloads into state
  - _looks_financial()    — boundary cases for the hallucination guard helper
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

from agent.graph.graph import (
    _extract_context_entities,
    _redact_prior_tool_messages,
    build_graph,
    should_escalate,
    should_use_tools,
    tool_result_collector_node,
)
from agent.verification import _looks_financial

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_state(
    messages: list,
    should_escalate_flag: bool = False,
    tool_results: list | None = None,
    **kwargs,
) -> dict:
    """Build a minimal AgentState dict for routing tests."""
    return {
        "messages": messages,
        "tool_results": tool_results or [],
        "verification_flags": [],
        "confidence": "HIGH",
        "reasoning_steps": 1,
        "conversation_id": "test-conv",
        "user_id": "test-user",
        "final_response": "",
        "should_escalate": should_escalate_flag,
        "turn_number": 1,
        "context_entities": {},
        **kwargs,
    }


def _tool_message(content: dict, name: str = "get_portfolio_summary") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content),
        name=name,
        tool_call_id="call-123",
    )


# ── should_use_tools ──────────────────────────────────────────────────────────


class TestShouldUseTools:
    def test_returns_tools_when_last_message_has_tool_calls(self):
        ai_msg = AIMessage(
            content="", tool_calls=[{"id": "call-1", "name": "get_portfolio_summary", "args": {}}]
        )
        state = _make_state([HumanMessage(content="Show me my portfolio"), ai_msg])
        assert should_use_tools(state) == "tools"

    def test_returns_verify_when_no_tool_calls(self):
        ai_msg = AIMessage(content="Your portfolio is worth $10,000.")
        state = _make_state([HumanMessage(content="Show me my portfolio"), ai_msg])
        assert should_use_tools(state) == "verify"

    def test_returns_verify_when_tool_calls_is_empty_list(self):
        ai_msg = AIMessage(content="Done.", tool_calls=[])
        state = _make_state([ai_msg])
        assert should_use_tools(state) == "verify"

    def test_uses_last_message_not_first(self):
        """Only the LAST message's tool_calls should determine routing."""
        first = AIMessage(
            content="", tool_calls=[{"id": "c1", "name": "get_portfolio_summary", "args": {}}]
        )
        second = AIMessage(content="Here is the summary.")
        state = _make_state([first, second])
        # Last message has no tool_calls → should route to verify
        assert should_use_tools(state) == "verify"


# ── should_escalate ───────────────────────────────────────────────────────────


class TestShouldEscalate:
    def test_returns_escalate_when_flag_is_true(self):
        state = _make_state([AIMessage(content="ok")], should_escalate_flag=True)
        assert should_escalate(state) == "escalate"

    def test_returns_end_when_flag_is_false(self):
        state = _make_state([AIMessage(content="ok")], should_escalate_flag=False)
        assert should_escalate(state) == END

    def test_returns_end_when_flag_missing_from_state(self):
        state = _make_state([AIMessage(content="ok")])
        state.pop("should_escalate")
        assert should_escalate(state) == END


# ── _extract_context_entities ────────────────────────────────────────────────


class TestExtractContextEntities:
    def test_extracts_tickers_from_holdings_tool_message(self):
        msg = _tool_message(
            {
                "holdings": [
                    {"symbol": "AAPL", "value": 1750},
                    {"symbol": "VTI", "value": 4200},
                    {"symbol": "MSFT", "value": 2050},
                ]
            }
        )
        result = _extract_context_entities([msg])
        assert "AAPL" in result["tickers"]
        assert "VTI" in result["tickers"]
        assert "MSFT" in result["tickers"]

    def test_extracts_single_symbol_from_market_data_message(self):
        msg = _tool_message(
            {"symbol": "NVDA", "current_price": 450.0, "status": "ok"},
            name="get_market_data",
        )
        result = _extract_context_entities([msg])
        assert "NVDA" in result["tickers"]

    def test_filters_stopwords_from_tickers(self):
        """HIGH, LOW, USD, BUY, SELL etc. must not appear in extracted tickers."""
        msg = _tool_message(
            {
                "holdings": [
                    {"symbol": "HIGH", "value": 100},  # stopword
                    {"symbol": "USD", "value": 100},  # stopword
                    {"symbol": "AAPL", "value": 1750},  # real ticker
                ]
            }
        )
        result = _extract_context_entities([msg])
        assert "HIGH" not in result["tickers"]
        assert "USD" not in result["tickers"]
        assert "AAPL" in result["tickers"]

    def test_extracts_sectors_from_diversification_message(self):
        msg = _tool_message(
            {
                "sector_breakdown": [
                    {"sector": "Technology", "percent": 60},
                    {"sector": "Healthcare", "percent": 20},
                ]
            }
        )
        result = _extract_context_entities([msg])
        assert "Technology" in result["sectors"]
        assert "Healthcare" in result["sectors"]

    def test_extracts_sectors_from_sectors_key(self):
        msg = _tool_message(
            {
                "sectors": [
                    {"name": "Financials", "weight": 0.3},
                ]
            }
        )
        result = _extract_context_entities([msg])
        assert "Financials" in result["sectors"]

    def test_ignores_non_tool_messages(self):
        """HumanMessage and AIMessage content must not contribute entities."""
        messages = [
            HumanMessage(content='{"holdings": [{"symbol": "FAKE"}]}'),
            AIMessage(content='{"symbol": "ALSONOTREAL"}'),
        ]
        result = _extract_context_entities(messages)
        assert "FAKE" not in result["tickers"]
        assert "ALSONOTREAL" not in result["tickers"]

    def test_handles_malformed_json_tool_message_gracefully(self):
        """Non-JSON ToolMessage content must not crash the extractor."""
        msg = ToolMessage(
            content="not valid json {{",
            name="get_portfolio_summary",
            tool_call_id="call-bad",
        )
        result = _extract_context_entities([msg])
        # Should return empty containers, not raise
        assert result["tickers"] == []
        assert result["sectors"] == []

    def test_returns_sorted_tickers(self):
        msg = _tool_message(
            {
                "holdings": [
                    {"symbol": "VTI", "value": 4000},
                    {"symbol": "AAPL", "value": 1000},
                    {"symbol": "MSFT", "value": 2000},
                ]
            }
        )
        result = _extract_context_entities([msg])
        assert result["tickers"] == sorted(result["tickers"])

    def test_empty_messages_returns_empty_containers(self):
        result = _extract_context_entities([])
        assert result == {"tickers": [], "sectors": [], "periods": []}

    def test_extracts_from_batch_market_data(self):
        msg = _tool_message(
            {
                "data": {"AAPL": {"price": 175}, "MSFT": {"price": 400}},
                "status": "ok",
            }
        )
        result = _extract_context_entities([msg])
        assert "AAPL" in result["tickers"]
        assert "MSFT" in result["tickers"]


# ── tool_result_collector_node ────────────────────────────────────────────────


class TestToolResultCollectorNode:
    def test_collects_tool_message_json_into_tool_results(self):
        payload = {"status": "ok", "total_value": 8000.0}
        msg = _tool_message(payload)
        state = _make_state([msg], tool_results=[])
        result = tool_result_collector_node(state)
        # _tool_call_id is injected for deduplication — strip it before comparing
        stripped = [
            {k: v for k, v in r.items() if k != "_tool_call_id"} for r in result["tool_results"]
        ]
        assert payload in stripped

    def test_accumulates_with_existing_tool_results(self):
        existing = {"status": "ok", "symbol": "AAPL"}
        new_payload = {"status": "ok", "total_value": 5000.0}
        msg = _tool_message(new_payload)
        state = _make_state([msg], tool_results=[existing])
        result = tool_result_collector_node(state)
        # _tool_call_id is injected for deduplication — strip it before comparing
        stripped = [
            {k: v for k, v in r.items() if k != "_tool_call_id"} for r in result["tool_results"]
        ]
        assert existing in stripped
        assert new_payload in stripped

    def test_ignores_non_tool_messages(self):
        ai_msg = AIMessage(content='{"status": "ok"}')
        state = _make_state([ai_msg], tool_results=[])
        result = tool_result_collector_node(state)
        assert result["tool_results"] == []

    def test_ignores_malformed_json_content(self):
        """Non-JSON ToolMessage must be skipped without crashing."""
        msg = ToolMessage(
            content="not json",
            name="get_portfolio_summary",
            tool_call_id="c",
        )
        state = _make_state([msg], tool_results=[])
        result = tool_result_collector_node(state)
        assert result["tool_results"] == []

    def test_ignores_non_dict_json(self):
        """A JSON array in ToolMessage content should be skipped."""
        msg = ToolMessage(
            content='["AAPL", "MSFT"]',
            name="get_market_data",
            tool_call_id="c",
        )
        state = _make_state([msg], tool_results=[])
        result = tool_result_collector_node(state)
        assert result["tool_results"] == []


# ── build_graph ───────────────────────────────────────────────────────────────


class TestBuildGraph:
    def test_compiles_without_error(self):
        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_compiles_with_memory_saver(self):
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Verify the graph contains the required architectural nodes."""
        graph = build_graph(checkpointer=None)
        node_names = set(graph.nodes.keys())
        for expected in ("reasoning", "tools", "collect_results", "verify", "escalate"):
            assert expected in node_names, (
                f"Expected node '{expected}' not found in compiled graph nodes: {node_names}"
            )


# ── _redact_prior_tool_messages ──────────────────────────────────────────────


class TestRedactPriorToolMessages:
    """
    Unit tests for the prior-turn tool-data redaction helper.

    The function must:
      - Leave ToolMessages from the CURRENT turn (after the last HumanMessage) intact
      - Replace ToolMessages from PRIOR turns with a placeholder
      - Pass HumanMessages and AIMessages through unchanged in all cases
    """

    _PLACEHOLDER = "[Stale tool result from a prior turn"

    def _tool_msg(
        self, content: str = '{"value": 1000}', name: str = "get_portfolio_summary"
    ) -> ToolMessage:
        return ToolMessage(content=content, name=name, tool_call_id="c-1")

    def test_noop_when_single_turn_no_prior_tool_messages(self):
        """Single turn: HumanMessage → AIMessage (tool call) → ToolMessage.
        No prior turns exist, so nothing should be redacted."""
        human = HumanMessage(content="What do I own?")
        ai = AIMessage(
            content="", tool_calls=[{"id": "c-1", "name": "get_portfolio_summary", "args": {}}]
        )
        tool = self._tool_msg()
        messages = [human, ai, tool]

        result = _redact_prior_tool_messages(messages)

        # ToolMessage is in the current turn — must remain intact
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert '{"value": 1000}' in tool_msgs[0].content

    def test_redacts_tool_messages_from_prior_turns(self):
        """Two turns: prior ToolMessage must get the stale placeholder."""
        prior_human = HumanMessage(content="Show my portfolio")
        prior_tool = self._tool_msg('{"total_value": 8000}')
        current_human = HumanMessage(content="What are my top 5 holdings?")

        messages = [prior_human, prior_tool, current_human]
        result = _redact_prior_tool_messages(messages)

        prior_tool_result = result[1]  # index of the prior ToolMessage
        assert isinstance(prior_tool_result, ToolMessage)
        assert self._PLACEHOLDER in prior_tool_result.content
        assert "8000" not in prior_tool_result.content

    def test_preserves_current_turn_tool_messages(self):
        """ToolMessage AFTER the last HumanMessage must not be redacted."""
        prior_human = HumanMessage(content="Turn 1")
        prior_tool = self._tool_msg('{"old": True}')
        current_human = HumanMessage(content="Turn 2")
        current_tool = ToolMessage(
            content='{"fresh": True}', name="get_portfolio_summary", tool_call_id="c-2"
        )
        messages = [prior_human, prior_tool, current_human, current_tool]
        result = _redact_prior_tool_messages(messages)

        # prior ToolMessage (index 1) → redacted
        assert self._PLACEHOLDER in result[1].content
        # current ToolMessage (index 3) → intact
        assert '{"fresh": True}' in result[3].content

    def test_preserves_human_and_ai_messages_unchanged(self):
        """HumanMessages and AIMessages must pass through unchanged in all cases."""
        human = HumanMessage(content="Hello")
        ai = AIMessage(content="Hi there!")
        messages = [human, ai]
        result = _redact_prior_tool_messages(messages)

        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"

    def test_placeholder_text_present_in_redacted_content(self):
        """Redacted content must contain the canonical placeholder so the LLM
        can see the signal to re-fetch — not just an empty string."""
        prior_human = HumanMessage(content="Q1")
        prior_tool = self._tool_msg('{"secret": 42}')
        current_human = HumanMessage(content="Q2")
        messages = [prior_human, prior_tool, current_human]
        result = _redact_prior_tool_messages(messages)

        assert self._PLACEHOLDER in result[1].content

    def test_multiple_prior_turns_all_tool_messages_redacted(self):
        """Three-turn history: ToolMessages from turns 1 and 2 are both replaced."""
        t1_human = HumanMessage(content="Turn 1")
        t1_tool = self._tool_msg('{"t": 1}')
        t2_human = HumanMessage(content="Turn 2")
        t2_ai = AIMessage(
            content="", tool_calls=[{"id": "c-2", "name": "get_portfolio_summary", "args": {}}]
        )
        t2_tool = ToolMessage(content='{"t": 2}', name="get_portfolio_summary", tool_call_id="c-2")
        t3_human = HumanMessage(content="Turn 3")  # current turn — no tool yet

        messages = [t1_human, t1_tool, t2_human, t2_ai, t2_tool, t3_human]
        result = _redact_prior_tool_messages(messages)

        # Both prior-turn ToolMessages must be redacted
        assert self._PLACEHOLDER in result[1].content  # t1_tool
        assert self._PLACEHOLDER in result[4].content  # t2_tool
        # HumanMessages and AIMessage are unchanged
        assert result[0].content == "Turn 1"
        assert result[2].content == "Turn 2"
        assert result[5].content == "Turn 3"


# ── _looks_financial helper ───────────────────────────────────────────────────


class TestLooksFinancial:
    """
    Direct unit tests for the private helper in agent/verification/__init__.py.
    Critical because it determines which numbers trigger hallucination warnings.
    """

    def test_year_2024_is_not_financial(self):
        assert _looks_financial("2024") is False

    def test_year_2010_boundary_is_not_financial(self):
        assert _looks_financial("2010") is False

    def test_year_2035_boundary_is_not_financial(self):
        assert _looks_financial("2035") is False

    def test_year_2009_is_financial(self):
        # 2009 is below the year range — treated as a financial figure
        assert _looks_financial("2009") is True

    def test_small_integer_below_10_is_not_financial(self):
        assert _looks_financial("5") is False
        assert _looks_financial("1") is False
        assert _looks_financial("9") is False

    def test_ten_point_five_is_financial(self):
        assert _looks_financial("10.5") is True

    def test_price_175_32_is_financial(self):
        assert _looks_financial("175.32") is True

    def test_zero_is_not_financial(self):
        assert _looks_financial("0") is False

    def test_large_portfolio_value_is_financial(self):
        assert _looks_financial("10000") is True

    def test_invalid_string_returns_false(self):
        assert _looks_financial("abc") is False
        assert _looks_financial("") is False
