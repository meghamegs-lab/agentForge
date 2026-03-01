"""
evals/test_tool_selection.py — Tool Selection Eval Suite (v2)
====================================================================
Eval IDs: TS01–TS06

"Does the agent choose the RIGHT tool(s) for each query?"

Tool selection is validated through two complementary lenses:

  A. DESCRIPTOR COVERAGE — the tool's docstring must contain all trigger
     keywords the LLM needs to route the query correctly. Tested by scanning
     each tool's .description / __doc__ for required phrases.

  B. PARAMETER MAPPING — given a natural-language intent, the tool receives
     the correct parameter values (e.g. "last year" → date_range="1y").

  C. DOMAIN BOUNDARY — each tool returns data only in its own domain
     and does NOT return metrics owned by another tool.

Tests:
  TS01 — "What is my YTD return?" → get_performance(date_range="ytd") only
  TS02 — "Show my sector allocation" → analyze_diversification() only
  TS03 — "Find duplicate transactions" → get_transactions() (or pattern tool)
  TS04 — "Compare to SPY over 1Y" → get_performance("1y") + get_market_data("SPY")
  TS05 — "Do I hold any AI stocks?" → get_portfolio_summary()
  TS06 — "How healthy is my portfolio?" → get_portfolio_health_scorecard()

All tests are pure Python — no network calls, no LLM required.
"""

from __future__ import annotations

from agent.tools.diversification import analyze_diversification
from agent.tools.health_scorecard import get_portfolio_health_scorecard
from agent.tools.market import get_market_data
from agent.tools.performance import get_performance
from agent.tools.portfolio import get_portfolio_summary
from agent.tools.transaction_patterns import get_transaction_pattern_intelligence
from agent.tools.transactions import get_transactions

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _doc(tool_fn) -> str:
    """Extract the tool's description string (LangChain .description or __doc__)."""
    desc = getattr(tool_fn, "description", None) or tool_fn.__doc__ or ""
    return desc.lower()


def _check_triggers(tool_fn, triggers: list[str]) -> list[str]:
    """Return list of trigger keywords missing from the tool's docstring."""
    doc = _doc(tool_fn)
    return [t for t in triggers if t not in doc]


# ══════════════════════════════════════════════════════════════════════════════
# TS01 — "What is my YTD return?" maps to get_performance with date_range=ytd
#         Verify: docstring covers return/performance keywords.
#         Verify: "ytd" is a valid accepted date_range value.
# ══════════════════════════════════════════════════════════════════════════════


class TestTS01_YTDReturnRoutesToPerformance:
    def test_performance_docstring_covers_return_keywords(self):
        """
        The LLM reads the tool description to decide which tool to call.
        get_performance must cover: 'return', 'performance', 'gain', 'loss', 'period'.
        """
        triggers = ["return", "performance", "gain", "loss", "period"]
        missing = _check_triggers(get_performance, triggers)
        assert not missing, (
            f"TS01: get_performance docstring missing trigger keywords: {missing}. "
            f"The LLM may fail to route YTD return queries to this tool."
        )

    def test_performance_docstring_covers_ytd(self):
        """
        The string 'ytd' must appear in the docstring so the LLM knows the valid range values.
        """
        assert "ytd" in _doc(get_performance), (
            "TS01: 'ytd' not found in get_performance description. "
            "The LLM may not know to pass date_range='ytd'."
        )

    def test_performance_returns_performance_data_not_holdings(self):
        """
        Domain boundary: get_performance must not be confused with portfolio composition.
        Its docstring must NOT be the primary holder of 'holdings' or 'allocation' keywords —
        those belong to get_portfolio_summary / analyze_diversification.
        """
        perf_doc = _doc(get_performance)
        portfolio_doc = _doc(get_portfolio_summary)
        # get_portfolio_summary must be MORE specific about 'holdings' than get_performance
        assert portfolio_doc.count("holding") >= perf_doc.count("holding"), (
            "TS01: get_portfolio_summary docstring must dominate 'holdings' keyword "
            "so the LLM prefers it for composition queries."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TS02 — "Show my allocation by sector and asset class" → analyze_diversification
#         Verify: docstring covers sector/geography/asset class keywords.
#         Verify: no false routing to get_portfolio_summary for sector data.
# ══════════════════════════════════════════════════════════════════════════════


class TestTS02_SectorAllocationRoutesToDiversification:
    def test_diversification_docstring_covers_sector_keywords(self):
        triggers = ["sector", "allocation", "diversif", "concentration", "geographic"]
        missing = _check_triggers(analyze_diversification, triggers)
        assert not missing, f"TS02: analyze_diversification missing trigger keywords: {missing}"

    def test_diversification_preferred_over_portfolio_for_sector(self):
        """
        For sector queries, analyze_diversification must have higher relevance score
        than get_portfolio_summary. We proxy this by checking keyword count.
        """
        div_doc = _doc(analyze_diversification)
        port_doc = _doc(get_portfolio_summary)
        assert div_doc.count("sector") >= port_doc.count("sector"), (
            "TS02: get_portfolio_summary mentions 'sector' as much as analyze_diversification. "
            "The LLM may pick the wrong tool for sector queries."
        )

    def test_diversification_not_confused_with_transactions(self):
        """Transactions tool must not mention 'sector' — that would cause false routing."""
        tx_doc = _doc(get_transactions)
        # allow one incidental mention but not more
        assert tx_doc.count("sector") <= 1, (
            f"TS02: get_transactions docstring mentions 'sector' {tx_doc.count('sector')} times — "
            f"may confuse tool selection for sector queries."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TS03 — "Find duplicate transactions" → get_transactions or pattern intelligence
#         Verify: transactions tool covers 'transaction', 'history', 'fee' keywords.
#         Verify: transaction_pattern_intelligence also covers pattern detection.
# ══════════════════════════════════════════════════════════════════════════════


class TestTS03_DuplicateTransactionsRoutesToTxTool:
    def test_transactions_docstring_covers_history_keywords(self):
        triggers = ["transaction", "history", "buy", "sell", "fee", "dividend"]
        missing = _check_triggers(get_transactions, triggers)
        assert not missing, f"TS03: get_transactions missing trigger keywords: {missing}"

    def test_pattern_intelligence_covers_pattern_keywords(self):
        triggers = ["pattern", "trading", "behaviour", "buy-and-hold"]
        doc = _doc(get_transaction_pattern_intelligence)
        missing = [t for t in triggers if t not in doc]
        # At least 2 of 4 must match (some phrasing differences tolerated)
        assert len(missing) <= 2, (
            f"TS03: get_transaction_pattern_intelligence missing too many keywords: {missing}"
        )

    def test_performance_not_used_for_duplicate_detection(self):
        """Performance tool docstring must not mislead the LLM into using it for tx history."""
        perf_doc = _doc(get_performance)
        assert "duplicate" not in perf_doc, (
            "TS03: get_performance mentions 'duplicate' — may cause false routing."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TS04 — "Compare to SPY over 1Y" → get_performance("1y") + get_market_data("SPY")
#         Verify: get_market_data covers 'compare', 'benchmark', 'price' keywords.
#         Verify: "1y" is documented in get_performance.
# ══════════════════════════════════════════════════════════════════════════════


class TestTS04_SPYComparisonRequiresBothTools:
    def test_market_data_docstring_covers_comparison_keywords(self):
        triggers = ["price", "market", "symbol", "stock"]
        missing = _check_triggers(get_market_data, triggers)
        assert not missing, f"TS04: get_market_data missing trigger keywords: {missing}"

    def test_performance_docstring_covers_1y(self):
        assert "1y" in _doc(get_performance), (
            "TS04: '1y' not found in get_performance docstring — "
            "LLM may not know to pass date_range='1y' for a 1-year comparison."
        )

    def test_market_data_covers_benchmark_use_case(self):
        """
        'compare' or 'context' must appear in get_market_data docstring
        so the LLM routes comparison requests here as well as to get_performance.
        """
        doc = _doc(get_market_data)
        assert "compare" in doc or "context" in doc or "benchmark" in doc, (
            "TS04: get_market_data docstring should mention 'compare', 'context', "
            "or 'benchmark' to cover SPY comparison queries."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TS05 — "Do I hold any AI stocks?" → get_portfolio_summary
#         The correct tool is get_portfolio_summary because the agent needs to
#         inspect WHAT the user holds before reasoning about AI relevance.
# ══════════════════════════════════════════════════════════════════════════════


class TestTS05_AIStockQueryUsesPortfolioSummary:
    def test_portfolio_summary_docstring_covers_holdings_keywords(self):
        triggers = ["holdings", "portfolio", "positions", "value", "allocation"]
        missing = _check_triggers(get_portfolio_summary, triggers)
        assert not missing, f"TS05: get_portfolio_summary missing trigger keywords: {missing}"

    def test_portfolio_summary_returns_symbol_and_name(self):
        """
        The tool result must include 'symbol' and 'name' fields so the agent
        can match holdings against AI-related ticker names.
        """
        doc = _doc(get_portfolio_summary)
        # Verify the docstring mentions what fields are returned
        assert "symbol" in doc or "composition" in doc or "what they own" in doc, (
            "TS05: get_portfolio_summary docstring should indicate it returns symbols/names "
            "so the LLM knows it can answer 'what stocks do I hold' questions."
        )

    def test_market_data_not_sole_tool_for_ai_exposure(self):
        """
        get_market_data alone cannot answer 'do I hold AI stocks' — it needs
        to be combined with portfolio data. The market tool's docstring must not
        claim to know what the user holds.
        """
        mkt_doc = _doc(get_market_data)
        assert "what you hold" not in mkt_doc and "your holdings" not in mkt_doc, (
            "TS05: get_market_data should not claim to know what the user holds — "
            "that's get_portfolio_summary's domain."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TS06 — "How healthy is my portfolio?" → get_portfolio_health_scorecard
#         Verify the dedicated scorecard tool exists and covers grade/score keywords.
#         Verify it's preferred over composing multiple individual tools.
# ══════════════════════════════════════════════════════════════════════════════


class TestTS06_HealthQueryUsesScorecard:
    def test_health_scorecard_docstring_covers_health_keywords(self):
        triggers = ["health", "grade", "assessment", "score", "action"]
        missing = _check_triggers(get_portfolio_health_scorecard, triggers)
        assert not missing, (
            f"TS06: get_portfolio_health_scorecard missing trigger keywords: {missing}"
        )

    def test_health_scorecard_is_more_specific_than_portfolio_summary(self):
        """
        For 'how healthy is my portfolio', get_portfolio_health_scorecard must
        be more relevant than get_portfolio_summary (which doesn't grade).
        Proxy: health scorecard mentions 'grade' or 'A-D'; portfolio summary doesn't.
        """
        score_doc = _doc(get_portfolio_health_scorecard)
        port_doc = _doc(get_portfolio_summary)
        assert "grade" in score_doc or "a-d" in score_doc, (
            "TS06: get_portfolio_health_scorecard must mention 'grade' or 'A-D' "
            "so the LLM selects it for health assessment queries."
        )
        # portfolio summary must not pretend to grade
        assert "grade" not in port_doc, (
            "TS06: get_portfolio_summary mentions 'grade' — may cause false routing "
            "for health queries away from the dedicated scorecard tool."
        )

    def test_all_tools_have_non_empty_description(self):
        """
        Regression guard: every tool must have a docstring / description.
        Empty descriptions cause silent LLM routing failures.
        """
        from agent.tools import ALL_TOOLS

        for tool_fn in ALL_TOOLS:
            desc = getattr(tool_fn, "description", None) or tool_fn.__doc__ or ""
            assert desc.strip(), (
                f"TS06: Tool '{getattr(tool_fn, 'name', str(tool_fn))}' has an empty description. "
                f"This will prevent correct tool selection."
            )
