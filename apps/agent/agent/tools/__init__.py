# Aggregates all LangChain @tool functions into the ALL_TOOLS list used by the agent graph.
from agent.tools.diversification import analyze_diversification
from agent.tools.fee_drag import get_fee_drag_analysis
from agent.tools.health_scorecard import get_portfolio_health_scorecard
from agent.tools.market import get_market_data
from agent.tools.market_context import get_market_context_overlay
from agent.tools.performance import get_performance
from agent.tools.portfolio import get_portfolio_summary
from agent.tools.proactive_monitor import get_proactive_risk_monitor
from agent.tools.rebalancing import get_rebalancing_plan
from agent.tools.transaction_patterns import get_transaction_pattern_intelligence
from agent.tools.transactions import get_transactions

ALL_TOOLS = [
    # Core tools (single Ghostfolio API call)
    get_portfolio_summary,
    get_performance,
    get_transactions,
    analyze_diversification,
    get_market_data,
    # Advanced multi-step tools
    get_fee_drag_analysis,
    get_portfolio_health_scorecard,
    get_rebalancing_plan,
    get_market_context_overlay,
    get_transaction_pattern_intelligence,
    get_proactive_risk_monitor,
]

__all__ = [
    "get_portfolio_summary",
    "get_performance",
    "get_transactions",
    "analyze_diversification",
    "get_market_data",
    "get_fee_drag_analysis",
    "get_portfolio_health_scorecard",
    "get_rebalancing_plan",
    "get_market_context_overlay",
    "get_transaction_pattern_intelligence",
    "get_proactive_risk_monitor",
    "ALL_TOOLS",
]
