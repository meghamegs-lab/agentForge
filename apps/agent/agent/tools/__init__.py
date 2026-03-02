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

# ── FIRE Goal Tracker (optional add-on) ───────────────────────────────────────
# Only registered when FIRE_TRACKER_ENABLED=true in .env.
# When disabled: zero tools added, zero DB connections opened, agent is unchanged.
from agent.config import settings  # noqa: E402 (import after list construction is intentional)

if settings.fire_tracker_enabled:
    from agent.tools.retirement import (
        calculate_retirement_projection,
        get_fire_progress,
        get_macro_data,
        get_retirement_goal,
        set_retirement_goal,
    )

    ALL_TOOLS.extend(
        [
            get_retirement_goal,
            set_retirement_goal,
            get_fire_progress,
            calculate_retirement_projection,
            get_macro_data,
        ]
    )

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
