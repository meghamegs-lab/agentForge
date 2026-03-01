# MCP server that exposes all 11 Ghostfolio tools via stdio so Claude Desktop and Cursor can use them.
"""
Fortio MCP Server
=================
Exposes all 11 Ghostfolio portfolio tools as an MCP (Model Context Protocol)
server so Claude Desktop, Cursor, and any other MCP-compatible AI host can
query the user's Ghostfolio portfolio using natural language.

Transport: stdio (stdin/stdout) — standard for local MCP servers.
  Runs as a subprocess of the MCP host. Secure: no open network port.

Usage:
    fortio mcp                 # via CLI (after pip install -e .)
    python -m agent.mcp.server # directly

Claude Desktop config (macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
                       Windows: %APPDATA%/Claude/claude_desktop_config.json):

    {
      "mcpServers": {
        "fortio": {
          "command": "fortio",
          "args": ["mcp"],
          "env": {
            "GHOSTFOLIO_BASE_URL": "https://your-ghostfolio.railway.app",
            "GHOSTFOLIO_ACCESS_TOKEN": "your-token"
          }
        }
      }
    }

Cursor: Settings → MCP → Add server → paste the same block.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import structlog
from langsmith import traceable
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# ── LangSmith tracing ──────────────────────────────────────────────────────────
# When launched directly (python -m agent.mcp.server or as a subprocess by an
# MCP host like Claude Desktop), cli.py is NOT imported, so the env vars are not
# set yet.  Push them here from pydantic-settings so LangSmith tracing works.
from agent.config import settings  # noqa: E402 — must come before langsmith reads os.environ

os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)
os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

# ── Import the underlying implementation functions directly ────────────────────
# We use the private _get_* functions, NOT the @tool-decorated LangChain
# versions.  This avoids LangGraph overhead and the "StructuredTool does not
# support sync invocation" issue that surfaces in tests.
from agent.clients.market import get_shared_market_client
from agent.tools.diversification import _analyze_diversification
from agent.tools.fee_drag import _fee_drag
from agent.tools.health_scorecard import _scorecard
from agent.tools.market_context import _market_context
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.proactive_monitor import _proactive_monitor
from agent.tools.rebalancing import _rebalancing_plan
from agent.tools.transaction_patterns import _transaction_patterns
from agent.tools.transactions import _get_transactions

# ── Redirect all logging to stderr ────────────────────────────────────────────
# stdout is reserved for the MCP JSON-RPC wire protocol.  Any bytes written to
# stdout that are not valid JSON-RPC messages will corrupt the framing and cause
# "Unexpected token" errors in the MCP host (Claude Desktop, Cursor, etc.).
# Must run after imports so all structlog loggers inherit this factory.
structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

log = structlog.get_logger()

# ── Server singleton ───────────────────────────────────────────────────────────

server = Server("fortio")
_market_client = get_shared_market_client()


# ── Helpers ───────────────────────────────────────────────────────────────────


# Serialises a tool result dict to a single MCP TextContent block for the JSON-RPC response.
def _ok(data: dict[str, Any]) -> list[types.TextContent]:
    """Serialize a tool result dict to a single MCP TextContent block."""
    return [types.TextContent(type="text", text=json.dumps(data, indent=2))]


# Thin async wrapper around MarketDataClient that mirrors the market.py tool without the @tool decorator.
async def _get_market_data_impl(
    symbols: str,
    metrics: str = "price,52w_range,market_cap",
) -> dict[str, Any]:
    """
    Thin async wrapper around MarketDataClient.
    Mirrors the logic in agent/tools/market.py without the @tool decorator.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"status": "error", "error": "No valid symbols provided"}
    if len(symbol_list) == 1:
        return await _market_client.get_quote(symbol_list[0])
    return await _market_client.get_batch_quotes(symbol_list)


# ── Tool registry ──────────────────────────────────────────────────────────────

_TOOLS: list[types.Tool] = [
    types.Tool(
        name="get_portfolio_summary",
        description=(
            "Retrieve the user's current Ghostfolio portfolio: all holdings, "
            "allocation percentages, current values, and total portfolio value. "
            "Use when asked about portfolio composition, what they own, or total value."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Optional account ID to filter. Leave empty for all accounts.",
                    "default": "",
                },
            },
        },
    ),
    types.Tool(
        name="get_performance",
        description=(
            "Get portfolio performance metrics: return percentage, absolute gain/loss, "
            "and current value for a given time period. Use when asked about returns, "
            "gains, losses, or how the portfolio has performed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "date_range": {
                    "type": "string",
                    "enum": ["1d", "wtd", "mtd", "ytd", "1y", "5y", "max"],
                    "description": "Time period for performance metrics.",
                    "default": "ytd",
                },
            },
        },
    ),
    types.Tool(
        name="get_transactions",
        description=(
            "Retrieve transaction history: buys, sells, dividends, and fees paid. "
            "Use when asked about trading history, past transactions, or fees."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Optional account filter. Leave empty for all accounts.",
                    "default": "",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date in ISO format e.g. '2024-01-01'. Optional.",
                    "default": "",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date in ISO format e.g. '2024-12-31'. Optional.",
                    "default": "",
                },
                "transaction_type": {
                    "type": "string",
                    "enum": ["BUY", "SELL", "DIVIDEND", "FEE", "INTEREST", ""],
                    "description": "Filter by transaction type. Leave empty for all types.",
                    "default": "",
                },
            },
        },
    ),
    types.Tool(
        name="analyze_diversification",
        description=(
            "Analyse portfolio diversification: sector and geographic breakdown, "
            "concentration risk score, and Herfindahl index. Use when asked about "
            "diversification, concentration, sector exposure, or geographic allocation."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_market_data",
        description=(
            "Get current market data for one or more stock/ETF symbols: live price, "
            "52-week range, market cap, and volume. Always call this for any price "
            "question — never answer from training knowledge."
        ),
        inputSchema={
            "type": "object",
            "required": ["symbols"],
            "properties": {
                "symbols": {
                    "type": "string",
                    "description": "Comma-separated ticker symbols e.g. 'AAPL,MSFT,VTI'.",
                },
                "metrics": {
                    "type": "string",
                    "description": "Comma-separated metrics: price, 52w_range, market_cap, volume.",
                    "default": "price,52w_range,market_cap",
                },
            },
        },
    ),
    types.Tool(
        name="get_fee_drag_analysis",
        description=(
            "Analyse how fees are dragging portfolio performance vs a zero-fee benchmark. "
            "Shows total fees paid, fee drag percentage, and projected long-term cost."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "date_range": {
                    "type": "string",
                    "enum": ["ytd", "1y", "5y", "max"],
                    "description": (
                        "Time period for fee analysis. Defaults to '1y'. "
                        "Use 'max' only when the user explicitly asks for lifetime/all-time fees — "
                        "it is slow on large portfolios."
                    ),
                    "default": "1y",
                },
            },
        },
    ),
    types.Tool(
        name="get_portfolio_health_scorecard",
        description=(
            "Generate an overall portfolio health score (0–100) with letter grade and "
            "actionable insights across diversification, performance, fees, and risk."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_rebalancing_plan",
        description=(
            "Generate a rebalancing plan with specific dollar amounts to buy/sell for "
            "each position to reach target allocations. Default targets: 55% US equity, "
            "25% international, 15% bonds, 5% cash."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target_us_equity_pct": {
                    "type": "number",
                    "description": "Target percentage for US equity holdings.",
                    "default": 55.0,
                },
                "target_intl_equity_pct": {
                    "type": "number",
                    "description": "Target percentage for international equity.",
                    "default": 25.0,
                },
                "target_bonds_pct": {
                    "type": "number",
                    "description": "Target percentage for bonds/fixed income.",
                    "default": 15.0,
                },
                "target_cash_pct": {
                    "type": "number",
                    "description": "Target percentage for cash/money market.",
                    "default": 5.0,
                },
            },
        },
    ),
    types.Tool(
        name="get_market_context_overlay",
        description=(
            "Map the user's specific holdings to a macro economic theme. Shows how each "
            "sector in the portfolio is positioned for the chosen regime (rising rates, "
            "recession, inflation, or bull market)."
        ),
        inputSchema={
            "type": "object",
            "required": ["macro_theme"],
            "properties": {
                "macro_theme": {
                    "type": "string",
                    "enum": ["rising_rates", "recession", "inflation", "bull_market"],
                    "description": "The macro economic theme to analyse the portfolio against.",
                },
            },
        },
    ),
    types.Tool(
        name="get_transaction_pattern_intelligence",
        description=(
            "Analyse behavioural patterns in the user's trading history: timing bias, "
            "holding periods, buy/sell frequency, and emotional trading signals."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_proactive_risk_monitor",
        description=(
            "Run a proactive risk check across the whole portfolio: concentration risk, "
            "volatility exposure, liquidity risk, and correlation risk. Optionally compare "
            "against a previous snapshot to surface changes since the last session."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "previous_snapshot_json": {
                    "type": "string",
                    "description": "JSON string of a prior risk snapshot for change detection. Leave empty for a fresh check.",
                    "default": "",
                },
            },
        },
    ),
]


# ── MCP protocol handlers ──────────────────────────────────────────────────────


# Advertises all 11 Fortio tools to the MCP host on the list_tools request.
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return the full list of Fortio tools to the MCP host."""
    return _TOOLS


# Dispatches a call_tool request from the MCP host to the matching private implementation function.
@server.call_tool()
@traceable(name="mcp_tool_call", run_type="tool")
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Dispatch a tool call from the MCP host to the appropriate implementation."""
    log.info("mcp_tool_call", tool=name)

    try:
        match name:
            case "get_portfolio_summary":
                result = await _get_portfolio_summary(
                    account_id=arguments.get("account_id", ""),
                )
            case "get_performance":
                result = await _get_performance(
                    date_range=arguments.get("date_range", "ytd"),
                )
            case "get_transactions":
                result = await _get_transactions(
                    account_id=arguments.get("account_id", ""),
                    date_from=arguments.get("date_from", ""),
                    date_to=arguments.get("date_to", ""),
                    transaction_type=arguments.get("transaction_type", ""),
                )
            case "analyze_diversification":
                result = await _analyze_diversification()
            case "get_market_data":
                result = await _get_market_data_impl(
                    symbols=arguments.get("symbols", ""),
                    metrics=arguments.get("metrics", "price,52w_range,market_cap"),
                )
            case "get_fee_drag_analysis":
                result = await _fee_drag(
                    date_range=arguments.get("date_range", "1y"),
                )
            case "get_portfolio_health_scorecard":
                result = await _scorecard()
            case "get_rebalancing_plan":
                result = await _rebalancing_plan(
                    tgt_us=float(arguments.get("target_us_equity_pct", 55.0)),
                    tgt_intl=float(arguments.get("target_intl_equity_pct", 25.0)),
                    tgt_bonds=float(arguments.get("target_bonds_pct", 15.0)),
                    tgt_cash=float(arguments.get("target_cash_pct", 5.0)),
                )
            case "get_market_context_overlay":
                result = await _market_context(
                    macro_theme=arguments.get("macro_theme", "rising_rates"),
                )
            case "get_transaction_pattern_intelligence":
                result = await _transaction_patterns()
            case "get_proactive_risk_monitor":
                result = await _proactive_monitor(
                    prev_snap_json=arguments.get("previous_snapshot_json", ""),
                )
            case _:
                result = {"status": "error", "error": f"Unknown tool: {name}"}

    except Exception as exc:
        log.error("mcp_tool_error", tool=name, error=str(exc))
        result = {"status": "error", "error": str(exc)}

    return _ok(result)


# ── Resources ──────────────────────────────────────────────────────────────────
# Resources let the MCP host passively load portfolio data as context —
# useful when Claude wants to reason about the portfolio without an explicit
# tool call, or to "attach" data before a conversation starts.


# Advertises the 3 passive portfolio resources (summary, performance, health) to the MCP host.
@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Advertise the portfolio data resources available to the MCP host."""
    return [
        types.Resource(
            uri="portfolio://summary",
            name="Portfolio Summary",
            description="Current holdings, allocations, and total portfolio value from Ghostfolio.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="portfolio://performance",
            name="Portfolio Performance (YTD)",
            description="Year-to-date performance metrics: return %, absolute gain/loss, current value.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="portfolio://health",
            name="Portfolio Health Scorecard",
            description="Overall portfolio health score (0–100) with letter grade and actionable insights.",
            mimeType="application/json",
        ),
    ]


# Fetches live portfolio data for the requested URI and returns it as a JSON string.
@server.read_resource()
async def read_resource(uri: str) -> str:
    """Fetch a portfolio resource by URI for passive context loading."""
    log.info("mcp_resource_read", uri=uri)

    match uri:
        case "portfolio://summary":
            data = await _get_portfolio_summary()
        case "portfolio://performance":
            data = await _get_performance(date_range="ytd")
        case "portfolio://health":
            data = await _scorecard()
        case _:
            data = {"status": "error", "error": f"Unknown resource URI: {uri}"}

    return json.dumps(data, indent=2)


# ── Prompts ────────────────────────────────────────────────────────────────────
# Prompt templates let MCP hosts offer slash-commands or pre-built context
# to their users.  Claude Desktop shows these in the prompt library.


# Advertises the portfolio-analysis prompt template to the MCP host (shown in Claude's prompt library).
@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """Advertise reusable Fortio prompt templates."""
    return [
        types.Prompt(
            name="portfolio-analysis",
            description=(
                "Load your full portfolio context (holdings + YTD performance + health score) "
                "and ask Fortio for a comprehensive analysis."
            ),
            arguments=[
                types.PromptArgument(
                    name="focus",
                    description="Focus area: 'risk', 'performance', 'fees', or 'all' (default).",
                    required=False,
                ),
            ],
        ),
    ]


# Fetches live portfolio data and embeds it into the portfolio-analysis prompt template.
@server.get_prompt()
async def get_prompt(
    name: str,
    arguments: dict[str, str] | None,
) -> types.GetPromptResult:
    """Return a rendered prompt template with live portfolio data embedded."""
    if name != "portfolio-analysis":
        raise ValueError(f"Unknown prompt: {name}")

    focus = (arguments or {}).get("focus", "all")

    # Fetch live data to embed as context in the prompt
    summary = await _get_portfolio_summary()
    performance = await _get_performance("ytd")
    health = await _scorecard()

    focus_instruction = {
        "risk": "Focus on concentration risk, sector exposure, and rebalancing needs.",
        "performance": "Focus on returns, period comparisons, and best/worst performers.",
        "fees": "Focus on fee drag, total fees paid, and cost reduction opportunities.",
        "all": "Provide a comprehensive analysis covering diversification, performance, fees, and risk.",
    }.get(
        focus,
        "Provide a comprehensive analysis covering diversification, performance, fees, and risk.",
    )

    prompt_text = (
        "You are Fortio, a personal finance assistant connected to Ghostfolio.\n\n"
        "## Current Portfolio Data\n\n"
        f"### Holdings\n```json\n{json.dumps(summary, indent=2)}\n```\n\n"
        f"### YTD Performance\n```json\n{json.dumps(performance, indent=2)}\n```\n\n"
        f"### Health Scorecard\n```json\n{json.dumps(health, indent=2)}\n```\n\n"
        f"## Your Task\n{focus_instruction}\n\n"
        "Base your analysis ONLY on the data above. "
        "Do not state specific numbers not present in the data provided."
    )

    return types.GetPromptResult(
        description=f"Fortio portfolio analysis — focus: {focus}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text),
            ),
        ],
    )


# ── Entry point ────────────────────────────────────────────────────────────────


# Starts the MCP server event loop on stdio — blocks until the MCP host disconnects.
async def serve() -> None:
    """Start the Fortio MCP server on stdio transport."""
    log.info("fortio_mcp_server_starting", transport="stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(serve())
