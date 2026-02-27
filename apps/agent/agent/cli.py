# Typer-based CLI for Fortio: ask, chat (REPL), serve (uvicorn), mcp (stdio), and version commands.
"""
Fortio CLI — command-line interface for the Fortio Finance Agent.
Uses Typer for commands and Rich for pretty terminal output.

Commands:
  fortio ask "your question"   - single question, print answer, exit
  fortio chat                  - interactive multi-turn REPL
  fortio serve                 - start the FastAPI / uvicorn server
  fortio version               - show version and config info

Run directly (without installing):
  python -m agent.cli ask "What's my portfolio?"

After installing the package (pip install -e .):
  fortio ask "What's my portfolio?"
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Optional

import structlog
import typer
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agent.graph.graph import build_graph

app = typer.Typer(
    name="fortio",
    help="Fortio – your Ghostfolio Finance AI Assistant 💼",
    add_completion=True,
    pretty_exceptions_show_locals=False,  # keep tracebacks clean in prod
)
console = Console()
log = structlog.get_logger()

# ── Shared graph singleton (MemorySaver: history lives for the process) ────────
_graph = None


# Returns the module-level agent graph singleton, creating it once with MemorySaver for the process lifetime.
def _get_graph():
    """Return the module-level agent graph singleton, building it once."""
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=MemorySaver())
    return _graph


# ── Suggestion data ────────────────────────────────────────────────────────────

# Welcome screen: topic categories with short example prompts (≤38 chars each)
_SUGGESTION_CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "📊 Portfolio",
        [
            "What does my portfolio look like?",
            "Give me a portfolio health scorecard",
            "What's my total portfolio value?",
        ],
    ),
    (
        "📈 Performance",
        [
            "How has my portfolio performed YTD?",
            "Which holdings performed best?",
            "Compare my returns to the market",
        ],
    ),
    (
        "⚖️ Risk & Diversification",
        [
            "Am I too concentrated in any sector?",
            "Generate a rebalancing plan for me",
            "Run a proactive risk check",
        ],
    ),
    (
        "💸 Fees & Trades",
        [
            "What are my biggest fees this year?",
            "Analyze my fee drag on performance",
            "What trading patterns do you see?",
        ],
    ),
    (
        "📡 Market Data",
        [
            "What's the current price of AAPL?",
            "Prices for my top 5 holdings",
            "How does the market affect my portfolio?",
        ],
    ),
]

# Per-tool follow-up suggestions shown after each agent response
_TOOL_FOLLOWUPS: dict[str, list[str]] = {
    "get_portfolio_summary": [
        "How has my portfolio performed this year?",
        "Am I too concentrated in any sector?",
        "Give me a full portfolio health scorecard",
    ],
    "get_performance": [
        "What fees are dragging my performance?",
        "How does my performance compare to the market?",
        "Which holdings are performing worst?",
    ],
    "get_transactions": [
        "What trading patterns do you see in my history?",
        "How much did I pay in total transaction fees?",
        "Analyze the fee drag on my portfolio",
    ],
    "analyze_diversification": [
        "Should I rebalance my portfolio?",
        "Run a proactive risk check on my portfolio",
        "How has my portfolio performed overall?",
    ],
    "get_market_data": [
        "How does this position fit into my overall portfolio?",
        "How does today's market context affect my holdings?",
        "What's my portfolio's total value right now?",
    ],
    "get_fee_drag_analysis": [
        "How has my performance been despite these fees?",
        "What's my overall portfolio health score?",
        "Should I rebalance to lower-cost holdings?",
    ],
    "get_portfolio_health_scorecard": [
        "What specific risks should I address first?",
        "Generate a rebalancing plan based on this score",
        "How concentrated am I in the tech sector?",
    ],
    "get_rebalancing_plan": [
        "What are the fee implications of rebalancing?",
        "How has my portfolio performed recently?",
        "What's my current diversification score?",
    ],
    "get_market_context_overlay": [
        "Should I adjust my portfolio for the current market?",
        "What's my portfolio's overall risk level?",
        "Are there sectors I should reduce exposure to?",
    ],
    "get_transaction_pattern_intelligence": [
        "How much am I paying in trading fees overall?",
        "What are my best and worst performing holdings?",
        "Am I trading too frequently for optimal returns?",
    ],
    "get_proactive_risk_monitor": [
        "Should I rebalance based on these risks?",
        "What's my current sector concentration?",
        "How has performance held up given these risks?",
    ],
}

# Default follow-ups when no specific tool was matched
_DEFAULT_FOLLOWUPS: list[str] = [
    "What does my portfolio look like?",
    "How has my portfolio performed this year?",
    "Am I too concentrated in any sector?",
]


# ── Core async invoke helper ───────────────────────────────────────────────────

# Sends one message through the agent graph and returns the final AgentState dict.
async def _invoke(message: str, conversation_id: str, user_id: str) -> dict:
    """Send one message through the agent graph and return the final state."""
    state = {
        "messages": [HumanMessage(content=message)],
        "tool_results": [],
        "verification_flags": [],
        "confidence": "HIGH",
        "reasoning_steps": 0,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "final_response": "",
        "should_escalate": False,
        # reasoning_node increments turn_number and populates context_entities;
        # MemorySaver persists both so prior-turn values are restored next call.
        "turn_number": 0,
        "context_entities": {},
    }
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": user_id,
        }
    }
    return await _get_graph().ainvoke(state, config=config)


# ── Rich rendering helpers ─────────────────────────────────────────────────────

_CONFIDENCE_COLOR: dict[str, str] = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}
_CONFIDENCE_EMOJI: dict[str, str] = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}


# Renders the agent's answer in a Rich panel with a colour-coded confidence badge and optional tool list.
def _render_response(final_state: dict, verbose: bool = False) -> None:
    """Render the agent's answer in a Rich panel with confidence badge."""
    answer = final_state.get("final_response", "")
    if not answer:
        last_msg = final_state["messages"][-1]
        content = getattr(last_msg, "content", "")
        answer = content if isinstance(content, str) else str(content)

    confidence: str = final_state.get("confidence", "MEDIUM")
    color = _CONFIDENCE_COLOR.get(confidence, "yellow")
    emoji = _CONFIDENCE_EMOJI.get(confidence, "🟡")

    flags: list[dict] = final_state.get("verification_flags", [])
    high_flags = [f for f in flags if f.get("severity") == "HIGH"]

    subtitle = f"{emoji} Confidence: [bold {color}]{confidence}[/bold {color}]"
    if high_flags:
        subtitle += f" | [bold red]⚠️ {len(high_flags)} high-severity flag(s)[/bold red]"

    console.print(
        Panel(
            Markdown(answer),
            title="[bold cyan]Fortio[/bold cyan]",
            subtitle=subtitle,
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if verbose:
        tool_msgs = [m for m in final_state.get("messages", []) if isinstance(m, ToolMessage)]
        if tool_msgs:
            console.print("[dim]Tools used this turn:[/dim]")
            for msg in tool_msgs:
                console.print(f"  [dim]• {msg.name}[/dim]")


# Shows 3 contextual follow-up suggestions keyed to the tools used; falls back to generic prompts.
def _render_followups(final_state: dict) -> None:
    """
    Show 3 contextual follow-up suggestions based on which tools were called.
    Falls back to generic suggestions when no tool was used.
    """
    tool_msgs = [m for m in final_state.get("messages", []) if isinstance(m, ToolMessage)]
    suggestions: list[str] = []

    for msg in tool_msgs:
        hits = _TOOL_FOLLOWUPS.get(msg.name or "", [])
        for s in hits:
            if s not in suggestions:
                suggestions.append(s)
        if len(suggestions) >= 3:
            break

    if not suggestions:
        suggestions = _DEFAULT_FOLLOWUPS

    suggestions = suggestions[:3]

    text = Text()
    for i, s in enumerate(suggestions):
        text.append(f"  {i + 1}. ", style="bold cyan")
        text.append(s, style="italic")
        if i < len(suggestions) - 1:
            text.append("\n")

    console.print(
        Panel(
            text,
            title="[dim]💡 Follow-up suggestions[/dim]",
            border_style="dim",
            padding=(0, 2),
        )
    )


# Renders the full topic → example-question grid as a two-column Rich panel on the welcome screen.
def _render_suggestions_table() -> None:
    """Render the full topic → example-questions suggestion grid."""
    # Two visible columns (cat header + bullets), no wrapping
    table = Table.grid(expand=False, padding=(0, 3))
    table.add_column(no_wrap=True)  # left block
    table.add_column(no_wrap=True)  # right block

    def _block(cat: str, qs: list[str]) -> str:
        """Build Rich markup for one category block."""
        lines = [f"[bold cyan]{cat}[/bold cyan]"]
        for q in qs:
            lines.append(f"[dim]  • {q}[/dim]")
        return "\n".join(lines)

    for i in range(0, len(_SUGGESTION_CATEGORIES), 2):
        left = _block(*_SUGGESTION_CATEGORIES[i])
        right = (
            _block(*_SUGGESTION_CATEGORIES[i + 1])
            if i + 1 < len(_SUGGESTION_CATEGORIES)
            else ""
        )
        table.add_row(left, right)
        table.add_row("", "")  # spacer row

    console.print(
        Panel(
            table,
            title="[bold cyan]💡 What can I help you with?[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


# Prints a Rich table listing every agent tool with a truncated one-line description (triggered by /tools).
def _render_tools_list() -> None:
    """Show all available agent tools with short one-line descriptions."""
    from agent.tools import ALL_TOOLS

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("What it does", no_wrap=True, style="dim")

    for tool in ALL_TOOLS:
        # Grab the first sentence and truncate to 55 chars max
        first_sentence = (tool.description or "").split(".")[0].strip()
        # Also split on newlines to avoid multi-line docstring preamble
        first_line = first_sentence.splitlines()[0].strip()
        label = first_line if len(first_line) <= 55 else first_line[:52] + "…"
        table.add_row(tool.name, label)

    console.print(
        Panel(
            table,
            title="[bold cyan]🔧 Available Tools[/bold cyan]",
            border_style="cyan",
            padding=(1, 1),
        )
    )


# ── Commands ───────────────────────────────────────────────────────────────────

# Sends a single question to the agent, prints the answer, and exits — no persistent REPL.
@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask Fortio"),
    user_id: str = typer.Option("cli_user", "--user-id", "-u", help="User ID for request context"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool call details"),
    suggest: bool = typer.Option(True, "--suggest/--no-suggest", help="Show follow-up suggestions"),
) -> None:
    """Ask Fortio a single question and print the answer."""
    conversation_id = str(uuid.uuid4())
    log.info("cli_ask", question_preview=question[:80], user_id=user_id)

    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
        try:
            final_state = asyncio.run(_invoke(question, conversation_id, user_id))
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    _render_response(final_state, verbose=verbose)
    if suggest:
        _render_followups(final_state)


# Launches an interactive multi-turn REPL; each turn appends to the same conversation_id.
@app.command()
def chat(
    user_id: str = typer.Option("cli_user", "--user-id", "-u", help="User ID for request context"),
    conversation_id: Optional[str] = typer.Option(
        None,
        "--conversation-id",
        "-c",
        help="Resume a prior conversation by its ID (same process only)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool call details per turn"),
    suggest: bool = typer.Option(True, "--suggest/--no-suggest", help="Show follow-up suggestions after each response"),
) -> None:
    """Start an interactive multi-turn chat REPL with Fortio."""
    cid = conversation_id or str(uuid.uuid4())

    # ── Welcome banner ────────────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold]Welcome to Fortio, your Ghostfolio Finance Assistant![/bold]\n\n"
            "[dim]Type your question below, or use a special command:[/dim]\n"
            "  [bold cyan]/help[/bold cyan]   show topic suggestions\n"
            "  [bold cyan]/tools[/bold cyan]  list all available agent tools\n"
            "  [bold cyan]/clear[/bold cyan]  print a visual separator\n"
            "  [bold cyan]exit[/bold cyan]    end the session\n\n"
            "[dim]⚠️  Fortio provides information for educational purposes only — not financial advice.[/dim]",
            title="[bold cyan]Fortio Chat[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print(f"[dim]Session: {cid}[/dim]\n")

    # Show the suggestion grid on startup
    _render_suggestions_table()

    # Inner coroutine that runs the entire REPL session in one event loop until the user types "exit".
    async def _repl() -> None:
        """Inner async REPL — runs the entire session in one event loop."""
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break

            if not user_input:
                continue

            # ── Special REPL commands ─────────────────────────────────────────
            if user_input.lower() in ("exit", "quit", "bye", "q"):
                console.print("[dim]Goodbye! 👋[/dim]")
                break

            if user_input.lower() == "/help":
                _render_suggestions_table()
                continue

            if user_input.lower() == "/tools":
                _render_tools_list()
                continue

            if user_input.lower() == "/clear":
                console.print(Rule(style="dim"))
                continue

            # ── Normal agent turn ─────────────────────────────────────────────
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                try:
                    final_state = await _invoke(user_input, cid, user_id)
                except Exception as exc:
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    continue

            _render_response(final_state, verbose=verbose)
            if suggest:
                _render_followups(final_state)
            console.print()  # blank line between turns

    asyncio.run(_repl())


# Starts the Fortio FastAPI server via uvicorn; supports reload mode for development.
@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8001, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable hot-reload (dev mode only)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of uvicorn worker processes"),
) -> None:
    """Start the Fortio FastAPI server with uvicorn."""
    import uvicorn

    effective_workers = workers if not reload else 1
    console.print(
        f"[bold cyan]Starting Fortio API[/bold cyan] → "
        f"[bold]http://{host}:{port}[/bold]  "
        f"[dim](reload={reload}, workers={effective_workers})[/dim]"
    )
    uvicorn.run(
        "agent.api.main:app",
        host=host,
        port=port,
        reload=reload,
        # uvicorn does not support workers > 1 with reload=True
        workers=effective_workers,
        log_level="info",
    )



# Starts the Fortio MCP server on stdio so Claude Desktop or Cursor can connect as an MCP host.
@app.command()
def mcp() -> None:
    """
    Start the Fortio MCP (Model Context Protocol) server on stdio.

    Allows Claude Desktop, Cursor, and other MCP-compatible AI hosts to query
    your Ghostfolio portfolio using natural language — no browser, no API keys
    in a chat box.

    Claude Desktop config
    (macOS: ~/Library/Application Support/Claude/claude_desktop_config.json):

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
    import asyncio
    from rich.console import Console as StderrConsole
    from agent.mcp.server import serve

    # IMPORTANT: stdout is the MCP JSON-RPC wire — ALL human-readable output
    # MUST go to stderr so it doesn't corrupt the protocol framing.
    err_console = StderrConsole(stderr=True)
    err_console.print(
        Panel(
            "[bold]Fortio MCP Server[/bold]\n\n"
            "[dim]Transport:[/dim] stdio (stdin/stdout)\n"
            "[dim]Exposing:[/dim] 11 Ghostfolio tools + 3 resources + 1 prompt template\n\n"
            "[dim]Waiting for MCP host connection...[/dim]\n"
            "[dim]Add this server to Claude Desktop or Cursor to connect.[/dim]",
            title="[bold cyan]\U0001f50c Fortio MCP[/bold cyan]",
            border_style="cyan",
        )
    )
    asyncio.run(serve())

# Prints the Fortio version string and the key settings active in the current environment.
@app.command()
def version() -> None:
    """Show Fortio version and active configuration."""
    from agent.config import settings

    console.print(
        Panel(
            "[bold]Fortio – Ghostfolio Finance AI Agent[/bold]\n\n"
            f"Version:        [cyan]0.1.0[/cyan]\n"
            f"Primary model:  [cyan]{settings.primary_model}[/cyan]\n"
            f"Fallback model: [cyan]{settings.fallback_model}[/cyan]\n"
            f"Environment:    [cyan]{settings.environment}[/cyan]\n"
            f"Checkpoint:     [cyan]{settings.checkpoint_backend}[/cyan]\n"
            f"Log level:      [cyan]{settings.log_level}[/cyan]",
            title="[bold cyan]Version Info[/bold cyan]",
            border_style="cyan",
        )
    )


# ── Module entry point ─────────────────────────────────────────────────────────
# Allows: python -m agent.cli ask "..."
if __name__ == "__main__":
    app()
