# Fortio — Agent Architecture Documentation

**Author:** Megha Shanthraj  
**Date:** February 2026  
**Production:** [fortio-agent-production.up.railway.app](https://ghostfolio-production-453e.up.railway.app)  
**Repo:** `apps/agent/` inside the [agentForge](https://github.com/meghamegs-lab/agentForge)

---

## 1. Domain & Use Cases

Finance - GhostFolio

### Why Finance?

Retail investors face a genuine information asymmetry: their portfolio data lives in
siloed dashboards (Ghostfolio, brokerage apps) that answer _what_ but not _so what_.
Asking "should I rebalance?" or "how much am I losing to fees?" requires cross-referencing
holdings, performance history, market prices, and transaction logs simultaneously — work
that takes minutes in a spreadsheet but seconds in natural language.

Fortio solves this by sitting **in front of Ghostfolio's REST API** and translating
user intent (natural language) into structured API calls, synthesis, and safe, verified answers.

### Concrete Problems Solved

| User Question                                    | Tools Invoked                                          | Value                                                           |
| ------------------------------------------------ | ------------------------------------------------------ | --------------------------------------------------------------- |
| "How am I doing this year?"                      | `get_performance(ytd)`                                 | Converts raw fraction → readable % + absolute gain              |
| "Am I too concentrated in tech?"                 | `get_portfolio_summary` → `analyze_diversification`    | Sector roll-up, concentration flag, diversification score       |
| "What did I pay in fees last year?"              | `get_transactions(type=FEE)` → `get_fee_drag_analysis` | Total fee cost, fee-drag % on portfolio, biggest drag positions |
| "Should I sell AAPL and buy Bitcoin?"            | `get_market_data(AAPL,BTC-USD)`                        | Live prices + mandatory disclaimer + hallucination check        |
| "How does my portfolio look given rising rates?" | `get_market_context_overlay`                           | Position-by-position rate-sensitivity, hedge suggestions        |
| "Generate a rebalancing plan"                    | `get_portfolio_summary` → `get_rebalancing_plan`       | Target-weight trades, drift %, estimated transaction count      |

---

## 2. Agent Architecture Framework

### Framework Choice: LangGraph `StateGraph`

Fortio uses **LangGraph** (not `AgentExecutor` or bare LCEL chains) because portfolio
analysis is inherently **multi-step and stateful**:

- A single question may require 2–4 sequential tool calls (e.g. fetch holdings → compute
  diversification → overlay market context → synthesise)
- Conversation history must persist across turns so "what about MSFT?" resolves correctly
  without the user repeating context
- Verification must run _after_ all tools complete, not mid-chain

### Graph Topology

```
  HumanMessage
       │
  ┌────▼────────┐
  │  reasoning  │  ← Claude Haiku (temp=0) with 11 tools bound
  └────┬────────┘
       │ tool_calls?
  ┌────▼────────┐        ┌────────────────────┐
  │    tools    │───────►│  collect_results   │
  └─────────────┘        └────────┬───────────┘
       │ (loop)                   │ (back to reasoning)
  ┌────▼────────┐
  │   verify    │  ← 5-stage verification pipeline
  └────┬────────┘
       │
  ┌────▼────────┐   HIGH severity?   ┌────────────┐
  │    END      │◄───────────────────┤  escalate  │
  └─────────────┘                    └────────────┘
```

**Key design decisions:**

- **Reasoning loop:** `collect_results` feeds back into `reasoning` so Claude can
  synthesise across multiple tool outputs before writing the final answer.
- **Context injection:** On turn ≥ 2, a dynamic "Active Conversation Context" block
  (tickers, sectors, periods extracted from `ToolMessage` history) is appended to the
  system prompt. This lets "what about those?" resolve to `["AAPL", "MSFT"]` without
  an extra tool call.
- **LLM:** `claude-haiku-4-5` primary (`temperature=0`); `gpt-4o-mini` fallback. Both
  are bound with all 11 tools at module import time — never rebuilt per-request.
- **Checkpointing:** FastAPI uses `AsyncPostgresSaver` (Railway Postgres); CLI uses
  `MemorySaver`. `thread_id = conversation_id` links turns.

### Tool Design

11 tools split into two tiers:

**Core tools** (single Ghostfolio API call):

| Tool                      | API endpoint                        | What it returns                                      |
| ------------------------- | ----------------------------------- | ---------------------------------------------------- |
| `get_portfolio_summary`   | `GET /api/v1/portfolio/holdings`    | Holdings, allocations, total value                   |
| `get_performance`         | `GET /api/v2/portfolio/performance` | Net performance %, absolute gain, net worth          |
| `get_transactions`        | `GET /api/v1/order`                 | Typed transactions, fee sum, filters                 |
| `analyze_diversification` | holdings (computed)                 | Sector weights, HHI-style score, concentration flags |
| `get_market_data`         | Yahoo Finance / yfinance            | Live price, 52-week range, batch quotes              |

**Advanced tools** (multi-step computation inside the tool):

`get_fee_drag_analysis`, `get_portfolio_health_scorecard`, `get_rebalancing_plan`,
`get_market_context_overlay`, `get_transaction_pattern_intelligence`, `get_proactive_risk_monitor`

All tools are `async def`, return structured `dict` (never raise), and use
`valueInBaseCurrency` (Ghostfolio v2 field) for position values.

---

## 3. Verification Strategy

Every LLM response passes through a **5-stage pipeline** before reaching the user:

```
response text + tool_results
        │
   1. check_disclaimer   → injects ⚠️ disclaimer if investment advice language detected
        │                  (keywords: buy, sell, invest, rebalance, recommend, …)
   2. check_hallucination → compares numeric claims in response against tool result numbers
        │                  • No tools ran → POTENTIAL_HALLUCINATION (MEDIUM)
        │                  • All tools failed → POTENTIAL_HALLUCINATION (HIGH) → triggers escalation
        │                  • Tool succeeded, number not in data → UNSUPPORTED_CLAIM (MEDIUM)
   3. check_freshness    → flags data_timestamp > 15 min (market) / 60 min (portfolio)
        │
   4. check_concentration → scans holdings for any position > 20% threshold
        │                   appends 🔔 warning to response text
   5. check_confidence   → assigns HIGH / MEDIUM / LOW
                           LOW forced when: predictions detected, no tool data,
                           speculative assets mentioned, or disclaimer triggered
```

**Why these checks?**

- **Disclaimer:** Finance agents have legal/ethical liability. Any advisory phrasing
  must carry the "not financial advice" caveat — automatically, not relying on the LLM.
- **Hallucination guard:** LLMs confidently cite training-data prices. Comparing response
  numbers against actual tool results catches the cases that matter: all-tools-failed + number stated.
- **Freshness:** Market prices from 45 minutes ago are materially wrong in volatile sessions.
- **Concentration:** A 90%-AAPL portfolio is a risk that must be surfaced even if the
  user didn't ask and even if the LLM downplays it.
- **Escalation node:** When `POTENTIAL_HALLUCINATION` at HIGH severity fires, the graph
  routes to `escalation_node` which replaces the response with a safe fallback and logs
  the incident — protecting users from confident fabrications.

---

## 4. Eval Results

### Test Suite Structure

| Suite                     | File                                    | Tests  | What it covers                                         |
| ------------------------- | --------------------------------------- | ------ | ------------------------------------------------------ |
| Unit — API schemas        | `tests/unit/api/`                       | 19     | Pydantic schema validation                             |
| Unit — Ghostfolio client  | `tests/unit/clients/`                   | 16     | HTTP mocking, auth, retry                              |
| Unit — Market client      | `tests/unit/clients/`                   | 10     | yfinance mocking, retry, fallback                      |
| Unit — Graph routing      | `tests/unit/graph/`                     | 28     | Routing logic, context extraction                      |
| Unit — Tools              | `tests/unit/tools/`                     | 19     | Tool output shapes, edge cases                         |
| Unit — Verification       | `tests/unit/verification/`              | 21     | All 5 pipeline stages                                  |
| Eval — Correctness        | `tests/eval/test_correctness.py`        | 12     | Math accuracy (%, sorts, sums, sector rollup)          |
| Eval — Tool selection     | `tests/eval/test_tool_selection.py`     | 10     | Docstring trigger keywords, domain boundary            |
| Eval — LLM tool selection | `tests/eval/test_llm_tool_selection.py` | 14     | LLM-driven tool routing, keyword coverage              |
| Eval — Tool execution     | `tests/eval/test_tool_execution.py`     | 16     | Advanced tool happy path + error cases                 |
| Eval — Multi-step         | `tests/eval/test_multi_step.py`         | 12     | Cross-tool data consistency                            |
| Eval — Edge cases         | `tests/eval/test_edge_cases.py`         | 10     | Unicode, empty portfolio, bad input                    |
| Eval — **Adversarial**    | `tests/eval/test_adversarial.py`        | **12** | Prompt injection, jailbreaks, fabricated numbers       |
| Adversarial (standalone)  | `tests/adversarial/test_adversarial.py` | —      | Safety / off-topic deflection (separate suite)         |
| LangSmith Experiments     | `tests/eval/ls_evals.py`                | 23     | Correctness, safety, latency, consistency scored evals |

### Running the Eval Suite

```bash
# All eval tests (fast, ~5–10 s, no network)
pytest tests/eval/ -v

# Unit tests only
pytest tests/unit/ -v

# Standalone adversarial / safety tests
pytest tests/adversarial/ -v

# Full suite with coverage
pytest tests/unit/ tests/eval/ --cov=agent --cov-report=term-missing

# LangSmith scored experiments (requires LANGCHAIN_API_KEY)
python tests/eval/ls_evals.py
python tests/eval/ls_evals.py --only correctness
python tests/eval/ls_evals.py --only safety
python tests/eval/ls_evals.py --prefix feat/my-branch
```

### Results (as of Feb 27, 2026)

```
223 passed in 10.68s     (unit + eval combined)
Coverage: 88.66% (total)   ← well above 40% required threshold
```

**All 223 tests pass. Zero failures.**

### Notable Findings During Development

| Bug found by tests                     | Root cause                                                                                                               | Fix                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| Performance tool always returned 0 %   | Ghostfolio v2 returns flat `netPerformancePercentage`, not nested `ytd.relativeChange`                                   | Rewrote parser to flat structure                 |
| 6 tools computed portfolio value as $0 | Field named `valueInBaseCurrency` not `value`                                                                            | Updated all 6 tools                              |
| Transaction account always empty       | `"Account"` (capital A) vs `"account"` (lowercase)                                                                       | Fixed field name                                 |
| Market data tests crashed              | `get_market_data` is async; tests used sync `.invoke()`                                                                  | Changed to `async def` + `.ainvoke()`            |
| `yfinance` intermittently empty        | Market closed; `history(period="1d")` returns nothing                                                                    | Added `"5d"` fallback + 3-attempt tenacity retry |
| Test asserted wrong severity           | `POTENTIAL_HALLUCINATION` only fires when _no_ tools ran or _all_ failed; successful-tool path emits `UNSUPPORTED_CLAIM` | Updated test assertions                          |

---

## 5. Observability Setup

### LangSmith Tracing

Every agent invocation is traced to **LangSmith** project `fortio-agent` automatically
via `LANGCHAIN_TRACING_V2=true`. Traces capture:

- Full message thread (human → AIMessage → ToolMessage × N → AIMessage)
- Tool call inputs and outputs per step
- Token counts (input / output / total) per LLM invocation
- Latency per node (reasoning, tools, verify)
- Verification flags and confidence level on the final state

**Insights gained from traces:**

1. The reasoning loop runs 2–3 turns for complex multi-tool queries (e.g. health scorecard);
   simple lookups resolve in 1 turn.
2. `get_market_data` latency spikes to 3–5 s when yfinance hits Yahoo's rate limiter —
   justifying the tenacity retry with exponential back-off (1–4 s window, 3 attempts).
3. `UNSUPPORTED_CLAIM` fires most often when the LLM reformulates a percentage (e.g.
   "approximately 12 %" vs the tool's `12.34`) — the number extraction regex catches
   integer approximations correctly.

### Structured Logging

All nodes and clients emit **structlog** JSON logs with context keys:
`tool`, `user_id`, `conversation_id`, `turn_number`, `confidence`, `flag_count`.
Railway aggregates these in its log stream; no additional log sink is configured yet.

### Eval Dataset in LangSmith (`ls_evals.py`)

`tests/eval/ls_evals.py` publishes a **60-example dataset** to LangSmith covering:

- Correctness examples (C-1 through C-8): expected numeric outputs for known inputs
- Latency examples (L-1 through L-4): max-seconds bounds per tool
- Consistency examples (K-1 through K-4): deterministic output for same inputs

These are run via `pytest tests/eval/ls_evals.py` and results appear in the LangSmith
"Datasets & Experiments" panel under the `fortio-agent` project.

---

## 6. Open Source Contribution

### What Was Released

**Fortio** is built on top of [Ghostfolio](https://github.com/ghostfolio/ghostfolio),
an MIT-licensed open-source wealth management platform.

During development, two API-level bugs were identified in Ghostfolio's Python integration:

1. **v2 portfolio performance endpoint returns a flat response object**, not a nested
   period-keyed structure. Documentation did not reflect this. The correct shape is:
   ```json
   { "performance": { "netPerformancePercentage": 0.123, "netPerformance": 987.65, ... } }
   ```
2. **`PortfolioPosition.valueInBaseCurrency`** (not `value`) is the correct field for
   position market value in the Ghostfolio TypeScript interface.

These findings are documented in the agent's `SETUP.md` and reflected in the corrected
tool implementations, which serve as a reference for any developer building a Python
client against Ghostfolio's API.

### Where to Find It

| Artifact               | Location                                                                                            |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| Full agent source      | `apps/agent/` in [github.com/meghamegs-lab/agentForge](https://github.com/meghamegs-lab/agentForge) |
| Live API + Swagger UI  | [fortio-agent-production.up.railway.app/docs](https://fortio-agent-production.up.railway.app/docs)  |
| LangSmith eval dataset | Project `fortio-agent` → Datasets → `fortio-correctness-v1`                                         |
| Adversarial eval suite | `apps/agent/tests/eval/test_adversarial.py` (12 tests, no LLM required)                             |
| Ghostfolio integration | [ghostfolio-production.up.railway.app](https://ghostfolio-production.up.railway.app)                |

### CLI Setup

Fortio ships a **Typer-powered CLI** (`fortio`) for terminal access without running a server.
Install it once from `apps/agent/` with the virtual environment active:

```bash
cd apps/agent
pip install -e .          # editable install — 'fortio' command becomes available
fortio --help
```

Key commands:

| Command                                | What it does                                                 |
| -------------------------------------- | ------------------------------------------------------------ |
| `fortio ask "…"`                       | Single question, prints answer, exits                        |
| `fortio ask "…" --verbose`             | Same, plus which tools were called and confidence level      |
| `fortio chat`                          | Interactive multi-turn REPL with `/help`, `/tools`, `/clear` |
| `fortio chat --verbose`                | REPL with verification flags shown after each response       |
| `fortio chat --conversation-id <uuid>` | Resume a prior in-process session                            |
| `fortio serve`                         | Start the FastAPI server (`uvicorn` on port 8001)            |
| `fortio serve --reload`                | Dev mode with hot-reload                                     |
| `fortio serve --workers 4`             | Production multi-worker mode                                 |
| `fortio demo`                          | Run all 11 tools in sequence; prints pass/fail summary       |
| `fortio mcp`                           | Start the MCP server (exposes all 11 tools via stdio)        |
| `fortio version`                       | Show active model, environment, and checkpoint backend       |

Inside `fortio chat` the REPL shows a rich panel per response — confidence badge
(`🟢 HIGH` / `🟡 MEDIUM` / `🔴 LOW`), tool call list, and follow-up suggestions —
all sourced from the verification pipeline output, not generated by the LLM.

**`fortio demo`** is a one-command diagnostic: it runs one targeted question per tool
in sequence and prints `Demo complete: 11/11 tools succeeded`. It is the fastest way
to confirm everything is wired up correctly after a fresh install or environment change.

### MCP Server

Fortio also exposes itself as a **Model Context Protocol (MCP) server** (`fortio mcp`),
making all 11 tools consumable by any MCP-compatible client (e.g. Claude Desktop,
other LLM agents). This is a zero-cost distribution path for the tool implementations.

**What the MCP server exposes:**

| Type          | Count | Details                                                                                     |
| ------------- | ----- | ------------------------------------------------------------------------------------------- |
| **Tools**     | 11    | All portfolio, performance, diversification, market, and risk tools                         |
| **Resources** | 3     | `portfolio://summary` · `portfolio://performance` · `portfolio://health`                    |
| **Prompts**   | 1     | `portfolio-analysis` with live pre-loaded context; focus: `risk`/`performance`/`fees`/`all` |

**Claude Desktop config** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "fortio": {
      "command": "fortio",
      "args": ["mcp"],
      "env": {
        "GHOSTFOLIO_BASE_URL": "http://localhost:3333",
        "GHOSTFOLIO_ACCESS_TOKEN": "your-security-token"
      }
    }
  }
}
```

**Cursor:** Settings → MCP → Add server → paste the same JSON block.

The MCP server runs over **stdio** (standard input/output). All human-readable output is
redirected to stderr so it doesn't corrupt the JSON-RPC wire protocol. The server
implementation is at `agent/mcp/server.py` and uses the `mcp` Python SDK.
