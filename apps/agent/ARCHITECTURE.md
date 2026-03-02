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
- **History redaction:** Before each LLM invocation, `_redact_prior_tool_messages()`
  strips the financial data payload out of `ToolMessage`s from prior turns, replacing
  them with a `[Stale tool result — call the tool again]` placeholder. This forces the
  LLM to re-fetch live data each turn instead of reading stale numbers from history,
  which prevents `POTENTIAL_HALLUCINATION` flags on multi-turn conversations.
- **Checkpointing:** FastAPI uses `AsyncPostgresSaver` (Railway Postgres); CLI uses
  `MemorySaver`. `thread_id = conversation_id` links turns.

### Tool Design

16 tools split into three tiers:

**Core tools** (5 — single Ghostfolio API call):

| Tool                      | API endpoint                        | What it returns                                      |
| ------------------------- | ----------------------------------- | ---------------------------------------------------- |
| `get_portfolio_summary`   | `GET /api/v1/portfolio/holdings`    | Holdings, allocations, total value                   |
| `get_performance`         | `GET /api/v2/portfolio/performance` | Net performance %, absolute gain, net worth          |
| `get_transactions`        | `GET /api/v1/order`                 | Typed transactions, fee sum, filters                 |
| `analyze_diversification` | holdings (computed)                 | Sector weights, HHI-style score, concentration flags |
| `get_market_data`         | Yahoo Finance / yfinance            | Live price, 52-week range, batch quotes              |

**Advanced tools** (6 — multi-step computation inside the tool):

`get_fee_drag_analysis`, `get_portfolio_health_scorecard`, `get_rebalancing_plan`,
`get_market_context_overlay`, `get_transaction_pattern_intelligence`, `get_proactive_risk_monitor`

**FIRE Goal Tracker tools** (5 — feature-flagged, opt-in via `FIRE_TRACKER_ENABLED=true`):

| Tool                              | Data Sources                   | What it returns                             |
| --------------------------------- | ------------------------------ | ------------------------------------------- |
| `set_retirement_goal`             | Postgres (`retirement_goals`)  | FIRE number (4% rule), years to retire      |
| `get_retirement_goal`             | Postgres                       | Saved goal parameters                       |
| `get_fire_progress`               | Postgres + Ghostfolio holdings | % toward FIRE number, dollar shortfall      |
| `calculate_retirement_projection` | Postgres + FRED (CPI + DGS10)  | Projected retirement date, real return      |
| `get_macro_data`                  | FRED API                       | Current CPI inflation, 10-yr Treasury yield |

All tools are `async def`, return structured `dict` (never raise), and use
`valueInBaseCurrency` (Ghostfolio v2 field) for position values.

---

## 2b. FIRE Goal Tracker — Stateful Extension

The FIRE tracker adds two new infrastructure layers when `FIRE_TRACKER_ENABLED=true`:

### Stateful data — `retirement_goals` Postgres table

| Column                     | Type        | Description                   |
| -------------------------- | ----------- | ----------------------------- |
| `user_id`                  | TEXT UNIQUE | Ties goal to Ghostfolio user  |
| `current_age`              | INTEGER     | User's age today              |
| `target_retirement_age`    | INTEGER     | Desired retirement age        |
| `target_annual_spending`   | NUMERIC     | $/year to spend in retirement |
| `safe_withdrawal_rate`     | NUMERIC     | Default 4% (0.04)             |
| `monthly_contribution`     | NUMERIC     | Monthly savings amount        |
| `expected_annual_return`   | NUMERIC     | Default 7%                    |
| `social_security_estimate` | NUMERIC     | Expected SS income            |

Table is created at FastAPI startup via `create_table_if_not_exists()` in `lifespan`. When
`FIRE_TRACKER_ENABLED=false`, the table is never created and no DB calls are made.

### FRED API client — `agent/clients/fred.py`

Fetches two FRED series:

| Series     | Name                  | Used for                                             |
| ---------- | --------------------- | ---------------------------------------------------- |
| `CPIAUCSL` | Consumer Price Index  | Year-over-year inflation for real-return calculation |
| `DGS10`    | 10-Year Treasury Rate | Risk-free rate benchmark                             |

Real return formula: `(1 + nominal_return) / (1 + inflation) − 1`

The client uses `tenacity` for exponential-back-off retry (3 attempts, 1–4 s window). If FRED
is unavailable, tools fall back to default macro values and append `macro_data_status: "default"`
to the tool output — the agent never errors out.

### New file layout (FIRE tracker only)

| File                        | Purpose                                        |
| --------------------------- | ---------------------------------------------- |
| `agent/clients/fred.py`     | Async FRED API client with tenacity retry      |
| `agent/db/__init__.py`      | DB package init                                |
| `agent/db/retirement.py`    | Postgres CRUD via psycopg2 + asyncio.to_thread |
| `agent/tools/retirement.py` | 5 LangChain `@tool` functions                  |

### Observability

- DB operations emit structlog events: `retirement_goal_saved`, `retirement_goal_deleted`
- Verification pipeline adds `FIRE_PROJECTION_SPECULATIVE` flag (LOW severity) on every projection response
- All tool calls traced in LangSmith under the `fortio-agent` project

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

| Suite                     | File                                        | Tests  | What it covers                                            |
| ------------------------- | ------------------------------------------- | ------ | --------------------------------------------------------- |
| Unit — API schemas        | `tests/unit/api/`                           | 30     | Pydantic schema validation                                |
| Unit — Clients            | `tests/unit/clients/`                       | 54     | HTTP mocking, auth, retry (Ghostfolio + market)           |
| Unit — FRED client        | `tests/unit/clients/test_fred_client.py`    | 20     | FRED HTTP mocking, retry, fallback defaults               |
| Unit — Graph routing      | `tests/unit/graph/`                         | 53     | Routing logic, context extraction, history redaction      |
| Unit — Tools              | `tests/unit/tools/`                         | 31     | Tool output shapes, edge cases                            |
| Unit — Retirement tools   | `tests/unit/tools/test_retirement_tools.py` | 37     | FIRE tool outputs, Postgres CRUD, FRED integration        |
| Unit — Verification       | `tests/unit/verification/`                  | 43     | All 5 pipeline stages                                     |
| Eval — Correctness        | `tests/evals/test_correctness.py`           | 11     | Math accuracy (%, sorts, sums, sector rollup)             |
| Eval — Tool selection     | `tests/evals/test_tool_selection.py`        | 28     | Docstring trigger keywords, domain boundary               |
| Eval — LLM tool selection | `tests/evals/test_llm_tool_selection.py`    | 16     | LLM-driven tool routing, keyword coverage                 |
| Eval — Tool execution     | `tests/evals/test_tool_execution.py`        | 12     | Advanced tool happy path + error cases                    |
| Eval — Multi-step         | `tests/evals/test_multi_step.py`            | 19     | Cross-tool data consistency                               |
| Eval — Edge cases         | `tests/evals/test_edge_cases.py`            | 28     | Unicode, empty portfolio, bad input                       |
| Eval — **Adversarial**    | `tests/evals/test_adversarial.py`           | **29** | Prompt injection, jailbreaks, fabricated numbers          |
| Eval — Safety             | `tests/evals/test_safety.py`                | 19     | Disclaimer, hallucination guard, confidence scoring       |
| Eval — Retirement/FIRE    | `tests/evals/test_retirement_eval.py`       | 35+    | 10 categories: projection math, multi-turn, FRED fallback |
| Adversarial (standalone)  | `tests/adversarial/test_adversarial.py`     | —      | Safety / off-topic deflection (separate suite)            |
| LangSmith Experiments     | `tests/evals/ls_evals.py`                   | 23     | Correctness, safety, latency, consistency scored evals    |

### Running the Eval Suite

```bash
# All eval tests (fast, ~5–10 s, no network)
pytest tests/evals/ -v

# Unit tests only
pytest tests/unit/ -v

# Standalone adversarial / safety tests
pytest tests/adversarial/ -v

# Full suite with coverage
pytest tests/unit/ tests/evals/ --cov=agent --cov-report=term-missing

# LangSmith scored experiments (requires LANGCHAIN_API_KEY)
python tests/evals/ls_evals.py
python tests/evals/ls_evals.py --only correctness
python tests/evals/ls_evals.py --only safety
python tests/evals/ls_evals.py --prefix feat/my-branch
```

### Results (as of Mar 1, 2026)

```
277 passed in 15.28s     (unit + eval combined, core suite — excluding LLM-live tests)
369+ passed              (with FIRE tracker tests: +20 FRED client, +37 retirement tools, +35 retirement evals)
Coverage: 83.20% (total)   ← well above 40% required threshold
```

**All tests pass. Zero failures.**

### Notable Findings During Development

| Bug found by tests                           | Root cause                                                                                                                          | Fix                                                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Performance tool always returned 0 %         | Ghostfolio v2 returns flat `netPerformancePercentage`, not nested `ytd.relativeChange`                                              | Rewrote parser to flat structure                                                                     |
| 6 tools computed portfolio value as $0       | Field named `valueInBaseCurrency` not `value`                                                                                       | Updated all 6 tools                                                                                  |
| Transaction account always empty             | `"Account"` (capital A) vs `"account"` (lowercase)                                                                                  | Fixed field name                                                                                     |
| Market data tests crashed                    | `get_market_data` is async; tests used sync `.invoke()`                                                                             | Changed to `async def` + `.ainvoke()`                                                                |
| `yfinance` intermittently empty              | Market closed; `history(period="1d")` returns nothing                                                                               | Added `"5d"` fallback + 3-attempt tenacity retry                                                     |
| Test asserted wrong severity                 | `POTENTIAL_HALLUCINATION` only fires when _no_ tools ran or _all_ failed; successful-tool path emits `UNSUPPORTED_CLAIM`            | Updated test assertions                                                                              |
| `POTENTIAL_HALLUCINATION` on multi-turn prod | In production (persistent history), the LLM read stale financial numbers from prior-turn `ToolMessage`s instead of re-calling tools | Added `_redact_prior_tool_messages()` in `graph.py` to strip prior-turn tool data before LLM context |

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

`tests/evals/ls_evals.py` publishes a **60-example dataset** to LangSmith covering:

- Correctness examples (C-1 through C-8): expected numeric outputs for known inputs
- Latency examples (L-1 through L-4): max-seconds bounds per tool
- Consistency examples (K-1 through K-4): deterministic output for same inputs

These are run via `pytest tests/evals/ls_evals.py` and results appear in the LangSmith
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
| Adversarial eval suite | `apps/agent/tests/evals/test_adversarial.py` (12 tests, no LLM required)                            |
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
