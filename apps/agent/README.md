# Fortio — Ghostfolio Finance AI Agent

A production-ready AI agent built on LangGraph that provides intelligent portfolio analysis
by integrating with [Ghostfolio](https://ghostfol.io), the open-source wealth management platform.

## Production URLs

| Service                                           | URL                                                                                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Ghostfolio App**                                | [ghostfolio-production.up.railway.app](https://ghostfolio-production-453e.up.railway.app/en/home)                                     |
| **Ghostfolio Demo Account to test Fortio Agent ** | [fortio-agent-production.up.railway.app](https://ghostfolio-production-453e.up.railway.app/en/p/e6b67d66-b727-4fa8-a3ac-01ec36d5dde1) |
| **Fortio Agent Docs**                             | [fortio-agent-production.up.railway.app/docs](https://fortio-agent-production.up.railway.app/docs)                                    |

## Stack

- **Agent:** LangGraph (state machine) + Claude Haiku (primary LLM) + GPT-4o-mini (fallback)
- **Tools:** 16 domain tools — 5 core (Ghostfolio REST API) + 6 advanced multi-step + 5 FIRE Goal Tracker (opt-in, feature-flagged)
- **Verification:** 5-stage pipeline (disclaimer, hallucination guard, freshness, concentration, confidence) + `FIRE_PROJECTION_SPECULATIVE` flag
- **Interfaces:** FastAPI REST API · Typer CLI (`fortio`) · MCP server (`fortio mcp`)
- **Observability:** LangSmith tracing + structured eval suite (370+ tests)
- **Deployment:** Railway (CI/CD via GitHub Actions)

---

## Fortio Agent exists under apps/agent folder https://github.com/meghamegs-lab/agentForge/tree/main/apps/agent

---

## Quick Start (30 minutes to first API call)

### 1. Clone and set up Python environment

\`\`\`bash

# From the monorepo root

cd apps/agent
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
\`\`\`

### 2. Get your API keys

| Key                 | Where to get it                  | Cost                     |
| ------------------- | -------------------------------- | ------------------------ |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys | ~$3/1M tokens            |
| `OPENAI_API_KEY`    | platform.openai.com → API Keys   | ~$5/1M tokens (fallback) |
| `LANGCHAIN_API_KEY` | smith.langchain.com → Settings   | Free tier available      |

### 3. Configure environment

\`\`\`bash
cp .env.example .env

# Edit .env and fill in your API keys

\`\`\`

### 4. Get a Ghostfolio bearer token (for authenticated endpoints)

\`\`\`bash
curl -X POST https://ghostfol.io/api/v1/auth/anonymous \
 -H "Content-Type: application/json" \
 -d '{"accessToken": "DEMO_ACCESS_TOKEN"}'

# Copy the authToken from the response into GHOSTFOLIO_ACCESS_TOKEN in .env

\`\`\`

### 5. Install the CLI and run tests

\`\`\`bash

# Install as editable so the `fortio` command is available

pip install -e .

# Unit tests (mocked, no network)

pytest tests/unit/ -v

# Eval suite (correctness, tool selection, edge cases, multi-step)

pytest tests/evals/ -v
\`\`\`
Eval Data Set -->[ https://github.com/meghamegs-lab/agentForge/tree/main/apps/agent/tests/evals/](https://github.com/meghamegs-lab/agentForge/blob/main/apps/agent/tests/evals/eval_suite.json)
---

## CLI — Command Line Interface

Fortio ships a **Typer-powered CLI** (`fortio`) for quick terminal access to the agent without
starting a server. It supports single-shot queries, an interactive REPL, server management,
and an MCP server mode.

### Install the CLI

#### Option A — Editable install (recommended for development)

\`\`\`bash
cd apps/agent
pip install -e .
fortio --help
\`\`\`

#### Option B — Run as a Python module (no install required)

\`\`\`bash
pip install -r requirements.txt
python -m agent.cli --help
\`\`\`

#### Option C — Install from PyPI (once published)

\`\`\`bash
pip install fortio-agent
fortio --help
\`\`\`

---

### CLI Commands

#### `fortio ask` — Single question, print answer, exit

\`\`\`bash
fortio ask "What does my portfolio look like?"
fortio ask "How concentrated am I in tech?" --verbose
fortio ask "What are my top performers?" --user-id alice
\`\`\`

#### `fortio chat` — Interactive multi-turn REPL

\`\`\`bash
fortio chat
fortio chat --conversation-id 550e8400-e29b-41d4-a716-446655440000
fortio chat --verbose
\`\`\`

Inside the REPL:

| Command      | Action                            |
| ------------ | --------------------------------- |
| `/help`      | Show topic suggestions grid       |
| `/tools`     | List all 11 available agent tools |
| `/clear`     | Print a visual separator          |
| `exit` / `q` | End the session                   |
| **Ctrl+C**   | Interrupt at any time             |

Example session:

\`\`\`
Fortio Chat ─────────────────────────────────────────
Welcome to Fortio, your Ghostfolio Finance Assistant!
...
Session: 4f3c2a1b-...

You: What's my overall portfolio value?
╭─ Fortio ──────────────────────────────────────────╮
│ Your total portfolio value is $124,580 across 12 │
│ holdings ... │
│ 🟢 Confidence: HIGH │
╰───────────────────────────────────────────────────╯
💡 Follow-up suggestions

1. How has my portfolio performed YTD?
2. Am I too concentrated in any sector?
3. Give me a portfolio health scorecard

You: quit
Goodbye! 👋
\`\`\`

#### `fortio serve` — Start the FastAPI server

\`\`\`bash
fortio serve # default: 0.0.0.0:8001
fortio serve --port 9000 # custom port
fortio serve --reload # dev mode with hot-reload
fortio serve --workers 4 # production multi-worker
\`\`\`

API docs: `http://localhost:8001/docs`

#### `fortio mcp` — Start the MCP server

\`\`\`bash
fortio mcp
\`\`\`

Starts the Fortio MCP (Model Context Protocol) server on **stdio**, exposing all 11 portfolio
tools + 3 resources + 1 prompt template to Claude Desktop, Cursor, or any MCP-compatible AI host.

See the [MCP section](#mcp--model-context-protocol) below for full setup instructions.

#### `fortio demo` — Run all 11 tools in sequence

\`\`\`bash
fortio demo
fortio demo --no-verbose # suppress per-tool detail
\`\`\`

Runs one targeted question per tool so all 11 tools are exercised end-to-end in a single command.
Each question is specifically designed to trigger exactly one tool. Prints a summary at the end:
`Demo complete: 11/11 tools succeeded`. Ideal for verifying a fresh install or demonstrating
the agent in a recorded session.

#### `fortio version` — Show active configuration

\`\`\`bash
fortio version
\`\`\`

\`\`\`
╭─ Version Info ──────────────────────────────────╮
│ Fortio – Ghostfolio Finance AI Agent │
│ │
│ Version: 0.1.0 │
│ Primary model: claude-haiku-4-5 │
│ Fallback model: gpt-4o-mini │
│ Environment: development │
│ Checkpoint: postgres │
│ Log level: INFO │
╰─────────────────────────────────────────────────╯
\`\`\`

---

### CLI vs Other Interfaces

| Interface                    | Best for                                | History                    | Persistence |
| ---------------------------- | --------------------------------------- | -------------------------- | ----------- |
| `fortio ask`                 | Quick one-off queries, scripting        | None                       | None        |
| `fortio chat`                | Interactive exploration in the terminal | In-memory (session only)   | None        |
| `fortio demo`                | Verify all 11 tools work end-to-end     | None                       | None        |
| `fortio serve` + `/api/chat` | Angular/Ghostfolio frontend             | Postgres (across restarts) | Full        |
| `fortio mcp`                 | Claude Desktop / Cursor integration     | Host-managed               | Host        |

> **Note:** `fortio chat`, `fortio ask`, and `fortio demo` use `MemorySaver` — conversation history is
> kept only for the lifetime of the process and is not persisted to Postgres.

---

## MCP — Model Context Protocol

Fortio can run as an **MCP server**, letting Claude Desktop, Cursor, and other MCP-compatible
hosts query your Ghostfolio portfolio using natural language — with no browser and no API keys
pasted into a chat box.

### What the MCP server exposes

| Type          | Count | Details                                                                |
| ------------- | ----- | ---------------------------------------------------------------------- |
| **Tools**     | 16    | All portfolio, performance, diversification, market, and FIRE tools    |
| **Resources** | 3     | `portfolio://summary`, `portfolio://performance`, `portfolio://health` |
| **Prompts**   | 1     | `portfolio-analysis` (pre-loads live data; focus: risk/perf/fees/all)  |

### Claude Desktop setup

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

\`\`\`json
{
"mcpServers": {
"fortio": {
"command": "fortio",
"args": ["mcp"],
"env": {
"GHOSTFOLIO_BASE_URL": "https://your-ghostfolio.railway.app",
"GHOSTFOLIO_ACCESS_TOKEN": "your-security-token"
}
}
}
}
\`\`\`

Restart Claude Desktop — the Fortio tools appear in the tool list automatically.

> **Local Ghostfolio?** Use `"GHOSTFOLIO_BASE_URL": "http://localhost:3333"` instead.

> **MCP server not connecting?** Confirm `fortio` is on `PATH` with `which fortio`. If not found, use the full path: `"command": "/path/to/.venv/bin/fortio"`.

### Cursor setup

1. Open Cursor → **Settings** → **MCP**
2. Click **Add server**
3. Paste the same JSON block as above
4. Restart Cursor — Fortio tools appear in the MCP tool list

### Using the `portfolio-analysis` prompt template

In Claude Desktop, open the **Prompt Library** and select **portfolio-analysis**.
You can optionally specify a focus area:

| Focus value   | What Claude analyses                                        |
| ------------- | ----------------------------------------------------------- |
| `risk`        | Concentration risk, sector exposure, rebalancing needs      |
| `performance` | Returns, period comparisons, best/worst performers          |
| `fees`        | Fee drag, total fees paid, cost reduction opportunities     |
| `all`         | Comprehensive: diversification, performance, fees, and risk |

The prompt fetches **live portfolio data** (holdings + YTD performance + health score) and
embeds it into context before Claude responds — no explicit tool calls required.

### MCP resources (passive context loading)

Resources let Claude load portfolio data as background context without an explicit tool call:

\`\`\`
portfolio://summary — current holdings, allocations, total value
portfolio://performance — YTD return %, absolute gain/loss, current value
portfolio://health — health score (0-100), letter grade, action items
\`\`\`

---

## Eval Suite

The agent ships with **370+ evaluation tests** across 9 files (zero real network calls — all mocked
with `respx`), plus a LangSmith experiment suite.

### Eval files

| File                                                                                 | Focus                                                                                      | Tests |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ----- |
| [`tests/evals/test_correctness.py`](./tests/evals/test_correctness.py)               | Arithmetic accuracy, percentage conversions, sort order, fee sums, sector rollup           | 11    |
| [`tests/evals/test_tool_selection.py`](./tests/evals/test_tool_selection.py)         | Tool docstring trigger keywords, domain boundary isolation, parameter mapping              | 28    |
| [`tests/evals/test_llm_tool_selection.py`](./tests/evals/test_llm_tool_selection.py) | LLM-driven tool selection routing and keyword coverage                                     | 16    |
| [`tests/evals/test_tool_execution.py`](./tests/evals/test_tool_execution.py)         | Advanced tool execution: happy path, error cases, edge inputs for all 6 advanced tools     | 12    |
| [`tests/evals/test_multi_step.py`](./tests/evals/test_multi_step.py)                 | Cross-tool consistency, referential integrity, multi-session proactive monitor             | 19    |
| [`tests/evals/test_edge_cases.py`](./tests/evals/test_edge_cases.py)                 | Dict/list format switching, zero-value holdings, unicode, large portfolios, invalid inputs | 28    |
| [`tests/evals/test_adversarial.py`](./tests/evals/test_adversarial.py)               | Prompt injection, jailbreaks, off-topic deflection, fabricated number detection            | 29    |
| [`tests/evals/test_safety.py`](./tests/evals/test_safety.py)                         | Verification pipeline stages: disclaimer, hallucination guard, confidence scoring          | 19    |
| [`tests/evals/test_retirement_eval.py`](./tests/evals/test_retirement_eval.py)       | FIRE Goal Tracker: projection math, multi-turn, FRED fallback (10 categories)              | 35+   |
| [`tests/evals/ls_evals.py`](./tests/evals/ls_evals.py)                               | LangSmith tracked experiments: correctness, safety, latency, consistency, tool-keywords    | 23    |

### Running the eval suite

\`\`\`bash

# All evals (fast, ~5-10 s, no network required)

pytest tests/evals/ -v

# Specific eval files

pytest tests/evals/test_correctness.py -v
pytest tests/evals/test_tool_selection.py -v
pytest tests/evals/test_llm_tool_selection.py -v
pytest tests/evals/test_tool_execution.py -v
pytest tests/evals/test_multi_step.py -v
pytest tests/evals/test_edge_cases.py -v
pytest tests/evals/test_adversarial.py -v
pytest tests/evals/test_safety.py -v

# Unit tests

pytest tests/unit/ -v

# Adversarial / safety tests (standalone suite)

pytest tests/adversarial/ -v

# With coverage report

pytest tests/unit/ tests/evals/ --cov=agent --cov-report=term-missing
\`\`\`

### LangSmith experiments (`ls_evals.py`)

`ls_evals.py` runs 5 scored eval types tracked in the LangSmith UI:

| Eval type       | What it tests                                                           | Examples |
| --------------- | ----------------------------------------------------------------------- | -------- |
| `correctness`   | Tool output matches numeric ground truth (totals, %, fee sums, sectors) | 6        |
| `safety`        | Verification pipeline catches disclaimer, hallucination, concentration  | 5        |
| `latency`       | Tool calls complete within 2 s on mocked I/O                            | 4        |
| `consistency`   | Same inputs → identical outputs across two runs                         | 3        |
| `tool-keywords` | Tool docstrings contain all LLM trigger keywords                        | 5        |

\`\`\`bash

# Prerequisites: LANGCHAIN_API_KEY must be set in .env

python tests/evals/ls_evals.py # all 5 eval types
python tests/evals/ls_evals.py --only safety # one type
python tests/evals/ls_evals.py --only latency
python tests/evals/ls_evals.py --prefix feat/my-branch # tag experiment run

# View results at: https://smith.langchain.com → Projects → fortio-evals

\`\`\`

### Evals in CI

Evals run automatically on every PR and push to `main` as part of the `test-agent` CI job.
They are **advisory** (`continue-on-error: true`) — results are visible in CI without blocking
deployment. Unit tests are the hard gate.

---

## FIRE Goal Tracker

Fortio ships an opt-in **FIRE (Financial Independence, Retire Early) planning** module backed
by live Federal Reserve data from the [FRED API](https://fred.stlouisfed.org).

### Enable it

```bash
# In .env
FIRE_TRACKER_ENABLED=true
FRED_API_KEY=your_key_here   # free — see SETUP.md Step 2b
```

### What users can ask

| Question                                        | Tools invoked                                 |
| ----------------------------------------------- | --------------------------------------------- |
| "Retire at 50, spend $80K/year"                 | `set_retirement_goal`                         |
| "What's my FIRE number?"                        | `get_retirement_goal`                         |
| "Am I on track to retire?"                      | `get_fire_progress` + `get_portfolio_summary` |
| "When can I retire at my current savings rate?" | `calculate_retirement_projection` + FRED      |
| "What if I save $500 more per month?"           | `calculate_retirement_projection` (override)  |
| "What's the current 10-year Treasury yield?"    | `get_macro_data`                              |

### How it works

Retirement projections fetch **live CPI inflation** and **10-year Treasury yields** from FRED:

```
Real return = (1 + nominal_return) / (1 + CPI_inflation) − 1
```

Goals persist in the `retirement_goals` Postgres table. CRUD API routes:

| Operation       | Route                                    |
| --------------- | ---------------------------------------- |
| Create / Update | `POST /api/goals/retirement`             |
| Read            | `GET /api/goals/retirement/{user_id}`    |
| Delete          | `DELETE /api/goals/retirement/{user_id}` |

Every projection response gets a `FIRE_PROJECTION_SPECULATIVE` flag (LOW severity) from the
verification pipeline — surfaced as a `⚠️ Flags detected` note in the chat UI.

> **Zero impact when disabled:** `FIRE_TRACKER_ENABLED=false` (default) — no tools registered,
> no DB tables created, no FRED calls made. Existing deployments are completely unaffected.

See [`BOUNTY.md`](./BOUNTY.md) for the full feature write-up.

---

## Verification Pipeline

Every final LLM response passes through a **5-stage verification pipeline**:

| Stage | Check                 | What it does                                                                |
| ----- | --------------------- | --------------------------------------------------------------------------- |
| 1     | `check_disclaimer`    | Detects investment-advice language; appends "not financial advice"          |
| 2     | `check_hallucination` | Extracts numbers from response; flags any not present in tool results       |
| 3     | `check_freshness`     | Rejects tool results older than `DATA_MAX_AGE_MINUTES` (default: 5 min)     |
| 4     | `check_concentration` | Flags any position > `PORTFOLIO_CONCENTRATION_THRESHOLD` (default: 20%)     |
| 5     | `check_confidence`    | Scores response HIGH/MEDIUM/LOW; escalates on HIGH severity + hallucination |

Flags are returned in the API response and surfaced as `⚠️ Flags detected` in the chat UI.

---

## Project Structure

\`\`\`
apps/agent/
├── agent/
│ ├── config.py # All settings via pydantic-settings
│ ├── prompts.py # System prompt + persona guardrails
│ ├── cli.py # Typer CLI: ask / chat / serve / mcp / version
│ ├── tools/ # 11 LangGraph tools
│ │ ├── **init**.py # ALL_TOOLS export
│ │ │ — Core tools (single Ghostfolio API call) —
│ │ ├── portfolio.py # get_portfolio_summary
│ │ ├── performance.py # get_performance
│ │ ├── transactions.py # get_transactions
│ │ ├── diversification.py # analyze_diversification
│ │ ├── market.py # get_market_data (Yahoo Finance via yfinance)
│ │ │ — Advanced multi-step tools —
│ │ ├── fee_drag.py # get_fee_drag_analysis
│ │ ├── health_scorecard.py# get_portfolio_health_scorecard
│ │ ├── rebalancing.py # get_rebalancing_plan
│ │ ├── market_context.py # get_market_context_overlay
│ │ ├── transaction_patterns.py # get_transaction_pattern_intelligence
│ │ └── proactive_monitor.py # get_proactive_risk_monitor
│ ├── verification/
│ │ └── **init**.py # 5-stage pipeline
│ ├── mcp/
│ │ └── server.py # MCP server: 11 tools + 3 resources + 1 prompt
│ ├── graph/
│ │ ├── state.py # AgentState TypedDict
│ │ └── graph.py # LangGraph assembly + routing
│ ├── clients/
│ │ ├── ghostfolio.py # Typed async Ghostfolio API client
│ │ └── market.py # yfinance async client (source: Yahoo Finance)
│ └── api/
│ ├── main.py # FastAPI app + /health + /api/chat
│ └── schemas.py # Request/response Pydantic models
│
├── tests/
│ ├── conftest.py
│ ├── unit/ # Fast mocked unit tests (hard gate in CI)
│ │ ├── tools/
│ │ ├── clients/
│ │ ├── graph/
│ │ ├── api/
│ │ └── verification/
│ ├── evals/ # Eval suite (advisory in CI)
│ │ ├── test_correctness.py
│ │ ├── test_tool_selection.py
│ │ ├── test_llm_tool_selection.py
│ │ ├── test_tool_execution.py
│ │ ├── test_multi_step.py
│ │ ├── test_edge_cases.py
│ │ ├── test_adversarial.py
│ │ ├── test_safety.py
│ │ └── ls_evals.py # LangSmith experiment runner
│ ├── integration/
│ └── adversarial/
│
├── docker/
│ └── docker-compose.yml # Ghostfolio + PostgreSQL + Redis
├── Dockerfile
├── railway.toml
├── pyproject.toml # Package metadata (fortio-agent on PyPI)
├── .env.example
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
\`\`\`

---

## LangSmith Observability Setup

1. Create free account at smith.langchain.com
2. Create a project called `fortio-agent`
3. Copy your API key to `.env` as `LANGCHAIN_API_KEY`
4. Set `LANGCHAIN_TRACING_V2=true`
5. Run the agent — traces appear automatically in your LangSmith dashboard

---

## Running the Full Docker Stack

\`\`\`bash

# From monorepo root

docker compose -f docker/docker-compose.yml up -d

# Ghostfolio UI → http://localhost:3333

# Agent FastAPI docs → http://localhost:8001/docs

\`\`\`

---

## Why Claude Over GPT-4o for This Project

| Capability                   | Claude Haiku 4.5                   | GPT-4o-mini                     |
| ---------------------------- | ---------------------------------- | ------------------------------- |
| Tool use accuracy            | ✅ Better multi-tool chaining      | ✅ Good                         |
| Financial reasoning          | ✅ Less hallucination on numbers   | ⚠️ Occasionally invents figures |
| Safety instruction following | ✅ Excellent at "never do X" rules | ✅ Good                         |
| Context window               | 200k tokens                        | 128k tokens                     |
| Cost                         | ~$0.80/1M tokens (input)           | ~$0.15/1M tokens (input)        |
| Structured output            | ✅ Native                          | ✅ Native                       |

The larger context window matters when passing full portfolio data + conversation history +
tool results in the same prompt.

---

## Open Source

This agent is published as `fortio-agent` on PyPI. The package metadata is defined in `pyproject.toml`.

### Install from PyPI

\`\`\`bash
pip install fortio-agent
\`\`\`

After installation, the `fortio` CLI command is available:

\`\`\`bash
fortio --help
fortio ask "What does my portfolio look like?"
fortio serve
\`\`\`

### Links

- **PyPI Package:** [pypi.org/project/fortio-agent](https://pypi.org/project/fortio-agent/)
- **Source Code:** [github.com/meghamegs-lab/agentForge](https://github.com/meghamegs-lab/agentForge)
- **Package Config:** `apps/agent/pyproject.toml` (metadata, dependencies, entry points)
