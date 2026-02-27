# Fortio — Ghostfolio Finance AI Agent

A production-ready AI agent built on LangGraph that provides intelligent portfolio analysis
by integrating with [Ghostfolio](https://ghostfol.io), the open-source wealth management platform.

## Production URLs

| Service               | URL                                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| **Ghostfolio App**    | [ghostfolio-production.up.railway.app](https://ghostfolio-production.up.railway.app)               |
| **Fortio Agent API**  | [fortio-agent-production.up.railway.app](https://fortio-agent-production.up.railway.app)           |
| **Fortio Agent Docs** | [fortio-agent-production.up.railway.app/docs](https://fortio-agent-production.up.railway.app/docs) |

## Stack

- **Agent:** LangGraph (state machine) + Claude Sonnet (primary LLM) + GPT-4o (fallback)
- **Tools:** 11 domain tools — 5 core (Ghostfolio REST API) + 6 advanced multi-step
- **Verification:** 5-stage pipeline (disclaimer, hallucination guard, freshness, concentration, confidence)
- **UI:** FastAPI REST API (production, deployed) + **Typer CLI** (terminal)
- **Observability:** LangSmith
- **Deployment:** Railway (CI/CD via GitHub Actions)

---

## Quick Start (30 minutes to first API call)

### 1. Clone and set up Python environment

```bash
# From the monorepo root
cd apps/agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Get your API keys

| Key                 | Where to get it                  | Cost                     |
| ------------------- | -------------------------------- | ------------------------ |
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys | ~$3/1M tokens            |
| `OPENAI_API_KEY`    | platform.openai.com → API Keys   | ~$5/1M tokens (fallback) |
| `LANGCHAIN_API_KEY` | smith.langchain.com → Settings   | Free tier available      |

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
# For MVP, the Ghostfolio demo API requires no token for public endpoints
```

### 4. Get a Ghostfolio bearer token (for authenticated endpoints)

```bash
# Using the demo instance — get the access token from ghostfol.io/en/demo
curl -X POST https://ghostfol.io/api/v1/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"accessToken": "DEMO_ACCESS_TOKEN"}'
# Copy the authToken from the response into GHOSTFOLIO_ACCESS_TOKEN in .env
```

### 5. Run tests

```bash
pytest tests/unit/ -v       # unit tests (mocked, no network)
pytest tests/eval/ -v       # eval suite (correctness, tool selection, edge cases)
```

---

## CLI — Command Line Interface

Fortio ships a **Typer-powered CLI** (`fortio`) for quick terminal access to the agent without
starting a server. It supports single-shot queries, an interactive REPL, and server management.

### Install as a CLI Tool

#### Option A — Editable install (recommended for development)

```bash
# From apps/agent/ with your virtual environment active
cd apps/agent
pip install -e .
```

After installation the `fortio` command is available system-wide in your virtual environment:

```bash
fortio --help
```

#### Option B — Install dependencies only (without registering the entry point)

```bash
pip install -r requirements.txt
# Then invoke via Python module:
python -m agent.cli --help
```

#### Option C — Install from PyPI (once published)

```bash
pip install fortio-agent
fortio --help
```

---

### CLI Commands

#### `fortio ask` — Single question, print answer, exit

```bash
# Basic question
fortio ask "What does my portfolio look like?"

# With verbose output (shows which tools were called)
fortio ask "How concentrated am I in tech?" --verbose

# Specify a user ID (for multi-user Ghostfolio setups)
fortio ask "What are my top performers?" --user-id alice
```

#### `fortio chat` — Interactive multi-turn REPL

```bash
# Start a new session
fortio chat

# Resume a prior session (history stored in-memory, same process only)
fortio chat --conversation-id 550e8400-e29b-41d4-a716-446655440000

# Show tool call details on every turn
fortio chat --verbose
```

Inside the REPL:

- Type your question and press **Enter**
- Type `exit`, `quit`, or `q` to end the session
- Press **Ctrl+C** to interrupt at any time

Example session:

```
Fortio Chat ─────────────────────────────────────────
 Welcome to Fortio, your Ghostfolio Finance Assistant!
 ...
Session: 4f3c2a1b-...

You: What's my overall portfolio value?
╭─ Fortio ──────────────────────────────────────────╮
│  Your total portfolio value is $124,580 across 12  │
│  holdings ...                                      │
│                                       🟢 Confidence: HIGH │
╰───────────────────────────────────────────────────╯

You: Which sector am I most exposed to?
...
You: quit
Goodbye! 👋
```

#### `fortio serve` — Start the FastAPI server

```bash
# Default: binds to 0.0.0.0:8001
fortio serve

# Custom port
fortio serve --port 9000

# Development mode with hot-reload
fortio serve --reload

# Multiple workers (production; incompatible with --reload)
fortio serve --workers 4
```

API docs are available at `http://localhost:8001/docs` once the server is running.

#### `fortio version` — Show active configuration

```bash
fortio version
```

```
╭─ Version Info ──────────────────────────────────╮
│  Fortio – Ghostfolio Finance AI Agent           │
│                                                 │
│  Version:        0.1.0                          │
│  Primary model:  claude-haiku-4-5               │
│  Fallback model: gpt-4o-mini                    │
│  Environment:    development                    │
│  Checkpoint:     postgres                       │
│  Log level:      INFO                           │
╰─────────────────────────────────────────────────╯
```

---

### CLI vs Other Interfaces

| Interface                    | Best for                                | History                    | Persistence |
| ---------------------------- | --------------------------------------- | -------------------------- | ----------- |
| `fortio ask`                 | Quick one-off queries, scripting        | None                       | None        |
| `fortio chat`                | Interactive exploration in the terminal | In-memory (session only)   | None        |
| `fortio serve` + `/api/chat` | Angular/Ghostfolio frontend             | Postgres (across restarts) | Full        |

> **Note:** `fortio chat` and `fortio ask` use `MemorySaver` — conversation history is
> kept only for the lifetime of the process and is not persisted to Postgres.

---

## Cursor IDE Setup — Get the Most Out of Claude

### Step 1: Enable Cursor Rules

The `.cursor/rules/agentforge.mdc` file is already in this repo.
Cursor automatically picks it up. To verify:

1. Open Cursor → Settings (Cmd+,) → Rules
2. Confirm `agentforge.mdc` appears under Project Rules

### Step 2: Set Claude as your model

1. Cursor → Settings → Models
2. Select **claude-sonnet-4-5** or **claude-opus-4-5** as default
3. Add your Anthropic API key under Settings → API Keys

### Step 3: Use Claude effectively in Cursor

#### For generating new tool code:

Open the command palette (Cmd+K) and type:

```
Create a new LangGraph tool following the pattern in agent/tools/portfolio.py
that [describe what you want]. Include unit tests in tests/unit/tools/.
```

#### For TDD — write tests first:

```
Write failing pytest unit tests for a tool called [name] that [behavior].
Use respx to mock httpx. Follow the pattern in tests/unit/tools/test_tools.py.
```

#### For debugging agent behavior:

```
I'm seeing this LangSmith trace [paste trace]. The agent is calling the wrong tool
for this query. What's wrong with the tool docstring or routing?
```

#### For the verification layer:

```
Add a new verification check called [name] to agent/verification/__init__.py
following the pattern of the existing 5 checks. Add it to run_verification_pipeline().
Write the unit tests first.
```

### Step 4: Cursor keyboard shortcuts for this project

| Shortcut      | Use                                    |
| ------------- | -------------------------------------- |
| `Cmd+K`       | Inline code generation / edit          |
| `Cmd+L`       | Open chat for longer context questions |
| `Cmd+Shift+L` | Add current file to chat context       |
| `@filename`   | Reference a specific file in chat      |
| `@codebase`   | Search across the whole project        |

### Step 5: Useful Cursor prompts for this project

**Scaffold a new tool:**

```
@agent/tools/portfolio.py Create a new tool file for [feature].
Follow the exact same structure: async private function, sync @tool wrapper,
asyncio.run(), error handling returning {"status": "error"}, data_timestamp.
```

**Fix a failing test:**

```
@tests/unit/tools/test_tools.py This test is failing: [paste error].
Fix the implementation without changing the test.
```

**Add a verification check:**

```
@agent/verification/__init__.py Add check number 6: [describe the check].
Must follow the signature: (response: str, tool_results: list[dict]) -> tuple[str, list[VerificationFlag]]
Add to run_verification_pipeline() and write the unit tests.
```

---

## Why Claude Over GPT-4o for This Project

| Capability                   | Claude Sonnet 4.5                  | GPT-4o                          |
| ---------------------------- | ---------------------------------- | ------------------------------- |
| Tool use accuracy            | ✅ Better multi-tool chaining      | ✅ Good                         |
| Financial reasoning          | ✅ Less hallucination on numbers   | ⚠️ Occasionally invents figures |
| Safety instruction following | ✅ Excellent at "never do X" rules | ✅ Good                         |
| Context window               | 200k tokens                        | 128k tokens                     |
| Cost                         | ~$3/1M tokens                      | ~$5/1M tokens                   |
| Structured output            | ✅ Native                          | ✅ Native                       |

The larger context window matters when you're passing full portfolio data (can be large) plus conversation history plus tool results all in the same prompt.

---

## Running the Full Docker Stack

```bash
# From monorepo root
docker compose -f docker/docker-compose.yml up -d

# Ghostfolio UI → http://localhost:3333
# Agent FastAPI docs → http://localhost:8001/docs
```

---

## Project Structure

```
apps/agent/
├── agent/
│   ├── config.py              # All settings via pydantic-settings
│   ├── prompts.py             # System prompt
│   ├── cli.py                 # Typer CLI (fortio ask / chat / serve / version)
│   ├── tools/                 # 11 LangGraph tools
│   │   ├── __init__.py        # ALL_TOOLS export
│   │   │   — Core tools (single Ghostfolio API call) —
│   │   ├── portfolio.py       # get_portfolio_summary
│   │   ├── performance.py     # get_performance
│   │   ├── transactions.py    # get_transactions
│   │   ├── diversification.py # analyze_diversification
│   │   ├── market.py          # get_market_data
│   │   │   — Advanced multi-step tools —
│   │   ├── fee_drag.py        # get_fee_drag_analysis
│   │   ├── health_scorecard.py# get_portfolio_health_scorecard
│   │   ├── rebalancing.py     # get_rebalancing_plan
│   │   ├── market_context.py  # get_market_context_overlay
│   │   ├── transaction_patterns.py # get_transaction_pattern_intelligence
│   │   └── proactive_monitor.py    # get_proactive_risk_monitor
│   ├── verification/
│   │   └── __init__.py        # All 5 verifiers + pipeline
│   ├── graph/
│   │   ├── state.py           # AgentState TypedDict
│   │   └── graph.py           # LangGraph assembly
│   ├── clients/
│   │   ├── ghostfolio.py      # Typed async API client
│   │   └── market.py          # yfinance client
│   ├── api/
│   │   ├── main.py            # FastAPI app + /health + /api/chat
│   │   └── schemas.py         # Request/response Pydantic models

├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── tools/             # Tool unit tests (mocked with respx)
│   │   └── verification/      # Verifier unit tests
│   ├── eval/
│   │   ├── test_correctness.py   # 12 arithmetic/structure accuracy tests
│   │   ├── test_tool_selection.py# 10 docstring coverage + domain boundary tests
│   │   └── test_edge_cases.py    # 10 edge case / resilience tests
│   ├── integration/
│   └── adversarial/
├── Dockerfile
├── railway.toml
├── .env.example
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

---

## Eval Suite

The agent ships with 32 evaluation tests (zero real network calls — all mocked with `respx`):

| File                                                                       | Focus                                                                                                              | Tests |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----- |
| [`tests/eval/test_correctness.py`](./tests/eval/test_correctness.py)       | Arithmetic accuracy, percentage conversions, sort order, fee sums, sector rollup                                   | 12    |
| [`tests/eval/test_tool_selection.py`](./tests/eval/test_tool_selection.py) | Tool docstring trigger coverage, domain boundary isolation, parameter mapping                                      | 10    |
| [`tests/eval/test_edge_cases.py`](./tests/eval/test_edge_cases.py)         | Dict vs list holdings format, zero-value holdings, missing fields, unicode names, large portfolios, invalid inputs | 10    |

```bash
pytest tests/eval/ -v               # run all evals
pytest tests/eval/test_correctness.py -v   # run one file
```

---

## LangSmith Observability Setup

1. Create free account at smith.langchain.com
2. Create a project called `fortio-agent`
3. Copy your API key to `.env` as `LANGCHAIN_API_KEY`
4. Set `LANGCHAIN_TRACING_V2=true`
5. Run the agent — traces appear automatically in your LangSmith dashboard

---

## Open Source

This agent is published as `fortio-agent` on PyPI:

```bash
pip install fortio-agent
```

See [PyPI page](https://pypi.org/project/fortio-agent/) for usage docs.
