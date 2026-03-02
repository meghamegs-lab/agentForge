# Fortio Agent — Setup & Run Guide

Complete guide to running Fortio locally, connecting it to Ghostfolio, using the CLI,
setting up MCP for Claude Desktop / Cursor, and running the eval suite.

---

## Prerequisites

| Tool           | Version   | Install                                    |
| -------------- | --------- | ------------------------------------------ |
| Python         | 3.12+     | python.org or `brew install python@3.12`   |
| Docker Desktop | latest    | docker.com/products/docker-desktop         |
| Git            | any       | git-scm.com                                |
| Node.js        | 20+ (LTS) | nodejs.org (required for Ghostfolio build) |

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/meghamegs-lab/agentForge.git
cd agentForge
```

---

## Step 2 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the following:

```bash
# ── Required ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...           # console.anthropic.com → API Keys

# ── Ghostfolio connection ────────────────────────────────────────────────────
# For local Docker setup (see Step 3)
GHOSTFOLIO_BASE_URL=http://localhost:3333
GHOSTFOLIO_ACCESS_TOKEN=               # leave empty until Step 4

# ── Optional — for LangSmith tracing ────────────────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...              # smith.langchain.com → Settings
LANGCHAIN_PROJECT=fortio-agent

# ── Optional — OpenAI fallback ──────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ── Optional — FIRE Goal Tracker (retire-early planning) ─────────────────────
FIRE_TRACKER_ENABLED=false            # set to true to enable FIRE tools
FRED_API_KEY=                         # free key — see Step 2b below
```

---

## Step 2b — Get a FRED API key (optional — for FIRE Goal Tracker)

If you plan to use the **FIRE Goal Tracker** (`FIRE_TRACKER_ENABLED=true`), you need a free
FRED API key from the Federal Reserve Bank of St. Louis:

1. Go to **[fred.stlouisfed.org](https://fred.stlouisfed.org)**
2. Click **My Account** → **Create an Account** — no credit card required
3. Verify your email, then log in
4. Click your name (top-right) → **API Keys** → **Request API Key**
5. Enter a short description (e.g. `"Fortio FIRE tracker"`) and agree to terms
6. Copy the key that appears into `.env`:

```bash
FRED_API_KEY=your_32_character_key_here
FIRE_TRACKER_ENABLED=true
```

> **Rate limit:** 120 req/min (free tier) — Fortio uses at most 2 FRED calls per question.  
> **No expiry:** Keys are permanent until manually revoked.

Skip this step if you only want the core portfolio tools (FIRE tracker is off by default).

---

## Step 3 — Start Ghostfolio + databases (Docker)

```bash
docker compose -f docker/docker-compose.yml up postgres redis ghostfolio -d
```

Wait ~60 seconds for Ghostfolio to initialise. Verify it's up:

```bash
curl http://localhost:3333/api/v1/health
# Expected: {"status":"ok"}
```

Ghostfolio UI → http://localhost:3333

---

## Step 4 — Create your Ghostfolio account and get an access token

1. Open http://localhost:3333 in your browser
2. Click **Get Started** → create an account
3. Navigate to **Settings** → **Security token** → copy the token
4. Add it to `.env`:
   ```
   GHOSTFOLIO_ACCESS_TOKEN=your-security-token-here
   ```

The `GhostfolioClient` uses this token to fetch bearer tokens automatically —
you do **not** need to handle token refresh manually.

> **Using the demo instance instead?**  
> Get your access token from [ghostfol.io/en/demo](https://ghostfol.io/en/demo)  
> and set `GHOSTFOLIO_BASE_URL=https://ghostfol.io`

---

## Step 5 — Add holdings in Ghostfolio

Go to http://localhost:3333 → **Portfolio** → **+ Add transaction**

Add at least 3–4 holdings so the agent has real data to analyse. The agent works best
with holdings from multiple asset classes (equity, bonds, ETFs).

---

## Step 6 — Set up the Python agent

```bash
cd apps/agent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install as editable to enable the `fortio` CLI command
pip install -e .
```

Verify the install:

```bash
fortio version
```

---

## Step 7 — Run the test suite

Always run tests before starting the agent to confirm the environment is correct.

### Unit tests (fast, mocked, no network needed)

```bash
# All unit tests
pytest tests/unit/ -v

# Specific sub-suites
pytest tests/unit/tools/ -v            # 11 tool tests
pytest tests/unit/verification/ -v    # 5-stage verification pipeline
pytest tests/unit/clients/ -v         # Ghostfolio + market data clients
pytest tests/unit/graph/ -v           # LangGraph routing
pytest tests/unit/api/ -v             # FastAPI schemas
```

### Eval suite (correctness, tool selection, edge cases, multi-step, adversarial)

```bash
pytest tests/evals/ -v
```

All eval tests are mocked — no real API calls, no LLM costs.

| File                         | What it tests                                             | Tests |
| ---------------------------- | --------------------------------------------------------- | ----- |
| `test_correctness.py`        | Arithmetic accuracy, percentage conversions, fee sums     | 11    |
| `test_tool_selection.py`     | Docstring trigger keywords, domain boundary isolation     | 28    |
| `test_llm_tool_selection.py` | LLM-driven tool selection routing and keyword coverage    | 16    |
| `test_tool_execution.py`     | Advanced tool happy path + error cases                    | 12    |
| `test_multi_step.py`         | Cross-tool consistency, referential integrity             | 19    |
| `test_edge_cases.py`         | Malformed data, unicode, large portfolios                 | 28    |
| `test_adversarial.py`        | Prompt injection, jailbreaks, fabricated number detection | 29    |
| `test_safety.py`             | Verification pipeline: disclaimer, hallucination, scoring | 19    |

### Adversarial tests (standalone safety suite)

```bash
pytest tests/adversarial/ -v           # safety / jailbreak / off-topic
```

> **Two adversarial test locations:**
>
> - `tests/evals/test_adversarial.py` — adversarial tests that run as part of the eval suite
> - `tests/adversarial/test_adversarial.py` — standalone adversarial/safety suite (run separately)

### FIRE Goal Tracker tests (requires `FIRE_TRACKER_ENABLED=true`)

```bash
pytest tests/unit/clients/test_fred_client.py -v       # 20 FRED client tests
pytest tests/unit/tools/test_retirement_tools.py -v    # 37 retirement tool unit tests
pytest tests/evals/test_retirement_eval.py -v           # 35+ FIRE eval tests (10 categories)
```

### Coverage report

```bash
pytest tests/unit/ tests/evals/ --cov=agent --cov-report=term-missing
```

---

## Step 8 — Start the agent

### Option A: CLI (development — no server needed)

```bash
# Single question
fortio ask "What does my portfolio look like?"
fortio ask "How concentrated am I in tech?" --verbose   # show tools + confidence

# Interactive multi-turn REPL
fortio chat
fortio chat --verbose   # show verification flags after each response

# Run all 11 tools end-to-end (confirms everything is wired up)
fortio demo
```

Inside `fortio chat`:

- Type your question and press Enter
- `/help` — show topic suggestions
- `/tools` — list all 11 agent tools
- `/clear` — visual separator
- `exit` or `q` — end the session

> **`fortio demo`** runs one targeted question per tool in sequence and prints a final
> summary (`Demo complete: 11/11 tools succeeded`). Use it as a quick smoke test after
> setup or any environment change.

### Option B: FastAPI server

```bash
# Start the API server (port 8001)
fortio serve

# Development mode with hot-reload
fortio serve --reload
```

- **Swagger docs:** http://localhost:8001/docs
- **Health check:** http://localhost:8001/health

### Option C: Full Docker stack (mirrors production)

```bash
# From the monorepo root — builds and starts all services
docker compose -f docker/docker-compose.yml up -d --build

docker compose -f docker/docker-compose.yml ps   # check status
docker compose -f docker/docker-compose.yml logs agent -f  # follow logs
```

---

## Step 9 — MCP Server (Claude Desktop / Cursor)

Fortio exposes all 11 tools as an MCP server so Claude Desktop and Cursor can query
your portfolio directly in their chat interfaces.

### What the MCP server provides

| Type          | Count | Details                                                                |
| ------------- | ----- | ---------------------------------------------------------------------- |
| **Tools**     | 11    | All portfolio, performance, diversification, market, and risk tools    |
| **Resources** | 3     | `portfolio://summary`, `portfolio://performance`, `portfolio://health` |
| **Prompts**   | 1     | `portfolio-analysis` with live pre-loaded portfolio context            |

### Start the MCP server (standalone)

```bash
fortio mcp
# Output goes to stderr; stdout is reserved for the MCP JSON-RPC wire
```

### Claude Desktop setup

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

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

Restart Claude Desktop. Fortio tools appear in the tool list automatically.

> **Using Railway?** Replace `http://localhost:3333` with your Railway Ghostfolio URL.

### Cursor setup

1. Open Cursor → **Settings** → **MCP**
2. Click **Add server**
3. Paste the same JSON block as the Claude Desktop config above
4. Restart Cursor — Fortio tools appear in the MCP tool panel automatically

### Using the `portfolio-analysis` prompt

In Claude Desktop, open the **Prompt Library** → select **portfolio-analysis**.
Choose a focus area:

| Focus         | What Claude analyses                                   |
| ------------- | ------------------------------------------------------ |
| `risk`        | Concentration risk, sector exposure, rebalancing needs |
| `performance` | Returns, period comparisons, best/worst performers     |
| `fees`        | Fee drag, total fees paid, cost reduction              |
| `all`         | Comprehensive analysis (default)                       |

Live portfolio data is embedded automatically — no tool calls needed.

---

## Step 10 — LangSmith Experiments

Run the scored LangSmith eval suite to benchmark correctness, safety, and latency:

```bash
# Requires LANGCHAIN_API_KEY in .env
python tests/evals/ls_evals.py

# Single eval type
python tests/evals/ls_evals.py --only correctness
python tests/evals/ls_evals.py --only safety
python tests/evals/ls_evals.py --only latency
python tests/evals/ls_evals.py --only consistency
python tests/evals/ls_evals.py --only tool-keywords

# Tag by branch for comparison
python tests/evals/ls_evals.py --prefix feat/my-branch
```

View results at: https://smith.langchain.com → Projects → **fortio-evals**

---

## Quick Reference: What Runs Where

| Service       | Local URL                  | Port  | How to Start                   |
| ------------- | -------------------------- | ----- | ------------------------------ |
| Ghostfolio UI | http://localhost:3333      | 3333  | `docker compose up ghostfolio` |
| Agent FastAPI | http://localhost:8001/docs | 8001  | `fortio serve` or Docker       |
| Agent CLI     | terminal                   | n/a   | `fortio chat`                  |
| Agent Demo    | terminal                   | n/a   | `fortio demo`                  |
| MCP server    | stdio (no port)            | n/a   | `fortio mcp`                   |
| PostgreSQL    | localhost:5432             | 5432  | `docker compose up postgres`   |
| Redis         | localhost:6379             | 6379  | `docker compose up redis`      |
| LangSmith     | smith.langchain.com        | cloud | env var only                   |

---

## CLI Commands Reference

| Command                                | What it does                                              |
| -------------------------------------- | --------------------------------------------------------- |
| `fortio ask "…"`                       | Single question, prints answer, exits                     |
| `fortio ask "…" --verbose`             | Same, plus tools called + confidence level                |
| `fortio chat`                          | Interactive multi-turn REPL                               |
| `fortio chat --verbose`                | REPL with verification flags shown after each response    |
| `fortio chat --conversation-id <uuid>` | Resume a prior in-process session                         |
| `fortio demo`                          | Run all 11 tools in sequence — smoke test for graders     |
| `fortio serve`                         | Start the FastAPI server on port 8001                     |
| `fortio serve --reload`                | Dev mode with hot-reload                                  |
| `fortio serve --workers 4`             | Production multi-worker mode                              |
| `fortio mcp`                           | Start the MCP server (stdio, for Claude Desktop / Cursor) |
| `fortio version`                       | Show active model, environment, and checkpoint config     |

---

## CLI vs Interfaces Cheat Sheet

| Interface                    | Best for                            | History persistence |
| ---------------------------- | ----------------------------------- | ------------------- |
| `fortio ask`                 | Quick one-off queries, scripting    | None (stateless)    |
| `fortio chat`                | Interactive exploration in terminal | In-memory only      |
| `fortio demo`                | Verify all 11 tools work            | None                |
| `fortio serve` + `/api/chat` | Angular/Ghostfolio frontend         | Postgres (full)     |
| `fortio mcp`                 | Claude Desktop / Cursor integration | Host-managed        |

---

## Troubleshooting

### Ghostfolio won't start

```bash
docker compose -f docker/docker-compose.yml logs ghostfolio
# Usually: postgres isn't ready yet — wait 30 s and retry
docker compose -f docker/docker-compose.yml restart ghostfolio
```

### Agent can't reach Ghostfolio (connection refused)

```bash
# Check GHOSTFOLIO_BASE_URL in .env:
# Local:           http://localhost:3333
# Docker internal: http://ghostfolio:3333  (used by agent container)
# Railway:         https://your-app.up.railway.app
```

### Bearer token expired (401 errors)

`GhostfolioClient` auto-refreshes bearer tokens. If you're testing manually:

```bash
curl -X POST http://localhost:3333/api/v1/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"accessToken": "YOUR_SECURITY_TOKEN"}'
```

### LangSmith not showing traces

```bash
# Check .env — all three must be set:
LANGCHAIN_TRACING_V2=true       # must be string "true"
LANGCHAIN_API_KEY=ls__...       # must start with ls__
LANGCHAIN_PROJECT=fortio-agent
```

### MCP server not connecting in Claude Desktop

1. Confirm `fortio` is on `PATH`: `which fortio`
2. If not found, use the full path in the config:
   ```json
   "command": "/Users/you/.venv/bin/fortio"
   ```
3. Check Claude Desktop logs for JSON-RPC errors
4. Verify `GHOSTFOLIO_ACCESS_TOKEN` is set in the MCP env block

### `fortio mcp` shows no output

That's correct — all human-readable output goes to stderr; stdout is the MCP wire protocol.
Run with stderr visible: `fortio mcp 2>&1 | head -20`

### Tests failing with import errors

```bash
# Ensure you're in the right directory with venv active
cd apps/agent
source .venv/bin/activate
pip install -e .
PYTHONPATH=. pytest tests/unit/ -v
```

---

## FIRE Goal Tracker (Optional Feature)

The FIRE (Financial Independence, Retire Early) Goal Tracker is an opt-in module that adds
retirement planning to Fortio using live Federal Reserve macro data (FRED API).

### Enable it

```bash
# In .env (see Step 2b for how to get a FRED API key)
FIRE_TRACKER_ENABLED=true
FRED_API_KEY=your_key_here
```

When enabled, 5 additional tools are registered and 4 new API routes become active:

### New tools

| Tool                              | What it does                                                    |
| --------------------------------- | --------------------------------------------------------------- |
| `set_retirement_goal`             | Save/update your FIRE goal (age, target spending, savings rate) |
| `get_retirement_goal`             | Retrieve your saved goal and computed FIRE number               |
| `get_fire_progress`               | Current % progress toward your FIRE number                      |
| `calculate_retirement_projection` | Project retirement date using live FRED CPI + DGS10             |
| `get_macro_data`                  | Fetch current CPI inflation rate and 10-year Treasury yield     |

### New API routes

| Operation       | Route                                    |
| --------------- | ---------------------------------------- |
| Create / Update | `POST /api/goals/retirement`             |
| Read            | `GET /api/goals/retirement/{user_id}`    |
| Delete          | `DELETE /api/goals/retirement/{user_id}` |

Goals are stored in the `retirement_goals` Postgres table, created automatically at startup.

### FRED data used

| FRED Series | Name                  | Used for                                           |
| ----------- | --------------------- | -------------------------------------------------- |
| `CPIAUCSL`  | Consumer Price Index  | Year-over-year inflation (real return calculation) |
| `DGS10`     | 10-Year Treasury Rate | Risk-free rate context for SWR                     |

> **Zero impact when disabled:** When `FIRE_TRACKER_ENABLED=false` (default), no tools are
> registered, no DB tables are created, and no FRED calls are made. Existing deployments are
> completely unaffected.

---

## CI/CD

GitHub Actions runs on every PR and push to `main`:

| Job            | Trigger                             | What it does                                 |
| -------------- | ----------------------------------- | -------------------------------------------- |
| `test-agent`   | Any change to `apps/agent/**`       | Unit tests (hard gate) + evals (advisory)    |
| `deploy-agent` | Push to `main`, agent files changed | Deploy to Railway (requires `RAILWAY_TOKEN`) |

Evals in CI use dummy API keys — they run on mocked data and never call real LLMs.

---

## Adding the AI Chat Widget to Ghostfolio

The Fortio AI chat widget (`GfAiChatComponent`) is a floating chat button (bottom-right corner)
that opens a full chat panel and connects to the Fortio FastAPI agent.

### How it works

The widget is an Angular standalone component in `libs/ui/src/lib/ai-chat/`. It takes one input:

```html
<gf-ai-chat [fortioApiUrl]="'https://your-fortio-agent.railway.app'" />
```

At runtime it calls `POST {fortioApiUrl}/api/chat` with the user's message and renders the
agent's answer, confidence badge, and verification flags inline.

### It's already integrated — just set one env var

The widget is already embedded in two places in the Ghostfolio frontend:

| Location                               | File                                                | Visible when      |
| -------------------------------------- | --------------------------------------------------- | ----------------- |
| **All authenticated pages** (floating) | `apps/client/src/app/app.component.html`            | User is logged in |
| **Public demo page**                   | `apps/client/src/app/pages/public/public-page.html` | Always            |

The widget URL comes from the Ghostfolio API server via the `/api/v1/info` endpoint.
To point it at your deployed agent, set one environment variable in the **Ghostfolio** service
(`.env` or Railway env vars):

```bash
FORTIO_API_URL=https://fortio-agent-production.up.railway.app
```

The data flow:

```
FORTIO_API_URL (env var)
  → apps/api/src/services/configuration/configuration.service.ts  (reads env)
  → apps/api/src/app/info/info.service.ts  (sets info.fortioApiUrl)
  → GET /api/v1/info  (served to Angular frontend)
  → app.component.html  (passes to widget as [fortioApiUrl])
  → GfAiChatComponent  (calls POST {fortioApiUrl}/api/chat)
```

### Adding the widget to a new Ghostfolio page

If you want to embed the widget on a page that does not already have it:

**Step 1 — Import the component** in the page's `@Component` decorator:

```typescript
// apps/client/src/app/pages/my-page/my-page.component.ts
import { GfAiChatComponent } from '@ghostfolio/ui/ai-chat';

@Component({
  imports: [
    GfAiChatComponent,
    // ... other existing imports
  ],
  // ...
})
export class MyPageComponent {
  public info: InfoItem;  // already available on most pages via DataService
}
```

**Step 2 — Add the tag to the template**:

```html
<!-- apps/client/src/app/pages/my-page/my-page.component.html -->

<!-- Place at the very bottom, outside any scroll containers -->
<gf-ai-chat [fortioApiUrl]="info?.fortioApiUrl || 'http://localhost:8001'" />
```

The widget uses `position: fixed` so it floats over all page content regardless of where
the tag is placed in the DOM.

**That's it** — no routing changes, no service injection, no NgModule registration needed.
It's a standalone component.

### Widget behaviour reference

| UI element               | Behaviour                                                           |
| ------------------------ | ------------------------------------------------------------------- |
| 💬 button (bottom-right) | Opens / closes the chat panel                                       |
| Confidence badge         | 🟢 HIGH · 🟡 MEDIUM · 🔴 LOW — colour-coded per response            |
| ⚠️ Flags detected        | Shown when any HIGH or MEDIUM verification flag is present          |
| Debug panel `{ }`        | Toggle to see raw API response (flags, tool calls, conversation ID) |
| Suggested prompts        | Clickable chips on the welcome screen for quick starts              |
