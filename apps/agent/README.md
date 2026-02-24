# Fortio — Ghostfolio Finance AI Agent

A production-ready AI agent built on LangGraph that provides intelligent portfolio analysis
by integrating with [Ghostfolio](https://ghostfol.io), the open-source wealth management platform.

## Stack
- **Agent:** LangGraph (state machine) + Claude Sonnet (primary LLM) + GPT-4o (fallback)
- **Tools:** 5 domain tools hitting Ghostfolio REST API + Yahoo Finance
- **Verification:** 5-stage pipeline (disclaimer, hallucination guard, freshness, concentration, confidence)
- **UI:** Chainlit (dev/demo) + FastAPI (production)
- **Observability:** LangSmith
- **Deployment:** Railway

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

| Key | Where to get it | Cost |
|-----|----------------|------|
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys | ~$3/1M tokens |
| `OPENAI_API_KEY` | platform.openai.com → API Keys | ~$5/1M tokens (fallback) |
| `LANGCHAIN_API_KEY` | smith.langchain.com → Settings | Free tier available |

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

### 5. Run the Chainlit UI
```bash
chainlit run agent/ui/chainlit_app.py
# Open http://localhost:8000
```

### 6. Run tests
```bash
pytest tests/unit/ -v
```

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
| Shortcut | Use |
|----------|-----|
| `Cmd+K` | Inline code generation / edit |
| `Cmd+L` | Open chat for longer context questions |
| `Cmd+Shift+L` | Add current file to chat context |
| `@filename` | Reference a specific file in chat |
| `@codebase` | Search across the whole project |

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

| Capability | Claude Sonnet 4.5 | GPT-4o |
|-----------|-------------------|--------|
| Tool use accuracy | ✅ Better multi-tool chaining | ✅ Good |
| Financial reasoning | ✅ Less hallucination on numbers | ⚠️ Occasionally invents figures |
| Safety instruction following | ✅ Excellent at "never do X" rules | ✅ Good |
| Context window | 200k tokens | 128k tokens |
| Cost | ~$3/1M tokens | ~$5/1M tokens |
| Structured output | ✅ Native | ✅ Native |

The larger context window matters when you're passing full portfolio data (can be large) plus conversation history plus tool results all in the same prompt.

---

## Running the Full Docker Stack (Friday submission)
```bash
# From monorepo root
docker compose -f docker/docker-compose.yml up -d

# Ghostfolio UI → http://localhost:3333
# Fortio Agent Chainlit UI → http://localhost:8000
# Agent FastAPI docs → http://localhost:8001/docs
```

---

## Project Structure
```
apps/agent/
├── agent/
│   ├── config.py              # All settings via pydantic-settings
│   ├── prompts.py             # System prompt
│   ├── tools/                 # 5 LangGraph tools
│   │   ├── portfolio.py       # get_portfolio_summary
│   │   ├── performance.py     # get_performance
│   │   ├── transactions.py    # get_transactions
│   │   ├── diversification.py # analyze_diversification
│   │   └── market.py          # get_market_data
│   ├── verification/
│   │   └── __init__.py        # All 5 verifiers + pipeline
│   ├── graph/
│   │   ├── state.py           # AgentState TypedDict
│   │   └── graph.py           # LangGraph assembly
│   ├── clients/
│   │   ├── ghostfolio.py      # Typed async API client
│   │   └── market.py          # yfinance client
│   ├── api/                   # FastAPI (coming Friday)
│   └── ui/
│       └── chainlit_app.py    # Chat interface
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── tools/             # Tool unit tests (mocked)
│   │   └── verification/      # Verifier unit tests
│   ├── integration/
│   ├── adversarial/
│   └── eval/
├── .cursor/rules/agentforge.mdc
├── .env.example
├── requirements.txt
└── pytest.ini
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
