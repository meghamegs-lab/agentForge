# AgentForge — Complete Setup & Run Guide

## Answers to All Your Questions

---

### Q1: Where is .env.example in the zip?
```
agentforge_scaffold.zip
└── .env.example          ← ROOT of the zip, same level as README.md
```
Unzip → you'll see `.env.example` immediately. Copy it:
```bash
cp .env.example .env
```

---


**The .cursor/rules/agentforge.mdc file I generated is auto-loaded by Cursor.**
Every time you ask Claude to write code, it already knows:
- Use LangGraph, not AgentExecutor
- Return error dicts, never raise
- Always include data_timestamp
- Use httpx, not requests

---

### Q3: Chainlit vs FastAPI — when to use which

**SHORT ANSWER: Use BOTH, for different audiences, at the same time.**

| | Chainlit | FastAPI |
|---|---|---|
| **What it is** | Chat UI (like ChatGPT) running on port 8000 | REST API running on port 8001 |
| **Who uses it** | YOU during development + demo reviewers | Ghostfolio's Angular frontend (later) |
| **When to use** | Right now — always | When you need to embed agent in Ghostfolio UI |
| **What it shows** | Chat bubbles, tool call traces inline, thumbs feedback | JSON responses, Swagger docs at /docs |
| **Auth** | Session-based (Chainlit handles it) | Bearer token (you implement) |
| **In production** | Can be used for public demo | MUST have this for Angular integration |

**Why both?** Chainlit is your working interface today. FastAPI is what Ghostfolio's Angular
frontend will call when you embed the chat panel. You don't choose between them — they
run simultaneously on different ports.

**Docker Compose starts both:**
```yaml
command: >
  sh -c "chainlit run agent/ui/chainlit_app.py --port 8000 &
         uvicorn agent.api.main:app --port 8001"
```

---

### Q4: Ghostfolio deployment — one platform, not two

**WHY we suggested Railway then Render:** Different tools for different stages.
That adds unnecessary complexity. Here is the SINGLE PLATFORM approach:

#### Option A: Railway only (RECOMMENDED — simplest)
```
Tuesday MVP:   Deploy agent only → point at ghostfol.io demo API (free)
Friday:        Add Ghostfolio as a second Railway service in same project
Both in one:   Railway project = Ghostfolio service + Agent service + Postgres + Redis
```
**Railway cost: ~$15-20/month total for 4 services**

#### Option B: Local Docker only (for the submission demo)
```
Run everything locally with docker compose up
Record a demo video of the working agent
Submit the video + GitHub repo link
No cloud deployment cost
```
**This is actually the safest option for the 24h deadline.**

#### Challenges if you use TWO platforms (Railway → Render):
- Two different CI/CD pipelines to configure
- Internal networking doesn't work across platforms (must use public URLs)
- Cross-platform latency adds 50-200ms per request
- Two sets of environment variables to keep in sync
- Different healthcheck and rollback mechanisms
- Double the debugging surface when things break

**RECOMMENDATION: Use Railway for everything OR local Docker for the demo.**
Don't split across platforms.

---

### Q5: Which skills.md files do you need?

I've generated a project-specific `SKILLS.md` at `.cursor/rules/SKILLS.md`.
Here's why you need it and what's in it:

| Skill in SKILLS.md | Why You Need It |
|---|---|
| Creating a new tool | So Cursor generates correct async pattern every time |
| Adding a Ghostfolio API client method | Prevents missing `@retry` decorator |
| Writing TDD tests | Correct `@respx.mock` pattern for httpx mocking |
| Adding a verification check | Correct signature + registration in pipeline |
| LangGraph node pattern | Prevents returning full state instead of diff |
| Common mistakes table | Claude in Cursor avoids the 8 most common errors |
| Ghostfolio endpoint reference | Cursor can autocomplete correct endpoint paths |

**How to use it:**
```
@SKILLS.md  Create a new tool that calls GET /api/v1/portfolio/investments
```
Cursor reads the skills file and generates code that follows all patterns exactly.

---

## Running the App Locally (Step by Step)

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+
- Git

### Step 1: Clone and structure
```bash
git clone https://github.com/meghamegs-lab/agentForge.git
cd agentForge
```

### Step 2: Copy environment file
```bash
cp .env.example .env
```
Open `.env` and fill in:
- `ANTHROPIC_API_KEY` — from console.anthropic.com
- `LANGCHAIN_API_KEY` — from smith.langchain.com (optional but recommended)
- Leave `GHOSTFOLIO_ACCESS_TOKEN` empty for now (step 5 below)

### Step 3: Start Ghostfolio + databases
```bash
docker compose -f docker/docker-compose.yml up postgres redis ghostfolio -d
```
Wait ~60 seconds for Ghostfolio to start. Check it's up:
```bash
curl http://localhost:3333/api/v1/health
# Should return: {"status":"ok"}
```

### Step 4: Create your Ghostfolio account
1. Open http://localhost:3333
2. Click "Get Started" → create account
3. Settings → Security token → copy the token
4. Add the token to `.env`:
   ```
   GHOSTFOLIO_ACCESS_TOKEN=your-token-here
   ```

### Step 5: Get a bearer token (for API calls)
```bash
curl -X POST http://localhost:3333/api/v1/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"accessToken": "YOUR_SECURITY_TOKEN"}'
# Copy the "authToken" from response — this is your bearer token
# NOT needed in .env — the GhostfolioClient fetches it automatically
```

### Step 6: Add some holdings in Ghostfolio
Go to http://localhost:3333 → Portfolio → + Add transaction
Add at least 3-4 holdings so the agent has data to work with.

### Step 7: Set up Python agent
```bash
cd apps/agent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 8: Run tests (TDD — do this before running the app)
```bash
pytest tests/unit/verification/ -v        # pure Python, no network needed
pytest tests/unit/tools/ -v               # uses respx mocks, no network
```
All should pass. If any fail, fix before proceeding.

### Step 9: Run the agent
```bash
# Terminal 1: Chainlit UI (chat interface)
chainlit run agent/ui/chainlit_app.py

# Terminal 2: FastAPI (REST API)
uvicorn agent.api.main:app --reload --port 8001
```

### Step 10: Open and test
- **Chainlit chat:** http://localhost:8000
  - Notice: the proactive risk monitor fires automatically when you open it
- **FastAPI docs:** http://localhost:8001/docs
- **Ghostfolio UI:** http://localhost:3333
- **LangSmith traces:** smith.langchain.com → your project

### Step 11: Run full Docker stack (optional — mirrors production)
```bash
# From repo root
docker compose -f docker/docker-compose.yml up -d --build

# Check all services are up
docker compose -f docker/docker-compose.yml ps
```

---

## Quick Reference: What Runs Where

| Service | Local URL | Port | How to Start |
|---|---|---|---|
| Ghostfolio UI | http://localhost:3333 | 3333 | Docker Compose |
| Ghostfolio API | http://localhost:3333/api/v1 | 3333 | Docker Compose |
| Agent Chainlit | http://localhost:8000 | 8000 | `chainlit run ...` or Docker |
| Agent FastAPI | http://localhost:8001/docs | 8001 | `uvicorn ...` or Docker |
| PostgreSQL | localhost:5432 | 5432 | Docker Compose |
| Redis | localhost:6379 | 6379 | Docker Compose |
| LangSmith | smith.langchain.com | cloud | env var only |

---

## Troubleshooting

**Ghostfolio won't start:**
```bash
docker compose logs ghostfolio   # check for DB connection errors
# Usually means postgres isn't ready yet — wait 30s and retry
```

**Agent can't reach Ghostfolio:**
```bash
# Check GHOSTFOLIO_BASE_URL in .env
# Local: http://localhost:3333
# Docker internal: http://ghostfolio:3333
```

**Bearer token expired (401 errors):**
```bash
# GhostfolioClient auto-refreshes — but if manual testing:
curl -X POST http://localhost:3333/api/v1/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"accessToken": "YOUR_SECURITY_TOKEN"}'
```

**LangSmith not showing traces:**
```bash
# Check .env:
LANGCHAIN_TRACING_V2=true       # must be string "true"
LANGCHAIN_API_KEY=ls__...       # must start with ls__
LANGCHAIN_PROJECT=ghostfolio-agent
```
