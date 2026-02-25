# Railway Deployment Guide

Deploy the full AgentForge stack to Railway from the `main` branch.
4 services: Ghostfolio · Fortio Agent · Postgres · Redis

---

## Architecture on Railway

```
Railway Project: agentforge
├── ghostfolio         (Docker, repo root Dockerfile)   → public URL :3333
├── fortio-agent       (Docker, apps/agent/Dockerfile)  → public URL :8001
├── postgres           (Railway managed Postgres)
└── redis              (Railway managed Redis)
```

Internal networking (agent → ghostfolio): `http://ghostfolio.railway.internal:3333`

---

## Step 1 — Create the Railway Project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Select **"Deploy from GitHub repo"**
3. Connect your GitHub and pick the `agentForge` repo (`meghamegs-lab/agentForge`)
4. Select branch: **`main`**
5. Click **"Add service"** (not "Deploy now" yet — we need 4 services)

---

## Step 2 — Add Postgres (Managed)

In the Railway project dashboard:

1. Click **"+ New"** → **"Database"** → **"Add PostgreSQL"**
2. Railway provisions it and auto-sets `DATABASE_URL` for linked services
3. Copy the connection details (you'll need them for Ghostfolio env vars)

---

## Step 3 — Add Redis (Managed)

1. Click **"+ New"** → **"Database"** → **"Add Redis"**
2. Railway provisions it and auto-sets `REDIS_URL`
3. Copy the connection details (you'll need `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`)

---

## Step 4 — Configure Ghostfolio Service

In Railway, the first GitHub-linked service defaults to the repo root.

**Settings → General:**

- Name: `ghostfolio`
- Root Directory: `/` (repo root)
- Branch: `main`
- Repo: `meghamegs-lab/agentForge`

**Settings → Build:**

- Builder: `Dockerfile`
- Dockerfile path: `Dockerfile`

**Settings → Deploy:**

- Start command: (leave empty — entrypoint.sh handles it)
- Health check path: `/api/v1/health`

### Ghostfolio Environment Variables

Set these in the Railway service → **Variables** tab:

```bash
# Database — copy from Railway Postgres service
DATABASE_URL=postgresql://postgres:PASSWORD@HOST:PORT/railway

# Redis — split from Railway Redis REDIS_URL (format: redis://:password@host:port)
REDIS_HOST=redis.railway.internal      # or the external host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Ghostfolio security — generate with: openssl rand -hex 32
ACCESS_TOKEN_SALT=generate_a_random_32char_hex_string
JWT_SECRET_KEY=generate_a_random_32char_hex_string

# Port (Railway sets this automatically — do NOT hardcode)
# PORT is injected by Railway

# Optional
NODE_ENV=production
```

---

## Step 5 — Configure Fortio Agent Service

In Railway: **"+ New"** → **"GitHub Repo"** → same repo (`meghamegs-lab/agentForge`), same branch

**Settings → General:**

- Name: `fortio-agent`
- Root Directory: `apps/agent` ← IMPORTANT
- Branch: `main`
- Repo: `meghamegs-lab/agentForge`

**Settings → Build:**

- Builder: `Dockerfile`
- Dockerfile path: `Dockerfile`

**Settings → Deploy:**

- Health check path: `/health`

### Agent Environment Variables

```bash
# LLM keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...              # fallback only, optional

# Ghostfolio connection
# Use Railway internal URL so traffic stays within the project (free, fast)
GHOSTFOLIO_BASE_URL=http://ghostfolio.railway.internal:3333
# Get this token AFTER Ghostfolio is deployed:
#   1. Open the Ghostfolio public URL
#   2. Create account → Account → Access tab → Security Token section
GHOSTFOLIO_ACCESS_TOKEN=your_ghostfolio_security_token

# Observability (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=fortio-agent

# Thresholds (optional — defaults are fine)
PORTFOLIO_CONCENTRATION_THRESHOLD=0.20
MARKET_DATA_FRESHNESS_MINUTES=15
MAX_TOOL_RETRIES=2
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

## Step 6 — Deploy Order

Deploy in this order (dependencies first):

1. **Postgres** — auto-deploys when provisioned ✅
2. **Redis** — auto-deploys when provisioned ✅
3. **Ghostfolio** — deploy → wait for health check to pass
4. **Fortio Agent** — deploy LAST (needs Ghostfolio URL to be live first,
   and you need the Ghostfolio security token before the agent will work)

---

## Step 7 — Get the Ghostfolio Access Token (after deploy)

Once Ghostfolio is running on Railway:

```bash
# 1. Open the Ghostfolio Railway public URL
# 2. Click "Get Started" → create admin account
# 3. Avatar → My Account → Access tab (🔑 icon) → Security Token section
#    Enter your current token and click "Generate" — copy the new token

# 4. Verify the token works:
curl -X POST https://ghostfolio-production.up.railway.app/api/v1/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"accessToken": "YOUR_COPIED_TOKEN"}'
# You should get back { "authToken": "..." }

# 5. Add it to the Fortio Agent service environment variables:
#    GHOSTFOLIO_ACCESS_TOKEN=YOUR_COPIED_TOKEN
# 6. Redeploy the agent service
```

---

## Step 8 — Verify Everything Works

```bash
# Ghostfolio health
curl https://ghostfolio-production.up.railway.app/api/v1/health

# Fortio Agent health
curl https://fortio-agent-production.up.railway.app/health

# Agent API docs
open https://fortio-agent-production.up.railway.app/docs

# Test a chat call
curl -X POST https://fortio-agent-production.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does my portfolio look like?", "user_id": "test"}'
```

---

## Service URLs Summary

| Service      | Internal URL                         | Public URL                                       |
| ------------ | ------------------------------------ | ------------------------------------------------ |
| Ghostfolio   | `ghostfolio.railway.internal:3333`   | `https://ghostfolio-production.up.railway.app`   |
| Fortio Agent | `fortio-agent.railway.internal:8001` | `https://fortio-agent-production.up.railway.app` |
| Postgres     | `postgres.railway.internal:5432`     | (private only)                                   |
| Redis        | `redis.railway.internal:6379`        | (private only)                                   |

---

## CI/CD — GitHub Actions Auto-Deploy

The workflow at `.github/workflows/ci-deploy.yml` runs automatically.

```
Push to main     →  test-agent + test-ghostfolio  →  deploy to Railway
Pull Request     →  test-agent + test-ghostfolio  (blocks merge on failure)
```

> ⚠️ Only pushes to `main` trigger deploys. No other branches run CI.

### Step A — Disable Railway Auto-Deploy

In Railway dashboard, for **each** service (ghostfolio, fortio-agent):

> Settings → General → Auto Deploy → **Off**

This ensures Railway only deploys when GitHub Actions says tests passed.

### Step B — Create a Railway API Token

1. railway.app → top-right avatar → **Account Settings**
2. **Tokens** tab → **Create Token**
3. Give it a name: `github-actions-deploy`
4. Copy the token (shown once)

### Step C — Get Your Project ID

1. railway.app → open your `agentforge` project
2. **Settings** (top-right gear) → **General**
3. Copy the **Project ID** (UUID format)

### Step D — Add GitHub Secrets

In your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret name          | Value                      |
| -------------------- | -------------------------- |
| `RAILWAY_TOKEN`      | The token from Step B      |
| `RAILWAY_PROJECT_ID` | The project ID from Step C |

### Step E — Enable the Production Environment (optional but recommended)

In GitHub repo → **Settings** → **Environments** → **New environment** → name it `production`.

You can add protection rules (e.g. require a reviewer to approve before deploying).
The workflow already uses `environment: production` — Railway deploy URL will show in your PR checks.

---

## If You Renamed the Repo

If Railway still shows the old repo name (e.g. `agentFortio`):

1. Go to Railway → your project → **each service** → **Settings** tab
2. Under **Source** → click **Disconnect**
3. Reconnect to `meghamegs-lab/agentForge`
4. Set **Branch** to `main`
5. Repeat for both `ghostfolio` and `fortio-agent` services

> GitHub automatically redirects old repo URLs, but Railway's UI won't update until you reconnect.

---

## Cost Estimate

Railway free tier gives $5/month credit. For full stack:

| Service      | Estimated cost          |
| ------------ | ----------------------- |
| Ghostfolio   | ~$5-8/month (512MB RAM) |
| Fortio Agent | ~$3-5/month (256MB RAM) |
| Postgres     | ~$1-2/month             |
| Redis        | ~$1-2/month             |
| **Total**    | **~$10-17/month**       |

Use Railway's **Hobby plan ($5/month)** which includes $5 credit — you'll pay ~$5-12 net.

---

## Troubleshooting

**Agent can't reach Ghostfolio (connection refused):**

- Check `GHOSTFOLIO_BASE_URL` uses internal URL: `http://ghostfolio.railway.internal:3333`
- Make sure both services are in the same Railway project

**Ghostfolio won't start (DB errors):**

- Check `DATABASE_URL` format: `postgresql://user:pass@host:port/db`
- Ensure Postgres service is healthy before deploying Ghostfolio

**Agent auth errors (401 from Ghostfolio):**

- `GHOSTFOLIO_ACCESS_TOKEN` is the security token, not the bearer token
- Get it from: Ghostfolio → Account → **Access tab** (🔑) → Security Token section
- The agent client exchanges it for a bearer token automatically

**Railway build fails for agent:**

- Make sure Root Directory is set to `apps/agent` in service settings
- The Dockerfile path should be `Dockerfile` (relative to `apps/agent/`)

**Railway still shows old repo name (`agentFortio`):**

- See "If You Renamed the Repo" section above
- GitHub redirects old URLs but Railway needs to be manually reconnected
