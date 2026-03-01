<div align="center">

[<img src="https://avatars.githubusercontent.com/u/82473144?s=200" width="100" alt="Ghostfolio logo">](https://ghostfol.io)

# Ghostfolio + Fortio

**Open Source Wealth Management Software · AI Finance Agent**

[**Ghostfol.io**](https://ghostfol.io) | [**Live Demo**](https://ghostfol.io/en/demo) | [**Ghostfolio Premium**](https://ghostfol.io/en/pricing) | [**FAQ**](https://ghostfol.io/en/faq) |
[**Blog**](https://ghostfol.io/en/blog) | [**LinkedIn**](https://www.linkedin.com/company/ghostfolio) | [**Slack**](https://join.slack.com/t/ghostfolio/shared_invite/zt-vsaan64h-F_I0fEo5M0P88lP9ibCxFg) | [**X**](https://x.com/ghostfolio_)

[![Shield: Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-Support-yellow?logo=buymeacoffee)](https://www.buymeacoffee.com/ghostfolio)
[![Shield: Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-limegreen.svg)](#contributing) [![Shield: Docker Pulls](https://img.shields.io/docker/pulls/ghostfolio/ghostfolio?label=Docker%20Pulls)](https://hub.docker.com/r/ghostfolio/ghostfolio)
[![Shield: License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-orange.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

### 🚀 Production Deployments (Railway)

| Service                                           | URL                                                                                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Ghostfolio App**                                | [ghostfolio-production.up.railway.app](https://ghostfolio-production-453e.up.railway.app/en/home)                                     |
| **Ghostfolio Demo Account to test Fortio Agent ** | [fortio-agent-production.up.railway.app](https://ghostfolio-production-453e.up.railway.app/en/p/e6b67d66-b727-4fa8-a3ac-01ec36d5dde1) |
| **Fortio Agent Docs**                             | [fortio-agent-production.up.railway.app/docs](https://fortio-agent-production.up.railway.app/docs)                                    |

</div>

**Ghostfolio** is an open source wealth management software built with web technology. The application empowers busy people to keep track of stocks, ETFs or cryptocurrencies and make solid, data-driven investment decisions. The software is designed for personal use in continuous operation.

<div align="center">

[<img src="./apps/client/src/assets/images/video-preview.jpg" width="600" alt="Preview image of the Ghostfolio video trailer">](https://www.youtube.com/watch?v=yY6ObSQVJZk)

</div>

## Fortio — AI Finance Agent

**Fortio** is an AI-powered portfolio assistant built on top of Ghostfolio, added to this fork. It uses [LangGraph](https://github.com/langchain-ai/langgraph) + Claude Sonnet as a stateful agent that reads your Ghostfolio portfolio data and answers natural-language questions about it.

### What Fortio Adds

| Layer             | Details                                                                                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Agent**         | LangGraph `StateGraph` + Claude Sonnet (primary) + GPT-4o (fallback)                                                                                                                                               |
| **Tools**         | 11 domain tools — 5 core (portfolio, performance, transactions, diversification, market) + 6 advanced (fee drag, health scorecard, rebalancing plan, market context, transaction patterns, proactive risk monitor) |
| **Verification**  | 5-stage pipeline: disclaimer injection · hallucination guard · data freshness · concentration risk · confidence scoring                                                                                            |
| **API**           | FastAPI REST endpoint (`POST /api/chat`) with multi-turn conversation history via Postgres checkpointing                                                                                                           |
| **UI**            | Embedded Angular chat widget in Ghostfolio (powered by the FastAPI REST endpoint)                                                                                                                                  |
| **Observability** | LangSmith tracing                                                                                                                                                                                                  |
| **Deployment**    | Railway (CI/CD via GitHub Actions)                                                                                                                                                                                 |

### Agent Architecture

```
reasoning → tools → collect_results → reasoning (loop) → verify → END
```

State is persisted per `conversation_id` in Postgres using LangGraph's `AsyncPostgresSaver`, so the LLM sees the full conversation history on every turn.

### Fortio Quick Start

See [`apps/agent/SETUP.md`](./apps/agent/SETUP.md) for the full 10-step setup guide. In brief:

```bash
cd apps/agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .                        # installs the `fortio` CLI command
cp .env.example .env                    # fill in ANTHROPIC_API_KEY + GHOSTFOLIO_ACCESS_TOKEN
fortio serve                            # FastAPI docs → http://localhost:8001/docs
```

### CLI — `fortio` Command

After `pip install -e .` the `fortio` CLI is available:

| Command                    | What it does                                                           |
| -------------------------- | ---------------------------------------------------------------------- |
| `fortio ask "…"`           | Single question, prints answer, exits                                  |
| `fortio ask "…" --verbose` | Same, plus tools called and confidence level                           |
| `fortio chat`              | Interactive multi-turn REPL with `/help`, `/tools`, `/clear`           |
| `fortio serve`             | Start the FastAPI server on port 8001                                  |
| `fortio serve --reload`    | Dev mode with hot-reload                                               |
| `fortio mcp`               | Start the MCP server (exposes all 11 tools to Claude Desktop / Cursor) |
| `fortio demo`              | Run all 11 tools in sequence — one-command proof of functionality      |
| `fortio version`           | Show active model, environment, and checkpoint config                  |

```bash
# Examples
fortio ask "What does my portfolio look like?"
fortio chat
fortio demo
fortio serve --reload
```

### MCP — Claude Desktop & Cursor Integration

Fortio exposes all 11 portfolio tools as an MCP server, letting Claude Desktop and Cursor
query your Ghostfolio data directly in their chat interfaces — no browser, no API keys in a chat box.

**Start the MCP server:**

```bash
fortio mcp
# stdout = MCP JSON-RPC wire; all human-readable output goes to stderr
```

**Claude Desktop** — edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

**Cursor** — Settings → MCP → Add server → paste the same JSON block.

> **Using Railway?** Replace `http://localhost:3333` with your Railway Ghostfolio URL.

The MCP server also exposes:

- **3 resources** — `portfolio://summary`, `portfolio://performance`, `portfolio://health`
- **1 prompt template** — `portfolio-analysis` with focus options: `risk` · `performance` · `fees` · `all`

### Eval Suite

The agent ships with **60+ evaluation tests** (all mocked — zero real API calls) plus a LangSmith experiment suite.

| Eval file                                                                                       | What it tests                                                                   | Tests |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----- |
| [`tests/evals/test_correctness.py`](./apps/agent/tests/evals/test_correctness.py)               | Arithmetic accuracy, % conversions, sort order, fee sums, sector rollup         | 12    |
| [`tests/evals/test_tool_selection.py`](./apps/agent/tests/evals/test_tool_selection.py)         | Tool docstring trigger keywords, domain boundary isolation, parameter mapping   | 10    |
| [`tests/evals/test_llm_tool_selection.py`](./apps/agent/tests/evals/test_llm_tool_selection.py) | LLM-driven tool selection routing and keyword coverage                          | 14    |
| [`tests/evals/test_tool_execution.py`](./apps/agent/tests/evals/test_tool_execution.py)         | Advanced tool happy path + error cases for all 6 advanced tools                 | 16    |
| [`tests/evals/test_multi_step.py`](./apps/agent/tests/evals/test_multi_step.py)                 | Cross-tool consistency, referential integrity, multi-session monitor            | 12    |
| [`tests/evals/test_edge_cases.py`](./apps/agent/tests/evals/test_edge_cases.py)                 | Dict vs list holdings, zero-value holdings, unicode names, large portfolios     | 10    |
| [`tests/evals/test_adversarial.py`](./apps/agent/tests/evals/test_adversarial.py)               | Prompt injection, jailbreaks, off-topic deflection, fabricated number detection | 12    |
| [`tests/evals/ls_evals.py`](./apps/agent/tests/evals/ls_evals.py)                               | LangSmith tracked experiments: correctness, safety, latency, consistency        | 23    |

Run the evals:

```bash
cd apps/agent

# All eval tests (fast, ~5–10 s, no network required)
pytest tests/evals/ -v

# All unit tests (mocked, no network)
pytest tests/unit/ -v

# Adversarial / safety tests
pytest tests/adversarial/ -v

# Full suite with coverage report
pytest tests/unit/ tests/evals/ --cov=agent --cov-report=term-missing

# LangSmith experiments (requires LANGCHAIN_API_KEY in .env)
python tests/evals/ls_evals.py
python tests/evals/ls_evals.py --only correctness
python tests/evals/ls_evals.py --only safety
```

---

## Ghostfolio Premium

Our official **[Ghostfolio Premium](https://ghostfol.io/en/pricing)** cloud offering is the easiest way to get started. Due to the time it saves, this will be the best option for most people. Revenue is used to cover operational costs for the hosting infrastructure and professional data providers, and to fund ongoing development.

If you prefer to run Ghostfolio on your own infrastructure, please find further instructions in the [Self-hosting](#self-hosting) section.

## Why Ghostfolio?

Ghostfolio is for you if you are...

- 💼 trading stocks, ETFs or cryptocurrencies on multiple platforms
- 🏦 pursuing a buy & hold strategy
- 🎯 interested in getting insights of your portfolio composition
- 👻 valuing privacy and data ownership
- 🧘 into minimalism
- 🧺 caring about diversifying your financial resources
- 🆓 interested in financial independence
- 🙅 saying no to spreadsheets
- 😎 still reading this list

## Features

- ✅ Create, update and delete transactions
- ✅ Multi account management
- ✅ Portfolio performance: Return on Average Investment (ROAI) for `Today`, `WTD`, `MTD`, `YTD`, `1Y`, `5Y`, `Max`
- ✅ Various charts
- ✅ Static analysis to identify potential risks in your portfolio
- ✅ Import and export transactions
- ✅ Dark Mode
- ✅ Zen Mode
- ✅ Progressive Web App (PWA) with a mobile-first design

<div align="center">

<img src="./apps/client/src/assets/images/screenshot.png" width="300" alt="Image of a phone showing the Ghostfolio app open">

</div>

## Technology Stack

Ghostfolio is a modern web application written in [TypeScript](https://www.typescriptlang.org) and organized as an [Nx](https://nx.dev) workspace.

### Backend

The backend is based on [NestJS](https://nestjs.com) using [PostgreSQL](https://www.postgresql.org) as a database together with [Prisma](https://www.prisma.io) and [Redis](https://redis.io) for caching.

### Frontend

The frontend is built with [Angular](https://angular.dev) and uses [Angular Material](https://material.angular.io) with utility classes from [Bootstrap](https://getbootstrap.com).

## Self-hosting

We provide official container images hosted on [Docker Hub](https://hub.docker.com/r/ghostfolio/ghostfolio) for `linux/amd64`, `linux/arm/v7` and `linux/arm64`.

<div align="center">

[<img src="./apps/client/src/assets/images/button-buy-me-a-coffee.png" width="150" alt="Buy me a coffee button"/>](https://www.buymeacoffee.com/ghostfolio)

</div>

### Supported Environment Variables

| Name                        | Type                  | Default Value         | Description                                                                                                                         |
| --------------------------- | --------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `ACCESS_TOKEN_SALT`         | `string`              |                       | A random string used as salt for access tokens                                                                                      |
| `API_KEY_COINGECKO_DEMO`    | `string` (optional)   |                       | The _CoinGecko_ Demo API key                                                                                                        |
| `API_KEY_COINGECKO_PRO`     | `string` (optional)   |                       | The _CoinGecko_ Pro API key                                                                                                         |
| `DATABASE_URL`              | `string`              |                       | The database connection URL, e.g. `postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}?sslmode=prefer` |
| `ENABLE_FEATURE_AUTH_TOKEN` | `boolean` (optional)  | `true`                | Enables authentication via security token                                                                                           |
| `HOST`                      | `string` (optional)   | `0.0.0.0`             | The host where the Ghostfolio application will run on                                                                               |
| `JWT_SECRET_KEY`            | `string`              |                       | A random string used for _JSON Web Tokens_ (JWT)                                                                                    |
| `LOG_LEVELS`                | `string[]` (optional) |                       | The logging levels for the Ghostfolio application, e.g. `["debug","error","log","warn"]`                                            |
| `PORT`                      | `number` (optional)   | `3333`                | The port where the Ghostfolio application will run on                                                                               |
| `POSTGRES_DB`               | `string`              |                       | The name of the _PostgreSQL_ database                                                                                               |
| `POSTGRES_PASSWORD`         | `string`              |                       | The password of the _PostgreSQL_ database                                                                                           |
| `POSTGRES_USER`             | `string`              |                       | The user of the _PostgreSQL_ database                                                                                               |
| `REDIS_DB`                  | `number` (optional)   | `0`                   | The database index of _Redis_                                                                                                       |
| `REDIS_HOST`                | `string`              |                       | The host where _Redis_ is running                                                                                                   |
| `REDIS_PASSWORD`            | `string`              |                       | The password of _Redis_                                                                                                             |
| `REDIS_PORT`                | `number`              |                       | The port where _Redis_ is running                                                                                                   |
| `REQUEST_TIMEOUT`           | `number` (optional)   | `2000`                | The timeout of network requests to data providers in milliseconds                                                                   |
| `ROOT_URL`                  | `string` (optional)   | `http://0.0.0.0:3333` | The root URL of the Ghostfolio application, used for generating callback URLs and external links.                                   |

#### OpenID Connect OIDC (Experimental)

| Name                       | Type                  | Default Value                        | Description                                                                                          |
| -------------------------- | --------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `ENABLE_FEATURE_AUTH_OIDC` | `boolean` (optional)  | `false`                              | Enables authentication via _OpenID Connect_                                                          |
| `OIDC_AUTHORIZATION_URL`   | `string` (optional)   |                                      | Manual override for the OIDC authorization endpoint (falls back to the discovery from the issuer)    |
| `OIDC_CALLBACK_URL`        | `string` (optional)   | `${ROOT_URL}/api/auth/oidc/callback` | The OIDC callback URL                                                                                |
| `OIDC_CLIENT_ID`           | `string`              |                                      | The OIDC client ID                                                                                   |
| `OIDC_CLIENT_SECRET`       | `string`              |                                      | The OIDC client secret                                                                               |
| `OIDC_ISSUER`              | `string`              |                                      | The OIDC issuer URL, used to discover the OIDC configuration via `/.well-known/openid-configuration` |
| `OIDC_SCOPE`               | `string[]` (optional) | `["openid"]`                         | The OIDC scope to request, e.g. `["email","openid","profile"]`                                       |
| `OIDC_TOKEN_URL`           | `string` (optional)   |                                      | Manual override for the OIDC token endpoint (falls back to the discovery from the issuer)            |
| `OIDC_USER_INFO_URL`       | `string` (optional)   |                                      | Manual override for the OIDC user info endpoint (falls back to the discovery from the issuer)        |

### Run with Docker Compose

#### Prerequisites

- Basic knowledge of Docker
- Installation of [Docker](https://www.docker.com/products/docker-desktop)
- Create a local copy of this Git repository (clone)
- Copy the file `.env.example` to `.env` and populate it with your data (`cp .env.example .env`)

#### a. Run environment

Run the following command to start the Docker images from [Docker Hub](https://hub.docker.com/r/ghostfolio/ghostfolio):

```bash
docker compose -f docker/docker-compose.yml up -d
```

#### b. Build and run environment

Run the following commands to build and start the Docker images:

```bash
docker compose -f docker/docker-compose.build.yml build
docker compose -f docker/docker-compose.build.yml up -d
```

#### Setup

1. Open http://localhost:3333 in your browser
1. Create a new user via _Get Started_ (this first user will get the role `ADMIN`)

#### Upgrade Version

1. Update the _Ghostfolio_ Docker image
   - Increase the version of the `ghostfolio/ghostfolio` Docker image in `docker/docker-compose.yml`
   - Run the following command if `ghostfolio:latest` is set:
     ```bash
     docker compose -f docker/docker-compose.yml pull
     ```

1. Run the following command to start the new Docker image:
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```
   The container will automatically apply any required database schema migrations during startup.

### Home Server Systems (Community)

Ghostfolio is available for various home server systems, including [CasaOS](https://github.com/bigbeartechworld/big-bear-casaos), [Home Assistant](https://github.com/lildude/ha-addon-ghostfolio), [Runtipi](https://www.runtipi.io/docs/apps-available), [TrueCharts](https://truecharts.org/charts/stable/ghostfolio), [Umbrel](https://apps.umbrel.com/app/ghostfolio), and [Unraid](https://unraid.net/community/apps?q=ghostfolio).

## Development

For detailed information on the environment setup and development process, please refer to [DEVELOPMENT.md](./DEVELOPMENT.md).

For the **Fortio Agent** (Python service), see [`apps/agent/README.md`](./apps/agent/README.md) and [`apps/agent/SETUP.md`](./apps/agent/SETUP.md). Deployment to Railway is documented in [RAILWAY.md](./RAILWAY.md).

## Public API

### Authorization: Bearer Token

Set the header for each request as follows:

```
"Authorization": "Bearer eyJh..."
```

You can get the _Bearer Token_ via `POST http://localhost:3333/api/v1/auth/anonymous` (Body: `{ "accessToken": "<INSERT_SECURITY_TOKEN_OF_ACCOUNT>" }`)

Deprecated: `GET http://localhost:3333/api/v1/auth/anonymous/<INSERT_SECURITY_TOKEN_OF_ACCOUNT>` or `curl -s http://localhost:3333/api/v1/auth/anonymous/<INSERT_SECURITY_TOKEN_OF_ACCOUNT>`.

### Health Check (experimental)

#### Request

`GET http://localhost:3333/api/v1/health`

**Info:** No Bearer Token is required for health check

#### Response

##### Success

`200 OK`

```
{
  "status": "OK"
}
```

### Import Activities

#### Prerequisites

[Bearer Token](#authorization-bearer-token) for authorization

#### Request

`POST http://localhost:3333/api/v1/import`

#### Body

```
{
  "activities": [
    {
      "currency": "USD",
      "dataSource": "YAHOO",
      "date": "2021-09-15T00:00:00.000Z",
      "fee": 19,
      "quantity": 5,
      "symbol": "MSFT",
      "type": "BUY",
      "unitPrice": 298.58
    }
  ]
}
```

| Field        | Type                | Description                                                         |
| ------------ | ------------------- | ------------------------------------------------------------------- |
| `accountId`  | `string` (optional) | Id of the account                                                   |
| `comment`    | `string` (optional) | Comment of the activity                                             |
| `currency`   | `string`            | `CHF` \| `EUR` \| `USD` etc.                                        |
| `dataSource` | `string`            | `COINGECKO` \| `GHOSTFOLIO` [^1] \| `MANUAL` \| `YAHOO`             |
| `date`       | `string`            | Date in the format `ISO-8601`                                       |
| `fee`        | `number`            | Fee of the activity                                                 |
| `quantity`   | `number`            | Quantity of the activity                                            |
| `symbol`     | `string`            | Symbol of the activity (suitable for `dataSource`)                  |
| `type`       | `string`            | `BUY` \| `DIVIDEND` \| `FEE` \| `INTEREST` \| `LIABILITY` \| `SELL` |
| `unitPrice`  | `number`            | Price per unit of the activity                                      |

#### Response

##### Success

`201 Created`

##### Error

`400 Bad Request`

```
{
  "error": "Bad Request",
  "message": [
    "activities.1 is a duplicate activity"
  ]
}
```

### Portfolio (experimental)

#### Prerequisites

Grant access of type _Public_ in the _Access_ tab of _My Ghostfolio_.

#### Request

`GET http://localhost:3333/api/v1/public/<INSERT_ACCESS_ID>/portfolio`

**Info:** No Bearer Token is required for authorization

#### Response

##### Success

```
{
  "performance": {
    "1d": {
      "relativeChange": 0 // normalized from -1 to 1
    };
    "ytd": {
      "relativeChange": 0 // normalized from -1 to 1
    },
    "max": {
      "relativeChange": 0 // normalized from -1 to 1
    }
  }
}
```

## Community Projects

Discover a variety of community projects for Ghostfolio: https://github.com/topics/ghostfolio

Are you building your own project? Add the `ghostfolio` topic to your _GitHub_ repository to get listed as well. [Learn more →](https://docs.github.com/en/articles/classifying-your-repository-with-topics)

## Contributing

Ghostfolio is **100% free** and **open source**. We encourage and support an active and healthy community that accepts contributions from the public - including you.

Not sure what to work on? We have [some ideas](https://github.com/ghostfolio/ghostfolio/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22%20no%3Aassignee), even for [newcomers](https://github.com/ghostfolio/ghostfolio/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22%20no%3Aassignee). Please join the Ghostfolio [Slack](https://join.slack.com/t/ghostfolio/shared_invite/zt-vsaan64h-F_I0fEo5M0P88lP9ibCxFg) channel or post to [@ghostfolio\_](https://x.com/ghostfolio_) on _X_. We would love to hear from you.

If you like to support this project, become a [**Sponsor**](https://github.com/sponsors/ghostfolio), get [**Ghostfolio Premium**](https://ghostfol.io/en/pricing) or [**Buy me a coffee**](https://www.buymeacoffee.com/ghostfolio).

## Sponsors

<div align="center">
  <a href="https://www.testmuai.com?utm_medium=sponsor&utm_source=ghostfolio" target="_blank" title="TestMu AI - AI Powered Testing Tool">
    <img alt="TestMu AI Logo" height="45" src="https://assets.testmuai.com/resources/images/logos/logo.svg" />
  </a>
</div>

## Analytics

![Alt](https://repobeats.axiom.co/api/embed/281a80b2d0c4af1162866c24c803f1f18e5ed60e.svg 'Repobeats analytics image')

## License

© 2021 - 2026 [Ghostfolio](https://ghostfol.io)

Licensed under the [AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.html).

[^1]: Available with [**Ghostfolio Premium**](https://ghostfol.io/en/pricing).
