# Bounty Submission — FIRE Goal Tracker

**Project:** Fortio (Ghostfolio AI Agent)
**Feature:** FIRE Goal Tracker + FRED Economic Data Integration
**Author:** Megha Shanthraj
**Date:** March 2026

---

## The Customer

**Self-directed FIRE (Financial Independence, Retire Early) investors** who obsessively track
their portfolio in Ghostfolio — but have no way to answer the most important question of all:
_"Am I on track to retire?"_

**Specific persona:**

> 34-year-old software engineer, $500K invested in index funds, contributing $2K/month.
> Uses Ghostfolio to track holdings and performance. Wants to retire at 50 but has no idea
> if that goal is achievable — or how inflation affects their timeline.

**Why they feel the pain:**  
Ghostfolio tells them _where they are_ (portfolio value, returns, fees). It cannot tell them
_where they're going_. Answering "Can I retire at 50?" requires cross-referencing portfolio
value, monthly contributions, expected returns, _and_ real-world inflation — work that today
takes a spreadsheet and 30 minutes. Fortio does it in one sentence.

---

## The Feature

**FIRE Goal Tracker** — a retirement planning add-on for Fortio that lets users set a
retirement goal once and ask natural-language questions about it in every subsequent turn.

### What users can now ask:

| Question                                                 | Tools invoked                                 |
| -------------------------------------------------------- | --------------------------------------------- |
| "Set my retirement goal — retire at 50, spend $80K/year" | `set_retirement_goal`                         |
| "What's my FIRE number?"                                 | `get_retirement_goal`                         |
| "Am I on track to retire?"                               | `get_fire_progress` + `get_portfolio_summary` |
| "When can I retire at my current savings rate?"          | `calculate_retirement_projection` + FRED      |
| "What if I save $3,000/month instead?"                   | `calculate_retirement_projection` (override)  |
| "How does today's inflation affect my timeline?"         | `get_macro_data` + FRED                       |
| "What's the current 10-year Treasury yield?"             | `get_macro_data`                              |

### Multi-turn conversation example:

```
Turn 1:
  User: "I want to retire at 50. I'm 35, plan to spend $80K/year."
  Agent: calls set_retirement_goal(current_age=35, target_retirement_age=50,
                                   target_annual_spending=80000)
         → "✅ Goal saved! Your FIRE number is $2,000,000 (using the 4% rule).
            You have 15 years to reach it."

Turn 2:
  User: "Am I on track?"
  Agent: calls get_fire_progress(user_id=...) + get_portfolio_summary
         → "You're 35% of the way there — $700K of your $2M FIRE number.
            Shortfall: $1.3M."

Turn 3:
  User: "When can I retire?"
  Agent: calls calculate_retirement_projection + FRED for real inflation
         → "At $2K/month contributions and 7% returns (3.68% real after today's
            3.2% inflation), you'll hit $2M in ~13.8 years — age 48.8. You're
            ahead of schedule! 🎉"

Turn 4:
  User: "What if I save $500 more per month?"
  Agent: calls calculate_retirement_projection(monthly_contribution_override=2500)
         → "Saving $2,500/month cuts 1.4 years off — you'd retire at 47.4."
```

---

## The Data Source

**FRED API — Federal Reserve Economic Data**  
Maintained by the Federal Reserve Bank of St. Louis.  
URL: `https://api.stlouisfed.org/fred/`  
Free API key: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

### Series integrated:

| FRED Series | Name                                       | Used for                         |
| ----------- | ------------------------------------------ | -------------------------------- |
| `CPIAUCSL`  | Consumer Price Index (seasonally adjusted) | Year-over-year inflation rate    |
| `DGS10`     | 10-Year US Treasury Constant Maturity Rate | Risk-free rate benchmark for SWR |

### Why FRED specifically:

- **Authoritative** — data comes directly from the Federal Reserve
- **Free** — no credit card, no tiers, 120 req/min rate limit
- **Reliable** — 99.9%+ uptime, data updated monthly (CPI) / daily (DGS10)
- **Relevant** — inflation directly impacts how much a retirement portfolio is worth in real terms

### How it's used:

The `calculate_retirement_projection` tool fetches CPI and DGS10 to:

1. Compute the **real return**: `(1 + nominal_return) / (1 + inflation) − 1`
2. Project the **inflation-adjusted FIRE number** (grows with inflation over time)
3. Provide context: "Is the 4% rule still valid given current rates?"

---

## Stateful Data

**Table: `retirement_goals` in Postgres (same Railway instance as LangGraph checkpoints)**

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

**CRUD operations:**

| Operation  | HTTP Route                               | Agent Tool                        |
| ---------- | ---------------------------------------- | --------------------------------- |
| **Create** | `POST /api/goals/retirement`             | `set_retirement_goal`             |
| **Read**   | `GET /api/goals/retirement/{user_id}`    | `get_retirement_goal`             |
| **Update** | `PUT /api/goals/retirement/{user_id}`    | `set_retirement_goal` (upsert)    |
| **Delete** | `DELETE /api/goals/retirement/{user_id}` | (API only — agent doesn't delete) |

---

## The Impact

### Quantifiable value:

- The average FIRE-focused investor needs their portfolio to last 40–50 years
- A 1-year delay in retirement costs ~$80–160K in forgone investment gains
- Knowing _exactly_ when you can retire (vs. guessing) is worth thousands of dollars per year
- The 4% rule in a 4.5% Treasury environment deserves re-examination — Fortio surfaces this automatically

### Why Ghostfolio didn't have this:

Ghostfolio is a portfolio _tracker_. It shows you where you are. It deliberately doesn't try to answer
"where are you going" — that requires user-specific goals (stateful!) and macroeconomic context (FRED!).
This is exactly the gap Fortio fills: bridging live portfolio data + user intent + macro context.

### Why this wins:

1. **Real customer pain** — 15M+ FIRE-curious investors lack a natural-language retirement planner
2. **Real data source** — FRED is the authoritative, free, government-backed macro data feed
3. **Full CRUD** — Goals persist across sessions via Postgres; agent reads and updates them
4. **Multi-turn coherence** — "What if I save $500 more?" resolves correctly because the goal is
   persisted and conversation history is checkpointed
5. **Reliable agent** — 57+ tests (unit + eval), verification flags for speculative projections,
   graceful FRED fallback when unavailable, never raises exceptions to the user

---

## Technical Implementation

### Files created:

| File                                        | Purpose                                        |
| ------------------------------------------- | ---------------------------------------------- |
| `agent/clients/fred.py`                     | Async FRED API client with tenacity retry      |
| `agent/db/__init__.py`                      | DB package init                                |
| `agent/db/retirement.py`                    | Postgres CRUD via psycopg2 + asyncio.to_thread |
| `agent/tools/retirement.py`                 | 5 LangChain `@tool` functions                  |
| `tests/unit/clients/test_fred_client.py`    | 20 FRED client unit tests                      |
| `tests/unit/tools/test_retirement_tools.py` | 37 tool unit tests                             |
| `tests/evals/test_retirement_eval.py`       | 35+ eval tests (10 categories)                 |

### Files modified:

| File                      | Change                                                   |
| ------------------------- | -------------------------------------------------------- |
| `agent/config.py`         | Added `fire_tracker_enabled`, `fred_api_key`             |
| `agent/tools/__init__.py` | Conditional tool registration when feature flag is on    |
| `agent/api/schemas.py`    | Added `RetirementGoalRequest`, `RetirementGoalResponse`  |
| `agent/api/main.py`       | 4 CRUD routes + `create_table_if_not_exists` in lifespan |

### Feature flag:

```bash
# Enable in .env:
FIRE_TRACKER_ENABLED=true
FRED_API_KEY=your_key_here   # free at fred.stlouisfed.org
```

When `false` (default): zero tools registered, zero DB tables created, zero FRED calls made.
Existing deployments are **completely unaffected**.

### External API calls per response:

| User question        | FRED calls          | Ghostfolio calls | DB calls |
| -------------------- | ------------------- | ---------------- | -------- |
| "Set my goal"        | 0                   | 0                | 1 UPSERT |
| "What's my goal?"    | 0                   | 0                | 1 SELECT |
| "Am I on track?"     | 0                   | 1                | 1 SELECT |
| "When can I retire?" | **2** (CPI + DGS10) | 1                | 1 SELECT |
| "What's inflation?"  | **2** (CPI + DGS10) | 0                | 0        |

### Observability:

- All tool calls traced in LangSmith under `fortio-agent` project
- DB operations emit structlog events: `retirement_goal_saved`, `retirement_goal_deleted`
- Verification pipeline adds `FIRE_PROJECTION_SPECULATIVE` flag (LOW severity) on every projection response
- If FRED is unavailable: `macro_data_status: "default"` in tool output; macro note appended to response
