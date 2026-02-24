# AgentForge Skills — Cursor IDE Reference
# Place this file at: .cursor/skills/SKILLS.md
# Reference it in chat: @SKILLS.md "build me a new tool following project patterns"

## What This Project Is

A Python AI agent (LangGraph + Claude Sonnet) that provides financial portfolio analysis
by connecting to the Ghostfolio REST API. The agent chains multiple tool calls,
runs a 5-stage verification pipeline on every response, and surfaces insights
that Ghostfolio's UI doesn't expose in natural language.

## Skill: Creating a New LangGraph Tool

Pattern to follow: any existing file in `agent/tools/`

```python
# Template for every new tool
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import Any
from langchain_core.tools import tool
from agent.clients.ghostfolio import GhostfolioClient, GhostfolioError

@tool
def your_tool_name(param: str = "default") -> dict[str, Any]:
    """
    One-line description of what this tool does.
    Use when users ask: [list 3-4 example queries].

    Args:
        param: Description of parameter and valid values.
    Returns:
        Description of returned dict fields.
    """
    return asyncio.run(_your_tool_name(param))

async def _your_tool_name(param: str) -> dict[str, Any]:
    try:
        async with GhostfolioClient() as client:
            data = await client.some_method()
        # process data...
        return {
            "status": "ok",
            # your fields...
            "data_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except GhostfolioError as e:
        return {"status": "error", "error": e.message}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

**Rules:**
- Always add `data_timestamp` — verification layer needs it
- Always return `{"status": "error", "error": "..."}` on failure — never raise
- Always use `asyncio.run()` wrapper — LangChain tools are synchronous
- Register in `agent/tools/__init__.py` → `ALL_TOOLS` list
- Write failing test first in `tests/unit/tools/test_tools.py`

## Skill: Adding a Ghostfolio API Client Method

Add to `agent/clients/ghostfolio.py`:

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def your_new_method(self, param: str) -> dict[str, Any]:
    bearer = await self._get_bearer_token()
    resp = await self._client.get(
        f"{self.base_url}/api/v1/your-endpoint",
        params={"key": param},
        headers=self._auth_headers(bearer),
    )
    if resp.status_code != 200:
        raise GhostfolioError(resp.status_code, resp.text)
    return resp.json()
```

## Skill: Writing TDD Tool Tests

Pattern from `tests/unit/tools/test_tools.py`:

```python
import respx, httpx, pytest

BASE_URL = "https://ghostfol.io"
AUTH = {"authToken": "test-token"}

class TestYourTool:
    @respx.mock
    async def test_happy_path(self):
        respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(200, json=AUTH))
        respx.get(f"{BASE_URL}/api/v1/your-endpoint").mock(
            return_value=httpx.Response(200, json={"your": "data"}))
        result = await _your_tool_name("param")
        assert result["status"] == "ok"
        assert "data_timestamp" in result

    @respx.mock
    async def test_network_error_returns_dict_not_exception(self):
        respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(200, json=AUTH))
        respx.get(f"{BASE_URL}/api/v1/your-endpoint").mock(
            return_value=httpx.Response(500, text="Server Error"))
        result = await _your_tool_name("param")
        assert result["status"] == "error"
        assert "error" in result
```

## Skill: Adding a Verification Check

Add to `agent/verification/__init__.py`:

```python
def check_your_rule(
    response: str, tool_results: list[dict]
) -> tuple[str, list[VerificationFlag]]:
    """Your check description."""
    flags = []
    # your logic
    if condition:
        flags.append({
            "type": "YOUR_FLAG_TYPE",
            "severity": "HIGH",  # HIGH, MEDIUM, LOW, INFO
            "message": "Human-readable description",
        })
    return response, flags
```

Then add to `run_verification_pipeline()`:
```python
response, flags = check_your_rule(response, tool_results)
all_flags.extend(flags)
```

## Skill: LangGraph Node Pattern

```python
async def your_node(state: AgentState) -> dict[str, Any]:
    """
    Nodes ONLY return the state keys they modify.
    Never return the full state.
    """
    # do something with state
    result = process(state["messages"])
    return {
        "messages": [result],        # only keys you changed
        "reasoning_steps": state.get("reasoning_steps", 0) + 1,
    }
```

## Skill: Proactive Tool (fires on session start)

The proactive monitor fires from `on_chat_start` in Chainlit:

```python
@cl.on_chat_start
async def on_chat_start():
    prev_snap = cl.user_session.get("portfolio_snapshot_json", "")
    result = get_proactive_risk_monitor.invoke({"previous_snapshot_json": prev_snap})
    if result.get("current_snapshot"):
        cl.user_session.set("portfolio_snapshot_json",
                            json.dumps(result["current_snapshot"]))
    # surface alerts to user...
```

## Common Mistakes (Cursor: watch for these in generated code)

| Wrong | Correct |
|-------|---------|
| `requests.get(...)` | `httpx.AsyncClient` |
| `print(f"debug")` | `import structlog; log = structlog.get_logger()` |
| `raise Exception(...)` in tools | `return {"status": "error", "error": str(e)}` |
| `from langchain.agents import AgentExecutor` | `from langgraph.graph import StateGraph` |
| Hardcoded `api_key="sk-..."` | `from agent.config import settings; settings.anthropic_api_key` |
| `import tool from langchain.tools` | `from langchain_core.tools import tool` |
| Missing `data_timestamp` in return | Always include it |
| Sync tool without `asyncio.run()` | Wrap async fn with `asyncio.run(_fn())` |

## Environment Variables Quick Reference

```
ANTHROPIC_API_KEY      → console.anthropic.com
OPENAI_API_KEY         → platform.openai.com (fallback)
GHOSTFOLIO_BASE_URL    → http://localhost:3333 (local) or https://ghostfol.io (demo)
GHOSTFOLIO_ACCESS_TOKEN → from POST /api/v1/auth/anonymous
LANGCHAIN_API_KEY      → smith.langchain.com
DATABASE_URL           → postgresql://postgres:postgres@localhost:5432/agentforge
REDIS_URL              → redis://localhost:6379/0
```

## Ghostfolio Endpoints Quick Reference

```
GET  /api/v1/portfolio/holdings         → all positions + sectors + countries
GET  /api/v1/portfolio/performance      → YTD/1Y/max returns (param: range)
GET  /api/v1/order                      → all transactions (params: accountId, dateFrom, dateTo)
GET  /api/v1/portfolio/investments      → cashflow timeline (param: groupBy)
GET  /api/v1/portfolio/dividends        → dividend income (params: groupBy, range)
GET  /api/v1/portfolio/holding/:ds/:sym → deep dive on one position (cost basis, history)
GET  /api/v1/portfolio/details          → full details + fireWealth + netWorth
GET  /api/v1/benchmark                  → benchmarks + market condition (BULL/BEAR/NEUTRAL)
GET  /api/v1/accounts                   → all accounts + types
GET  /api/v1/symbol/lookup?query=X      → search for any ticker
GET  /api/v1/symbol/:ds/:sym            → asset profile (sector, country, currency)
POST /api/v1/auth/anonymous             → exchange token for JWT bearer
```
