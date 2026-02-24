"""
Typed async client for the Ghostfolio REST API.
Handles auth, retries, and structured error responses.

Performance note
----------------
Use `get_shared_client()` inside agent tools instead of creating a new
`GhostfolioClient()` every call.  The singleton keeps the bearer token alive
across every tool invocation in the same process, saving one auth round-trip
(~200-400 ms) per tool call after the first.
"""
from __future__ import annotations

import httpx
from typing import Any
from tenacity import retry, stop_after_attempt, wait_fixed

from agent.config import settings


class GhostfolioError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Ghostfolio API error {status_code}: {message}")


class GhostfolioClient:
    """Async HTTP client for Ghostfolio REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        access_token: str | None = None,
        public_access_id: str | None = None,
    ):
        self.base_url = (base_url or settings.ghostfolio_base_url).rstrip("/")
        self.access_token = access_token or settings.ghostfolio_access_token
        self.public_access_id = public_access_id or settings.ghostfolio_public_access_id
        self._bearer_token: str | None = None
        # Increase pool size; keep-alive avoids TCP handshake on every request
        self._client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def _get_bearer_token(self) -> str:
        """Exchange the security token for a JWT bearer token (cached per instance)."""
        if self._bearer_token:
            return self._bearer_token
        resp = await self._client.post(
            f"{self.base_url}/api/v1/auth/anonymous",
            json={"accessToken": self.access_token},
        )
        if resp.status_code not in (200, 201):
            raise GhostfolioError(resp.status_code, "Failed to authenticate")
        self._bearer_token = resp.json().get("authToken", "")
        return self._bearer_token

    def _invalidate_token(self) -> None:
        """Clear the cached bearer token so the next call will re-authenticate."""
        self._bearer_token = None

    def _auth_headers(self, bearer: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {bearer}"}

    # Retry: 2 attempts with a fixed 0.3 s gap (fast fail, avoids multi-second
    # exponential back-off that was adding latency to error paths).
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.3))
    async def get_portfolio_holdings(self) -> dict[str, Any]:
        """GET /api/v1/portfolio/holdings — returns all positions."""
        bearer = await self._get_bearer_token()
        resp = await self._client.get(
            f"{self.base_url}/api/v1/portfolio/holdings",
            headers=self._auth_headers(bearer),
        )
        if resp.status_code == 401:
            self._invalidate_token()
            raise GhostfolioError(401, "Token expired — will retry with fresh auth")
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.3))
    async def get_portfolio_performance(self, date_range: str = "max") -> dict[str, Any]:
        """GET /api/v1/portfolio/performance — YTD, 1Y, max returns."""
        bearer = await self._get_bearer_token()
        resp = await self._client.get(
            f"{self.base_url}/api/v1/portfolio/performance",
            params={"range": date_range},
            headers=self._auth_headers(bearer),
        )
        if resp.status_code == 401:
            self._invalidate_token()
            raise GhostfolioError(401, "Token expired — will retry with fresh auth")
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.3))
    async def get_orders(
        self,
        account_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """GET /api/v1/order — transaction history."""
        bearer = await self._get_bearer_token()
        params: dict[str, str] = {}
        if account_id:
            params["accountId"] = account_id
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        resp = await self._client.get(
            f"{self.base_url}/api/v1/order",
            params=params,
            headers=self._auth_headers(bearer),
        )
        if resp.status_code == 401:
            self._invalidate_token()
            raise GhostfolioError(401, "Token expired — will retry with fresh auth")
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    async def get_public_portfolio(self) -> dict[str, Any]:
        """GET /api/v1/public/{access_id}/portfolio — no auth required."""
        resp = await self._client.get(
            f"{self.base_url}/api/v1/public/{self.public_access_id}/portfolio"
        )
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GhostfolioClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


# ── Module-level Singleton ────────────────────────────────────────────────────
# Using a single shared client means the bearer token is fetched ONCE and
# reused for every subsequent tool call in the same process.
# Without this, every tool (portfolio, performance, transactions …) would each
# make its own POST /auth/anonymous, adding 200-400 ms latency per tool.

_shared_client: GhostfolioClient | None = None


def get_shared_client() -> GhostfolioClient:
    """
    Return the process-wide GhostfolioClient singleton.

    The bearer token is cached on the instance, so after the first
    authenticated request, all subsequent tool calls skip the auth round-trip.
    Thread-safe enough for asyncio (single-threaded event loop).
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = GhostfolioClient()
    return _shared_client
