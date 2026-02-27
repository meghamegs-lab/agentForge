# Async HTTP client for the Ghostfolio REST API with bearer-token caching and automatic retry logic.
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

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

from agent.config import settings


class GhostfolioError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Ghostfolio API error {status_code}: {message}")


class GhostfolioClient:
    """Async HTTP client for Ghostfolio REST API."""

    # ── API endpoint paths ────────────────────────────────────────────────────
    # Centralised here so a Ghostfolio API upgrade only requires changes in
    # one place. Note: /portfolio/performance moved to v2 in Ghostfolio ≥2.x;
    # all other endpoints remain on v1.
    _ENDPOINTS: dict[str, str] = {
        "auth":        "api/v1/auth/anonymous",
        "holdings":    "api/v1/portfolio/holdings",
        "performance": "api/v2/portfolio/performance",  # v2-only endpoint
        "orders":      "api/v1/order",
        "public":      "api/v1/public/{access_id}/portfolio",
    }

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

    # Builds a full URL from a named endpoint key, with optional path substitutions.
    def _url(self, key: str, **kwargs: str) -> str:
        """Return the full URL for a named Ghostfolio endpoint.

        Looks up the path template from _ENDPOINTS, applies any keyword
        substitutions (e.g. access_id=...), and prepends the configured base URL.
        """
        path = self._ENDPOINTS[key].format(**kwargs)
        return f"{self.base_url}/{path}"

    # Exchanges the static access token for a JWT bearer token; caches it for the lifetime of the instance.
    async def _get_bearer_token(self) -> str:
        """Exchange the security token for a JWT bearer token (cached per instance)."""
        if self._bearer_token:
            return self._bearer_token
        resp = await self._client.post(
            self._url("auth"),
            json={"accessToken": self.access_token},
        )
        if resp.status_code not in (200, 201):
            raise GhostfolioError(resp.status_code, "Failed to authenticate")
        self._bearer_token = resp.json().get("authToken", "")
        return self._bearer_token

    # Forces a fresh login on the next API call by discarding the cached bearer token.
    def _invalidate_token(self) -> None:
        """Clear the cached bearer token so the next call will re-authenticate."""
        self._bearer_token = None

    # Builds the Authorization: Bearer header dict from a token string.
    def _auth_headers(self, bearer: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {bearer}"}

    # Retry: 2 attempts with a fixed 0.3 s gap (fast fail, avoids multi-second
    # exponential back-off that was adding latency to error paths).
    # Fetches all current portfolio positions from GET /api/v1/portfolio/holdings (retries on 401).
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.3))
    async def get_portfolio_holdings(self) -> dict[str, Any]:
        """GET /api/v1/portfolio/holdings — returns all positions."""
        bearer = await self._get_bearer_token()
        resp = await self._client.get(
            self._url("holdings"),
            headers=self._auth_headers(bearer),
        )
        if resp.status_code == 401:
            self._invalidate_token()
            raise GhostfolioError(401, "Token expired — will retry with fresh auth")
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    # Fetches portfolio performance metrics (returns, gains) for the given date range.
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.3))
    async def get_portfolio_performance(self, date_range: str = "max") -> dict[str, Any]:
        """GET /api/v2/portfolio/performance — YTD, 1Y, max returns."""
        bearer = await self._get_bearer_token()
        resp = await self._client.get(
            self._url("performance"),
            params={"range": date_range},
            headers=self._auth_headers(bearer),
        )
        if resp.status_code == 401:
            self._invalidate_token()
            raise GhostfolioError(401, "Token expired — will retry with fresh auth")
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    # Fetches transaction order history from GET /api/v1/order with optional account/date/type filters.
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
            self._url("orders"),
            params=params,
            headers=self._auth_headers(bearer),
        )
        if resp.status_code == 401:
            self._invalidate_token()
            raise GhostfolioError(401, "Token expired — will retry with fresh auth")
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    # Fetches the public portfolio view using the public access ID — no authentication required.
    async def get_public_portfolio(self) -> dict[str, Any]:
        """GET /api/v1/public/{access_id}/portfolio — no auth required."""
        resp = await self._client.get(
            self._url("public", access_id=self.public_access_id)
        )
        if resp.status_code != 200:
            raise GhostfolioError(resp.status_code, resp.text)
        return resp.json()

    # Closes the underlying httpx.AsyncClient and releases its connection pool.
    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GhostfolioClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


# ── Module-level Singleton ────────────────────────────────────────────────────
# Using a single shared client means the bearer token is fetched ONCE and
# reused for every subsequent tool call in the same process.
# Without this, every tool (portfolio, performance, transactions …) would each
# make its own POST /auth/anonymous, adding 200-400 ms latency per tool.

_shared_client: GhostfolioClient | None = None


# Returns the process-wide GhostfolioClient singleton, creating it on the first call.
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
