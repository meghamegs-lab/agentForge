"""
Unit tests for agent/clients/ghostfolio.py
===========================================
Tests GhostfolioClient bearer-token caching, 401 retry logic, all four
API endpoints, and the module-level singleton.

All HTTP calls are intercepted with respx — no real network requests.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from agent.clients.ghostfolio import (
    GhostfolioClient,
    GhostfolioError,
    get_shared_client,
)
from agent.config import settings

BASE_URL = settings.ghostfolio_base_url.rstrip("/")
AUTH_TOKEN = "test-bearer-xyz"
AUTH_RESP = {"authToken": AUTH_TOKEN}


def _register_auth(token: str = AUTH_TOKEN) -> None:
    """Register mock auth endpoint inside a respx.mock block."""
    respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json={"authToken": token})
    )


# ── GhostfolioError ──────────────────────────────────────────────────────────

class TestGhostfolioError:
    def test_carries_status_code_and_message(self):
        err = GhostfolioError(404, "resource not found")
        assert err.status_code == 404
        assert err.message == "resource not found"

    def test_str_representation_includes_both(self):
        err = GhostfolioError(500, "internal server error")
        assert "500" in str(err)
        assert "internal server error" in str(err)

    def test_is_exception_subclass(self):
        assert issubclass(GhostfolioError, Exception)


# ── Bearer token caching ─────────────────────────────────────────────────────

class TestBearerTokenCaching:
    @respx.mock
    async def test_token_fetched_once_and_reused(self):
        """
        The auth POST should only be called ONCE even if multiple API methods
        are invoked.  The bearer token must be cached on the instance.
        """
        auth_route = respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(200, json=AUTH_RESP)
        )
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json={"holdings": []})
        )
        respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
            return_value=httpx.Response(200, json={"performance": {}})
        )

        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_portfolio_holdings()
        await client.get_portfolio_performance()

        # Auth endpoint called exactly once despite two API calls
        assert auth_route.call_count == 1

    @respx.mock
    async def test_token_stored_on_instance(self):
        """After first authenticated call, _bearer_token is populated."""
        _register_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json={"holdings": []})
        )

        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        assert client._bearer_token is None

        await client.get_portfolio_holdings()
        assert client._bearer_token == AUTH_TOKEN

    @respx.mock
    async def test_invalidate_token_clears_cache(self):
        """_invalidate_token() must clear the cached bearer so next call re-auths."""
        _register_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json={"holdings": []})
        )

        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_portfolio_holdings()
        assert client._bearer_token == AUTH_TOKEN

        client._invalidate_token()
        assert client._bearer_token is None


# ── 401 retry with fresh token ────────────────────────────────────────────────

class TestTokenRefreshOn401:
    @respx.mock
    async def test_401_triggers_token_invalidation_and_retry(self):
        """
        On a 401 response the client must:
        1. Invalidate the cached token
        2. Re-authenticate on the next attempt
        3. Retry the original request with the new token
        tenacity retries at most 2 times.
        """
        # First auth call returns token-A; second returns token-B
        auth_route = respx.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
            side_effect=[
                httpx.Response(200, json={"authToken": "token-A"}),
                httpx.Response(200, json={"authToken": "token-B"}),
            ]
        )
        # Holdings: first call returns 401, second returns 200
        holdings_route = respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            side_effect=[
                httpx.Response(401),
                httpx.Response(200, json={"holdings": []}),
            ]
        )

        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        result = await client.get_portfolio_holdings()

        # Auth must have been called twice (first fetch + re-auth after 401)
        assert auth_route.call_count == 2
        # Holdings must have been called twice (original + retry)
        assert holdings_route.call_count == 2
        assert result == {"holdings": []}


# ── get_portfolio_holdings ────────────────────────────────────────────────────

class TestGetPortfolioHoldings:
    @respx.mock
    async def test_returns_parsed_json(self):
        _register_auth()
        payload = {"holdings": [{"symbol": "AAPL", "value": 1750.0}]}
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        result = await client.get_portfolio_holdings()
        assert result == payload

    @respx.mock
    async def test_sends_bearer_authorization_header(self):
        _register_auth()
        holdings_route = respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(200, json={"holdings": []})
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_portfolio_holdings()

        sent_headers = holdings_route.calls.last.request.headers
        assert sent_headers["Authorization"] == f"Bearer {AUTH_TOKEN}"

    @respx.mock
    async def test_non_200_raises_error(self):
        """
        A non-200, non-401 response must cause an exception to be raised.
        After tenacity exhausts its retries, it wraps the GhostfolioError in
        RetryError — so we just assert that SOME exception propagates out.
        """
        _register_auth()
        respx.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        with pytest.raises(Exception):
            await client.get_portfolio_holdings()


# ── get_portfolio_performance ─────────────────────────────────────────────────

class TestGetPortfolioPerformance:
    @respx.mock
    async def test_passes_range_query_param(self):
        _register_auth()
        perf_route = respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
            return_value=httpx.Response(200, json={"performance": {}})
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_portfolio_performance(date_range="1y")

        # Verify the `range` query param was sent
        url = str(perf_route.calls.last.request.url)
        assert "range=1y" in url

    @respx.mock
    async def test_default_range_is_max(self):
        _register_auth()
        perf_route = respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
            return_value=httpx.Response(200, json={"performance": {}})
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_portfolio_performance()

        url = str(perf_route.calls.last.request.url)
        assert "range=max" in url

    @respx.mock
    async def test_returns_parsed_json(self):
        _register_auth()
        payload = {"performance": {"ytd": {"relativeChange": 0.12}}}
        respx.get(f"{BASE_URL}/api/v1/portfolio/performance").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        result = await client.get_portfolio_performance()
        assert result == payload


# ── get_orders ────────────────────────────────────────────────────────────────

class TestGetOrders:
    @respx.mock
    async def test_no_params_sends_no_query_string(self):
        _register_auth()
        orders_route = respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json={"activities": []})
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_orders()

        url = str(orders_route.calls.last.request.url)
        # No optional params → no query string beyond the base path
        assert "accountId" not in url
        assert "dateFrom" not in url
        assert "dateTo" not in url

    @respx.mock
    async def test_optional_params_passed_when_provided(self):
        _register_auth()
        orders_route = respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json={"activities": []})
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        await client.get_orders(
            account_id="acc-1",
            date_from="2024-01-01",
            date_to="2024-12-31",
        )

        url = str(orders_route.calls.last.request.url)
        assert "accountId=acc-1" in url
        assert "dateFrom=2024-01-01" in url
        assert "dateTo=2024-12-31" in url

    @respx.mock
    async def test_returns_activities_list(self):
        _register_auth()
        payload = {"activities": [{"id": "t1", "type": "BUY"}]}
        respx.get(f"{BASE_URL}/api/v1/order").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = GhostfolioClient(base_url=BASE_URL, access_token="tok")
        result = await client.get_orders()
        assert result == payload


# ── get_public_portfolio ──────────────────────────────────────────────────────

class TestGetPublicPortfolio:
    @respx.mock
    async def test_no_authorization_header_sent(self):
        """Public endpoint must NOT include an Authorization header."""
        pub_id = "public-abc-123"
        pub_route = respx.get(
            f"{BASE_URL}/api/v1/public/{pub_id}/portfolio"
        ).mock(
            return_value=httpx.Response(200, json={"portfolio": {}})
        )
        client = GhostfolioClient(
            base_url=BASE_URL,
            access_token="tok",
            public_access_id=pub_id,
        )
        await client.get_public_portfolio()

        sent_headers = pub_route.calls.last.request.headers
        assert "authorization" not in sent_headers

    @respx.mock
    async def test_returns_parsed_json(self):
        pub_id = "pub-id-999"
        payload = {"portfolio": {"totalValue": 5000}}
        respx.get(f"{BASE_URL}/api/v1/public/{pub_id}/portfolio").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = GhostfolioClient(
            base_url=BASE_URL,
            access_token="tok",
            public_access_id=pub_id,
        )
        result = await client.get_public_portfolio()
        assert result == payload

    @respx.mock
    async def test_non_200_raises_ghostfolio_error(self):
        pub_id = "pub-bad"
        respx.get(f"{BASE_URL}/api/v1/public/{pub_id}/portfolio").mock(
            return_value=httpx.Response(404, text="Not Found")
        )
        client = GhostfolioClient(
            base_url=BASE_URL,
            access_token="tok",
            public_access_id=pub_id,
        )
        with pytest.raises(GhostfolioError) as exc_info:
            await client.get_public_portfolio()
        assert exc_info.value.status_code == 404


# ── Module-level singleton ────────────────────────────────────────────────────

class TestGetSharedClient:
    def test_returns_ghostfolio_client_instance(self):
        import agent.clients.ghostfolio as mod

        # Reset the singleton so we get a fresh one for this test
        mod._shared_client = None

        client = get_shared_client()
        assert isinstance(client, GhostfolioClient)

    def test_same_instance_on_repeated_calls(self):
        import agent.clients.ghostfolio as mod

        mod._shared_client = None  # Reset

        c1 = get_shared_client()
        c2 = get_shared_client()
        assert c1 is c2

    def test_singleton_is_reused_after_creation(self):
        import agent.clients.ghostfolio as mod

        # First call creates it; second call reuses it
        mod._shared_client = None
        first = get_shared_client()
        mod._shared_client = first  # ensure it's stored

        second = get_shared_client()
        assert first is second
