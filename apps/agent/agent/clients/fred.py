# Async HTTP client for the FRED (Federal Reserve Economic Data) API with tenacity retry.
"""
FRED API client — fetches macroeconomic data for the FIRE Goal Tracker.

What is FRED?
-------------
FRED (Federal Reserve Economic Data) is a free public database maintained by the
Federal Reserve Bank of St. Louis. It contains 800,000+ economic time series.
Base URL: https://api.stlouisfed.org/fred/

Series used by Fortio:
  CPIAUCSL  — Consumer Price Index (all urban consumers, seasonally adjusted)
              Used to compute the year-over-year inflation rate.
  DGS10     — 10-Year US Treasury Constant Maturity Rate
              Used as the risk-free rate for safe withdrawal rate comparisons.

API Key:
---------
Free registration at https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in .env.  If empty, all calls return a structured error dict.

Error handling:
---------------
All public methods return a structured dict and NEVER raise.
On failure:  {"status": "error", "error": "<reason>", "series_id": "<id>"}
On success:  {"status": "ok", "series_id": "<id>", "observations": [...], ...}

Retry policy:
-------------
Up to 3 attempts with exponential back-off (1 s → 2 s → 4 s) on any
network-level exception (ConnectionError, Timeout, HTTP 5xx).
HTTP 400 / 403 / 404 are NOT retried — they signal a config error.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.config import settings

log = structlog.get_logger()

_FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Status codes that are worth retrying (transient server/network issues).
# 400 Bad Request (bad params) and 403 Forbidden (bad key) are permanent.
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


class FredClient:
    """
    Async HTTP client for the FRED REST API.

    Usage inside tools:
        from agent.clients.fred import get_shared_fred_client
        client = get_shared_fred_client()
        inflation = await client.get_current_inflation_rate()
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Use None as sentinel so FredClient(api_key="") is respected as "no key"
        # rather than falling back to settings. This allows tests to exercise the
        # missing-key error path even when FRED_API_KEY is configured in .env.
        self._api_key = api_key if api_key is not None else settings.fred_api_key

    # ── Internal: raw series fetch (with tenacity retry) ──────────────────────

    @retry(
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    async def _fetch_series(self, series_id: str, limit: int = 13) -> dict[str, Any]:
        """
        Raw FRED API call — retried on transient network errors.

        Returns the raw JSON from FRED:
          {"observations": [{"date": "2026-01-01", "value": "314.175"}, ...]}

        Raises on HTTP errors so tenacity can retry 5xx responses.
        Raises ValueError on 400/403 (permanent errors — no retry).
        """
        if not self._api_key:
            raise ValueError(
                "FRED_API_KEY is not set. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
                "and add FRED_API_KEY=<your_key> to apps/agent/.env"
            )

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_FRED_BASE_URL}/series/observations", params=params)

        if resp.status_code in _RETRYABLE_HTTP_CODES:
            # Raise so tenacity retries
            resp.raise_for_status()

        if resp.status_code == 400:
            body = resp.json()
            raise ValueError(f"FRED 400 Bad Request: {body.get('error_message', resp.text)}")

        if resp.status_code == 403:
            raise ValueError("FRED 403 Forbidden: check your FRED_API_KEY")

        resp.raise_for_status()
        return resp.json()

    # ── Public async methods ───────────────────────────────────────────────────

    async def get_series(self, series_id: str, limit: int = 13) -> dict[str, Any]:
        """
        Fetch FRED series observations.  Never raises — returns structured dict.

        Args:
            series_id: FRED series ID, e.g. "CPIAUCSL" or "DGS10"
            limit:     Number of most-recent observations to fetch (default 13)

        Returns:
            {"status": "ok", "series_id": "...", "observations": [...], "data_timestamp": "..."}
            or
            {"status": "error", "error": "...", "series_id": "..."}
        """
        try:
            data = await self._fetch_series(series_id, limit)
            observations = data.get("observations", [])
            # Filter out FRED's placeholder "." values (missing data points)
            valid_obs = [
                {"date": o["date"], "value": float(o["value"])}
                for o in observations
                if o.get("value", ".") != "."
            ]
            return {
                "status": "ok",
                "series_id": series_id,
                "observations": valid_obs,
                "count": len(valid_obs),
                "data_timestamp": datetime.now(UTC).isoformat(),
                "source": "FRED / Federal Reserve Bank of St. Louis",
            }
        except RetryError as e:
            cause = e.last_attempt.exception()
            error_msg = str(cause) if cause else str(e)
            log.error("fred_retry_exhausted", series_id=series_id, error=error_msg)
            return {"status": "error", "error": error_msg, "series_id": series_id}
        except ValueError as e:
            # Config errors (missing key, bad key) — no retry, clear message
            log.error("fred_config_error", series_id=series_id, error=str(e))
            return {"status": "error", "error": str(e), "series_id": series_id}
        except Exception as e:
            log.error("fred_unexpected_error", series_id=series_id, error=str(e))
            return {"status": "error", "error": str(e), "series_id": series_id}

    async def get_current_inflation_rate(self) -> dict[str, Any]:
        """
        Compute year-over-year CPI inflation rate from FRED CPIAUCSL series.

        Returns:
            {
                "status": "ok",
                "inflation_rate_yoy": 0.032,      # 3.2% as a decimal
                "inflation_rate_pct": 3.2,         # same as a percentage
                "latest_cpi": 314.175,
                "cpi_12m_ago": 304.702,
                "latest_cpi_date": "2026-01-01",
                "data_timestamp": "..."
            }
        """
        result = await self.get_series("CPIAUCSL", limit=13)
        if result["status"] != "ok":
            return {
                "status": "error",
                "error": result.get("error", "Failed to fetch CPI data"),
                "series_id": "CPIAUCSL",
            }

        obs = result["observations"]
        if len(obs) < 2:
            return {
                "status": "error",
                "error": "Insufficient CPI data to compute YoY inflation (need ≥2 months)",
                "series_id": "CPIAUCSL",
            }

        # obs is sorted desc (most recent first)
        latest_cpi = obs[0]["value"]
        latest_date = obs[0]["date"]

        # 12 months ago — obs[12] if available, else obs[-1]
        cpi_12m_ago = obs[min(12, len(obs) - 1)]["value"]

        if cpi_12m_ago == 0:
            return {
                "status": "error",
                "error": "CPI 12 months ago was 0 — cannot compute inflation rate",
                "series_id": "CPIAUCSL",
            }

        yoy = (latest_cpi - cpi_12m_ago) / cpi_12m_ago

        return {
            "status": "ok",
            "series_id": "CPIAUCSL",
            "inflation_rate_yoy": round(yoy, 4),  # decimal, e.g. 0.032
            "inflation_rate_pct": round(yoy * 100, 2),  # %, e.g. 3.2
            "latest_cpi": latest_cpi,
            "cpi_12m_ago": cpi_12m_ago,
            "latest_cpi_date": latest_date,
            "data_timestamp": result["data_timestamp"],
            "source": result["source"],
        }

    async def get_risk_free_rate(self) -> dict[str, Any]:
        """
        Fetch the current 10-Year US Treasury yield from FRED DGS10.

        Returns:
            {
                "status": "ok",
                "risk_free_rate": 0.0445,    # decimal, e.g. 4.45%
                "risk_free_rate_pct": 4.45,  # percentage
                "rate_date": "2026-02-28",
                "data_timestamp": "..."
            }
        """
        result = await self.get_series("DGS10", limit=5)
        if result["status"] != "ok":
            return {
                "status": "error",
                "error": result.get("error", "Failed to fetch 10Y Treasury data"),
                "series_id": "DGS10",
            }

        obs = result["observations"]
        if not obs:
            return {
                "status": "error",
                "error": "No 10Y Treasury data available",
                "series_id": "DGS10",
            }

        # DGS10 is already in % (e.g. 4.45 means 4.45%)
        rate_pct = obs[0]["value"]
        rate_decimal = rate_pct / 100.0

        return {
            "status": "ok",
            "series_id": "DGS10",
            "risk_free_rate": round(rate_decimal, 4),
            "risk_free_rate_pct": round(rate_pct, 2),
            "rate_date": obs[0]["date"],
            "data_timestamp": result["data_timestamp"],
            "source": result["source"],
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
# Same pattern as GhostfolioClient's get_shared_client() — one instance per process.

_shared_fred_client: FredClient | None = None


def get_shared_fred_client() -> FredClient:
    """
    Returns the shared FredClient singleton (created on first call).
    Import and call this inside tools — do not instantiate FredClient directly.
    """
    global _shared_fred_client
    if _shared_fred_client is None:
        _shared_fred_client = FredClient()
    return _shared_fred_client
