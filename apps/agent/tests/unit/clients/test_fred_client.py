"""
Unit tests for agent/clients/fred.py
======================================
Tests FredClient.get_series(), get_current_inflation_rate(), and get_risk_free_rate()
with mocked HTTP calls via respx — no real network requests, no FRED API key required.

All test observations use realistic FRED JSON shapes so the parsing logic is
exercised exactly as it would be in production.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent.clients.fred import FredClient

# ── Helpers ───────────────────────────────────────────────────────────────────

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred_response(observations: list[dict]) -> httpx.Response:
    """Build a mock FRED API JSON response with the given observations."""
    return httpx.Response(
        200,
        json={
            "realtime_start": "2026-02-28",
            "realtime_end": "2026-02-28",
            "observation_start": "1600-01-01",
            "observation_end": "9999-12-31",
            "units": "lin",
            "output_type": 1,
            "file_type": "json",
            "order_by": "observation_date",
            "sort_order": "desc",
            "count": len(observations),
            "offset": 0,
            "limit": len(observations),
            "observations": observations,
        },
    )


# 13 CPI observations (most-recent first, descending)
_CPI_OBS = [
    {"date": "2026-01-01", "value": "320.000"},   # latest
    {"date": "2025-12-01", "value": "319.200"},
    {"date": "2025-11-01", "value": "318.500"},
    {"date": "2025-10-01", "value": "317.800"},
    {"date": "2025-09-01", "value": "317.000"},
    {"date": "2025-08-01", "value": "316.300"},
    {"date": "2025-07-01", "value": "315.500"},
    {"date": "2025-06-01", "value": "314.800"},
    {"date": "2025-05-01", "value": "314.000"},
    {"date": "2025-04-01", "value": "313.300"},
    {"date": "2025-03-01", "value": "312.500"},
    {"date": "2025-02-01", "value": "311.800"},
    {"date": "2025-01-01", "value": "310.000"},   # 12 months ago
]
# YoY = (320 - 310) / 310 ≈ 0.03226

_DGS10_OBS = [
    {"date": "2026-02-28", "value": "4.45"},  # latest 10Y yield (%)
    {"date": "2026-02-27", "value": "4.43"},
    {"date": "2026-02-26", "value": "4.40"},
]


def _client_with_key() -> FredClient:
    """FredClient with a fake API key (avoids the 'key not set' guard)."""
    return FredClient(api_key="test-key-12345")


# ── TestGetSeries ──────────────────────────────────────────────────────────────


class TestGetSeries:
    @respx.mock
    async def test_ok_status_on_200(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS[:3]))
        result = await _client_with_key().get_series("CPIAUCSL", limit=3)
        assert result["status"] == "ok"

    @respx.mock
    async def test_series_id_echoed_in_result(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS[:3]))
        result = await _client_with_key().get_series("CPIAUCSL", limit=3)
        assert result["series_id"] == "CPIAUCSL"

    @respx.mock
    async def test_observations_parsed_as_float_dicts(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS[:2]))
        result = await _client_with_key().get_series("CPIAUCSL", limit=2)
        obs = result["observations"]
        assert len(obs) == 2
        assert isinstance(obs[0]["value"], float)
        assert obs[0]["value"] == pytest.approx(320.0)

    @respx.mock
    async def test_dot_placeholder_values_are_filtered_out(self):
        """FRED uses '.' for missing data points — they must be excluded."""
        obs_with_dot = [
            {"date": "2026-01-01", "value": "320.0"},
            {"date": "2025-12-01", "value": "."},  # missing — should be dropped
            {"date": "2025-11-01", "value": "318.5"},
        ]
        respx.get(FRED_BASE).mock(return_value=_fred_response(obs_with_dot))
        result = await _client_with_key().get_series("CPIAUCSL", limit=3)
        assert result["count"] == 2
        dates = [o["date"] for o in result["observations"]]
        assert "2025-12-01" not in dates

    @respx.mock
    async def test_result_includes_data_timestamp(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS[:1]))
        result = await _client_with_key().get_series("CPIAUCSL", limit=1)
        assert "data_timestamp" in result

    async def test_missing_api_key_returns_error_not_raises(self):
        """An empty API key must return a structured error dict, never raise."""
        client = FredClient(api_key="")  # no key
        # No respx mock needed — error fires before any HTTP call
        result = await client.get_series("CPIAUCSL")
        assert result["status"] == "error"
        assert "FRED_API_KEY" in result["error"]

    @respx.mock
    async def test_network_error_returns_structured_error(self):
        """ConnectError must be caught and returned as a structured dict."""
        respx.get(FRED_BASE).mock(side_effect=httpx.ConnectError("connection refused"))
        result = await _client_with_key().get_series("CPIAUCSL")
        assert result["status"] == "error"
        assert "error" in result

    @respx.mock
    async def test_500_response_returns_error(self):
        respx.get(FRED_BASE).mock(return_value=httpx.Response(500, text="Internal Server Error"))
        result = await _client_with_key().get_series("CPIAUCSL")
        assert result["status"] == "error"


# ── TestGetCurrentInflationRate ────────────────────────────────────────────────


class TestGetCurrentInflationRate:
    @respx.mock
    async def test_returns_ok_status(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS))
        result = await _client_with_key().get_current_inflation_rate()
        assert result["status"] == "ok"

    @respx.mock
    async def test_inflation_rate_is_correct(self):
        """YoY = (320 - 310) / 310 ≈ 0.03226"""
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS))
        result = await _client_with_key().get_current_inflation_rate()
        expected_yoy = (320.0 - 310.0) / 310.0
        assert result["inflation_rate_yoy"] == pytest.approx(expected_yoy, abs=0.001)

    @respx.mock
    async def test_inflation_pct_matches_yoy_times_100(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS))
        result = await _client_with_key().get_current_inflation_rate()
        assert abs(result["inflation_rate_pct"] - result["inflation_rate_yoy"] * 100) < 0.01

    @respx.mock
    async def test_latest_cpi_date_is_present(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS))
        result = await _client_with_key().get_current_inflation_rate()
        assert result["latest_cpi_date"] == "2026-01-01"

    @respx.mock
    async def test_insufficient_data_returns_error(self):
        """Only 1 observation — can't compute YoY."""
        respx.get(FRED_BASE).mock(return_value=_fred_response(_CPI_OBS[:1]))
        result = await _client_with_key().get_current_inflation_rate()
        assert result["status"] == "error"
        assert "Insufficient" in result["error"]

    async def test_missing_key_returns_error(self):
        result = await FredClient(api_key="").get_current_inflation_rate()
        assert result["status"] == "error"


# ── TestGetRiskFreeRate ────────────────────────────────────────────────────────


class TestGetRiskFreeRate:
    @respx.mock
    async def test_returns_ok_status(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_DGS10_OBS))
        result = await _client_with_key().get_risk_free_rate()
        assert result["status"] == "ok"

    @respx.mock
    async def test_yield_is_converted_from_pct_to_decimal(self):
        """DGS10 = 4.45 in FRED (%) → should be returned as 0.0445 decimal."""
        respx.get(FRED_BASE).mock(return_value=_fred_response(_DGS10_OBS))
        result = await _client_with_key().get_risk_free_rate()
        assert result["risk_free_rate"] == pytest.approx(0.0445, abs=0.0001)
        assert result["risk_free_rate_pct"] == pytest.approx(4.45, abs=0.01)

    @respx.mock
    async def test_rate_date_in_result(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response(_DGS10_OBS))
        result = await _client_with_key().get_risk_free_rate()
        assert result["rate_date"] == "2026-02-28"

    @respx.mock
    async def test_empty_observations_returns_error(self):
        respx.get(FRED_BASE).mock(return_value=_fred_response([]))
        result = await _client_with_key().get_risk_free_rate()
        assert result["status"] == "error"

    async def test_missing_key_returns_error(self):
        result = await FredClient(api_key="").get_risk_free_rate()
        assert result["status"] == "error"
