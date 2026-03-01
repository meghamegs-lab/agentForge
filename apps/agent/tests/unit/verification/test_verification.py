"""
Unit tests for all 5 verification checks.
Pure Python logic — no LLM, no network calls.
"""

from datetime import UTC, datetime, timedelta

from agent.verification import (
    check_concentration,
    check_confidence,
    check_disclaimer,
    check_freshness,
    check_hallucination,
    run_verification_pipeline,
)

# ── 1. Disclaimer ──────────────────────────────────────────────────────────────


class TestDisclaimer:
    def test_appends_disclaimer_when_investment_keywords_present(self):
        response = "You should rebalance your portfolio toward bonds."
        result, flags = check_disclaimer(response, [])
        assert "Not financial advice" in result
        assert len(flags) == 1
        assert flags[0]["type"] == "DISCLAIMER_ADDED"

    def test_does_not_append_disclaimer_for_factual_query(self):
        response = "Your portfolio is currently worth $10,000."
        result, flags = check_disclaimer(response, [])
        assert "Not financial advice" not in result
        assert len(flags) == 0

    def test_disclaimer_not_duplicated_on_second_call(self):
        response = "I recommend you invest in index funds."
        result1, _ = check_disclaimer(response, [])
        result2, flags2 = check_disclaimer(result1, [])
        assert result2.count("Not financial advice") == 1

    def test_triggers_on_buy_more_directive(self):
        _, flags = check_disclaimer("You might want to buy more AAPL.", [])
        assert any(f["type"] == "DISCLAIMER_ADDED" for f in flags)

    def test_triggers_on_diversify_keyword(self):
        _, flags = check_disclaimer("Consider diversifying into international stocks.", [])
        assert any(f["type"] == "DISCLAIMER_ADDED" for f in flags)

    def test_no_disclaimer_for_buy_and_hold_description(self):
        # "buy-and-hold" is a strategy description, not investment advice.
        # Regression: \bbuy\b matches "buy-and-hold" because '-' is a non-word char.
        response = "That's a solid return for a buy-and-hold investor with zero fees."
        result, flags = check_disclaimer(response, [])
        assert "Not financial advice" not in result
        assert len(flags) == 0

    def test_no_disclaimer_for_investor_or_investing_descriptive(self):
        # "investor" and "started investing" describe facts — not advice.
        response = "Your return since you started investing in January 2024 is 14.1%."
        result, flags = check_disclaimer(response, [])
        assert "Not financial advice" not in result
        assert len(flags) == 0

    def test_no_disclaimer_for_performance_analysis_response(self):
        # Full realistic returns response should NOT trigger disclaimer.
        response = (
            "Your portfolio has grown 14.1% over the last 5 years, turning $23,655 into $26,825. "
            "That's a solid return for a buy-and-hold investor with zero fees. "
            "Your return since you started investing in January 2024 is positive overall."
        )
        result, flags = check_disclaimer(response, [])
        assert "Not financial advice" not in result
        assert len(flags) == 0

    def test_disclaimer_fires_for_explicit_buy_directive(self):
        _, flags = check_disclaimer("You should buy more AAPL to reduce cash drag.", [])
        assert any(f["type"] == "DISCLAIMER_ADDED" for f in flags)

    def test_disclaimer_fires_for_rebalance(self):
        _, flags = check_disclaimer("You should rebalance toward bonds.", [])
        assert any(f["type"] == "DISCLAIMER_ADDED" for f in flags)


# ── 2. Hallucination Guard ─────────────────────────────────────────────────────


class TestHallucinationGuard:
    def test_passes_when_numbers_present_in_tool_results(self):
        tool_results = [{"current_value": 1750.50, "data_timestamp": "2024-01-01T00:00:00Z"}]
        response = "Your portfolio value is $1750.50."
        _, flags = check_hallucination(response, tool_results)
        assert not any(f["type"] == "POTENTIAL_HALLUCINATION" for f in flags)

    def test_flags_number_not_in_tool_results(self):
        tool_results = [{"current_value": 1000.00, "data_timestamp": "2024-01-01T00:00:00Z"}]
        response = "Your portfolio is worth $99999.99 today."
        _, flags = check_hallucination(response, tool_results)
        assert any(f["severity"] == "MEDIUM" for f in flags)

    def test_flags_when_no_tool_called_but_financial_number_stated(self):
        _, flags = check_hallucination("AAPL is trading at $175.32 today.", [])
        assert any(f["type"] == "POTENTIAL_HALLUCINATION" for f in flags)
        # No tools called + specific financial number → HIGH severity:
        # the LLM is presenting training-data guesses as verified facts.
        assert flags[0]["severity"] == "HIGH"

    def test_no_unsupported_claim_for_trillion_formatted_market_cap(self):
        # LLM says "$3.88 trillion" but tool result has raw integer 3880000000000.
        # After suffix normalisation both become "3880000000000" → no false flag.
        tool_results = [
            {
                "status": "ok",
                "symbol": "AAPL",
                "current_price": 213.5,
                "market_cap": 3880000000000,
                "data_timestamp": "2024-01-01T00:00:00Z",
            }
        ]
        response = "Apple's market cap is $3.88 trillion and the price is $213.50."
        _, flags = check_hallucination(response, tool_results)
        unsupported = [f for f in flags if f["type"] == "UNSUPPORTED_CLAIM"]
        assert len(unsupported) == 0

    def test_ignores_years_as_financial_numbers(self):
        tool_results = [{"data_timestamp": "2024-01-01T00:00:00Z"}]
        _, flags = check_hallucination("Performance since 2022 has been strong.", tool_results)
        unsupported = [f for f in flags if f["type"] == "UNSUPPORTED_CLAIM"]
        assert len(unsupported) == 0

    def test_passes_with_empty_response(self):
        _, flags = check_hallucination("", [{"data_timestamp": "2024-01-01T00:00:00Z"}])
        assert len(flags) == 0


# ── 3. Freshness ───────────────────────────────────────────────────────────────


class TestFreshness:
    def test_passes_when_timestamp_is_recent(self):
        now = datetime.now(UTC).isoformat()
        _, flags = check_freshness("result", [{"current_price": 100, "data_timestamp": now}])
        stale = [f for f in flags if f["type"] == "STALE_DATA"]
        assert len(stale) == 0

    def test_warns_when_market_data_is_stale(self):
        old_ts = (datetime.now(UTC) - timedelta(minutes=20)).isoformat()
        _, flags = check_freshness("result", [{"current_price": 100, "data_timestamp": old_ts}])
        assert any(f["type"] == "STALE_DATA" for f in flags)

    def test_flags_missing_timestamp(self):
        _, flags = check_freshness("result", [{"current_price": 100}])
        assert any(f["type"] == "MISSING_TIMESTAMP" for f in flags)

    def test_portfolio_data_has_longer_threshold(self):
        # 45 min old portfolio data should NOT be flagged (threshold=60min)
        ts_45min_ago = (datetime.now(UTC) - timedelta(minutes=45)).isoformat()
        _, flags = check_freshness("result", [{"holdings": [], "data_timestamp": ts_45min_ago}])
        stale = [f for f in flags if f["type"] == "STALE_DATA"]
        assert len(stale) == 0


# ── 4. Concentration ───────────────────────────────────────────────────────────


class TestConcentration:
    def test_flags_position_above_20_percent(self):
        tool_results = [
            {
                "holdings": [
                    {"symbol": "AAPL", "allocation_percent": 45.0, "current_value": 4500},
                    {"symbol": "VTI", "allocation_percent": 55.0, "current_value": 5500},
                ]
            }
        ]
        response, flags = check_concentration("Your portfolio summary.", tool_results)
        assert any(f["type"] == "CONCENTRATION_RISK" for f in flags)
        assert "AAPL" in response

    def test_no_flag_when_all_positions_below_threshold(self):
        tool_results = [
            {
                "holdings": [
                    {"symbol": "AAPL", "allocation_percent": 15.0, "current_value": 1500},
                    {"symbol": "VTI", "allocation_percent": 18.0, "current_value": 1800},
                    {"symbol": "MSFT", "allocation_percent": 12.0, "current_value": 1200},
                ]
            }
        ]
        _, flags = check_concentration("Portfolio summary.", tool_results)
        assert not any(f["type"] == "CONCENTRATION_RISK" for f in flags)

    def test_uses_diversification_tool_flags(self):
        tool_results = [
            {
                "concentration_flags": [
                    {
                        "symbol": "TSLA",
                        "severity": "HIGH",
                        "message": "TSLA is 40.0% of your portfolio",
                    }
                ]
            }
        ]
        response, flags = check_concentration("Analysis result.", tool_results)
        assert any(f["severity"] == "HIGH" for f in flags)
        assert "TSLA" in response

    def test_warning_appended_to_response(self):
        tool_results = [
            {
                "holdings": [
                    {"symbol": "BIG", "allocation_percent": 60.0, "current_value": 6000},
                ]
            }
        ]
        response, _ = check_concentration("Here is your portfolio.", tool_results)
        assert "Concentration Risk" in response


# ── 5. Confidence Scoring ──────────────────────────────────────────────────────


class TestConfidence:
    def test_high_confidence_for_direct_data_lookup(self):
        tool_results = [{"status": "ok", "data_timestamp": "2024-01-01T00:00:00Z"}]
        _, flags, confidence = check_confidence("Your portfolio is worth $10,000.", tool_results, 1)
        assert confidence == "HIGH"

    def test_medium_confidence_for_multi_tool_chain(self):
        # 3+ reasoning steps = genuine multi-round analysis → MEDIUM
        tool_results = [{"a": 1}, {"b": 2}]
        _, flags, confidence = check_confidence("Based on analysis...", tool_results, 3)
        assert confidence == "MEDIUM"

    def test_high_confidence_for_parallel_price_lookup(self):
        # Two parallel price lookups (one reasoning round) → still HIGH
        # This is the "AAPL and MSFT prices" scenario: the LLM calls both tools
        # simultaneously in step 1, synthesises in step 2 → reasoning_steps = 2
        tool_results = [
            {"status": "ok", "symbol": "AAPL", "data_timestamp": "2024-01-01T00:00:00Z"},
            {"status": "ok", "symbol": "MSFT", "data_timestamp": "2024-01-01T00:00:00Z"},
        ]
        _, flags, confidence = check_confidence(
            "AAPL is $213.50 and MSFT is $420.30.", tool_results, 2
        )
        assert confidence == "HIGH"

    def test_low_confidence_for_predictions(self):
        _, flags, confidence = check_confidence("AAPL will likely reach $200 next year.", [], 1)
        assert confidence == "LOW"

    def test_low_confidence_when_no_tool_data(self):
        _, flags, confidence = check_confidence("Your balance might be around $5000.", [], 1)
        assert confidence == "LOW"


# ── Full Pipeline ──────────────────────────────────────────────────────────────


class TestVerificationPipeline:
    def test_pipeline_returns_all_required_fields(self, now_iso):
        tool_results = [{"holdings": [], "data_timestamp": now_iso}]
        result = run_verification_pipeline("Your portfolio is empty.", tool_results)
        assert "response" in result
        assert "confidence" in result
        assert "verification_flags" in result
        assert "flag_count" in result
        assert "has_high_severity" in result

    def test_pipeline_adds_disclaimer_for_advice(self, now_iso):
        result = run_verification_pipeline(
            "You should rebalance your portfolio.", [{"data_timestamp": now_iso}]
        )
        assert "Not financial advice" in result["response"]

    def test_pipeline_handles_empty_response(self, now_iso):
        result = run_verification_pipeline("", [{"data_timestamp": now_iso}])
        assert result["response"] == "" or isinstance(result["response"], str)
