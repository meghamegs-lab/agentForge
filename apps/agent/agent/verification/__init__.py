# Five-stage verification pipeline: disclaimer → hallucination → freshness → concentration → confidence.
"""
Verification Layer — All 5 production checks for the finance agent.

Each verifier takes the agent response dict and the tool_results list,
returns (modified_response, list_of_flags).

Pipeline runs in order: disclaimer → hallucination → freshness → concentration → confidence
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from agent.config import settings

# ─── Types ────────────────────────────────────────────────────────────────────

VerificationFlag = dict[str, str]  # {type, severity, message}


# ─── 1. Disclaimer Injection ──────────────────────────────────────────────────

INVESTMENT_KEYWORDS = {
    "buy", "sell", "invest", "rebalance", "recommend", "should i",
    "allocate", "diversify", "move", "shift", "rotate", "switch",
    "portfolio change", "what to do", "advice",
}

DISCLAIMER_TEXT = (
    "\n\n⚠️ **Not financial advice.** This information is for educational purposes only. "
    "Always consult a qualified financial advisor before making investment decisions."
)


# Appends a financial disclaimer to the response when investment-advice language is detected.
def check_disclaimer(
    response: str, tool_results: list[dict]
) -> tuple[str, list[VerificationFlag]]:
    """
    Appends the standard financial disclaimer whenever the response contains
    investment suggestion language. Idempotent — won't add twice.
    """
    flags: list[VerificationFlag] = []
    response_lower = response.lower()

    needs_disclaimer = any(kw in response_lower for kw in INVESTMENT_KEYWORDS)

    if needs_disclaimer and DISCLAIMER_TEXT.strip() not in response:
        response = response + DISCLAIMER_TEXT
        flags.append({
            "type": "DISCLAIMER_ADDED",
            "severity": "INFO",
            "message": "Investment language detected — disclaimer appended",
        })

    return response, flags


# ─── 2. Hallucination Guard ───────────────────────────────────────────────────

# Extracts all numeric values (prices, percentages, dollar amounts) from a text string using regex.
def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric values from text (prices, percentages, dollar amounts)."""
    # Match: $1,234.56 | 12.34% | 1234.56 | 1,234
    pattern = r"\$?[\d,]+\.?\d*%?"
    matches = re.findall(pattern, text)
    # Normalize: remove $ , % for comparison
    return {re.sub(r"[$,%]", "", m).replace(",", "") for m in matches if len(m) > 1}


# Flattens all tool result dicts to JSON and extracts every numeric value from them.
def _extract_numbers_from_tool_results(tool_results: list[dict]) -> set[str]:
    """Flatten all numeric values from tool result dicts."""
    text = json.dumps(tool_results)
    return _extract_numbers(text)


# Returns True if every tool result carries an error/failure status — meaning no real data was fetched.
def _all_tools_failed(tool_results: list[dict]) -> bool:
    """
    Return True if every tool result has a failure status
    (error, price_unavailable) — meaning no real data was retrieved.
    """
    failure_statuses = {"error", "price_unavailable"}
    return bool(tool_results) and all(
        r.get("status") in failure_statuses for r in tool_results
    )


# Flags financial numbers in the response that cannot be traced back to any tool result.
def check_hallucination(
    response: str, tool_results: list[dict]
) -> tuple[str, list[VerificationFlag]]:
    """
    Flags numeric claims in the response that don't appear in any tool result.

    Two HIGH-severity escalation cases:
    1. No tools were called at all but LLM states financial numbers.
    2. ALL tools returned errors / price_unavailable but LLM still states
       financial numbers — this means the LLM guessed despite having no data.

    Skips small integers (counts, years) and round percentages below 5.
    """
    flags: list[VerificationFlag] = []

    if not tool_results:
        # No tools were called — any financial number is fabricated
        nums_in_response = _extract_numbers(response)
        financial_nums = {n for n in nums_in_response if _looks_financial(n)}
        if financial_nums:
            flags.append({
                "type": "POTENTIAL_HALLUCINATION",
                "severity": "HIGH",
                "message": (
                    f"Response contains financial numbers {financial_nums} "
                    "but no tool was called to retrieve data"
                ),
            })
        return response, flags

    # All tools failed — check if LLM guessed anyway
    if _all_tools_failed(tool_results):
        nums_in_response = _extract_numbers(response)
        financial_nums = {n for n in nums_in_response if _looks_financial(n)}
        if financial_nums:
            flags.append({
                "type": "POTENTIAL_HALLUCINATION",
                "severity": "HIGH",
                "message": (
                    f"All data tools returned errors but response still contains "
                    f"financial numbers {financial_nums} — likely guessed from training data"
                ),
            })
        return response, flags

    # At least one tool succeeded — check for unsupported numbers
    tool_nums = _extract_numbers_from_tool_results(tool_results)
    response_nums = _extract_numbers(response)

    unsupported = {
        n for n in response_nums
        if n not in tool_nums and _looks_financial(n)
    }

    if unsupported:
        flags.append({
            "type": "UNSUPPORTED_CLAIM",
            "severity": "MEDIUM",
            "message": (
                f"These values in the response could not be verified in tool results: "
                f"{unsupported}. Verify before trusting."
            ),
        })

    return response, flags


# Returns True if the number string represents a meaningful financial value (not a year or tiny count).
def _looks_financial(num_str: str) -> bool:
    """Return True if the number looks like a financial value worth flagging."""
    try:
        val = float(num_str)
        # Skip: years (2020-2030), small counts (1-9), common round percentages
        if 2010 <= val <= 2035:
            return False
        return not (val < 10 and val == int(val))
    except ValueError:
        return False


# ─── 3. Data Freshness Check ──────────────────────────────────────────────────

# Warns when any tool result's data_timestamp is older than the configured freshness threshold.
def check_freshness(
    response: str, tool_results: list[dict]
) -> tuple[str, list[VerificationFlag]]:
    """
    Inspects data_timestamp fields in tool results.
    Market data: warn if >15 minutes old.
    Portfolio data: warn if >60 minutes old.
    """
    flags: list[VerificationFlag] = []
    now = datetime.now(UTC)
    market_threshold = timedelta(minutes=settings.market_data_freshness_minutes)
    portfolio_threshold = timedelta(minutes=60)

    for result in tool_results:
        ts_str = result.get("data_timestamp")
        if not ts_str:
            flags.append({
                "type": "MISSING_TIMESTAMP",
                "severity": "LOW",
                "message": "A tool result is missing a data timestamp — freshness unknown",
            })
            continue

        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = now - ts

            # Detect if this is market data (has current_price field) or portfolio data
            is_market_data = "current_price" in result or "quotes" in result
            threshold = market_threshold if is_market_data else portfolio_threshold

            if age > threshold:
                flags.append({
                    "type": "STALE_DATA",
                    "severity": "MEDIUM",
                    "message": (
                        f"{'Market' if is_market_data else 'Portfolio'} data is "
                        f"{int(age.total_seconds() / 60)} minutes old — "
                        f"prices may not reflect current market conditions"
                    ),
                })
        except (ValueError, TypeError):
            flags.append({
                "type": "INVALID_TIMESTAMP",
                "severity": "LOW",
                "message": f"Could not parse data timestamp: {ts_str}",
            })

    return response, flags


# ─── 4. Concentration Warning ─────────────────────────────────────────────────

# Appends a concentration warning to the response if any portfolio position exceeds the threshold.
def check_concentration(
    response: str, tool_results: list[dict]
) -> tuple[str, list[VerificationFlag]]:
    """
    Scans tool results for concentration flags produced by analyze_diversification.
    Also directly inspects portfolio holdings for >threshold positions.
    Appends a concentration warning to the response if risk is detected.
    """
    flags: list[VerificationFlag] = []
    threshold = settings.portfolio_concentration_threshold

    concentration_warnings = []

    for result in tool_results:
        # From analyze_diversification tool
        if "concentration_flags" in result:
            for flag in result["concentration_flags"]:
                concentration_warnings.append(flag)
                flags.append({
                    "type": "CONCENTRATION_RISK",
                    "severity": flag.get("severity", "MEDIUM"),
                    "message": flag.get("message", ""),
                })

        # From portfolio summary — direct check
        if "holdings" in result:
            for holding in (result.get("holdings") or []):
                alloc = holding.get("allocation_percent", 0) / 100
                if alloc >= threshold:
                    symbol = holding.get("symbol", "Unknown")
                    msg = (
                        f"{symbol} is {holding['allocation_percent']}% of your portfolio "
                        f"(above {int(threshold*100)}% threshold)"
                    )
                    if not any(w.get("symbol") == symbol for w in concentration_warnings):
                        concentration_warnings.append({"symbol": symbol, "message": msg})
                        flags.append({
                            "type": "CONCENTRATION_RISK",
                            "severity": "MEDIUM",
                            "message": msg,
                        })

    if concentration_warnings:
        warning_lines = "\n".join(f"  • {w['message']}" for w in concentration_warnings)
        response += (
            f"\n\n🔔 **Concentration Risk Detected:**\n{warning_lines}\n"
            f"Consider diversifying to reduce single-position risk."
        )

    return response, flags


# ─── 5. Confidence Scoring ────────────────────────────────────────────────────

PREDICTION_KEYWORDS = {
    "will", "forecast", "predict", "expect", "future", "next year",
    "going to", "likely to", "probably", "might", "could reach",
}

HEDGING_KEYWORDS = {
    "uncertain", "unclear", "approximately", "roughly", "around", "about",
    "estimate", "may", "might", "could",
}


# Scores the agent response as HIGH, MEDIUM, or LOW confidence based on tool usage and response language.
def check_confidence(
    response: str,
    tool_results: list[dict],
    reasoning_steps: int = 1,
) -> tuple[str, list[VerificationFlag], str]:
    """
    Assigns a confidence level (HIGH / MEDIUM / LOW) to the agent's response.

    HIGH: Direct data lookup, single tool call, no predictions
    MEDIUM: Multi-tool reasoning, some inference
    LOW: Predictive claims, no tool data, or high hallucination risk

    Returns (response, flags, confidence_level)
    """
    flags: list[VerificationFlag] = []
    response_lower = response.lower()

    # Detect prediction language
    has_predictions = any(kw in response_lower for kw in PREDICTION_KEYWORDS)
    has_hedging = any(kw in response_lower for kw in HEDGING_KEYWORDS)
    has_tool_data = len(tool_results) > 0
    is_multi_step = reasoning_steps > 1 or len(tool_results) > 1

    if not has_tool_data or has_predictions:
        confidence = "LOW"
        flags.append({
            "type": "LOW_CONFIDENCE",
            "severity": "INFO",
            "message": (
                "Response involves predictions or was generated without tool data — "
                "treat with caution"
            ),
        })
    elif is_multi_step or has_hedging:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    return response, flags, confidence


# ─── Pipeline ─────────────────────────────────────────────────────────────────

# Runs all 5 checks in sequence and returns the final response, consolidated flags, and confidence level.
def run_verification_pipeline(
    response: str,
    tool_results: list[dict],
    reasoning_steps: int = 1,
) -> dict[str, Any]:
    """
    Run all 5 verification checks in order.
    Returns the (possibly modified) response, all flags, and confidence level.
    """
    all_flags: list[VerificationFlag] = []

    # 1. Disclaimer
    response, flags = check_disclaimer(response, tool_results)
    all_flags.extend(flags)

    # 2. Hallucination guard
    response, flags = check_hallucination(response, tool_results)
    all_flags.extend(flags)

    # 3. Freshness
    response, flags = check_freshness(response, tool_results)
    all_flags.extend(flags)

    # 4. Concentration
    response, flags = check_concentration(response, tool_results)
    all_flags.extend(flags)

    # 5. Confidence scoring
    response, flags, confidence = check_confidence(response, tool_results, reasoning_steps)
    all_flags.extend(flags)

    return {
        "response": response,
        "confidence": confidence,
        "verification_flags": all_flags,
        "flag_count": len(all_flags),
        "has_high_severity": any(f["severity"] == "HIGH" for f in all_flags),
    }
