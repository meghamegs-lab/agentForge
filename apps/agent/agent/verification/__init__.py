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
    "buy",
    "sell",
    "invest",
    "rebalance",
    "recommend",
    "should i",
    "allocate",
    "diversify",
    "move",
    "shift",
    "rotate",
    "switch",
    "portfolio change",
    "what to do",
    "advice",
}

DISCLAIMER_TEXT = (
    "\n\n⚠️ **Not financial advice.** This information is for educational purposes only. "
    "Always consult a qualified financial advisor before making investment decisions."
)


# Appends a financial disclaimer to the response when investment-advice language is detected.
def check_disclaimer(response: str, tool_results: list[dict]) -> tuple[str, list[VerificationFlag]]:
    """
    Appends the standard financial disclaimer whenever the response contains
    investment suggestion language. Idempotent — won't add twice.
    """
    flags: list[VerificationFlag] = []
    response_lower = response.lower()

    needs_disclaimer = any(kw in response_lower for kw in INVESTMENT_KEYWORDS)

    if needs_disclaimer and DISCLAIMER_TEXT.strip() not in response:
        response = response + DISCLAIMER_TEXT
        flags.append(
            {
                "type": "DISCLAIMER_ADDED",
                "severity": "INFO",
                "message": "Investment language detected — disclaimer appended",
            }
        )

    return response, flags


# ─── 2. Hallucination Guard ───────────────────────────────────────────────────


# Maps common large-number suffixes to their multipliers (longest/most-specific first).
_SUFFIX_MAP: list[tuple[str, float]] = [
    (r"(?:trillion|T)(?=\W|$)", 1e12),
    (r"(?:billion|B)(?=\W|$)", 1e9),
    (r"(?:million|M)(?=\W|$)", 1e6),
    (r"(?:thousand|K)(?=\W|$)", 1e3),
]


# Extracts all numeric values from text, normalising large-number suffixes to full integers.
def _extract_numbers(text: str) -> set[str]:
    """
    Extract all numeric values from text (prices, percentages, dollar amounts).

    Handles suffix notation so LLM-formatted numbers ("$3.88 trillion", "1.5B")
    are normalised to the same raw integer as the tool-result JSON value
    ("3880000000000", "1500000000") — preventing false UNSUPPORTED_CLAIM flags
    when the LLM correctly formats a large number for readability.

    Suffix-matched spans are excluded from the standard number scan so the
    base value (e.g. "3.88") is not double-counted as a separate number.
    """
    results: set[str] = set()
    suffix_spans: list[tuple[int, int]] = []  # character spans already handled by suffix rules

    for suffix_re, multiplier in _SUFFIX_MAP:
        pattern = rf"\$?([\d,]+\.?\d*)\s*{suffix_re}"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val_str = m.group(1).replace(",", "")
            try:
                full_val = float(val_str) * multiplier
                results.add(f"{full_val:.0f}")
                suffix_spans.append(m.span())
            except ValueError:
                pass

    # Standard numbers — skip spans already handled by a suffix rule above
    for m in re.finditer(r"\$?[\d,]+\.?\d*%?", text):
        if any(start <= m.start() and m.end() <= end for start, end in suffix_spans):
            continue
        clean = re.sub(r"[$,%]", "", m.group()).replace(",", "")
        if len(clean) > 1:
            results.add(clean)

    return results


# Returns True when two numeric strings are within `rel_tol` of each other (default 5 %).
def _nums_close(a: str, b: str, rel_tol: float = 0.05) -> bool:
    """
    Fuzzy-compare two numeric strings.

    Handles rounding that the LLM applies to large numbers (e.g. reporting
    volume as "171.7M" when the raw value is 171,731,114 — a 0.02 % difference
    that should NOT trigger an UNSUPPORTED_CLAIM flag).

    A 5 % relative tolerance is intentionally loose enough to absorb
    reasonable rounding but tight enough to catch a fabricated price
    (e.g. "$178.50" vs the real "$213.00" — a ~16 % difference).
    """
    try:
        fa, fb = float(a), float(b)
        if fa == 0 and fb == 0:
            return True
        if fa == 0 or fb == 0:
            return False
        return abs(fa - fb) / max(abs(fa), abs(fb)) <= rel_tol
    except ValueError:
        return False


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
    return bool(tool_results) and all(r.get("status") in failure_statuses for r in tool_results)


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
        # No tools were called — if the LLM still cites specific financial
        # numbers (dollar amounts, precise percentages), this is HIGH severity:
        # the LLM is presenting training-data guesses as verified portfolio facts.
        nums_in_response = _extract_numbers(response)
        financial_nums = {n for n in nums_in_response if _looks_financial(n)}
        if financial_nums:
            flags.append(
                {
                    "type": "POTENTIAL_HALLUCINATION",
                    "severity": "HIGH",
                    "message": (
                        f"Response contains financial numbers {financial_nums} "
                        "but no tool was called to retrieve data — "
                        "these figures are unverified and may be fabricated from training data"
                    ),
                }
            )
        return response, flags

    # All tools failed — check if LLM guessed anyway
    if _all_tools_failed(tool_results):
        nums_in_response = _extract_numbers(response)
        financial_nums = {n for n in nums_in_response if _looks_financial(n)}
        if financial_nums:
            flags.append(
                {
                    "type": "POTENTIAL_HALLUCINATION",
                    "severity": "HIGH",
                    "message": (
                        f"All data tools returned errors but response still contains "
                        f"financial numbers {financial_nums} — likely guessed from training data"
                    ),
                }
            )
        return response, flags

    # At least one tool succeeded — check for unsupported numbers
    tool_nums = _extract_numbers_from_tool_results(tool_results)
    response_nums = _extract_numbers(response)

    # A number is "unsupported" only when it cannot be traced (exactly or within 5 %)
    # to any value in the tool results.  The fuzzy match absorbs:
    #   • rounding  ("171.7M" ≈ 171,731,114)
    #   • precision differences  ("$175.32" from a tool value of "175.32")
    # while still catching fabricated values that differ by more than 5 %.
    unsupported = {
        n
        for n in response_nums
        if n not in tool_nums
        and not any(_nums_close(n, t) for t in tool_nums)
        and _looks_financial(n)
    }

    if unsupported:
        flags.append(
            {
                "type": "UNSUPPORTED_CLAIM",
                "severity": "MEDIUM",
                "message": (
                    f"These values in the response could not be verified in tool results: "
                    f"{unsupported}. Verify before trusting."
                ),
            }
        )

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


# ─── UUID / Display-name helpers ─────────────────────────────────────────────

# Matches standard UUID v4 — used to detect Ghostfolio internal IDs used as symbols.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _display_name(holding: dict) -> str:
    """
    Return the most human-readable identifier for a holding.

    Ghostfolio uses UUIDs as the 'symbol' field for manually-entered assets,
    cash positions, and unlisted holdings. This helper prefers the 'name' field
    and falls back gracefully so users never see a raw UUID in a warning.

    Output format:
      - Real ticker + name:  "Apple Inc. (AAPL)"
      - UUID symbol + name:  "Cash EUR"
      - Real ticker only:    "AAPL"
      - Both UUID / missing: "Unknown Asset"
    """
    symbol = holding.get("symbol", "")
    name = holding.get("name", "")
    symbol_is_uuid = bool(symbol and _UUID_RE.match(symbol))
    name_is_uuid = bool(name and _UUID_RE.match(name))

    has_real_symbol = bool(symbol and not symbol_is_uuid)
    has_real_name = bool(name and not name_is_uuid)

    if has_real_name and has_real_symbol:
        return f"{name} ({symbol})"
    if has_real_name:
        return name
    if has_real_symbol:
        return symbol
    # Last resort — both are UUIDs or empty
    return name or symbol or "Unknown Asset"


# ─── 3. Data Freshness Check ──────────────────────────────────────────────────


# Warns when any tool result's data_timestamp is older than the configured freshness threshold.
def check_freshness(response: str, tool_results: list[dict]) -> tuple[str, list[VerificationFlag]]:
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
            flags.append(
                {
                    "type": "MISSING_TIMESTAMP",
                    "severity": "LOW",
                    "message": "A tool result is missing a data timestamp — freshness unknown",
                }
            )
            continue

        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = now - ts

            # Detect if this is market data (has current_price field) or portfolio data
            is_market_data = "current_price" in result or "quotes" in result
            threshold = market_threshold if is_market_data else portfolio_threshold

            if age > threshold:
                flags.append(
                    {
                        "type": "STALE_DATA",
                        "severity": "MEDIUM",
                        "message": (
                            f"{'Market' if is_market_data else 'Portfolio'} data is "
                            f"{int(age.total_seconds() / 60)} minutes old — "
                            f"prices may not reflect current market conditions"
                        ),
                    }
                )
        except (ValueError, TypeError):
            flags.append(
                {
                    "type": "INVALID_TIMESTAMP",
                    "severity": "LOW",
                    "message": f"Could not parse data timestamp: {ts_str}",
                }
            )

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
                flags.append(
                    {
                        "type": "CONCENTRATION_RISK",
                        "severity": flag.get("severity", "MEDIUM"),
                        "message": flag.get("message", ""),
                    }
                )

        # From portfolio summary — direct check
        if "holdings" in result:
            for holding in result.get("holdings") or []:
                alloc = holding.get("allocation_percent", 0) / 100
                if alloc >= threshold:
                    display = _display_name(holding)
                    symbol = holding.get("symbol", "")  # raw key for dedup only
                    msg = (
                        f"{display} is {holding['allocation_percent']}% of your portfolio "
                        f"(above {int(threshold * 100)}% threshold)"
                    )
                    if not any(w.get("_key") == symbol for w in concentration_warnings):
                        concentration_warnings.append({"_key": symbol, "message": msg})
                        flags.append(
                            {
                                "type": "CONCENTRATION_RISK",
                                "severity": "MEDIUM",
                                "message": msg,
                            }
                        )

    if concentration_warnings:
        warning_lines = "\n".join(f"  • {w['message']}" for w in concentration_warnings)
        response += (
            f"\n\n🔔 **Concentration Risk Detected:**\n{warning_lines}\n"
            f"Consider diversifying to reduce single-position risk."
        )

    return response, flags


# ─── 5. Confidence Scoring ────────────────────────────────────────────────────

PREDICTION_KEYWORDS = {
    "will",
    "forecast",
    "predict",
    "expect",
    "future",
    "next year",
    "going to",
    "likely to",
    "probably",
    "might",
    "could reach",
}

HEDGING_KEYWORDS = {
    "uncertain",
    "unclear",
    "approximately",
    "roughly",
    "around",
    "about",
    "estimate",
    "may",
    "might",
    "could",
}

SPECULATIVE_KEYWORDS = {
    "bitcoin",
    "crypto",
    "cryptocurrency",
    "ethereum",
    "solana",
    "dogecoin",
    "nft",
    "meme stock",
    "all in",
    "double down",
    "yolo",
    "gamble",
    "put it all",
    "everything into",
}


# Scores the agent response as HIGH, MEDIUM, or LOW confidence based on tool usage and response language.
def check_confidence(
    response: str,
    tool_results: list[dict],
    reasoning_steps: int = 1,
) -> tuple[str, list[VerificationFlag], str]:
    """
    Assigns a confidence level (HIGH / MEDIUM / LOW) to the agent's response.

    HIGH: Tool data present, ≤2 reasoning steps (normal single or parallel lookup), no predictions
    MEDIUM: Genuine multi-round analysis (3+ reasoning steps) or hedging language
    LOW: Predictive claims, no tool data, or high hallucination risk

    Returns (response, flags, confidence_level)
    """
    flags: list[VerificationFlag] = []
    response_lower = response.lower()

    # Detect prediction language
    has_predictions = any(kw in response_lower for kw in PREDICTION_KEYWORDS)
    has_hedging = any(kw in response_lower for kw in HEDGING_KEYWORDS)
    has_speculative = any(kw in response_lower for kw in SPECULATIVE_KEYWORDS)
    has_tool_data = len(tool_results) > 0
    # "Multi-step" means the graph looped back for a second round of tool calls
    # (3+ reasoning steps).  The normal flow is always 2 steps:
    #   step 1 — LLM decides which tools to call (may call many in parallel)
    #   step 2 — LLM synthesises tool results into a final response
    # Parallel price lookups ("AAPL and MSFT prices") take exactly 2 steps and
    # should remain HIGH confidence; only genuine multi-round analysis (step 3+)
    # warrants MEDIUM.
    is_multi_step = reasoning_steps > 2

    if not has_tool_data or has_predictions or has_speculative:
        confidence = "LOW"
        flags.append(
            {
                "type": "LOW_CONFIDENCE",
                "severity": "INFO",
                "message": (
                    "Response involves predictions, speculative assets, or was generated "
                    "without tool data — treat with caution"
                ),
            }
        )
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

    # 1. Hallucination guard
    response, flags = check_hallucination(response, tool_results)
    all_flags.extend(flags)

    # 2. Freshness
    response, flags = check_freshness(response, tool_results)
    all_flags.extend(flags)

    # 3. Concentration  ← runs before disclaimer so its block appears first
    response, flags = check_concentration(response, tool_results)
    all_flags.extend(flags)

    # 4. Disclaimer  ← always last text appended, so "Not financial advice." is always the final line
    response, flags = check_disclaimer(response, tool_results)
    all_flags.extend(flags)
    disclaimer_added = any(f["type"] == "DISCLAIMER_ADDED" for f in flags)

    # 5. Confidence scoring
    response, flags, confidence = check_confidence(response, tool_results, reasoning_steps)
    all_flags.extend(flags)

    # If investment advice language triggered a disclaimer, confidence must be
    # LOW regardless of tool data or multi-step reasoning — we are not financial
    # advisors and any direct recommendation carries inherent uncertainty.
    if disclaimer_added and confidence != "LOW":
        confidence = "LOW"
        all_flags.append(
            {
                "type": "LOW_CONFIDENCE",
                "severity": "INFO",
                "message": (
                    "Response contains investment advice language (disclaimer triggered) — "
                    "confidence downgraded to LOW"
                ),
            }
        )

    return {
        "response": response,
        "confidence": confidence,
        "verification_flags": all_flags,
        "flag_count": len(all_flags),
        "has_high_severity": any(f["severity"] == "HIGH" for f in all_flags),
    }
