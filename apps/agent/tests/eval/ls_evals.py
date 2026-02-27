#!/usr/bin/env python3
"""
LangSmith Evaluation Suite — tests/eval/ls_evals.py
====================================================
Uploads example datasets to LangSmith and runs scored, tracked evaluations.
Results appear in the LangSmith UI so you can compare across runs / branches.

Eval Types
----------
1. correctness   — tool outputs match numeric ground truth            (6 examples)
2. safety        — verification pipeline flags issues correctly       (5 examples)
3. latency       — tool calls complete within time bounds             (4 examples)
4. consistency   — same inputs → identical outputs (deterministic)   (3 examples)
5. tool-keywords — tool docstrings contain LLM trigger keywords      (5 examples)

The first 4 evals use mocked HTTP (respx) — zero real network calls, no LLM cost.
The tool-keywords eval checks source code quality, also no LLM.

Prerequisites
-------------
    LANGCHAIN_API_KEY   must be set (from .env or environment)
    LANGCHAIN_PROJECT   optional — defaults to "fortio-evals"
    LANGCHAIN_TRACING_V2=true  optional — enables automatic run tracing

Run
---
    # All evals (all are fast, no LLM):
    python tests/eval/ls_evals.py

    # Single eval type:
    python tests/eval/ls_evals.py --only correctness
    python tests/eval/ls_evals.py --only safety
    python tests/eval/ls_evals.py --only latency
    python tests/eval/ls_evals.py --only consistency
    python tests/eval/ls_evals.py --only tool-keywords

    # Custom LangSmith experiment prefix (e.g. for a branch):
    python tests/eval/ls_evals.py --prefix feat/my-branch

View results
------------
    https://smith.langchain.com  → Projects → fortio-evals
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import time
from typing import Any

import httpx
import respx
import structlog

from agent.config import settings
from agent.tools.diversification import _analyze_diversification
from agent.tools.performance import _get_performance
from agent.tools.portfolio import _get_portfolio_summary
from agent.tools.transactions import _get_transactions
from agent.verification import (
    check_concentration,
    check_disclaimer,
    check_hallucination,
)

log = structlog.get_logger()

BASE_URL   = settings.ghostfolio_base_url.rstrip("/")
_AUTH_RESP = {"authToken": "ls-eval-token-abc"}


# ══════════════════════════════════════════════════════════════════════════════
# Shared Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously (fresh event loop each call)."""
    return asyncio.run(coro)


def _mock_auth(router: respx.MockRouter) -> None:
    """Register the Ghostfolio anonymous-auth endpoint on an active respx router."""
    router.post(f"{BASE_URL}/api/v1/auth/anonymous").mock(
        return_value=httpx.Response(200, json=_AUTH_RESP)
    )


def _get_ls_client():
    """Return a LangSmith Client or None if the API key is missing."""
    try:
        from langsmith import Client  # noqa: PLC0415
        return Client()
    except Exception as exc:
        print(f"⚠  LangSmith client unavailable: {exc}", file=sys.stderr)
        return None


def _get_or_create_dataset(client, name: str, description: str):
    """Return an existing LangSmith dataset or create a new one."""
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        return client.create_dataset(name, description=description)


def _upsert_examples(client, dataset_id: str, examples: list[dict]) -> None:
    """Add examples to a dataset, skipping any already present (by eval_id)."""
    existing_ids = {
        ex.metadata.get("eval_id")
        for ex in client.list_examples(dataset_id=dataset_id)
        if ex.metadata and ex.metadata.get("eval_id")
    }
    new_examples = [
        e for e in examples
        if e.get("metadata", {}).get("eval_id") not in existing_ids
    ]
    if new_examples:
        client.create_examples(
            inputs=[e["inputs"] for e in new_examples],
            outputs=[e.get("outputs", {}) for e in new_examples],
            metadata=[e.get("metadata", {}) for e in new_examples],
            dataset_id=dataset_id,
        )
    added, kept = len(new_examples), len(existing_ids)
    print(f"  ↳ dataset ready: {added} new examples added, {kept} already present")


def _avg_score(results, metric_key: str) -> tuple[float, int]:
    """
    Extract (average_score, count) from an ExperimentResults object.
    Handles different LangSmith SDK versions gracefully.
    """
    # Preferred: to_pandas() gives a clean DataFrame
    try:
        df = results.to_pandas()
        col = f"feedback.{metric_key}"
        if col in df.columns:
            series = df[col].dropna()
            return float(series.mean()), len(series)
    except Exception:
        pass

    # Fallback: iterate over results directly
    scores: list[float] = []
    try:
        for r in results:
            for er in (r.get("evaluation_results") or {}).get("results", []):
                if getattr(er, "key", None) == metric_key:
                    if er.score is not None:
                        scores.append(float(er.score))
    except Exception:
        pass

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, len(scores)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORRECTNESS EVAL
#    "Does the tool return numerically accurate data?"
#    Checks: totals, allocations, percentage conversions, fee sums, sector %
# ══════════════════════════════════════════════════════════════════════════════

_CORRECTNESS_EXAMPLES: list[dict] = [
    # ── C-1: Portfolio total value = arithmetic sum of holdings ────────────────
    {
        "metadata": {"eval_id": "corr-portfolio-total"},
        "inputs": {
            "tool": "portfolio_summary",
            "api_response": {
                "holdings": [
                    {"symbol": "AAPL", "name": "Apple",     "quantity": 10, "value": 1750.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [], "countries": []},
                    {"symbol": "VTI",  "name": "Vanguard",  "quantity": 20, "value": 4200.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                     "sectors": [], "countries": []},
                    {"symbol": "MSFT", "name": "Microsoft", "quantity": 5,  "value": 2050.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [], "countries": []},
                ]
            },
        },
        "outputs": {"total_value": 8000.00, "position_count": 3},
    },
    # ── C-2: Individual allocation percentages are correct ─────────────────────
    {
        "metadata": {"eval_id": "corr-allocation-pct"},
        "inputs": {
            "tool": "portfolio_summary",
            "api_response": {
                "holdings": [
                    {"symbol": "AAPL", "name": "Apple",   "quantity": 10, "value": 1000.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [], "countries": []},
                    {"symbol": "VTI",  "name": "Vanguard", "quantity": 30, "value": 3000.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                     "sectors": [], "countries": []},
                ]
            },
        },
        "outputs": {"aapl_alloc_pct": 25.00, "vti_alloc_pct": 75.00},
    },
    # ── C-3: Performance: raw fraction → percentage (positive) ─────────────────
    {
        "metadata": {"eval_id": "corr-perf-positive"},
        "inputs": {
            "tool": "performance",
            "date_range": "ytd",
            "api_response": {
                # Flat v2 API format — no nested period keys
                "performance": {
                    "netPerformancePercentage": 0.1234,
                    "netPerformance": 987.65,
                    "currentValueInBaseCurrency": 8987.65,
                    "totalInvestment": 8000.00,
                    "currentNetWorth": 8987.65,
                }
            },
        },
        "outputs": {"relative_change_pct": 12.34},
    },
    # ── C-4: Performance: negative returns preserved correctly ─────────────────
    {
        "metadata": {"eval_id": "corr-perf-negative"},
        "inputs": {
            "tool": "performance",
            "date_range": "ytd",
            "api_response": {
                # Flat v2 API format — no nested period keys
                "performance": {
                    "netPerformancePercentage": -0.085,
                    "netPerformance": -750.00,
                    "currentValueInBaseCurrency": 8150.00,
                    "totalInvestment": 8900.00,
                    "currentNetWorth": 8150.00,
                }
            },
        },
        "outputs": {"relative_change_pct": -8.50},
    },
    # ── C-5: Transactions: fee total = exact sum ───────────────────────────────
    {
        "metadata": {"eval_id": "corr-tx-fees"},
        "inputs": {
            "tool": "transactions",
            "api_response": {
                "activities": [
                    {"id": "t1", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                     "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                     "quantity": 5, "unitPrice": 170.00, "fee": 4.99,
                     "currency": "USD", "Account": {"name": "Brokerage"}},
                    {"id": "t2", "date": "2024-02-01T00:00:00Z", "type": "BUY",
                     "SymbolProfile": {"symbol": "MSFT", "name": "Microsoft"},
                     "quantity": 2, "unitPrice": 400.00, "fee": 1.50,
                     "currency": "USD", "Account": {"name": "Brokerage"}},
                    {"id": "t3", "date": "2024-03-01T00:00:00Z", "type": "DIVIDEND",
                     "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                     "quantity": 5, "unitPrice": 0.25, "fee": 0.00,
                     "currency": "USD", "Account": {"name": "Brokerage"}},
                ]
            },
        },
        "outputs": {"total_fees_paid": 6.49},
    },
    # ── C-6: Diversification: sector weights rolled up correctly ───────────────
    {
        "metadata": {"eval_id": "corr-diversification-sectors"},
        "inputs": {
            "tool": "diversification",
            "api_response": {
                "holdings": [
                    {"symbol": "AAPL", "name": "Apple",   "quantity": 10, "value": 2000.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [{"name": "Technology", "weight": 1.0}],
                     "countries": [{"name": "United States", "weight": 1.0}]},
                    {"symbol": "VTI",  "name": "Vanguard", "quantity": 10, "value": 2000.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                     "sectors": [{"name": "Technology", "weight": 0.5},
                                 {"name": "Healthcare",  "weight": 0.5}],
                     "countries": [{"name": "United States", "weight": 1.0}]},
                ]
            },
        },
        "outputs": {"technology_pct": 75.0, "healthcare_pct": 25.0},
    },
]


def _correctness_target(inputs: dict) -> dict:
    """
    Run the specified tool against a mocked Ghostfolio API.
    Returns a flat dict of key metrics for the evaluator to score.
    """
    tool     = inputs["tool"]
    api_resp = inputs["api_response"]

    with respx.mock() as router:
        _mock_auth(router)

        if tool == "portfolio_summary":
            router.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            result = _run(_get_portfolio_summary())
            by_sym = {h["symbol"]: h.get("allocation_percent", 0)
                      for h in result.get("holdings", [])}
            return {
                "status":         result.get("status"),
                "total_value":    result.get("total_value"),
                "position_count": result.get("position_count"),
                # Per-symbol allocation keys, e.g. "aapl_alloc_pct"
                **{f"{sym.lower()}_alloc_pct": pct for sym, pct in by_sym.items()},
            }

        if tool == "performance":
            router.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            result = _run(_get_performance(inputs.get("date_range", "ytd")))
            perf   = result.get("performance", {})
            return {
                "status":              result.get("status"),
                "relative_change_pct": perf.get("relative_change_pct"),
                "absolute_change":     perf.get("absolute_change"),
            }

        if tool == "transactions":
            router.get(f"{BASE_URL}/api/v1/order").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            result  = _run(_get_transactions())
            summary = result.get("summary", {})
            return {
                "status":          result.get("status"),
                "total_fees_paid": summary.get("total_fees_paid"),
            }

        if tool == "diversification":
            router.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            result  = _run(_analyze_diversification())
            sectors = {s["name"]: s["percent"]
                       for s in result.get("sector_breakdown", [])}
            return {
                "status": result.get("status"),
                **{f"{name.lower().replace(' ', '_')}_pct": pct
                   for name, pct in sectors.items()},
            }

        return {"status": "error", "error": f"Unknown tool: {tool}"}


def _correctness_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """
    Score = fraction of reference fields that match within tolerance.
    Numeric: tolerance ±0.1    |    String/bool: exact match.
    """
    if outputs.get("status") == "error":
        return {"key": "correctness", "score": 0.0}

    scores: list[float] = []
    for key, expected in reference_outputs.items():
        actual = outputs.get(key)
        if actual is None:
            scores.append(0.0)
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            scores.append(1.0 if abs(actual - expected) <= 0.1 else 0.0)
        else:
            scores.append(1.0 if actual == expected else 0.0)

    score = sum(scores) / len(scores) if scores else 0.0
    return {"key": "correctness", "score": score}


def run_correctness_eval(client, experiment_prefix: str = "fortio") -> None:
    from langsmith import evaluate as ls_evaluate  # noqa: PLC0415

    print("\n── 1. Correctness Eval ──────────────────────────────────────────────────")
    ds = _get_or_create_dataset(
        client, "fortio-correctness",
        "Ground-truth numeric accuracy for Fortio tool outputs"
    )
    _upsert_examples(client, ds.id, _CORRECTNESS_EXAMPLES)

    results = ls_evaluate(
        _correctness_target,
        data="fortio-correctness",
        evaluators=[_correctness_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )
    avg, count = _avg_score(results, "correctness")
    print(f"  ✅ Correctness score: {avg:.1%}  ({count} examples scored)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SAFETY EVAL
#    "Does the verification pipeline catch issues correctly?"
#    Checks: disclaimer injection, hallucination guard, concentration flag
# ══════════════════════════════════════════════════════════════════════════════

_SAFETY_EXAMPLES: list[dict] = [
    # ── S-1: Investment-advice language triggers disclaimer ────────────────────
    {
        "metadata": {"eval_id": "safety-disclaimer-triggers"},
        "inputs": {
            "check": "disclaimer",
            "response": "You should rebalance your portfolio toward bonds right now.",
            "tool_results": [],
        },
        "outputs": {"disclaimer_added": True, "expected_flag": "DISCLAIMER_ADDED"},
    },
    # ── S-2: Factual statement does NOT trigger disclaimer ─────────────────────
    {
        "metadata": {"eval_id": "safety-disclaimer-no-trigger"},
        "inputs": {
            "check": "disclaimer",
            "response": "Your portfolio is currently worth $10,000.",
            "tool_results": [],
        },
        "outputs": {"disclaimer_added": False},
    },
    # ── S-3: Number not in tool results → hallucination flagged ───────────────
    {
        "metadata": {"eval_id": "safety-hallucination-flagged"},
        "inputs": {
            "check": "hallucination",
            "response": "Your portfolio is worth $99,999.99 today.",
            "tool_results": [
                {"current_value": 1000.00, "data_timestamp": "2024-01-01T00:00:00Z"}
            ],
        },
        "outputs": {"hallucination_flagged": True},
    },
    # ── S-4: Number matches tool result → no hallucination flag ───────────────
    {
        "metadata": {"eval_id": "safety-hallucination-clean"},
        "inputs": {
            "check": "hallucination",
            "response": "Your portfolio value is $1,750.50.",
            "tool_results": [
                {"current_value": 1750.50, "data_timestamp": "2024-01-01T00:00:00Z"}
            ],
        },
        "outputs": {"hallucination_flagged": False},
    },
    # ── S-5: Single position >25 % → concentration risk flagged ───────────────
    {
        "metadata": {"eval_id": "safety-concentration-high"},
        "inputs": {
            "check": "concentration",
            "response": "Your portfolio looks healthy.",
            "tool_results": [
                {
                    "holdings": [
                        {"symbol": "AAPL", "allocation_percent": 85.0},
                        {"symbol": "VTI",  "allocation_percent": 15.0},
                    ]
                }
            ],
        },
        "outputs": {"concentration_flagged": True},
    },
]


def _safety_target(inputs: dict) -> dict:
    """Run one verification check and return boolean flag indicators."""
    check       = inputs["check"]
    response    = inputs["response"]
    tool_results = inputs.get("tool_results", [])

    if check == "disclaimer":
        _, flags = check_disclaimer(response, tool_results)
        return {
            "disclaimer_added":   any(f["type"] == "DISCLAIMER_ADDED" for f in flags),
            "flag_types":         [f["type"] for f in flags],
        }

    if check == "hallucination":
        _, flags = check_hallucination(response, tool_results)
        return {
            "hallucination_flagged": any(f["type"] == "POTENTIAL_HALLUCINATION" for f in flags),
            "flag_types":            [f["type"] for f in flags],
        }

    if check == "concentration":
        _, flags = check_concentration(response, tool_results)
        return {
            "concentration_flagged": any(f["type"] == "CONCENTRATION_RISK" for f in flags),
            "flag_types":            [f["type"] for f in flags],
        }

    return {"error": f"Unknown check type: {check}"}


def _safety_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Score = fraction of expected boolean fields that match."""
    bool_keys = ["disclaimer_added", "hallucination_flagged", "concentration_flagged"]
    scores: list[float] = []

    for key in bool_keys:
        if key in reference_outputs:
            scores.append(1.0 if outputs.get(key) == reference_outputs[key] else 0.0)

    # Also check that the expected flag type actually appeared
    if "expected_flag" in reference_outputs:
        expected = reference_outputs["expected_flag"]
        actual   = outputs.get("flag_types", [])
        scores.append(1.0 if expected in actual else 0.0)

    score = sum(scores) / len(scores) if scores else 0.0
    return {"key": "safety", "score": score}


def run_safety_eval(client, experiment_prefix: str = "fortio") -> None:
    from langsmith import evaluate as ls_evaluate  # noqa: PLC0415

    print("\n── 2. Safety Eval ───────────────────────────────────────────────────────")
    ds = _get_or_create_dataset(
        client, "fortio-safety",
        "Verification pipeline correctness — disclaimer, hallucination, concentration"
    )
    _upsert_examples(client, ds.id, _SAFETY_EXAMPLES)

    results = ls_evaluate(
        _safety_target,
        data="fortio-safety",
        evaluators=[_safety_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )
    avg, count = _avg_score(results, "safety")
    print(f"  ✅ Safety score: {avg:.1%}  ({count} examples scored)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LATENCY EVAL
#    "Do tool calls complete within acceptable time bounds?"
#    Threshold: each tool call must finish in ≤ 2 s on mocked I/O.
# ══════════════════════════════════════════════════════════════════════════════

_LATENCY_EXAMPLES: list[dict] = [
    {
        "metadata": {"eval_id": "latency-portfolio"},
        "inputs": {
            "tool": "portfolio_summary",
            "api_response": {
                "holdings": [
                    {"symbol": "AAPL", "name": "Apple", "quantity": 10, "value": 1750.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [], "countries": []},
                ]
            },
            "max_seconds": 2.0,
        },
        "outputs": {"within_bound": True},
    },
    {
        "metadata": {"eval_id": "latency-performance"},
        "inputs": {
            "tool": "performance",
            "date_range": "ytd",
            "api_response": {
                # Flat v2 API format
                "performance": {
                    "netPerformancePercentage": 0.12,
                    "netPerformance": 900.0,
                    "currentValueInBaseCurrency": 8900.0,
                    "totalInvestment": 8000.0,
                    "currentNetWorth": 8900.0,
                }
            },
            "max_seconds": 2.0,
        },
        "outputs": {"within_bound": True},
    },
    {
        "metadata": {"eval_id": "latency-transactions"},
        "inputs": {
            "tool": "transactions",
            "api_response": {
                "activities": [
                    {"id": "t1", "date": "2024-01-01T00:00:00Z", "type": "BUY",
                     "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                     "quantity": 10, "unitPrice": 170.00, "fee": 4.99,
                     "currency": "USD", "Account": {"name": "Brokerage"}},
                ]
            },
            "max_seconds": 2.0,
        },
        "outputs": {"within_bound": True},
    },
    {
        "metadata": {"eval_id": "latency-diversification"},
        "inputs": {
            "tool": "diversification",
            "api_response": {
                "holdings": [
                    {"symbol": "AAPL", "name": "Apple", "quantity": 10, "value": 2000.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [{"name": "Technology", "weight": 1.0}],
                     "countries": [{"name": "United States", "weight": 1.0}]},
                ]
            },
            "max_seconds": 2.0,
        },
        "outputs": {"within_bound": True},
    },
]


def _latency_target(inputs: dict) -> dict:
    """Run the tool, time it, and return latency metrics."""
    tool     = inputs["tool"]
    api_resp = inputs["api_response"]
    max_sec  = inputs.get("max_seconds", 2.0)

    t0 = time.perf_counter()

    with respx.mock() as router:
        _mock_auth(router)

        if tool == "portfolio_summary":
            router.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            _run(_get_portfolio_summary())

        elif tool == "performance":
            router.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            _run(_get_performance(inputs.get("date_range", "ytd")))

        elif tool == "transactions":
            router.get(f"{BASE_URL}/api/v1/order").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            _run(_get_transactions())

        elif tool == "diversification":
            router.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            _run(_analyze_diversification())

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": round(elapsed, 4),
        "within_bound":    elapsed <= max_sec,
        "max_seconds":     max_sec,
    }


def _latency_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Score 1.0 if within bound, 0.0 otherwise. Comment includes actual time."""
    within  = outputs.get("within_bound", False)
    elapsed = outputs.get("elapsed_seconds", -1)
    return {
        "key":     "latency_ok",
        "score":   1.0 if within else 0.0,
        "comment": f"{elapsed:.3f}s",
    }


def run_latency_eval(client, experiment_prefix: str = "fortio") -> None:
    from langsmith import evaluate as ls_evaluate  # noqa: PLC0415

    print("\n── 3. Latency Eval ──────────────────────────────────────────────────────")
    ds = _get_or_create_dataset(
        client, "fortio-latency",
        "Tool call latency within acceptable bounds (≤2 s on mocked I/O)"
    )
    _upsert_examples(client, ds.id, _LATENCY_EXAMPLES)

    results = ls_evaluate(
        _latency_target,
        data="fortio-latency",
        evaluators=[_latency_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )
    avg, count = _avg_score(results, "latency_ok")
    print(f"  ✅ Latency score: {avg:.1%}  ({count} examples scored)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSISTENCY EVAL
#    "Does the same input always produce the same output?"
#    Runs each tool twice with identical mocked data; expects bit-for-bit match.
# ══════════════════════════════════════════════════════════════════════════════

_CONSISTENCY_EXAMPLES: list[dict] = [
    {
        "metadata": {"eval_id": "consistency-portfolio"},
        "inputs": {
            "tool": "portfolio_summary",
            "api_response": {
                "holdings": [
                    {"symbol": "AAPL", "name": "Apple",   "quantity": 10, "value": 1750.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "STOCK",
                     "sectors": [], "countries": []},
                    {"symbol": "VTI",  "name": "Vanguard", "quantity": 20, "value": 4200.00,
                     "currency": "USD", "assetClass": "EQUITY", "assetSubClass": "ETF",
                     "sectors": [], "countries": []},
                ]
            },
        },
        "outputs": {"deterministic": True},
    },
    {
        "metadata": {"eval_id": "consistency-performance"},
        "inputs": {
            "tool": "performance",
            "date_range": "ytd",
            "api_response": {
                # Flat v2 API format
                "performance": {
                    "netPerformancePercentage": 0.12,
                    "netPerformance": 900.0,
                    "currentValueInBaseCurrency": 8900.0,
                    "totalInvestment": 8000.0,
                    "currentNetWorth": 8900.0,
                }
            },
        },
        "outputs": {"deterministic": True},
    },
    {
        "metadata": {"eval_id": "consistency-transactions"},
        "inputs": {
            "tool": "transactions",
            "api_response": {
                "activities": [
                    {"id": "t1", "date": "2024-06-01T00:00:00Z", "type": "BUY",
                     "SymbolProfile": {"symbol": "AAPL", "name": "Apple"},
                     "quantity": 10, "unitPrice": 170.00, "fee": 4.99,
                     "currency": "USD", "Account": {"name": "Brokerage"}},
                ]
            },
        },
        "outputs": {"deterministic": True},
    },
]


def _run_tool_once(tool: str, api_resp: dict, date_range: str = "ytd") -> dict:
    """Run one tool call with mocked API. Helper for the consistency target."""
    with respx.mock() as router:
        _mock_auth(router)
        if tool == "portfolio_summary":
            router.get(f"{BASE_URL}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            return _run(_get_portfolio_summary())
        if tool == "performance":
            router.get(f"{BASE_URL}/api/v2/portfolio/performance").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            return _run(_get_performance(date_range))
        if tool == "transactions":
            router.get(f"{BASE_URL}/api/v1/order").mock(
                return_value=httpx.Response(200, json=api_resp)
            )
            return _run(_get_transactions())
    return {"status": "error"}


def _strip_volatile(d: dict) -> dict:
    """Remove non-deterministic timestamp fields before comparing."""
    return {k: v for k, v in d.items()
            if k not in {"data_timestamp", "generated_at", "retrieved_at"}}


def _consistency_target(inputs: dict) -> dict:
    """Run the same tool twice with identical mock data; compare both results."""
    tool       = inputs["tool"]
    api_resp   = inputs["api_response"]
    date_range = inputs.get("date_range", "ytd")

    run1 = _strip_volatile(_run_tool_once(tool, api_resp, date_range))
    run2 = _strip_volatile(_run_tool_once(tool, api_resp, date_range))

    diff_keys = [k for k in set(list(run1) + list(run2)) if run1.get(k) != run2.get(k)]
    return {
        "deterministic": run1 == run2,
        "diff_keys":     diff_keys,
        "run1_status":   run1.get("status"),
        "run2_status":   run2.get("status"),
    }


def _consistency_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    deterministic = outputs.get("deterministic", False)
    diff_keys     = outputs.get("diff_keys", [])
    comment       = "ok" if deterministic else f"differs on: {diff_keys}"
    return {"key": "deterministic", "score": 1.0 if deterministic else 0.0, "comment": comment}


def run_consistency_eval(client, experiment_prefix: str = "fortio") -> None:
    from langsmith import evaluate as ls_evaluate  # noqa: PLC0415

    print("\n── 4. Consistency Eval ──────────────────────────────────────────────────")
    ds = _get_or_create_dataset(
        client, "fortio-consistency",
        "Same inputs always produce identical tool outputs"
    )
    _upsert_examples(client, ds.id, _CONSISTENCY_EXAMPLES)

    results = ls_evaluate(
        _consistency_target,
        data="fortio-consistency",
        evaluators=[_consistency_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )
    avg, count = _avg_score(results, "deterministic")
    print(f"  ✅ Consistency score: {avg:.1%}  ({count} examples scored)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TOOL KEYWORDS EVAL
#    "Do tool docstrings contain the trigger keywords the LLM needs?"
#    Each tool's description must include words the LLM uses to decide which
#    tool to invoke.  Missing keywords → wrong tool selected → silent failure.
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_KEYWORD_EXAMPLES: list[dict] = [
    {
        "metadata": {"eval_id": "keywords-portfolio"},
        "inputs": {
            "tool_module": "agent.tools.portfolio",
            "tool_attr":   "get_portfolio_summary",
        },
        "outputs": {
            "required_keywords": ["holdings", "allocation", "portfolio", "positions", "value"]
        },
    },
    {
        "metadata": {"eval_id": "keywords-performance"},
        "inputs": {
            "tool_module": "agent.tools.performance",
            "tool_attr":   "get_performance",
        },
        "outputs": {
            "required_keywords": ["performance", "return", "gain", "loss", "period"]
        },
    },
    {
        "metadata": {"eval_id": "keywords-transactions"},
        "inputs": {
            "tool_module": "agent.tools.transactions",
            "tool_attr":   "get_transactions",
        },
        "outputs": {
            "required_keywords": ["transaction", "history", "fee", "dividend", "buy", "sell"]
        },
    },
    {
        "metadata": {"eval_id": "keywords-diversification"},
        "inputs": {
            "tool_module": "agent.tools.diversification",
            "tool_attr":   "analyze_diversification",
        },
        "outputs": {
            "required_keywords": [
                "diversification", "concentration", "sector", "geographic", "rebalancing"
            ]
        },
    },
    {
        "metadata": {"eval_id": "keywords-market"},
        "inputs": {
            "tool_module": "agent.tools.market",
            "tool_attr":   "get_market_data",
        },
        "outputs": {
            "required_keywords": ["price", "stock", "symbol", "market", "52"]
        },
    },
]


def _tool_keywords_target(inputs: dict) -> dict:
    """Import the named tool and extract its description / docstring."""
    module  = importlib.import_module(inputs["tool_module"])
    tool_fn = getattr(module, inputs["tool_attr"])
    doc     = (getattr(tool_fn, "description", None) or tool_fn.__doc__ or "").lower()
    return {"doc": doc, "tool_name": inputs["tool_attr"]}


def _tool_keywords_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Score = fraction of required keywords present in the docstring."""
    doc      = outputs.get("doc", "")
    required = reference_outputs.get("required_keywords", [])
    if not required:
        return {"key": "keyword_coverage", "score": 0.0}

    hits    = [kw for kw in required if kw in doc]
    missing = [kw for kw in required if kw not in doc]
    score   = len(hits) / len(required)
    comment = f"{len(hits)}/{len(required)} keywords found"
    if missing:
        comment += f"; missing: {missing}"

    return {"key": "keyword_coverage", "score": score, "comment": comment}


def run_tool_keywords_eval(client, experiment_prefix: str = "fortio") -> None:
    from langsmith import evaluate as ls_evaluate  # noqa: PLC0415

    print("\n── 5. Tool Keywords Eval ────────────────────────────────────────────────")
    ds = _get_or_create_dataset(
        client, "fortio-tool-keywords",
        "Tool docstrings contain required LLM trigger keywords for correct tool selection"
    )
    _upsert_examples(client, ds.id, _TOOL_KEYWORD_EXAMPLES)

    results = ls_evaluate(
        _tool_keywords_target,
        data="fortio-tool-keywords",
        evaluators=[_tool_keywords_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )
    avg, count = _avg_score(results, "keyword_coverage")
    print(f"  ✅ Tool Keywords score: {avg:.1%}  ({count} examples scored)")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

ALL_EVALS: dict[str, Any] = {
    "correctness":   run_correctness_eval,
    "safety":        run_safety_eval,
    "latency":       run_latency_eval,
    "consistency":   run_consistency_eval,
    "tool-keywords": run_tool_keywords_eval,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LangSmith-tracked evaluations for the Fortio agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/eval/ls_evals.py
  python tests/eval/ls_evals.py --only correctness
  python tests/eval/ls_evals.py --only safety
  python tests/eval/ls_evals.py --only latency
  python tests/eval/ls_evals.py --only consistency
  python tests/eval/ls_evals.py --only tool-keywords
  python tests/eval/ls_evals.py --prefix feat/my-branch
        """.strip(),
    )
    parser.add_argument(
        "--only",
        choices=list(ALL_EVALS.keys()),
        metavar="TYPE",
        help=f"Run only this eval type. Choices: {', '.join(ALL_EVALS.keys())}",
    )
    parser.add_argument(
        "--prefix",
        default="fortio",
        help="LangSmith experiment name prefix (default: fortio)",
    )
    args = parser.parse_args()

    # Validate API key
    api_key = (
        os.environ.get("LANGCHAIN_API_KEY")
        or os.environ.get("LANGSMITH_API_KEY")
    )
    if not api_key:
        print(
            "❌  LANGCHAIN_API_KEY not set.\n"
            "   Add it to apps/agent/.env:  LANGCHAIN_API_KEY=lsv2_pt_...\n"
            "   Or export it:               export LANGCHAIN_API_KEY=lsv2_pt_...",
            file=sys.stderr,
        )
        return 1

    # Default project name for LangSmith grouping
    project = os.environ.get("LANGCHAIN_PROJECT", "fortio-evals")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)

    client = _get_ls_client()
    if client is None:
        return 1

    to_run = {args.only: ALL_EVALS[args.only]} if args.only else dict(ALL_EVALS)

    print(f"\n{'═' * 68}")
    print(f"  Fortio LangSmith Evals")
    print(f"  Project  : {project}")
    print(f"  Prefix   : {args.prefix}")
    print(f"  Running  : {', '.join(to_run.keys())}")
    print(f"{'═' * 68}")

    errors: list[str] = []
    for name, fn in to_run.items():
        try:
            fn(client, experiment_prefix=args.prefix)
        except Exception as exc:
            print(f"  ❌ {name} failed: {exc}", file=sys.stderr)
            errors.append(name)

    print(f"\n{'═' * 68}")
    if errors:
        print(f"  ⚠  {len(errors)} eval(s) encountered errors: {errors}")
        print(f"  Partial results may still appear in LangSmith.")
        return 1
    else:
        print(f"  ✅ All {len(to_run)} eval(s) completed successfully.")
        print(f"  View results → https://smith.langchain.com")
        print(f"                 Projects → {project}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
