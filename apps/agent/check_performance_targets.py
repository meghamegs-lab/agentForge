#!/usr/bin/env python3
"""
check_performance_targets.py  ─  Fortio Agent Performance Target Checker
=========================================================================
Validates all 6 SLO targets against the existing eval suite.

  Target 1  End-to-end latency     < 5 s    single-tool  (measured with mocked HTTP)
  Target 2  Multi-step latency     < 15 s   3+ tool chain (measured with mocked HTTP)
  Target 3  Tool success rate      > 95 %   tool-exercising tests pass
  Target 4  Eval pass rate         > 80 %   overall eval suite pass rate
  Target 5  Hallucination rate     < 5 %    adversarial inputs NOT caught by verifier
  Target 6  Verification accuracy  > 90 %   safety + adversarial tests pass rate

Run from apps/agent/:
    python check_performance_targets.py            # full suite
    python check_performance_targets.py --fast     # skip latency measurement
    python check_performance_targets.py --no-llm   # exclude LLM-in-the-loop tests
    python check_performance_targets.py --verbose  # print full pytest output

Notes
-----
  - Latency is measured with fully mocked HTTP (respx). Real API calls add ~500ms–2s
    per call. Mocked proxy budgets: single <200ms, multi-step <1000ms.
  - Hallucination rate = fraction of adversarial inputs the verification layer
    FAILED to catch (ADV tests that FAIL = a missed detection).
  - Verification accuracy = fraction of safety + adversarial tests that PASS.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ROOT = Path(__file__).parent  # apps/agent/


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    name: str
    value: float            # actual measured value
    threshold: float        # SLO threshold
    unit: str               # "s" | "%"
    higher_is_better: bool  # True for rates, False for latency/error rates
    detail: str = ""
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.higher_is_better:
            self.passed = self.value >= self.threshold
        else:
            self.passed = self.value <= self.threshold

    def _value_str(self) -> str:
        if self.unit == "s":
            return f"{self.value:.3f}s"
        return f"{self.value:.1f}%"

    def _threshold_str(self) -> str:
        direction = "≥" if self.higher_is_better else "≤"
        if self.unit == "s":
            return f"{direction} {self.threshold}s"
        return f"{direction} {self.threshold}%"

    def status_line(self) -> str:
        icon  = f"{GREEN}✅{RESET}" if self.passed else f"{RED}❌{RESET}"
        label = f"{BOLD}{self.name:<37}{RESET}"
        val   = f"{self._value_str():>10}"
        thr   = f"(target {self._threshold_str()})"
        det   = f"  [{self.detail}]" if self.detail else ""
        return f"  {icon}  {label}  {val}  {thr}{det}"


# ── Step 1: Run pytest and parse results ──────────────────────────────────────

def _run_pytest(extra_args: list[str] | None = None) -> dict:
    """
    Run the full eval suite via subprocess.
    Returns a dict with counts and per-nodeid pass/fail booleans.
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/evals/",
        "-v", "--tb=no",
        "--no-header",
        "--override-ini=addopts=",  # strip pytest.ini addopts (removes coverage overhead)
    ] + (extra_args or [])

    print(f"  $ {' '.join(cmd[2:])}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT
    )
    stdout = result.stdout + result.stderr

    # Parse individual test outcomes from -v output
    # Format: "tests/evals/test_foo.py::test_bar PASSED"
    test_results: dict[str, bool] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if " PASSED" in stripped:
            node = stripped.split(" PASSED")[0].strip()
            if "::" in node:
                test_results[node] = True
        elif " FAILED" in stripped or " ERROR" in stripped:
            node = stripped.split(" FAILED")[0].split(" ERROR")[0].strip()
            if "::" in node:
                test_results[node] = False

    passed = sum(1 for v in test_results.values() if v)
    failed = sum(1 for v in test_results.values() if not v)
    total  = len(test_results)

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0.0,
        "test_results": test_results,
        "stdout": stdout,
        "returncode": result.returncode,
    }


def _filter_by_files(
    test_results: dict[str, bool], file_names: set[str]
) -> tuple[int, int]:
    """Return (passed, total) for tests whose file name is in file_names."""
    passed = total = 0
    for nodeid, ok in test_results.items():
        fname = nodeid.split("/")[-1].split("::")[0].replace(".py", "")
        if fname in file_names:
            total += 1
            if ok:
                passed += 1
    return passed, total


# ── Step 2: Latency measurement ───────────────────────────────────────────────

def _make_fixtures() -> tuple[dict, dict]:
    """Return shared mock API response fixtures."""
    holdings = {
        "holdings": [
            {
                "symbol": "AAPL", "name": "Apple Inc.", "quantity": 10,
                "valueInBaseCurrency": 1750.00, "currency": "USD",
                "assetClass": "EQUITY", "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "VTI", "name": "Vanguard ETF", "quantity": 20,
                "valueInBaseCurrency": 4200.00, "currency": "USD",
                "assetClass": "EQUITY", "assetSubClass": "ETF",
                "sectors": [
                    {"name": "Technology", "weight": 0.30},
                    {"name": "Other", "weight": 0.70},
                ],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
            {
                "symbol": "MSFT", "name": "Microsoft", "quantity": 5,
                "valueInBaseCurrency": 2050.00, "currency": "USD",
                "assetClass": "EQUITY", "assetSubClass": "STOCK",
                "sectors": [{"name": "Technology", "weight": 1.0}],
                "countries": [{"name": "United States", "weight": 1.0}],
            },
        ]
    }
    perf = {
        "performance": {
            "netPerformancePercentage": 0.1234,
            "netPerformance": 987.65,
            "currentValueInBaseCurrency": 8987.65,
            "totalInvestment": 8000.00,
            "currentNetWorth": 8987.65,
        }
    }
    return holdings, perf


def _timed_runs(coro_factory, n: int = 3) -> float:
    """Run `coro_factory()` n times synchronously, return median elapsed seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        coro_factory()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[n // 2]  # median


def _measure_latency() -> tuple[MetricResult, MetricResult]:
    """
    Time two representative tool calls with fully mocked HTTP (respx).

    Mocked proxy budgets map to production targets:
      Single-tool  mocked ≤ 200ms  →  production target < 5s
      Multi-step   mocked ≤ 1000ms →  production target < 15s

    The mocked budget is intentionally strict so that if Python overhead alone
    is too high, we catch it before it reaches production.
    """
    import httpx
    import respx
    from agent.config import settings

    BASE = settings.ghostfolio_base_url.rstrip("/")
    AUTH = {"authToken": "perf-check-token"}
    holdings_fixture, perf_fixture = _make_fixtures()

    # ── Single-tool: get_portfolio_summary (1 Ghostfolio API call) ────────────
    from agent.tools.portfolio import _get_portfolio_summary

    def _run_single():
        with respx.mock() as r:
            r.post(f"{BASE}/api/v1/auth/anonymous").mock(
                return_value=httpx.Response(200, json=AUTH)
            )
            r.get(f"{BASE}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=holdings_fixture)
            )
            asyncio.run(_get_portfolio_summary())

    t_single = _timed_runs(_run_single)

    # ── Multi-step: health scorecard (3+ internal API calls + computation) ────
    from agent.tools.health_scorecard import _scorecard

    def _run_multi():
        with respx.mock() as r:
            r.post(f"{BASE}/api/v1/auth/anonymous").mock(
                return_value=httpx.Response(200, json=AUTH)
            )
            r.get(f"{BASE}/api/v1/portfolio/holdings").mock(
                return_value=httpx.Response(200, json=holdings_fixture)
            )
            r.get(f"{BASE}/api/v2/portfolio/performance").mock(
                return_value=httpx.Response(200, json=perf_fixture)
            )
            asyncio.run(_scorecard())

    t_multi = _timed_runs(_run_multi)

    return (
        MetricResult(
            name="End-to-end latency (single-tool)",
            value=round(t_single, 3),
            threshold=0.200,  # 200ms mocked proxy → <5s production
            unit="s",
            higher_is_better=False,
            detail="mocked HTTP; production target <5s",
        ),
        MetricResult(
            name="Multi-step latency (3+ tools)",
            value=round(t_multi, 3),
            threshold=1.000,  # 1s mocked proxy → <15s production
            unit="s",
            higher_is_better=False,
            detail="mocked HTTP; production target <15s",
        ),
    )


# ── Steps 3-6: Derived metrics from pytest results ────────────────────────────

def _metric_tool_success(tr: dict[str, bool]) -> MetricResult:
    """
    Tool success rate: tests that directly exercise tool execution.
    Files: test_correctness, test_tool_execution, test_multi_step, test_edge_cases.
    """
    passed, total = _filter_by_files(tr, {
        "test_correctness", "test_tool_execution",
        "test_multi_step", "test_edge_cases",
    })
    rate = (passed / total * 100) if total else 0.0
    return MetricResult(
        name="Tool success rate",
        value=round(rate, 1),
        threshold=95.0,
        unit="%",
        higher_is_better=True,
        detail=f"{passed}/{total} tool tests passed",
    )


def _metric_eval_pass_rate(counts: dict) -> MetricResult:
    """Overall eval suite pass rate across all tests/evals/ files."""
    rate = counts["pass_rate"] * 100
    return MetricResult(
        name="Eval pass rate",
        value=round(rate, 1),
        threshold=80.0,
        unit="%",
        higher_is_better=True,
        detail=f"{counts['passed']}/{counts['total']} tests passed",
    )


def _metric_hallucination_rate(tr: dict[str, bool]) -> MetricResult:
    """
    Hallucination rate = fraction of adversarial inputs NOT caught by the
    verification layer.

      ADV test PASSES → guard correctly detected the attack  ✅
      ADV test FAILS  → guard let the hallucination through  ❌ (a miss)

    rate = (ADV tests that FAILED / total ADV tests) × 100
    Target: < 5%
    """
    passed, total = _filter_by_files(tr, {"test_adversarial"})
    missed = total - passed
    rate = (missed / total * 100) if total else 0.0
    return MetricResult(
        name="Hallucination rate",
        value=round(rate, 1),
        threshold=5.0,
        unit="%",
        higher_is_better=False,
        detail=f"{missed}/{total} adversarial inputs not caught",
    )


def _metric_verification_accuracy(tr: dict[str, bool]) -> MetricResult:
    """
    Verification accuracy = fraction of safety + adversarial tests that pass.
    Covers: correct flag injection, no false positives, idempotency, edge cases.
    Target: > 90%
    """
    passed, total = _filter_by_files(tr, {"test_safety", "test_adversarial"})
    rate = (passed / total * 100) if total else 0.0
    return MetricResult(
        name="Verification accuracy",
        value=round(rate, 1),
        threshold=90.0,
        unit="%",
        higher_is_better=True,
        detail=f"{passed}/{total} verification checks correct",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}▶ {title}{RESET}")


def _bar(value: float, threshold: float, higher_is_better: bool, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if higher_is_better:
        fill = min(int(value / 100 * width), width)
        colour = GREEN if value >= threshold else RED
    else:
        # For latency/error rate: full bar = bad, empty = good
        fill = min(int((value / (threshold * 2)) * width), width)
        colour = GREEN if value <= threshold else RED
    bar = "█" * fill + "░" * (width - fill)
    return f"{colour}[{bar}]{RESET}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fortio agent performance target checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip direct latency measurement (faster run)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Exclude LLM-in-the-loop tests (avoids API cost)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full pytest output after the dashboard",
    )
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 74}{RESET}")
    print(f"{BOLD}  Fortio Agent — Performance Target Checker{RESET}")
    print(f"{BOLD}{'═' * 74}{RESET}")

    metrics: list[MetricResult] = []

    # ── Step 1: Run pytest ────────────────────────────────────────────────────
    _section("Running eval suite (may take 30–90 s)...")
    extra: list[str] = []
    if args.no_llm:
        extra += ["--ignore=tests/evals/test_llm_tool_selection.py"]
        print(f"  {YELLOW}ℹ  LLM-in-the-loop tests excluded (--no-llm){RESET}")

    pytest_data = _run_pytest(extra_args=extra)
    colour_p = GREEN if pytest_data["failed"] == 0 else YELLOW
    print(
        f"\n  pytest finished: "
        f"{GREEN}{pytest_data['passed']} passed{RESET}, "
        f"{RED}{pytest_data['failed']} failed{RESET} "
        f"(total: {colour_p}{pytest_data['total']}{RESET})"
    )

    # ── Step 2: Latency ───────────────────────────────────────────────────────
    if args.fast:
        print(f"\n{YELLOW}  ⚡ Skipping latency measurement (--fast){RESET}")
    else:
        _section("Measuring tool latency (3 runs each, reporting median)...")
        try:
            lat_single, lat_multi = _measure_latency()
            metrics.extend([lat_single, lat_multi])
            s_ok = f"{GREEN}OK{RESET}" if lat_single.passed else f"{RED}EXCEEDED{RESET}"
            m_ok = f"{GREEN}OK{RESET}" if lat_multi.passed else f"{RED}EXCEEDED{RESET}"
            print(f"  Single-tool  median: {lat_single.value:.3f}s  [{s_ok}]  (budget 200ms)")
            print(f"  Multi-step   median: {lat_multi.value:.3f}s  [{m_ok}]  (budget 1000ms)")
        except Exception as exc:
            print(f"  {RED}Error measuring latency: {exc}{RESET}")
            print(f"  {YELLOW}Make sure you're running from apps/agent/ with the venv active.{RESET}")
            metrics.extend([
                MetricResult(
                    "End-to-end latency (single-tool)", 999.0, 0.200, "s", False,
                    detail="measurement error — run from apps/agent/ with venv active"
                ),
                MetricResult(
                    "Multi-step latency (3+ tools)", 999.0, 1.000, "s", False,
                    detail="measurement error — run from apps/agent/ with venv active"
                ),
            ])

    # ── Steps 3-6: Derived metrics ────────────────────────────────────────────
    _section("Computing metrics from pytest results...")
    tr = pytest_data["test_results"]

    tool_p, tool_t = _filter_by_files(
        tr, {"test_correctness", "test_tool_execution", "test_multi_step", "test_edge_cases"}
    )
    adv_p, adv_t  = _filter_by_files(tr, {"test_adversarial"})
    ver_p, ver_t  = _filter_by_files(tr, {"test_safety", "test_adversarial"})

    print(f"  Tool-exercising tests : {tool_p}/{tool_t}")
    print(f"  Adversarial tests     : {adv_p}/{adv_t}")
    print(f"  Verification tests    : {ver_p}/{ver_t}")

    metrics.extend([
        _metric_tool_success(tr),
        _metric_eval_pass_rate(pytest_data),
        _metric_hallucination_rate(tr),
        _metric_verification_accuracy(tr),
    ])

    # ── Dashboard ─────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 74}{RESET}")
    print(f"{BOLD}  Performance Target Dashboard{RESET}")
    print(f"{BOLD}{'═' * 74}{RESET}\n")

    passed_count = 0
    for m in metrics:
        print(m.status_line())
        if m.passed:
            passed_count += 1

    total_metrics = len(metrics)
    print(f"\n{BOLD}  {'─' * 70}{RESET}")

    all_pass = passed_count == total_metrics
    if all_pass:
        overall = f"{GREEN}{BOLD}ALL {total_metrics} TARGETS MET ✅{RESET}"
    else:
        failed_count = total_metrics - passed_count
        overall = (
            f"{RED}{BOLD}{failed_count} TARGET(S) NOT MET ❌  "
            f"({passed_count}/{total_metrics} passed){RESET}"
        )
    print(f"\n  Overall: {overall}\n")

    # ── Legend ────────────────────────────────────────────────────────────────
    print(f"{BOLD}{'═' * 74}{RESET}")
    print(f"{CYAN}{BOLD}  Metric definitions:{RESET}")
    print(f"""
  1  Latency (single)      Median of 3 runs of _get_portfolio_summary() with
                           mocked HTTP. Proxy budget ≤200ms → production <5s.

  2  Latency (multi-step)  Median of 3 runs of _scorecard() (3+ API calls).
                           Proxy budget ≤1000ms → production <15s.

  3  Tool success rate     Pass rate of test_correctness + test_tool_execution
                           + test_multi_step + test_edge_cases  (target >95%)

  4  Eval pass rate        Overall pass rate of ALL tests/evals/ files.
                           (target >80%)

  5  Hallucination rate    Fraction of test_adversarial tests that FAIL.
                           FAIL = verification layer missed an attack/hallucination.
                           (target <5%)

  6  Verification accuracy Pass rate of test_safety + test_adversarial.
                           (target >90%)
""")
    print(f"  {YELLOW}Tip: use --no-llm to skip LLM API calls (free, fast).{RESET}")
    print(f"  {YELLOW}Tip: use --verbose to print the full pytest log.{RESET}")
    print(f"  {YELLOW}Tip: run LLM evals separately →  pytest tests/evals/test_llm_tool_selection.py -v{RESET}")
    print(f"\n{BOLD}{'═' * 74}{RESET}\n")

    if args.verbose:
        print(f"\n{BOLD}── Raw pytest output ──────────────────────────────────────────────────{RESET}")
        print(pytest_data["stdout"])

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
