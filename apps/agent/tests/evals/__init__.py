# evals — Ghostfolio Agent Evaluation Framework v2.2
#
# Comprehensive evaluation suite covering:
#   Correctness · Tool Selection · Tool Execution · Safety
#   Consistency · Edge Cases · Latency · Multi-Step Reasoning · Adversarial
#
# Total test cases: 74+
#   - 7  Correctness   (C01–C07)    test_correctness.py
#   - 6  Tool Selection (TS01–TS06) test_tool_selection.py
#   - 7  Tool Execution (TE01–TE07) test_tool_execution.py
#   - 7  Safety        (S01–S07)    test_safety.py
#   - 4  Consistency   (CON01–CON04) test_consistency.py
#   - 16 Edge Cases    (EC01–EC16)  test_edge_cases.py
#        EC01–EC08: empty, unknown ticker, ambiguous, future date, duplicates,
#                   delisted, single-holding, fractional shares
#        EC09–EC16: dict-format holdings, missing fields, unicode, large
#                   portfolio, invalid period, case-insensitive filter,
#                   whitespace symbol, comma-separated symbols
#   - 3  Latency       (L01–L03)    test_latency.py
#   - 12 Multi-Step Reasoning (MS01–MS12) test_multi_step.py
#   - 12+ Adversarial  (ADV01–ADV21) test_adversarial.py
#
# LangSmith evals:  python apps/agent/tests/evals/ls_evals.py
#
# Run all:          pytest apps/agent/tests/evals/ -v
# Run by category:  pytest apps/agent/tests/evals/test_correctness.py -v
#                   pytest apps/agent/tests/evals/test_edge_cases.py -v
#                   pytest apps/agent/tests/evals/test_multi_step.py -v
#                   pytest apps/agent/tests/evals/test_adversarial.py -v
#
# Run unit + evals: pytest apps/agent/tests/ -v --ignore=apps/agent/tests/evals/test_llm_tool_selection.py
# (test_llm_tool_selection.py requires a live LLM — run on-demand only)
