# evalsNew — Ghostfolio Agent Evaluation Framework v2.2
#
# Comprehensive evaluation suite covering:
#   Correctness · Tool Selection · Tool Execution · Safety
#   Consistency · Edge Cases · Latency · Multi-Step Reasoning · Adversarial
#
# This is the single consolidated eval directory (eval/ has been merged here).
#
# Total test cases: 74
#   - 7  Correctness (C01–C07)
#   - 6  Tool Selection (TS01–TS06)
#   - 7  Tool Execution (TE01–TE07)
#   - 7  Safety (S01–S07)
#   - 4  Consistency (CON01–CON04)
#   - 16 Edge Cases (EC01–EC16)
#        EC01–EC08: empty, unknown ticker, ambiguous, future date, duplicates,
#                   delisted, single-holding, fractional shares
#        EC09–EC16: dict-format holdings, missing fields, unicode, large
#                   portfolio, invalid period, case-insensitive filter,
#                   whitespace symbol, comma-separated symbols
#   - 3  Latency (L01–L03)
#   - 12 Multi-Step Reasoning (MS01–MS12)
#   - 12 Adversarial (ADV01–ADV12)
#
# LangSmith evals:  python apps/agent/tests/evalsNew/ls_evals.py
#
# Run all:          pytest apps/agent/tests/evalsNew/ -v
# Run by category:  pytest apps/agent/tests/evalsNew/test_correctness_v2.py -v
#                   pytest apps/agent/tests/evalsNew/test_edge_cases_v2.py -v
#                   pytest apps/agent/tests/evalsNew/test_multi_step_v2.py -v
#                   pytest apps/agent/tests/evalsNew/test_adversarial_v2.py -v
