# Verification Pipeline — Flow & Flag Reference

## Pipeline Overview

The verification pipeline runs **5 stages in order** after every LLM response.
Each stage can add flags and/or modify the response text.

```
LLM Response + Tool Results
          │
          ▼
╔═══════════════════════════════════╗
║   STAGE 1 — DISCLAIMER CHECK      ║
╚═══════════════════════════════════╝
Does response contain any of:
buy, sell, invest, rebalance, recommend,
"should i", allocate, diversify, move, shift,
rotate, switch, "portfolio change",
"what to do", advice
          │
    ┌─────┴─────┐
   YES          NO
    │            │
    ▼            │
⚑ DISCLAIMER_ADDED          │
  severity: INFO             │
  → Appends disclaimer text  │
  → Sets disclaimer_added=True
    │            │
    └─────┬──────┘
          │
          ▼
╔═══════════════════════════════════╗
║   STAGE 2 — HALLUCINATION GUARD   ║
╚═══════════════════════════════════╝
          │
    ┌─────┴──────────────────────┐
No tool        Some tools        All tools
called         succeeded         failed
    │               │                │
    ▼               ▼                ▼
Does response   Do financial     Does response
have financial  numbers in       have financial
numbers?        response match   numbers?
    │           tool results?        │
  YES │ NO        │                YES │ NO
    │   │   ┌────┴────┐              │   │
    ▼   │  YES        NO             ▼   │
⚑ POTENTIAL│           │        ⚑ POTENTIAL│
  HALLUCINA-│       ⚑ UNSUPPORTED   HALLUCINA-│
  TION      │         CLAIM         TION      │
  severity: │         severity:     severity: │
  MEDIUM    │         MEDIUM        HIGH      │
  (no tools │         (numbers in   (guessed  │
  = opinion)│         response not  from      │
    │   │   │         in any tool   training  │
    │   │   │         result)       data)     │
    └───┴───┴──────────────────────────────┘
          │
          ▼
╔═══════════════════════════════════╗
║   STAGE 3 — DATA FRESHNESS        ║
╚═══════════════════════════════════╝
For each tool result:
          │
    ┌─────┴──────────────────────────┐
No timestamp    Valid timestamp      Unparseable
present         present              timestamp
    │               │                    │
    ▼               ▼                    ▼
⚑ MISSING_    Is it market data   ⚑ INVALID_
  TIMESTAMP   (has current_price       TIMESTAMP
  severity:   or quotes field)?        severity:
  LOW             │                    LOW
           ┌──────┴──────┐
        YES (market)   NO (portfolio)
           │               │
     threshold:       threshold:
      15 min            60 min
           │               │
     Is age > threshold?
           │
      YES  │  NO
       │   │   │
       ▼   │   │
⚑ STALE_DATA   │
  severity:     │
  MEDIUM        │
  (shows age    │
  in minutes)   │
       │        │
       └────────┘
          │
          ▼
╔═══════════════════════════════════╗
║   STAGE 4 — CONCENTRATION RISK    ║
╚═══════════════════════════════════╝
For each tool result:
          │
    ┌─────┴──────────────────────────┐
Has             Has "holdings"       Neither
"concentration_ field (portfolio     → skip
flags" field    summary)?
(diversification│
tool output)?   │
    │           ▼
    │    Any holding >= 20%
    │    allocation_percent?
    │           │
    │      YES  │  NO
    │       │   │   │
    ▼       ▼   │   │
⚑ CONCENTRATION_RISK    │
  severity: MEDIUM  (from│
  holding direct check)  │
  — also appends warning │
    text to response     │
                         │
⚑ CONCENTRATION_RISK     │
  severity: (inherited   │
  from diversification   │
  tool's own severity)   │
          │
          ▼
╔═══════════════════════════════════╗
║   STAGE 5 — CONFIDENCE SCORING    ║
╚═══════════════════════════════════╝

          Conditions evaluated:
          ┌─────────────────────────────────────┐
          │ has_tool_data  = tool_results > 0   │
          │ has_predictions = "will","forecast", │
          │   "predict","expect","future",       │
          │   "next year","going to","likely to",│
          │   "probably","might","could reach"   │
          │ has_speculative = "bitcoin","crypto", │
          │   "ethereum","solana","dogecoin",    │
          │   "nft","meme stock","all in",       │
          │   "double down","yolo","gamble",     │
          │   "put it all","everything into"     │
          │ has_hedging = "uncertain","unclear", │
          │   "approximately","roughly","around",│
          │   "about","estimate","may","might",  │
          │   "could"                            │
          │ is_multi_step = reasoning_steps > 1  │
          │   OR len(tool_results) > 1           │
          └─────────────────────────────────────┘

    ┌──────────────────────────────────┐
    │                                  │
    ▼                                  │
NOT has_tool_data                      │
OR has_predictions                     │
OR has_speculative?                    │
    │                                  │
  YES │  NO                            │
    │   │                              │
    ▼   ▼                              │
 LOW  is_multi_step                    │
      OR has_hedging?                  │
          │                            │
       YES │ NO                        │
        │   │                          │
        ▼   ▼                          │
      MEDIUM HIGH                      │
        │   │                          │
        └───┘                          │
          │                            │
⚑ LOW_CONFIDENCE (only if LOW)         │
  severity: INFO                       │
  "predictions/speculative/no data"    │
          │                            │
          ▼                            │
╔════════════════════════════╗         │
║  PIPELINE POST-PROCESSING  ║◄────────┘
╚════════════════════════════╝
Was disclaimer_added = True
AND confidence != LOW?
          │
       YES│ NO
          │   │
          ▼   │
  Override → LOW
⚑ LOW_CONFIDENCE
  severity: INFO
  "disclaimer triggered —
   confidence downgraded"
          │
          ▼
╔═════════════════════════════════════════╗
║  FINAL OUTPUT                           ║
║  {                                      ║
║    response:  (possibly modified text)  ║
║    confidence: HIGH | MEDIUM | LOW      ║
║    verification_flags: [...]            ║
║    flag_count: N                        ║
║    has_high_severity: true | false      ║
║  }                                      ║
╚═════════════════════════════════════════╝
```

---

## All Flag Types — Quick Reference

| Flag                      | Stage | Severity   | Trigger                                                          |
| ------------------------- | ----- | ---------- | ---------------------------------------------------------------- |
| `DISCLAIMER_ADDED`        | 1     | INFO       | Response contains investment advice keywords                     |
| `POTENTIAL_HALLUCINATION` | 2     | **MEDIUM** | No tools called, but response has financial numbers              |
| `POTENTIAL_HALLUCINATION` | 2     | **HIGH**   | All tools failed, but response still cites financial numbers     |
| `UNSUPPORTED_CLAIM`       | 2     | **MEDIUM** | Financial numbers in response can't be traced to any tool result |
| `MISSING_TIMESTAMP`       | 3     | LOW        | A tool result has no `data_timestamp` field                      |
| `STALE_DATA`              | 3     | **MEDIUM** | Market data > 15 min old, or portfolio data > 60 min old         |
| `INVALID_TIMESTAMP`       | 3     | LOW        | `data_timestamp` exists but can't be parsed                      |
| `CONCENTRATION_RISK`      | 4     | **MEDIUM** | Any holding ≥ 20% of portfolio                                   |
| `LOW_CONFIDENCE`          | 5     | INFO       | No tool data, predictions, or speculative language detected      |
| `LOW_CONFIDENCE`          | Post  | INFO       | Disclaimer was added but confidence wasn't already LOW           |

---

## Confidence Decision Table

| has_tool_data | has_predictions | has_speculative | is_multi_step | has_hedging | disclaimer_added | → Confidence         |
| ------------- | --------------- | --------------- | ------------- | ----------- | ---------------- | -------------------- |
| ❌            | any             | any             | any           | any         | any              | **LOW**              |
| ✅            | ✅              | any             | any           | any         | any              | **LOW**              |
| ✅            | ❌              | ✅              | any           | any         | any              | **LOW**              |
| ✅            | ❌              | ❌              | any           | any         | ✅               | **LOW** _(override)_ |
| ✅            | ❌              | ❌              | ✅            | any         | ❌               | **MEDIUM**           |
| ✅            | ❌              | ❌              | ❌            | ✅          | ❌               | **MEDIUM**           |
| ✅            | ❌              | ❌              | ❌            | ❌          | ❌               | **HIGH**             |

---

## Severity Escalation Summary

```
INFO    → Informational only; no action required
LOW     → Minor data quality issue; note to user
MEDIUM  → Data could be unreliable; flag visibly
HIGH    → Potential hallucination; triggers escalation path in graph
```

> **Note:** `has_high_severity: true` in the pipeline output means at least one
> `HIGH` flag was raised (currently only `POTENTIAL_HALLUCINATION` when all tools
> failed). The graph's `should_escalate()` router uses this combined with
> `should_escalate` state field to decide whether to route to the escalation node.
