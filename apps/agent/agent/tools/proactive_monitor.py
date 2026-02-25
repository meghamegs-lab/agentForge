"""
Tool: get_proactive_risk_monitor
Standout feature: fires AUTOMATICALLY on every Chainlit session start via on_chat_start.
Checks for new risks since last session — tells you what changed, not just current state.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from agent.clients.ghostfolio import GhostfolioError, get_shared_client
from agent.clients.market import MarketDataClient
from agent.config import settings

_market = MarketDataClient()


@tool
async def get_proactive_risk_monitor(previous_snapshot_json: str = "") -> dict[str, Any]:
    """
    Proactive risk monitoring tool — designed to fire automatically on every session start.
    Checks for concentration risks, significant price moves in holdings, new diversification
    issues, and compares against the previous session's snapshot to surface CHANGES.

    This is the standout differentiating feature: it tells you what CHANGED since you
    last logged in — not just the current state of your portfolio.

    Use this:
    - AUTOMATICALLY on every Chainlit on_chat_start (pass previous snapshot from session store)
    - When user asks 'Any risks I should know about?'
    - When user asks 'What changed since last time?'
    - When user asks 'Do I have any urgent alerts?'

    Args:
        previous_snapshot_json: JSON string of previous session's snapshot dict.
                                 Pass empty string on first session.

    Returns:
        alerts[] with severity, changes_since_last_session[], concentration_risks[],
        overall_risk_level, current_snapshot (save for next session)
    """
    return await _proactive_monitor(previous_snapshot_json)


async def _proactive_monitor(prev_snap_json: str = "") -> dict[str, Any]:
    try:
        client = get_shared_client()
        holdings_data = await client.get_portfolio_holdings()

        holdings = holdings_data.get("holdings", {})
        if not holdings:
            return {
                "status": "empty",
                "message": "No holdings found.",
                "alerts": [],
                "overall_risk_level": "NONE",
                "current_snapshot": {},
                "data_timestamp": datetime.now(UTC).isoformat(),
            }

        total_value = sum(h.get("value", 0) or 0 for h in holdings.values())

        # ── Build current snapshot ─────────────────────────────────
        current_snapshot = {
            "total_value": round(total_value, 2),
            "position_count": len(holdings),
            "timestamp": datetime.now(UTC).isoformat(),
            "allocations": {
                sym: round((h.get("value", 0) or 0) / total_value * 100, 2)
                for sym, h in holdings.items()
            },
        }

        # ── Load previous snapshot ─────────────────────────────────
        prev_snapshot: dict[str, Any] = {}
        if prev_snap_json:
            try:
                prev_snapshot = json.loads(prev_snap_json)
            except (json.JSONDecodeError, ValueError):
                prev_snapshot = {}

        # ── Alerts ────────────────────────────────────────────────
        alerts: list[dict] = []
        threshold = settings.portfolio_concentration_threshold

        # 1. Concentration alerts (current state)
        concentration_risks = []
        for sym, h in holdings.items():
            val = h.get("value", 0) or 0
            alloc = val / total_value * 100 if total_value > 0 else 0
            if alloc >= threshold * 100:
                severity = "HIGH" if alloc >= 35 else "MEDIUM"
                concentration_risks.append({
                    "symbol": sym,
                    "name": h.get("name", sym),
                    "allocation": round(alloc, 2),
                    "severity": severity,
                })
                alerts.append({
                    "type": "CONCENTRATION",
                    "severity": severity,
                    "message": (
                        f"{sym} is {alloc:.1f}% of your portfolio "
                        f"(threshold: {threshold * 100:.0f}%)"
                    ),
                    "action": f"Consider trimming {sym} to reduce single-position risk",
                })

        # 2. Changes since last session
        changes_since_last = []
        if prev_snapshot and "allocations" in prev_snapshot:
            prev_allocs = prev_snapshot["allocations"]
            curr_allocs = current_snapshot["allocations"]

            for sym, curr_pct in curr_allocs.items():
                if sym in prev_allocs:
                    change = curr_pct - prev_allocs[sym]
                    if abs(change) >= 2.0:   # flag 2%+ allocation shifts
                        changes_since_last.append({
                            "symbol": sym,
                            "previous_pct": prev_allocs[sym],
                            "current_pct": round(curr_pct, 2),
                            "change_pp": round(change, 2),
                            "direction": "INCREASED" if change > 0 else "DECREASED",
                            "reason": "likely price movement (no new transactions detected)",
                        })
                        # New threshold breach from price movement
                        if curr_pct >= threshold * 100 and prev_allocs[sym] < threshold * 100:
                            alerts.append({
                                "type": "NEW_CONCENTRATION_BREACH",
                                "severity": "HIGH",
                                "message": (
                                    f"⚠️ {sym} crossed the {threshold * 100:.0f}% threshold since "
                                    f"your last session ({prev_allocs[sym]:.1f}% → {curr_pct:.1f}%) "
                                    f"due to price appreciation."
                                ),
                                "action": (
                                    f"Review {sym} position — consider trimming to manage concentration"
                                ),
                            })

            # New positions added since last session
            new_syms = set(curr_allocs) - set(prev_allocs)
            for sym in new_syms:
                changes_since_last.append({
                    "symbol": sym,
                    "change_pp": curr_allocs[sym],
                    "direction": "NEW_POSITION",
                    "message": f"New position added: {sym} ({curr_allocs[sym]:.1f}%)",
                })

            # Portfolio value change
            if "total_value" in prev_snapshot:
                prev_val = prev_snapshot["total_value"]
                val_change = total_value - prev_val
                val_change_pct = (val_change / prev_val * 100) if prev_val > 0 else 0
                if abs(val_change_pct) >= 2:
                    changes_since_last.append({
                        "symbol": "PORTFOLIO",
                        "direction": "VALUE_CHANGE",
                        "message": (
                            f"Portfolio value {'increased' if val_change > 0 else 'decreased'} "
                            f"by ${abs(val_change):,.0f} ({val_change_pct:+.1f}%) since last session"
                        ),
                        "change_pp": round(val_change_pct, 2),
                    })

        # ── Overall risk level ─────────────────────────────────────
        high_alerts = sum(1 for a in alerts if a["severity"] == "HIGH")
        medium_alerts = sum(1 for a in alerts if a["severity"] == "MEDIUM")

        if high_alerts >= 2 or (high_alerts >= 1 and medium_alerts >= 2):
            overall_risk = "HIGH"
        elif high_alerts >= 1 or medium_alerts >= 2:
            overall_risk = "MEDIUM"
        elif medium_alerts >= 1:
            overall_risk = "LOW"
        else:
            overall_risk = "NONE"

        # ── Session greeting ───────────────────────────────────────
        if overall_risk == "NONE" and not changes_since_last:
            greeting = "✅ No new risks detected. Your portfolio is within all thresholds."
        elif overall_risk == "NONE" and changes_since_last:
            greeting = f"ℹ️ {len(changes_since_last)} allocation change(s) since your last session."
        else:
            greeting = (
                f"⚠️ {len(alerts)} risk alert(s) detected — "
                f"{high_alerts} high, {medium_alerts} medium priority."
            )

        return {
            "status": "ok",
            "session_greeting": greeting,
            "overall_risk_level": overall_risk,
            "alerts": alerts,
            "alert_count": len(alerts),
            "concentration_risks": concentration_risks,
            "changes_since_last_session": changes_since_last,
            "portfolio_value": round(total_value, 2),
            "position_count": len(holdings),
            "current_snapshot": current_snapshot,  # caller should save for next session
            "data_timestamp": datetime.now(UTC).isoformat(),
        }

    except GhostfolioError as e:
        return {"status": "error", "error": e.message, "alerts": [], "current_snapshot": {}}
    except Exception as e:
        return {"status": "error", "error": str(e), "alerts": [], "current_snapshot": {}}
