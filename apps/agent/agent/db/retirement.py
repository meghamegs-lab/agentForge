# Async Postgres CRUD for the retirement_goals table (FIRE Goal Tracker).
"""
Retirement Goals — Database Layer
==================================
Stores one retirement goal per user_id in the `retirement_goals` table.

Schema
------
    id                      UUID PRIMARY KEY
    user_id                 TEXT UNIQUE NOT NULL   (matches user_id in AgentState)
    current_age             INTEGER NOT NULL
    target_retirement_age   INTEGER NOT NULL
    target_annual_spending  NUMERIC(15,2) NOT NULL  ($ per year in retirement)
    safe_withdrawal_rate    NUMERIC(5,4) DEFAULT 0.04
    monthly_contribution    NUMERIC(15,2) DEFAULT 0
    expected_annual_return  NUMERIC(5,4) DEFAULT 0.07
    social_security_estimate NUMERIC(15,2) DEFAULT 0
    created_at              TIMESTAMPTZ DEFAULT now()
    updated_at              TIMESTAMPTZ DEFAULT now()

Why psycopg2 + asyncio.to_thread?
----------------------------------
psycopg2 is already a project dependency (psycopg2-binary).  Running
sync DB calls in asyncio.to_thread() keeps the event loop non-blocking,
matching the yfinance pattern already used in agent/clients/market.py.
No new dependencies are needed.

All public functions:
    create_table_if_not_exists(database_url) — idempotent startup step
    get_goal(database_url, user_id)          — SELECT one row or None
    upsert_goal(database_url, user_id, **fields) — INSERT ... ON CONFLICT DO UPDATE
    delete_goal(database_url, user_id)       — DELETE, returns True if row existed

Error handling:
    All sync helpers raise on error.
    All async wrappers catch and return {"status": "error", "error": "..."}.
"""

from __future__ import annotations

import asyncio
from typing import Any

import psycopg2
import psycopg2.extras
import structlog

log = structlog.get_logger()

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS retirement_goals (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                  TEXT NOT NULL UNIQUE,
    current_age              INTEGER NOT NULL,
    target_retirement_age    INTEGER NOT NULL,
    target_annual_spending   NUMERIC(15,2) NOT NULL,
    safe_withdrawal_rate     NUMERIC(5,4)  NOT NULL DEFAULT 0.04,
    monthly_contribution     NUMERIC(15,2) NOT NULL DEFAULT 0,
    expected_annual_return   NUMERIC(5,4)  NOT NULL DEFAULT 0.07,
    social_security_estimate NUMERIC(15,2) NOT NULL DEFAULT 0,
    created_at               TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ   NOT NULL DEFAULT now()
);
"""

_UPSERT_SQL = """
INSERT INTO retirement_goals (
    user_id, current_age, target_retirement_age, target_annual_spending,
    safe_withdrawal_rate, monthly_contribution, expected_annual_return,
    social_security_estimate, updated_at
)
VALUES (%(user_id)s, %(current_age)s, %(target_retirement_age)s, %(target_annual_spending)s,
        %(safe_withdrawal_rate)s, %(monthly_contribution)s, %(expected_annual_return)s,
        %(social_security_estimate)s, now())
ON CONFLICT (user_id) DO UPDATE SET
    current_age              = EXCLUDED.current_age,
    target_retirement_age    = EXCLUDED.target_retirement_age,
    target_annual_spending   = EXCLUDED.target_annual_spending,
    safe_withdrawal_rate     = EXCLUDED.safe_withdrawal_rate,
    monthly_contribution     = EXCLUDED.monthly_contribution,
    expected_annual_return   = EXCLUDED.expected_annual_return,
    social_security_estimate = EXCLUDED.social_security_estimate,
    updated_at               = now()
RETURNING *;
"""

_SELECT_SQL = "SELECT * FROM retirement_goals WHERE user_id = %s;"
_DELETE_SQL = "DELETE FROM retirement_goals WHERE user_id = %s RETURNING user_id;"


# ── Sync helpers (run inside asyncio.to_thread) ────────────────────────────────


def _create_table_sync(database_url: str) -> None:
    """Create the retirement_goals table if it doesn't exist. Idempotent."""
    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
    finally:
        conn.close()


def _get_goal_sync(database_url: str, user_id: str) -> dict[str, Any] | None:
    """SELECT one retirement goal by user_id. Returns None if not found."""
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SELECT_SQL, (user_id,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _upsert_goal_sync(database_url: str, fields: dict[str, Any]) -> dict[str, Any]:
    """INSERT or UPDATE a retirement goal. Returns the saved row."""
    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_UPSERT_SQL, fields)
            row = cur.fetchone()
        return dict(row) if row else fields
    finally:
        conn.close()


def _delete_goal_sync(database_url: str, user_id: str) -> bool:
    """DELETE a retirement goal. Returns True if a row was deleted."""
    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(_DELETE_SQL, (user_id,))
            deleted = cur.fetchone()
        return deleted is not None
    finally:
        conn.close()


# ── Async public API ──────────────────────────────────────────────────────────


def _row_to_goal_dict(row: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise a Postgres row dict for use in tool responses.
    Converts Decimal → float, datetime → ISO string, UUID → string.
    """
    return {
        "user_id": str(row.get("user_id", "")),
        "current_age": int(row.get("current_age", 0)),
        "target_retirement_age": int(row.get("target_retirement_age", 0)),
        "target_annual_spending": float(row.get("target_annual_spending", 0)),
        "safe_withdrawal_rate": float(row.get("safe_withdrawal_rate", 0.04)),
        "monthly_contribution": float(row.get("monthly_contribution", 0)),
        "expected_annual_return": float(row.get("expected_annual_return", 0.07)),
        "social_security_estimate": float(row.get("social_security_estimate", 0)),
        # Derived field — FIRE number = annual spending ÷ SWR (e.g. $80K ÷ 0.04 = $2M)
        "fire_number": round(
            float(row.get("target_annual_spending", 0))
            / max(float(row.get("safe_withdrawal_rate", 0.04)), 0.001),
            2,
        ),
        "years_to_target": int(row.get("target_retirement_age", 0))
        - int(row.get("current_age", 0)),
        "created_at": (
            row["created_at"].isoformat()
            if hasattr(row.get("created_at"), "isoformat")
            else str(row.get("created_at", ""))
        ),
        "updated_at": (
            row["updated_at"].isoformat()
            if hasattr(row.get("updated_at"), "isoformat")
            else str(row.get("updated_at", ""))
        ),
    }


async def create_table_if_not_exists(database_url: str) -> None:
    """
    Create the retirement_goals table if it doesn't exist.
    Called from FastAPI lifespan — safe to call on every startup (idempotent).
    Logs a warning if database_url is empty (feature flag off or misconfigured).
    """
    if not database_url:
        log.warning("retirement_db_skip", reason="DATABASE_URL is empty — skipping table creation")
        return
    try:
        await asyncio.to_thread(_create_table_sync, database_url)
        log.info("retirement_table_ready")
    except Exception as exc:
        log.error("retirement_table_error", error=str(exc))


async def get_goal(database_url: str, user_id: str) -> dict[str, Any] | None:
    """
    Fetch a user's retirement goal from Postgres.

    Returns:
        Normalised goal dict if found, None if the user has no saved goal.
        On DB error: raises (callers should wrap in try/except).
    """
    row = await asyncio.to_thread(_get_goal_sync, database_url, user_id)
    return _row_to_goal_dict(row) if row else None


async def upsert_goal(
    database_url: str,
    user_id: str,
    current_age: int,
    target_retirement_age: int,
    target_annual_spending: float,
    safe_withdrawal_rate: float = 0.04,
    monthly_contribution: float = 0.0,
    expected_annual_return: float = 0.07,
    social_security_estimate: float = 0.0,
) -> dict[str, Any]:
    """
    Insert or update a retirement goal (upsert on user_id).

    Returns the saved goal dict (normalised), including the computed fire_number.
    Raises on DB error.
    """
    fields = {
        "user_id": user_id,
        "current_age": current_age,
        "target_retirement_age": target_retirement_age,
        "target_annual_spending": target_annual_spending,
        "safe_withdrawal_rate": safe_withdrawal_rate,
        "monthly_contribution": monthly_contribution,
        "expected_annual_return": expected_annual_return,
        "social_security_estimate": social_security_estimate,
    }
    row = await asyncio.to_thread(_upsert_goal_sync, database_url, fields)
    return _row_to_goal_dict(row)


async def delete_goal(database_url: str, user_id: str) -> bool:
    """
    Delete a user's retirement goal.

    Returns True if a row was deleted, False if the user had no goal.
    Raises on DB error.
    """
    return await asyncio.to_thread(_delete_goal_sync, database_url, user_id)


# ── Validation helpers (used by tools and API routes) ─────────────────────────


def validate_goal_fields(
    current_age: int,
    target_retirement_age: int,
    target_annual_spending: float,
    safe_withdrawal_rate: float,
) -> list[str]:
    """
    Validate retirement goal field values. Returns a list of error messages.
    Empty list means all fields are valid.

    Rules:
      - current_age: 18–100
      - target_retirement_age: > current_age, ≤ 100
      - target_annual_spending: > 0
      - safe_withdrawal_rate: 1%–10% (0.01–0.10)
    """
    errors: list[str] = []

    if not (18 <= current_age <= 100):
        errors.append(f"current_age must be between 18 and 100 (got {current_age})")

    if target_retirement_age <= current_age:
        errors.append(
            f"target_retirement_age ({target_retirement_age}) must be greater than "
            f"current_age ({current_age})"
        )

    if target_retirement_age > 100:
        errors.append(f"target_retirement_age must be ≤ 100 (got {target_retirement_age})")

    if target_annual_spending <= 0:
        errors.append(f"target_annual_spending must be > 0 (got {target_annual_spending})")

    if not (0.01 <= safe_withdrawal_rate <= 0.10):
        errors.append(
            f"safe_withdrawal_rate must be between 0.01 (1%) and 0.10 (10%) "
            f"(got {safe_withdrawal_rate})"
        )

    return errors
