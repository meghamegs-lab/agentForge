# Redis-backed exact-match query cache for identical user queries.
"""
Semantic query cache for Fortio.

Caches the full ChatResponse payload (answer, confidence, flags, tool_calls,
turn_number, context_entities) in Redis, keyed by a SHA-256 hash of the
normalised (user_id, query) pair.

Only single-turn (fresh-session) queries are cached — multi-turn queries
depend on conversation history and MUST NOT be served from cache.

Cache misses and Redis errors are both handled silently; the agent falls
through to a fresh LLM call in either case.

Configuration
-------------
    SEMANTIC_CACHE_ENABLED=true          (default)
    SEMANTIC_CACHE_TTL_SECONDS=300       (5 minutes, default)
    REDIS_URL=redis://localhost:6379/1   (default — DB 1 is Fortio's logical DB)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from agent.config import settings

log = structlog.get_logger()

# Module-level Redis client — initialised lazily on first use.
_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    """Return (or create) the shared async Redis client."""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _make_key(user_id: str, query: str) -> str:
    """
    Build a stable, user-scoped cache key from a normalised query string.

    Normalisation: lowercase + collapse consecutive whitespace.
    The key is a short SHA-256 digest to avoid Redis key-length limits and
    to prevent query text from leaking into monitoring dashboards.
    """
    normalised = " ".join(query.lower().split())  # lowercase + collapse whitespace
    raw = f"fortio:v1:{user_id}:{normalised}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"fortio:qcache:{digest}"


async def get_cached_response(user_id: str, query: str) -> dict[str, Any] | None:
    """
    Return a cached response dict for this (user_id, query) pair, or None.

    Returns None (cache miss) when:
    - Caching is disabled via settings.semantic_cache_enabled
    - The key does not exist in Redis
    - Redis is unavailable (exception swallowed, never raises)
    """
    if not settings.semantic_cache_enabled:
        return None
    try:
        key = _make_key(user_id, query)
        raw = await _get_redis().get(key)
        if raw:
            log.info("query_cache_hit", user_id=user_id, preview=query[:60])
            return json.loads(raw)
    except Exception as exc:
        log.warning("query_cache_get_error", error=str(exc))
    return None


async def set_cached_response(
    user_id: str,
    query: str,
    response: dict[str, Any],
) -> None:
    """
    Store a serialisable response dict in Redis for TTL seconds.

    Silently no-ops if caching is disabled or Redis is unavailable.
    Non-serialisable values are coerced to strings via json default=str.
    """
    if not settings.semantic_cache_enabled:
        return
    try:
        key = _make_key(user_id, query)
        payload = json.dumps(response, default=str)
        await _get_redis().setex(key, settings.semantic_cache_ttl_seconds, payload)
        log.info(
            "query_cache_set",
            user_id=user_id,
            preview=query[:60],
            ttl=settings.semantic_cache_ttl_seconds,
        )
    except Exception as exc:
        log.warning("query_cache_set_error", error=str(exc))
