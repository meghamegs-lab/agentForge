# Async market-data client wrapping yfinance; all blocking calls are dispatched to a thread pool.
"""
Market data client using yfinance.
Returns structured dicts with timestamps for freshness checking.

Implementation note
-------------------
yfinance is a synchronous library.  To avoid blocking the asyncio event loop,
all network calls are dispatched to a thread-pool executor via asyncio.to_thread().

Reliability note
----------------
yfinance is a scraper, not an official API — Yahoo Finance rate-limits it and
occasionally returns empty/error responses on transient network hiccups.

Three strategies are layered to handle this:
  1. Browser-like session: a shared requests.Session with realistic headers is
     passed to every yf.Ticker() call so Yahoo Finance does not block cloud IPs
     (Railway, AWS, GCP, etc.) that it recognises as bots.
  2. Period fallback: if period="1d" returns empty (market closed / weekend),
     we retry the same request with period="5d" to get the most recent close.
  3. Tenacity retry: up to 3 attempts with exponential back-off (1 s → 2 s → 4 s)
     for genuine network-level exceptions (ConnectionError, Timeout, HTTP 429, etc.).
     Empty-data results are NOT retried — the period fallback already handles those.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

# ── yfinance 0.2.38+ session note ─────────────────────────────────────────────
# Newer yfinance uses curl_cffi internally for its HTTP session.
# Do NOT pass a requests.Session — yfinance will raise:
#   "Yahoo API requires curl_cffi session not requests.sessions.Session"
# Let yfinance manage its own session by omitting the session= argument.


class MarketDataClient:
    """Async-safe market data client backed by yfinance (run in thread pool)."""

    # ── internal sync helpers (run in thread pool) ────────────────────────────

    # Synchronous yfinance fetch for one symbol (price, 52w range, market cap) — runs in thread pool.
    # Retries up to 3× on network exceptions; falls back to period="5d" for market-closed days.
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _fetch_quote_sync(self, symbol: str) -> dict[str, Any]:
        """Synchronous yfinance fetch — called from asyncio.to_thread().

        Network exceptions (ConnectionError, Timeout, HTTP 429, etc.) propagate
        so tenacity can retry them.  Empty-data results (market closed / invalid
        symbol) return a price_unavailable dict — no exception, no retry needed
        because the period fallback inside already tried '5d'.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        # Fix 2: Try "1d" first; fall back to "5d" for market-closed / weekend days
        # where the current-day bar is not yet published by Yahoo Finance.
        hist = ticker.history(period="1d")
        if hist.empty:
            hist = ticker.history(period="5d")

        if hist.empty:
            # Genuinely no data (invalid symbol or exchange offline) — don't raise,
            # so tenacity does NOT retry this case (retrying won't help).
            return {
                "status": "price_unavailable",
                "symbol": symbol,
                "error": "No market data returned — market may be closed or symbol invalid",
                "instruction": (
                    "Tell the user the price is currently unavailable. "
                    "Do NOT guess or use training data."
                ),
            }

        current_price = float(hist["Close"].iloc[-1])

        # Anchor data_timestamp to market close (4 PM ET) on the bar's date so the
        # freshness checker reports an accurate age. yfinance daily bars are indexed
        # at midnight ET (05:00 UTC), not at the 4 PM ET close, which would inflate
        # the reported age by ~16 hours on weekends.
        # On weekends, hist.index[-1] is Friday's bar → we set the timestamp to
        # Friday 4 PM ET so the age reflects time since the actual last close.
        try:
            bar_date = hist.index[-1].date()
            ET = ZoneInfo("America/New_York")
            market_close = datetime(
                bar_date.year, bar_date.month, bar_date.day, 16, 0, 0, tzinfo=ET
            )
            data_timestamp = market_close.astimezone(UTC).isoformat()
        except Exception:
            # Fallback: if date extraction or timezone conversion fails, use now()
            data_timestamp = datetime.now(UTC).isoformat()

        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "current_price": current_price,
            "currency": getattr(info, "currency", "USD"),
            "fifty_two_week_high": getattr(info, "year_high", None),
            "fifty_two_week_low": getattr(info, "year_low", None),
            "market_cap": getattr(info, "market_cap", None),
            "volume": getattr(info, "three_month_average_volume", None),
            "data_timestamp": data_timestamp,
            "source": "Yahoo Finance (via yfinance)",
        }

    # ── public async API ──────────────────────────────────────────────────────

    # Async wrapper that offloads _fetch_quote_sync to a thread pool via asyncio.to_thread().
    # The outer try/except catches any exception that survives all tenacity retry attempts.
    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Async get_quote — runs yfinance in thread pool with tenacity retry."""
        try:
            return await asyncio.to_thread(self._fetch_quote_sync, symbol)
        except RetryError as e:
            # All 3 retry attempts exhausted — unwrap to surface the original error message.
            cause = e.last_attempt.exception()
            error_msg = str(cause) if cause else str(e)
            return {
                "status": "price_unavailable",
                "symbol": symbol,
                "error": error_msg,
                "instruction": (
                    "Tell the user the price is currently unavailable due to a data "
                    "provider error. Do NOT guess or use training data."
                ),
            }
        except Exception as e:
            # Unexpected error outside of tenacity (e.g. asyncio.to_thread failure).
            return {
                "status": "price_unavailable",
                "symbol": symbol,
                "error": str(e),
                "instruction": (
                    "Tell the user the price is currently unavailable due to a data "
                    "provider error. Do NOT guess or use training data."
                ),
            }

    # Concurrently fetches quotes for multiple symbols and combines the results into a single dict.
    async def get_batch_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """Async batch — fetches all symbols concurrently in the thread pool."""
        results = await asyncio.gather(*[self.get_quote(s) for s in symbols])
        return {
            "status": "ok",
            "quotes": {s: r for s, r in zip(symbols, results, strict=False)},
            "data_timestamp": datetime.now(UTC).isoformat(),
            "source": "Yahoo Finance (via yfinance)",
        }


_shared_market_client: MarketDataClient | None = None


# Returns the process-wide MarketDataClient singleton, creating it on the first call.
def get_shared_market_client() -> MarketDataClient:
    """
    Return the process-wide MarketDataClient singleton.

    MarketDataClient is stateless (no auth token to cache), so a single instance
    shared across all tool modules avoids redundant object allocations.
    Thread-safe enough for asyncio (single-threaded event loop).
    """
    global _shared_market_client
    if _shared_market_client is None:
        _shared_market_client = MarketDataClient()
    return _shared_market_client
