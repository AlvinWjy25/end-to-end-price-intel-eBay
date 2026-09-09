"""
ebay_token_manager.py

Wraps ingest.py's get_access_token() with expiry tracking and
automatic refresh, so the FastAPI service doesn't need to fetch a
fresh token on every request (unnecessary network round-trip) nor
risk silently failing once the token from startup expires (~2 hours,
per eBay's OAuth Client Credentials flow).

NOTE: ingest.py's get_access_token() currently only returns the
access_token string, discarding the `expires_in` field eBay's OAuth
response actually includes. Rather than modify ingest.py (a working
batch-ingestion script, best left alone to avoid affecting the
existing ingest pipeline), this wrapper assumes eBay's documented
default of ~2 hours (7200s) and applies a safety buffer. If ingest.py
is ever updated to expose expires_in directly, prefer switching this
module to use the real value instead of the assumed constant.
"""

from __future__ import annotations

import time
import threading

from src.ingestion.ingest import get_access_token  # adjust import path if needed


# eBay's Client Credentials tokens are documented as ~2 hours
# (7200s). Refresh a bit early to avoid a request racing against
# expiry mid-flight.
_ASSUMED_TOKEN_LIFETIME_SECONDS = 7200
_REFRESH_SAFETY_BUFFER_SECONDS = 300  # refresh 5 min before assumed expiry


class EbayTokenManager:
    """Thread-safe holder for the current eBay access token, refreshing
    it transparently when it's close to (assumed) expiry.

    Usage:
        token_manager = EbayTokenManager()
        token = token_manager.get_token()  # fetches on first call,
                                            # reuses/refreshes after
    """

    def __init__(self):
        self._token: str | None = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    def _is_stale(self) -> bool:
        if self._token is None:
            return True
        age = time.monotonic() - self._fetched_at
        return age >= (_ASSUMED_TOKEN_LIFETIME_SECONDS - _REFRESH_SAFETY_BUFFER_SECONDS)

    def get_token(self) -> str:
        # Fast path: token still fresh, no lock contention needed for
        # the common case.
        if not self._is_stale():
            return self._token

        with self._lock:
            # Re-check after acquiring the lock in case another
            # thread already refreshed while we were waiting.
            if self._is_stale():
                self._token = get_access_token()
                self._fetched_at = time.monotonic()
            return self._token
