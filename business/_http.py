"""
Shared HTTP utilities for backend API clients.

Per RULES.md §1, every API client must implement:
- timeout
- retry (3 attempts, exponential backoff)
- fallback return on error

This module centralises the retry/backoff logic so individual clients
(datdota_client, dltv_client, discovery) don't reinvent it.
"""

from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request

from .exceptions import HTTPClientError
from typing import Any, Dict, Optional, Tuple


# HTTP statuses that are safe to retry. 4xx (except 429) are not — they
# indicate a client-side problem that won't fix itself.
RETRYABLE_STATUSES: Tuple[int, ...] = (429, 500, 502, 503, 504)


def _backoff_sleep(
    attempt: int,
    base: float = 1.0,
    cap: float = 30.0,
    jitter: float = 0.5,
) -> float:
    """Exponential backoff with jitter. Returns the actual delay applied."""
    delay = min(cap, base * (2 ** attempt))
    delay += random.uniform(0, jitter * base)
    time.sleep(delay)
    return delay


def request_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    retryable_statuses: Tuple[int, ...] = RETRYABLE_STATUSES,
) -> Optional[Any]:
    """Fetch JSON with exponential backoff and retry.

    Returns the parsed JSON object, or None on failure (after exhausting retries
    or hitting a non-retryable error). Never raises.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    headers = headers or {}

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or 200
                if status == 200:
                    body = resp.read()
                    if not body:
                        return None
                    return json.loads(body.decode("utf-8"))
                if status in retryable_statuses:
                    # Honor Retry-After when server provides it
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.strip().isdigit():
                        time.sleep(min(int(retry_after), 60))
                    elif attempt < retries - 1:
                        _backoff_sleep(attempt, backoff_base, backoff_cap)
                    continue
                # Non-retryable HTTP error (4xx, etc.)
                return None
        except (OSError, ValueError, HTTPClientError):
            # urllib raises URLError (OSError subclass) on network failures;
            # json.JSONDecodeError inherits from ValueError; HTTPClientError
            # covers anything already wrapped by the client layer.  All of
            # these should trigger an exponential backoff and retry.
            if attempt < retries - 1:
                _backoff_sleep(attempt, backoff_base, backoff_cap)

    return None
