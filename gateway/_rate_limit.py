"""
Token-bucket rate limiter, used by the gateway's middleware.

Why a custom implementation?  In-process is enough for 0.1.1
(single gateway instance).  The interface (`RateLimiter.try_consume`)
is small enough that a Redis-backed swap in 0.2.x is a one-file
change.

Algorithm
---------
Each (api_key, ip) tuple gets a bucket with:
  - `capacity` = burst (e.g. 10)
  - `refill_rate` = rpm / 60  (tokens per second)

On every request:
  1. Refill the bucket since the last touch (capped at capacity).
  2. If `tokens >= 1`, decrement and let the request through.
  3. Otherwise reject with 429 + Retry-After.

Why per (api_key, ip)?  A single API key shared by many users
shouldn't get all of them throttled by one user spamming; and a
single IP without a key should still be limited to keep the
surface small.  The tuple is the conservative choice.

`RATE_LIMIT_RPM=0` disables the limiter — handy for local dev.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple


# Bucket-eviction horizon: any bucket that hasn't been touched in
# this many seconds is dropped from the in-memory dict, so a
# long-running gateway doesn't accumulate dead state.
BUCKET_TTL_SEC = 3600.0


@dataclass
class _Bucket:
    tokens: float
    last_refill: float
    last_seen: float = field(default_factory=time.monotonic)

    def refill(self, rate: float, capacity: float, now: float) -> None:
        """Top up `tokens` by `rate * elapsed` since the last touch."""
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(capacity, self.tokens + elapsed * rate)
        self.last_refill = now
        self.last_seen = now


class RateLimiter:
    """Per-(key, ip) token bucket with `try_consume()` semantics.

    Thread-safe — uses a single lock around the bucket dict.  The
    critical section is a dict lookup + a couple of float ops, so
    lock contention is negligible compared to the I/O the gateway
    is doing anyway.
    """

    def __init__(self, rpm: int, burst: int) -> None:
        self._rpm = int(rpm)
        self._burst = max(1, int(burst))
        # tokens-per-second refill rate
        self._rate = (self._rpm / 60.0) if self._rpm > 0 else 0.0
        self._enabled = self._rpm > 0
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def capacity(self) -> int:
        return self._burst

    def try_consume(self, key: str, ip: str, cost: float = 1.0) -> Tuple[bool, int]:
        """Try to spend `cost` tokens from the (key, ip) bucket.

        Returns `(allowed, retry_after_seconds)`.  When the limiter
        is disabled, `allowed` is always True and `retry_after`
        is 0 — callers don't have to special-case dev mode.
        """
        if not self._enabled:
            return True, 0
        now = time.monotonic()
        with self._lock:
            self._evict_if_due(now)
            bucket = self._buckets.get((key, ip))
            if bucket is None:
                bucket = _Bucket(tokens=float(self._burst), last_refill=now)
                self._buckets[(key, ip)] = bucket
            bucket.refill(self._rate, float(self._burst), now)
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0
            # Compute Retry-After: how long until enough tokens
            # accumulate to cover one more `cost`.
            needed = cost - bucket.tokens
            retry_after = int(math.ceil(needed / self._rate)) if self._rate > 0 else 60
            return False, max(1, retry_after)

    def _evict_if_due(self, now: float) -> None:
        """Drop buckets idle for longer than `BUCKET_TTL_SEC`.

        Called under the lock on every `try_consume`.  Cheap when
        the dict is small (a single linear scan of last_seen); for
        a 10k+ active dict this would warrant a sorted structure.
        """
        if len(self._buckets) < 64:
            return
        cutoff = now - BUCKET_TTL_SEC
        dead = [k for k, b in self._buckets.items() if b.last_seen < cutoff]
        for k in dead:
            self._buckets.pop(k, None)

    # ---- introspection (for tests) ---- #

    def bucket_count(self) -> int:
        """How many active buckets the limiter currently tracks."""
        with self._lock:
            return len(self._buckets)

    def snapshot(self) -> Dict[Tuple[str, str], float]:
        """Return a copy of `(key, ip) -> tokens` for tests / debug."""
        with self._lock:
            return {k: v.tokens for k, v in self._buckets.items()}
