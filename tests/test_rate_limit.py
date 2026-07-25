"""
Unit tests for `gateway._rate_limit.RateLimiter` — the per-(key, ip)
token bucket that backs the 0.1.1 rate-limit middleware.

The middleware layer (which plugs into Starlette) is tested
through `TestClient` in `test_gateway.py`; these tests cover the
bucket math directly because the test surface is small and the
edge cases (refill timing, eviction, disabled mode) are easier to
exercise in isolation.
"""

from __future__ import annotations

import time

import pytest

from gateway._rate_limit import BUCKET_TTL_SEC, RateLimiter


# --------------------------------------------------------------------------- #
# Disabled mode (RATE_LIMIT_RPM=0)
# --------------------------------------------------------------------------- #

class TestDisabled:
    def test_disabled_when_rpm_zero(self):
        r = RateLimiter(rpm=0, burst=10)
        assert not r.enabled
        for _ in range(1000):
            allowed, retry = r.try_consume("k", "1.1.1.1")
            assert allowed is True
            assert retry == 0

    def test_disabled_does_not_track_buckets(self):
        r = RateLimiter(rpm=0, burst=10)
        for _ in range(5):
            r.try_consume("k", "1.1.1.1")
        # No buckets were ever created in disabled mode.
        assert r.bucket_count() == 0


# --------------------------------------------------------------------------- #
# Basic consume / refill
# --------------------------------------------------------------------------- #

class TestBasicConsume:
    def test_first_request_uses_full_burst(self):
        r = RateLimiter(rpm=60, burst=5)
        for _ in range(5):
            allowed, _ = r.try_consume("k", "1.1.1.1")
            assert allowed is True

    def test_burst_exhaustion_returns_429_with_retry_after(self):
        r = RateLimiter(rpm=60, burst=3)
        for _ in range(3):
            r.try_consume("k", "1.1.1.1")
        allowed, retry = r.try_consume("k", "1.1.1.1")
        assert allowed is False
        assert retry >= 1

    def test_refill_over_time(self):
        r = RateLimiter(rpm=60, burst=2)
        # Drain the bucket.
        r.try_consume("k", "1.1.1.1")
        r.try_consume("k", "1.1.1.1")
        allowed, _ = r.try_consume("k", "1.1.1.1")
        assert allowed is False

        # 60 rpm = 1 token / sec.  Wait 1.1 s and try again.
        time.sleep(1.1)
        allowed, _ = r.try_consume("k", "1.1.1.1")
        assert allowed is True


# --------------------------------------------------------------------------- #
# Per-key isolation
# --------------------------------------------------------------------------- #

class TestPerKeyIsolation:
    def test_different_keys_have_independent_buckets(self):
        r = RateLimiter(rpm=60, burst=2)
        # Drain `k1`.
        r.try_consume("k1", "1.1.1.1")
        r.try_consume("k1", "1.1.1.1")
        allowed, _ = r.try_consume("k1", "1.1.1.1")
        assert allowed is False
        # `k2` is still fresh.
        allowed, _ = r.try_consume("k2", "1.1.1.1")
        assert allowed is True

    def test_different_ips_have_independent_buckets(self):
        r = RateLimiter(rpm=60, burst=1)
        r.try_consume("k", "1.1.1.1")
        allowed, _ = r.try_consume("k", "1.1.1.1")
        assert allowed is False
        allowed, _ = r.try_consume("k", "2.2.2.2")
        assert allowed is True

    def test_different_key_pairs_are_distinct(self):
        r = RateLimiter(rpm=60, burst=1)
        # Same key, different IPs → two buckets.
        r.try_consume("k", "1.1.1.1")
        allowed, _ = r.try_consume("k", "1.1.1.1")
        assert allowed is False
        allowed, _ = r.try_consume("k", "2.2.2.2")
        assert allowed is True


# --------------------------------------------------------------------------- #
# Capacity / properties
# --------------------------------------------------------------------------- #

class TestProperties:
    def test_capacity_is_burst(self):
        r = RateLimiter(rpm=60, burst=7)
        assert r.capacity == 7

    def test_burst_floor_is_one(self):
        # 0 burst would be a denial-of-service; we silently clamp to 1.
        r = RateLimiter(rpm=60, burst=0)
        assert r.capacity == 1
        allowed, _ = r.try_consume("k", "1.1.1.1")
        assert allowed is True


# --------------------------------------------------------------------------- #
# Snapshot / introspection
# --------------------------------------------------------------------------- #

class TestSnapshot:
    def test_snapshot_reflects_consumes(self):
        r = RateLimiter(rpm=60, burst=5)
        r.try_consume("k", "1.1.1.1")
        r.try_consume("k", "1.1.1.1")
        snap = r.snapshot()
        # Two tokens consumed → 3 left.
        assert snap[("k", "1.1.1.1")] == pytest.approx(3.0, abs=0.5)

    def test_snapshot_includes_all_active_keys(self):
        r = RateLimiter(rpm=60, burst=2)
        r.try_consume("k1", "1.1.1.1")
        r.try_consume("k2", "2.2.2.2")
        snap = r.snapshot()
        assert ("k1", "1.1.1.1") in snap
        assert ("k2", "2.2.2.2") in snap


# --------------------------------------------------------------------------- #
# Eviction
# --------------------------------------------------------------------------- #

class TestEviction:
    def test_eviction_does_not_run_for_small_dicts(self):
        # The fast path: don't bother scanning when there are < 64
        # buckets.  We just want to confirm the call doesn't blow up.
        r = RateLimiter(rpm=60, burst=2)
        for i in range(10):
            r.try_consume(f"k{i}", "1.1.1.1")
        assert r.bucket_count() == 10
