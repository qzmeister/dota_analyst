"""
Unit tests for the dltv_browser cache layer.

We deliberately don't spin up Playwright here — that's an
integration test that needs the chromium binary.  These tests
verify the round-trip through `ml_data/player_wr_cache.json`:
write an entry, read it back, respect the TTL.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_ml_data(monkeypatch, tmp_path):
    """Redirect ML_DATA_DIR to a temp dir."""
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path))
    import importlib
    from business import dltv_browser
    importlib.reload(dltv_browser)
    yield tmp_path
    importlib.reload(dltv_browser)


class TestCacheIO:
    def test_read_missing_returns_empty(self, tmp_ml_data):
        from business import dltv_browser
        assert dltv_browser._read_cache() == {}

    def test_write_then_read_round_trip(self, tmp_ml_data):
        from business import dltv_browser
        dltv_browser._write_cache({"s42": {"ts": time.time(), "rates": {"YatoroG": 65.0}}})
        cache = dltv_browser._read_cache()
        assert "s42" in cache
        assert cache["s42"]["rates"]["YatoroG"] == 65.0

    def test_corrupt_cache_returns_empty(self, tmp_ml_data):
        from business import dltv_browser
        cache_file = Path(tmp_ml_data) / "player_wr_cache.json"
        cache_file.write_text("not json at all", encoding="utf-8")
        assert dltv_browser._read_cache() == {}


class TestGetCachedPlayerWinrates:
    def test_returns_none_for_missing(self, tmp_ml_data):
        from business import dltv_browser
        assert dltv_browser.get_cached_player_winrates(42) is None

    def test_returns_rates_for_fresh_entry(self, tmp_ml_data):
        from business import dltv_browser
        dltv_browser._write_cache({
            "s100": {"ts": time.time(), "rates": {"YatoroG": 65.0, "Collapse": 70.0}},
        })
        out = dltv_browser.get_cached_player_winrates(100)
        assert out == {"YatoroG": 65.0, "Collapse": 70.0}

    def test_returns_none_for_stale_entry(self, tmp_ml_data):
        from business import dltv_browser
        old_ts = time.time() - (dltv_browser.PLAYER_WR_TTL_SEC + 60)
        dltv_browser._write_cache({
            "s100": {"ts": old_ts, "rates": {"YatoroG": 65.0}},
        })
        assert dltv_browser.get_cached_player_winrates(100) is None

    def test_skips_non_numeric_rates(self, tmp_ml_data):
        """Defensive: a bad write shouldn't make us return strings as rates."""
        from business import dltv_browser
        dltv_browser._write_cache({
            "s100": {"ts": time.time(), "rates": {"YatoroG": 65.0, "bad": "high"}},
        })
        out = dltv_browser.get_cached_player_winrates(100)
        assert out == {"YatoroG": 65.0}


class TestUpdatePlayerWrCache:
    """`update_player_wr_cache` writes the cache.  We mock `fetch_player_winrates`
    to keep the test fast and offline."""

    def test_records_rates_on_success(self, tmp_ml_data):
        from business import dltv_browser
        with patch.object(
            dltv_browser, "fetch_player_winrates",
            return_value={"YatoroG": 65.0, "Collapse": 70.0},
        ):
            out = dltv_browser.update_player_wr_cache(100, "https://example.com/matches/100")
        assert out == {"YatoroG": 65.0, "Collapse": 70.0}
        cached = dltv_browser.get_cached_player_winrates(100)
        assert cached == {"YatoroG": 65.0, "Collapse": 70.0}

    def test_empty_rates_cached_too(self, tmp_ml_data):
        """An empty result (page rendered, no WR visible) is a valid
        signal — we still cache it so we don't hammer dltv.org."""
        from business import dltv_browser
        with patch.object(dltv_browser, "fetch_player_winrates", return_value={}):
            out = dltv_browser.update_player_wr_cache(100, "https://example.com")
        assert out == {}
        # Subsequent read returns the empty dict (not None — we still
        # want the get_cached_player_winrates() to skip the re-fetch
        # even when the result is empty).
        cached = dltv_browser.get_cached_player_winrates(100)
        assert cached == {}

    def test_discovery_error_caches_failure(self, tmp_ml_data):
        """When the upstream is down we cache the error so we don't
        retry within the TTL window."""
        from business import dltv_browser
        from business.exceptions import DiscoveryError
        with patch.object(
            dltv_browser, "fetch_player_winrates",
            side_effect=DiscoveryError("chromium missing"),
        ):
            out = dltv_browser.update_player_wr_cache(100, "https://example.com")
        assert out is None
        # And the cache reflects the failure (so the next poll skips).
        cache = dltv_browser._read_cache()
        assert "s100" in cache
        assert cache["s100"].get("error", "").startswith("chromium missing")


# ========================================================================== #
# v0.4.0.3: match_state cache failure semantics
#
# The previous `update_match_state_cache` wrote `match_state: {}`
# on Playwright fetch failure, overwriting any real prior state,
# and `get_cached_match_state` then returned that empty dict.
# Two regressions this caused:
#   1. A single failed fetch blanked the live card for 1h
#      (MATCH_STATE_TTL_SEC) even if the publisher's NEXT tick
#      would have succeeded.
#   2. The user saw cards "blink to empty" every time the
#      socket reconnected (because the watchlist fallback path
#      had no real state to overlay).
#
# The fix: on failure, keep the previous real `match_state` if
# we have one, only stamp `error` + `error_ts`, and return None
# from `get_cached_match_state` whenever the dict is empty
# (so callers fall through to the socket state instead of
# rendering an empty card).
# ========================================================================== #

class TestUpdateMatchStateCacheFailures:
    """Pin the v0.4.0.3 cache-failure semantics."""

    def _patched_fetch(self, dltv_browser, side_effect=None, return_value=None):
        kwargs = {}
        if side_effect is not None:
            kwargs["side_effect"] = side_effect
        if return_value is not None:
            kwargs["return_value"] = return_value
        return patch.object(dltv_browser, "fetch_match_state", **kwargs)

    def test_failure_preserves_prior_real_state(self, tmp_ml_data):
        from business import dltv_browser
        from business.exceptions import DiscoveryError
        # Seed a real state.
        dltv_browser._write_cache({
            "s100": {
                "ts": time.time(),
                "url": "https://example.com/matches/100",
                "match_state": {"picks": {"radiant": [1, 2, 3]}},
            },
        })
        with self._patched_fetch(dltv_browser, DiscoveryError("goto timeout")):
            out = dltv_browser.update_match_state_cache(
                100, "https://example.com/matches/100",
            )
        assert out is None
        # The real match_state must be preserved.
        state = dltv_browser.get_cached_match_state(100)
        assert state == {"picks": {"radiant": [1, 2, 3]}}
        # And the error marker is stamped.
        cache = dltv_browser._read_cache()
        assert "s100" in cache
        assert cache["s100"].get("error", "").startswith("goto timeout")
        assert cache["s100"].get("error_ts") is not None

    def test_failure_with_no_prior_state_writes_empty(self, tmp_ml_data):
        from business import dltv_browser
        from business.exceptions import DiscoveryError
        with self._patched_fetch(dltv_browser, DiscoveryError("chromium missing")):
            out = dltv_browser.update_match_state_cache(
                100, "https://example.com/matches/100",
            )
        assert out is None
        # No prior real state → empty marker, but the read
        # still returns None (the v0.4.0.3 contract).
        assert dltv_browser.get_cached_match_state(100) is None

    def test_success_clears_error_marker(self, tmp_ml_data):
        from business import dltv_browser
        from business.exceptions import DiscoveryError
        # First write a failure marker.
        dltv_browser._write_cache({
            "s100": {
                "ts": time.time(),
                "url": "https://example.com/matches/100",
                "match_state": {},
                "error": "old error",
                "error_ts": time.time(),
            },
        })
        with self._patched_fetch(dltv_browser, return_value={"picks": {"radiant": [5]}}):
            out = dltv_browser.update_match_state_cache(
                100, "https://example.com/matches/100",
            )
        assert out == {"picks": {"radiant": [5]}}
        cache = dltv_browser._read_cache()
        # Error marker gone.
        assert "error" not in cache["s100"]
        assert "error_ts" not in cache["s100"]

    def test_steam_alias_preserves_prior_state(self, tmp_ml_data):
        from business import dltv_browser
        from business.exceptions import DiscoveryError
        # Seed real state under the steam alias key.
        dltv_browser._write_cache({
            "s8920753023": {
                "ts": time.time(),
                "url": "https://example.com/matches/8920753023",
                "match_state": {"picks": {"radiant": [7, 8]}},
            },
        })
        with self._patched_fetch(dltv_browser, DiscoveryError("timeout")):
            dltv_browser.update_match_state_cache(
                8920753023, "https://example.com/matches/8920753023",
                steam_id=8920753023,
            )
        # Both keys (s{series_id} and s{steam_id} = same here)
        # preserve the real state.
        assert dltv_browser.get_cached_match_state(8920753023) == {"picks": {"radiant": [7, 8]}}
        assert dltv_browser.get_cached_match_state_by_steam(8920753023) == {"picks": {"radiant": [7, 8]}}


class TestGetCachedMatchStateFailure:
    """Pin the v0.4.0.3 `get_cached_match_state` empty-state semantics."""

    def test_empty_state_after_failure_returns_none(self, tmp_ml_data):
        from business import dltv_browser
        dltv_browser._write_cache({
            "s100": {
                "ts": time.time(),
                "url": "https://example.com",
                "match_state": {},
                "error": "timeout",
                "error_ts": time.time(),
            },
        })
        # v0.4.0.3: empty {} is no longer returned as a valid
        # cache hit — that was the root cause of the "card goes
        # blank" regression.  Callers fall through to socket
        # state instead of rendering an empty card.
        assert dltv_browser.get_cached_match_state(100) is None

    def test_empty_state_with_no_error_returns_none(self, tmp_ml_data):
        from business import dltv_browser
        dltv_browser._write_cache({
            "s100": {
                "ts": time.time(),
                "url": "https://example.com",
                "match_state": {},
            },
        })
        # No error marker, but still empty — the contract is
        # "empty match_state is not useful data" regardless
        # of cause.
        assert dltv_browser.get_cached_match_state(100) is None

    def test_real_state_returned_normally(self, tmp_ml_data):
        from business import dltv_browser
        dltv_browser._write_cache({
            "s100": {
                "ts": time.time(),
                "url": "https://example.com",
                "match_state": {"picks": {"radiant": [1]}},
            },
        })
        assert dltv_browser.get_cached_match_state(100) == {"picks": {"radiant": [1]}}

    def test_by_steam_returns_none_for_empty(self, tmp_ml_data):
        from business import dltv_browser
        dltv_browser._write_cache({
            "s8920753023": {
                "ts": time.time(),
                "url": "https://example.com",
                "match_state": {},
                "error": "timeout",
                "error_ts": time.time(),
            },
        })
        # Watchlist path: empty state is also treated as a
        # cache miss for the alias key.
        assert dltv_browser.get_cached_match_state_by_steam(8920753023) is None
