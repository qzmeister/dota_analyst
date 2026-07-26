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
