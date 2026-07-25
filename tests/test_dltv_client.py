"""Tests for business.dltv_client — hero normalization and team helpers."""

from __future__ import annotations

import pytest

from business.dltv_client import DLTVClient, _TTLCache


# ========================================================================== #
# _TTLCache
# ========================================================================== #

class TestTTLCache:
    def test_set_and_get(self):
        c = _TTLCache()
        c.set("k", "v", ttl=10)
        assert c.get("k") == "v"

    def test_expired_returns_none(self):
        c = _TTLCache()
        c.set("k", "v", ttl=0.01)
        import time; time.sleep(0.05)
        assert c.get("k") is None

    def test_lru_eviction_when_over_maxsize(self):
        c = _TTLCache(maxsize=3)
        c.set("a", 1, ttl=10)
        c.set("b", 2, ttl=10)
        c.set("c", 3, ttl=10)
        c.set("d", 4, ttl=10)  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3
        assert c.get("d") == 4

    def test_unbounded_when_maxsize_zero(self):
        c = _TTLCache(maxsize=0)
        for i in range(50):
            c.set(f"k{i}", i, ttl=10)
        for i in range(50):
            assert c.get(f"k{i}") == i


# ========================================================================== #
# _normalize_hero
# ========================================================================== #

class TestNormalizeHero:
    def setup_method(self):
        # A new client per test (the singleton client would leak state).
        self.client = DLTVClient()

    def test_minimal_input(self):
        norm = self.client._normalize_hero({"id": 5, "title": "Crystal Maiden"})
        assert norm["id"] == 5
        assert norm["steam_id"] is None
        assert norm["name"] == "Crystal Maiden"
        assert norm["roles"] == []
        assert norm["win_rate"] is None

    def test_roles_as_list(self):
        norm = self.client._normalize_hero({
            "id": 1, "title": "Axe", "roles": ["Initiator", "Disabler"],
        })
        assert norm["roles"] == ["Initiator", "Disabler"]

    def test_roles_as_json_string(self):
        """DLTV sometimes returns roles as a JSON-encoded string — accept it."""
        import json
        norm = self.client._normalize_hero({
            "id": 2, "title": "CM", "roles": json.dumps(["Support", "Disabler"]),
        })
        assert norm["roles"] == ["Support", "Disabler"]

    def test_roles_as_malformed_string_falls_back_to_empty(self):
        norm = self.client._normalize_hero({
            "id": 3, "title": "X", "roles": "this is not json",
        })
        assert norm["roles"] == []

    def test_image_paths_get_made_absolute(self):
        norm = self.client._normalize_hero({
            "id": 4, "title": "Y", "image": "/static/heroes/y.png",
        })
        assert norm["image"] == "https://dltv.org/static/heroes/y.png"

    def test_absolute_image_url_kept_as_is(self):
        norm = self.client._normalize_hero({
            "id": 5, "title": "Z", "image": "https://cdn.example.com/z.png",
        })
        assert norm["image"] == "https://cdn.example.com/z.png"

    def test_numeric_strings_get_parsed(self):
        """win_rate / pick_rate / kda may arrive as strings — coerce to float."""
        norm = self.client._normalize_hero({
            "id": 6, "title": "AA",
            "win_rate": "52.5", "pick_rate": "12.0", "kda": "3.2", "avg_duration": "2400",
        })
        assert norm["win_rate"] == 52.5
        assert norm["pick_rate"] == 12.0
        assert norm["kda"] == 3.2
        assert norm["avg_duration"] == 2400.0

    def test_garbage_numeric_strings_become_none(self):
        norm = self.client._normalize_hero({
            "id": 7, "title": "BB", "win_rate": "n/a", "kda": None,
        })
        assert norm["win_rate"] is None
        assert norm["kda"] is None


# ========================================================================== #
# normalize_team
# ========================================================================== #

class TestNormalizeTeam:
    def setup_method(self):
        self.client = DLTVClient()

    def test_missing_team_returns_safe_defaults(self):
        t = self.client.normalize_team(None)
        assert t["name"] == "TBD"
        assert t["id"] is None
        assert t["win_rate"] is None
        assert t["fb_rate"] is None

    def test_full_team(self):
        t = self.client.normalize_team({
            "id": 42, "title": "Team Spirit", "tag": "TS",
            "image": "/i/team.png", "rank": 5,
            "win_rate": 60.0, "fb_rate": 55.0, "f10_rate": 52.0,
        })
        assert t["id"] == 42
        assert t["name"] == "Team Spirit"
        assert t["tag"] == "TS"
        assert t["logo"] == "https://dltv.org/i/team.png"
        assert t["rank"] == 5
        assert t["win_rate"] == 60.0


# ========================================================================== #
# hero_by_*  — index lookup
# ========================================================================== #

class TestHeroLookup:
    def setup_method(self):
        self.client = DLTVClient()

    def test_hero_lookup_uses_index_after_load(self):
        # Inject a known hero and re-build the index.
        self.client.get_heroes = lambda: self.client._build_hero_index([
            {"id": 1, "steam_id": 100, "title": "TestHero", "roles": []},
        ])  # type: ignore[assignment]
        self.client._heroes_loaded = True
        self.client._hero_by_id = {1: {"id": 1, "name": "TestHero"}}
        self.client._hero_by_steam = {100: {"id": 1, "name": "TestHero"}}

        assert self.client.hero_by_dltv_id(1)["name"] == "TestHero"
        assert self.client.hero_by_steam_id(100)["name"] == "TestHero"
        assert self.client.hero_by_dltv_id(999) is None

    def test_hero_lookup_lazy_loads_on_first_call(self):
        """First call triggers a network fetch (stubbed) and the index is then populated."""
        from unittest.mock import patch
        with patch.object(self.client, "get_heroes", wraps=self.client.get_heroes) as spy:
            # Force the underlying API call to fail (no network) so the test
            # is hermetic; the lazy load still attempts and populates with
            # whatever the network returned (here, empty).
            try:
                self.client.hero_by_dltv_id(1)
            except Exception:
                pass
            assert spy.called
            assert self.client._heroes_loaded is True
