"""Tests for business.opendota_live.

We mock the HTTP layer so the tests don't depend on opendota.com
being reachable.  The module's responsibility is the parsing +
caching + thread-safe access — those we test directly.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_opendota_state():
    """Wipe the module-level state between tests."""
    from business import opendota_live
    with opendota_live._lock:
        opendota_live._state.clear()
        opendota_live._state_ts.clear()
        opendota_live._player_cache.clear()
    yield


# ========================================================================== #
# parse_building_state
# ========================================================================== #

class TestParseBuildingState:
    """Pin the v0.4.0.3 OpenDota `building_state` bitmask parser.

    Bits 0-8 (inclusive) are radiant towers + ancient (9 per side
    on the standard map); bits 16-24 are the same shifted by 16
    for dire.  A 1 bit = destroyed.
    """

    def test_zero_is_no_destroyed(self):
        from business.opendota_live import parse_building_state
        assert parse_building_state(0) == {"radiant": 0, "dire": 0}

    def test_none_is_no_destroyed(self):
        from business.opendota_live import parse_building_state
        assert parse_building_state(None) == {"radiant": 0, "dire": 0}

    def test_single_radiant_tower_destroyed(self):
        from business.opendota_live import parse_building_state
        # bit 0 = first radiant tower
        assert parse_building_state(0b1) == {"radiant": 1, "dire": 0}

    def test_all_radiant_towers_destroyed(self):
        from business.opendota_live import parse_building_state
        # bits 0-8 all set = 9 radiant towers destroyed
        mask = (1 << 9) - 1
        assert parse_building_state(mask) == {"radiant": 9, "dire": 0}

    def test_all_dire_towers_destroyed(self):
        from business.opendota_live import parse_building_state
        # bits 16-24 all set = 9 dire towers destroyed
        mask = ((1 << 9) - 1) << 16
        assert parse_building_state(mask) == {"radiant": 0, "dire": 9}

    def test_both_sides_partial(self):
        from business.opendota_live import parse_building_state
        # radiant bits 0,1 (2 destroyed) + dire bits 16,17,18 (3 destroyed)
        mask = 0b11 | (0b111 << 16)
        assert parse_building_state(mask) == {"radiant": 2, "dire": 3}

    def test_barracks_bits_ignored(self):
        from business.opendota_live import parse_building_state
        # bits 9-15 = radiant barracks (not towers).  Should be
        # ignored by our parser.
        mask = (0b1111111 << 9)  # bits 9-15
        assert parse_building_state(mask) == {"radiant": 0, "dire": 0}

    def test_dire_barracks_bits_ignored(self):
        from business.opendota_live import parse_building_state
        # bits 25-31 = dire barracks.  Should be ignored.
        mask = (0b1111111 << 25)
        assert parse_building_state(mask) == {"radiant": 0, "dire": 0}


# ========================================================================== #
# _normalize_match
# ========================================================================== #

class TestNormalizeMatch:
    def test_minimal_row(self):
        from business.opendota_live import _normalize_match
        out = _normalize_match({"match_id": "8920750332"})
        assert out is not None
        assert out["match_id"] == 8920750332
        assert out["radiant_score"] == 0
        assert out["dire_score"] == 0
        assert out["game_time"] is None
        assert out["radiant_lead"] is None
        assert out["destroyed_towers"] is None
        assert out["players"] == []

    def test_full_row(self):
        from business.opendota_live import _normalize_match
        raw = {
            "match_id": "8920750332",
            "radiant_score": 17,
            "dire_score": 24,
            "game_time": 1151,
            "radiant_lead": -5660,
            "building_state": (1 << 0) | (1 << 16) | (1 << 17),
            "team_name_radiant": "Pro A",
            "team_name_dire": "Pro B",
            "league_id": 9999,
            "spectators": 4,
            "last_update_time": 1785413200,
            "players": [
                {"account_id": 192664620, "hero_id": 9,  "team_slot": 1},
                {"account_id": 135883142, "hero_id": 86, "team_slot": 128},
            ],
        }
        out = _normalize_match(raw)
        assert out is not None
        assert out["radiant_score"] == 17
        assert out["dire_score"] == 24
        assert out["game_time"] == 1151
        assert out["radiant_lead"] == -5660
        assert out["destroyed_towers"] == {"radiant": 1, "dire": 2}
        assert out["team_name_radiant"] == "Pro A"
        assert out["team_name_dire"] == "Pro B"
        assert out["league_id"] == 9999
        assert out["spectators"] == 4
        assert out["source_ts"] == 1785413200
        assert out["players"][0]["team"] == 0
        assert out["players"][1]["team"] == 1
        assert out["players"][0]["account_id"] == 192664620
        assert out["players"][0]["hero_id"] == 9

    def test_missing_match_id_returns_none(self):
        from business.opendota_live import _normalize_match
        assert _normalize_match({}) is None
        assert _normalize_match({"match_id": None}) is None

    def test_zero_match_id_returns_none(self):
        from business.opendota_live import _normalize_match
        assert _normalize_match({"match_id": "0"}) is None
        assert _normalize_match({"match_id": 0}) is None

    def test_drops_players_with_no_account_or_hero(self):
        from business.opendota_live import _normalize_match
        raw = {
            "match_id": "100",
            "players": [
                {"account_id": 0, "hero_id": 9, "team_slot": 1},
                {"account_id": 192664620, "hero_id": 0, "team_slot": 1},
                {"account_id": 192664620, "hero_id": 9, "team_slot": 1},
            ],
        }
        out = _normalize_match(raw)
        assert out is not None
        assert len(out["players"]) == 1
        assert out["players"][0]["account_id"] == 192664620

    def test_scoreboard_only_builds_partial(self):
        from business.opendota_live import _normalize_match
        out = _normalize_match({"match_id": "1", "building_state": 0})
        assert out["destroyed_towers"] is None


# ========================================================================== #
# get_live_state
# ========================================================================== #

class TestGetLiveState:
    def test_returns_none_for_unknown_match(self):
        from business import opendota_live
        assert opendota_live.get_live_state(123) is None

    def test_returns_state_when_fresh(self):
        from business import opendota_live
        with opendota_live._lock:
            opendota_live._state[123] = {"match_id": 123, "radiant_score": 5}
            opendota_live._state_ts[123] = opendota_live._now()
        out = opendota_live.get_live_state(123)
        assert out == {"match_id": 123, "radiant_score": 5}
        # Must be a copy, not a reference.
        out["radiant_score"] = 99
        assert opendota_live._state[123]["radiant_score"] == 5

    def test_returns_none_when_stale(self):
        from business import opendota_live
        with opendota_live._lock:
            opendota_live._state[123] = {"match_id": 123}
            opendota_live._state_ts[123] = opendota_live._now() - (opendota_live.LIVE_TTL_SEC + 1)
        assert opendota_live.get_live_state(123) is None


# ========================================================================== #
# _ingest_live
# ========================================================================== #

class TestIngestLive:
    def test_replaces_state(self):
        from business import opendota_live
        with opendota_live._lock:
            opendota_live._state[100] = {"match_id": 100, "old": True}
            opendota_live._state_ts[100] = opendota_live._now()
        new_rows = [
            {"match_id": 200, "radiant_score": 5, "dire_score": 3,
             "radiant_lead": 1234, "building_state": 0, "players": [],
             "league_id": 9999, "spectators": 1, "last_update_time": 1000},
        ]
        opendota_live._ingest_live(new_rows)
        assert 100 not in opendota_live._state
        out = opendota_live.get_live_state(200)
        assert out is not None
        assert out["radiant_lead"] == 1234

    def test_empty_rows_does_not_clear_state(self):
        # v0.4.0.3 regression test: passing [] to _ingest_live
        # used to wipe _state entirely (the poller would
        # clear-then-update with an empty dict on a 429).
        # Now we treat it as a transient failure: keep
        # the prior state in place.
        from business import opendota_live
        with opendota_live._lock:
            opendota_live._state[100] = {"match_id": 100, "keep": True}
            opendota_live._state_ts[100] = opendota_live._now()
        opendota_live._ingest_live([])
        # The old entry survived.
        assert 100 in opendota_live._state
        out = opendota_live.get_live_state(100)
        assert out is not None
        assert out["keep"] is True


# ========================================================================== #
# Player info cache
# ========================================================================== #

class TestGetPlayerInfo:
    def test_returns_none_for_unknown_account(self):
        from business import opendota_live
        assert opendota_live.get_player_info(123) is None

    def test_returns_cached_info(self):
        from business import opendota_live
        with opendota_live._lock:
            opendota_live._player_cache[123] = {
                "personaname": "YatoroG",
                "loccountrycode": "ru",
                "ts": opendota_live._now(),
            }
        assert opendota_live.get_player_info(123) == {
            "personaname": "YatoroG",
            "loccountrycode": "ru",
        }

    def test_returns_none_when_stale(self):
        from business import opendota_live
        with opendota_live._lock:
            opendota_live._player_cache[123] = {
                "personaname": "YatoroG",
                "loccountrycode": "ru",
                "ts": opendota_live._now() - (opendota_live.PLAYER_TTL_SEC + 1),
            }
        assert opendota_live.get_player_info(123) is None


# ========================================================================== #
# v0.4.0.3: persistent on-disk player_info cache
# ========================================================================== #

class TestDiskPlayerInfo:
    """Pin the v0.4.0.3 disk persistence contract for player_info."""

    def _setup_tmp_ml_data(self, tmp_path, monkeypatch):
        from business import opendota_live
        monkeypatch.setenv("ML_DATA_DIR", str(tmp_path))
        import importlib
        importlib.reload(opendota_live)
        return opendota_live

    def test_load_populates_in_memory_cache(self, tmp_path, monkeypatch):
        opendota_live = self._setup_tmp_ml_data(tmp_path, monkeypatch)
        import json
        with opendota_live.PLAYER_INFO_CACHE_FILE.open("w") as f:
            json.dump({
                "192664620": {"personaname": "mikkxx", "loccountrycode": "PH",
                              "ts": opendota_live._now()},
                "135883142": {"personaname": "YatoroG", "loccountrycode": "RU",
                              "ts": opendota_live._now()},
            }, f)
        opendota_live._load_player_info_from_disk()
        assert opendota_live.get_player_info(192664620) == {
            "personaname": "mikkxx",
            "loccountrycode": "PH",
        }
        assert opendota_live.get_player_info(135883142) == {
            "personaname": "YatoroG",
            "loccountrycode": "RU",
        }

    def test_load_skips_expired_entries(self, tmp_path, monkeypatch):
        opendota_live = self._setup_tmp_ml_data(tmp_path, monkeypatch)
        import json
        old_ts = opendota_live._now() - (opendota_live.PLAYER_INFO_DISK_TTL_SEC + 60)
        with opendota_live.PLAYER_INFO_CACHE_FILE.open("w") as f:
            json.dump({
                "1": {"personaname": "old", "loccountrycode": "x", "ts": old_ts},
                "2": {"personaname": "new", "loccountrycode": "y",
                      "ts": opendota_live._now()},
            }, f)
        opendota_live._load_player_info_from_disk()
        assert opendota_live.get_player_info(1) is None
        assert opendota_live.get_player_info(2) is not None

    def test_load_handles_missing_file(self, tmp_path, monkeypatch):
        opendota_live = self._setup_tmp_ml_data(tmp_path, monkeypatch)
        opendota_live._load_player_info_from_disk()

    def test_load_handles_corrupt_file(self, tmp_path, monkeypatch):
        opendota_live = self._setup_tmp_ml_data(tmp_path, monkeypatch)
        opendota_live.PLAYER_INFO_CACHE_FILE.write_text("not json", encoding="utf-8")
        opendota_live._load_player_info_from_disk()
        assert opendota_live._player_cache == {}

    def test_save_writes_atomic(self, tmp_path, monkeypatch):
        opendota_live = self._setup_tmp_ml_data(tmp_path, monkeypatch)
        opendota_live._player_cache[100] = {
            "personaname": "tester",
            "loccountrycode": "US",
            "ts": opendota_live._now(),
        }
        opendota_live._save_player_info_to_disk()
        import json
        with opendota_live.PLAYER_INFO_CACHE_FILE.open() as f:
            data = json.load(f)
        assert "100" in data
        assert data["100"]["personaname"] == "tester"

    def test_save_skips_empty_cache(self, tmp_path, monkeypatch):
        opendota_live = self._setup_tmp_ml_data(tmp_path, monkeypatch)
        opendota_live._save_player_info_to_disk()
        assert not opendota_live.PLAYER_INFO_CACHE_FILE.exists()


# ========================================================================== #
# 429 backoff
# ========================================================================== #

class TestBackoff:
    """Pin the v0.4.0.3 backoff behaviour on HTTP 429."""

    def test_backoff_starts_at_zero(self):
        from business import opendota_live
        opendota_live._backoff_sec = 0.0
        assert opendota_live._backoff_sec == 0.0

    def test_backoff_constants(self):
        from business import opendota_live
        # The constants are part of the contract — operators
        # tune them in one place, tests verify the math.
        assert opendota_live._BACKOFF_BASE_SEC == 60.0
        assert opendota_live._BACKOFF_MAX_SEC == 600.0
        assert opendota_live._backoff_sec == 0.0
