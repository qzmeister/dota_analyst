"""
Unit tests for `business.ml.targets` — the per-match label extractors
for the regression heads.

The 0.2.1 trainer iterates these once and shares the resulting
matrix across every head; getting the filter logic right is the
foundation for every metric we'll ever report.
"""

from __future__ import annotations

import pytest

from business.ml.targets import (
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    extract_target,
    iter_clean_targets,
    target_duration_minutes,
    target_kills,
    target_towers,
)


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# target_towers  (0.2.2)
# --------------------------------------------------------------------------- #

class TestTargetTowers:
    def test_returns_none_when_no_bitmask(self):
        # 0.2.1 corpus — no tower_radiant / tower_dire fields.
        m = _match()
        assert target_towers(m) is None

    def test_decodes_dltv_bitmask(self):
        # Radiant destroyed 6 of 11, Dire destroyed 4 of 11.
        # 0b1110111 = 119 → bin().count("1") == 6.
        # 0b11110 = 30 → bin().count("1") == 4.
        m = _match()
        m["tower_radiant"] = 0b1110111
        m["tower_dire"] = 0b11110
        assert target_towers(m) == 10

    def test_caps_at_11_per_side(self):
        # A corrupted bitmask with 15 set bits should still be capped
        # to 11 per side (the DLTV convention is exactly 11 bits).
        m = _match()
        m["tower_radiant"] = 0xFFFF  # way more than 11
        m["tower_dire"] = 0
        assert target_towers(m) == 11

    def test_returns_none_for_non_dict(self):
        assert target_towers(None) is None
        assert target_towers("not a match") is None


# --------------------------------------------------------------------------- #
# target_kills
# --------------------------------------------------------------------------- #


def _match(
    radiant_kills=(0, 0, 0, 0, 0),
    dire_kills=(0, 0, 0, 0, 0),
    duration=30 * 60,
    radiant_victory=True,
    has_error=False,
    radiant_heroes=(1, 2, 3, 4, 5),
    dire_heroes=(6, 7, 8, 9, 10),
):
    radiant = {
        "team": {"name": "R"},
        "player_performances": [
            {"performance": {"hero": {"valve_id": h}, "kills": k}}
            for h, k in zip(radiant_heroes, radiant_kills)
        ],
    }
    dire = {
        "team": {"name": "D"},
        "player_performances": [
            {"performance": {"hero": {"valve_id": h}, "kills": k}}
            for h, k in zip(dire_heroes, dire_kills)
        ],
    }
    return {
        "match_id": 1,
        "duration": duration,
        "radiant_victory": radiant_victory,
        "has_error": has_error,
        "radiant": radiant,
        "dire": dire,
    }


# --------------------------------------------------------------------------- #
# target_kills
# --------------------------------------------------------------------------- #

class TestTargetKills:
    def test_sums_all_ten_players(self):
        m = _match(radiant_kills=(5, 4, 3, 2, 1), dire_kills=(6, 5, 4, 3, 2))
        # 15 + 20 = 35
        assert target_kills(m) == 35

    def test_zero_kills(self):
        m = _match()
        assert target_kills(m) == 0

    def test_returns_none_when_no_performances(self):
        m = {"radiant": {}, "dire": {}}
        assert target_kills(m) is None

    def test_skips_players_without_kills_field(self):
        # If a player dict has no `kills`, treat it as 0.
        m = _match()
        m["radiant"]["player_performances"][0] = {"performance": {"hero": {"valve_id": 1}}}
        m["dire"]["player_performances"][0] = {"performance": {"hero": {"valve_id": 6}}}
        # Only 4 players per side have kills, all zeros — sum is 0.
        assert target_kills(m) == 0

    def test_non_dict_match_returns_none(self):
        assert target_kills(None) is None
        assert target_kills("not a match") is None


# --------------------------------------------------------------------------- #
# target_duration_minutes
# --------------------------------------------------------------------------- #

class TestTargetDuration:
    def test_seconds_to_minutes(self):
        m = _match(duration=30 * 60)
        assert target_duration_minutes(m) == pytest.approx(30.0)

    def test_short_game_filtered(self):
        m = _match(duration=MIN_DURATION_SEC - 1)
        assert target_duration_minutes(m) is None

    def test_long_game_filtered(self):
        m = _match(duration=MAX_DURATION_SEC + 1)
        assert target_duration_minutes(m) is None

    def test_exact_boundary_accepted(self):
        m = _match(duration=MIN_DURATION_SEC)
        assert target_duration_minutes(m) == pytest.approx(10.0)
        m = _match(duration=MAX_DURATION_SEC)
        assert target_duration_minutes(m) == pytest.approx(90.0)

    def test_missing_duration(self):
        m = _match()
        m["duration"] = None
        assert target_duration_minutes(m) is None


# --------------------------------------------------------------------------- #
# extract_target
# --------------------------------------------------------------------------- #

class TestExtractTarget:
    def test_happy_path(self):
        m = _match(radiant_kills=(5, 4, 3, 2, 1), dire_kills=(6, 5, 4, 3, 2),
                   duration=35 * 60, radiant_victory=True)
        t = extract_target(m)
        assert t is not None
        assert t.winner == 1
        assert t.kills_total == 35
        assert t.duration_minutes == pytest.approx(35.0)
        assert t.radiant_hero_ids == [1, 2, 3, 4, 5]
        assert t.dire_hero_ids == [6, 7, 8, 9, 10]

    def test_dire_winner_yields_zero(self):
        m = _match(radiant_victory=False)
        t = extract_target(m)
        assert t is not None
        assert t.winner == 0

    def test_errored_match_skipped(self):
        m = _match(has_error=True)
        assert extract_target(m) is None

    def test_missing_winner_skipped(self):
        m = _match()
        del m["radiant_victory"]
        assert extract_target(m) is None

    def test_wrong_hero_count_skipped(self):
        m = _match(radiant_heroes=(1, 2, 3, 4))  # only 4 radiant heroes
        assert extract_target(m) is None

    def test_short_game_skipped(self):
        m = _match(duration=60)  # 1 min — remake
        assert extract_target(m) is None

    def test_no_player_performances_skipped(self):
        m = _match()
        m["radiant"]["player_performances"] = []
        assert extract_target(m) is None


# --------------------------------------------------------------------------- #
# iter_clean_targets
# --------------------------------------------------------------------------- #

class TestIterCleanTargets:
    def test_returns_list(self):
        matches = [_match(), _match(radiant_victory=False)]
        out = iter_clean_targets(matches)
        assert isinstance(out, list)
        assert len(out) == 2

    def test_drops_unusable_rows(self):
        good = _match()
        bad = _match(has_error=True)
        out = iter_clean_targets([good, bad])
        assert len(out) == 1

    def test_empty_input(self):
        assert iter_clean_targets([]) == []

    def test_accepts_generator(self):
        # `iter_clean_targets` calls `list(...)` internally so a
        # one-shot generator is safe.
        def gen():
            yield _match()
            yield _match(radiant_victory=False)
        out = iter_clean_targets(gen())
        assert len(out) == 2
