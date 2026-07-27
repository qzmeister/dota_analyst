"""Tests for backend.analysis — the heuristic draft-analysis engine.

These tests are pure-function tests, so they don't need any mocks or
network. They cover the main prediction branches in `analyze()` and
the bitmask helper `decode_towers()`.
"""

from __future__ import annotations

import pytest

from business.analysis import (
    analyze,
    analyze_map_with_verdict,
    decode_towers,
    map_verdicts,
    FALLBACK_DURATION_MIN,
    FALLBACK_WR,
    BASE_TOTAL_KILLS,
    KILLS_FLOOR,
    KILLS_CEILING,
)


# ========================================================================== #
# analyze() — happy path & invariants
# ========================================================================== #

class TestAnalyzeBasics:
    def test_returns_all_six_predictions(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        for key in ("winner", "kills", "duration_min", "towers",
                    "first_to_15", "multikill", "confidence"):
            assert key in result, f"missing prediction: {key}"

    def test_winner_probability_in_range(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        assert 50 <= result["winner"]["probability"] <= 100
        assert 0 <= result["winner"]["prob_radiant"] <= 100

    def test_stronger_team_favored(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        """Team A has higher win_rate (62 vs 48) — should be predicted to win."""
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        # probability in [0..100] where the reported prob_radiant > 50 means A wins
        assert result["winner"]["prob_radiant"] > 50
        assert result["winner"]["team"] == sample_team_a["name"]

    def test_kills_within_bounds(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        assert KILLS_FLOOR <= result["kills"]["total"] <= KILLS_CEILING
        assert result["kills"]["radiant"] + result["kills"]["dire"] == result["kills"]["total"]

    def test_duration_reasonable(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        # Avg Dota 2 pro match: 25-55 min. Sanity clamp.
        assert 20 <= result["duration_min"] <= 60

    def test_empty_heroes_uses_fallbacks(self, sample_team_a, sample_team_b):
        """No heroes at all — should still return a result using fallback values."""
        result = analyze(sample_team_a, sample_team_b, [], [])
        # confidence scales with completeness; 0 heroes -> low confidence
        assert 0.0 <= result["confidence"] <= 1.0
        # Winner is still computed from team aggregates
        assert "team" in result["winner"]
        assert "probability" in result["winner"]

    def test_confidence_scales_with_completeness(self, sample_team_a, sample_team_b):
        """More heroes picked => higher confidence (up to the configured cap)."""
        empty = analyze(sample_team_a, sample_team_b, [], [])
        # build 10 heroes with neutral stats
        heroes = [{"id": i, "name": f"H{i}", "win_rate": 50, "kda": 3, "avg_duration": 38 * 60, "roles": []} for i in range(10)]
        full = analyze(sample_team_a, sample_team_b, heroes[:5], heroes[5:])
        assert full["confidence"] > empty["confidence"]

    def test_over_under_side_consistent(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        """over/under side must be one of {over, under} and threshold positive."""
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        for key in ("kills_total_over_under", "total_over_under"):
            ou = result[key]
            assert ou["side"] in ("over", "under")
            assert isinstance(ou["threshold"], int)
            assert ou["threshold"] > 0

    def test_multikill_level_is_one_of_three(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        assert result["multikill"]["level"] in ("Low", "Medium", "High")

    def test_over_under_threshold_format_includes_mmss(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        formatted = result["total_over_under"]["formatted"]
        # "MM:SS" format
        import re
        assert re.match(r"^\d+:\d{2}$", formatted), formatted

    def test_towers_over_under_present_and_consistent(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        """v0.3.24g: `towers_over_under` mirrors the kills/duration
        over-under shape.  Side ∈ {over, under}, threshold > 0, and
        when the predicted total is below the threshold the bet is
        on the "over" side (heuristic over-estimates so we bet
        contrarian)."""
        heroes_a, heroes_b = sample_heroes_balanced
        result = analyze(sample_team_a, sample_team_b, heroes_a, heroes_b)
        ou = result.get("towers_over_under")
        assert ou is not None, "analyze() must include towers_over_under"
        assert ou["side"] in ("over", "under")
        assert isinstance(ou["threshold"], int)
        assert ou["threshold"] > 0
        towers_total = result["towers"]["total"]
        # Contrarian logic: low total -> over, high total -> under.
        # The boundary is TOWER_OVER_UNDER_THRESHOLD (10).
        from business.analysis import TOWER_OVER_UNDER_THRESHOLD, BET_THRESHOLD_OFFSET
        if towers_total >= TOWER_OVER_UNDER_THRESHOLD:
            assert ou["side"] == "under"
            assert ou["threshold"] == towers_total
        else:
            assert ou["side"] == "over"
            assert ou["threshold"] == towers_total + BET_THRESHOLD_OFFSET


# ========================================================================== #
# decode_towers()
# ========================================================================== #

class TestDecodeTowers:
    def test_zero_mask(self):
        # all towers still standing -> 0 destroyed
        assert decode_towers(0) == 0

    def test_full_mask(self):
        # all 11 bits set -> 11 destroyed
        assert decode_towers((1 << 11) - 1) == 11

    def test_specific_value(self):
        # 1792 = 0b11100000000 -> 3 destroyed (verified empirically in code)
        assert decode_towers(1792) == 3

    def test_capped_at_eleven(self):
        # any value with all 11 bits set -> exactly 11, never more
        assert decode_towers(0xFFFF) == 11
        assert decode_towers(0b11111111111) == 11

    def test_non_int_returns_none(self):
        assert decode_towers(None) is None
        assert decode_towers("0b101") is None
        assert decode_towers([1, 0, 1]) is None
        assert decode_towers(0.0) is None  # float rejected


# ========================================================================== #
# map_verdicts() — comparing predictions to actuals
# ========================================================================== #

class TestMapVerdicts:
    @pytest.fixture
    def base_prediction(self):
        return {
            "winner": {"team": "Team Spirit", "probability": 60, "prob_radiant": 60},
            "kills": {"total": 48, "radiant": 26, "dire": 22},
            "duration_min": 38.0,
            "kills_total_over_under": {"side": "over", "threshold": 47},
            "total_over_under": {"side": "under", "threshold": 38, "formatted": "38:00"},
            "towers": {"total": 11, "radiant": 8, "dire": 3},
            "first_to_15": {"team": "Team Spirit", "probability": 60},
            "first_blood": {"team": "Team Spirit", "probability": 60},
            "multikill": {"level": "Medium", "likely_side": "Team Spirit"},
        }

    def test_winner_correct(self, base_prediction):
        actual = {"winner_team": "Team Spirit", "duration_min": 38.0, "kills_total": 48,
                  "towers_total": 11, "fb_side": "radiant", "f15_side": "radiant"}
        verdicts = map_verdicts(base_prediction, actual, "Team Spirit", "Opponent")
        assert verdicts["winner"] is True

    def test_winner_incorrect(self, base_prediction):
        actual = {"winner_team": "Opponent", "duration_min": 38.0, "kills_total": 48,
                  "towers_total": 11, "fb_side": "dire", "f15_side": "dire"}
        verdicts = map_verdicts(base_prediction, actual, "Team Spirit", "Opponent")
        assert verdicts["winner"] is False

    def test_kills_over_under(self, base_prediction):
        # base_prediction says "over 47" — actual 50 should be win, 40 should be loss
        actual_50 = {"winner_team": "Team Spirit", "duration_min": 38.0, "kills_total": 50,
                     "towers_total": 11, "fb_side": "radiant", "f15_side": "radiant"}
        actual_40 = dict(actual_50, kills_total=40)
        assert map_verdicts(base_prediction, actual_50, "A", "B")["kills_total"] is True
        assert map_verdicts(base_prediction, actual_40, "A", "B")["kills_total"] is False

    def test_duration_under_equality(self, base_prediction):
        # base_prediction says "under 38" -> wins if actual <= 38
        actual_eq = {"winner_team": "A", "duration_min": 38.0, "kills_total": 48,
                     "towers_total": 11, "fb_side": "radiant", "f15_side": "radiant"}
        actual_over = dict(actual_eq, duration_min=39.0)
        assert map_verdicts(base_prediction, actual_eq, "A", "B")["duration"] is True
        assert map_verdicts(base_prediction, actual_over, "A", "B")["duration"] is False

    def test_missing_actual_returns_none(self, base_prediction):
        actual = {"winner_team": "A", "duration_min": 38.0, "kills_total": None,
                  "towers_total": None, "fb_side": None, "f15_side": None}
        verdicts = map_verdicts(base_prediction, actual, "A", "B")
        # Winner can still be checked if string, but numeric verdicts must be None
        assert verdicts["kills_total"] is None
        assert verdicts["towers_total"] is None

    def test_side_translation_radiant_dire(self):
        """fb_side and f15_side are 'radiant'/'dire' — must translate to team_a/team_b."""
        prediction = {
            "winner": {"team": "A", "probability": 60, "prob_radiant": 60},
            "kills": {"total": 48, "radiant": 26, "dire": 22},
            "duration_min": 38.0,
            "kills_total_over_under": {"side": "over", "threshold": 47},
            "total_over_under": {"side": "under", "threshold": 38, "formatted": "38:00"},
            "towers": {"total": 11, "radiant": 8, "dire": 3},
            "first_to_15": {"team": "A", "probability": 60},
            "first_blood": {"team": "A", "probability": 60},
            "multikill": {"level": "Medium", "likely_side": "A"},
        }
        actual = {"winner_team": "A", "duration_min": 38.0, "kills_total": 48,
                  "towers_total": 11, "fb_side": "radiant", "f15_side": "radiant"}
        verdicts = map_verdicts(prediction, actual, "A", "B")
        assert verdicts["first_blood"] is True
        assert verdicts["first_to_15"] is True

    def test_side_translation_dire(self):
        """'dire' must map to team_b_name, not team_a_name."""
        prediction = {
            "winner": {"team": "B", "probability": 60, "prob_radiant": 40},
            "kills": {"total": 48, "radiant": 22, "dire": 26},
            "duration_min": 38.0,
            "kills_total_over_under": {"side": "over", "threshold": 47},
            "total_over_under": {"side": "under", "threshold": 38, "formatted": "38:00"},
            "towers": {"total": 11, "radiant": 3, "dire": 8},
            "first_to_15": {"team": "B", "probability": 60},
            "first_blood": {"team": "B", "probability": 60},
            "multikill": {"level": "Medium", "likely_side": "B"},
        }
        actual = {"winner_team": "B", "duration_min": 38.0, "kills_total": 48,
                  "towers_total": 11, "fb_side": "dire", "f15_side": "dire"}
        verdicts = map_verdicts(prediction, actual, "A", "B")
        assert verdicts["first_blood"] is True
        assert verdicts["first_to_15"] is True


# ========================================================================== #
# analyze_map_with_verdict() — integration of analyze + map_verdicts
# ========================================================================== #

class TestAnalyzeMapWithVerdict:
    def test_returns_prediction_and_verdict(self, sample_team_a, sample_team_b, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        actual = {
            "winner_team": sample_team_a["name"],
            "duration_min": 38.0,
            "kills_total": 48,
            "towers_total": 11,
            "fb_side": "radiant",
            "f15_side": "radiant",
        }
        result = analyze_map_with_verdict(sample_team_a, sample_team_b, heroes_a, heroes_b, actual)
        assert "prediction" in result
        assert "verdict" in result
        # first_blood must be in the prediction now (analyze() didn't expose it before)
        assert "first_blood" in result["prediction"]
