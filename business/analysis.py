"""
Heuristic draft-analysis engine (MVP).

Produces the six predictions requested for a live match, from data we actually
have: team aggregates (win_rate, first-blood rate, first-10 rate, rank),
the drafted heroes and their meta (win_rate, avg_duration, kda, roles).

These are transparent heuristics tuned to pro-scene (patch 7.41d) baselines.
They are structured so a trained model / DatDota lane pairs can later replace
individual terms without changing the public shape of `analyze()`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

# Pro-scene baselines (patch 7.41d ballpark)
BASE_TOTAL_KILLS = 46.0
BASE_KDA = 3.0
FALLBACK_DURATION_MIN = 38.0
FALLBACK_WR = 50.0
FALLBACK_FB = 50.0
FALLBACK_F10 = 50.0

# roles that correlate with kill-heavy / teamfight games
FIGHT_ROLES = {"Nuker", "Initiator", "Disabler", "Escape"}


# ==========================================================================
# Algorithm calibration constants (pro-scene patch 7.41d ballpark).
# These are empirical and should be re-calibrated when the patch meta shifts.
# ==========================================================================

# --- Winner-probability logistic ----------------------------------------
# Each term is a z-score contribution; the final z is squashed by _logistic().
WINNER_WR_WEIGHT = 0.045        # per 1pp team win-rate difference
WINNER_DRAFT_WEIGHT = 0.020     # per 1pp mean per-hero meta win-rate difference
WINNER_RANK_WEIGHT = 0.015      # per 1 rank inversion (lower rank = better)
WINNER_RANK_CLAMP = 40          # cap rank delta to avoid extreme games

# --- Total kills ---------------------------------------------------------
KILLS_KDA_SLOPE = 6.0           # extra kills per +1 KDA above BASE_KDA
KILLS_FLOOR = 30                # minimum predicted total kills
KILLS_CEILING = 78              # maximum predicted total kills
KILLS_SHARE_BASE = 0.35         # favored side's share when p=0.5
KILLS_SHARE_SLOPE = 0.30        # how strongly winner prob shifts the share

# --- Over/Under bet thresholds -----------------------------------------
# The heuristic tends to over-estimate, so we bet the "contrarian" side
# (under if predicted is high, over if predicted is low).
KILLS_OVER_UNDER_THRESHOLD = 50     # kills: predicted >= 50 -> bet under
DURATION_OVER_UNDER_THRESHOLD = 40  # minutes: predicted >= 40 -> bet under
BET_THRESHOLD_OFFSET = 1            # offset added on the "over" side

# --- Tower calibration (function of dominance) -------------------------
# dominance = abs(p_a - 0.5) * 2  — 0 for balanced, 1 for one-sided.
TOWER_OVER_UNDER_THRESHOLD = 10     # towers (total): >= 10 -> bet under
                                    # 11 max per side / 22 max total — the
                                    # theoretical ceiling, so the under bet
                                    # has a built-in upper bound.
TOWER_BALANCED = 5              # towers per side in a 50/50 game
TOWER_WIN_SLOPE = 6             # extra towers the favored side takes
TOWER_LOSE_SLOPE = 4            # towers the losing side gets at most

# --- First-to-15 (early aggression) ------------------------------------
EARLY_FB_WEIGHT = 0.4           # weight of first-blood rate
EARLY_F10_WEIGHT = 0.6          # weight of first-10-minutes rate
FIRST15_EARLY_SLOPE = 0.05      # multiplier on early-rate difference
FIRST15_WINNER_SLOPE = 0.4      # bonus from overall winner prob deviation

# --- Multikill / rampage potential -------------------------------------
MULTIKILL_KILLS_THRESHOLD = 56  # if total_kills > this, add bonus
MULTIKILL_KILLS_BONUS = 2
MULTIKILL_HIGH_SCORE = 7        # fight_score >= this  -> "High"
MULTIKILL_MEDIUM_SCORE = 4      # fight_score >= this  -> "Medium"

# --- Confidence ---------------------------------------------------------
CONFIDENCE_BASE = 0.4           # base confidence when nothing is known
CONFIDENCE_COMPLETENESS = 0.6   # max uplift from full draft (0..1)


def _logistic(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mean(values: List[Optional[float]], fallback: float) -> float:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else fallback


def _val(team: Dict, key: str, fallback: float) -> float:
    v = team.get(key)
    return float(v) if isinstance(v, (int, float)) else fallback


def analyze(
    team_a: Dict,
    team_b: Dict,
    heroes_a: List[Dict],
    heroes_b: List[Dict],
) -> Dict:
    """
    team_a / team_b : normalized team dicts (radiant = A, dire = B).
    heroes_a / heroes_b : lists of hero-meta dicts for each side's picks.
    Returns a dict with the six predictions plus helper fields.
    """
    heroes_a = [h for h in heroes_a if h]
    heroes_b = [h for h in heroes_b if h]
    all_heroes = heroes_a + heroes_b

    wr_a = _val(team_a, "win_rate", FALLBACK_WR)
    wr_b = _val(team_b, "win_rate", FALLBACK_WR)
    fb_a = _val(team_a, "fb_rate", FALLBACK_FB)
    fb_b = _val(team_b, "fb_rate", FALLBACK_FB)
    f10_a = _val(team_a, "f10_rate", FALLBACK_F10)
    f10_b = _val(team_b, "f10_rate", FALLBACK_F10)
    rank_a = team_a.get("rank")
    rank_b = team_b.get("rank")

    # ---- draft quality (mean per-hero meta win-rate) ---- #
    draft_a = _mean([h.get("win_rate") for h in heroes_a], FALLBACK_WR)
    draft_b = _mean([h.get("win_rate") for h in heroes_b], FALLBACK_WR)

    # confidence scales with how complete the draft is (0..10 heroes)
    completeness = _clamp(len(all_heroes) / 10.0, 0.0, 1.0)

    # =================================================================== #
    # 3. Winner probability
    # =================================================================== #
    z = WINNER_WR_WEIGHT * (wr_a - wr_b)
    z += WINNER_DRAFT_WEIGHT * (draft_a - draft_b)
    if isinstance(rank_a, int) and isinstance(rank_b, int) and rank_a > 0 and rank_b > 0:
        z += WINNER_RANK_WEIGHT * _clamp(rank_b - rank_a, -WINNER_RANK_CLAMP, WINNER_RANK_CLAMP)
    p_a = _logistic(z)
    winner_team = team_a if p_a >= 0.5 else team_b
    winner_conf = max(p_a, 1 - p_a)

    # =================================================================== #
    # 3. Total kills
    # =================================================================== #
    mean_kda = _mean([h.get("kda") for h in all_heroes], BASE_KDA)
    total_kills = _clamp(
        BASE_TOTAL_KILLS + (mean_kda - BASE_KDA) * KILLS_KDA_SLOPE,
        KILLS_FLOOR,
        KILLS_CEILING,
    )
    share_a = KILLS_SHARE_BASE + KILLS_SHARE_SLOPE * p_a    # favored side scores more
    kills_a = round(total_kills * share_a)
    kills_b = round(total_kills * (1 - share_a))

    # --- Тотальная ставка: Over/Under ---
    if total_kills >= KILLS_OVER_UNDER_THRESHOLD:
        # Высокие убийства — ставим Under (будет меньше или равно)
        kills_side = "under"
        kills_threshold = int(round(total_kills))
    else:
        # Низкие убийства — ставим Over (будет больше)
        kills_side = "over"
        kills_threshold = int(round(total_kills)) + BET_THRESHOLD_OFFSET

    # =================================================================== #
    # 2. Potential duration
    # =================================================================== #
    dur_sec = _mean([h.get("avg_duration") for h in all_heroes], FALLBACK_DURATION_MIN * 60)
    dur_min = round(dur_sec / 60.0, 1)

    # --- Тотальная ставка: Over/Under ---
    # Преобразуем минуты в формат MM:SS для отображения
    pred_dur_min_int = int(round(dur_min))
    pred_dur_mmss = f"{pred_dur_min_int // 60}:{str(pred_dur_min_int % 60).zfill(2)}"

    # Логика: если матчи обычно длинные (>40 мин), делаем ставку "Тотал больше X минут"
    # Если короткое время — "Тотал меньше X+1 минут"
    if dur_min >= DURATION_OVER_UNDER_THRESHOLD:
        # Длинные матчи — ставим Under (матч не будет дольше предсказанного)
        total_side = "under"
        total_threshold = pred_dur_min_int
    else:
        # Короткие матчи — ставим Over
        total_side = "over"
        total_threshold = pred_dur_min_int + BET_THRESHOLD_OFFSET

    # =================================================================== #
    # 4. Potential towers destroyed
    # =================================================================== #
    dominance = abs(p_a - 0.5) * 2.0                            # 0..1
    win_tw = int(round(TOWER_BALANCED + TOWER_WIN_SLOPE * dominance))   # 5..11
    lose_tw = int(round(TOWER_BALANCED - TOWER_LOSE_SLOPE * dominance))  # 1..5
    if p_a >= 0.5:
        towers_a, towers_b = win_tw, lose_tw
    else:
        towers_a, towers_b = lose_tw, win_tw
    towers_total = towers_a + towers_b
    # --- Тотальная ставка: Over/Under (same shape as kills/duration) ---
    if towers_total >= TOWER_OVER_UNDER_THRESHOLD:
        towers_side = "under"
        towers_threshold = towers_total
    else:
        towers_side = "over"
        towers_threshold = towers_total + BET_THRESHOLD_OFFSET

    # =================================================================== #
    # 5. First to 15 kills (early aggression / lane strength proxy)
    # =================================================================== #
    early_a = EARLY_FB_WEIGHT * fb_a + EARLY_F10_WEIGHT * f10_a
    early_b = EARLY_FB_WEIGHT * fb_b + EARLY_F10_WEIGHT * f10_b
    zf = FIRST15_EARLY_SLOPE * (early_a - early_b) + FIRST15_WINNER_SLOPE * (p_a - 0.5)
    p_first_a = _logistic(zf)
    first15_team = team_a if p_first_a >= 0.5 else team_b
    first15_conf = max(p_first_a, 1 - p_first_a)

    # =================================================================== #
    # 6. Ultra-kill / Rampage potential
    # =================================================================== #
    fight_a = sum(1 for h in heroes_a if set(h.get("roles") or []) & FIGHT_ROLES)
    fight_b = sum(1 for h in heroes_b if set(h.get("roles") or []) & FIGHT_ROLES)
    fight_score = fight_a + fight_b + (
        MULTIKILL_KILLS_BONUS if total_kills > MULTIKILL_KILLS_THRESHOLD else 0
    )
    if fight_score >= MULTIKILL_HIGH_SCORE:
        multikill = "High"
    elif fight_score >= MULTIKILL_MEDIUM_SCORE:
        multikill = "Medium"
    else:
        multikill = "Low"
    if fight_a == fight_b:
        multikill_side = winner_team["name"]
    else:
        multikill_side = team_a["name"] if fight_a > fight_b else team_b["name"]

    def pct(x: float) -> int:
        return int(round(x * 100))

    return {
        "confidence": round(CONFIDENCE_BASE + CONFIDENCE_COMPLETENESS * completeness, 2),  # overall data confidence
        "winner": {
            "team": winner_team["name"],
            "probability": pct(winner_conf),
            "prob_radiant": pct(p_a),
        },
        "kills": {
            "total": int(round(total_kills)),
            "radiant": kills_a,
            "dire": kills_b,
        },
        "kills_total_over_under": {
            "side": kills_side,
            "threshold": kills_threshold,
        },
        "duration_min": dur_min,
        "total_over_under": {
            "side": total_side,
            "threshold": total_threshold,
            "formatted": pred_dur_mmss,
        },
        "towers": {
            "total": towers_total,
            "radiant": towers_a,
            "dire": towers_b,
        },
        "towers_over_under": {
            "side": towers_side,
            "threshold": towers_threshold,
        },
        "first_to_15": {
            "team": first15_team["name"],
            "probability": pct(first15_conf),
        },
        "multikill": {
            "level": multikill,          # Low / Medium / High
            "likely_side": multikill_side,
        },
    }


# --------------------------------------------------------------------------- #
# Helpers for post-match per-map analysis
# --------------------------------------------------------------------------- #

# Tower bitmask: 11 bits — one tower per bit. A SET bit = tower STILL STANDING
# at the end of the game (verified empirically: Dire 1792 = 0b11100000000 has
# 7 bits set and in-game Dire destroyed 7 of 11 Radiant towers).
# So destroyed = popcount(mask), NOT 11 - popcount.
_TOWER_TOTAL = 11


def decode_towers(mask) -> Optional[int]:
    """Return number of towers DESTROYED from a DLTV bitmask, or None if missing.

    Each of the 11 bits represents one tower on the enemy side; a set bit means
    that tower has been knocked down. `decode_towers(1792)` → 7 destroyed.
    """
    if not isinstance(mask, int):
        return None
    return min(_TOWER_TOTAL, bin(mask).count("1"))


def _side_team_name(actual_side: Optional[str], radiant_name: str, dire_name: str) -> Optional[str]:
    """Translate 'radiant' / 'dire' into the actual team name used by the UI."""
    if not actual_side:
        return None
    return radiant_name if actual_side == "radiant" else dire_name


def map_verdicts(
    prediction: Dict,
    actual: Dict,
    team_a_name: str,
    team_b_name: str,
) -> Dict:
    """Compare a per-map prediction against reality and return a per-metric verdict.

    Checks:
      - winner:          predicted winner must equal actual
      - duration_over_under: "over" bet wins if actual > threshold, "under" wins if actual <= threshold
      - kills_total_over_under: "over" bet wins if actual > threshold, "under" wins if actual <= threshold
      - towers_total:    predicted total towers must EQUAL actual
      - first_blood:     predicted side matches actual
      - first_to_15:     predicted side matches actual
    """
    def _eq(a, b) -> Optional[bool]:
        if a is None or b is None:
            return None
        return str(a).lower() == str(b).lower()

    def _exact(pred, act) -> Optional[bool]:
        if not isinstance(pred, (int, float)) or not isinstance(act, (int, float)):
            return None
        return pred == act

    # Winner
    pred_winner = (prediction.get("winner") or {}).get("team")
    actual_winner = actual.get("winner_team")
    v_winner = _eq(pred_winner, actual_winner)

    # Duration OVER/UNDER bet
    # Convert actual seconds to minutes for comparison
    actual_dur_min = actual.get("duration_min", 0)
    pred_total = prediction.get("total_over_under")
    if pred_total and isinstance(actual_dur_min, (int, float)):
        side = pred_total.get("side")
        threshold = pred_total.get("threshold")
        if isinstance(threshold, (int, float)):
            if side == "over":
                # Bet: матч будет больше N минут — победа если actual > threshold
                v_dur = actual_dur_min > threshold
            else:  # under
                # Bet: матч будет меньше или равен N минут — победа если actual <= threshold
                v_dur = actual_dur_min <= threshold
        else:
            v_dur = None
    else:
        v_dur = None

    # total kills
    pred_kills_total = prediction.get("kills_total_over_under")
    actual_kills = actual.get("kills_total", 0)
    if pred_kills_total and isinstance(actual_kills, (int, float)):
        side = pred_kills_total.get("side")
        threshold = pred_kills_total.get("threshold")
        if isinstance(threshold, (int, float)):
            if side == "over":
                # Bet: матч будет больше N киллов — победа если actual > threshold
                v_kills = actual_kills > threshold
            else:  # under
                # Bet: матч будет меньше или равен N киллов — победа если actual <= threshold
                v_kills = actual_kills <= threshold
        else:
            v_kills = None
    else:
        v_kills = None

    # total towers destroyed
    pred_tw = (prediction.get("towers") or {}).get("total")
    actual_tw = actual.get("towers_total")
    v_towers = _exact(pred_tw, actual_tw)

    # first blood — prediction: team with higher fb_rate; actual: recorded side
    pred_fb = (prediction.get("first_blood") or {}).get("team")
    actual_fb = _side_team_name(actual.get("fb_side"), team_a_name, team_b_name)
    v_fb = _eq(pred_fb, actual_fb)

    # first to 15 kills
    pred_f15 = (prediction.get("first_to_15") or {}).get("team")
    actual_f15 = _side_team_name(actual.get("f15_side"), team_a_name, team_b_name)
    v_f15 = _eq(pred_f15, actual_f15)

    return {
        "winner": v_winner,
        "duration": v_dur,
        "kills_total": v_kills,
        "towers_total": v_towers,
        "first_blood": v_fb,
        "first_to_15": v_f15,
    }


def analyze_map_with_verdict(
    team_a: Dict,
    team_b: Dict,
    heroes_a: List[Dict],
    heroes_b: List[Dict],
    actual: Dict,
    engine=None,
) -> Dict:
    """Run analyze() for one finished map and attach verdicts vs. actual outcome.

    `actual` is a dict with the map's recorded facts:
      - winner_team (str, the team name that won)
      - duration_min (float)
      - kills_total (int)
      - towers_total (int or None)
      - fb_side ("radiant" | "dire" | None)
      - f15_side ("radiant" | "dire" | None)

    `engine` is an optional `IPredictionEngine` (any object exposing
    `.analyze(team_a, team_b, heroes_a, heroes_b)`). When None, the
    pure heuristic is used (preserves the v0.0.x behaviour). Pass an
    `MLEngine` here to evaluate the model against historical matches.
    """
    if engine is None:
        pred = analyze(team_a, team_b, heroes_a, heroes_b)
    else:
        pred = engine.analyze(team_a, team_b, heroes_a, heroes_b)
    # Augment prediction with first-blood pick (analyze() didn't expose it)
    fb_a = _val(team_a, "fb_rate", FALLBACK_FB)
    fb_b = _val(team_b, "fb_rate", FALLBACK_FB)
    pred["first_blood"] = {
        "team": team_a["name"] if fb_a >= fb_b else team_b["name"],
        "probability": int(round(max(fb_a, fb_b))),
    }
    verdict = map_verdicts(pred, actual, team_a["name"], team_b["name"])
    return {"prediction": pred, "verdict": verdict}
