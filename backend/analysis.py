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
    z = 0.045 * (wr_a - wr_b)
    z += 0.020 * (draft_a - draft_b)
    if isinstance(rank_a, int) and isinstance(rank_b, int) and rank_a > 0 and rank_b > 0:
        z += 0.015 * _clamp(rank_b - rank_a, -40, 40)
    p_a = _logistic(z)
    winner_team = team_a if p_a >= 0.5 else team_b
    winner_conf = max(p_a, 1 - p_a)

    # =================================================================== #
    # 3. Total kills
    # =================================================================== #
    mean_kda = _mean([h.get("kda") for h in all_heroes], BASE_KDA)
    total_kills = _clamp(BASE_TOTAL_KILLS + (mean_kda - BASE_KDA) * 6.0, 30, 78)
    share_a = 0.35 + 0.30 * p_a               # favored side scores more
    kills_a = round(total_kills * share_a)
    kills_b = round(total_kills * (1 - share_a))
    
    # --- Тотальная ставка: Over/Under ---
    if total_kills >= 50:
        # Высокие убийства — ставим Under (будет меньше или равно)
        kills_side = "under"
        kills_threshold = int(round(total_kills)) + 0.5
    else:
        # Низкие убийства — ставим Over (будет больше)
        kills_side = "over"
        kills_threshold = int(round(total_kills)) - 0.5
    
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
    if dur_min >= 40:
        # Длинные матчи — ставим Under (матч не будет дольше предсказанного)
        total_side = "under"
        total_threshold = pred_dur_min_int
    else:
        # Короткие матчи — ставим Over
        total_side = "over"
        total_threshold = pred_dur_min_int + 1

    # =================================================================== #
    # 4. Potential towers destroyed
    # =================================================================== #
    dominance = abs(p_a - 0.5) * 2.0                     # 0..1
    win_tw = int(round(5 + 6 * dominance))               # 5..11
    lose_tw = int(round(5 - 4 * dominance))              # 1..5
    if p_a >= 0.5:
        towers_a, towers_b = win_tw, lose_tw
    else:
        towers_a, towers_b = lose_tw, win_tw

    # =================================================================== #
    # 5. First to 15 kills (early aggression / lane strength proxy)
    # =================================================================== #
    early_a = 0.4 * fb_a + 0.6 * f10_a
    early_b = 0.4 * fb_b + 0.6 * f10_b
    zf = 0.05 * (early_a - early_b) + 0.4 * (p_a - 0.5)
    p_first_a = _logistic(zf)
    first15_team = team_a if p_first_a >= 0.5 else team_b
    first15_conf = max(p_first_a, 1 - p_first_a)

    # =================================================================== #
    # 6. Ultra-kill / Rampage potential
    # =================================================================== #
    fight_a = sum(1 for h in heroes_a if set(h.get("roles") or []) & FIGHT_ROLES)
    fight_b = sum(1 for h in heroes_b if set(h.get("roles") or []) & FIGHT_ROLES)
    fight_score = fight_a + fight_b + (2 if total_kills > 56 else 0)
    if fight_score >= 7:
        multikill = "High"
    elif fight_score >= 4:
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
        "confidence": round(0.4 + 0.6 * completeness, 2),  # overall data confidence
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
            "total": towers_a + towers_b,
            "radiant": towers_a,
            "dire": towers_b,
        },
        "first_to_15": {
            "team": first15_team["name"],
            "probability": pct(first15_conf),
        },
        "first_blood": {
            "team": team_a["name"] if fb_a >= fb_b else team_b["name"],
            "probability": pct(max(fb_a, fb_b) / 100.0),
        },
        "first_tower": {
            "team": team_a["name"] if p_a >= 0.5 else team_b["name"],
            "probability": pct(max(p_a, 1.0 - p_a)),
        },
        "multikill": {
            "level": multikill,          # Low / Medium / High
            "likely_side": multikill_side,
        },
    }


def analyze_prematch_series(team_a: Dict, team_b: Dict, bo: str) -> Dict:
    """Produce pre-draft markets using only team form/rank and series format.

    This intentionally avoids hero, lane and live-game inputs: it is suitable
    for cards shown before a draft exists.
    """
    wr_a, wr_b = _val(team_a, "win_rate", FALLBACK_WR), _val(team_b, "win_rate", FALLBACK_WR)
    rank_a, rank_b = team_a.get("rank"), team_b.get("rank")
    z = 0.045 * (wr_a - wr_b)
    if isinstance(rank_a, int) and isinstance(rank_b, int) and rank_a > 0 and rank_b > 0:
        z += 0.015 * _clamp(rank_b - rank_a, -40, 40)
    probability_a = _logistic(z)
    favourite = team_a if probability_a >= 0.5 else team_b
    confidence = int(round(max(probability_a, 1.0 - probability_a) * 100))
    bo_number = int(bo[2:]) if isinstance(bo, str) and bo.lower().startswith("bo") and bo[2:].isdigit() else 3
    close_series = abs(probability_a - 0.5) < 0.12
    if bo_number == 2:
        score, total = "1:1", 2
    elif bo_number == 5:
        score, total = ("3:2", 5) if close_series else ("3:1", 4)
    else:
        score, total = ("2:1", 3) if close_series else ("2:0", 2)
    return {
        "winner": {"team": favourite["name"], "probability": confidence},
        "first_map": {"team": favourite["name"], "probability": confidence},
        "series_score": {"favourite": favourite["name"], "score": score},
        "total_maps": {
            "side": "exact" if bo_number == 2 else ("over" if total == bo_number else "under"),
            "threshold": 2 if bo_number == 2 else bo_number - 0.5,
        },
        "first_blood": {"team": team_a["name"] if _val(team_a, "fb_rate", FALLBACK_FB) >= _val(team_b, "fb_rate", FALLBACK_FB) else team_b["name"]},
        "first_tower": {"team": favourite["name"]},
        "confidence": confidence,
        "source": "prematch_team_form",
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
    if pred_total:
        side = pred_total.get("side")
        threshold = pred_total.get("threshold")
        if side == "over":
            # Bet: матч будет больше N минут — победа если actual > threshold
            v_dur = actual_dur_min > threshold
        else:  # under
            # Bet: матч будет меньше или равен N минут — победа если actual <= threshold
            v_dur = actual_dur_min <= threshold
    else:
        v_dur = None

    # total kills
    pred_kills_total = prediction.get("kills_total_over_under")
    actual_kills = actual.get("kills_total", 0)
    if pred_kills_total:
        side = pred_kills_total.get("side")
        threshold = pred_kills_total.get("threshold")
        if side == "over":
            # Bet: матч будет больше N киллов — победа если actual > threshold
            v_kills = actual_kills > threshold
        else:  # under
            # Bet: матч будет меньше или равен N киллов — победа если actual <= threshold
            v_kills = actual_kills <= threshold
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
) -> Dict:
    """Run analyze() for one finished map and attach verdicts vs. actual outcome.

    `actual` is a dict with the map's recorded facts:
      - winner_team (str, the team name that won)
      - duration_min (float)
      - kills_total (int)
      - towers_total (int or None)
      - fb_side ("radiant" | "dire" | None)
      - f15_side ("radiant" | "dire" | None)
    """
    pred = analyze(team_a, team_b, heroes_a, heroes_b)
    # Augment prediction with first-blood pick (analyze() didn't expose it)
    fb_a = _val(team_a, "fb_rate", FALLBACK_FB)
    fb_b = _val(team_b, "fb_rate", FALLBACK_FB)
    pred["first_blood"] = {
        "team": team_a["name"] if fb_a >= fb_b else team_b["name"],
        "probability": int(round(max(fb_a, fb_b))),
    }
    verdict = map_verdicts(pred, actual, team_a["name"], team_b["name"])
    return {"prediction": pred, "verdict": verdict}
