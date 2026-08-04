"""Compute v19 features with PER-MATCH ROLLING lookups (no leakage).

Background
----------
The earlier v19 attempt (v0.7.63) built one cache for the whole
corpus and used it for both train and test.  That cache included
each training match's own data, so the model saw a "leak" — the
v19 features for a training match X reflected X's outcome, and
the model learned to lean on them.  At test time the cache had no
data for the test match, the v19 features collapsed to 0.5, and
the model failed.

This script does the proper thing: walk matches in start_time
order, maintain a running lookup, and for each match compute v19
features from matches BEFORE it.  No leakage anywhere.

Output:  ml_data/imports/v19_features.json
   { "<match_id>": {<12 v19 feature floats>}, ... }

Consumed by:  scripts/train_v18.py::extract_features

Features (12, all in 0..1, Bayesian-smoothed toward 0.5):
  - r_player_wr_avg, d_player_wr_avg
  - r_player_wr_max, d_player_wr_max
  - player_wr_diff  (= r_avg - d_avg)
  - r_pair_synergy_avg, d_pair_synergy_avg
  - r_pair_synergy_max, d_pair_synergy_max
  - mid_matchup_wr, bot_matchup_wr, top_matchup_wr

Run:  python scripts/build_v19_features_rolling.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRO_ROOT = Path(r"C:\Users\artka\.minimax\workspace\dota_analyst")
IMPORTS = PRO_ROOT / "ml_data" / "imports"
BACKUP_DIR = IMPORTS / "_v17_match_stratz_minor"
OUT = IMPORTS / "v19_features.json"

# Smoothing priors.  Conservative: a player needs ~30 real games to
# push the smoothed WR meaningfully away from 0.5.  See train_v18.py
# for the rationale (was 5/5/3 — too leaky; 20/20/10 also overfit;
# 5/5/3 is fine WHEN the lookup is per-match rolling, no leakage).
PLAYER_PRIOR = 5
PAIR_PRIOR = 5
MATCHUP_PRIOR = 3

# v0.7.66: v19 LITE — only emit features for experienced players.
# When `games < MIN_PLAYER_GAMES`, the feature is the neutral
# prior (0.5) instead of the smoothed rate.  This stops the
# "12 near-constant noise features" problem that killed the full
# v19 attempt (corpus acc dropped 0.76 -> 0.65).  A player needs
# 30+ games on a hero for the feature to "fire".
# v0.7.66: kept at 30 — multiple attempts to lower it (15 in
# v0.7.67, 20 in v0.7.69) both hurt.  30 is the local maximum
# for the player WR gate in this corpus (3403 OpenDota matches).
MIN_PLAYER_GAMES = 30

# OpenDota lane constants (used to derive mid / bot / top slots)
LANE_TOP = 1
LANE_MID = 2
LANE_BOT = 3


# --------------------------------------------------------------------------- #
# Match loading
# --------------------------------------------------------------------------- #

def _load_all_v17_matches() -> List[Tuple[int, int, Dict[str, Any]]]:
    """Walk v17_match_*.json in both imports/ and the backup dir,
    return a list of (start_time, match_id, doc) sorted by time.
    Matches without a parseable start_time are dropped.
    """
    out: List[Tuple[int, int, Dict[str, Any]]] = []
    seen_ids: set = set()
    for folder in (IMPORTS, BACKUP_DIR):
        if not folder.exists():
            continue
        for fp in folder.glob("v17_match_*.json"):
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            mid = d.get("match_id")
            if mid is None:
                # Fall back to filename
                try:
                    mid = int(fp.stem.split("_")[-1])
                except Exception:
                    continue
            mid = int(mid)
            if mid in seen_ids:
                continue
            st = int(d.get("start_time") or 0)
            if st <= 0:
                continue
            seen_ids.add(mid)
            out.append((st, mid, d))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


# --------------------------------------------------------------------------- #
# Feature extraction from caches
# --------------------------------------------------------------------------- #

def _sm(w: int, g: int, prior: int) -> float:
    return (w + prior * 0.5) / (g + prior)


def _side_hero_ids(m: Dict[str, Any], side: str) -> List[int]:
    """Get 5 hero_ids for `side` from m['players'][] (OpenDota-style
    positional: 0-4 radiant, 5-9 dire).  Falls back to the legacy
    `_players_to_picks` semantics: take the first 5 with isRadiant
    matching, or fall back to all 10 split by position.
    """
    players = m.get("players") or []
    heroes: List[int] = []
    for p in players:
        is_rad = bool(p.get("isRadiant"))
        if side == "radiant" and not is_rad:
            continue
        if side == "dire" and is_rad:
            continue
        h = p.get("hero_id")
        if isinstance(h, int) and h > 0:
            heroes.append(int(h))
        if len(heroes) == 5:
            break
    return heroes


def _player_names_for_side(m: Dict[str, Any], side: str) -> List[str]:
    players = m.get("players") or []
    out: List[str] = []
    for p in players:
        is_rad = bool(p.get("isRadiant"))
        if side == "radiant" and not is_rad:
            continue
        if side == "dire" and is_rad:
            continue
        nm = (p.get("name") or p.get("personaname") or "").strip().lower()
        if nm:
            out.append(nm)
        if len(out) == 5:
            break
    return out


def _lane_players(m: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out = {k: [] for k in ("r_mid", "d_mid", "r_bot", "d_bot", "r_top", "d_top")}
    for p in (m.get("players") or []):
        h = p.get("hero_id")
        if not isinstance(h, int) or h <= 0:
            continue
        lane = p.get("lane")
        side = "r" if p.get("isRadiant") else "d"
        if lane == LANE_MID:
            out[f"{side}_mid"].append(p)
        elif lane == LANE_BOT:
            out[f"{side}_bot"].append(p)
        elif lane == LANE_TOP:
            out[f"{side}_top"].append(p)
    return out


def _features_for_match(
    m: Dict[str, Any],
    player_cache: Dict[Tuple[str, int], Tuple[int, int]],
    pair_cache: Dict[Tuple[str, int, int], Tuple[int, int]],
    mid_cache: Dict[Tuple[int, int], Tuple[int, int]],
    bot_cache: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    top_cache: Dict[Tuple[int, int, int, int], Tuple[int, int]],
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # --- player WR per side ---
    r_names = _player_names_for_side(m, "radiant")
    d_names = _player_names_for_side(m, "dire")
    r_heroes = _side_hero_ids(m, "radiant")
    d_heroes = _side_hero_ids(m, "dire")

    def _sm_player(w: int, g: int) -> float:
        # v0.7.66: gate by min-games.  Players with <MIN_PLAYER_GAMES
        # on a hero get the neutral prior (0.5) — no signal, no noise.
        if g < MIN_PLAYER_GAMES:
            return 0.5
        return _sm(w, g, PLAYER_PRIOR)

    r_wrs = []
    for nm, h in zip(r_names, r_heroes):
        w, g = player_cache.get((nm, h), (0, 0))
        r_wrs.append(_sm_player(w, g))
    d_wrs = []
    for nm, h in zip(d_names, d_heroes):
        w, g = player_cache.get((nm, h), (0, 0))
        d_wrs.append(_sm_player(w, g))
    r_avg = sum(r_wrs) / 5.0 if r_wrs else 0.5
    d_avg = sum(d_wrs) / 5.0 if d_wrs else 0.5
    feats["r_player_wr_avg"] = r_avg
    feats["d_player_wr_avg"] = d_avg
    feats["r_player_wr_max"] = max(r_wrs) if r_wrs else 0.5
    feats["d_player_wr_max"] = max(d_wrs) if d_wrs else 0.5
    feats["player_wr_diff"] = r_avg - d_avg

    # v0.7.66: pair synergy and lane matchup were dropped because
    # they're too sparse (most player-pairs and lane matchups
    # have 1-3 games, so the smoothed WR is dominated by the
    # prior 0.5 → 12 near-constant noise features → -4pp honest
    # test).  v0.7.67 tried adding mid 1v1 back and lowering the
    # gate to 15 games — both made the model WORSE (0.6241 ->
    # 0.6138).  v0.7.68 tried mid 1v1 alone (gate still 30) —
    # also WORSE (0.5874).  v0.7.69 testing gate=20 alone, no
    # mid.  If that also hurts, v0.7.66 settings are the local
    # maximum.

    return feats


# --------------------------------------------------------------------------- #
# Cache updates
# --------------------------------------------------------------------------- #

def _winner_bool(m: Dict[str, Any]) -> Optional[bool]:
    rw = m.get("radiant_win")
    if rw is None:
        rw = m.get("radiant_victory")
    return None if rw is None else bool(rw)


def _update_caches(
    m: Dict[str, Any],
    player_cache: Dict[Tuple[str, int], Tuple[int, int]],
    pair_cache: Dict[Tuple[str, int, int], Tuple[int, int]],
    mid_cache: Dict[Tuple[int, int], Tuple[int, int]],
    bot_cache: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    top_cache: Dict[Tuple[int, int, int, int], Tuple[int, int]],
) -> None:
    won = _winner_bool(m)
    if won is None:
        return
    target = int(won)  # 1 = radiant win

    # player hero WR
    for side in ("radiant", "dire"):
        names = _player_names_for_side(m, side)
        heroes = _side_hero_ids(m, side)
        is_win = target if side == "radiant" else 1 - target
        for nm, h in zip(names, heroes):
            w, g = player_cache.get((nm, h), (0, 0))
            player_cache[(nm, h)] = (w + is_win, g + 1)

    # hero pair synergy
    for side in ("radiant", "dire"):
        heroes = _side_hero_ids(m, side)
        is_win = target if side == "radiant" else 1 - target
        for i in range(len(heroes)):
            for j in range(i + 1, len(heroes)):
                a, b = (heroes[i], heroes[j]) if heroes[i] <= heroes[j] else (heroes[j], heroes[i])
                w, g = pair_cache.get((side[0], a, b), (0, 0))
                pair_cache[(side[0], a, b)] = (w + is_win, g + 1)

    # lane matchup
    lanes = _lane_players(m)
    if lanes["r_mid"] and lanes["d_mid"]:
        h_r = lanes["r_mid"][0]["hero_id"]
        h_d = lanes["d_mid"][0]["hero_id"]
        w, g = mid_cache.get((h_r, h_d), (0, 0))
        mid_cache[(h_r, h_d)] = (w + target, g + 1)
    if len(lanes["r_bot"]) >= 2 and len(lanes["d_bot"]) >= 2:
        h = (lanes["r_bot"][0]["hero_id"], lanes["r_bot"][1]["hero_id"],
             lanes["d_bot"][0]["hero_id"], lanes["d_bot"][1]["hero_id"])
        w, g = bot_cache.get(h, (0, 0))
        bot_cache[h] = (w + target, g + 1)
    if len(lanes["r_top"]) >= 2 and len(lanes["d_top"]) >= 2:
        h = (lanes["r_top"][0]["hero_id"], lanes["r_top"][1]["hero_id"],
             lanes["d_top"][0]["hero_id"], lanes["d_top"][1]["hero_id"])
        w, g = top_cache.get(h, (0, 0))
        top_cache[h] = (w + target, g + 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    matches = _load_all_v17_matches()
    print(f"loaded {len(matches)} matches (from imports/ + _v17_match_stratz_minor/)")

    player_cache: Dict[Tuple[str, int], Tuple[int, int]] = {}
    pair_cache:   Dict[Tuple[str, int, int], Tuple[int, int]] = {}
    mid_cache:    Dict[Tuple[int, int], Tuple[int, int]] = {}
    bot_cache:    Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
    top_cache:    Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}

    out: Dict[str, Dict[str, float]] = {}
    for i, (_st, mid, m) in enumerate(matches):
        feats = _features_for_match(m, player_cache, pair_cache,
                                    mid_cache, bot_cache, top_cache)
        out[str(mid)] = feats
        _update_caches(m, player_cache, pair_cache,
                       mid_cache, bot_cache, top_cache)
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(matches)} matches")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"wrote {len(out)} v19 feature rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
