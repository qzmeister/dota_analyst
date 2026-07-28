"""Normalize OpenDota match blobs into the v17 training schema.

Reads `ml_data/imports/v17_match_<id>.json` (raw OpenDota /api/matches
payloads) and writes `ml_data/full_matches/<id>.json` in the
extended v17 schema:

    {
      # Existing v0.3.11 schema (kept for backward compat with v16 trainer)
      "match_id": 8902338515,
      "patch": "7.40",            # str, normalised
      "duration": 1869,
      "radiant_win": true,
      "radiant_team_id": 9824702,
      "dire_team_id":  9823272,
      "start_time": 1784382596,
      "leagueid": 19785,
      "league_tier": "premium",
      "series_id": 1121633,
      "series_type": 1,
      "players": [
        # 10 players, slots 0-4 radiant, 128-132 dire
        {
          "account_id": 1044002267,
          "name": "Satanic",
          "hero_id": 19,
          "player_slot": 0,
          "isRadiant": true,
          "kills": 10, "deaths": 2, "assists": 9,
          "kda": 6.33,
          "gold_per_min": 765,
          "xp_per_min": 950,
          "last_hits": 378, "denies": 8,
          "net_worth": 23660,
          "lane": 1, "lane_role": 1,
          "lane_efficiency_pct": 81,
          "obs_placed": 0, "sen_placed": 1,
          "hero_damage": 19020,
          "tower_damage": 9215,
          "hero_healing": 0,
          "stuns": 20.36,
          "level": 23,
          "rank_tier": 80,
          "rune_pickups": 1, "creeps_stacked": 8, "camps_stacked": 3,
          "ancient_kills": 14, "neutral_kills": 175, "roshan_kills": 1,
          "purchase_tpscroll": 4,
          "teamfight_participation": 0.67,
          "ability_upgrades_arr": [5108, 5106, 5108, ...],   # ability ids
          "items": [63, 249, 939, 117, 1, 112],            # final items
          "item_neutral": 1605,
          "gold_t": [0, 335, 664, ...],                    # 32-point time-series
          "xp_t":   [0,  90, 418, ...],
          "lh_t":   [0,   3,   9, ...],
          "dn_t":   [0,   0,   1, ...],
          "lane_pos": {"76": ..., "77": ...},               # lane position time-series
        },
        ...
      ],
      "picks_bans": [
        # 24 entries: alternating ban/pick per team
        {"is_pick": false, "hero_id": 145, "team": 1, "order": 0},
        ...
      ],
      "objectives": [
        # towers, racks, Roshan
        {"time": 1492, "type": "building_kill", "key": "npc_dota_tower", ...},
        ...
      ],
      "teamfights": [
        {"start": 600, "end": 720, "last_death": 720, "deaths": 5,
         "players": [...]},
        ...
      ],
      "radiant_gold_adv": [-373, -100, ...],  # 32-point time-series
      "radiant_xp_adv":   [-99,  -30, ...],

      # NEW v17 targets
      "v17_targets": {
        "kills_total": 37,            # radiant_score + dire_score
        "duration_sec": 1869,
        "first_15_kills": 7,           # kills before game time 900
        "winner": "radiant"           # "radiant" or "dire"
      },

      # NEW v17 features
      "v17_features": {
        "patch": "7.40",
        "is_top_team_radiant": 1,     # 0/1 flags
        "is_top_team_dire": 0,
        "team_tier": "premium",        # league.tier
        "recency_weight": 0.95,       # 0..1; 1.0 for matches <7d old
      },

      "v17_outliers": {              # sanity-check flags
        "duration_ok": true,          # 300 < duration < 7200
        "kills_ok": true,             # 0 < kills_total < 100
      }
    }

We do NOT drop existing v0.3.11 fields — backwards compat means the
v16 trainer can still load these files (it ignores unknown keys).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"
FULL_MATCHES = PRO_ROOT / "ml_data" / "full_matches"

# OpenDota `patch` is an int — 22 = 7.33d, 60 = 7.40 etc.  We
# build a "major.minor" string for the trainer to use.  If the
# int isn't in the lookup, fall back to a hex-y "patch_<int>" so
# downstream code never breaks on a future patch.
_PATCH_LOOKUP = {
    22: "7.33d", 23: "7.34", 24: "7.35",  # historical examples
    # Real OpenDota patch values are usually sequential integers
    # mapping to version buckets; we rebuild from /api/constants/patch
    # when possible, but the static map below covers the
    # common case where the imports file isn't there.
}
# Patch id -> "X.YY" — filled at import time from the latest
# v17_phase7_patch_info.json (see `init_patch_map`).
_PATCH_MAP: Dict[int, str] = dict(_PATCH_LOOKUP)


def init_patch_map() -> None:
    """Read the OpenDota patch constants file (if present) and
    populate _PATCH_MAP.  The OpenDota payload gives a list of
    `{patch, name}` (e.g. name = "7.40") which lets us reverse-map
    a patch int to a human-readable version string.
    """
    p = IMPORTS / "v17_phase7_patch_info.json"
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
    except Exception:
        return
    if not isinstance(data, list):
        return
    for entry in data:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("patch_id") or entry.get("id") or entry.get("patch")
        name = entry.get("name") or entry.get("patch_name")
        if isinstance(pid, int) and isinstance(name, str) and name:
            _PATCH_MAP[pid] = name


def _patch_str(patch_int: Optional[int]) -> str:
    if patch_int is None:
        return ""
    if patch_int in _PATCH_MAP:
        return _PATCH_MAP[patch_int]
    # Fallback: 60 = "patch_60" so the trainer never crashes.
    return f"patch_{patch_int}"


# Per-player slot layout: 0-4 = radiant, 128-132 = dire.
def _side_from_slot(slot: int) -> str:
    return "radiant" if slot < 128 else "dire"


# v0.3.11 had `picks_bans` ordered as bans-first or picks-first
# depending on game mode; OpenDota uses `order` (0..23) so we keep
# the original array and let the trainer sort by order when it
# needs to split by team / pick.
def _extract_picks_bans(picks_bans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for pb in picks_bans or []:
        if not isinstance(pb, dict):
            continue
        out.append({
            "is_pick": bool(pb.get("is_pick")),
            "hero_id": pb.get("hero_id"),
            "team": pb.get("team"),  # 0 = radiant, 1 = dire
            "order": pb.get("order"),
        })
    return out


def _build_player_entry(p: Dict[str, Any]) -> Dict[str, Any]:
    """Compress an OpenDota player blob into the v17 schema."""
    out: Dict[str, Any] = {
        "account_id":    p.get("account_id"),
        "name":          p.get("name") or p.get("personaname"),
        "hero_id":       p.get("hero_id"),
        "player_slot":   p.get("player_slot"),
        "isRadiant":     p.get("isRadiant"),
        "kills":         p.get("kills"),
        "deaths":        p.get("deaths"),
        "assists":       p.get("assists"),
        "kda":           p.get("kda"),
        "gold_per_min":  p.get("gold_per_min"),
        "xp_per_min":    p.get("xp_per_min"),
        "last_hits":     p.get("last_hits"),
        "denies":        p.get("denies"),
        "net_worth":     p.get("net_worth"),
        "lane":          p.get("lane"),
        "lane_role":     p.get("lane_role"),
        "lane_efficiency_pct": p.get("lane_efficiency_pct"),
        "obs_placed":    p.get("obs_placed"),
        "sen_placed":    p.get("sen_placed"),
        "hero_damage":   p.get("hero_damage"),
        "tower_damage":  p.get("tower_damage"),
        "hero_healing":  p.get("hero_healing"),
        "stuns":         p.get("stuns"),
        "level":         p.get("level"),
        "rank_tier":     p.get("rank_tier"),
        "rune_pickups":  p.get("rune_pickups"),
        "creeps_stacked":p.get("creeps_stacked"),
        "camps_stacked": p.get("camps_stacked"),
        "ancient_kills": p.get("ancient_kills"),
        "neutral_kills": p.get("neutral_kills"),
        "roshan_kills":  p.get("roshan_kills"),
        "purchase_tpscroll": p.get("purchase_tpscroll"),
        "purchase_ward_sentry": p.get("purchase_ward_sentry"),
        "teamfight_participation": p.get("teamfight_participation"),
        "ability_upgrades_arr": list(p.get("ability_upgrades_arr") or []),
        "items": [
            p.get("item_0"), p.get("item_1"), p.get("item_2"),
            p.get("item_3"), p.get("item_4"), p.get("item_5"),
        ],
        "item_neutral": p.get("item_neutral"),
        "item_neutral2": p.get("item_neutral2"),
        "gold_t":  list(p.get("gold_t")  or []),
        "xp_t":    list(p.get("xp_t")    or []),
        "lh_t":    list(p.get("lh_t")    or []),
        "dn_t":    list(p.get("dn_t")    or []),
        "lane_pos": dict(p.get("lane_pos") or {}),
    }
    return out


def _first_n_kills(players: List[Dict[str, Any]], n_sec: int) -> int:
    """Count hero kills across both teams up to `n_sec` game time.

    Uses the per-player `kills_log` field which is `[{time, key}, ...]`.
    A hero kill has `key` starting with `npc_dota_hero_`; a
    neutral-creep kill starts with `npc_dota_creep_` or
    `npc_dota_neutral_` (which we filter out).  `time` is in
    seconds since match start (negative = pre-game, ignore).
    """
    count = 0
    for p in players:
        for k in p.get("kills_log") or []:
            if not isinstance(k, dict):
                continue
            t = k.get("time")
            if not isinstance(t, (int, float)) or t < 0 or t > n_sec:
                continue
            key = str(k.get("key") or "")
            if not key.startswith("npc_dota_hero_"):
                continue
            count += 1
    return count


def _recency_weight(start_time: int, now_ts: Optional[int] = None) -> float:
    """Exponential decay weight: 1.0 for matches <7d old, ~0.5 at 30d,
    ~0.1 at 90d.  Used by the trainer to weight recent samples more.
    """
    if start_time <= 0:
        return 0.5
    now = now_ts or int(time.time())
    days = max(0, (now - start_time) / 86400.0)
    return 0.5 ** (days / 30.0)  # half-life 30 days


def _outlier_flags(duration: int, kills: int) -> Dict[str, bool]:
    """Sanity-check the target values; the trainer can drop or
    winsorize on these flags.
    """
    return {
        "duration_ok": 300 <= duration <= 7200,    # 5min-2h
        "kills_ok":    0 <= kills <= 100,
    }


def normalize_one(raw: Dict[str, Any],
                  top_team_ids: Optional[set] = None) -> Dict[str, Any]:
    """Take an OpenDota match payload, return a normalised v17 dict."""
    mid = raw.get("match_id")
    if mid is None:
        raise ValueError("raw match has no match_id")
    duration = int(raw.get("duration") or 0)
    radiant_score = int(raw.get("radiant_score") or 0)
    dire_score    = int(raw.get("dire_score")    or 0)
    kills_total   = radiant_score + dire_score

    players_raw = raw.get("players") or []
    players = [_build_player_entry(p) for p in players_raw if isinstance(p, dict)]

    first_15 = _first_n_kills(players_raw, 900)  # 15 min = 900 sec

    winner = "radiant" if raw.get("radiant_win") else "dire"

    patch = _patch_str(raw.get("patch"))

    league = raw.get("league") or {}
    league_tier = league.get("tier") or "unknown"

    r_team = raw.get("radiant_team_id")
    d_team = raw.get("dire_team_id")

    targets = {
        "kills_total": kills_total,
        "duration_sec": duration,
        "first_15_kills": first_15,
        "winner": winner,
    }

    if top_team_ids is not None:
        r_top = int(r_team) in top_team_ids if r_team else False
        d_top = int(d_team) in top_team_ids if d_team else False
    else:
        r_top = d_top = False

    v17_features = {
        "patch": patch,
        "is_top_team_radiant": int(r_top),
        "is_top_team_dire":    int(d_top),
        "team_tier":            league_tier,
        "recency_weight":       _recency_weight(int(raw.get("start_time") or 0)),
    }

    return {
        "match_id":         int(mid),
        "patch":            patch,
        "duration":         duration,
        "radiant_win":      bool(raw.get("radiant_win")),
        "radiant_team_id":  r_team,
        "dire_team_id":     d_team,
        "start_time":       int(raw.get("start_time") or 0),
        "leagueid":         raw.get("leagueid"),
        "league_tier":      league_tier,
        "series_id":        raw.get("series_id"),
        "series_type":      raw.get("series_type"),
        "radiant_score":    radiant_score,
        "dire_score":       dire_score,
        "players":          players,
        "picks_bans":       _extract_picks_bans(raw.get("picks_bans") or []),
        "objectives":       list(raw.get("objectives") or []),
        "teamfights":       list(raw.get("teamfights") or []),
        "radiant_gold_adv": list(raw.get("radiant_gold_adv") or []),
        "radiant_xp_adv":   list(raw.get("radiant_xp_adv")   or []),
        "v17_targets":      targets,
        "v17_features":     v17_features,
        "v17_outliers":     _outlier_flags(duration, kills_total),
    }


def main(argv: List[str]) -> int:
    init_patch_map()
    if not IMPORTS.exists():
        print(f"missing {IMPORTS}", file=sys.stderr)
        return 2
    FULL_MATCHES.mkdir(parents=True, exist_ok=True)
    # top-team filter (optional)
    top_ids: Optional[set] = None
    p = IMPORTS / "v17_phase1_top_teams.json"
    if p.exists() and "--all" not in argv:
        try:
            teams = json.loads(p.read_text())
            top_ids = {int(t["team_id"]) for t in teams if t.get("team_id") is not None}
        except Exception:
            top_ids = None
    files = sorted(IMPORTS.glob("v17_match_*.json"))
    written = 0
    skipped = 0
    for i, src in enumerate(files, 1):
        try:
            raw = json.loads(src.read_text())
            if not isinstance(raw, dict):
                continue
            norm = normalize_one(raw, top_team_ids=top_ids)
            target = FULL_MATCHES / f"{norm['match_id']}.json"
            with open(target, "w", encoding="utf-8") as f:
                json.dump(norm, f, ensure_ascii=False)
            written += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"  skip {src.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if i % 50 == 0:
            print(f"  normalised {i} of {len(files)}", file=sys.stderr)
    print(f"[normalizer] wrote {written} normalised matches, {skipped} skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
