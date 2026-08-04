"""Normalize Stratz-format match files in ml_data/full_matches/ to the
canonical OpenDota-compatible schema the v18 trainer expects.

Background
----------
The corpus has two schemas side by side:

  A) OpenDota  (3403 files, "good" corpus)
       - top-level: radiant_win, radiant_team_id, dire_team_id,
         duration, patch, start_time (unix sec), players[10],
         picks_bans[], v17_features, league_tier
       - this is what scripts/train_v18.py consumes directly

  B) Stratz  (1824 files, "raw DLTV/Stratz dump")
       - top-level: radiant_victory (not radiant_win),
         radiant{team, player_performances[]}, dire{team, player_performances[]},
         start_date (unix MILLI), patch, duration, league{league_id, name},
         frames{times, radiant_networth_advantage}
       - has team_id (radiant.team.valve_id), winner (radiant_victory),
         per-player hero/kda/gpm/xpm/items — but in a different shape
       - missing: players[] (OpenDota shape), picks_bans, v17_features,
         league_tier

This script converts (B) → (A) in place.  It's idempotent: files that
already have a `players` list of length >= 10 are skipped.

Output: every Stratz file in ml_data/full_matches/ gets the canonical
keys added.  Original Stratz keys are preserved (so we don't lose
player_performances / frames / derived_series which might be useful
later).

Run:  python scripts/normalize_full_matches.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CORPUS_GLOB = r"C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\full_matches\*.json"


def _is_already_canonical(d: Dict[str, Any]) -> bool:
    """A file is canonical if it already has a `players` list of length >= 10
    and a top-level `radiant_win`.  We don't check every field, just the
    ones the v18 trainer requires."""
    players = d.get("players")
    return (
        isinstance(players, list)
        and len(players) >= 10
        and "radiant_win" in d
    )


def _is_stratz_format(d: Dict[str, Any]) -> bool:
    return (
        "radiant_victory" in d
        and isinstance(d.get("radiant"), dict)
        and isinstance(d.get("dire"), dict)
        and "player_performances" in d.get("radiant", {})
    )


def _build_player(slot: int, perf: Dict[str, Any], is_radiant: bool) -> Dict[str, Any]:
    """Translate a Stratz player_performance record into the OpenDota
    `players[]` shape that train_v18.py expects.

    slot: 0..9, 0-4 are radiant, 5-9 are dire (OpenDota convention).
    is_radiant: True if the player is on Radiant.
    """
    p = perf.get("player", {}) or {}
    perf_inner = perf.get("performance", {}) or {}
    hero = perf_inner.get("hero", {}) or {}
    steam32 = p.get("steam32")
    return {
        "account_id": steam32,
        "name": p.get("nickname") or "",
        "hero_id": hero.get("valve_id"),
        "player_slot": slot,
        "isRadiant": is_radiant,
        # performance fields — the trainer only consumes hero_id, but we
        # record the rest for downstream consumers (player_hero_stats, etc.)
        "kills": perf_inner.get("kills"),
        "deaths": perf_inner.get("deaths"),
        "assists": perf_inner.get("assists"),
        "gold_per_min": perf_inner.get("gpm"),
        "xp_per_min": perf_inner.get("xpm"),
        "level": perf_inner.get("level"),
        "hero_damage": perf_inner.get("hero_damage"),
        "hero_healing": perf_inner.get("hero_healing"),
    }


def _normalize_one(d: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Return (normalized_doc, was_changed, list_of_actions)."""
    actions: List[str] = []

    # 1) winner
    if "radiant_win" not in d and "radiant_victory" in d:
        d["radiant_win"] = bool(d["radiant_victory"])
        actions.append("radiant_win <- radiant_victory")

    # 2) team ids (Stratz: radiant.team.valve_id)
    rad = d.get("radiant") or {}
    dire = d.get("dire") or {}
    if not d.get("radiant_team_id") and isinstance(rad.get("team"), dict):
        rid = rad["team"].get("valve_id")
        if rid is not None:
            d["radiant_team_id"] = int(rid)
            actions.append(f"radiant_team_id <- {rid}")
    if not d.get("dire_team_id") and isinstance(dire.get("team"), dict):
        did = dire["team"].get("valve_id")
        if did is not None:
            d["dire_team_id"] = int(did)
            actions.append(f"dire_team_id <- {did}")

    # 3) start_time: Stratz uses millis, OpenDota uses seconds
    if not d.get("start_time"):
        if d.get("start_time"):
            pass  # already set
        elif d.get("start_date"):
            ms = int(d["start_date"])
            d["start_time"] = ms // 1000
            actions.append(f"start_time <- start_date/1000 ({d['start_time']})")

    # 4) league id (Stratz: league.league_id)
    if not d.get("leagueid"):
        league = d.get("league") or {}
        if league.get("league_id") is not None:
            d["leagueid"] = int(league["league_id"])
            actions.append(f"leagueid <- league.league_id ({d['leagueid']})")

    # 5) players[] (the big one)
    if not (isinstance(d.get("players"), list) and len(d["players"]) >= 10):
        rad_perfs = rad.get("player_performances") or []
        dire_perfs = dire.get("player_performances") or []
        players: List[Dict[str, Any]] = []
        for i, perf in enumerate(rad_perfs[:5]):
            players.append(_build_player(i, perf, is_radiant=True))
        for i, perf in enumerate(dire_perfs[:5]):
            players.append(_build_player(5 + i, perf, is_radiant=False))
        if len(players) == 10:
            d["players"] = players
            actions.append(f"players[] built (10 from {len(rad_perfs)}+{len(dire_perfs)} perfs)")
        else:
            actions.append(
                f"players[] INCOMPLETE ({len(players)}/10: "
                f"rad={len(rad_perfs)}, dire={len(dire_perfs)})"
            )

    # 6) Optional: best-of series info (Stratz has `derived_series`)
    ds = d.get("derived_series")
    if isinstance(ds, dict) and not d.get("series_type"):
        # Stratz doesn't always encode Bo1/Bo3, leave None if absent
        pass

    return d, bool(actions), actions


def main(argv: List[str]) -> int:
    files = sorted(glob.glob(CORPUS_GLOB))
    print(f"scanning {len(files)} files in {CORPUS_GLOB}")

    canonical = 0
    stratz_total = 0
    stratz_changed = 0
    stratz_incomplete_players = 0
    no_team_id = 0
    errors: List[str] = []
    sample_actions: List[Tuple[str, List[str]]] = []

    for fp in files:
        try:
            with open(fp, encoding="utf-8-sig") as f:
                d = json.load(f)
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: parse error: {e}")
            continue

        if _is_already_canonical(d):
            canonical += 1
            continue

        if not _is_stratz_format(d):
            # unknown format — leave alone
            continue

        stratz_total += 1
        d2, changed, actions = _normalize_one(d)
        if not d2.get("radiant_team_id") or not d2.get("dire_team_id"):
            no_team_id += 1
        if not (isinstance(d2.get("players"), list) and len(d2["players"]) == 10):
            stratz_incomplete_players += 1
        if changed:
            try:
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(d2, f, ensure_ascii=False, separators=(",", ":"))
                stratz_changed += 1
                if len(sample_actions) < 3:
                    sample_actions.append((os.path.basename(fp), actions))
            except Exception as e:
                errors.append(f"{os.path.basename(fp)}: write error: {e}")

    print()
    print(f"  already-canonical (OpenDota):  {canonical}")
    print(f"  stratz-format seen:            {stratz_total}")
    print(f"  stratz normalized (written):   {stratz_changed}")
    print(f"  stratz with <10 players:       {stratz_incomplete_players}")
    print(f"  stratz missing team_id:        {no_team_id}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")

    if sample_actions:
        print()
        print("Sample normalizations:")
        for name, actions in sample_actions:
            print(f"  {name}:")
            for a in actions:
                print(f"    - {a}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
