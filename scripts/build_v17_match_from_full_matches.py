"""Build v17_match_*.json files from full_matches/*.json for matches
that don't already have a v17_match counterpart.

Background
----------
The v18 trainer (scripts/train_v18.py) reads from
`ml_data/imports/v17_match_*.json`.  The standard pipeline is:

    v17_match_*.json (raw OpenDota /matches/{id})  --normalize_v17.py-->
    full_matches/*.json (normalized form)

For the 3403 OpenDota matches this has already been done: each
v17_match_*.json has a corresponding full_matches/<id>.json.

The 1824 Stratz-format matches (post-normalize_full_matches.py) live
only in full_matches/.  They have the OpenDota-compat fields
(radiant_win, players[10], team_ids, start_time, etc.) but the
trainer needs them as v17_match_*.json, and it also requires
`radiant_score` + `dire_score` to be non-zero (else the match is
silently dropped).

This script:
  1. walks ml_data/full_matches/*.json
  2. for any <id>.json whose ml_data/imports/v17_match_<id>.json
     doesn't exist, builds a minimal v17_match_<id>.json with the
     required fields, computing radiant_score/dire_score by
     summing players[0..4].kills and players[5..9].kills.
  3. is idempotent: never overwrites an existing v17_match_*.json.

Run:  python scripts/build_v17_match_from_full_matches.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PRO_ROOT = Path(r"C:\Users\artka\.minimax\workspace\dota_analyst")
FULL_MATCHES = PRO_ROOT / "ml_data" / "full_matches"
IMPORTS = PRO_ROOT / "ml_data" / "imports"


def _compute_scores(players: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Return (radiant_kills, dire_kills, players_with_no_kill_field).

    Players are positional: 0-4 radiant, 5-9 dire.  If a player has
    no `kills` key (older OpenDota format), treat it as 0.
    """
    r_kills = sum(int(p.get("kills") or 0) for p in players[:5])
    d_kills = sum(int(p.get("kills") or 0) for p in players[5:10])
    no_kill = sum(1 for p in players if p.get("kills") is None)
    return r_kills, d_kills, no_kill


def _is_interesting(d: Dict[str, Any]) -> bool:
    """A file is worth promoting to v17_match if it has a winner label
    and a 10-player list (i.e. the trainer will accept it)."""
    return (
        isinstance(d.get("players"), list)
        and len(d["players"]) >= 10
        and "radiant_win" in d
    )


def main() -> int:
    full_files = sorted(glob.glob(str(FULL_MATCHES / "*.json")))
    print(f"scanning {len(full_files)} files in {FULL_MATCHES}")

    promoted = 0
    skipped_already = 0
    skipped_no_winner = 0
    skipped_no_players = 0
    skipped_zero_kills = 0
    no_kill_field = 0
    sample: List[Tuple[str, int, int]] = []

    for fp in full_files:
        match_id = os.path.splitext(os.path.basename(fp))[0]
        out_path = IMPORTS / f"v17_match_{match_id}.json"
        if out_path.exists():
            skipped_already += 1
            continue

        try:
            with open(fp, encoding="utf-8-sig") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  parse error {match_id}: {e}")
            continue

        if not isinstance(d.get("players"), list) or len(d["players"]) < 10:
            skipped_no_players += 1
            continue
        if "radiant_win" not in d:
            skipped_no_winner += 1
            continue

        r_kills, d_kills, no_kill = _compute_scores(d["players"])
        if no_kill:
            no_kill_field += no_kill
        if r_kills == 0 and d_kills == 0:
            # Trainer's `if radiant_score == 0 and dire_score == 0: return None`
            # would drop this.  Skip rather than inject fake scores.
            skipped_zero_kills += 1
            continue

        # Build a minimal v17_match payload.  We keep all the
        # canonical fields the trainer needs, plus the original
        # Stratz wrappers (radiant/dire) so nothing is lost.
        out: Dict[str, Any] = {
            "version": 1,
            "match_id": d.get("match_id", int(match_id)),
            "players": d["players"],
            "radiant_win": d["radiant_win"],
            "radiant_team_id": d.get("radiant_team_id"),
            "dire_team_id": d.get("dire_team_id"),
            "leagueid": d.get("leagueid"),
            "start_time": d.get("start_time"),
            "duration": d.get("duration"),
            "patch": d.get("patch"),
            "radiant_score": r_kills,
            "dire_score": d_kills,
            "draft_timings": d.get("draft_timings") or [],
            "radiant_gold_adv": d.get("radiant_gold_adv") or [],
            "radiant_xp_adv": d.get("radiant_xp_adv") or [],
            "series_id": d.get("series_id"),
            "series_type": d.get("series_type"),
            "pre_game_duration": d.get("pre_game_duration"),
            # Preserve the original Stratz wrappers for downstream consumers
            # (player_hero_stats.py, build_v17_from_stratz, future work).
            "radiant": d.get("radiant"),
            "dire": d.get("dire"),
            "radiant_victory": d.get("radiant_victory"),
        }

        # Make sure the imports dir exists
        IMPORTS.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        promoted += 1
        if len(sample) < 5:
            sample.append((match_id, r_kills, d_kills))

    print()
    print(f"  already had v17_match:        {skipped_already}")
    print(f"  promoted (new v17_match):     {promoted}")
    print(f"  skipped (no players[10]):     {skipped_no_players}")
    print(f"  skipped (no radiant_win):     {skipped_no_winner}")
    print(f"  skipped (zero kills):         {skipped_zero_kills}")
    print(f"  players missing .kills field: {no_kill_field}")
    if sample:
        print()
        print("Sample promotions (match_id, radiant_kills, dire_kills):")
        for s in sample:
            print(f"  {s[0]}  {s[1]}-{s[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
