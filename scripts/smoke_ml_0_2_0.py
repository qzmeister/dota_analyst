"""
Smoke test for the 0.2.0 ML engine: load 5 real matches from
`ml_data/full_matches/`, run both HeuristicEngine and MLEngine on them,
print a side-by-side comparison.

Run with:  python -m scripts.smoke_ml_0_2_0   (from project root)
or:        cd dota_analyst && python scripts/smoke_ml_0_2_0.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `business` importable when run as `python scripts/smoke_ml_0_2_0.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.ml.engine import (  # noqa: E402
    HeuristicEngine,
    get_default_engine,
    reset_default_engine,
)


def synth_team(m: dict, side: str) -> dict:
    """Build the minimal team dict that `analyze()` requires."""
    t = m.get(side) or {}
    team = t.get("team") or {}
    return {
        "name": team.get("name") or f"Team-{side}",
        "win_rate": 50.0,  # no team aggregates in full_matches
        "fb_rate": 50.0,
        "f10_rate": 50.0,
        "rank": None,
    }


def synth_heroes(m: dict, side: str) -> list:
    """Build the minimal hero-meta list that `analyze()` + MLEngine need.

    We only need the keys `analyze()` reads (win_rate, roles, kda) plus
    the key MLEngine reads (`steam_id`). Everything else is irrelevant.
    """
    out: list = []
    for p in (m.get(side) or {}).get("player_performances") or []:
        h = (p.get("performance") or {}).get("hero") or {}
        vid = h.get("valve_id")
        if vid is None:
            continue
        out.append({
            "id": vid,
            "steam_id": vid,         # MLEngine uses this
            "name": h.get("short_name"),
            "win_rate": 50.0,
            "avg_duration": 38 * 60,
            "kda": 3.0,
            "roles": [],
        })
        if len(out) == 5:
            break
    return out


def main() -> int:
    data_dir = _ROOT / "ml_data" / "full_matches"
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        return 1

    paths = sorted(data_dir.glob("*.json"))[:5]
    if not paths:
        print("no match files found", file=sys.stderr)
        return 1

    matches = []
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            matches.append(json.load(fh))

    # Heuristic baseline (no model file lookup, deterministic).
    heuristic = HeuristicEngine()

    # ML: rebuild the singleton so we pick up the env / latest model.
    reset_default_engine()
    ml = get_default_engine()
    print(f"engine: {ml.name}\n")

    header = (
        f"{'match_id':>11} {'actual':>6} "
        f"{'heur.team':<22} {'heur.p':>6} {'heur.pr':>7} | "
        f"{'ml.team':<22} {'ml.p':>6} {'ml.pr':>7} {'src':<10}"
    )
    print(header)
    print("-" * len(header))

    correct_heur = 0
    correct_ml = 0
    for m in matches:
        team_a = synth_team(m, "radiant")
        team_b = synth_team(m, "dire")
        heroes_a = synth_heroes(m, "radiant")
        heroes_b = synth_heroes(m, "dire")
        if len(heroes_a) != 5 or len(heroes_b) != 5:
            continue

        actual = "radiant" if m.get("radiant_victory") else "dire"

        h_res = heuristic.analyze(team_a, team_b, heroes_a, heroes_b)
        m_res = ml.analyze(team_a, team_b, heroes_a, heroes_b)

        if h_res["winner"]["team"] == (team_a["name"] if actual == "radiant" else team_b["name"]):
            correct_heur += 1
        if m_res["winner"]["team"] == (team_a["name"] if actual == "radiant" else team_b["name"]):
            correct_ml += 1

        print(
            f"{m.get('match_id', 0):>11} {actual:>6} "
            f"{h_res['winner']['team'][:22]:<22} {h_res['winner']['probability']:>6} {h_res['winner']['prob_radiant']:>7} | "
            f"{m_res['winner']['team'][:22]:<22} {m_res['winner']['probability']:>6} {m_res['winner']['prob_radiant']:>7} {m_res['winner'].get('source', '?'):<10}"
        )

    print()
    print(f"heuristic correct: {correct_heur}/{len(matches)}")
    print(f"ml       correct: {correct_ml}/{len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
