"""
Smoke test for the 0.2.1 multi-target ML engine: load 5 real matches
from `ml_data/full_matches/`, run both HeuristicEngine and MLEngine
on them, print a side-by-side comparison of every block that ML
overrides in 0.2.1 (winner, kills, duration).

Run with:  python scripts/smoke_ml_0_2_1.py   (from project root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.ml.engine import (  # noqa: E402
    HeuristicEngine,
    get_default_engine,
    reset_default_engine,
)


def synth_team(m: dict, side: str) -> dict:
    t = (m.get(side) or {}).get("team") or {}
    return {
        "name": t.get("name") or f"Team-{side}",
        "win_rate": 50.0,  # no team aggregates in full_matches
        "fb_rate": 50.0,
        "f10_rate": 50.0,
        "rank": None,
    }


def synth_heroes(m: dict, side: str) -> list:
    out: list = []
    for p in (m.get(side) or {}).get("player_performances") or []:
        h = (p.get("performance") or {}).get("hero") or {}
        vid = h.get("valve_id")
        if vid is None:
            continue
        out.append({
            "id": vid,
            "steam_id": vid,
            "name": h.get("short_name"),
            "win_rate": 50.0,
            "avg_duration": 38 * 60,
            "kda": 3.0,
            "roles": [],
        })
        if len(out) == 5:
            break
    return out


def _winner_arrow(pred: dict, team_a: dict, team_b: dict, actual_side: str) -> str:
    """Mark correct ('OK') or wrong ('X') winner prediction."""
    if not pred or "team" not in pred:
        return "?"
    actual = team_a["name"] if actual_side == "radiant" else team_b["name"]
    return "OK" if pred["team"] == actual else "X"


def _fmt(v, fmt: str) -> str:
    if v is None:
        return "n/a"
    return format(v, fmt)


def main() -> int:
    data_dir = _ROOT / "ml_data" / "full_matches"
    paths = sorted(data_dir.glob("*.json"))[:5]
    if not paths:
        print("no match files found", file=sys.stderr)
        return 1

    matches = []
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            matches.append(json.load(fh))

    heuristic = HeuristicEngine()
    reset_default_engine()
    ml = get_default_engine()
    print(f"engine: {ml.name}\n")

    print(f"{'match_id':>11} {'actual':>6}  {'h_win':>4} {'m_win':>4}  "
          f"{'h_kills':>7} {'m_kills':>7} {'act_kills':>10}  "
          f"{'h_dur':>6} {'m_dur':>6} {'act_dur':>7}  "
          f"{'m_src':>10}")
    print("-" * 110)

    n_h_winner = n_m_winner = 0
    abs_h_kills = abs_m_kills = 0
    abs_h_dur = abs_m_dur = 0
    n_kills = n_dur = 0

    for m in matches:
        team_a = synth_team(m, "radiant")
        team_b = synth_team(m, "dire")
        heroes_a = synth_heroes(m, "radiant")
        heroes_b = synth_heroes(m, "dire")
        if len(heroes_a) != 5 or len(heroes_b) != 5:
            continue

        actual = "radiant" if m.get("radiant_victory") else "dire"
        actual_kills = sum(
            (p.get("performance") or {}).get("kills", 0) or 0
            for side in ("radiant", "dire")
            for p in (m.get(side) or {}).get("player_performances") or []
        )
        actual_dur = round(m.get("duration", 0) / 60, 1)

        h_res = heuristic.analyze(team_a, team_b, heroes_a, heroes_b)
        m_res = ml.analyze(team_a, team_b, heroes_a, heroes_b)

        h_win = h_res["winner"]["team"]
        m_win = m_res["winner"]["team"]
        h_kills = h_res["kills"]["total"]
        m_kills = m_res["kills"]["total"]
        h_dur = h_res["duration_min"]
        m_dur = m_res["duration_min"]

        h_arrow = _winner_arrow(h_res["winner"], team_a, team_b, actual)
        m_arrow = _winner_arrow(m_res["winner"], team_a, team_b, actual)

        if h_arrow == "OK":
            n_h_winner += 1
        if m_arrow == "OK":
            n_m_winner += 1
        abs_h_kills += abs(h_kills - actual_kills)
        abs_m_kills += abs(m_kills - actual_kills)
        n_kills += 1
        abs_h_dur += abs(h_dur - actual_dur)
        abs_m_dur += abs(m_dur - actual_dur)
        n_dur += 1

        print(
            f"{m.get('match_id', 0):>11} {actual:>6}  "
            f"{h_arrow:>4} {m_arrow:>4}  "
            f"{h_kills:>7} {m_kills:>7} {actual_kills:>10}  "
            f"{_fmt(h_dur, '.1f'):>6} {_fmt(m_dur, '.1f'):>6} {actual_dur:>7.1f}  "
            f"{m_res['winner'].get('source', '?'):>10}"
        )

    print()
    print(f"  winner correct:  heuristic {n_h_winner}/5   ml {n_m_winner}/5")
    if n_kills:
        print(f"  kills MAE:       heuristic {abs_h_kills / n_kills:.1f}   ml {abs_m_kills / n_kills:.1f}")
    if n_dur:
        print(f"  duration MAE:    heuristic {abs_h_dur / n_dur:.1f}   ml {abs_m_dur / n_dur:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
