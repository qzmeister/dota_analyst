"""Quick diagnostic: how many matches, teams, lane info, etc."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "ml_data" / "full_matches"
files = sorted(p.glob("*.json"))
print(f"total files: {len(files)}")

teams: set[int] = set()
team_counts: dict[int, int] = defaultdict(int)
lane_ok = 0
lane_missing: list[str] = []
hero_count_per_match: list[int] = []
for f in files:
    d = json.load(f.open(encoding="utf-8"))
    r = (d.get("radiant") or {}).get("team") or {}
    dd = (d.get("dire") or {}).get("team") or {}
    r_vid = r.get("valve_id")
    d_vid = dd.get("valve_id")
    if isinstance(r_vid, int) and isinstance(d_vid, int):
        teams.add(r_vid)
        teams.add(d_vid)
        team_counts[r_vid] += 1
        team_counts[d_vid] += 1
    lanes_ok = True
    n_heroes = 0
    for side in ("radiant", "dire"):
        for pp in (d.get(side) or {}).get("player_performances") or []:
            n_heroes += 1
            if not (pp.get("performance") or {}).get("lane"):
                lanes_ok = False
    hero_count_per_match.append(n_heroes)
    if lanes_ok:
        lane_ok += 1
    else:
        lane_missing.append(f.name)

print(f"unique teams: {len(teams)}")
print(f"matches with full lane info: {lane_ok}")
print(f"matches missing some lane: {len(lane_missing)}")
if team_counts:
    vals = sorted(team_counts.values())
    print(f"team match-count min/median/max: {vals[0]}/{vals[len(vals)//2]}/{vals[-1]}")
    print(f"team count distribution (1/2/3-5/6+):")
    one = sum(1 for v in vals if v == 1)
    two = sum(1 for v in vals if v == 2)
    small = sum(1 for v in vals if 3 <= v <= 5)
    med = sum(1 for v in vals if 6 <= v <= 20)
    large = sum(1 for v in vals if v > 20)
    print(f"  1 match:   {one}")
    print(f"  2 matches: {two}")
    print(f"  3-5:       {small}")
    print(f"  6-20:      {med}")
    print(f"  >20:       {large}")

# Hero distribution
n_hero_counts = defaultdict(int)
for n in hero_count_per_match:
    n_hero_counts[n] += 1
print(f"player_performances per match: {dict(n_hero_counts)}")
