"""Distribution of lane values in raw match data."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "ml_data" / "full_matches"
files = sorted(p.glob("*.json"))

counter = Counter()
total_slots = 0
n_with_lane = 0
for f in files:
    d = json.load(f.open(encoding="utf-8"))
    for side in ("radiant", "dire"):
        for pp in (d.get(side) or {}).get("player_performances") or []:
            total_slots += 1
            li = (pp.get("laneInfo") or {}).get("lane")
            if li:
                counter[li] += 1
                n_with_lane += 1
            else:
                counter["<MISSING>"] += 1

print(f"total slots: {total_slots}")
print(f"with laneInfo.lane: {n_with_lane}")
print("distribution:")
for k, v in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"  {k:>15}: {v} ({v/total_slots:.1%})")

# Check switchTo as well
switch_counter = Counter()
for f in files[:200]:  # sample
    d = json.load(f.open(encoding="utf-8"))
    for side in ("radiant", "dire"):
        for pp in (d.get(side) or {}).get("player_performances") or []:
            li = pp.get("laneInfo") or {}
            sw = li.get("switchTo")
            if sw:
                switch_counter[sw] += 1
print("\nswitchTo distribution (200 matches sample):")
for k, v in sorted(switch_counter.items(), key=lambda x: -x[1]):
    print(f"  {k:>15}: {v}")
