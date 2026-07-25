"""Diagnose: where is `lane` in the match JSON?"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "ml_data" / "full_matches"
f = sorted(p.glob("*.json"))[0]
d = json.load(f.open(encoding="utf-8"))
print(f"file: {f.name}")
print("top-level keys:", list(d.keys()))
for side in ("radiant", "dire"):
    print(f"\n=== {side} ===")
    print("  top-level keys:", list((d.get(side) or {}).keys()))
    pps = (d.get(side) or {}).get("player_performances") or []
    if pps:
        print(f"  player_performances[0] keys: {list(pps[0].keys())}")
        perf = pps[0].get("performance") or {}
        print(f"  performance[0] keys: {list(perf.keys())}")
        if "lane" in perf:
            print(f"  lane value: {perf.get('lane')!r}")
        else:
            print("  no 'lane' in performance")
        # check nested
        for k, v in perf.items():
            if isinstance(v, dict) and "lane" in v:
                print(f"  found lane in performance.{k}: {v['lane']!r}")
print("\n=== Search all nested for 'lane' or 'laneInfo' ===")
def find_lane(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("lane", "laneInfo", "lane_id", "laneRole"):
                print(f"  {path}.{k} = {v!r}")
            find_lane(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2]):
            find_lane(v, f"{path}[{i}]")
find_lane(d)
