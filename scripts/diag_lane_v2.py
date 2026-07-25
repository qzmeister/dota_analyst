"""Test alternative lane-pair definitions."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business.ml.features import lane_heroes_from_match

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "ml_data" / "full_matches"
files = sorted(p.glob("*.json"))

# Variants:
#  A) bot=BOTTOM+ROAM, top_pair=TOP+TOP (proxy for offlane+jungler), mid=MID
#  B) bot=BOTTOM+ROAM, top_pair=TOP+any_non_BOTTOM_non_MID, mid=MID
#  C) skip top_pair entirely

variant_A = {"bot": 0, "top": 0, "mid": 0}
variant_B = {"bot": 0, "top": 0, "mid": 0}
for f in files:
    d = json.load(f.open(encoding="utf-8"))
    lanes = lane_heroes_from_match(d)
    r = lanes["radiant"]; dd = lanes["dire"]
    # A
    if r["BOT_CARRY"] and r["BOT_SUPPORT"] and dd["BOT_CARRY"] and dd["BOT_SUPPORT"]:
        variant_A["bot"] += 1
    if r["TOP"] and dd["TOP"]:
        variant_A["top"] += 1
    if r["MID"] and dd["MID"]:
        variant_A["mid"] += 1
    # B (drop the ROAM requirement; pos5 may also be on BOTTOM sometimes)
    bot_r = r["BOT_CARRY"] is not None
    bot_d = dd["BOT_CARRY"] is not None
    if bot_r and bot_d:
        variant_B["bot"] += 1
    if r["TOP"] and dd["TOP"]:
        variant_B["top"] += 1
    if r["MID"] and dd["MID"]:
        variant_B["mid"] += 1

print(f"Variant A: BOTTOM+ROAM / TOP+TOP / MID")
for k, v in variant_A.items():
    print(f"  {k}-pair usable: {v} / {len(files)} = {v/len(files):.1%}")
print()
print(f"Variant B: BOTTOM (any) / TOP+TOP / MID")
for k, v in variant_B.items():
    print(f"  {k}-pair usable: {v} / {len(files)} = {v/len(files):.1%}")

# Check: how often does BOTTOM side have 2 BOTTOM players?
two_bottom_per_side = 0
for f in files:
    d = json.load(f.open(encoding="utf-8"))
    for side in ("radiant", "dire"):
        bottoms = []
        for pp in (d.get(side) or {}).get("player_performances") or []:
            li = (pp.get("laneInfo") or {}).get("lane")
            if li == "BOTTOM":
                vid = (pp.get("performance") or {}).get("hero", {}).get("valve_id")
                if isinstance(vid, int):
                    bottoms.append(vid)
        if len(bottoms) >= 2:
            two_bottom_per_side += 1
print(f"\ntwo BOTTOM players per side: {two_bottom_per_side} / {2*len(files)} = {two_bottom_per_side/(2*len(files)):.1%}")
