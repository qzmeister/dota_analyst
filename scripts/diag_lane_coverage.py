"""Coverage of the 5 roles we use for lane features."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from business.ml.features import lane_heroes_from_match

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "ml_data" / "full_matches"
files = sorted(p.glob("*.json"))

# We use: bot_pair(BOT_CARRY+BOT_SUPPORT), top_jungle(TOP+JUNGLE), mid_matchup(MID)
# For each side, we need BOT_CARRY + BOT_SUPPORT + TOP + JUNGLE + MID = 5 cells.
# A match is "bot pair usable" iff both BOT_CARRY and BOT_SUPPORT are filled.
# "top_jungle usable" iff both TOP and JUNGLE are filled.
# "mid matchup usable" iff both radiant MID and dire MID are filled.

usability = {"bot": 0, "tj": 0, "mid": 0}
total = 0
for f in files:
    d = json.load(f.open(encoding="utf-8"))
    lanes = lane_heroes_from_match(d)
    total += 1
    r = lanes["radiant"]; dd = lanes["dire"]
    if r["BOT_CARRY"] is not None and r["BOT_SUPPORT"] is not None and \
       dd["BOT_CARRY"] is not None and dd["BOT_SUPPORT"] is not None:
        usability["bot"] += 1
    if r["TOP"] is not None and r["JUNGLE"] is not None and \
       dd["TOP"] is not None and dd["JUNGLE"] is not None:
        usability["tj"] += 1
    if r["MID"] is not None and dd["MID"] is not None:
        usability["mid"] += 1

print(f"total: {total}")
for k, v in usability.items():
    print(f"  {k}-pair usable: {v} / {total} = {v/total:.1%}")
