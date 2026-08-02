"""v0.7.36: Compare v0.7.0 tier vs Stratz premium-list.

User feedback (v0.7.27):
  a team winning 80% in tier-3 amateur leagues is NOT the
  same as a team winning 60% in MAJOR+.  So our v0.7.0
  percentile-tier (computed from 540 OpenDota pro matches
  with Bayesian Glicko) can be misleading.

This script answers:
  - Who is in v0.7.0 = premium (14) but NOT in Stratz
    premium-list of 64?  -> phantom premium, v0.7.0 over-rated
  - Who is in Stratz premium-list but v0.7.0 = minor (467)
    or professional (40)?  -> dark horse, v0.7.0 under-rated

Then proposes a RECOMMENDED tier assignment based on a
blend of v0.7.0 and Stratz signal.

Inputs:
  ml_data/imports/v18_top_teams.json     (v0.7.0 computed)
  ml_data/imports/stratz_premium_teams.json (v0.7.33 dump)

Output (printed):
  - Coverage summary
  - Phantom premium list (if any)
  - Top 20 dark horses (Stratz premium, v0.7.0 minor/pro)
  - Proposed tier-recommendation histogram
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = ROOT / "ml_data" / "imports"

V18_FILE = IMPORTS / "v18_top_teams.json"
STRATZ_FILE = IMPORTS / "stratz_premium_teams.json"


def _load_v18() -> Dict[int, Dict[str, Any]]:
    """Load v18_top_teams.json as {team_id: row}."""
    rows = json.loads(V18_FILE.read_text(encoding="utf-8"))
    return {int(r["team_id"]): r for r in rows if r.get("team_id") is not None}


def _load_stratz() -> Dict[int, Dict[str, Any]]:
    """Load stratz_premium_teams.json as {team_id: row}."""
    rows = json.loads(STRATZ_FILE.read_text(encoding="utf-8"))
    return {int(r["team_id"]): r for r in rows if r.get("team_id") is not None}


def _tier_label(t: int) -> str:
    return {0: "minor", 1: "professional", 2: "premium"}.get(t, f"tier{t}")


def main() -> int:
    v18 = _load_v18()
    stratz = _load_stratz()
    print(f"v18 teams: {len(v18)}")
    print(f"stratz premium teams: {len(stratz)}")

    # Coverage
    stratz_ids = set(stratz.keys())
    v18_ids = set(v18.keys())
    in_both = stratz_ids & v18_ids
    in_stratz_only = stratz_ids - v18_ids
    in_v18_only = v18_ids - stratz_ids
    print(f"  in both: {len(in_both)}")
    print(f"  stratz-only (new teams not in v18 seed): {len(in_stratz_only)}")
    print(f"  v18-only (not in MAJOR+ per Stratz): {len(in_v18_only)}")
    if in_stratz_only:
        print(f"    examples: {sorted(in_stratz_only)[:10]}")

    # ----- Phantom premium: v0.7.0=premium but not in Stratz MAJOR+ -----
    print(f"\n=== PHANTOM PREMIUM (v0.7.0=premium, NOT in Stratz MAJOR+) ===")
    phantoms = [tid for tid in v18
                if v18[tid].get("tier") == 2 and tid not in stratz_ids]
    print(f"  count: {len(phantoms)}")
    for tid in sorted(phantoms):
        v = v18[tid]
        name = v.get("name") or f"id={tid}"
        print(f"    {tid:>10d}  v18_rating={v.get('rating', 0):.0f}  "
              f"v18_W/L={v.get('wins', 0)}/{v.get('losses', 0)}  "
              f"{name[:30]}")

    # ----- Dark horses: in Stratz MAJOR+ but v0.7.0 = minor or professional -----
    print(f"\n=== DARK HORSES (in Stratz MAJOR+, v0.7.0=minor or pro) ===")
    dark = []
    for tid in in_both:
        v = v18[tid]
        if v.get("tier") == 2:
            continue  # already in premium
        s = stratz[tid]
        dark.append({
            "team_id": tid,
            "v18_tier": v.get("tier", 0),
            "v18_rating": v.get("rating", 0),
            "v18_W/L": (v.get("wins", 0), v.get("losses", 0)),
            "stratz_premium_count": s.get("premium_leagues_count", 0),
            "stratz_max_tier": s.get("max_tier"),
            "stratz_total_prize": s.get("total_prize", 0),
        })
    # sort by stratz signal
    dark.sort(key=lambda r: (-r["stratz_premium_count"],
                              -r["stratz_total_prize"]))
    print(f"  count: {len(dark)}")
    for d in dark[:20]:
        v_tier = _tier_label(d["v18_tier"])
        # v0.7.36: cast prize to int (Stratz file may have float
        # if it was generated before v0.7.34)
        prize = int(round(d["stratz_total_prize"] or 0))
        print(f"    {d['team_id']:>10d}  v18={v_tier:<13s}  "
              f"stratz={d['stratz_premium_count']:>2d} лиг  "
              f"max={d['stratz_max_tier']:<13s}  "
              f"prize=${prize:>9d}  "
              f"v18_W/L={d['v18_W/L'][0]}/{d['v18_W/L'][1]}")

    # ----- Same-tier cohort summary -----
    print(f"\n=== CROSS-TAB (rows=v0.7.0 tier, cols=Stratz presence) ===")
    cross = {0: {"in_stratz": 0, "not_in_stratz": 0},
             1: {"in_stratz": 0, "not_in_stratz": 0},
             2: {"in_stratz": 0, "not_in_stratz": 0}}
    for tid, v in v18.items():
        t = v.get("tier", 0)
        if t not in cross:
            continue
        if tid in stratz_ids:
            cross[t]["in_stratz"] += 1
        else:
            cross[t]["not_in_stratz"] += 1
    print(f"  {'v18_tier':<13s}  {'in_stratz':>10s}  {'not_in_stratz':>13s}  "
          f"{'total':>6s}  {'%_in_stratz':>11s}")
    for t in (0, 1, 2):
        c = cross[t]
        total = c["in_stratz"] + c["not_in_stratz"]
        pct = c["in_stratz"] / total * 100 if total else 0
        print(f"  {_tier_label(t):<13s}  {c['in_stratz']:>10d}  "
              f"{c['not_in_stratz']:>13d}  {total:>6d}  {pct:>10.1f}%")

    # ----- Proposed tier merge -----
    # Rules:
    #   - v0.7.0 premium + in Stratz MAJOR+     -> keep premium (0.7 confirmed)
    #   - v0.7.0 premium + NOT in Stratz        -> demote to professional
    #   - v0.7.0 professional + in Stratz MAJOR+ (>=3 leagues) -> promote to premium
    #   - v0.7.0 professional + in Stratz MAJOR+ (<3)  -> keep professional (verified)
    #   - v0.7.0 minor + in Stratz MAJOR+ (>=3 leagues or INTERNATIONAL) -> promote to professional
    #   - v0.7.0 minor + in Stratz MAJOR+ (<3, all MAJOR) -> keep minor
    #   - v0.7.0 X + not in Stratz               -> keep v0.7.0
    print(f"\n=== PROPOSED TIER MERGE (v0.7.0 + Stratz signal) ===")
    changes: Dict[str, int] = {
        "premium->premium": 0,
        "premium->professional": 0,
        "professional->premium": 0,
        "professional->professional": 0,
        "professional->minor": 0,
        "minor->professional": 0,
        "minor->minor": 0,
        "new_team_no_v18": 0,
    }
    new_team_examples: List[int] = []
    for tid, v in v18.items():
        old_tier = v.get("tier", 0)
        if tid in stratz_ids:
            s = stratz[tid]
            cnt = s.get("premium_leagues_count", 0)
            max_t = s.get("max_tier")
            if old_tier == 2:
                changes["premium->premium"] += 1
            elif old_tier == 1:
                if cnt >= 3:
                    changes["professional->premium"] += 1
                else:
                    changes["professional->professional"] += 1
            else:  # minor
                if cnt >= 3 or max_t == "INTERNATIONAL":
                    changes["minor->professional"] += 1
                else:
                    changes["minor->minor"] += 1
        else:
            if old_tier == 2:
                changes["premium->professional"] += 1
            elif old_tier == 1:
                changes["professional->professional"] += 1
            else:
                changes["minor->minor"] += 1
    for tid in in_stratz_only:
        changes["new_team_no_v18"] += 1
        if len(new_team_examples) < 5:
            new_team_examples.append(tid)
    for k, c in changes.items():
        print(f"  {k:<30s}  {c:>4d}")
    print(f"  (new Stratz teams without v18 entry: "
          f"{changes['new_team_no_v18']}  e.g. {new_team_examples})")

    # Net effect
    new_premium = (changes["premium->premium"]
                   + changes["professional->premium"])
    new_professional = (changes["premium->professional"]
                        + changes["professional->professional"]
                        + changes["minor->professional"])
    new_minor = changes["minor->minor"]
    print(f"\n  PROPOSED premium     : {new_premium}  "
          f"(was {cross[2]['in_stratz'] + cross[2]['not_in_stratz']})")
    print(f"  PROPOSED professional: {new_professional}  "
          f"(was {cross[1]['in_stratz'] + cross[1]['not_in_stratz']})")
    print(f"  PROPOSED minor       : {new_minor}  "
          f"(was {cross[0]['in_stratz'] + cross[0]['not_in_stratz']})")
    print(f"  NEW (no v18 entry)   : {changes['new_team_no_v18']}")
    print(f"  TOTAL (v18 + new)    : "
          f"{new_premium + new_professional + new_minor + changes['new_team_no_v18']}  "
          f"(was {len(v18)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
