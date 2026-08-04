"""Compute per-team ratings from ml_data/imports/v17_match_*.json.

v17 shipped with v17_phase1_top_teams.json containing only 30 teams
(2024-2025 era top 30).  Our corpus has 540 unique team_ids
spanning 2015-2026, so 98% of teams fall into the "minor" tier
(tier=0) and the model can't distinguish them.

This script walks every match file, accumulates per-team
(wins, losses, last_match_time, name, tag), computes a Glicko-like
rating with recency weighting, and writes the result to
ml_data/imports/v18_top_teams.json.  We keep the same JSON shape
as v17_phase1_top_teams.json so the existing _tier_for() lookup
in business/v18_predict.py and business/v17_predict.py works
without code changes — just swap the filename.

Rating formula
--------------
For each team we compute a smoothed Glicko-style rating:

    raw_rating = 1500 + 400 * (wins - losses) / (wins + losses)
    recency_factor = exp(-(now - last_match) / (270 days))
    rating = 1500 + 0.7 * (raw_rating - 1500) * recency_factor

Plus a Bayesian smoothing for low-sample teams:

    effective_n = wins + losses + 5
    rating = (effective_n * raw_rating + 5 * 1500) / (effective_n + 5)

This pulls teams with <5 games toward 1500 (the prior) so a single
lucky upset doesn't push a brand-new team to 1900.

Output: ml_data/imports/v18_top_teams.json

Run:  python scripts/compute_team_ratings.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"
INPUT_TOP = IMPORTS / "v17_phase1_top_teams.json"
OUTPUT = IMPORTS / "v18_top_teams.json"

# Glicko parameters
INIT_RATING = 1500.0
PRIOR_N = 5              # Bayesian smoothing prior
RECENCY_HALF_LIFE_DAYS = 270.0
RECENCY_WEIGHT = 0.7     # how much the recency factor contributes
# v0.7.0: a team must have played at least this many matches to
# be eligible for any tier above minor.  Without this, a 4-0
# qualifier team ranks above a 100-game pro team with 50% WR,
# which inverts the v17 Glicko-based tier system.  The v18
# corpus has 288 teams with <5 games (mostly academy /
# qualifier rosters) and only 47 teams with 30+ games (real
# pro scene).  Setting MIN_GAMES=10 keeps the latter 30+ teams
# in the running and demotes the former to "minor" regardless
# of win rate.
MIN_GAMES = 10


def list_match_files() -> List[Path]:
    return sorted(IMPORTS.glob("v17_match_*.json"))


def load_match(p: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "radiant_win" not in d:
        return None
    return d


def compute_ratings() -> List[Dict[str, Any]]:
    files = list_match_files()
    print(f"  scanning {len(files)} match files")
    # team_id -> {wins, losses, last_match_time, name?, tag?}
    acc: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "wins": 0, "losses": 0, "last_match_time": 0,
        "name": None, "tag": None, "match_id": None,
    })
    skipped = 0
    for p in files:
        m = load_match(p)
        if m is None:
            skipped += 1
            continue
        radiant_win = bool(m.get("radiant_win"))
        r_id = m.get("radiant_team_id")
        d_id = m.get("dire_team_id")
        st = int(m.get("start_time") or 0)
        mid = m.get("match_id")
        if not r_id or not d_id:
            continue
        # Radiant name and tag are not in the /matches payload, but
        # sometimes the v17 collector included them in the league
        # wrapper.  Try several keys; otherwise stay None.
        # (The heuristic doesn't need name/tag for the tier
        # lookup, but downstream UI may.)
        # Win/loss bookkeeping: if radiant won, radiant gets
        # the win and dire gets the loss.  We had this
        # inverted in the v0.7.0 first pass (which is why
        # every "premium" team had a 25% WR in the train+test
        # stat — they were actually the worst teams).
        if radiant_win:
            acc[int(r_id)]["wins"] += 1
            acc[int(d_id)]["losses"] += 1
        else:
            acc[int(r_id)]["losses"] += 1
            acc[int(d_id)]["wins"] += 1
        for tid in (int(r_id), int(d_id)):
            if st > acc[tid]["last_match_time"]:
                acc[tid]["last_match_time"] = st
                if mid:
                    acc[tid]["match_id"] = mid
    print(f"  skipped {skipped} unparseable")
    print(f"  unique team_ids: {len(acc)}")
    return _score(acc)


def _score(acc: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply Bayesian + recency smoothing, return sorted list."""
    now = max(int(time.time()),
              max((t["last_match_time"] for t in acc.values()), default=0))
    out: List[Dict[str, Any]] = []
    for tid, t in acc.items():
        w = t["wins"]
        l = t["losses"]
        if w + l == 0:
            continue
        # Bayesian smoothing: pulls low-sample teams to 1500.
        raw = INIT_RATING + 400.0 * (w - l) / max(1, w + l)
        eff_n = w + l + PRIOR_N
        smoothed = (eff_n * raw + PRIOR_N * INIT_RATING) / (eff_n + PRIOR_N)
        # Recency: weight by exp(-(now - last) / half_life).
        age_days = max(0.0, (now - t["last_match_time"]) / 86400.0)
        recency = math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
        # Final rating: blend 1500 baseline with smoothed * recency.
        rating = INIT_RATING + RECENCY_WEIGHT * (smoothed - INIT_RATING) * recency
        out.append({
            "team_id": tid,
            "rating": round(rating, 2),
            "wins": w,
            "losses": l,
            "last_match_time": t["last_match_time"],
            "match_id": t["match_id"],
            "name": t.get("name"),
            "tag": t.get("tag"),
            # Source field so downstream code can tell which
            # snapshot produced the row.
            "source": "v18_compute_team_ratings",
        })
    # Sort by rating desc.
    out.sort(key=lambda r: -r["rating"])
    # v0.7.60: tier is assigned by PURE RATING PERCENTILE over ALL
    # 540 teams we've ever seen — no MIN_GAMES gate.  The earlier
    # v0.7.0 logic (MIN_GAMES=10) silently forced 73% of teams to
    # tier=0 even when their rating was high (e.g. 8-0 rookies
    # with rating 1680), which meant the v18 model saw
    # `r_premium=1` in < 1% of matches and learned that the tier
    # features were noise.  The new rule:
    #   * top 10% by rating → premium (tier=2)
    #   * next 30% → professional (tier=1)
    #   * bottom 60% → minor (tier=0)
    # applied to ALL teams.  This gives the model a useful
    # gradient: ~10% of matches now have r_premium=1, ~40% have
    # r_top_team=1.  The new `min_games_eligible` flag is
    # informational only — it's no longer a gate.
    n = len(out)
    p10 = int(n * 0.10)
    p40 = int(n * 0.40)
    eligible_tiers: Dict[int, int] = {}
    for i, r in enumerate(out):
        if i < p10:
            eligible_tiers[int(r["team_id"])] = 2
        elif i < p40:
            eligible_tiers[int(r["team_id"])] = 1
        else:
            eligible_tiers[int(r["team_id"])] = 0
    for r in out:
        tid = int(r["team_id"])
        r["tier"] = eligible_tiers.get(tid, 0)
        # Compute the team's rating percentile for diagnostics.
        rank = next((i for i, t in enumerate(out) if int(t["team_id"]) == tid), None)
        r["tier_percentile"] = (1.0 - (rank / max(1, n))) if rank is not None else None
        r["min_games_eligible"] = (r["wins"] + r["losses"]) >= MIN_GAMES
    return out


def main() -> int:
    print("=" * 78)
    print("Compute team ratings from v17_match_*.json corpus")
    print("=" * 78)
    print()
    rows = compute_ratings()
    # Print distribution by tier
    by_tier: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    for r in rows:
        by_tier[int(r.get("tier", 0))] += 1
    tier_names = {0: "minor", 1: "professional", 2: "premium"}
    print(f"  tier distribution (percentile-based):")
    for tier, count in sorted(by_tier.items()):
        print(f"    tier={tier} ({tier_names[tier]:12s}): {count:>4d} teams "
              f"({100.0*count/max(1,len(rows)):.1f}%)")
    # Compare with v17 top teams
    if INPUT_TOP.exists():
        old = json.loads(INPUT_TOP.read_text(encoding="utf-8"))
        old_ids = {int(t["team_id"]) for t in old if t.get("team_id") is not None}
        new_ids = {int(t["team_id"]) for t in rows}
        in_both = old_ids & new_ids
        only_new = new_ids - old_ids
        only_old = old_ids - new_ids
        print(f"  v17 top list: {len(old_ids)} teams")
        print(f"  v18 new list: {len(new_ids)} teams")
        print(f"    in both: {len(in_both)}")
        print(f"    only in v18 (new coverage): {len(only_new)}")
        print(f"    only in v17 (historic): {len(only_old)}")
    print()
    print(f"  top 10 (premium tier):")
    for r in rows[:10]:
        print(f"    team_id={r['team_id']:>10d}  rating={r['rating']:>7.2f}  "
              f"tier={r['tier']}  wins={r['wins']:>4d}  losses={r['losses']:>4d}")
    print()
    print(f"  tier boundary (around 30% mark, tier=0 starts):")
    if len(rows) >= 5:
        for r in rows[max(0, len(rows) - 5):]:
            print(f"    team_id={r['team_id']:>10d}  rating={r['rating']:>7.2f}  "
                  f"tier={r['tier']}  wins={r['wins']:>4d}  losses={r['losses']:>4d}")
    print()
    print(f"  bottom 5:")
    for r in rows[-5:]:
        print(f"    team_id={r['team_id']:>10d}  rating={r['rating']:>7.2f}  "
              f"tier={r['tier']}  wins={r['wins']:>4d}  losses={r['losses']:>4d}")
    OUTPUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {OUTPUT}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
