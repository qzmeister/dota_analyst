"""v0.7.26: Compare Stratz win_rate vs our computed tier (v0.7.0).

The Stratz dump from v0.7.21-25 gave us winCount/lossCount/
lastMatchDateTime for our 540 seed teams.  This script asks:
are those numbers actually consistent with our v0.7.0
Bayesian-smoothed Glicko + percentile-based tier?

If yes: Stratz data is a useful cross-check (and v18 can
       blend them for higher-quality tier signal).
If no:  Stratz data is noise, ignore.

Outputs:
  - Coverage stats (% of 540 teams in Stratz)
  - Mean Stratz win_rate per v18 tier (premium / pro / minor)
  - Pearson correlation: tier vs win_rate
  - Top 10 teams with the largest tier-win_rate disagreement
  - Sanity check on last_match recency

Usage:
    python scripts/compare_stratz_vs_v18.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = ROOT / "ml_data" / "imports"

V18_FILE = IMPORTS / "v18_top_teams.json"
STRATZ_FILE = IMPORTS / "stratz_team_ratings.json"


def _load() -> tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Load both files.  Returns (v18_list, stratz_by_id)."""
    v18 = json.loads(V18_FILE.read_text(encoding="utf-8"))
    stratz_raw = json.loads(STRATZ_FILE.read_text(encoding="utf-8"))
    stratz_by_id = {int(r["id"]): r for r in stratz_raw
                    if r.get("id") is not None}
    return v18, stratz_by_id


def _tier_label(t: int) -> str:
    return {0: "minor", 1: "professional", 2: "premium"}.get(t, f"tier{t}")


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation, no numpy."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return 0.0
    return num / (dx2 ** 0.5 * dy2 ** 0.5)


def main() -> int:
    v18, stratz = _load()
    print(f"v18 teams: {len(v18)}")
    print(f"stratz teams (in dump): {len(stratz)}")

    # Join
    joined = []
    missing = []
    for row in v18:
        tid = int(row["team_id"])
        if tid in stratz:
            j = {
                "team_id": tid,
                "tier": row.get("tier", 0),
                "rating": row.get("rating"),
                "v18_wins": row.get("wins", 0),
                "v18_losses": row.get("losses", 0),
                "v18_last_match": row.get("last_match_time"),
                "v18_name": row.get("name"),
                "stratz_win_rate": stratz[tid].get("win_rate"),
                "stratz_games": stratz[tid].get("games_total"),
                "stratz_win_count": stratz[tid].get("winCount"),
                "stratz_loss_count": stratz[tid].get("lossCount"),
                "stratz_last_match": stratz[tid].get("last_match_iso"),
                "stratz_name": stratz[tid].get("name"),
            }
            joined.append(j)
        else:
            missing.append(tid)

    print(f"joined (both v18 and stratz): {len(joined)}")
    print(f"v18 teams NOT in stratz dump: {len(missing)}")
    if missing[:5]:
        print(f"  examples: {missing[:5]}")

    # 1) Coverage
    coverage = len(joined) / max(len(v18), 1)
    print(f"\n=== COVERAGE ===")
    print(f"  {len(joined)}/{len(v18)} = {coverage:.1%}")

    # 2) Stratz win_rate per v18 tier
    print(f"\n=== STRATZ win_rate PER V18 TIER ===")
    by_tier: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    for j in joined:
        by_tier[j["tier"]].append(j["stratz_win_rate"])
    for t in (2, 1, 0):
        rates = by_tier[t]
        if not rates:
            print(f"  tier={_tier_label(t)} (n=0): no data")
            continue
        rates_sorted = sorted(rates)
        n = len(rates)
        median = rates_sorted[n // 2]
        p25 = rates_sorted[n // 4]
        p75 = rates_sorted[(3 * n) // 4]
        mean = sum(rates) / n
        print(f"  tier={_tier_label(t):<13s} n={n:>3d}  "
              f"mean={mean:.3f}  median={median:.3f}  "
              f"p25={p25:.3f}  p75={p75:.3f}  "
              f"min={min(rates):.3f}  max={max(rates):.3f}")

    # 3) Correlation: tier vs win_rate
    tiers = [j["tier"] for j in joined]
    wrs = [j["stratz_win_rate"] for j in joined]
    corr = _pearson(tiers, wrs)
    print(f"\n=== CORRELATION ===")
    print(f"  Pearson(tier, stratz_win_rate) = {corr:+.4f}")
    if corr > 0.5:
        print(f"  -> STRONG POSITIVE: higher tier <-> higher Stratz win_rate. "
              f"Stratz data is a useful cross-check.")
    elif corr > 0.2:
        print(f"  -> WEAK POSITIVE: tier and Stratz win_rate agree, but "
              f"there's meaningful disagreement.  Use as feature, not ground truth.")
    elif corr > -0.2:
        print(f"  -> NEAR ZERO: tier and Stratz win_rate are independent. "
              f"Stratz data adds orthogonal signal (good for stacking).")
    else:
        print(f"  -> NEGATIVE: tier and Stratz win_rate disagree strongly. "
              f"Investigate -- probably one is broken.")

    # 4) Top 10 disagreements: where tier says premium but Stratz says low win_rate
    print(f"\n=== TOP 10 'PROMISED PREMIUM, STRATZ DISAGREES' ===")
    # premium (tier=2) with low win_rate
    promises = [j for j in joined if j["tier"] == 2 and j["stratz_win_rate"] < 0.5]
    promises.sort(key=lambda j: j["stratz_win_rate"])
    for j in promises[:10]:
        name = j["stratz_name"] or f"id={j['team_id']}"
        print(f"  tier=premium  stratz_win_rate={j['stratz_win_rate']:.3f}  "
              f"v18 W/L={j['v18_wins']}/{j['v18_losses']}  "
              f"stratz W/L={j['stratz_win_count']}/{j['stratz_loss_count']}  "
              f"{name[:30]}")

    # 5) Top 10 reverse: tier=minor (0) but Stratz says high win_rate
    print(f"\n=== TOP 10 'LISTED AS MINOR, STRATZ SAYS PREMIUM' ===")
    dark_horses = [j for j in joined
                   if j["tier"] == 0 and j["stratz_win_rate"] > 0.6
                   and (j["stratz_games"] or 0) >= 30]
    dark_horses.sort(key=lambda j: -j["stratz_win_rate"])
    for j in dark_horses[:10]:
        name = j["stratz_name"] or f"id={j['team_id']}"
        print(f"  tier=minor    stratz_win_rate={j['stratz_win_rate']:.3f}  "
              f"v18 W/L={j['v18_wins']}/{j['v18_losses']}  "
              f"stratz W/L={j['stratz_win_count']}/{j['stratz_loss_count']}  "
              f"v18_last={j['v18_last_match']}  "
              f"{name[:30]}")

    # 6) Recency: when was each team's last Stratz match?
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    print(f"\n=== STRATZ LAST-MATCH RECENCY (out of {len(joined)} teams) ===")
    buckets = {"<30d": 0, "30-90d": 0, "90-180d": 0, "180-365d": 0, ">365d": 0, "n/a": 0}
    for j in joined:
        lmd = j["stratz_last_match"]
        if not lmd:
            buckets["n/a"] += 1
            continue
        try:
            ts = int(datetime.fromisoformat(
                lmd.replace("Z", "+00:00")
            ).timestamp())
        except Exception:
            buckets["n/a"] += 1
            continue
        days = (now_ts - ts) / 86400
        if days < 30:
            buckets["<30d"] += 1
        elif days < 90:
            buckets["30-90d"] += 1
        elif days < 180:
            buckets["90-180d"] += 1
        elif days < 365:
            buckets["180-365d"] += 1
        else:
            buckets[">365d"] += 1
    for k, v in buckets.items():
        print(f"  {k:<10s} {v:>4d}  ({v / max(len(joined), 1):.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
