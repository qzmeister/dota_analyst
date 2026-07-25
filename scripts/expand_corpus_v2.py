"""
Expand the ML corpus by fetching matches from DatDota tier-1 leagues.

Strategy:
  1. List all tier-1 leagues from `/api/leagues?limit=2000`.
  2. For each league, get its match list from `/api/matches?leagueId=X`.
  3. For each match ID not already in `full_matches/`, fetch full
     details from `/api/matches/{id}` and save as `{id}.json`.

DatDota has no strict rate limit, but we sleep between calls
to be polite.  Set --max to control runtime.

Run with:
    python scripts/expand_corpus_v2.py --max 1500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.datdota_client import _http_json, get_match_details


def fetch_tier1_leagues() -> List[Dict]:
    """Return all tier-1 leagues from DatDota's /api/leagues."""
    r = _http_json('https://api.datdota.com/api/leagues?limit=2000')
    if not isinstance(r, dict) or 'data' not in r:
        return []
    return [L for L in r['data'] if L.get('tier', {}).get('id') == 1]


def fetch_league_matches(league_id: int) -> List[Dict]:
    """Return all match summaries for a league."""
    r = _http_json(f'https://api.datdota.com/api/matches?leagueId={league_id}&limit=500')
    if not isinstance(r, dict) or 'data' not in r:
        return []
    return r['data']


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="ml_data/full_matches")
    p.add_argument("--max", type=int, default=1500)
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--leagues-limit", type=int, default=200)
    args = p.parse_args()

    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: fetching tier-1 leagues...")
    leagues = fetch_tier1_leagues()
    print(f"  {len(leagues)} tier-1 leagues (using up to {args.leagues_limit})")
    leagues = leagues[: args.leagues_limit]

    print("Step 2: collecting match IDs from each league...")
    existing = {int(p.stem) for p in out_dir.glob("*.json")}
    print(f"  {len(existing)} matches already in {out_dir.name}")

    all_target_ids: Set[int] = set()
    for i, L in enumerate(leagues, 1):
        try:
            ms = fetch_league_matches(L['leagueId'])
        except Exception as exc:
            print(f"  [{i}/{len(leagues)}] league {L['leagueId']}: HTTP error {exc}")
            time.sleep(args.delay)
            continue
        for m in ms:
            mid = int(m.get('matchId', 0))
            if mid and mid not in existing:
                all_target_ids.add(mid)
        if i % 20 == 0:
            print(f"  [{i}/{len(leagues)}]  collected={len(all_target_ids)} unique new IDs")
        time.sleep(args.delay * 0.3)  # less delay between list calls

    targets = sorted(all_target_ids)[: args.max]
    print(f"\n  total new IDs to fetch: {len(targets)} (capped at {args.max})")

    if args.dry_run:
        print("  DRY RUN — not actually fetching")
        for tid in targets[:10]:
            print(f"  would fetch {tid}")
        return 0

    print("\nStep 3: fetching full match details...")
    n_ok = 0
    n_err = 0
    t0 = time.time()
    for i, mid in enumerate(targets, 1):
        try:
            r = get_match_details(mid)
        except Exception as exc:
            print(f"  [{i}/{len(targets)}] {mid}: HTTP error {exc}")
            n_err += 1
            time.sleep(args.delay)
            continue
        if r is None or "data" not in r:
            n_err += 1
            time.sleep(args.delay)
            continue
        data = r["data"]
        if not isinstance(data, dict) or data.get("has_error"):
            n_err += 1
            time.sleep(args.delay)
            continue
        if "radiant_victory" not in data:
            n_err += 1
            time.sleep(args.delay)
            continue
        out_path = out_dir / f"{mid}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n_ok += 1
        if i % 25 == 0:
            elapsed = time.time() - t0
            rate = n_ok / elapsed if elapsed > 0 else 0
            eta = (len(targets) - n_ok) / rate if rate > 0 else 0
            print(f"  [{i}/{len(targets)}] ok={n_ok} err={n_err}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s")
        time.sleep(args.delay)

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.0f}s.  saved={n_ok}  err={n_err}")
    print(f"Total in {out_dir.name}: {len(existing) + n_ok} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
