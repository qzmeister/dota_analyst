"""
Expand the ML corpus by fetching unsaved matches from DatDota.

Walks `ml_data/datdota_tier1_matches.json`, finds IDs not yet
present in `ml_data/full_matches/`, fetches each via DatDota's
public `/api/matches/{id}` endpoint, and writes the JSON.

DatDota has no strict rate limit but we sleep 0.5s between calls
to be polite.  With ~900 remaining matches this is ~7-8 minutes.

Run with:
    python scripts/expand_corpus.py --max 300
    python scripts/expand_corpus.py --max 100 --delay 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.datdota_client import get_match_details


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matches-list", default="ml_data/datdota_tier1_matches.json")
    p.add_argument("--out-dir", default="ml_data/full_matches")
    p.add_argument("--max", type=int, default=300,
                   help="max number of NEW matches to fetch (default 300)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="seconds between API calls (default 0.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be fetched, don't actually fetch")
    args = p.parse_args()

    matches_list = _ROOT / args.matches_list
    out_dir = _ROOT / args.out_dir
    if not matches_list.is_file():
        print(f"ERROR: {matches_list} not found")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    with matches_list.open("r", encoding="utf-8") as fh:
        all_matches = json.load(fh)
    print(f"loaded {len(all_matches)} IDs from {matches_list.name}")

    existing = {int(p.stem) for p in out_dir.glob("*.json")}
    print(f"{len(existing)} matches already in {out_dir.name}")

    targets = [m for m in all_matches if int(m["matchId"]) not in existing]
    print(f"{len(targets)} matches to fetch (capped at {args.max})")
    targets = targets[: args.max]

    if args.dry_run:
        for m in targets[:10]:
            print(f"  would fetch {m['matchId']}  ({m.get('tournament_name', '?')})")
        return 0

    n_ok = 0
    n_err = 0
    t0 = time.time()
    for i, m in enumerate(targets, 1):
        mid = int(m["matchId"])
        try:
            r = get_match_details(mid)
        except Exception as exc:
            print(f"  [{i}/{len(targets)}] {mid}: HTTP error {exc}")
            n_err += 1
            time.sleep(args.delay)
            continue
        if r is None or "data" not in r:
            print(f"  [{i}/{len(targets)}] {mid}: no data")
            n_err += 1
            time.sleep(args.delay)
            continue
        data = r["data"]
        # Same format check as iter_clean_targets — keep matches
        # that have radiant_victory, player_performances, etc.
        if not isinstance(data, dict) or data.get("has_error"):
            print(f"  [{i}/{len(targets)}] {mid}: has_error or wrong type")
            n_err += 1
            time.sleep(args.delay)
            continue
        if "radiant_victory" not in data:
            print(f"  [{i}/{len(targets)}] {mid}: no radiant_victory")
            n_err += 1
            time.sleep(args.delay)
            continue
        out_path = out_dir / f"{mid}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n_ok += 1
        if i % 25 == 0:
            elapsed = time.time() - t0
            rate = n_ok / elapsed if elapsed > 0 else 0
            eta = (args.max - n_ok) / rate if rate > 0 else 0
            print(f"  [{i}/{len(targets)}] {mid}: ok  (saved={n_ok} err={n_err}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s)")
        time.sleep(args.delay)

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.0f}s.  saved={n_ok}  err={n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
