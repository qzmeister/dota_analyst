"""Stage 2 data collector: fetch additional OpenDota pro matches
to bring the corpus from 3403 to 5000-10000.

Walks OpenDota /proMatches (paginated) and /matches/{id} for each
match.  Re-uses the existing v17_match_*.json format so the
trainer can ingest without changes.

Saves to ml_data/imports/v17_match_{id}.json.  Skips matches
that already exist on disk (idempotent).

Rate-limit: 1.1s per call to be polite to OpenDota.  5000 matches
= 92 minutes at this rate.

Can be safely interrupted and resumed -- the script tracks which
match IDs have been processed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

USER_AGENT = "dota-analyst/0.7.1 (research; +https://github.com/qzmeister/dota_analyst)"
OPENDOTA = "https://api.opendota.com/api"
PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"

# OpenDota asks for <= 1 req/sec on the public API; we sleep a
# little longer to be polite.
RATE_SLEEP_SEC = float(os.environ.get("V17_RATE_SLEEP", "1.1"))

# How many /proMatches pages to walk.  Each page is 100 matches.
# 30 pages = 3000 more matches (cumulative 6403).
PAGES_TO_FETCH = int(os.environ.get("V17_PAGES", "30"))


def _http_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Single-shot GET with retries; returns parsed JSON or None."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                if not body:
                    return None
                return json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = RATE_SLEEP_SEC * (2 ** attempt)
            print(f"  retry {attempt+1}: {type(exc).__name__} {str(exc)[:80]} (sleep {wait:.1f}s)", file=sys.stderr)
            time.sleep(wait)
    print(f"  GIVE UP: {url} -> {type(last).__name__ if last else '?'}", file=sys.stderr)
    return None


def list_existing_match_ids() -> Set[int]:
    """Return the set of match_ids we already have on disk."""
    out: Set[int] = set()
    for p in IMPORTS.glob("v17_match_*.json"):
        try:
            mid = int(p.stem.replace("v17_match_", ""))
            out.add(mid)
        except ValueError:
            continue
    return out


def fetch_pro_matches_pages(n_pages: int) -> List[Dict[str, Any]]:
    """Fetch the most recent N pages of OpenDota /proMatches."""
    out: List[Dict[str, Any]] = []
    for page in range(n_pages):
        url = f"{OPENDOTA}/proMatches"
        params = {"less_than_match_id": 10**10 - 1 - page * 100}
        # OpenDota doesn't accept pagination directly; the
        # trick is to use `less_than_match_id` to walk back in
        # time.  The endpoint returns 100 most-recent matches
        # with match_id < less_than_match_id.
        print(f"  [page {page+1}/{n_pages}] fetching < {10**10 - 1 - page * 100}...", file=sys.stderr)
        data = _http_json(url, params=params if page > 0 else None)
        if not isinstance(data, list):
            print(f"  page {page+1}: got non-list, stopping", file=sys.stderr)
            break
        out.extend(data)
        if len(data) < 100:
            print(f"  page {page+1}: only {len(data)} matches (end of feed)", file=sys.stderr)
            break
        time.sleep(RATE_SLEEP_SEC)
    return out


def fetch_and_save_match(mid: int) -> bool:
    """Fetch /matches/{id} and save to v17_match_{id}.json.
    Returns True if saved, False if skipped or failed."""
    target = IMPORTS / f"v17_match_{mid}.json"
    if target.exists():
        return False
    raw = _http_json(f"{OPENDOTA}/matches/{mid}")
    if not isinstance(raw, dict):
        return False
    if raw.get("game_mode") not in (1, 2, 3, 4, 5, 12, 22):
        return False
    if not raw.get("radiant_team_id") or not raw.get("dire_team_id"):
        return False
    target.write_text(json.dumps(raw), encoding="utf-8")
    return True


def main() -> int:
    print("=" * 78)
    print(f"Stage 2 data collector -- fetching {PAGES_TO_FETCH} pages of /proMatches")
    print("=" * 78)
    print()

    print("Step 1: enumerate existing matches on disk")
    existing = list_existing_match_ids()
    print(f"  already have {len(existing)} matches in ml_data/imports/")
    print()

    print("Step 2: fetch /proMatches pages")
    pro_matches = fetch_pro_matches_pages(PAGES_TO_FETCH)
    print(f"  -> {len(pro_matches)} pro matches seen")
    print()

    new_ids = []
    for m in pro_matches:
        mid = m.get("match_id")
        if mid is None:
            continue
        if mid in existing:
            continue
        new_ids.append(mid)
    print(f"  {len(new_ids)} new match_ids to fetch (not on disk)")
    print()

    print("Step 3: fetch /matches/{id} for each new id")
    saved = 0
    failed = 0
    t0 = time.time()
    for i, mid in enumerate(new_ids, 1):
        if fetch_and_save_match(mid):
            saved += 1
        else:
            failed += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(new_ids) - i)
            print(f"  [{i}/{len(new_ids)}]  saved={saved}  failed={failed}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m", file=sys.stderr)
        time.sleep(RATE_SLEEP_SEC)
    print()
    print(f"  DONE: {saved} saved, {failed} failed/skipped")
    print(f"  total matches on disk now: {len(existing) + saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
