"""Polite OpenDota /matches collector with exponential backoff.

Earlier collect scripts hammered OpenDota with no backoff, so
the public API now returns 429 to our IP for an unknown
duration.  This script:

  - Spreads requests over time (configurable, default 2.5s/call
    so we stay well under the 60 req/min hard limit even with
    a tiny burst tolerance)
  - Exponential backoff on 429: 30s -> 60s -> 120s -> 240s
    (capped at 10 min) before retry
  - Idempotent: skips matches we already have on disk
  - Resumable: tracks last successful start_time in
    ml_data/imports/_v18_s2_progress.json

Expected runtime for 5000 matches: 5000 * 2.5s = 208 minutes
(~3.5 hours).  Slow but works.  If rate limits clear
(typically 1-24h after a hot period), we can drop the delay
to 1.1s and finish in ~90 min.
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

USER_AGENT = "dota-analyst/0.7.4 (research; +https://github.com/qzmeister/dota_analyst)"
OPENDOTA = "https://api.opendota.com/api"
PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"
PROGRESS_FILE = IMPORTS / "_v18_s2_progress.json"

# How many /matches/{id} to fetch in this run
TARGET_NEW = int(os.environ.get("V18_S2_NEW", "3000"))
# Seconds between calls; OpenDota asks for <= 1/s but we use
# 2.5s to be polite and stay clear of the 429 envelope.
SLEEP_BETWEEN = float(os.environ.get("V18_S2_SLEEP", "2.5"))
# Max consecutive failures before giving up
MAX_BACKOFF_SEC = 600  # 10 min


def _http_json(url: str, max_retries: int = 5) -> Optional[Any]:
    """Single-shot GET with exponential backoff on 429."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    backoff = 30.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if not body:
                    return None
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(backoff, MAX_BACKOFF_SEC)
                print(f"  429: backing off {wait}s (attempt {attempt+1})", file=sys.stderr)
                time.sleep(wait)
                backoff *= 2
            elif e.code in (500, 502, 503, 504):
                wait = min(backoff, MAX_BACKOFF_SEC)
                print(f"  {e.code}: backing off {wait}s", file=sys.stderr)
                time.sleep(wait)
                backoff *= 2
            else:
                print(f"  HTTP {e.code}: {e.read()[:200].decode('utf-8', errors='replace')}", file=sys.stderr)
                return None
        except Exception as exc:
            print(f"  {type(exc).__name__}: {str(exc)[:100]}", file=sys.stderr)
            return None
    return None


def list_existing_match_ids() -> Set[int]:
    out: Set[int] = set()
    for p in IMPORTS.glob("v17_match_*.json"):
        try:
            out.add(int(p.stem.replace("v17_match_", "")))
        except ValueError:
            continue
    return out


def fetch_match_ids_in_window(start_ts: int, end_ts: int,
                               cap: int = 5000) -> List[int]:
    """Query /explorer for pro match_ids in [start_ts, end_ts)."""
    sql = (
        "SELECT match_id FROM matches "
        f"WHERE start_time >= {start_ts} AND start_time < {end_ts} "
        "AND leagueid IS NOT NULL "
        "AND radiant_team_id IS NOT NULL "
        "AND dire_team_id IS NOT NULL "
        "ORDER BY start_time DESC LIMIT 5000"
    )
    url = f"{OPENDOTA}/explorer?{urllib.parse.urlencode({'sql': sql})}"
    data = _http_json(url)
    if not isinstance(data, dict):
        return []
    rows = data.get("rows") or []
    return [int(r["match_id"]) for r in rows if r.get("match_id") is not None]


def fetch_and_save(mid: int) -> bool:
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


def load_progress() -> Dict[str, Any]:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(p: Dict[str, Any]) -> None:
    PROGRESS_FILE.write_text(json.dumps(p, indent=2), encoding="utf-8")


def main() -> int:
    print("=" * 78)
    print(f"Polite OpenDota /matches collector")
    print(f"  target_new={TARGET_NEW}  sleep={SLEEP_BETWEEN}s/call")
    print("=" * 78)
    print()

    progress = load_progress()
    print(f"  saved progress: saved={progress.get('saved', 0)}  "
          f"failed={progress.get('failed', 0)}  "
          f"last_start={progress.get('last_start', 'n/a')}")

    existing = list_existing_match_ids()
    print(f"  already have {len(existing)} matches on disk")
    print()

    # Window: last 12 months.  The /explorer SQL is the only
    # endpoint that gives date-range filtering on public.
    now = int(time.time())
    one_year_ago = now - 365 * 86400
    print(f"Step 1: query /explorer for matches in last 365 days")
    ids = fetch_match_ids_in_window(one_year_ago, now)
    print(f"  /explorer returned {len(ids)} match_ids")
    new_ids = [i for i in ids if i not in existing]
    print(f"  of which {len(new_ids)} are not on disk")
    if len(new_ids) > TARGET_NEW:
        new_ids = new_ids[:TARGET_NEW]
    print(f"  -> targeting {len(new_ids)} for this run")
    print()

    print(f"Step 2: fetch /matches/{{id}} (sleep {SLEEP_BETWEEN}s between calls)")
    saved = 0
    failed = 0
    t0 = time.time()
    n = len(new_ids)
    for i, mid in enumerate(new_ids, 1):
        if fetch_and_save(mid):
            saved += 1
        else:
            failed += 1
        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / max(0.1, elapsed)
            eta = (n - i) / max(0.01, rate)
            print(f"  [{i}/{n}]  saved={saved}  failed={failed}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}m", file=sys.stderr)
            save_progress({"saved": saved + progress.get("saved", 0),
                           "failed": failed + progress.get("failed", 0),
                           "last_start": now})
        time.sleep(SLEEP_BETWEEN)
    print()
    print(f"  DONE: {saved} saved, {failed} failed/skipped")
    print(f"  total matches on disk now: {len(existing) + saved}")
    save_progress({"saved": saved + progress.get("saved", 0),
                   "failed": failed + progress.get("failed", 0),
                   "last_start": now,
                   "completed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
