"""Stage 2 data collector v2 -- use OpenDota /explorer (the
/proMatches endpoint times out from this network; /explorer is
faster anyway because it returns a SQL-shaped result).

Walks matches in a date range we don't have on disk, then
fetches /matches/{id} for each new one.
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

# How far back to look.  We have data through 2026-07; we want
# matches from 2025-01 to 2026-04 (extra 9 months of pro games
# for the older eras).  The first run is conservatively scoped
# to ~3 months to stay under the 30-min PowerShell limit.
START_TIME = int(os.environ.get("V18_S2_START", "1735689600"))   # 2025-01-01 UTC
END_TIME   = int(os.environ.get("V18_S2_END",   "1743465600"))   # 2025-04-01 UTC
BATCH_SIZE = int(os.environ.get("V18_S2_BATCH", "500"))

RATE_SLEEP_SEC = float(os.environ.get("V18_S2_SLEEP", "0.4"))


def _http_json(url: str) -> Any:
    """Single-shot GET with retries; returns parsed JSON or None."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
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
    out: Set[int] = set()
    for p in IMPORTS.glob("v17_match_*.json"):
        try:
            mid = int(p.stem.replace("v17_match_", ""))
            out.add(mid)
        except ValueError:
            continue
    return out


def fetch_match_ids_in_window(start_ts: int, end_ts: int) -> List[int]:
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


def main() -> int:
    print("=" * 78)
    print(f"Stage 2 data collector v2 -- explorer SQL")
    print(f"  window: {START_TIME} ({time.strftime('%Y-%m-%d', time.gmtime(START_TIME))})"
          f" .. {END_TIME} ({time.strftime('%Y-%m-%d', time.gmtime(END_TIME))})")
    print("=" * 78)
    print()

    print("Step 1: enumerate existing matches on disk")
    existing = list_existing_match_ids()
    print(f"  already have {len(existing)} matches")
    print()

    print("Step 2: query /explorer for match_ids in window")
    ids = fetch_match_ids_in_window(START_TIME, END_TIME)
    print(f"  /explorer returned {len(ids)} match_ids")
    new_ids = [i for i in ids if i not in existing]
    print(f"  of which {len(new_ids)} are not on disk")
    print()

    print("Step 3: fetch /matches/{id} for each new id")
    saved = failed = 0
    t0 = time.time()
    n = len(new_ids)
    for i, mid in enumerate(new_ids, 1):
        if fetch_and_save(mid):
            saved += 1
        else:
            failed += 1
        if i % 25 == 0:
            elapsed = time.time() - t0
            rate = i / max(0.1, elapsed)
            eta = (n - i) / max(0.01, rate)
            print(f"  [{i}/{n}]  saved={saved}  failed={failed}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}m", file=sys.stderr)
        time.sleep(RATE_SLEEP_SEC)
    print()
    print(f"  DONE: {saved} saved, {failed} failed/skipped")
    print(f"  total matches on disk now: {len(existing) + saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
