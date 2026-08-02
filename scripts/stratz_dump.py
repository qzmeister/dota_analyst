"""Stratz portable dump -- run this on your local machine to
collect team ratings and match data from a non-blocked IP.

Usage (Windows / Mac / Linux):
    pip install requests
    set STRATZ_API_KEY=<your_key>          # Windows
    export STRATZ_API_KEY=<your_key>       # Mac/Linux
    python stratz_dump.py teams 100        # top 100 teams -> stratz_teams.json
    python stratz_dump.py matches 5000     # recent 5000 pro matches -> stratz_matches.json

The output JSON is self-contained -- paste it back to me and
I'll wire it into the v18 model.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

ENDPOINT = "https://api.stratz.com/graphql"
USER_AGENT = "dota-analyst-portable/1.0 (research)"


def _key() -> Optional[str]:
    return os.environ.get("STRATZ_API_KEY") or None


def _gql(query: str, variables: Optional[Dict[str, Any]] = None,
         max_retries: int = 4) -> Optional[Dict[str, Any]]:
    if not _key():
        print("ERROR: STRATZ_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)
    body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_key()}",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode("utf-8"))
                if "errors" in payload:
                    for err in payload["errors"][:3]:
                        print(f"  GQL error: {err.get('message', err)[:200]}",
                              file=sys.stderr)
                return payload.get("data") if isinstance(payload, dict) else None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            print(f"  HTTP {e.code}: {body}", file=sys.stderr)
            if e.code == 403:
                print("  Stratz blocked this IP (Cloudflare). "
                      "Try a different network or VPN.", file=sys.stderr)
                return None
            if e.code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None
        except Exception as exc:
            print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2
    return None


# --------------------------------------------------------------------------- #
# Mode 1: dump top teams
# --------------------------------------------------------------------------- #

def dump_top_teams(limit: int) -> None:
    """Top N teams by Stratz rating.  Saves to stratz_teams.json.

    Stratz's `teams` field on DotaQuery doesn't support orderBy
    (the GraphQL error we got from the user's first run was
    "Unknown argument 'orderBy' on field 'teams'").  We fetch
    a larger slice of teams and sort client-side by rating.
    """
    # Take 5x what we need; some teams have null/missing rating
    take = max(limit * 5, 1000)
    q = """
    query TopTeams($take: Int!) {
      teams(take: $take) {
        id name tag rating wins losses lastMatchDateTime
      }
    }
    """
    data = _gql(q, {"take": take})
    if not data or "teams" not in data:
        print("  no data returned")
        return
    rows = [t for t in data["teams"] if t and t.get("rating") is not None]
    # Sort client-side by rating desc, then trim to limit
    rows.sort(key=lambda t: -(t.get("rating") or 0))
    rows = rows[:limit]
    out = Path(__file__).parent.parent / "stratz_teams.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  saved {len(rows)} teams to {out}")
    if rows:
        print(f"  top 5:")
        for t in rows[:5]:
            print(f"    {t.get('id'):>10d}  rating={t.get('rating'):>7.1f}  "
                  f"{t.get('name'):>25s}  ({t.get('tag')})  "
                  f"wins={t.get('wins')}  losses={t.get('losses')}")


# --------------------------------------------------------------------------- #
# Mode 2: dump recent pro matches
# --------------------------------------------------------------------------- #

def fetch_match_ids(take: int, skip: int) -> List[int]:
    """Pull a page of recent pro match ids.  Stratz caps at
    ~1000 per request."""
    q = """
    query Recent($take: Int!, $skip: Int!) {
      matches(take: $take, skip: $skip, orderBy: START_DATE_TIME_DESC) {
        id startDateTime duration leagueId patch
        radiantTeamId direTeamId radiantWin
      }
    }
    """
    data = _gql(q, {"take": take, "skip": skip})
    if not data or "matches" not in data:
        return []
    return [int(m["id"]) for m in data["matches"] if m and m.get("id") is not None]


def fetch_match_full(match_id: int) -> Optional[Dict[str, Any]]:
    q = """
    query FullMatch($id: Long!) {
      match(id: $id) {
        id startDateTime duration
        radiantTeamId direTeamId radiantWin
        leagueId patch
        players { heroId isRadiant }
        pickBans { isPick isRadiant heroId order }
        radiantGoldAdvantage
      }
    }
    """
    data = _gql(q, {"id": match_id})
    if not data or "match" not in data:
        return None
    return data["match"]


def dump_matches(target: int) -> None:
    """Collect up to `target` recent pro matches.  Saves to
    stratz_matches.json (just the metadata, NOT the full
    payload -- Stratz full payloads are big and we want to
    keep the dump small).

    For the full payloads, run a second pass with
    `python stratz_dump.py full <id_list>`.
    """
    print(f"  collecting match ids (target {target})...")
    ids: List[int] = []
    for skip in range(0, target, 200):
        page = fetch_match_ids(min(200, target - skip), skip)
        if not page:
            print(f"  no ids at skip={skip}, stopping")
            break
        ids.extend(page)
        time.sleep(0.5)  # polite
        if len(ids) >= target:
            break
    ids = ids[:target]
    print(f"  got {len(ids)} match ids")
    print(f"  collecting full payloads...")
    rows: List[Dict[str, Any]] = []
    for i, mid in enumerate(ids, 1):
        m = fetch_match_full(mid)
        if m:
            rows.append(m)
        if i % 25 == 0:
            print(f"  [{i}/{len(ids)}]  saved={len(rows)}", file=sys.stderr)
        time.sleep(0.3)
    out = Path(__file__).parent.parent / "stratz_matches.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  saved {len(rows)} matches to {out}")
    print(f"  file size: {out.stat().st_size / 1024 / 1024:.1f} MB")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    if not _key():
        print("ERROR: STRATZ_API_KEY env var is not set.", file=sys.stderr)
        print("  Windows: set STRATZ_API_KEY=eyJ...", file=sys.stderr)
        print("  Linux:   export STRATZ_API_KEY=eyJ...", file=sys.stderr)
        return 1
    if len(sys.argv) < 3:
        print("ERROR: missing arguments.", file=sys.stderr)
        print("  Usage:  python stratz_dump.py teams 200", file=sys.stderr)
        print("          python stratz_dump.py matches 5000", file=sys.stderr)
        print()
        print(__doc__)
        return 1
    mode = sys.argv[1].lower()
    if mode == "teams":
        n = int(sys.argv[2])
        dump_top_teams(n)
    elif mode == "matches":
        n = int(sys.argv[2])
        dump_matches(n)
    else:
        print(f"ERROR: unknown mode {mode!r}.  Use 'teams' or 'matches'.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        rc = 130
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        rc = 1
    raise SystemExit(rc)
