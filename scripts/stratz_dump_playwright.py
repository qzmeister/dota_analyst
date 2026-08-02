"""Stratz portable dump via Playwright (solves Cloudflare JS
challenge automatically, then reuses the session cookies
for the GraphQL POST requests).

Why Playwright?  The user's first round got "Unknown argument
'take'" from Stratz -- meaning the key authenticated and the
request reached Stratz.  The second round got HTML 403 with
"<!DOCTYPE html>...Just a moment..." -- Cloudflare's bot-
challenge page.  Python's urllib doesn't run JavaScript so
it can't solve the challenge.  Playwright Chromium DOES
run JS, so it passes the challenge, then reuses the
session's cf_clearance cookie for the GraphQL API.

Usage (Windows):
    pip install playwright requests
    playwright install chromium
    set STRATZ_API_KEY=<key>
    python stratz_dump_playwright.py teams 200
    python stratz_dump_playwright.py matches 5000

Output: stratz_teams.json / stratz_matches.json in the
parent directory.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PRO_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.stratz.com/graphql"
PASS_THROUGH_PAGES = (
    "https://stratz.com/",
    "https://stratz.com/api",
)
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def _key() -> str:
    k = os.environ.get("STRATZ_API_KEY")
    if not k:
        print("ERROR: STRATZ_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)
    return k


def _warm_up_cookies() -> Dict[str, str]:
    """Open a Playwright Chromium, visit a regular Stratz page
    so the JS challenge runs and Cloudflare issues a
    cf_clearance cookie.  Return that cookie as a dict for
    use in the requests session."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.  pip install playwright && "
              "playwright install chromium", file=sys.stderr)
        sys.exit(1)

    print("  warming up cookies via headless Chromium...")
    cookies: Dict[str, str] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        for url in PASS_THROUGH_PAGES:
            try:
                resp = page.goto(url, timeout=30_000, wait_until="domcontentloaded")
                print(f"  visited {url}: {resp.status if resp else 'no response'}")
                # CF challenge is async; give it a few seconds to
                # mark the session as "verified"
                time.sleep(3)
            except Exception as exc:
                print(f"  {url}: {type(exc).__name__}: {str(exc)[:100]}",
                      file=sys.stderr)
        for c in context.cookies():
            cookies[c["name"]] = c["value"]
        browser.close()
    print(f"  cookies collected: {list(cookies.keys())}")
    return cookies


def _gql_with_cookies(query: str, variables: Optional[Dict[str, Any]],
                       cookies: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """POST the GraphQL query to Stratz with the warmed-up
    Cloudflare cookies plus the user's API key."""
    sess = requests.Session()
    sess.headers.update(HEADERS)
    sess.headers["Authorization"] = f"Bearer {_key()}"
    for k, v in cookies.items():
        sess.cookies.set(k, v)
    body = json.dumps({"query": query, "variables": variables or {}})
    try:
        r = sess.post(ENDPOINT, data=body, timeout=60)
    except Exception as exc:
        print(f"  POST failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None
    try:
        payload = r.json()
    except Exception:
        print(f"  non-JSON response: {r.text[:200]}", file=sys.stderr)
        return None
    if "errors" in payload:
        for e in payload["errors"][:3]:
            print(f"  GQL error: {e.get('message', e)[:200]}", file=sys.stderr)
    return payload.get("data") if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Mode 1: probe to find the right query shape
# --------------------------------------------------------------------------- #

def probe(cookies: Dict[str, str]) -> None:
    """Dump the raw JSON of a few candidate queries so we
    can see which shape Stratz actually returns."""
    candidates = [
        ("leagues(take:1) sanity", '{ leagues(take: 1) { id name } }'),
        ("teams(request: take+isPro)",
         '{ teams(request: { take: 5, isPro: true }) { id name tag rating wins losses } }'),
        ("teams(request) with nodes",
         '{ teams(request: { take: 5, isPro: true }) { nodes { id name rating } } }'),
        ("matches(request) with nodes",
         '{ matches(request: { take: 3 }) { nodes { id startDateTime duration leagueId patch radiantTeamId direTeamId radiantWin } } }'),
    ]
    for label, q in candidates:
        print(f"--- {label} ---")
        data = _gql_with_cookies(q, None, cookies)
        if data is None:
            print("  (no data, see errors above)")
        else:
            # Pretty-print, but truncate long lists
            s = json.dumps(data, indent=2, default=str)
            if len(s) > 1500:
                s = s[:1500] + f"\n... ({len(s)-1500} more bytes)"
            print(s)
        print()


# --------------------------------------------------------------------------- #
# Mode 2: dump top teams
# --------------------------------------------------------------------------- #

def dump_top_teams(cookies: Dict[str, str], limit: int) -> None:
    """Take ~5x what we need and sort client-side by rating
    (Stratz's `teams` doesn't accept orderBy -- confirmed in
    the user's first run, "Unknown argument 'orderBy' on
    field 'teams' of type 'DotaQuery'.")."""
    take = max(limit * 5, 1000)
    # Try the .nodes pattern first; fall back to direct list
    q_nodes = ('{ teams(request: { take: ' + str(take) +
               ', isPro: true }) { nodes { id name tag rating wins losses lastMatchDateTime } } }')
    data = _gql_with_cookies(q_nodes, None, cookies)
    rows: List[Dict[str, Any]] = []
    if data and "teams" in data and data["teams"]:
        t = data["teams"]
        if isinstance(t, dict) and "nodes" in t:
            rows = [n for n in t["nodes"] if n]
        elif isinstance(t, list):
            rows = t
    if not rows:
        # Try the simpler shape
        q_simple = ('{ teams(request: { take: ' + str(take) +
                    ', isPro: true }) { id name tag rating wins losses lastMatchDateTime } }')
        data = _gql_with_cookies(q_simple, None, cookies)
        if data and "teams" in data:
            t = data["teams"]
            if isinstance(t, list):
                rows = t
            elif isinstance(t, dict):
                rows = [t]
    if not rows:
        print("  no teams data -- run `probe` to see the actual shape")
        return
    rows = [r for r in rows if r and r.get("rating") is not None]
    rows.sort(key=lambda r: -(r.get("rating") or 0))
    rows = rows[:limit]
    out = PRO_ROOT / "stratz_teams.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  saved {len(rows)} teams to {out}")
    for r in rows[:5]:
        print(f"    {r.get('id'):>10d}  rating={r.get('rating'):>7.1f}  "
              f"{r.get('name'):>25s}  ({r.get('tag')})  "
              f"wins={r.get('wins')}  losses={r.get('losses')}")


# --------------------------------------------------------------------------- #
# Mode 3: dump recent pro matches
# --------------------------------------------------------------------------- #

def fetch_match_ids(cookies: Dict[str, str], take: int,
                     skip: int) -> List[int]:
    q = ('{ matches(request: { take: ' + str(take) + ', skip: ' +
         str(skip) + ' }) { nodes { id startDateTime } } }')
    data = _gql_with_cookies(q, None, cookies)
    if not data or "matches" not in data:
        return []
    m = data["matches"]
    if isinstance(m, dict) and "nodes" in m:
        return [int(x["id"]) for x in m["nodes"] if x and x.get("id") is not None]
    if isinstance(m, list):
        return [int(x["id"]) for x in m if x and x.get("id") is not None]
    return []


def fetch_match_full(cookies: Dict[str, str], match_id: int) -> Optional[Dict[str, Any]]:
    q = ('{ match(id: ' + str(match_id) + ') { '
         'id startDateTime duration leagueId patch '
         'radiantTeamId direTeamId radiantWin '
         'players { heroId isRadiant } '
         'pickBans { isPick isRadiant heroId order } '
         '} }')
    data = _gql_with_cookies(q, None, cookies)
    if not data or "match" not in data:
        return None
    return data["match"]


def dump_matches(cookies: Dict[str, str], target: int) -> None:
    print(f"  collecting match ids (target {target})...")
    ids: List[int] = []
    for skip in range(0, target, 200):
        page = fetch_match_ids(cookies, min(200, target - skip), skip)
        if not page:
            print(f"  no ids at skip={skip}, stopping")
            break
        ids.extend(page)
        time.sleep(0.5)
        if len(ids) >= target:
            break
    ids = ids[:target]
    print(f"  got {len(ids)} match ids")
    print(f"  collecting full payloads...")
    rows: List[Dict[str, Any]] = []
    for i, mid in enumerate(ids, 1):
        m = fetch_match_full(cookies, mid)
        if m:
            rows.append(m)
        if i % 25 == 0:
            print(f"  [{i}/{len(ids)}]  saved={len(rows)}", file=sys.stderr)
        time.sleep(0.3)
    out = PRO_ROOT / "stratz_matches.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  saved {len(rows)} matches to {out}")
    print(f"  file size: {out.stat().st_size / 1024 / 1024:.1f} MB")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    if not _key():
        return 1
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    mode = sys.argv[1].lower()
    cookies = _warm_up_cookies()
    if mode == "probe":
        probe(cookies)
    elif mode == "teams":
        dump_top_teams(cookies, int(sys.argv[2]))
    elif mode == "matches":
        dump_matches(cookies, int(sys.argv[2]))
    else:
        print(f"unknown mode: {mode}.  Use 'probe', 'teams', or 'matches'.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
