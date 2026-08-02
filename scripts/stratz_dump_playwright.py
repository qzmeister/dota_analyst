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
    """v0.7.17 schema discovery: find out what fields
    TeamType and MatchType actually have, and what the
    matches() query shape is.

    v0.7.16 found via error responses that:
      - `teams(teamIds: [Int]!)` is the only valid args
      - topTeams / proTeams don't exist
      - TeamType doesn't have rating/wins/losses

    So we use __type introspection to discover the real
    field names, then query by teamIds."""
    queries = [
        ("TeamType fields (what's on a team object)",
         '{ __type(name: "TeamType") { fields { name type { name kind ofType { name } } } } }'),
        ("MatchType fields (what's on a match object)",
         '{ __type(name: "MatchType") { fields { name type { name kind ofType { name } } } } }'),
        ("DotaQuery root fields + args (top-level queries)",
         '{ __type(name: "DotaQuery") { fields { name args { name } } } }'),
        ("matches(args) -- what args does it take?",
         '{ __type(name: "DotaQuery") { fields(includeDeprecated: true) { '
         'name args { name type { name kind ofType { name } } } } } }'),
        ("teams(teamIds:[9572001]) sanity",
         '{ teams(teamIds: [9572001]) { id name tag } }'),
    ]
    for label, q in queries:
        print(f"--- {label} ---")
        r = _post_raw(q, cookies)
        print(f"  HTTP {r.status_code}")
        try:
            payload = r.json()
            if "errors" in payload:
                print("  ERRORS:")
                for e in payload["errors"][:3]:
                    print(f"    {e.get('message', '?')[:300]}")
            else:
                s = json.dumps(payload, indent=2, default=str)
                if len(s) > 3000:
                    s = s[:3000] + f"\n... ({len(s)-3000} more bytes)"
                print(f"  {s}")
        except Exception:
            print(f"  non-JSON: {r.text[:300]}")
        print()


def _load_existing_team_ids() -> Optional[List[int]]:
    """Load team_ids from our v18_top_teams.json so we can
    seed the `teams(teamIds: ...)` query.  Stratz's `teams`
    field is gated by `teamIds: [Int]!` -- we can't query
    'all teams' or 'top teams', we have to provide IDs."""
    p = PRO_ROOT / "ml_data" / "imports" / "v18_top_teams.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [int(t["team_id"]) for t in data if t.get("team_id") is not None]
    except Exception:
        return None


def dump_top_teams(cookies: Dict[str, str], limit: int) -> None:
    """Query Stratz for teams by their IDs (from our
    v18_top_teams.json).  Stratz doesn't have a 'top teams'
    field; we provide the IDs and sort by whatever rating
    field is actually on TeamType (discovered via probe)."""
    team_ids = _load_existing_team_ids()
    if not team_ids:
        print("  no v18_top_teams.json found -- run probe first",
              "then we need to seed with team_ids from somewhere")
        return
    team_ids = team_ids[:limit]
    all_rows: List[Dict[str, Any]] = []
    for chunk_start in range(0, len(team_ids), 50):
        chunk = team_ids[chunk_start:chunk_start + 50]
        ids_csv = ", ".join(str(i) for i in chunk)
        # Conservative field selection -- only `id name tag`,
        # which we KNOW exist (the user's first successful
        # query used them).  We'll discover other fields via
        # probe and re-query if needed.
        q = '{ teams(teamIds: [' + ids_csv + ']) { id name tag } }'
        r = _post_raw(q, cookies)
        if r.status_code == 200:
            try:
                payload = r.json()
                rows = (payload.get("data") or {}).get("teams") or []
                all_rows.extend([x for x in rows if x])
            except Exception:
                pass
        time.sleep(0.3)
    if not all_rows:
        print("  no team data returned; run `probe` to see why")
        return
    out = PRO_ROOT / "stratz_teams.json"
    out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"  saved {len(all_rows)} teams to {out}")
    for r in all_rows[:5]:
        print(f"    {r.get('id'):>10d}  {r.get('name'):>25s}  ({r.get('tag')})")


# --------------------------------------------------------------------------- #
# Mode 2: dump top teams
# --------------------------------------------------------------------------- #

def _post_raw(query: str, cookies: Dict[str, str]) -> requests.Response:
    """POST a GraphQL query and return the raw response, so
    callers can dump the full JSON when needed."""
    sess = requests.Session()
    sess.headers.update(HEADERS)
    sess.headers["Authorization"] = f"Bearer {_key()}"
    for k, v in cookies.items():
        sess.cookies.set(k, v)
    body = json.dumps({"query": query})
    return sess.post(ENDPOINT, data=body, timeout=60)


def _dump_top_teams_legacy(cookies: Dict[str, str], limit: int) -> None:
    """LEGACY: tries 6 candidate query shapes that all returned
    400.  Kept for reference (and so we can re-test if Stratz's
    schema ever changes).  NOT wired into the CLI -- the real
    path is dump_top_teams() above, which seeds from
    v18_top_teams.json + uses the field list we discover via
    __type introspection in `probe` mode.
    """
    take = max(limit * 5, 1000)
    candidates = [
        ('{ teams(request: { take: ' + str(take) + ', isPro: true }) '
         '{ nodes { id name tag rating wins losses lastMatchDateTime } } }'),
        ('{ teams(request: { take: ' + str(take) + ', isPro: true }) '
         '{ id name tag rating wins losses lastMatchDateTime } }'),
        ('{ topTeams { id name tag rating wins losses } }'),
        ('{ proTeams { id name tag rating wins losses } }'),
        ('{ teams(isPro: true) { id name rating } }'),
    ]
    for q in candidates:
        r = _post_raw(q, cookies)
        print(f"  tried: {q[:80]}...")
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            try:
                payload = r.json()
                data = payload.get("data", {})
                t = data.get("teams") or data.get("topTeams") or data.get("proTeams")
                if t is None:
                    print(f"  200 but no data: {json.dumps(payload)[:300]}")
                    continue
                if isinstance(t, dict) and "nodes" in t:
                    rows = [n for n in t["nodes"] if n]
                elif isinstance(t, list):
                    rows = [n for n in t if n]
                else:
                    rows = [t]
                rows = [r for r in rows if r and r.get("rating") is not None]
                if not rows:
                    print(f"  200 but no rows: {json.dumps(payload)[:300]}")
                    continue
                rows.sort(key=lambda r: -(r.get("rating") or 0))
                rows = rows[:limit]
                out = PRO_ROOT / "stratz_teams.json"
                out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"  saved {len(rows)} teams to {out}")
                for r2 in rows[:5]:
                    print(f"    {r2.get('id'):>10d}  rating={r2.get('rating'):>7.1f}  "
                          f"{r2.get('name'):>25s}  ({r2.get('tag')})  "
                          f"wins={r2.get('wins')}  losses={r2.get('losses')}")
                return
            except Exception as exc:
                print(f"  200 but parse failed: {exc}")
                continue
        else:
            # 400 / etc -- print the full body so the user
            # can see the full error message and the
            # valid argument list (Stratz returns
            # extensions.allowedArgumentNames or similar).
            try:
                payload = r.json()
                print(f"  full error JSON:")
                print("  " + json.dumps(payload, indent=2, default=str)[:1500]
                      .replace("\n", "\n  "))
            except Exception:
                print(f"  body: {r.text[:300]}")
    print()
    print("  no candidate query shape worked.")
    print("  hint: try `python stratz_dump_playwright.py probe`")
    print("        to see all 6 candidate queries and their errors.")


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
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    mode = sys.argv[1].lower()
    cookies = _warm_up_cookies()
    if mode == "probe":
        # `probe` takes no extra arg -- the third argv slot is ignored
        # (kept for backward-compat: `probe 0` still works).
        probe(cookies)
    elif mode == "teams":
        if len(sys.argv) < 3:
            print("ERROR: `teams` mode needs a count, e.g. `teams 200`",
                  file=sys.stderr)
            return 1
        dump_top_teams(cookies, int(sys.argv[2]))
    elif mode == "matches":
        if len(sys.argv) < 3:
            print("ERROR: `matches` mode needs a count, e.g. `matches 5000`",
                  file=sys.stderr)
            return 1
        dump_matches(cookies, int(sys.argv[2]))
    else:
        print(f"unknown mode: {mode}.  Use 'probe', 'teams', or 'matches'.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
