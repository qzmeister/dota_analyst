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
    python stratz_dump_playwright.py probe
    python stratz_dump_playwright.py teams 540
    python stratz_dump_playwright.py expand 200
    python stratz_dump_playwright.py matches 5000

Output (next to the script, in C:\\Users\\artka\\Downloads when
run from there):
    stratz_schema_discovery.json   (probe: full __type output)
    stratz_teams.json              (teams: winCount/lossCount per seed)
    stratz_teams_expanded.json     (expand: NEW teams via leagues)
    stratz_matches.json            (matches: full payload per id)

Schema findings (v0.7.20) and pipeline plan are in
docs/STRATZ_SCHEMA_NOTES.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
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

# v0.7.23: Stratz free tier caps `teams(teamIds: [...])` at 5 IDs
# per request.  50/100 chunked queries return
#   "You have surpassed the maximum take value of : 5"
# v0.7.25: 540 teams / 5 per chunk = 108 chunks.  At 0.5s
# sleep that's 54s of sleep + ~30-90s request time + ~30s
# warmup = 114-174s -- fits under the 180s PowerShell default
# timeout.  _post_raw_with_retry() (v0.7.24) handles the few
# 429s that come from creeping over the 30 req/min limit.
TEAM_CHUNK_SIZE = 5
TEAM_CHUNK_SLEEP_SEC = 0.5


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
    """Schema discovery via __type introspection.

    Phase 1 (v0.7.17): TeamType / MatchType / DotaQuery
    Phase 2 (v0.7.20): LeagueRequestType input shape,
                        leaderboard structure, sample leagues
                        call -- we need to know how to find
                        teamIds we don't have in our seed.

    Each result is both printed AND saved to
    stratz_schema_discovery.json so we have a permanent record
    (network is flaky, output truncation is risky)."""
    queries = [
        # ---- Phase 1: what we already know works ----
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
        # ---- Phase 2: team discovery path ----
        ("LeagueRequestType input shape (for leagues() filter)",
         '{ __type(name: "LeagueRequestType") { inputFields { '
         'name type { name kind ofType { name kind ofType { name } } } } } }'),
        ("LeagueType fields (what's on a league object)",
         '{ __type(name: "LeagueType") { fields { name type { name kind ofType { name } } } } }'),
        ("leaderboard { __typename } (what does it return?)",
         '{ leaderboard { __typename } }'),
        ("leagues(request: { take: 5 }) { id name tier } (sample)",
         '{ leagues(request: { take: 5 }) { id name tier } }'),
        ("teams(teamIds:[9572001]) with win/loss/iso/etc fields",
         '{ teams(teamIds: [9572001]) { '
         'id name tag isPro isLocked countryCode countryName '
         'winCount lossCount lastMatchDateTime dateCreated '
         '} }'),
    ]
    all_results: Dict[str, Any] = {}
    for label, q in queries:
        print(f"--- {label} ---")
        r = _post_raw(q, cookies)
        print(f"  HTTP {r.status_code}")
        try:
            payload = r.json()
            all_results[label] = payload
            if "errors" in payload:
                print("  ERRORS:")
                for e in payload["errors"][:3]:
                    print(f"    {e.get('message', '?')[:300]}")
            else:
                s = json.dumps(payload, indent=2, default=str)
                # v0.7.20: increased truncation limit from 3000 to
                # 50000 bytes so we can read most nested types
                # without losing structure.  Full result is still
                # saved to JSON file below.
                if len(s) > 50000:
                    s = s[:50000] + f"\n... ({len(s)-50000} more bytes)"
                print(f"  {s}")
        except Exception:
            print(f"  non-JSON: {r.text[:300]}")
            all_results[label] = {"_raw": r.text[:1000]}
        print()
    # v0.7.20: persist full output (no truncation) so we
    # have a permanent record even if the network drops.
    # Save next to the script (Downloads dir when run from
    # there) rather than PRO_ROOT, which would land in
    # C:\Users\artka\ when this script lives in Downloads.
    out = Path(__file__).resolve().parent / "stratz_schema_discovery.json"
    out.write_text(json.dumps(all_results, indent=2, default=str),
                   encoding="utf-8")
    print(f"saved full output to {out}")
    print(f"file size: {out.stat().st_size / 1024:.1f} KB")


def _load_existing_team_ids() -> Optional[List[int]]:
    """Load team_ids from our v18_top_teams.json so we can
    seed the `teams(teamIds: ...)` query.  Stratz's `teams`
    field is gated by `teamIds: [Int]!` -- we can't query
    'all teams' or 'top teams', we have to provide IDs.

    v0.7.22: search multiple candidate locations because
    the script can live in Downloads (PRO_ROOT = C:\\Users\\artka)
    OR in the project's scripts/ dir (PRO_ROOT = project root).
    """
    # v0.7.22: hardcoded fallback for the project root, since
    # when this script is in Downloads, PRO_ROOT is the user's
    # home dir, not the project.
    DEFAULT_PROJECT_ROOT = Path(
        r"C:\Users\artka\.minimax\workspace\dota_analyst"
    )
    candidates = []
    # 1) env var override (DOTA_ANALYST_HOME or DOTA_ANALYST_ROOT)
    env = os.environ.get("DOTA_ANALYST_HOME") or os.environ.get("DOTA_ANALYST_ROOT")
    if env:
        candidates.append(Path(env) / "ml_data" / "imports" / "v18_top_teams.json")
    # 2) PRO_ROOT (works when script lives in project/scripts/)
    candidates.append(PRO_ROOT / "ml_data" / "imports" / "v18_top_teams.json")
    # 3) hardcoded default project root
    candidates.append(
        DEFAULT_PROJECT_ROOT / "ml_data" / "imports" / "v18_top_teams.json"
    )
    # 4) CWD-relative (works when run from project root)
    candidates.append(
        Path.cwd() / "ml_data" / "imports" / "v18_top_teams.json"
    )
    # 5) parent of CWD (works when run from project/scripts/)
    candidates.append(
        Path.cwd().parent / "ml_data" / "imports" / "v18_top_teams.json"
    )
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                ids = [int(t["team_id"]) for t in data
                       if t.get("team_id") is not None]
                print(f"  seed: {p}  ({len(ids)} teams)")
                return ids
            except Exception as exc:
                print(f"  seed: {p}  parse failed: {exc}", file=sys.stderr)
    print(f"  seed: no v18_top_teams.json found.  tried:", file=sys.stderr)
    for p in candidates:
        print(f"    - {p}", file=sys.stderr)
    return None


def dump_top_teams(cookies: Dict[str, str], limit: int) -> None:
    """v0.7.21: enrich our 540 seed teams with Stratz's
    winCount / lossCount / lastMatchDateTime.

    v0.7.20 __type introspection confirmed the TeamType field
    list (see docs/STRATZ_SCHEMA_NOTES.md):
      - id, name, tag, isPro, isLocked, countryCode, countryName
      - winCount, lossCount, lastMatchDateTime, dateCreated

    No `rating` field exists -- Stratz's skill proxy is just
    win_count / (win_count + loss_count), plus recency filter on
    lastMatchDateTime.

    Output: stratz_teams.json next to the script, with
    `win_rate` and `last_match_iso` pre-computed for v18 to
    consume directly.

    For each chunk of 50 teamIds we send a single GraphQL query.
    With 540 teams we need 11 queries at ~0.5s each = ~6s total.
    """
    team_ids = _load_existing_team_ids()
    if not team_ids:
        print("  no v18_top_teams.json found -- need team IDs to seed.",
              "Either run scripts/compute_team_ratings.py first,",
              "or run the `expand` mode after fixing network access.")
        return
    team_ids = team_ids[:limit]
    print(f"  loaded {len(team_ids)} seed team_ids from v18_top_teams.json")
    print(f"  chunking: {TEAM_CHUNK_SIZE} ids/req, "
          f"{TEAM_CHUNK_SLEEP_SEC}s sleep "
          f"≈ {TEAM_CHUNK_SIZE/TEAM_CHUNK_SLEEP_SEC:.1f} ids/s")
    # v0.7.21: full field set.  All confirmed present by probe.
    # `coachSteamAccountId` is skipped -- not useful for tier rating.
    fields = (
        "id name tag isPro isLocked countryCode countryName "
        "winCount lossCount lastMatchDateTime dateCreated"
    )
    all_rows: List[Dict[str, Any]] = []
    failed_chunks = 0
    total_chunks = (len(team_ids) + TEAM_CHUNK_SIZE - 1) // TEAM_CHUNK_SIZE
    for chunk_idx, chunk_start in enumerate(range(0, len(team_ids),
                                                   TEAM_CHUNK_SIZE)):
        chunk = team_ids[chunk_start:chunk_start + TEAM_CHUNK_SIZE]
        ids_csv = ", ".join(str(i) for i in chunk)
        q = "{ teams(teamIds: [" + ids_csv + "]) { " + fields + " } }"
        r = _post_raw_with_retry(q, cookies)
        if r.status_code == 200:
            try:
                payload = r.json()
                if "errors" in payload and payload["errors"]:
                    err_msgs = [e.get("message", "?")[:100]
                                for e in payload["errors"][:2]]
                    # 429 / rate-limit / chunk-too-big: print
                    # full body once for diagnosis, then bail
                    # if the same error repeats.
                    print(f"  chunk {chunk_idx}/{total_chunks}: GQL errors: "
                          f"{err_msgs}")
                    if ("surpassed the maximum take" in err_msgs[0]
                            or "rate limit" in err_msgs[0].lower()):
                        print(f"  -> Stratz hard limit hit. Aborting this "
                              f"chunk; adjust TEAM_CHUNK_SIZE.")
                    failed_chunks += 1
                rows = (payload.get("data") or {}).get("teams") or []
                all_rows.extend([x for x in rows if x])
            except Exception as exc:
                print(f"  chunk {chunk_idx}/{total_chunks}: parse failed: {exc}")
                failed_chunks += 1
        else:
            print(f"  chunk {chunk_idx}/{total_chunks}: HTTP {r.status_code}")
            failed_chunks += 1
        if chunk_idx % 20 == 0 and chunk_idx > 0:
            print(f"    [{chunk_start + len(chunk)}/{len(team_ids)}]  "
                  f"saved={len(all_rows)}")
        time.sleep(TEAM_CHUNK_SLEEP_SEC)
    if not all_rows:
        print("  no team data returned; run `probe` to see why")
        return
    # v0.7.21: compute win_rate and last_match_iso for v18
    # consumption.  Filter out teams with 0 games.
    enriched: List[Dict[str, Any]] = []
    for row in all_rows:
        wc = row.get("winCount") or 0
        lc = row.get("lossCount") or 0
        if wc + lc == 0:
            continue
        lmd = row.get("lastMatchDateTime")
        row["win_rate"] = round(wc / (wc + lc), 4)
        row["games_total"] = wc + lc
        if lmd:
            try:
                # v0.7.26: use timezone-aware datetime to avoid
                # DeprecationWarning on datetime.utcfromtimestamp()
                # in Python 3.12+.  Output format is identical
                # (ISO 8601 with trailing 'Z').
                row["last_match_iso"] = (
                    datetime.fromtimestamp(int(lmd), timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            except Exception:
                row["last_match_iso"] = None
        else:
            row["last_match_iso"] = None
        enriched.append(row)
    # Save next to the script (Downloads dir) so the user can
    # easily copy it into the project later.
    out = Path(__file__).resolve().parent / "stratz_teams.json"
    out.write_text(json.dumps(enriched, indent=2, default=str),
                   encoding="utf-8")
    print(f"  saved {len(enriched)} teams to {out}")
    print(f"  file size: {out.stat().st_size / 1024:.1f} KB")
    if failed_chunks:
        print(f"  WARNING: {failed_chunks} chunks failed; "
              f"may have fewer teams than expected")
    # Show top 5 by win_rate (min 30 games to filter noise)
    top = [r for r in enriched if r["games_total"] >= 30]
    top.sort(key=lambda r: -r["win_rate"])
    print(f"  top 5 by win_rate (>=30 games):")
    for r in top[:5]:
        last = r.get("last_match_iso") or "?"
        print(f"    {r.get('id'):>10d}  win_rate={r['win_rate']:.3f}  "
              f"games={r['games_total']:>5d}  last={last}  "
              f"{r.get('name', '?')[:25]}")
    # Bottom 5
    bot = sorted(top, key=lambda r: r["win_rate"])
    print(f"  bottom 5 by win_rate (>=30 games):")
    for r in bot[:5]:
        last = r.get("last_match_iso") or "?"
        print(f"    {r.get('id'):>10d}  win_rate={r['win_rate']:.3f}  "
              f"games={r['games_total']:>5d}  last={last}  "
              f"{r.get('name', '?')[:25]}")


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


def _post_raw_with_retry(query: str, cookies: Dict[str, str],
                          max_retries: int = 4) -> requests.Response:
    """v0.7.24: same as _post_raw but retries on 429 / rate
    limit with exponential backoff (1s -> 2s -> 4s -> 8s).

    The Stratz free tier is 30 req/min.  540 teams / 5 per
    chunk = 108 chunks at TEAM_CHUNK_SLEEP_SEC = 1.0s already
    pushes us to ~50-60 req/min during the actual dump, so a
    few 429s are likely.  Better to wait and retry than to
    fail the whole run.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = _post_raw(query, cookies)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            backoff = 2 ** attempt
            print(f"    POST exception: {type(exc).__name__}; "
                  f"backoff {backoff}s (attempt {attempt+1}/{max_retries})",
                  file=sys.stderr)
            time.sleep(backoff)
            continue
        if r.status_code == 429:
            if attempt == max_retries:
                return r
            backoff = 2 ** attempt
            print(f"    429 rate limit; backoff {backoff}s "
                  f"(attempt {attempt+1}/{max_retries})", file=sys.stderr)
            time.sleep(backoff)
            continue
        return r
    if last_exc:
        raise last_exc
    # unreachable
    return _post_raw(query, cookies)


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
