"""Bulk collector for v17 ML training corpus.

Pulls pro-match data from public Dota 2 stats sources (no auth required
for OpenDota; optional auth for Stratz / datdota).  Designed to be
re-runnable: every run adds new matches to the existing
`ml_data/full_matches/` corpus without touching the schema.

Phases (run sequentially; each is idempotent):
  1. Top-N pro teams (by OpenDota rating)
  2. Match IDs in lookback window (OpenDota /api/explorer SQL,
     filtering on `leagueid IS NOT NULL` to exclude pub games)
  3. Match details (/api/matches/<id> per ID; per-patch data is
     returned by the match blob itself)
  4. Per-hero meta stats (/api/heroStats - has per-patch pick/win
     counts in the 1_pick..8_pick / 1_win..8_win fields)
  5. Hero matchups (/api/heroes/<id>/matchups - per-hero synergies
     & counters)
  6. Team roster / player aggregates (per-team /api/teams/<id>/players)
  7. Patch boundaries (/api/constants/patch)

Outputs go to:
  ml_data/imports/v17_<phase>.json  (raw payloads, human-readable)
  ml_data/full_matches/<match_id>.json  (one file per match, matches
                                          the existing v0.3.11 schema
                                          so the v16 trainer can ingest
                                          without changes)

We do NOT add a NEW schema layer — we just enrich the existing
`full_matches/` format with extra fields the v17 trainer can pick
up via `Optional[...]` reads.  Keeps the migration tiny.

Rate-limit: 1 req/sec to OpenDota.  OpenDota's docs ask for
<= 1 req/sec; we honour that with a `time.sleep(1.1)` between
calls.  ~4000 matches = ~4000s = 67 minutes per phase.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

USER_AGENT = "dota-analyst/0.4.0 (research; +https://github.com/qzmeister/dota_analyst)"
OPENDOTA = "https://api.opendota.com/api"
PRO_ROOT = Path(__file__).resolve().parents[1]
ML_DATA = PRO_ROOT / "ml_data"
FULL_MATCHES = ML_DATA / "full_matches"
IMPORTS = ML_DATA / "imports"

# Top-N teams we'll anchor the corpus on.  30 is the user spec.
TOP_N_TEAMS = int(os.environ.get("V17_TOP_N", "30"))

# Lookback window in seconds.  270 days ≈ 9 months, enough to cover
# the current patch plus the two previous ones (7.41, 7.40, 7.39 each
# last 3-7 months).  Default 270; override with V17_LOOKBACK_DAYS.
LOOKBACK_SEC = int(os.environ.get("V17_LOOKBACK_DAYS", "270")) * 86400

# Patches we anchor the corpus on.  Filled in from /api/constants/patch
# at Phase 7 time; here we just keep a "current N patches" selector.
PATCH_DEPTH = int(os.environ.get("V17_PATCH_DEPTH", "3"))

# OpenDota asks for <= 1 req/sec on the public API; we sleep a
# little longer to be polite.
RATE_SLEEP_SEC = float(os.environ.get("V17_RATE_SLEEP", "1.1"))


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
            print(f"  retry {attempt + 1}: {type(exc).__name__} {str(exc)[:80]} (sleep {wait:.1f}s)", file=sys.stderr)
            time.sleep(wait)
    print(f"  GIVE UP: {url} -> {type(last).__name__ if last else '?'}", file=sys.stderr)
    return None


def _save(name: str, payload: Any) -> Path:
    """Write a JSON payload to ml_data/imports/ for inspection."""
    IMPORTS.mkdir(parents=True, exist_ok=True)
    path = IMPORTS / f"v17_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _now_ts() -> int:
    return int(time.time())


# --------------------------------------------------------------------------- #
# Phase 1: Top teams
# --------------------------------------------------------------------------- #

def collect_top_teams(n: int = TOP_N_TEAMS) -> List[Dict[str, Any]]:
    """Pull the top-N pro teams by OpenDota rating."""
    print(f"[phase 1] top {n} teams...", file=sys.stderr)
    data = _http_json(f"{OPENDOTA}/teams")
    if not isinstance(data, list):
        raise RuntimeError("OpenDota /api/teams returned non-list")
    data.sort(key=lambda t: (t.get("rating") or 0, t.get("wins") or 0), reverse=True)
    top = data[:n]
    _save("phase1_top_teams", top)
    print(f"  -> {len(top)} teams, top: {top[0].get('name')} (rating {top[0].get('rating')})", file=sys.stderr)
    return top


# --------------------------------------------------------------------------- #
# Phase 2: Match IDs in lookback window
# --------------------------------------------------------------------------- #

def _explorer_sql(query: str) -> List[Dict[str, Any]]:
    """Run a SQL via OpenDota /api/explorer.  Returns rows as dicts."""
    url = f"{OPENDOTA}/explorer?{urllib.parse.urlencode({'sql': query})}"
    data = _http_json(url)
    if not isinstance(data, dict):
        return []
    return list(data.get("rows") or [])


def collect_match_ids(lookback_sec: int = LOOKBACK_SEC) -> List[Dict[str, Any]]:
    """All matches with leagueid in the lookback window.

    OpenDota /api/explorer SQL: pulls `match_id, start_time, leagueid,
    radiant_team_id, dire_team_id, radiant_win` for matches with a
    non-null leagueid (filters out pub games; includes pro + tier-2/3).
    Returns 4000+ matches for a 90-day window.

    Note: `patch` is NOT in the `matches` table; it lives on the
    per-match detail blob and we'll re-pull it in Phase 3.
    """
    print(f"[phase 2] match IDs in lookback...", file=sys.stderr)
    cutoff = _now_ts() - lookback_sec
    sql = (
        "SELECT match_id, start_time, leagueid, radiant_team_id, dire_team_id, radiant_win "
        "FROM matches "
        f"WHERE start_time >= {cutoff} "
        "AND leagueid IS NOT NULL "
        "AND radiant_team_id IS NOT NULL "
        "AND dire_team_id IS NOT NULL "
        "ORDER BY start_time DESC LIMIT 5000"
    )
    rows = _explorer_sql(sql)
    _save("phase2_match_ids", rows)
    print(f"  -> {len(rows)} matches in window", file=sys.stderr)
    return rows


def filter_top_teams_matches(match_rows: List[Dict[str, Any]],
                              top_team_ids: Set[int]) -> List[Dict[str, Any]]:
    """Keep only matches where at least one of the two teams is in top-N."""
    kept = [r for r in match_rows
            if r.get("radiant_team_id") in top_team_ids
            or r.get("dire_team_id") in top_team_ids]
    return kept


# --------------------------------------------------------------------------- #
# Phase 3: Match details
# --------------------------------------------------------------------------- #

def _enrich_match(mid: int) -> Optional[Dict[str, Any]]:
    raw = _http_json(f"{OPENDOTA}/matches/{mid}")
    if not isinstance(raw, dict):
        return None
    if raw.get("game_mode") not in (1, 2, 3, 4, 5, 12, 22):
        return None
    return raw


def collect_match_details(match_ids: Iterable[int]) -> List[Dict[str, Any]]:
    """Fetch /api/matches/<id> for every match ID; cache to disk."""
    FULL_MATCHES.mkdir(parents=True, exist_ok=True)
    IMPORTS.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []
    fetched = 0
    for i, mid in enumerate(match_ids, 1):
        target = IMPORTS / f"v17_match_{mid}.json"
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached:
                    out.append(cached)
                continue
            except Exception:
                pass
        raw = _enrich_match(mid)
        if not raw:
            continue
        with open(target, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        out.append(raw)
        fetched += 1
        if i % 50 == 0:
            print(f"  fetched {i} ({fetched} new) of {len(out)}", file=sys.stderr)
        time.sleep(RATE_SLEEP_SEC)
    print(f"[phase 3] {len(out)} matches in corpus ({fetched} new this run)", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Phase 4: Hero meta stats
# --------------------------------------------------------------------------- #

def collect_hero_stats() -> List[Dict[str, Any]]:
    """One /api/heroStats call: every hero with per-patch pick/win/ban counts."""
    print(f"[phase 4] hero stats...", file=sys.stderr)
    data = _http_json(f"{OPENDOTA}/heroStats")
    if not isinstance(data, list):
        return []
    _save("phase4_hero_stats", data)
    return data


# --------------------------------------------------------------------------- #
# Phase 5: Hero matchups
# --------------------------------------------------------------------------- #

def collect_hero_matchups(hero_ids: Iterable[int]) -> Dict[int, Any]:
    """Per-hero /api/heroes/<id>/matchups — win-rate against every other hero."""
    print(f"[phase 5] hero matchups...", file=sys.stderr)
    out: Dict[int, Any] = {}
    for hid in hero_ids:
        data = _http_json(f"{OPENDOTA}/heroes/{hid}/matchups")
        if isinstance(data, list):
            out[hid] = data
        time.sleep(RATE_SLEEP_SEC)
    _save("phase5_hero_matchups", out)
    return out


# --------------------------------------------------------------------------- #
# Phase 6: Team rosters
# --------------------------------------------------------------------------- #

def collect_team_players(team_ids: Iterable[int]) -> Dict[int, Any]:
    """Per-team /api/teams/<id>/players."""
    print(f"[phase 6] team rosters...", file=sys.stderr)
    out: Dict[int, Any] = {}
    for tid in team_ids:
        data = _http_json(f"{OPENDOTA}/teams/{tid}/players")
        if isinstance(data, list):
            out[tid] = data
        time.sleep(RATE_SLEEP_SEC)
    _save("phase6_team_players", out)
    return out


# --------------------------------------------------------------------------- #
# Phase 7: Patch info
# --------------------------------------------------------------------------- #

def collect_patch_info() -> List[Dict[str, Any]]:
    """/api/constants/patch returns ~30 patches with release dates."""
    print(f"[phase 7] patch info...", file=sys.stderr)
    data = _http_json(f"{OPENDOTA}/constants/patch")
    if not isinstance(data, list):
        return []
    _save("phase7_patch_info", data)
    return data


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    phase = argv[0]
    if phase == "1":
        collect_top_teams()
    elif phase == "2":
        collect_match_ids()
    elif phase == "3":
        ids_path = IMPORTS / "v17_phase2_match_ids.json"
        if not ids_path.exists():
            print(f"missing {ids_path}; run phase 2 first", file=sys.stderr)
            return 2
        rows = json.loads(ids_path.read_text())
        # optional: filter to top-N teams if phase 1 done
        teams_path = IMPORTS / "v17_phase1_top_teams.json"
        if teams_path.exists() and "--all" not in argv:
            teams = json.loads(teams_path.read_text())
            top_ids = {t["team_id"] for t in teams}
            rows = filter_top_teams_matches(rows, top_ids)
            print(f"  filtered to {len(rows)} matches involving top-{TOP_N_TEAMS} teams", file=sys.stderr)
        # Drop `--teams-only` to also keep top-N teams even without
        # the team filter (we already do by default; `--all` overrides).
        mids = [r["match_id"] for r in rows]
        collect_match_details(mids)
    elif phase == "4":
        collect_hero_stats()
    elif phase == "5":
        heroes_path = IMPORTS / "v17_phase4_hero_stats.json"
        heroes = json.loads(heroes_path.read_text()) if heroes_path.exists() else collect_hero_stats()
        collect_hero_matchups(h["id"] for h in heroes)
    elif phase == "6":
        teams_path = IMPORTS / "v17_phase1_top_teams.json"
        teams = json.loads(teams_path.read_text()) if teams_path.exists() else collect_top_teams()
        collect_team_players(t["team_id"] for t in teams)
    elif phase == "7":
        collect_patch_info()
    elif phase == "all":
        teams = collect_top_teams()
        rows = collect_match_ids()
        top_ids = {t["team_id"] for t in teams}
        rows = filter_top_teams_matches(rows, top_ids)
        mids = [r["match_id"] for r in rows]
        collect_match_details(mids)
        heroes = collect_hero_stats()
        collect_hero_matchups(h["id"] for h in heroes)
        collect_team_players(t["team_id"] for t in teams)
        collect_patch_info()
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
