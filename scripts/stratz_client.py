"""Stratz GraphQL client for Dota 2 data.

Stratz (https://stratz.com/) provides match history, team
ratings, hero stats, and more via a public GraphQL endpoint.
Free tier is rate-limited (~30 req/min, no auth required)
but plenty for our use case.

The API key is OPTIONAL.  Without it we can still:
  - List teams by id
  - Get current team ratings
  - Query recent matches by league

With an API key (set STRATZ_API_KEY env var):
  - Higher rate limits
  - Historical match data (back to 2017)

Endpoint: https://api.stratz.com/graphql
Auth header: `Authorization: Bearer <key>` (when key is set)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

ENDPOINT = "https://api.stratz.com/graphql"
PRO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"


def _api_key() -> Optional[str]:
    """Return the Stratz API key from env, or None if not set."""
    return os.environ.get("STRATZ_API_KEY") or None


def _headers() -> Dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "User-Agent": "dota-analyst/0.7.4 (research; +https://github.com/qzmeister/dota_analyst)",
    }
    key = _api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _gql(query: str, variables: Optional[Dict[str, Any]] = None,
         timeout: float = 60.0) -> Optional[Dict[str, Any]]:
    """Run a GraphQL query against Stratz.  Returns the `data`
    block (or None on transport / parse error)."""
    body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"  GQL error: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
        return None
    if "errors" in payload:
        # Stratz returns 200 + errors[] for GraphQL problems
        for err in payload["errors"][:3]:
            print(f"  GQL error: {err.get('message', err)[:200]}", file=sys.stderr)
    return payload.get("data") if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def list_top_teams(limit: int = 200) -> List[Dict[str, Any]]:
    """Top teams by Stratz rating.  Returns at most `limit` rows.

    Each row: {teamId, name, tag, rating, wins, losses, ...}
    Stratz's `rating` field is the team's current Glicko/Elo
    score.  We treat it as the authoritative tier signal.
    """
    q = """
    query TopTeams($take: Int!) {
      teams(orderBy: RATING, take: $take) {
        id
        name
        tag
        rating
        wins
        losses
        lastMatchDateTime
      }
    }
    """
    data = _gql(q, {"take": limit})
    if not data or "teams" not in data:
        return []
    return [t for t in data["teams"] if t is not None]


def get_team_ratings(team_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Bulk fetch team ratings.  Returns {team_id: {rating, ...}}."""
    if not team_ids:
        return {}
    q = """
    query TeamRatings($ids: [Long!]!) {
      teams(ids: $ids) {
        id
        name
        tag
        rating
        wins
        losses
      }
    }
    """
    out: Dict[int, Dict[str, Any]] = {}
    # Stratz's `ids` filter accepts a list; cap each request at
    # 100 ids to stay under the response size limit.
    for i in range(0, len(team_ids), 100):
        chunk = team_ids[i:i + 100]
        data = _gql(q, {"ids": chunk})
        if not data or "teams" not in data:
            continue
        for t in data["teams"]:
            if t is None:
                continue
            try:
                tid = int(t.get("id") or 0)
            except (TypeError, ValueError):
                continue
            if tid:
                out[tid] = t
    return out


def list_matches_by_league(league_id: int, take: int = 100,
                            skip: int = 0) -> List[Dict[str, Any]]:
    """Recent matches in a league.  Returns at most `take` rows."""
    q = """
    query LeagueMatches($id: Int!, $take: Int!, $skip: Int!) {
      league(id: $id) {
        id
        name
        matches(take: $take, skip: $skip) {
          id
          startDateTime
          duration
          radiantTeamId
          direTeamId
          radiantWin
          leagueId
          patch
          players {
            heroId
            team
            isRadiant
          }
          didRadiantWin
        }
      }
    }
    """
    data = _gql(q, {"id": league_id, "take": take, "skip": skip})
    if not data or "league" not in data or not data["league"]:
        return []
    return data["league"].get("matches") or []


def list_recent_pro_matches(take: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Top-level list of recent pro matches (no league filter)."""
    q = """
    query RecentMatches($take: Int!, $skip: Int!) {
      matches(
        take: $take
        skip: $skip
        orderBy: START_DATE_TIME_DESC
      ) {
        id
        startDateTime
        duration
        radiantTeamId
        direTeamId
        radiantWin
        leagueId
        patch
        didRadiantWin
        players {
          heroId
          team
          isRadiant
        }
      }
    }
    """
    data = _gql(q, {"take": take, "skip": skip})
    if not data or "matches" not in data:
        return []
    return [m for m in data["matches"] if m is not None]


def get_match_full(match_id: int) -> Optional[Dict[str, Any]]:
    """One full match payload (with players, gold/xp adv, etc.)."""
    q = """
    query FullMatch($id: Long!) {
      match(id: $id) {
        id
        startDateTime
        duration
        radiantTeamId
        direTeamId
        radiantWin
        leagueId
        patch
        players {
          heroId
          team
          isRadiant
        }
        didRadiantWin
        radiantGoldAdvantage
        direGoldAdvantage
        pickBans {
          isPick
          isRadiant
          heroId
          order
        }
      }
    }
    """
    data = _gql(q, {"id": match_id})
    if not data or "match" not in data:
        return None
    return data["match"]


# --------------------------------------------------------------------------- #
# Quick connectivity test
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 78)
    print("Stratz connectivity test")
    print("=" * 78)
    print(f"  API key: {'set' if _api_key() else 'NOT set (free tier)'}")
    print()

    print("Test 1: list top 5 teams")
    teams = list_top_teams(5)
    print(f"  -> {len(teams)} teams")
    for t in teams[:5]:
        print(f"    id={t.get('id'):>10d}  rating={t.get('rating')}  "
              f"name={t.get('name'):>20s}  tag={t.get('tag')}")
    print()

    print("Test 2: list 5 recent pro matches")
    matches = list_recent_pro_matches(5)
    print(f"  -> {len(matches)} matches")
    for m in matches[:5]:
        print(f"    id={m.get('id'):>10d}  startDateTime={m.get('startDateTime')}  "
              f"radiantTeamId={m.get('radiantTeamId')}  patch={m.get('patch')}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
