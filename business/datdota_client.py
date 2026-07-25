"""
DatDota API Client for Dota Analyst.

Professional Dota 2 statistics API with:
- Tournament/league data
- Match history with full details
- Player/team stats
- Draft analysis

Rate limits:
- 3 second minimum between requests (callers should `time.sleep(3)` between
  collection calls; the HTTP layer only handles transient transport errors)
- 500 requests/day (no key)
- Contact Noxville on Discord for higher limits

HTTP retry / exponential backoff lives in `business._http`.
"""

import os
import time
from typing import Optional, Dict, List, Any

from ._http import request_json
from ._logging import get_logger

log = get_logger(__name__)


BASE_URL = "https://api.datdota.com"

# Headers shared across all DatDota calls.
_HEADERS = {
    "User-Agent": "DotaAnalyst/DatDota/1.0",
    "Accept": "application/json",
}


def _http_json(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: float = 10.0,
    retries: int = 3,
) -> Optional[Any]:
    """Fetch JSON with exponential backoff and retry.

    Thin wrapper around `request_json` that injects the default DatDota headers
    and respects the daily rate budget. The 3-second inter-request sleep used
    by collection scripts is a separate caller concern (see collect_* fns).
    """
    merged = {**_HEADERS, **(headers or {})}
    return request_json(
        url=url,
        headers=merged,
        params=params,
        timeout=timeout,
        retries=retries,
        backoff_base=2.0,  # DatDota likes 2-3s between attempts
        backoff_cap=30.0,
    )


# ========================================================================= #
# ENDPOINTS
# ========================================================================= #

def get_leagues(limit: int = 100, offset: int = 0) -> Optional[Dict]:
    """
    Get list of leagues/tournaments.
    """
    url = f"{BASE_URL}/api/leagues"
    params = {"limit": limit, "offset": offset}
    return _http_json(url, params)


def get_league_matches(league_id: int) -> Optional[Dict]:
    """
    Get all matches for a specific league/tournament.
    """
    url = f"{BASE_URL}/api/leagues/{league_id}"
    return _http_json(url)


def get_match_details(match_id: int) -> Optional[Dict]:
    """
    Get full match details.
    """
    url = f"{BASE_URL}/api/matches/{match_id}"
    return _http_json(url)


def get_teams(limit: int = 100) -> Optional[Dict]:
    """Get list of professional teams."""
    url = f"{BASE_URL}/api/teams"
    params = {"limit": limit}
    return _http_json(url, params)


def get_team_matches(team_id: int, limit: int = 50) -> Optional[Dict]:
    """Get recent matches for a team."""
    url = f"{BASE_URL}/api/teams/{team_id}/matches"
    params = {"limit": limit}
    return _http_json(url, params)


# ========================================================================= #
# BULK COLLECTION
# ========================================================================= #

def collect_tier1_tournaments() -> List[Dict]:
    """
    Collect all Tier 1 tournaments from DatDota.
    """
    log.info("Fetching all leagues from DatDota...")

    all_leagues = []
    offset = 0
    batch_size = 100

    while True:
        data = get_leagues(limit=batch_size, offset=offset)
        if not data or "data" not in data:
            break

        leagues = data["data"]
        if not leagues:
            break

        all_leagues.extend(leagues)
        offset += batch_size

        # Rate limit
        time.sleep(3)

        log.info("Fetched %d leagues so far", len(all_leagues))

    # Filter Tier 1 (tier.id = 1 = PREMIUM)
    tier1 = [l for l in all_leagues if l.get("tier", {}).get("id") == 1]

    log.info("Found %d Tier 1 tournaments", len(tier1))
    for league in tier1[:10]:
        log.info("  - %s (ID: %s)", league.get('name'), league.get('leagueId'))

    return tier1


def collect_tournament_matches(league_id: int, league_name: str) -> List[Dict]:
    """
    Collect all matches from a specific tournament.
    """
    log.info("Fetching matches for: %s", league_name)

    data = get_league_matches(league_id)

    if not data or "data" not in data:
        log.error("No data returned for league %s", league_id)
        return []

    league_data = data["data"]
    matches_data = league_data.get("matches", {})
    matches = matches_data.get("data", [])

    log.info("Total: %d matches", len(matches))
    log.info("Stats: %s radiant wins, %s dire wins",
             matches_data.get('radiantWins', 0), matches_data.get('direWins', 0))
    log.info("Avg duration: %.1fs", matches_data.get('avgDuration', 0))

    return matches


def collect_all_tier1_matches() -> List[Dict]:
    """
    Main entry point: collect all matches from Tier 1 tournaments.

    Returns list of match objects ready for ML training.
    """
    log.info("=" * 60)
    log.info("DatDota Tier 1 Tournament Collection")
    log.info("=" * 60)

    # Step 1: Get Tier 1 tournaments
    tier1_leagues = collect_tier1_tournaments()

    if not tier1_leagues:
        log.error("No Tier 1 tournaments found!")
        return []

    # Step 2: Collect matches from each tournament
    all_matches = []

    for league in tier1_leagues:
        league_id = league.get("leagueId")
        league_name = league.get("name")

        matches = collect_tournament_matches(league_id, league_name)

        # Add tournament metadata
        for match in matches:
            match["tournament_name"] = league_name
            match["tournament_tier"] = 1
            match["league_id"] = league_id

        all_matches.extend(matches)

        # Rate limit between tournaments
        time.sleep(3)

    # Deduplicate by match_id
    seen_ids = set()
    unique_matches = []

    for match in all_matches:
        mid = match.get("match_id")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique_matches.append(match)

    # Save
    output_file = "ml_data/datdota_tier1_matches.json"
    os.makedirs("ml_data", exist_ok=True)

    with open(output_file, 'w') as f:
        import json
        json.dump(unique_matches, f, indent=2)

    log.info("=" * 60)
    log.info("DONE! Collected %d unique Tier 1 matches", len(unique_matches))
    log.info("Saved to: %s", output_file)

    return unique_matches


# ========================================================================= #
# TEST
# ========================================================================= #

def test_api():
    """Test basic API connectivity."""
    log.info("Testing DatDota API connectivity...")

    # Test 1: Get leagues
    log.info("1. Fetching leagues...")
    leagues = get_leagues(limit=5)
    if leagues and "data" in leagues:
        log.info("[OK] Got %d leagues", len(leagues['data']))
        for l in leagues["data"][:3]:
            tier_name = l.get("tier", {}).get("name", "Unknown")
            log.info("    - %s [%s]", l.get('name'), tier_name)
    else:
        log.error("[FAIL] leagues fetch failed")

    time.sleep(3)

    # Test 2: Get league matches
    log.info("2. Fetching league matches (Esports World Cup 2026)...")
    league_data = get_league_matches(19785)
    if league_data and "data" in league_data:
        matches = league_data["data"].get("matches", {}).get("data", [])
        log.info("[OK] Got %d matches", len(matches))
        if matches:
            first = matches[0]
            log.info("    - Match %s: %ss, radiant_win=%s",
                     first.get('matchId'), first.get('duration'), first.get('radiantVictory'))
    else:
        log.error("[FAIL] league matches fetch failed")

    log.info("API connectivity OK!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_api()
    else:
        # Full collection
        matches = collect_all_tier1_matches()
