"""
DatDota API Client for Dota Analyst.

Professional Dota 2 statistics API with:
- Tournament/league data
- Match history with full details
- Player/team stats
- Draft analysis

Rate limits:
- 3 second minimum between requests
- 500 requests/day (no key)
- Contact Noxville on Discord for higher limits
"""

import os
import json
import time
import urllib.request
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta


BASE_URL = "https://api.datdota.com"


def _http_json(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: float = 10.0, retries: int = 3) -> Optional[Any]:
    """Fetch JSON with rate limiting and retry logic."""
    
    # Build query string
    if params:
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{url}?{query}"
    
    last_exc = None
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            
            # Add headers
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            
            req.add_header("User-Agent", "DotaAnalyst/DatDota/1.0")
            req.add_header("Accept", "application/json")
            
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 429:
                    print(f"[datdota] Rate limited! Waiting 10s...")
                    time.sleep(10)
                    continue
                
                if resp.status != 200:
                    print(f"[datdota] HTTP {resp.status}")
                    return None
                
                data = json.loads(resp.read().decode("utf-8"))
                return data
                
        except Exception as e:
            last_exc = e
            print(f"[datdota] Error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)  # Respect rate limit
    
    return None


# ========================================================================= #
# ENDPOINTS
# ========================================================================= #

def get_leagues(limit: int = 100, offset: int = 0) -> Optional[Dict]:
    """
    Get list of leagues/tournaments.
    
    Returns:
    {
        "data": [
            {
                "leagueId": 13234,
                "name": "DreamLeague Season 29",
                "tier": {"id": 1, "name": "PREMIUM"},
                "first": "2026-05-13T00:00:00.000+00:00",
                "last": "2026-05-24T00:00:00.000+00:00",
                "count": 120,
                ...
            }
        ]
    }
    """
    url = f"{BASE_URL}/api/leagues"
    params = {"limit": limit, "offset": offset}
    
    return _http_json(url, params)


def get_league_matches(league_id: int) -> Optional[Dict]:
    """
    Get all matches for a specific league/tournament.
    
    Returns:
    {
        "data": {
            "league": {"leagueId": 19785, "name": "...", "tier": 1},
            "matches": {
                "radiantWins": 72,
                "direWins": 85,
                "avgDuration": 2601.15,
                "total": 157,
                "data": [
                    {
                        "matchId": 8885183102,
                        "startDate": "2026-07-07T09:24:39.000+00:00",
                        "duration": 2623,
                        "radiantVictory": false
                    }
                ]
            }
        }
    }
    """
    url = f"{BASE_URL}/api/leagues/{league_id}"
    
    return _http_json(url)


def get_match_details(match_id: int) -> Optional[Dict]:
    """
    Get full match details including:
    - Player stats
    - Hero picks/bans
    - Timeline data
    - Gold/XP graphs
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
    
    Filters for tournaments with tier.id = 1 (PREMIUM)
    """
    print("[DatDota] Fetching all leagues...")
    
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
        
        print(f"   Fetched {len(all_leagues)} leagues so far...")
    
    # Filter Tier 1 (tier.id = 1 = PREMIUM)
    tier1 = [l for l in all_leagues if l.get("tier", {}).get("id") == 1]
    
    print(f"\n[DatDota] Found {len(tier1)} Tier 1 tournaments:")
    for league in tier1[:10]:
        print(f"   - {league.get('name')} (ID: {league.get('leagueId')})")
    
    return tier1


def collect_tournament_matches(league_id: int, league_name: str) -> List[Dict]:
    """
    Collect all matches from a specific tournament.
    
    Args:
        league_id: DatDota league ID
        league_name: Tournament name (for logging)
    """
    print(f"\n[DatDota] Fetching matches for: {league_name}")
    
    data = get_league_matches(league_id)
    
    if not data or "data" not in data:
        print(f"   [ERROR] No data returned")
        return []
    
    league_data = data["data"]
    matches_data = league_data.get("matches", {})
    matches = matches_data.get("data", [])
    
    print(f"   [OK] Total: {len(matches)} matches")
    print(f"   Stats: {matches_data.get('radiantWins', 0)} radiant wins, {matches_data.get('direWins', 0)} dire wins")
    print(f"   Avg duration: {matches_data.get('avgDuration', 0):.1f}s")
    
    return matches


def collect_all_tier1_matches() -> List[Dict]:
    """
    Main entry point: collect all matches from Tier 1 tournaments.
    
    Returns list of match objects ready for ML training.
    """
    print("="*60)
    print("DatDota Tier 1 Tournament Collection")
    print("="*60 + "\n")
    
    # Step 1: Get Tier 1 tournaments
    tier1_leagues = collect_tier1_tournaments()
    
    if not tier1_leagues:
        print("[ERROR] No Tier 1 tournaments found!")
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
        json.dump(unique_matches, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DONE! Collected {len(unique_matches)} unique Tier 1 matches")
    print(f"Saved to: {output_file}\n")
    
    return unique_matches


# ========================================================================= #
# TEST
# ========================================================================= #

def test_api():
    """Test basic API connectivity."""
    print("[TEST] Testing DatDota API connectivity...\n")
    
    # Test 1: Get leagues
    print("1. Fetching leagues...")
    leagues = get_leagues(limit=5)
    if leagues and "data" in leagues:
        print(f"   [OK] Got {len(leagues['data'])} leagues")
        for l in leagues["data"][:3]:
            tier_name = l.get("tier", {}).get("name", "Unknown")
            print(f"      - {l.get('name')} [{tier_name}]")
    else:
        print("   [FAIL] Failed")
    
    time.sleep(3)
    
    # Test 2: Get league matches
    print("\n2. Fetching league matches (Esports World Cup 2026)...")
    league_data = get_league_matches(19785)
    if league_data and "data" in league_data:
        matches = league_data["data"].get("matches", {}).get("data", [])
        print(f"   [OK] Got {len(matches)} matches")
        if matches:
            first = matches[0]
            print(f"      - Match {first.get('matchId')}: {first.get('duration')}s, radiant_win={first.get('radiantVictory')}")
    else:
        print("   [FAIL] Failed")
    
    print("\n[TEST] API connectivity OK!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_api()
    else:
        # Full collection
        matches = collect_all_tier1_matches()
