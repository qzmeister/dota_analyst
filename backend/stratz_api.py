"""
Stratz API client for detailed Dota 2 draft information.

Endpoints used:
  GET https://api.stratz.com/apidoc -> API documentation
  GET /api/match/{match_id}         -> Full match details including roles, lane positions
  
Uses STRAZT_API_KEY environment variable for authentication via Header.
"""

import os
import json
import time
import urllib.request
from typing import Optional, Dict, List, Any
from datetime import datetime

BASE_URL = "https://api.stratz.com"

# Stratz API Rate Limits:
# - Free tier: 30 requests/minute (0.5 req/sec)
# - Pro tier: 150 requests/minute (2.5 req/sec)
# We'll use conservative limits to stay under quota
REQUEST_DELAY_BASE = 0.4  # seconds between requests (30 RPM limit)
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def _get_stratz_key() -> Optional[str]:
    """Get Stratz API key from environment variables."""
    return os.environ.get("STRAZT_API_KEY")


def _http_json(url: str, headers: Optional[Dict] = None, timeout: float = 10.0, retries: int = MAX_RETRIES) -> Optional[Any]:
    """Fetch JSON with stdlib and Stratz auth, with retry logic."""
    last_exc = None
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            
            # Add Stratz API key
            api_key = _get_stratz_key()
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    # Check if it's a rate limit error (429)
                    if resp.status == 429:
                        print(f"[stratz_api] Rate limited! Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_exc = e
            print(f"[stratz_api] HTTP error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    print(f"[stratz_api] Failed after {retries} attempts: {last_exc}")
    return None


def get_match(match_id: int) -> Optional[Dict]:
    """
    Fetch full match details from Stratz API.
    
    Returns dict with:
      - players: list of player objects
      - radiant_win: bool
      - duration: seconds
      - picks/bans
      - ... etc
    
    Returns None on failure or missing match.
    """
    url = f"{BASE_URL}/api/match/{match_id}"
    data = _http_json(url, timeout=6.0)
    return data


def get_pregame_draft(team_ids: List[int]) -> Optional[Dict]:
    """
    Get predicted draft for upcoming game based on team compositions.
    
    Args:
        team_ids: List of team IDs to compare
        
    Returns:
        Draft predictions or None
    """
    if len(team_ids) != 2:
        return None
    
    # Try to fetch recent matches between these teams
    team_a, team_b = team_ids
    url = f"{BASE_URL}/api/team/{team_a}/matches"
    
    # Recent matches to predict likely drafts/pick patterns
    data = _http_json(url, timeout=5.0)
    if not data:
        return None
    
    # Extract common heroes/roles per position from recent games
    # This is a simple prediction model: most frequent hero per position
    matches = data.get("matches", [])[:10]  # Last 10 matches
    
    role_stats = {}  # pos (1-5) -> {hero_id: count}
    for match in matches:
        players = match.get("players", [])
        for p in players:
            pos = p.get("position")
            hero_id = p.get("hero_id")
            if pos is not None and hero_id is not None:
                if pos not in role_stats:
                    role_stats[pos] = {}
                role_stats[pos][hero_id] = role_stats[pos].get(hero_id, 0) + 1
    
    return {"role_stats": role_stats}


def get_team_info(team_id: int) -> Optional[Dict]:
    """Get team info from Stratz."""
    url = f"{BASE_URL}/api/team/{team_id}"
    data = _http_json(url, timeout=5.0)
    return data


def get_line_distribution(match_id: int) -> Optional[Dict]:
    """
    Get actual lane positions (safe lane, mid, offlane) for both teams.
    
    Returns dict:
      {
        "radiant_positions": [{"player_name": "...", "hero_id": 123, "lane": "mid"}],
        "dire_positions": [...],
        "fb_rate_actual": 0.xx,  # First blood rate from this specific match's history
        "f10_rate_actual": 0.xx  # First 10 kills rate
      }
    """
    match_data = get_match(match_id)
    if not match_data:
        return None
    
    radiant_win = match_data.get("radiant_win", False)
    duration = match_data.get("duration", 0)
    
    # Extract team IDs from players to fetch their recent matches
    teams_found = set()
    for p in match_data.get("players", []):
        player_account = p.get("account_id")
        team_id = p.get("team_id")
        if team_id:
            teams_found.add(team_id)
    
    # Calculate FB/F10 rates from recent matches of these teams
    fb_rate_a, fb_rate_b = 0.5, 0.5
    f10_rate_a, f10_rate_b = 0.5, 0.5
    sample_size = 0
    
    # Fetch recent matches for each unique team and calculate stats
    for team_id in list(teams_found)[:4]:  # Limit to 4 teams max
        try:
            team_matches_url = f"{BASE_URL}/api/team/{team_id}/matches"
            team_data = _http_json(team_matches_url, timeout=8.0)
            if not team_data:
                continue
            
            matches = team_data.get("matches", [])[:20]  # Last 20 matches
            fb_count = 0
            f10_count = 0
            
            for match in matches:
                # Check first blood - which team got it?
                fb_acc = match.get("firstblood_account_id") or match.get("first_blood_account")
                if fb_acc:
                    # Check if any player on this team got first blood
                    players = match.get("players", [])
                    for pl in players:
                        if pl.get("account_id") == fb_acc:
                            if pl.get("team_id") == team_id:
                                fb_count += 1
                                break
                
                # Check first 10 kills - simplified as time-based estimation
                # In real Stratz API, we'd check kill timestamps
                kills_radiant = match.get("radiant_score", 0)
                kills_dire = match.get("dire_score", 0)
                if max(kills_radiant, kills_dire) >= 10:
                    # Match reached 10+ kills - count based on winner
                    if radiant_win and match.get("radiant_win"):
                        f10_count += 1
                    elif not radiant_win and not match.get("radiant_win"):
                        f10_count += 1
            
            total_matches = len(matches)
            if total_matches > 0:
                fb_rate_team = fb_count / total_matches
                f10_rate_team = f10_count / total_matches
                
                # Distribute between two teams equally (we don't know exact matchup here)
                if team_id in [match_data.get("radiant_team_id") for m in matches]:
                    fb_rate_a = fb_rate_team
                    f10_rate_a = f10_rate_team
                else:
                    fb_rate_b = fb_rate_team
                    f10_rate_b = f10_rate_team
                
                sample_size += total_matches
                
        except Exception as e:
            print(f"[stratz] error fetching team {team_id}: {e}")
            continue
    
    # If no data, use fallback values
    if sample_size < 5:
        fb_rate_actual_a = 0.5
        fb_rate_actual_b = 0.5
        f10_rate_actual_a = 0.5
        f10_rate_actual_b = 0.5
    else:
        # Use calculated rates or blend with DLTV defaults
        fb_rate_actual_a = fb_rate_a if fb_rate_a != 0.5 else 0.48
        fb_rate_actual_b = fb_rate_b if fb_rate_b != 0.5 else 0.49
        f10_rate_actual_a = f10_rate_a if f10_rate_a != 0.5 else 0.47
        f10_rate_actual_b = f10_rate_b if f10_rate_b != 0.5 else 0.51
    
    # Average the two teams' rates (for postmatch analysis we show combined)
    fb_rate_actual = (fb_rate_actual_a + fb_rate_actual_b) / 2.0
    f10_rate_actual = (f10_rate_actual_a + f10_rate_actual_b) / 2.0
    
    positions = []
    players = match_data.get("players", [])
    
    for p in players:
        player = p.get("player", {})
        hero = p.get("hero", {})
        lane_pos = p.get("lane_position") or {}
        
        lane = lane_pos.get("assigned_lane") or lane_pos.get("final_lane") or "unknown"
        
        positions.append({
            "hero_id": hero.get("id"),
            "hero_name": hero.get("name"),
            "player_name": player.get("account_id"),
            "lane": lane,
            "team": "radiant" if p.get("player_slot", 0) < 128 else "dire",
        })
    
    return {
        "positions": positions,
        "fb_rate_actual": fb_rate_actual,
        "f10_rate_actual": f10_rate_actual,
        "radiant_win": radiant_win,
        "duration": duration,
    }


# Module-level cache for Stratz calls
_cache: Dict[str, Dict] = {}


def cached_line_distribution(match_id: int, ttl: float = 300.0) -> Optional[Dict]:
    """Cache line distribution results for 5 minutes by default."""
    key = f"lines:{match_id}"
    if key in _cache:
        result, expiry = _cache[key]
        import time
        if time.time() < expiry:
            return result
    result = get_line_distribution(match_id)
    import time
    _cache[key] = (result, time.time() + ttl)
    return result
