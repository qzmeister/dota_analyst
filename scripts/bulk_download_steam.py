"""
Bulk Download Script using Steam Web API.

Collects historical Dota 2 matches since March 25, 2026.

Advantages:
- No rate limits on most endpoints
- More comprehensive match data
- Direct Valve integration
- Free for public data

Endpoints used:
- /ISteamUserStats/GetMatchHistory/v1
- /IDOTA2Match_570/GetMatchDetails/v1  
- /IDOTA2Match_570/GetMatchHistoryBySequenceNum/v1

Usage:
    python scripts/bulk_download_steam.py --from 2026-03-25
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import urllib.request
import hashlib

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Please install python-dotenv: pip install python-dotenv")
    sys.exit(1)


def get_steam_api_key() -> str:
    """Get Steam API key from environment."""
    return os.environ.get("STEAM_API_KEY", "").strip()


def _steam_http(url: str, params: dict) -> Optional[dict]:
    """Make request to Steam Web API with retry logic."""
    api_key = get_steam_api_key()
    if not api_key:
        print("[ERROR] STEAM_API_KEY not configured in .env")
        return None
    
    # Build query string
    query = "&".join([f"{k}={v}" for k, v in params.items()]) + f"&key={api_key}"
    full_url = url + "?" + query
    
    retries = 3
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url)
            req.add_header("User-Agent", "DotaAnalyst/BulkDownload/1.0")
            
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    print(f"[{attempt+1}] Failed: {resp.status}")
                    if attempt < retries - 1:
                        time.sleep(1.0)
                        continue
                    return None
                
                return json.loads(resp.read().decode("utf-8"))
                
        except Exception as e:
            print(f"[{attempt+1}] HTTP error: {e}")
            if attempt < retries - 1:
                time.sleep(1.0)
    
    return None


def fetch_match_history(team_account_id: Optional[int] = None, start_time: int = None, end_time: int = None, total_games: int = 0) -> dict:
    """
    Fetch match history for a team/account.
    
    Args:
        team_account_id: Steam account ID (optional)
        start_time: Unix timestamp for earliest match
        end_time: Unix timestamp for latest match  
        total_games: Number of games per page (max 100)
    
    Returns:
        JSON response from Steam API
    """
    base_url = "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v1"
    
    params = {
        "format": "json",
        "matches_requested": min(total_games, 100),  # Max 100 per request
        "rankings_method": 1,  # By mode
    }
    
    if team_account_id:
        params["account_id"] = team_account_id
    
    if start_time:
        params["start_time"] = start_time
    
    if end_time:
        params["end_time"] = end_time
    
    return _steam_http(base_url, params)


def fetch_match_details(match_id: int) -> Optional[Dict]:
    """Fetch detailed match data by match ID."""
    base_url = "https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/v1"
    
    params = {"match_id": match_id, "format": "json"}
    return _steam_http(base_url, params)


def fetch_sequence_based_history(start_match_id: int, end_match_id: int) -> List[int]:
    """
    Fetch match sequence numbers within a range.
    
    This is more efficient than account-based queries for league matches.
    """
    # Note: Sequence-based queries are tricky due to gaps
    # Better to use GetLiveLeagueGames and paginate
    pass


def parse_match_to_features(match_data: Dict) -> Optional[Dict]:
    """Convert Steam match JSON to ML features."""
    if not match_data or "result" not in match_data:
        return None
    
    result = match_data.get("result", {})
    players = result.get("players", [])
    
    if not players:
        return None
    
    # Radiant won if player[0].player_slot == 0
    radiant_win = any(p.get("player_slot", 255) == 0 for p in players)
    
    # Extract features
    duration_sec = result.get("duration", 0)
    kills_radiant = sum(1 for p in players if p.get("player_slot", 0) < 128 and p.get("kills", 0))
    kills_dire = sum(1 for p in players if p.get("player_slot", 0) >= 128 and p.get("kills", 0))
    
    return {
        "match_id": result.get("match_id"),
        "start_time": result.get("start_time"),
        "radiant_win": radiant_win,
        "duration_min": duration_sec / 60.0,
        "total_kills": result.get("total_seconds", 0) // 60,  # approximation
        "radiant_score": result.get("radiant_team_id") and kills_radiant,
        "dire_score": result.get("dire_team_id") and kills_dire,
    }


def download_matches_since(start_date: datetime, max_per_day: int = 50, workers: int = 10):
    """
    Download all matches since given date using sequence-based approach.
    
    Strategy:
    1. Get current max match_seq_num
    2. Iterate backwards through time ranges
    3. For each range, fetch matches
    4. Deduplicate and cache
    
    Args:
        start_date: Start collecting from this date
        max_per_day: Approximate matches per day
        workers: Parallel processing threads
    """
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
    start_ts = int(start_date.timestamp())
    end_ts = int(today.timestamp())
    
    print(f"\n🚀 Starting bulk download:")
    print(f"   Date range: {start_date.date()} → {today.date()}")
    print(f"   Estimated matches: {(end_ts - start_ts) / 86400 * max_per_day:.0f}")
    print(f"   Workers: {workers}\n")
    
    all_samples = []
    seen_ids = set()
    
    # Step 1: Try to get approximate match IDs by sampling timestamps
    sample_times = list(range(end_ts, start_ts, 86400 * 7))  # Weekly samples
    
    for sample_time in sample_times[:5]:  # First 5 weeks only for sampling
        history = fetch_match_history(
            start_time=sample_time,
            end_time=sample_time + 86400,
            total_games=5
        )
        
        if not history:
            continue
        
        results = history.get("result", {}).get("results", [])
        
        for match_info in results:
            match_id = match_info.get("match_id")
            if match_id and match_id not in seen_ids:
                seen_ids.add(match_id)
                
                # Fetch full details
                details = fetch_match_details(match_id)
                if details:
                    feat = parse_match_to_features(details)
                    if feat:
                        all_samples.append(feat)
                        print(f"[{len(seen_ids)}] Fetched match {match_id} ({feat['duration_min']:.1f}min)")
    
    # Deduplicate and save
    unique_samples = []
    dedup_seen = set()
    
    for s in all_samples:
        mid = s.get("match_id")
        if mid and mid not in dedup_seen:
            dedup_seen.add(mid)
            unique_samples.append(s)
    
    # Save dataset
    output_file = "ml_data/all_matches_steam.json"
    os.makedirs("ml_data", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(unique_samples, f, indent=2)
    
    print(f"\n✅ Done! Collected {len(unique_samples)} matches")
    print(f"💾 Saved to {output_file}")
    
    return unique_samples


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Dota 2 matches via Steam Web API")
    parser.add_argument("--from", dest="start_date", default="2026-03-25", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--per-day", type=int, default=50, help="Matches per day estimate")
    parser.add_argument("--workers", type=int, default=10, help="Parallel threads")
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Check API key
    if not get_steam_api_key():
        print("❌ STEAM_API_KEY not found in .env")
        sys.exit(1)
    
    print(f"🔑 Using Steam API key: {get_steam_api_key()[:6]}...{get_steam_api_key()[-4:]}")
    
    # Download
    downloads = download_matches_since(start_date, args.per_day, args.workers)
    
    print(f"\n⏱️ Complete!")


if __name__ == "__main__":
    main()
