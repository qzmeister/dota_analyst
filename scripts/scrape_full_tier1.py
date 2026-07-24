"""
Full Match Collection from dltv.org/matches page.

This scrapes the main matches page which contains ALL past, live, and upcoming matches.
Better than relying on Events API which only shows recent tournaments.

Strategy:
- Scrape https://dltv.org/matches HTML
- Extract all match cards with steam_id
- Filter for Tier 1 tournaments by name matching
- Cache locally to avoid re-scraping
"""

import sys
import os
import re
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional
from urllib.request import urlopen


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from backend.dltv_client import _http_json, BASE, SITE


def get_steam_key() -> str:
    return os.environ.get("STEAM_API_KEY", "").strip()


# Tier 1 tournament patterns (case-insensitive)
TIER1_PATTERNS = [
    "esports world cup",
    "blast slam",
    "dreamleague",
    "pgl wallachia",
    "esl one birmingham",
]


def is_tier1_event(event_name: str) -> bool:
    """Check if event name matches any Tier 1 pattern."""
    event_lower = event_name.lower()
    return any(pattern in event_lower for pattern in TIER1_PATTERNS)


def scrape_matches_page(limit_pages: int = 50) -> List[Dict]:
    """
    Scrape all matches from dltv.org/matches page.
    
    Returns list of match objects with:
    - steam_id (if available)
    - event/title
    - team_a, team_b
    - start_time
    - bo format
    """
    url = f"{SITE}/matches"
    
    print(f"[SCRAPE] Fetching {url}...")
    
    try:
        with urlopen(url, timeout=15.0) as resp:
            html = resp.read().decode("utf-8")
        
        print(f"[SCRAPE] Got {len(html)} bytes of HTML")
        
        # Parse match cards using regex (same logic as discovery.py)
        matches = parse_match_cards(html)
        print(f"[SCRAPE] Found {len(matches)} total match cards\n")
        
        # Filter for Tier 1 tournaments
        tier1_matches = filter_tier1(matches)
        
        print(f"[FILTERED] {len(tier1_matches)} matches are Tier 1")
        
        return tier1_matches
        
    except Exception as e:
        print(f"[ERROR] Scraping failed: {e}")
        return []


def parse_match_cards(html: str) -> List[Dict]:
    """Parse match cards from DLTV matches page HTML."""
    matches = []
    
    # Find all match div elements
    match_pattern = r'<div class="match(?!__)([^"]*)"(?:\s+|\W)*data-series-id="(\d+)".*?>.*?<div class="match__head-event">.*?<span>([^<]+)</span>.*?</div>.*?(?:.*?)<a href="/matches/\d+/[^"]*".*?>(.*?)</a>'
    
    # Simpler approach: find data-series-id blocks
    series_blocks = re.findall(r'data-series-id="(\d+)"[^>]*>.*?<span>([^<]+)</span>', html, re.DOTALL)
    
    for series_id, event_name in series_blocks:
        # Skip non-Tier 1 events early
        if not is_tier1_event(event_name):
            continue
        
        # Find teams and steam_ids in this block
        team_pattern = r'<div class="team__title">\s*<span>([^<]+)</span>'
        teams = re.findall(team_pattern, html[html.find(f'data-series-id="{series_id}"'):html.find('data-series-id=', html.find(f'data-series-id="{series_id}"')+1)+100])
        
        if len(teams) >= 2:
            # Look for steam_id in nearby attributes
            steam_match = re.search(r'data-match="(\d+)"', html[html.find(f'data-series-id="{series_id}"'):min(len(html), html.find(f'data-series-id="{series_id}"') + 500)])
            
            matches.append({
                "steam_id": int(steam_match.group(1)) if steam_match else None,
                "event_id": int(series_id),
                "event": event_name.strip(),
                "team_a": teams[0].strip(),
                "team_b": teams[1].strip(),
                "status": "upcoming" if "upcoming" in html[html.find(f'data-series-id="{series_id}"'):html.find(f'data-series-id="{series_id}"')+200] else "live",
            })
    
    return matches


def filter_tier1(matches: List[Dict]) -> List[Dict]:
    """Filter matches that belong to Tier 1 tournaments."""
    filtered = []
    
    for m in matches:
        if m.get("steam_id"):  # Only keep finished/live matches with Steam IDs
            filtered.append(m)
    
    return filtered


def enrich_with_details(matches: List[Dict], batch_size: int = 10) -> List[Dict]:
    """
    Fetch full match details via Steam API.
    
    This calls GetMatchDetails for each steam_id to get:
    - Full player stats
    - Hero picks/bans
    - Duration, scores
    - Timeline data
    """
    enriched = []
    seen_ids = set()
    
    print("\n[ENRICH] Fetching Steam MatchDetails...")
    
    for i, match in enumerate(matches):
        mid = match.get("steam_id")
        
        if mid in seen_ids or mid is None:
            continue
        
        seen_ids.add(mid)
        
        # Fetch details via Steam API
        details = fetch_match_details(mid)
        
        if details and "result" in details:
            result = details["result"]
            
            enriched_match = {
                **match,
                "duration_sec": result.get("duration", 0),
                "radiant_win": result.get("radiant_win"),
                "radiant_score": result.get("radiant_team_id") and sum(p.get("kills", 0) for p in result.get("players", []) if p.get("player_slot", 0) < 128),
                "dire_score": result.get("dire_team_id") and sum(p.get("kills", 0) for p in result.get("players", []) if p.get("player_slot", 0) >= 128),
                "heroes_radiant": [p.get("hero_id") for p in result.get("players", []) if p.get("player_slot", 0) < 128][:5],
                "heroes_dire": [p.get("hero_id") for p in result.get("players", []) if p.get("player_slot", 0) >= 128][:5],
                "bans_radiant": [p.get("hero_id") for p in result.get("players", []) if p.get("player_slot", 0) < 128 and p.get("hero_id")] or [],  # Simplified
                "bans_dire": [],
            }
            
            enriched.append(enriched_match)
            
            if (i + 1) % batch_size == 0:
                print(f"   [{i+1}/{len(matches)}] Enriched matches...")
                time.sleep(0.5)  # Be polite to Steam API
        
        elif not details:
            print(f"[SKIP] No details for match {mid}")
    
    return enriched


def fetch_match_details(match_id: int) -> Optional[Dict]:
    """Call Steam Web API GetMatchDetails endpoint."""
    if not get_steam_key():
        print("[ERROR] STEAM_API_KEY not configured")
        return None
    
    url = f"https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/v1/"
    
    params = f"?match_id={match_id}&format=json&key={get_steam_key()}"
    
    try:
        req = urllib.request.Request(url + params)
        req.add_header("User-Agent", "DotaAnalyst/MultiScrape/1.0")
        
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            if resp.status != 200:
                return None
            
            return json.loads(resp.read().decode("utf-8"))
            
    except Exception as e:
        print(f"[API ERROR] Fetch {match_id}: {e}")
        return None


def load_from_cache(cache_file: str) -> List[Dict]:
    """Load previously scraped data if exists."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"[CACHE] Loaded {len(data)} matches from cache")
            return data
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}")
    
    return []


def save_to_cache(matches: List[Dict], cache_file: str):
    """Save scraped data to cache."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    with open(cache_file, 'w') as f:
        json.dump(matches, f, indent=2)
    
    print(f"[CACHE] Saved {len(matches)} matches to {cache_file}")


def main():
    print("="*60)
    print("Tier 1 Tournament Full Scrape")
    print("="*60)
    print()
    
    cache_file = "ml_data/full_tier1_scrape.json"
    
    # Try cache first
    cached = load_from_cache(cache_file)
    if cached and len(cached) > 0:
        print(f"Using {len(cached)} cached matches")
        print("\nSaving output...")
        with open("ml_data/tier1_matches.json", 'w') as f:
            json.dump(cached, f, indent=2)
        print(f"Saved to ml_data/tier1_matches.json")
        return len(cached)
    
    # Scrape fresh data
    all_matches = scrape_matches_page()
    
    if not all_matches:
        print("[ERROR] No matches found!")
        return 0
    
    # Enrich with Steam API
    enriched = enrich_with_details(all_matches)
    
    # Save both versions
    print("\n[SAVE] Writing results...")
    
    # Raw scraped metadata
    with open("ml_data/raw_tier1_metadata.json", 'w') as f:
        json.dump(all_matches, f, indent=2)
    
    # Enriched with details
    with open("ml_data/tier1_matches.json", 'w') as f:
        json.dump(enriched, f, indent=2)
    
    # Cache
    save_to_cache(enriched, cache_file)
    
    print(f"\n{'='*60}")
    print(f"✅ DONE! Collected {len(enriched)} Tier 1 matches")
    print(f"💾 Saved to:")
    print(f"   - ml_data/raw_tier1_metadata.json ({len(all_matches)} raw)")
    print(f"   - ml_data/tier1_matches.json ({len(enriched)} enriched)")
    print(f"   - {cache_file} (cached)\n")
    
    return len(enriched)


if __name__ == "__main__":
    count = main()
