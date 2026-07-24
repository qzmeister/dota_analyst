"""
Tier 1 Tournament Data Collector for Dota Analyst.

Collects match history from only the top-tier tournaments listed:
- Esports World Cup 2026 (Jul 07-19)
- BLAST SLAM VII (May 26 – Jun 07)
- DreamLeague Season 29 (May 13–24)
- PGL Wallachia Season 8 (Apr 18–26)
- ESL One Birmingham 2026 (Mar 22–29)
- PGL Wallachia Season 7 (Mar 07–15)
- DreamLeague Season 28 (Feb 16 – Mar 01)

Uses DLTV discovery to get tournament slugs, then fetches matches via Steam API.

Output: ~200-300 high-quality matches from top tournaments only!
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: Please install python-dotenv: pip install python-dotenv")
    sys.exit(1)


def get_steam_key() -> str:
    return os.environ.get("STEAM_API_KEY", "").strip()


# Known Tier 1 tournaments from your list
TIER1_TOURNAMENTS = [
    {
        "name": "Esports World Cup 2026",
        "start_date": "2026-07-07",
        "end_date": "2026-07-19",
        "prize_pool": "$2,000,000",
        "location": "France Paris",
    },
    {
        "name": "BLAST SLAM VII",
        "start_date": "2026-05-26",
        "end_date": "2026-06-07",
        "prize_pool": "$1,000,000",
        "location": "Denmark Copenhagen",
    },
    {
        "name": "DreamLeague Season 29",
        "start_date": "2026-05-13",
        "end_date": "2026-05-24",
        "prize_pool": "$1,000,000",
        "location": "Europe",
    },
    {
        "name": "PGL Wallachia Season 8",
        "start_date": "2026-04-18",
        "end_date": "2026-04-26",
        "prize_pool": "$1,000,000",
        "location": "Romania Bucharest",
    },
    {
        "name": "ESL One Birmingham 2026",
        "start_date": "2026-03-22",
        "end_date": "2026-03-29",
        "prize_pool": "$1,000,000",
        "location": "United Kingdom Birmingham",
    },
    {
        "name": "PGL Wallachia Season 7",
        "start_date": "2026-03-07",
        "end_date": "2026-03-15",
        "prize_pool": "$1,000,000",
        "location": "Romania Bucharest",
    },
    {
        "name": "DreamLeague Season 28",
        "start_date": "2026-02-16",
        "end_date": "2026-03-01",
        "prize_pool": "$1,000,000",
        "location": "Europe",
    },
]


# We'll discover these tournaments via DLTV scraper
# First, let's find their event IDs/slugs

print("[Steam API] Using key: " + get_steam_key()[:6] + "..." + get_steam_key()[-4:])

from backend.discovery import tracker
from backend.dltv_client import client


def find_tournament_by_name(event_title: str) -> Optional[int]:
    """Find DLTV event ID by tournament name."""
    events = client.get_events()
    
    for event in events:
        if event_title.lower() in event.get("title", "").lower():
            return event.get("id"), event.get("title")
    
    return None


def fetch_tournament_matches(series_id: int) -> List[Dict]:
    """Fetch all series/maps for a tournament."""
    try:
        series_list = client.get_series(series_id)
        if not series_list:
            return []
        
        matches = []
        for series in series_list:
            for map_data in series.get("maps", []):
                if map_data.get("steam_id"):
                    matches.append({
                        "series_id": series.get("id"),
                        "map_id": map_data.get("id"),
                        "steam_id": map_data.get("steam_id"),
                        "team_a": series.get("first_team", {}).get("title"),
                        "team_b": series.get("second_team", {}).get("title"),
                        "event": series.get("event_title"),
                    })
        
        return matches
        
    except Exception as e:
        print(f"[ERROR] Fetching series {series_id}: {e}")
        return []


def main():
    print("="*60)
    print("Tier 1 Tournament Data Collector")
    print("="*60 + "\n")
    
    # Step 1: Discover all current events from DLTV
    print("[Discovery] Scanning DLTV for tournaments...")
    all_events = tracker.get_live_and_prematch()[1] + tracker.get_live_and_prematch()[0]
    event_titles = set(m.get("event") for m in all_events if m.get("event"))
    
    print(f"Found {len(event_titles)} unique event titles:")
    for title in sorted(event_titles):
        print(f"   [EVENT] {title}")
    
    # Step 2: Match against known Tier 1 tournaments
    tier1_events = {}
    unknown_events = []
    
    for title in event_titles:
        found = False
        for t1_tournament in TIER1_TOURNAMENTS:
            target_name = t1_tournament["name"]
            
            # Check if this title matches our Tier 1 list
            if target_name.lower() in title.lower() or title.lower() in target_name.lower():
                print(f"FOUND: {target_name} <- {title}")
                
                # Find DLTV event ID
                event_info = find_tournament_by_name(target_name)
                if event_info:
                    event_id, event_title = event_info
                    tier1_events[target_name] = {
                        "dltv_event_id": event_id,
                        "full_title": event_title,
                        "date_range": f"{t1_tournament['start_date']} - {t1_tournament['end_date']}",
                    }
                else:
                    print(f"   WARNING: Could not find DLTV event ID for {target_name}")
                
                found = True
                break
        
        if not found:
            unknown_events.append(title)
    
    print("\n" + "="*60 + "\n")
    print(f"Identified {len(tier1_events)} Tier 1 tournaments:")
    for name, info in tier1_events.items():
        print(f"   [ID] {name} (DLTV ID: {info['dltv_event_id']})")
    
    # Step 3: Fetch all matches from these tournaments
    print("\n" + "="*60 + "\n")
    print("Fetching matches from identified tournaments...\n")
    
    all_matches = []
    
    for tournament_name, info in tier1_events.items():
        print(f"Processing: {tournament_name}")
        
        try:
            matches = fetch_tournament_matches(info["dltv_event_id"])
            print(f"   -> Found {len(matches)} matches\n")
            all_matches.extend(matches)
            
        except Exception as e:
            print(f"   ERROR: {e}\n")
    
    # Deduplicate by steam_id
    seen_ids = set()
    unique_matches = []
    
    for match in all_matches:
        mid = match.get("steam_id")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique_matches.append(match)
    
    print("-"*60)
    print(f"Total matches collected: {len(unique_matches)}")
    print(f"Ready for enrichment with Steam API MatchDetails!\n")
    
    # Save intermediate result
    output_file = "ml_data/tier1_matches.json"
    os.makedirs("ml_data", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(unique_matches, f, indent=2)
    
    print(f"Saved to: {output_file}")
    print("\n[NOTE] Next step: Enrich these matches with full details!")
    print("   Use: python scripts/enrich_tier1_matches.py\n")
    
    return unique_matches


if __name__ == "__main__":
    matches = main()
