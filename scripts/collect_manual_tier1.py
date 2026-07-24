"""
Manual Tier 1 Tournament Collection.

Manually specifies tournament slugs to fetch, regardless of live/prematch status.
This is better than relying on discovery for historical data.

Usage:
    python scripts/collect_manual_tier1.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.dltv_client import client
import json


def main():
    print("[MANUAL] Collecting Tier 1 matches...")
    
    # Known Tier 1 tournaments with their URL slugs
    TIER1_TOURNAMENTS = [
        {"name": "Esports World Cup 2026", "slug": "esports-world-cup-2026"},
        {"name": "BLAST SLAM VII", "slug": "blast-slam-vii"},
        {"name": "DreamLeague Season 29", "slug": "dreamleague-season-29"},
        {"name": "PGL Wallachia Season 8", "slug": "pgl-wallachia-season-8"},
        {"name": "ESL One Birmingham 2026", "slug": "esl-one-birmingham-2026"},
        {"name": "PGL Wallachia Season 7", "slug": "pgl-wallachia-season-7"},
        {"name": "DreamLeague Season 28", "slug": "dreamleague-season-28"},
    ]
    
    all_matches = []
    
    print(f"Fetching {len(TIER1_TOURNAMENTS)} tournaments...")
    
    for t in TIER1_TOURNAMENTS:
        print(f"\n[FETCHING] {t['name']} ({t['slug']})")
        
        try:
            event_id = find_event_by_slug(t["slug"])
            if event_id:
                matches = fetch_all_series(event_id)
                
                for match in matches:
                    # Only add if steam_id exists and not duplicate
                    mid = match.get("steam_id")
                    if mid and not any(m.get("steam_id") == mid for m in all_matches):
                        all_matches.append({
                            **match,
                            "tournament": t["name"],
                        })
                
                print(f"   -> Found {len(matches)} unique matches")
            else:
                print(f"   WARNING: Could not find event ID for {t['slug']}")
                
        except Exception as e:
            print(f"   ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"Total Tier 1 matches collected: {len(all_matches)}")
    
    # Save
    output_file = "ml_data/tier1_matches.json"
    os.makedirs("ml_data", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_matches, f, indent=2)
    
    print(f"Saved to: {output_file}\n")
    return len(all_matches)


def find_event_by_slug(slug: str) -> int:
    """Find DLTV event ID by URL slug."""
    events = client.get_events()
    
    for event in events:
        title = event.get("title", "")
        if slug.lower() in title.lower() or title.lower() in slug.lower():
            return event.get("id")
    
    # Also try getting all series from active events
    for event in events:
        try:
            series_list = client.get_series(event.get("id"))
            for series in series_list or []:
                if series.get("slug") and slug.lower() in series.get("slug").lower():
                    return event.get("id")
        except:
            continue
    
    return None


def fetch_all_series(event_id: int):
    """Fetch all maps from a tournament's series."""
    try:
        series_list = client.get_series(event_id) or []
        matches = []
        
        for series in series_list:
            for map_data in series.get("maps") or []:
                if map_data.get("steam_id"):
                    matches.append({
                        "steam_id": map_data["steam_id"],
                        "team_a": series.get("first_team", {}).get("title"),
                        "team_b": series.get("second_team", {}).get("title"),
                    })
        
        return matches
        
    except Exception as e:
        print(f"[ERROR] Fetching series {event_id}: {e}")
        return []


if __name__ == "__main__":
    count = main()
    print(f"\nDone! Collected {count} matches from Tier 1 tournaments.")
