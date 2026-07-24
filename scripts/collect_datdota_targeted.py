"""
Targeted Tier 1 Tournament Collection via DatDota API.

Collects ONLY the 7 specific tournaments the user requested:
1. Esports World Cup 2026 (ID: 19785)
2. BLAST SLAM VII
3. DreamLeague Season 29
4. PGL Wallachia Season 8
5. ESL One Birmingham 2026
6. PGL Wallachia Season 7
7. DreamLeague Season 28

Much faster than fetching all leagues!
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.datdota_client import get_league_matches, get_leagues


# Known Tier 1 tournament league IDs (from DatDota)
# All 8 tournaments the user requested
TIER1_TOURNAMENTS = [
    {"name_pattern": "Esports World Cup 2026", "league_id": 19785},
    {"name_pattern": "BLAST SLAM VII", "league_id": 19101},
    {"name_pattern": "DreamLeague Season 29", "league_id": 19696},
    {"name_pattern": "PGL Wallachia 2026 Season 8", "league_id": 19543},
    {"name_pattern": "ESL One Birmingham 2026", "league_id": 19422},
    {"name_pattern": "PGL Wallachia 2026 Season 7", "league_id": 19435},
    {"name_pattern": "DreamLeague Season 28", "league_id": 19269},
    {"name_pattern": "1win Essence I", "league_id": 19656},
]


def find_league_id(name_pattern: str) -> int:
    """Search for a league by name pattern and return its ID."""
    print(f"[SEARCH] Looking for: {name_pattern}")
    
    # Fetch first 500 leagues (should cover recent Tier 1)
    data = get_leagues(limit=500, offset=0)
    if not data or "data" not in data:
        return None
    
    for league in data["data"]:
        league_name = league.get("name") or ""
        if name_pattern.lower() in league_name.lower():
            league_id = league.get("leagueId")
            tier = league.get("tier", {}).get("name", "Unknown")
            print(f"   [FOUND] {league_name} (ID: {league_id}, Tier: {tier})")
            return league_id
    
    print(f"   [NOT FOUND]")
    return None


def main():
    print("="*60)
    print("Targeted Tier 1 Tournament Collection")
    print("="*60)
    print()
    
    # Step 1: Find league IDs for tournaments we don't know yet
    print("[STEP 1] Finding league IDs...")
    tournaments_to_fetch = []
    
    for t in TIER1_TOURNAMENTS:
        if t["league_id"]:
            tournaments_to_fetch.append(t)
        else:
            league_id = find_league_id(t["name_pattern"])
            if league_id:
                tournaments_to_fetch.append({
                    "name_pattern": t["name_pattern"],
                    "league_id": league_id
                })
            time.sleep(3)  # Rate limit
    
    print(f"\n[STEP 1] Found {len(tournaments_to_fetch)} tournaments to fetch\n")
    
    # Step 2: Fetch matches for each tournament
    print("[STEP 2] Fetching matches...")
    all_matches = []
    
    for t in tournaments_to_fetch:
        league_id = t["league_id"]
        name = t["name_pattern"]
        
        print(f"\n{'='*60}")
        print(f"Tournament: {name} (ID: {league_id})")
        print(f"{'='*60}")
        
        data = get_league_matches(league_id)
        
        if not data or "data" not in data:
            print(f"[ERROR] No data for {name}")
            time.sleep(3)
            continue
        
        league_data = data["data"]
        matches_data = league_data.get("matches", {})
        matches = matches_data.get("data", [])
        
        print(f"Matches: {len(matches)}")
        print(f"Radiant wins: {matches_data.get('radiantWins', 0)}")
        print(f"Dire wins: {matches_data.get('direWins', 0)}")
        print(f"Avg duration: {matches_data.get('avgDuration', 0):.1f}s")
        
        # Add metadata to each match
        for match in matches:
            match["tournament_name"] = name
            match["tournament_tier"] = 1
            match["league_id"] = league_id
        
        all_matches.extend(matches)
        
        # Rate limit
        time.sleep(3)
    
    # Step 3: Deduplicate and save
    print(f"\n{'='*60}")
    print("[STEP 3] Saving data...")
    
    seen_ids = set()
    unique_matches = []
    
    for match in all_matches:
        mid = match.get("matchId")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique_matches.append(match)
    
    output_file = "ml_data/datdota_tier1_matches.json"
    os.makedirs("ml_data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_matches, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"DONE! Collected {len(unique_matches)} unique Tier 1 matches")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Summary by tournament
    print("\nSummary by tournament:")
    by_tournament = {}
    for match in unique_matches:
        t_name = match.get("tournament_name", "Unknown")
        by_tournament[t_name] = by_tournament.get(t_name, 0) + 1
    
    for t_name, count in sorted(by_tournament.items()):
        print(f"   {t_name}: {count} matches")


if __name__ == "__main__":
    main()
