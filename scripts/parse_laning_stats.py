"""
Parser for DatDota Laning Statistics CSV export.

Converts raw CSV into structured team laning profiles for ML features.

CSV columns (24 total):
- Team, Games, Lanes Won
- NW Adv, LVL Adv
- Overall: Exc%, Won%, Draw%, Lost%, Terr%
- Lane 1 (Safe): Exc%, Won%, Draw%, Lost%, Terr%
- Lane 2 (Offlane): Exc%, Won%, Draw%, Lost%, Terr%
- Lane 3 (Mid): Exc%, Won%, Draw%, Lost%, Terr%
- FB%, Twr Dest, Twr Lost, Twr Δ
"""

import csv
import json
import os
from typing import Dict, List, Optional


CSV_PATH = "ml_data/datdota_laning_stats.csv"
OUTPUT_PATH = "ml_data/team_laning_profiles.json"


def parse_laning_csv(csv_path: str = CSV_PATH) -> List[Dict]:
    """
    Parse DatDota laning stats CSV into structured team profiles.
    
    Returns list of team dicts with laning statistics.
    """
    teams = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header row
        
        for row in reader:
            if len(row) < 24:
                continue  # Skip malformed rows
            
            team_name = row[0].strip()
            
            # Skip teams with very few games (unreliable stats)
            games = int(row[1])
            if games < 5:
                continue
            
            profile = {
                "team_name": team_name,
                "games": games,
                "lanes_won_ratio": float(row[2]),
                
                # Overall advantages
                "net_worth_advantage": float(row[3]),
                "level_advantage": float(row[4]),
                
                # Overall lane performance
                "overall": {
                    "exceed_pct": float(row[5]),
                    "win_pct": float(row[6]),
                    "draw_pct": float(row[7]),
                    "loss_pct": float(row[8]),
                    "terrorize_pct": float(row[9]),
                },
                
                # Lane-specific performance (3 lanes)
                # DatDota order: Safe Lane, Offlane, Mid
                "safe_lane": {
                    "exceed_pct": float(row[10]),
                    "win_pct": float(row[11]),
                    "draw_pct": float(row[12]),
                    "loss_pct": float(row[13]),
                    "terrorize_pct": float(row[14]),
                },
                "offlane": {
                    "exceed_pct": float(row[15]),
                    "win_pct": float(row[16]),
                    "draw_pct": float(row[17]),
                    "loss_pct": float(row[18]),
                    "terrorize_pct": float(row[19]),
                },
                "mid": {
                    "exceed_pct": float(row[20]),
                    "win_pct": float(row[21]),
                    "draw_pct": float(row[22]),
                    "loss_pct": float(row[23]),
                    "terrorize_pct": float(row[24]) if len(row) > 24 else 0.0,
                },
                
                # Early game & tower control
                "first_blood_pct": float(row[25]) if len(row) > 25 else 0.0,
                "tower_destruction_rate": float(row[26]) if len(row) > 26 else 0.0,
                "tower_lost_rate": float(row[27]) if len(row) > 27 else 0.0,
                "tower_delta": float(row[28]) if len(row) > 28 else 0.0,
            }
            
            teams.append(profile)
    
    return teams


def build_laning_lookup(teams: List[Dict]) -> Dict[str, Dict]:
    """
    Build a lookup dict: team_name -> profile
    
    Normalizes team names for fuzzy matching.
    """
    lookup = {}
    
    for team in teams:
        # Normalize name for matching
        name = team["team_name"].lower().strip()
        name = name.replace("  ", " ")  # Remove double spaces
        
        lookup[name] = team
        
        # Also add common variations
        if " " in name:
            # Add without spaces
            lookup[name.replace(" ", "")] = team
        
        # Add abbreviated forms
        words = name.split()
        if len(words) > 1:
            # First letter of each word
            abbrev = "".join(w[0] for w in words if w)
            lookup[abbrev] = team
    
    return lookup


def get_team_laning_stats(team_name: str, lookup: Dict[str, Dict]) -> Optional[Dict]:
    """
    Get laning stats for a team by name (fuzzy match).
    
    Args:
        team_name: Team name to look up
        lookup: Team lookup dict from build_laning_lookup()
    
    Returns:
        Team profile dict or None if not found
    """
    # Try exact match first
    name = team_name.lower().strip()
    if name in lookup:
        return lookup[name]
    
    # Try without spaces
    name_no_spaces = name.replace(" ", "")
    if name_no_spaces in lookup:
        return lookup[name_no_spaces]
    
    # Try partial match
    for key, profile in lookup.items():
        if name in key or key in name:
            return profile
    
    return None


def main():
    """Parse CSV and save team laning profiles."""
    print("="*60)
    print("DatDota Laning Stats Parser")
    print("="*60)
    
    # Parse CSV
    print(f"\n[1/3] Parsing {CSV_PATH}...")
    teams = parse_laning_csv()
    print(f"   [OK] Parsed {len(teams)} teams with 5+ games")
    
    # Build lookup
    print(f"\n[2/3] Building team lookup...")
    lookup = build_laning_lookup(teams)
    print(f"   [OK] Created lookup with {len(lookup)} entries")
    
    # Save to JSON
    print(f"\n[3/3] Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    output_data = {
        "metadata": {
            "source": "DatDota Laning Statistics",
            "patch": "7.41",
            "tier": "1,2",
            "date_range": "2010-01-01 to 2026-07-24",
            "total_teams": len(teams),
        },
        "teams": teams,
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"   [OK] Saved {len(teams)} team profiles")
    
    # Print top teams by net worth advantage
    print(f"\n{'='*60}")
    print("TOP 10 TEAMS BY NET WORTH ADVANTAGE:")
    print(f"{'='*60}")
    
    sorted_teams = sorted(teams, key=lambda t: t["net_worth_advantage"], reverse=True)
    
    for i, team in enumerate(sorted_teams[:10], 1):
        print(f"{i:2d}. {team['team_name']:<25s} "
              f"NW: {team['net_worth_advantage']:+7.1f}  "
              f"LVL: {team['level_advantage']:+.2f}  "
              f"Games: {team['games']}")
    
    print(f"\n{'='*60}")
    print("TOP 10 TEAMS BY SAFE LANE WIN RATE:")
    print(f"{'='*60}")
    
    sorted_safe = sorted(teams, key=lambda t: t["safe_lane"]["win_pct"], reverse=True)
    
    for i, team in enumerate(sorted_safe[:10], 1):
        safe = team["safe_lane"]
        print(f"{i:2d}. {team['team_name']:<25s} "
              f"Safe Win%: {safe['win_pct']:5.1f}%  "
              f"Exc%: {safe['exceed_pct']:5.1f}%  "
              f"Games: {team['games']}")
    
    print(f"\n[OK] Done! Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
