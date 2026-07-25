"""
Tournament Tier Weighting
Weight training data by tournament tier.

Tier 1: Premier tournaments (The International, DreamLeague, ESL, PGL, etc.)
Tier 2: Major tournaments
Tier 3: Minor/Qualifier tournaments

Output: ml_data/tournament_tiers.json
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/tournament_tiers.json")

# Tier 1 tournaments (Premier)
TIER1_KEYWORDS = [
    'the international', 'ti', 'dreamleague', 'esl', 'pgl', 
    'riot games', 'valve', 'major', 'championship',
    'epicenter', 'mdl macau', 'one birmingham', 'buku',
    'esports world cup', 'ewc', 'cave da'
]

# Tier 2 tournaments (Major)
TIER2_KEYWORDS = [
    'league', 'cup', 'masters', 'invitational', 'open',
    'challenge', 'series', 'classic', 'pro', 'elite'
]


def determine_tier(league_name: str) -> int:
    """Determine tournament tier from league name."""
    name_lower = league_name.lower()
    
    # Check Tier 1 first
    for keyword in TIER1_KEYWORDS:
        if keyword in name_lower:
            return 1
    
    # Check Tier 2
    for keyword in TIER2_KEYWORDS:
        if keyword in name_lower:
            return 2
    
    # Default to Tier 3
    return 3


def analyze_tournament_tiers():
    """Analyze tournament tiers and match weights."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"Loaded {len(matches)} matches")
    
    # Track leagues
    leagues = defaultdict(lambda: {
        'matches': 0,
        'tier': 3,
        'weight': 0.4
    })
    
    for match in matches:
        league_name = match.get('league', {}).get('name', 'Unknown')
        leagues[league_name]['matches'] += 1
    
    # Determine tiers
    for league_name in leagues:
        tier = determine_tier(league_name)
        leagues[league_name]['tier'] = tier
        
        # Assign weights
        if tier == 1:
            leagues[league_name]['weight'] = 1.0
        elif tier == 2:
            leagues[league_name]['weight'] = 0.7
        else:
            leagues[league_name]['weight'] = 0.4
    
    # Count by tier
    tier_counts = defaultdict(int)
    for league, data in leagues.items():
        tier_counts[data['tier']] += data['matches']
    
    # Sort by matches
    leagues = dict(sorted(leagues.items(), key=lambda x: x[1]['matches'], reverse=True))
    
    return leagues, tier_counts


def main():
    print("Analyzing tournament tiers...")
    leagues, tier_counts = analyze_tournament_tiers()
    
    total_matches = sum(tier_counts.values())
    
    print(f"\nResults:")
    print(f"  Total leagues: {len(leagues)}")
    print(f"\n  Tier distribution:")
    for tier in [1, 2, 3]:
        count = tier_counts[tier]
        pct = count / total_matches * 100
        weight = {1: 1.0, 2: 0.7, 3: 0.4}[tier]
        print(f"    Tier {tier}: {count} matches ({pct:.1f}%), weight: {weight}")
    
    print(f"\n  Top 15 leagues:")
    for i, (league, data) in enumerate(list(leagues.items())[:15], 1):
        print(f"    {i}. {league}: Tier {data['tier']}, {data['matches']} matches, weight {data['weight']}")
    
    # Save to JSON
    result = {
        'total_leagues': len(leagues),
        'tier_counts': dict(tier_counts),
        'tier_weights': {1: 1.0, 2: 0.7, 3: 0.4},
        'leagues': leagues
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
