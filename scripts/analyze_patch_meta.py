"""
Patch Meta Analysis
Track hero win rates and pick rates across patches.

Output: ml_data/patch_meta.json
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/patch_meta.json")


def analyze_patch_meta():
    """Analyze hero stats per patch."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"Loaded {len(matches)} matches")
    
    # Track hero stats per patch
    patch_heroes = defaultdict(lambda: defaultdict(lambda: {
        'picks': 0,
        'wins': 0,
        'total_gpm': 0,
        'total_xpm': 0
    }))
    
    patch_matches = defaultdict(int)
    
    for i, match in enumerate(matches, 1):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(matches)} matches")
        
        patch = match.get('patch', 'unknown')
        radiant_victory = match['radiant_victory']
        patch_matches[patch] += 1
        
        for side in ['radiant', 'dire']:
            for p in match[side]['player_performances']:
                hero_name = p['performance']['hero']['short_name']
                won = (side == 'radiant') == radiant_victory
                
                hero_stats = patch_heroes[patch][hero_name]
                hero_stats['picks'] += 1
                hero_stats['wins'] += 1 if won else 0
                hero_stats['total_gpm'] += p['performance']['gpm'] or 0
                hero_stats['total_xpm'] += p['performance']['xpm'] or 0
    
    # Calculate rates
    result = {}
    for patch, heroes in patch_heroes.items():
        total_matches = patch_matches[patch]
        result[patch] = {
            'total_matches': total_matches,
            'heroes': {}
        }
        
        for hero, stats in heroes.items():
            if stats['picks'] < 3:
                continue
            
            picks = stats['picks']
            result[patch]['heroes'][hero] = {
                'picks': picks,
                'pick_rate': round(picks / (total_matches * 5) * 100, 2),  # % of total hero slots
                'win_rate': round(stats['wins'] / picks * 100, 2),
                'avg_gpm': round(stats['total_gpm'] / picks, 1),
                'avg_xpm': round(stats['total_xpm'] / picks, 1)
            }
        
        # Sort by pick rate
        result[patch]['heroes'] = dict(sorted(
            result[patch]['heroes'].items(),
            key=lambda x: x[1]['pick_rate'],
            reverse=True
        ))
    
    return result


def main():
    print("Analyzing patch meta...")
    patch_meta = analyze_patch_meta()
    
    print(f"\nResults:")
    for patch, data in sorted(patch_meta.items()):
        print(f"\n  Patch {patch}: {data['total_matches']} matches")
        top_heroes = list(data['heroes'].items())[:10]
        print(f"    Top 10 heroes by pick rate:")
        for hero, stats in top_heroes:
            print(f"      {hero}: {stats['pick_rate']}% pick rate, {stats['win_rate']}% WR")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(patch_meta, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
