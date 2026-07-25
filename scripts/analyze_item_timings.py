"""
Item Timing Analysis
Track when key items are purchased and correlate with win rate.

Output: ml_data/item_timings.json
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/item_timings.json")

# Key items to track
KEY_ITEMS = {
    'item_boots': 'Boots',
    'item_power_treads': 'Power Treads',
    'item_phase_boots': 'Phase Boots',
    'item_travel_boots': 'Travel Boots',
    'item_blink': 'Blink Dagger',
    'item_black_king_bar': 'BKB',
    'item_butterfly': 'Butterfly',
    'item_daedalus': 'Daedalus',
    'item_radiance': 'Radiance',
    'item_mjollnir': 'Mjollnir',
    'item_satanic': 'Satanic',
    'item_heart': 'Heart of Tarrasque',
    'item_shivas_guard': 'Shiva\'s Guard',
    'item_sheepstick': 'Scythe of Vyse',
    'item_orchid': 'Orchid Malevolence',
    'item_bloodthorn': 'Bloodthorn',
    'item_nullifier': 'Nullifier',
    'item_hurricane_pike': 'Hurricane Pike',
    'item_manta': 'Manta Style',
    'item_linkens': 'Linken\'s Sphere',
    'item_silver_edge': 'Silver Edge',
    'item_abyssal_blade': 'Abyssal Blade',
    'item_assault': 'Assault Cuirass',
    'item_crimson_guard': 'Crimson Guard',
    'item_pipe': 'Pipe of Insight',
    'item_force_staff': 'Force Staff',
    'item_ghost_scepter': 'Ghost Scepter',
    'item_glimmer_cape': 'Glimmer Cape',
    'item_aeon_disk': 'Aeon Disk',
    'item_solar_crest': 'Solar Crest',
    'item_medallion': 'Medallion of Courage',
    'item_ward_observer': 'Observer Ward',
    'item_ward_sentry': 'Sentry Ward',
    'item_dust': 'Dust of Appearance',
    'item_smoke': 'Smoke of Deception',
    'item_gem': 'Gem of True Sight'
}


def analyze_item_timings():
    """Analyze item purchase timings."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"Loaded {len(matches)} matches")
    
    # Track item timings per hero
    hero_items = defaultdict(lambda: defaultdict(lambda: {
        'purchases': 0,
        'total_time': 0,
        'wins': 0,
        'times': []
    }))
    
    for i, match in enumerate(matches, 1):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(matches)} matches")
        
        radiant_victory = match['radiant_victory']
        
        for side in ['radiant', 'dire']:
            for p in match[side]['player_performances']:
                hero_name = p['performance']['hero']['short_name']
                won = (side == 'radiant') == radiant_victory
                items = p['performance'].get('items', [])
                
                for item in items:
                    item_name = item.get('name')
                    item_time = item.get('time', 0)
                    
                    # Skip pre-game items (negative time) and wards
                    if item_time < 0 or item_name in ['item_ward_observer', 'item_ward_sentry', 'item_dust', 'item_smoke', 'item_gem', 'item_tpscroll']:
                        continue
                    
                    if item_name in KEY_ITEMS:
                        item_stats = hero_items[hero_name][item_name]
                        item_stats['purchases'] += 1
                        item_stats['total_time'] += item_time
                        item_stats['wins'] += 1 if won else 0
                        item_stats['times'].append(item_time)
    
    # Calculate averages
    result = {}
    for hero, items in hero_items.items():
        result[hero] = {}
        for item_name, stats in items.items():
            if stats['purchases'] < 5:
                continue
            
            times = stats['times']
            avg_time = stats['total_time'] / stats['purchases']
            
            # Calculate percentiles
            times_sorted = sorted(times)
            n = len(times_sorted)
            early_threshold = times_sorted[int(n * 0.25)] if n >= 4 else avg_time
            
            result[hero][item_name] = {
                'display_name': KEY_ITEMS.get(item_name, item_name),
                'purchases': stats['purchases'],
                'avg_time_sec': round(avg_time, 0),
                'avg_time_min': round(avg_time / 60, 2),
                'win_rate': round(stats['wins'] / stats['purchases'] * 100, 2),
                'early_threshold_sec': round(early_threshold, 0),
                'min_time': round(min(times), 0),
                'max_time': round(max(times), 0)
            }
        
        # Sort by purchases
        result[hero] = dict(sorted(result[hero].items(), key=lambda x: x[1]['purchases'], reverse=True))
    
    return result


def main():
    print("Analyzing item timings...")
    item_timings = analyze_item_timings()
    
    print(f"\nResults:")
    print(f"  Heroes with item data: {len(item_timings)}")
    
    # Show sample
    sample_heroes = list(item_timings.keys())[:5]
    for hero in sample_heroes:
        print(f"\n  {hero}:")
        for item, stats in list(item_timings[hero].items())[:5]:
            print(f"    {stats['display_name']}: {stats['purchases']} purchases, avg {stats['avg_time_min']:.1f} min, WR {stats['win_rate']:.1f}%")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(item_timings, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
