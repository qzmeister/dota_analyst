"""
Hero Pair Lane Dominance Analysis
Calculate lane pair strength using GPM difference vs direct opponent pair.

Output: ml_data/hero_pair_stats.json
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/hero_pair_stats.json")


def get_lane_pairs(match: Dict) -> List[Dict]:
    """Extract lane pairs from a match."""
    pairs = []
    
    for side in ['radiant', 'dire']:
        team_data = match[side]
        team_name = team_data['team']['name']
        players = team_data['player_performances']
        
        # Group players by lane
        lanes = defaultdict(list)
        for p in players:
            lane_info = p.get('laneInfo', {})
            lane = lane_info.get('lane')
            if lane and lane != 'JUNGLE':
                hero_id = p['performance']['hero']['valve_id']
                hero_name = p['performance']['hero']['short_name']
                gpm = p['performance']['gpm'] or 0
                lanes[lane].append({
                    'hero_id': hero_id,
                    'hero_name': hero_name,
                    'gpm': gpm,
                    'player': p['player']['nickname']
                })
        
        # Create pairs for lanes with 2 players
        for lane, heroes in lanes.items():
            if len(heroes) == 2:
                pair_key = f"{heroes[0]['hero_id']}_{heroes[1]['hero_id']}_{lane.lower()}"
                pairs.append({
                    'pair_key': pair_key,
                    'heroes': [heroes[0]['hero_id'], heroes[1]['hero_id']],
                    'hero_names': [heroes[0]['hero_name'], heroes[1]['hero_name']],
                    'lane': lane,
                    'side': side,
                    'team': team_name,
                    'total_gpm': heroes[0]['gpm'] + heroes[1]['gpm'],
                    'match_id': match['match_id'],
                    'won': (side == 'radiant') == match['radiant_victory']
                })
    
    return pairs


def analyze_lane_dominance(all_pairs: List[Dict]) -> Dict:
    """Analyze lane dominance for each pair vs opponents."""
    # Group pairs by match and lane
    match_lanes = defaultdict(list)
    for pair in all_pairs:
        key = (pair['match_id'], pair['lane'])
        match_lanes[key].append(pair)
    
    pair_stats = defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'total_gpm': 0,
        'dominance_wins': 0,  # Won lane (higher GPM)
        'dominance_losses': 0,
        'strong_vs': defaultdict(int),
        'weak_vs': defaultdict(int)
    })
    
    # For each match+lane, compare radiant vs dire pairs
    for (match_id, lane), pairs in match_lanes.items():
        radiant_pairs = [p for p in pairs if p['side'] == 'radiant']
        dire_pairs = [p for p in pairs if p['side'] == 'dire']
        
        # Compare each radiant pair vs each dire pair in same lane
        for r_pair in radiant_pairs:
            for d_pair in dire_pairs:
                r_key = r_pair['pair_key']
                d_key = d_pair['pair_key']
                
                # Radiant pair stats
                r_stats = pair_stats[r_key]
                r_stats['matches'] += 1
                r_stats['wins'] += 1 if r_pair['won'] else 0
                r_stats['total_gpm'] += r_pair['total_gpm']
                
                gpm_diff = r_pair['total_gpm'] - d_pair['total_gpm']
                if gpm_diff > 0:
                    r_stats['dominance_wins'] += 1
                    r_stats['strong_vs'][d_key] += 1
                else:
                    r_stats['dominance_losses'] += 1
                    r_stats['weak_vs'][d_key] += 1
                
                # Dire pair stats
                d_stats = pair_stats[d_key]
                d_stats['matches'] += 1
                d_stats['wins'] += 1 if d_pair['won'] else 0
                d_stats['total_gpm'] += d_pair['total_gpm']
                
                if gpm_diff < 0:
                    d_stats['dominance_wins'] += 1
                    d_stats['strong_vs'][r_key] += 1
                else:
                    d_stats['dominance_losses'] += 1
                    d_stats['weak_vs'][r_key] += 1
    
    # Calculate averages and format output
    result = {}
    for pair_key, stats in pair_stats.items():
        if stats['matches'] < 3:  # Skip pairs with < 3 matchups
            continue
        
        avg_gpm = stats['total_gpm'] / stats['matches']
        lane_win_rate = stats['dominance_wins'] / stats['matches'] if stats['matches'] > 0 else 0
        
        # Get top opponents
        strong_vs = dict(sorted(stats['strong_vs'].items(), key=lambda x: x[1], reverse=True)[:3])
        weak_vs = dict(sorted(stats['weak_vs'].items(), key=lambda x: x[1], reverse=True)[:3])
        
        result[pair_key] = {
            'pair_key': pair_key,
            'matches': stats['matches'],
            'win_rate': round(stats['wins'] / stats['matches'], 3),
            'lane_win_rate': round(lane_win_rate, 3),
            'avg_total_gpm': round(avg_gpm, 1),
            'strong_vs': strong_vs,
            'weak_vs': weak_vs
        }
    
    return result


def main():
    print("Loading matches...")
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    print(f"Loaded {len(matches)} matches")
    
    print("Extracting lane pairs...")
    all_pairs = []
    for i, match in enumerate(matches, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(matches)} matches")
        try:
            pairs = get_lane_pairs(match)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  Error processing match {match.get('match_id')}: {e}")
    
    print(f"Extracted {len(all_pairs)} lane pairs")
    
    print("Analyzing lane dominance...")
    pair_stats = analyze_lane_dominance(all_pairs)
    
    # Sort by matches
    pair_stats = dict(sorted(pair_stats.items(), key=lambda x: x[1]['matches'], reverse=True))
    
    print(f"\nResults:")
    print(f"  Unique pairs with 3+ matches: {len(pair_stats)}")
    print(f"  Top 10 pairs by matches:")
    for i, (pair, stats) in enumerate(list(pair_stats.items())[:10], 1):
        print(f"    {i}. {pair}: {stats['matches']} matches, WR: {stats['win_rate']:.1%}, Lane WR: {stats['lane_win_rate']:.1%}")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(pair_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
