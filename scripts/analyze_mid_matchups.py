"""
Mid Lane Matchup Comparison
Compare all mid heroes and their matchups.

Output: ml_data/mid_matchups.json
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/mid_matchups.json")


def get_mid_players(match: Dict) -> List[Dict]:
    """Extract mid lane players from a match."""
    mid_players = []
    
    for side in ['radiant', 'dire']:
        team_data = match[side]
        players = team_data['player_performances']
        
        for p in players:
            lane_info = p.get('laneInfo', {})
            lane = lane_info.get('lane')
            meta_lane = lane_info.get('metaLane')
            
            # Check if player is mid (lane=MIDDLE)
            if lane == 'MIDDLE':
                perf = p['performance']
                mid_players.append({
                    'hero_id': perf['hero']['valve_id'],
                    'hero_name': perf['hero']['short_name'],
                    'player': p['player']['nickname'],
                    'team': team_data['team']['name'],
                    'side': side,
                    'gpm': perf['gpm'] or 0,
                    'xpm': perf['xpm'] or 0,
                    'kills': perf['kills'] or 0,
                    'deaths': perf['deaths'] or 0,
                    'assists': perf['assists'] or 0,
                    'hero_damage': perf['hero_damage'] or 0,
                    'level': perf['level'] or 0,
                    'won': (side == 'radiant') == match['radiant_victory'],
                    'match_id': match['match_id']
                })
                break  # Only one mid per team
    
    return mid_players


def analyze_mid_matchups(all_mids: List[Dict]) -> Dict:
    """Analyze mid hero matchups."""
    # Group mids by match
    match_mids = defaultdict(list)
    for mid in all_mids:
        match_mids[mid['match_id']].append(mid)
    
    hero_stats = defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'total_gpm': 0,
        'total_xpm': 0,
        'total_kills': 0,
        'total_deaths': 0,
        'total_hero_damage': 0,
        'matchup_wins': 0,
        'matchup_losses': 0,
        'strong_vs': defaultdict(int),
        'weak_vs': defaultdict(int)
    })
    
    # For each match, compare radiant mid vs dire mid
    for match_id, mids in match_mids.items():
        if len(mids) != 2:
            continue
        
        radiant_mid = next((m for m in mids if m['side'] == 'radiant'), None)
        dire_mid = next((m for m in mids if m['side'] == 'dire'), None)
        
        if not radiant_mid or not dire_mid:
            continue
        
        r_hero = radiant_mid['hero_name']
        d_hero = dire_mid['hero_name']
        
        # Radiant mid stats
        r_stats = hero_stats[r_hero]
        r_stats['matches'] += 1
        r_stats['wins'] += 1 if radiant_mid['won'] else 0
        r_stats['total_gpm'] += radiant_mid['gpm']
        r_stats['total_xpm'] += radiant_mid['xpm']
        r_stats['total_kills'] += radiant_mid['kills']
        r_stats['total_deaths'] += radiant_mid['deaths']
        r_stats['total_hero_damage'] += radiant_mid['hero_damage']
        
        gpm_diff = radiant_mid['gpm'] - dire_mid['gpm']
        if gpm_diff > 0:
            r_stats['matchup_wins'] += 1
            r_stats['strong_vs'][d_hero] += 1
        else:
            r_stats['matchup_losses'] += 1
            r_stats['weak_vs'][d_hero] += 1
        
        # Dire mid stats
        d_stats = hero_stats[d_hero]
        d_stats['matches'] += 1
        d_stats['wins'] += 1 if dire_mid['won'] else 0
        d_stats['total_gpm'] += dire_mid['gpm']
        d_stats['total_xpm'] += dire_mid['xpm']
        d_stats['total_kills'] += dire_mid['kills']
        d_stats['total_deaths'] += dire_mid['deaths']
        d_stats['total_hero_damage'] += dire_mid['hero_damage']
        
        if gpm_diff < 0:
            d_stats['matchup_wins'] += 1
            d_stats['strong_vs'][r_hero] += 1
        else:
            d_stats['matchup_losses'] += 1
            d_stats['weak_vs'][r_hero] += 1
    
    # Calculate averages and format output
    result = {}
    for hero, stats in hero_stats.items():
        if stats['matches'] < 3:
            continue
        
        matches = stats['matches']
        strong_vs = dict(sorted(stats['strong_vs'].items(), key=lambda x: x[1], reverse=True)[:5])
        weak_vs = dict(sorted(stats['weak_vs'].items(), key=lambda x: x[1], reverse=True)[:5])
        
        result[hero] = {
            'hero': hero,
            'matches': matches,
            'win_rate': round(stats['wins'] / matches, 3),
            'avg_gpm': round(stats['total_gpm'] / matches, 1),
            'avg_xpm': round(stats['total_xpm'] / matches, 1),
            'avg_kills': round(stats['total_kills'] / matches, 2),
            'avg_deaths': round(stats['total_deaths'] / matches, 2),
            'avg_hero_damage': round(stats['total_hero_damage'] / matches, 0),
            'matchup_win_rate': round(stats['matchup_wins'] / matches, 3) if matches > 0 else 0,
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
    
    print("Extracting mid lane players...")
    all_mids = []
    for i, match in enumerate(matches, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(matches)} matches")
        try:
            mids = get_mid_players(match)
            all_mids.extend(mids)
        except Exception as e:
            print(f"  Error processing match {match.get('match_id')}: {e}")
    
    print(f"Extracted {len(all_mids)} mid player performances")
    
    print("Analyzing mid matchups...")
    mid_stats = analyze_mid_matchups(all_mids)
    
    # Sort by matches
    mid_stats = dict(sorted(mid_stats.items(), key=lambda x: x[1]['matches'], reverse=True))
    
    print(f"\nResults:")
    print(f"  Unique mid heroes with 3+ matches: {len(mid_stats)}")
    print(f"  Top 15 mid heroes by matches:")
    for i, (hero, stats) in enumerate(list(mid_stats.items())[:15], 1):
        print(f"    {i}. {hero}: {stats['matches']} matches, WR: {stats['win_rate']:.1%}, Avg GPM: {stats['avg_gpm']}")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(mid_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
