"""
Extract Advanced Hero & Player Features
From existing 1,111 match JSONs:
  1. Hero-specific stats (win rate, avg KDA, GPM per hero)
  2. Player-hero combinations (player performance on specific heroes)
  3. Hero vs hero matchups (counters)
  4. Hero pair synergy (how well two heroes work together)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
ML_DATA = Path("ml_data")


def load_all_matches() -> List[Dict]:
    """Load all match JSONs."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    return matches


def extract_hero_stats(matches: List[Dict]) -> Dict:
    """Extract per-hero statistics."""
    print("\nExtracting hero-specific stats...")
    
    hero_stats = defaultdict(lambda: {
        'picks': 0,
        'wins': 0,
        'total_kills': 0,
        'total_deaths': 0,
        'total_assists': 0,
        'total_gpm': 0,
        'total_xpm': 0,
        'total_hero_damage': 0,
        'total_duration': 0
    })
    
    for match in matches:
        radiant_victory = match['radiant_victory']
        
        for side in ['radiant', 'dire']:
            for p in match[side]['player_performances']:
                hero = p['performance']['hero']['short_name']
                won = (side == 'radiant') == radiant_victory
                perf = p['performance']
                
                stats = hero_stats[hero]
                stats['picks'] += 1
                stats['wins'] += 1 if won else 0
                stats['total_kills'] += perf['kills'] or 0
                stats['total_deaths'] += perf['deaths'] or 0
                stats['total_assists'] += perf['assists'] or 0
                stats['total_gpm'] += perf['gpm'] or 0
                stats['total_xpm'] += perf['xpm'] or 0
                stats['total_hero_damage'] += perf['hero_damage'] or 0
                stats['total_duration'] += match['duration'] / 60.0
    
    # Calculate averages
    result = {}
    for hero, stats in hero_stats.items():
        if stats['picks'] < 5:
            continue
        
        picks = stats['picks']
        result[hero] = {
            'picks': picks,
            'win_rate': round(stats['wins'] / picks, 3),
            'avg_kills': round(stats['total_kills'] / picks, 2),
            'avg_deaths': round(stats['total_deaths'] / picks, 2),
            'avg_assists': round(stats['total_assists'] / picks, 2),
            'avg_kda': round((stats['total_kills'] + stats['total_assists']) / max(stats['total_deaths'], 1) / picks, 2),
            'avg_gpm': round(stats['total_gpm'] / picks, 1),
            'avg_xpm': round(stats['total_xpm'] / picks, 1),
            'avg_hero_damage': round(stats['total_hero_damage'] / picks, 0),
            'avg_duration': round(stats['total_duration'] / picks, 1)
        }
    
    print(f"  Extracted stats for {len(result)} heroes (5+ picks)")
    return result


def extract_player_hero_stats(matches: List[Dict]) -> Dict:
    """Extract player performance on specific heroes."""
    print("\nExtracting player-hero combinations...")
    
    player_hero = defaultdict(lambda: defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'total_kills': 0,
        'total_deaths': 0,
        'total_assists': 0,
        'total_gpm': 0,
        'total_xpm': 0
    }))
    
    for match in matches:
        radiant_victory = match['radiant_victory']
        
        for side in ['radiant', 'dire']:
            for p in match[side]['player_performances']:
                player = p['player']['nickname']
                hero = p['performance']['hero']['short_name']
                won = (side == 'radiant') == radiant_victory
                perf = p['performance']
                
                stats = player_hero[player][hero]
                stats['matches'] += 1
                stats['wins'] += 1 if won else 0
                stats['total_kills'] += perf['kills'] or 0
                stats['total_deaths'] += perf['deaths'] or 0
                stats['total_assists'] += perf['assists'] or 0
                stats['total_gpm'] += perf['gpm'] or 0
                stats['total_xpm'] += perf['xpm'] or 0
    
    # Calculate averages (only for players with 3+ matches on a hero)
    result = {}
    for player, heroes in player_hero.items():
        result[player] = {}
        for hero, stats in heroes.items():
            if stats['matches'] < 3:
                continue
            
            matches = stats['matches']
            result[player][hero] = {
                'matches': matches,
                'win_rate': round(stats['wins'] / matches, 3),
                'avg_kills': round(stats['total_kills'] / matches, 2),
                'avg_deaths': round(stats['total_deaths'] / matches, 2),
                'avg_assists': round(stats['total_assists'] / matches, 2),
                'avg_kda': round((stats['total_kills'] + stats['total_assists']) / max(stats['total_deaths'], 1) / matches, 2),
                'avg_gpm': round(stats['total_gpm'] / matches, 1),
                'avg_xpm': round(stats['total_xpm'] / matches, 1)
            }
    
    # Count players with hero data
    players_with_data = sum(1 for p, h in result.items() if len(h) > 0)
    print(f"  Extracted {players_with_data} players with hero-specific stats")
    return result


def extract_hero_matchups(matches: List[Dict]) -> Dict:
    """Extract hero vs hero matchups (who counters whom)."""
    print("\nExtracting hero vs hero matchups...")
    
    # Track matchups in same lane (mid vs mid, safe vs off, etc.)
    matchups = defaultdict(lambda: defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'total_gpm_diff': 0
    }))
    
    for match in matches:
        radiant_victory = match['radiant_victory']
        
        # Group players by lane
        radiant_lanes = defaultdict(list)
        dire_lanes = defaultdict(list)
        
        for p in match['radiant']['player_performances']:
            lane = p.get('laneInfo', {}).get('lane')
            if lane and lane != 'JUNGLE':
                radiant_lanes[lane].append(p)
        
        for p in match['dire']['player_performances']:
            lane = p.get('laneInfo', {}).get('lane')
            if lane and lane != 'JUNGLE':
                dire_lanes[lane].append(p)
        
        # Compare heroes in same lane
        for lane in radiant_lanes:
            if lane in dire_lanes:
                for r_p in radiant_lanes[lane]:
                    for d_p in dire_lanes[lane]:
                        r_hero = r_p['performance']['hero']['short_name']
                        d_hero = d_p['performance']['hero']['short_name']
                        r_gpm = r_p['performance']['gpm'] or 0
                        d_gpm = d_p['performance']['gpm'] or 0
                        
                        # Radiant hero perspective
                        matchup_r = matchups[r_hero][d_hero]
                        matchup_r['matches'] += 1
                        matchup_r['wins'] += 1 if (r_gpm > d_gpm) else 0
                        matchup_r['total_gpm_diff'] += (r_gpm - d_gpm)
                        
                        # Dire hero perspective
                        matchup_d = matchups[d_hero][r_hero]
                        matchup_d['matches'] += 1
                        matchup_d['wins'] += 1 if (d_gpm > r_gpm) else 0
                        matchup_d['total_gpm_diff'] += (d_gpm - r_gpm)
    
    # Calculate win rates (only for matchups with 3+ games)
    result = {}
    for hero, opponents in matchups.items():
        result[hero] = {}
        for opponent, stats in opponents.items():
            if stats['matches'] < 3:
                continue
            
            result[hero][opponent] = {
                'matches': stats['matches'],
                'win_rate': round(stats['wins'] / stats['matches'], 3),
                'avg_gpm_diff': round(stats['total_gpm_diff'] / stats['matches'], 1)
            }
    
    print(f"  Extracted {len(result)} heroes with matchup data")
    return result


def extract_hero_pair_synergy(matches: List[Dict]) -> Dict:
    """Extract hero pair synergy (how well two heroes work together)."""
    print("\nExtracting hero pair synergy...")
    
    pairs = defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'total_team_kills': 0,
        'total_duration': 0
    })
    
    for match in matches:
        radiant_victory = match['radiant_victory']
        
        for side in ['radiant', 'dire']:
            heroes = [p['performance']['hero']['short_name'] for p in match[side]['player_performances']]
            won = (side == 'radiant') == radiant_victory
            team_kills = sum(p['performance']['kills'] or 0 for p in match[side]['player_performances'])
            
            # Generate all pairs (10 pairs per team)
            for i in range(len(heroes)):
                for j in range(i+1, len(heroes)):
                    pair_key = '_'.join(sorted([heroes[i], heroes[j]]))
                    pairs[pair_key]['matches'] += 1
                    pairs[pair_key]['wins'] += 1 if won else 0
                    pairs[pair_key]['total_team_kills'] += team_kills
                    pairs[pair_key]['total_duration'] += match['duration'] / 60.0
    
    # Calculate stats (only for pairs with 5+ matches)
    result = {}
    for pair, stats in pairs.items():
        if stats['matches'] < 5:
            continue
        
        matches = stats['matches']
        result[pair] = {
            'matches': matches,
            'win_rate': round(stats['wins'] / matches, 3),
            'avg_team_kills': round(stats['total_team_kills'] / matches, 2),
            'avg_duration': round(stats['total_duration'] / matches, 1)
        }
    
    print(f"  Extracted {len(result)} hero pairs with synergy data (5+ matches)")
    return result


def main():
    print("=" * 60)
    print("Extracting Advanced Hero & Player Features")
    print("=" * 60)
    
    matches = load_all_matches()
    print(f"\nLoaded {len(matches)} matches")
    
    # Extract all features
    hero_stats = extract_hero_stats(matches)
    player_hero_stats = extract_player_hero_stats(matches)
    hero_matchups = extract_hero_matchups(matches)
    hero_pair_synergy = extract_hero_pair_synergy(matches)
    
    # Save all to JSON
    output = {
        'hero_stats': hero_stats,
        'player_hero_stats': player_hero_stats,
        'hero_matchups': hero_matchups,
        'hero_pair_synergy': hero_pair_synergy,
        'metadata': {
            'total_matches': len(matches),
            'heroes_with_stats': len(hero_stats),
            'players_with_hero_data': sum(1 for p, h in player_hero_stats.items() if len(h) > 0),
            'heroes_with_matchups': len(hero_matchups),
            'pairs_with_synergy': len(hero_pair_synergy)
        }
    }
    
    output_path = ML_DATA / "advanced_hero_features.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nSaved to {output_path}")
    print(f"\nSummary:")
    print(f"  Heroes with stats: {len(hero_stats)}")
    print(f"  Players with hero data: {output['metadata']['players_with_hero_data']}")
    print(f"  Heroes with matchups: {len(hero_matchups)}")
    print(f"  Hero pairs with synergy: {len(hero_pair_synergy)}")


if __name__ == "__main__":
    main()
