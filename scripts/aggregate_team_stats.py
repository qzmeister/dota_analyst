"""
Team Statistics Aggregator
Collect per-team stats across all collected matches.

Output: ml_data/team_stats.json
"""
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/team_stats.json")


def load_all_matches() -> List[Dict]:
    """Load all match JSON files."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    return matches


def extract_team_stats(match: Dict) -> Dict:
    """Extract statistics for both teams from a match."""
    duration_min = match['duration'] / 60.0
    radiant_victory = match['radiant_victory']
    
    teams = {}
    
    for side in ['radiant', 'dire']:
        team_data = match[side]
        team_name = team_data['team']['name']
        players = team_data['player_performances']
        
        # Basic stats
        total_kills = sum(p['performance']['kills'] or 0 for p in players)
        total_deaths = sum(p['performance']['deaths'] or 0 for p in players)
        total_assists = sum(p['performance']['assists'] or 0 for p in players)
        avg_gpm = sum(p['performance']['gpm'] or 0 for p in players) / 5
        avg_xpm = sum(p['performance']['xpm'] or 0 for p in players) / 5
        total_building_damage = sum(p['performance']['building_damage'] or 0 for p in players)
        total_hero_damage = sum(p['performance']['hero_damage'] or 0 for p in players)
        
        # First blood detection (simplified - check if team has early kills)
        # Note: actual first blood timing requires timeline data
        has_early_kill = any(p['performance']['kills'] >= 2 for p in players)
        
        # Networth advantage from frames
        nw_advantage = 0
        if match.get('frames') and match['frames'].get('radiant_networth_advantage'):
            nw_series = match['frames']['radiant_networth_advantage']
            if nw_series:
                nw_advantage = nw_series[-1] if side == 'radiant' else -nw_series[-1]
        
        teams[team_name] = {
            'kills': total_kills,
            'deaths': total_deaths,
            'assists': total_assists,
            'avg_gpm': avg_gpm,
            'avg_xpm': avg_xpm,
            'building_damage': total_building_damage,
            'hero_damage': total_hero_damage,
            'duration_min': duration_min,
            'won': (side == 'radiant') == radiant_victory,
            'nw_advantage': nw_advantage,
            'patch': match.get('patch', 'unknown'),
            'league': match.get('league', {}).get('name', 'unknown')
        }
    
    return teams


def aggregate_team_stats(all_teams: List[Dict]) -> Dict:
    """Aggregate stats across all matches for each team."""
    team_agg = defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'total_kills': 0,
        'total_deaths': 0,
        'total_assists': 0,
        'total_gpm': 0,
        'total_xpm': 0,
        'total_building_damage': 0,
        'total_hero_damage': 0,
        'total_duration': 0,
        'total_nw_advantage': 0,
        'patches': defaultdict(int),
        'leagues': defaultdict(int)
    })
    
    for match_teams in all_teams:
        for team_name, stats in match_teams.items():
            t = team_agg[team_name]
            t['matches'] += 1
            t['wins'] += 1 if stats['won'] else 0
            t['total_kills'] += stats['kills']
            t['total_deaths'] += stats['deaths']
            t['total_assists'] += stats['assists']
            t['total_gpm'] += stats['avg_gpm']
            t['total_xpm'] += stats['avg_xpm']
            t['total_building_damage'] += stats['building_damage']
            t['total_hero_damage'] += stats['hero_damage']
            t['total_duration'] += stats['duration_min']
            t['total_nw_advantage'] += stats['nw_advantage']
            t['patches'][stats['patch']] += 1
            t['leagues'][stats['league']] += 1
    
    # Calculate averages
    result = {}
    for team_name, t in team_agg.items():
        matches = t['matches']
        if matches < 2:  # Skip teams with only 1 match
            continue
        
        result[team_name] = {
            'matches': matches,
            'win_rate': round(t['wins'] / matches, 3),
            'avg_kills': round(t['total_kills'] / matches, 2),
            'avg_deaths': round(t['total_deaths'] / matches, 2),
            'avg_assists': round(t['total_assists'] / matches, 2),
            'avg_gpm': round(t['total_gpm'] / matches, 1),
            'avg_xpm': round(t['total_xpm'] / matches, 1),
            'avg_building_damage': round(t['total_building_damage'] / matches, 0),
            'avg_hero_damage': round(t['total_hero_damage'] / matches, 0),
            'avg_duration_min': round(t['total_duration'] / matches, 2),
            'avg_nw_advantage': round(t['total_nw_advantage'] / matches, 0),
            'top_patches': dict(sorted(t['patches'].items(), key=lambda x: x[1], reverse=True)[:3]),
            'top_leagues': dict(sorted(t['leagues'].items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    return result


def main():
    print("Loading matches...")
    matches = load_all_matches()
    print(f"Loaded {len(matches)} matches")
    
    print("Extracting team stats from matches...")
    all_teams = []
    for i, match in enumerate(matches, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(matches)} matches")
        try:
            teams = extract_team_stats(match)
            all_teams.append(teams)
        except Exception as e:
            print(f"  Error processing match {match.get('match_id')}: {e}")
    
    print("Aggregating team statistics...")
    team_stats = aggregate_team_stats(all_teams)
    
    # Sort by matches played
    team_stats = dict(sorted(team_stats.items(), key=lambda x: x[1]['matches'], reverse=True))
    
    print(f"\nResults:")
    print(f"  Teams with 2+ matches: {len(team_stats)}")
    print(f"  Top 10 teams by matches played:")
    for i, (team, stats) in enumerate(list(team_stats.items())[:10], 1):
        print(f"    {i}. {team}: {stats['matches']} matches, WR: {stats['win_rate']:.1%}")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(team_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
