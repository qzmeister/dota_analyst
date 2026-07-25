"""
Player Form Tracking
Track individual player performance over recent matches.

Output: ml_data/player_form.json
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/player_form.json")


def analyze_player_form():
    """Analyze player performance trends."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"Loaded {len(matches)} matches")
    
    # Sort matches by date
    matches.sort(key=lambda m: m.get('start_date', 0))
    
    # Track player performances
    player_history = defaultdict(list)
    
    for i, match in enumerate(matches, 1):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(matches)} matches")
        
        match_id = match['match_id']
        radiant_victory = match['radiant_victory']
        
        for side in ['radiant', 'dire']:
            for p in match[side]['player_performances']:
                player_name = p['player']['nickname']
                hero_name = p['performance']['hero']['short_name']
                won = (side == 'radiant') == radiant_victory
                
                perf = p['performance']
                kda = (perf['kills'] or 0) + (perf['assists'] or 0) - (perf['deaths'] or 0)
                
                player_history[player_name].append({
                    'match_id': match_id,
                    'hero': hero_name,
                    'won': won,
                    'kills': perf['kills'] or 0,
                    'deaths': perf['deaths'] or 0,
                    'assists': perf['assists'] or 0,
                    'gpm': perf['gpm'] or 0,
                    'xpm': perf['xpm'] or 0,
                    'kda': kda,
                    'hero_damage': perf['hero_damage'] or 0
                })
    
    # Calculate form metrics
    result = {}
    for player, history in player_history.items():
        if len(history) < 5:
            continue
        
        # Recent matches (last 10)
        recent = history[-10:]
        
        # Calculate trends
        recent_wins = sum(1 for m in recent if m['won'])
        recent_kda = sum(m['kda'] for m in recent) / len(recent)
        recent_gpm = sum(m['gpm'] for m in recent) / len(recent)
        recent_xpm = sum(m['xpm'] for m in recent) / len(recent)
        
        # Hero pool
        heroes_played = list(set(m['hero'] for m in history))
        
        # All-time stats
        all_wins = sum(1 for m in history if m['won'])
        all_kda = sum(m['kda'] for m in history) / len(history)
        all_gpm = sum(m['gpm'] for m in history) / len(history)
        
        # Form delta (recent vs all-time)
        form_delta = {
            'win_rate_delta': round((recent_wins / len(recent)) - (all_wins / len(history)), 3),
            'kda_delta': round(recent_kda - all_kda, 2),
            'gpm_delta': round(recent_gpm - all_gpm, 1),
            'xpm_delta': round(recent_xpm - (sum(m['xpm'] for m in history) / len(history)), 1)
        }
        
        result[player] = {
            'total_matches': len(history),
            'win_rate': round(all_wins / len(history), 3),
            'recent_matches': len(recent),
            'recent_win_rate': round(recent_wins / len(recent), 3),
            'avg_kda': round(all_kda, 2),
            'recent_avg_kda': round(recent_kda, 2),
            'avg_gpm': round(all_gpm, 1),
            'recent_avg_gpm': round(recent_gpm, 1),
            'avg_xpm': round(sum(m['xpm'] for m in history) / len(history), 1),
            'recent_avg_xpm': round(recent_xpm, 1),
            'hero_pool': heroes_played,
            'hero_pool_size': len(heroes_played),
            'form_delta': form_delta
        }
    
    # Sort by matches
    result = dict(sorted(result.items(), key=lambda x: x[1]['total_matches'], reverse=True))
    
    return result


def main():
    print("Analyzing player form...")
    player_form = analyze_player_form()
    
    print(f"\nResults:")
    print(f"  Players with 5+ matches: {len(player_form)}")
    
    # Show top players
    print(f"\n  Top 15 players by matches:")
    for i, (player, stats) in enumerate(list(player_form.items())[:15], 1):
        form_trend = "+" if stats['form_delta']['win_rate_delta'] > 0 else "-" if stats['form_delta']['win_rate_delta'] < 0 else "="
        print(f"    {i}. {player}: {stats['total_matches']} matches, WR {stats['win_rate']:.1%}, Recent WR {stats['recent_win_rate']:.1%} {form_trend}")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(player_form, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
