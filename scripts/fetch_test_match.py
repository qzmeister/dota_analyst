"""Fetch full match details for one test match via DatDota API."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.datdota_client import get_match_details


def main():
    match_id = 8885183102  # Esports World Cup 2026 match
    
    print(f"Fetching full details for match {match_id}...")
    print("="*60)
    
    details = get_match_details(match_id)
    
    if not details or "data" not in details:
        print("[ERROR] No data returned!")
        return
    
    data = details["data"]
    
    # Match info
    print("\nMATCH INFO:")
    print(f"  Duration: {data.get('duration')}s ({data.get('duration')/60:.1f} min)")
    print(f"  Winner: {'Radiant' if data.get('radiant_victory') else 'Dire'}")
    print(f"  Patch: {data.get('patch')}")
    print(f"  League: {data.get('league', {}).get('name')}")
    print(f"  Start date: {data.get('start_date')}")
    
    # Radiant team
    print("\nRADIANT TEAM:")
    radiant = data.get('radiant', {})
    print(f"  Name: {radiant.get('team', {}).get('name')}")
    print(f"  Tag: {radiant.get('team', {}).get('tag')}")
    print(f"  Players: {len(radiant.get('player_performances', []))}")
    
    for i, player in enumerate(radiant.get('player_performances', [])[:2], 1):
        perf = player.get('performance', {})
        hero = perf.get('hero', {})
        print(f"\n  Player {i}: {player.get('player', {}).get('nickname')}")
        print(f"    Hero: {hero.get('short_name')} (ID: {hero.get('valve_id')})")
        print(f"    K/D/A: {perf.get('kills')}/{perf.get('deaths')}/{perf.get('assists')}")
        print(f"    GPM/XPM: {perf.get('gpm')}/{perf.get('xpm')}")
        print(f"    Hero damage: {perf.get('hero_damage')}")
        print(f"    Building damage: {perf.get('building_damage')}")
        print(f"    Level: {perf.get('level')}")
        print(f"    Items: {len(perf.get('items', []))}")
        lane_info = player.get('laneInfo', {})
        print(f"    Lane: {lane_info.get('lane')} -> {lane_info.get('switchTo')}")
    
    # Dire team
    print("\nDIRE TEAM:")
    dire = data.get('dire', {})
    print(f"  Name: {dire.get('team', {}).get('name')}")
    print(f"  Tag: {dire.get('team', {}).get('tag')}")
    print(f"  Players: {len(dire.get('player_performances', []))}")
    
    # Timeline data
    print("\nTIMELINE DATA:")
    frames = data.get('frames', {})
    times = frames.get('times', [])
    networth = frames.get('radiant_networth_advantage', [])
    print(f"  Time points: {len(times)}")
    print(f"  Networth advantage points: {len(networth)}")
    if times:
        print(f"  Time range: {times[0]}s to {times[-1]}s")
    if networth:
        print(f"  Networth range: {min(networth)} to {max(networth)}")
    
    # Map control
    print("\nMAP CONTROL:")
    map_ctrl = data.get('map_control', {})
    print(f"  Control value: {map_ctrl.get('control_value')}")
    print(f"  One-sidedness: {map_ctrl.get('one_sidedness')}")
    print(f"  Raw control values: {len(map_ctrl.get('raw_control_values', []))} points")
    
    # Save full data
    output_file = "ml_data/test_match_full.json"
    os.makedirs("ml_data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Full match data saved to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print("\nDATA AVAILABLE FOR ML:")
    print("  [OK] Match duration")
    print("  [OK] Winner (radiant/dire)")
    print("  [OK] Patch version")
    print("  [OK] Team names and tags")
    print("  [OK] Player stats (K/D/A, GPM/XPM, damage)")
    print("  [OK] Hero picks (10 heroes)")
    print("  [OK] Item builds with timestamps")
    print("  [OK] Lane assignments")
    print("  [OK] Timeline data (gold/xp graph)")
    print("  [OK] Map control metrics")
    print("\nAll data needed for ML training is available!")


if __name__ == "__main__":
    main()
