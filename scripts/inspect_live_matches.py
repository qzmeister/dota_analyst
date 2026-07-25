"""Inspect DatDota live matches data structure."""

import sys
sys.path.insert(0, 'f:\\Projects\\Dota_analyst')

from backend.datdota_client import _http_json
import json

print("Fetching /api/matches?live=true...")
result = _http_json("https://api.datdota.com/api/matches?live=true")

if not result or 'data' not in result:
    print("[FAIL] No data")
    sys.exit(1)

matches = result['data']
print(f"\n[OK] Got {len(matches)} matches")

# Check first few matches
print("\n" + "="*60)
print("FIRST 5 MATCHES:")
print("="*60)

for i, match in enumerate(matches[:5], 1):
    print(f"\n{i}. Match ID: {match.get('matchId')}")
    print(f"   League: {match.get('league', {}).get('name')}")
    print(f"   Teams: {match.get('radiant', {}).get('name')} vs {match.get('dire', {}).get('name')}")
    print(f"   Start: {match.get('startDate')}")
    print(f"   Duration: {match.get('duration')}s")
    print(f"   Winner: {match.get('radiantVictory')}")
    
    # Check if match is live (no winner yet, duration > 0)
    if match.get('radiantVictory') is None and match.get('duration', 0) > 0:
        print(f"   >>> LIVE MATCH! <<<")

# Count live vs completed
live_count = 0
completed_count = 0
upcoming_count = 0

for match in matches:
    duration = match.get('duration', 0)
    winner = match.get('radiantVictory')
    
    if winner is None and duration > 0:
        live_count += 1
    elif winner is not None:
        completed_count += 1
    else:
        upcoming_count += 1

print("\n" + "="*60)
print("MATCH STATUS BREAKDOWN:")
print("="*60)
print(f"  Live (in progress): {live_count}")
print(f"  Completed: {completed_count}")
print(f"  Upcoming: {upcoming_count}")
print(f"  Total: {len(matches)}")

# Show sample of live matches
if live_count > 0:
    print("\n" + "="*60)
    print("SAMPLE LIVE MATCHES:")
    print("="*60)
    
    live_matches = [m for m in matches if m.get('radiantVictory') is None and m.get('duration', 0) > 0]
    
    for i, match in enumerate(live_matches[:3], 1):
        print(f"\n{i}. {match.get('radiant', {}).get('name')} vs {match.get('dire', {}).get('name')}")
        print(f"   Duration: {match.get('duration')}s ({match.get('duration')//60}m)")
        print(f"   League: {match.get('league', {}).get('name')}")
        
        # Show full structure
        print(f"\n   Full match structure:")
        print(json.dumps(match, indent=4, default=str)[:1000])
