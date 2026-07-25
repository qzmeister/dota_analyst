"""Check if DatDota match details returns live data for in-progress matches."""

import sys
sys.path.insert(0, 'f:\\Projects\\Dota_analyst')

from backend.datdota_client import get_match_details
import json

# Get a known live match ID from Steam API or DLTV
# For testing, let's try a recent match that might be live

print("Testing DatDota match details for live data...")
print("="*60)

# Try fetching a match that we know exists
# Let's use a match from our collection that was recently played
test_match_id = 8910909413  # From the previous result

print(f"\nFetching match {test_match_id}...")
match = get_match_details(test_match_id)

if not match:
    print("[FAIL] No data")
    sys.exit(1)

print(f"[OK] Got match data")

# Check structure
data = match.get('data', {})
print(f"\nMatch keys: {list(data.keys())}")

# Check if there's live-specific data
if 'liveData' in data or 'live' in data:
    print("\n>>> FOUND LIVE DATA! <<<")
    print(json.dumps(data.get('liveData', data.get('live')), indent=2)[:2000])

# Check match status
match_info = data.get('match', {})
print(f"\nMatch info:")
print(f"  ID: {match_info.get('matchId')}")
print(f"  Duration: {match_info.get('duration')}s")
print(f"  Winner: {match_info.get('radiantVictory')}")
print(f"  Start: {match_info.get('startDate')}")

# Check for timeline/graph data (could be live)
if 'graphData' in data:
    graph = data['graphData']
    print(f"\nGraph data keys: {list(graph.keys())}")
    if 'gold' in graph or 'xp' in graph:
        print(f"  Gold diff points: {len(graph.get('gold', {}).get('data', []))}")
        print(f"  XP diff points: {len(graph.get('xp', {}).get('data', []))}")

# Check for player stats
if 'players' in data:
    players = data['players']
    print(f"\nPlayers: {len(players)} total")
    if players:
        print(f"  First player keys: {list(players[0].keys())}")

# Save full structure for inspection
with open('ml_data/sample_match_structure.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)

print(f"\n[OK] Saved full structure to ml_data/sample_match_structure.json")
print(f"     Inspect this file to see all available fields")
