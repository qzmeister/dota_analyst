import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.datdota_client import get_leagues

data = get_leagues(limit=500)
leagues = data.get('data', [])

print(f"Total leagues: {len(leagues)}")
print("\nSearching for missing tournaments...")

targets = ['PGL Wallachia', 'Wallachia']

for target in targets:
    print(f"\n[SEARCH] {target}")
    found_any = False
    for league in leagues:
        name = league.get('name') or ''
        if target.lower() in name.lower():
            print(f"   FOUND: {name} (ID: {league.get('leagueId')}, Tier: {league.get('tier', {}).get('name')})")
            found_any = True
    if not found_any:
        print(f"   NOT FOUND")
