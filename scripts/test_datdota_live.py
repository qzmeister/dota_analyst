"""Test DatDota API endpoints for live matches."""

import sys
sys.path.insert(0, 'f:\\Projects\\Dota_analyst')

from backend.datdota_client import _http_json
import time

endpoints = [
    '/api/live',
    '/api/matches/current',
    '/api/live/matches',
    '/api/leagues/live',
    '/api/matches?live=true',
    '/api/streams',  # DatDota has streaming/live data
]

for endpoint in endpoints:
    url = f"https://api.datdota.com{endpoint}"
    print(f"\nTesting: {endpoint}")
    result = _http_json(url)
    if result:
        print(f"  [OK] Got data!")
        print(f"  Keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        if isinstance(result, dict) and 'data' in result:
            data = result['data']
            if isinstance(data, list):
                print(f"  Items: {len(data)}")
                if data:
                    print(f"  First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else data[0]}")
            elif isinstance(data, dict):
                print(f"  Data keys: {list(data.keys())}")
    else:
        print(f"  [FAIL] No data")
    
    time.sleep(3)
