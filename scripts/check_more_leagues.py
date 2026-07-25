"""Quick check: what other tier-1 leagues does DatDota expose?"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.datdota_client import _http_json

# List a few leagues
r = _http_json('https://api.datdota.com/api/leagues?limit=5')
if isinstance(r, dict) and 'data' in r:
    data = r['data']
    if isinstance(data, list):
        print(f'leagues: {len(data)} (truncated to 5)')
        for L in data[:5]:
            print(f'  {L}')
    else:
        print('data is', type(data))
        print('first 2:', data[:2] if isinstance(data, list) else 'n/a')
else:
    print('shape:', type(r), list(r.keys()) if isinstance(r, dict) else 'n/a')
