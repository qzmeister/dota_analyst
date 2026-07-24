"""
Collect Tier 1 tournament data via DatDota API.

Usage:
    python scripts/collect_datdota_tier1.py
    
This will:
1. Fetch all Tier 1 tournaments (prize pool >= $1M)
2. Download all matches from each tournament
3. Save to ml_data/datdota_tier1_matches.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.datdota_client import collect_all_tier1_matches


def main():
    print("="*60)
    print("DatDota Tier 1 Tournament Collector")
    print("="*60)
    print()
    
    # Collect
    matches = collect_all_tier1_matches()
    
    if not matches:
        print("[ERROR] No matches collected!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Collected {len(matches)} Tier 1 matches")
    print(f"{'='*60}\n")
    
    # Next steps
    print("Next steps:")
    print("1. Enrich with Steam API MatchDetails (optional)")
    print("2. Extract ML features")
    print("3. Train models")
    print()


if __name__ == "__main__":
    main()
