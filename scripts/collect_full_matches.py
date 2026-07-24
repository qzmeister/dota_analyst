"""
Full match details collector via DatDota API.

Fetches complete match data (players, items, timeline) for all Tier 1 matches.
Supports resume - tracks progress and skips already fetched matches.

Rate limit: 500 requests/day, 3 seconds between requests.
Strategy: fetch ~400 matches per day, save progress, resume next day.

Usage:
    python scripts/collect_full_matches.py
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.datdota_client import get_match_details


# Paths
MATCHES_LIST = "ml_data/datdota_tier1_matches.json"
FULL_MATCHES_DIR = "ml_data/full_matches"
PROGRESS_FILE = "ml_data/collection_progress.json"
DAILY_LIMIT = 450  # Stay under 500 to be safe
REQUEST_DELAY = 3.0  # seconds between requests


def load_progress():
    """Load collection progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "fetched_ids": [],
        "failed_ids": [],
        "last_run": None,
        "total_fetched": 0,
        "days_run": 0
    }


def save_progress(progress):
    """Save collection progress to file."""
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_match_list():
    """Load the list of all Tier 1 matches."""
    if not os.path.exists(MATCHES_LIST):
        print(f"[ERROR] Match list not found: {MATCHES_LIST}")
        print("Run: python scripts/collect_datdota_targeted.py")
        sys.exit(1)
    
    with open(MATCHES_LIST, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    print(f"Loaded {len(matches)} matches from {MATCHES_LIST}")
    return matches


def sort_matches_by_date(matches):
    """Sort matches by start date (most recent first)."""
    def get_date(m):
        # DatDota uses 'startDate' field
        return m.get('startDate', '')
    
    # Sort descending (most recent first)
    return sorted(matches, key=get_date, reverse=True)


def fetch_match(match_id, output_dir):
    """Fetch full match details and save to file."""
    details = get_match_details(match_id)
    
    if not details or "data" not in details:
        return False
    
    data = details["data"]
    
    # Save to individual file
    output_file = os.path.join(output_dir, f"{match_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return True


def main():
    print("="*60)
    print("DatDota Full Match Details Collector")
    print("="*60)
    print()
    
    # Load progress
    progress = load_progress()
    fetched_ids = set(progress["fetched_ids"])
    failed_ids = set(progress["failed_ids"])
    
    print(f"Progress: {len(fetched_ids)} fetched, {len(failed_ids)} failed")
    if progress["last_run"]:
        print(f"Last run: {progress['last_run']}")
    print()
    
    # Load match list
    all_matches = load_match_list()
    
    # Sort by date (most recent first)
    all_matches = sort_matches_by_date(all_matches)
    
    # Filter out already fetched
    remaining = [m for m in all_matches if m.get('matchId') not in fetched_ids]
    
    print(f"\nRemaining to fetch: {len(remaining)} matches")
    print(f"Daily limit: {DAILY_LIMIT} requests")
    print()
    
    if not remaining:
        print("[DONE] All matches already fetched!")
        return
    
    # Create output directory
    os.makedirs(FULL_MATCHES_DIR, exist_ok=True)
    
    # Fetch matches
    fetched_today = 0
    failed_today = []
    
    for i, match in enumerate(remaining, 1):
        match_id = match.get('matchId')
        tournament = match.get('tournament_name', 'Unknown')
        start_date = match.get('startDate', 'Unknown')
        
        # Check daily limit
        if fetched_today >= DAILY_LIMIT:
            print(f"\n[LIMIT] Reached daily limit ({DAILY_LIMIT} requests)")
            print(f"Resume tomorrow to continue!")
            break
        
        print(f"[{i}/{len(remaining)}] Fetching match {match_id}...")
        print(f"   Tournament: {tournament}")
        print(f"   Date: {start_date}")
        
        # Fetch
        success = fetch_match(match_id, FULL_MATCHES_DIR)
        
        if success:
            fetched_ids.add(match_id)
            progress["fetched_ids"] = list(fetched_ids)
            progress["total_fetched"] = len(fetched_ids)
            fetched_today += 1
            print(f"   [OK] Saved to {FULL_MATCHES_DIR}/{match_id}.json")
        else:
            failed_ids.add(match_id)
            progress["failed_ids"] = list(failed_ids)
            failed_today.append(match_id)
            print(f"   [FAIL] Could not fetch match {match_id}")
        
        # Save progress after each match
        save_progress(progress)
        
        # Rate limit
        if i < len(remaining) and fetched_today < DAILY_LIMIT:
            time.sleep(REQUEST_DELAY)
    
    # Final summary
    progress["days_run"] += 1
    save_progress(progress)
    
    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY:")
    print(f"  Fetched today: {fetched_today}")
    print(f"  Failed today: {len(failed_today)}")
    print(f"  Total fetched: {len(fetched_ids)}/{len(all_matches)}")
    print(f"  Remaining: {len(all_matches) - len(fetched_ids)}")
    print(f"  Days run: {progress['days_run']}")
    print(f"{'='*60}")
    
    if len(fetched_ids) >= len(all_matches):
        print("\n[COMPLETE] All matches fetched!")
        print(f"Data saved to: {FULL_MATCHES_DIR}/")
    else:
        print(f"\n[PAUSED] Resume tomorrow:")
        print(f"  python scripts/collect_full_matches.py")


if __name__ == "__main__":
    main()
