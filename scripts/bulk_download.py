"""
Bulk Data Collection Script for Dota Analyst ML Training.

Downloads historical match data from Stratz API since March 25, 2026.

Rate limits:
- Free tier: 30 requests/minute
- Estimated time: 90 days × ~500 matches/day = 45,000 matches
- At 0.4s/request: ~6 hours total (serial)
- With batching: ~2-3 hours recommended

Usage:
    python -m scripts.bulk_download --from 2026-03-25 --teams "123,456"
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from backend.stratz_api import _http_json, BASE_URL, REQUEST_DELAY_BASE
except ImportError:
    print("⚠️  Please ensure python-dotenv is installed and .env exists")
    sys.exit(1)


def get_team_ids_from_stratz() -> List[int]:
    """
    Fetch list of pro teams from Stratz.
    
    Returns a curated list of top-tier teams based on common knowledge.
    For best results, add more team IDs manually.
    
    Known Pro Teams (Stratz IDs):
    - Team Liquid: 2163
    - OG: 25869725
    - Evil Geniuses: 39
    - Virtus.pro: 15
    - Team Spirit: 82148507
    - gGaming: 8254400
    - and many more...
    """
    # Add known team IDs here
    known_teams = [
        82148507,   # Team Spirit
        25869725,   # OG
        39,         # Evil Geniuses
        2163,       # Team Liquid
        15,         # Virtus.pro
        8254400,    # gGaming
        58514931,   # Example team ID (user's provided)
    ]
    
    return known_teams


def fetch_team_matches(team_id: int, max_matches: int = 200, cache_file: Optional[str] = None) -> List[dict]:
    """
    Fetch matches for a single team with rate limiting and caching.
    
    Args:
        team_id: Stratz team ID
        max_matches: Maximum matches to fetch
        cache_file: Path to cache file (optional)
    
    Returns:
        List of match dicts
    """
    print(f"[{team_id}] Fetching {max_matches} matches...")
    
    # Check cache first
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if isinstance(cached, list):
                    print(f"[{team_id}] Loaded {len(cached)} matches from cache")
                    return cached
        except Exception as e:
            print(f"[{team_id}] Cache load failed: {e}")
    
    url = f"{BASE_URL}/api/team/{team_id}/matches"
    params = {"limit": min(max_matches, 200), "order": "desc"}
    
    headers = {}
    api_key = os.environ.get("STRAZT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "DotaAnalyst/BulkDownload/1.0")
        for k, v in headers.items():
            req.add_header(k, v)
        
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            if resp.status != 200:
                print(f"[{team_id}] Failed: status {resp.status}")
                return []
            
            data = json.loads(resp.read().decode("utf-8"))
            matches = data.get("matches", [])[:max_matches]
            
            # Save to cache
            if cache_file:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(matches, f)
                    print(f"[{team_id}] Cached {len(matches)} matches")
                except Exception as e:
                    print(f"[{team_id}] Cache save failed: {e}")
            
            return matches
            
    except Exception as e:
        print(f"[{team_id}] HTTP error: {e}")
        # Return cached data even if fresh fetch failed
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []


def filter_matches_by_date(matches: List[dict], start_date: datetime) -> List[dict]:
    """Filter matches that occurred after start_date."""
    filtered = []
    
    for match in matches:
        # Try different timestamp fields
        ts = match.get("start_time") or match.get("timestamp") or match.get("date")
        
        if not ts:
            continue
        
        try:
            # Parse ISO format
            if isinstance(ts, str):
                ts = ts.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts)
            else:
                dt = datetime.utcfromtimestamp(ts)
            
            # Normalize to UTC
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            
            if dt >= start_date:
                filtered.append(match)
        except Exception as e:
            print(f"[filter] Invalid date: {ts} - {e}")
            continue
    
    return filtered


def collect_all_history(start_date: datetime, team_ids: List[int], max_per_team: int = 200, workers: int = 5):
    """
    Collect match history for multiple teams in parallel.
    
    Args:
        start_date: Start collecting from this date
        team_ids: List of team IDs
        max_per_team: Max matches per team
        workers: Number of parallel threads (keep low due to rate limits!)
    """
    all_samples = []
    team_cache_dir = "ml_data/teams"
    os.makedirs(team_cache_dir, exist_ok=True)
    
    total_time_needed = len(team_ids) * max_per_team * REQUEST_DELAY_BASE / 60.0
    print(f"\n⏱️  Estimated time: {total_time_needed:.1f} minutes")
    print(f"📊 Teams: {len(team_ids)}, Matches/team: {max_per_team}, Workers: {workers}\n")
    
    # Process teams in batches
    for batch_start in range(0, len(team_ids), workers):
        batch_end = min(batch_start + workers, len(team_ids))
        batch = team_ids[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_start//workers + 1}: Teams {batch[0]}-{batch[-1]}")
        print('='*60)
        
        batch_tasks = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for team_id in batch:
                cache_file = os.path.join(team_cache_dir, f"team_{team_id}.json")
                future = executor.submit(fetch_team_matches, team_id, max_per_team, cache_file)
                batch_tasks.append((team_id, future))
                
                # Respect rate limit between requests from same thread
                time.sleep(REQUEST_DELAY_BASE)
            
            # Collect results
            for team_id, future in batch_tasks:
                try:
                    matches = future.result(timeout=30.0)
                    
                    # Filter by date
                    filtered = filter_matches_by_date(matches, start_date)
                    
                    print(f"[{team_id}] Fresh: {len(filtered)}/{len(matches)} matches since {start_date.date()}")
                    
                    all_samples.extend(filtered)
                    
                except Exception as e:
                    print(f"[{team_id}] Error: {e}")
        
        # Delay between batches
        time.sleep(2.0)
    
    # Deduplicate matches by ID
    seen_match_ids = set()
    unique_samples = []
    
    for sample in all_samples:
        mid = sample.get("match_id") or sample.get("id")
        if mid and mid not in seen_match_ids:
            seen_match_ids.add(mid)
            unique_samples.append(sample)
    
    # Save combined dataset
    output_file = "ml_data/all_matches.json"
    with open(output_file, 'w') as f:
        json.dump(unique_samples, f, indent=2)
    
    print(f"\n✅ Done! Collected {len(unique_samples)} unique matches")
    print(f"💾 Saved to {output_file}")
    
    return unique_samples


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical Dota 2 matches from Stratz API")
    parser.add_argument("--from", dest="start_date", default="2026-03-25", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--teams", help="Team IDs (comma-separated), defaults to known teams")
    parser.add_argument("--per-team", type=int, default=200, help="Max matches per team")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (low due to rate limits)")
    
    args = parser.parse_args()
    
    # Parse start date
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
    except ValueError as e:
        print(f"Error parsing date: {e}")
        sys.exit(1)
    
    # Get team IDs
    if args.teams:
        team_ids = [int(tid.strip()) for tid in args.teams.split(",") if tid.strip()]
    else:
        team_ids = get_team_ids_from_stratz()
        print(f"⚠️  Using default team list: {team_ids}")
    
    if not team_ids:
        print("❌ No team IDs provided!")
        sys.exit(1)
    
    # Collect
    print(f"\n🚀 Starting collection:")
    print(f"   From: {start_date.date()}")
    print(f"   Teams: {len(team_ids)}")
    print(f"   Per team: {args.per_team}")
    print(f"   Workers: {args.workers}\n")
    
    start_time = time.time()
    samples = collect_all_history(start_date, team_ids, args.per_team, args.workers)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
