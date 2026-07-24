"""
Quick Tier 1 collection using discovery.py logic.

Simple approach: use existing _split_match_blocks parser from discovery module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urllib.request import urlopen
from backend.discovery import _split_match_blocks, _TEAM_NAME
import re


def get_tier1_matches_from_html(html: str) -> list:
    """Parse DLTV matches page and extract Tier 1 tournaments."""
    
    TIER1_PATTERNS = [
        "esports world cup",
        "blast slam",
        "dreamleague", 
        "pgl wallachia",
        "esl one birmingham",
    ]
    
    tier1_matches = []
    
    # Split into match blocks
    blocks = _split_match_blocks(html)
    
    print(f"[PARSER] Found {len(blocks)} match blocks\n")
    
    for cls, attrs, body in blocks:
        # Check if this is live or upcoming match
        if "live" not in cls and "upcoming" not in cls:
            continue
        
        series_id = attrs.get("series_id")
        steam_id = attrs.get("match_id")
        
        if not series_id:
            continue
        
        # Extract event name
        event_m = re.search(r'<div class="match__head-event">.*?<span>([^<]{2,})</span>', body, re.DOTALL)
        event_name = event_m.group(1).strip() if event_m else "Unknown"
        
        # Check if Tier 1
        if any(pattern.lower() in event_name.lower() for pattern in TIER1_PATTERNS):
            # Extract teams
            team_names = _TEAM_NAME.findall(body)
            team_a = team_names[0].strip() if len(team_names) > 0 else "TBD"
            team_b = team_names[1].strip() if len(team_names) > 1 else "TBD"
            
            tier1_matches.append({
                "steam_id": int(steam_id) if steam_id else None,
                "event_id": int(series_id),
                "event": event_name,
                "team_a": team_a,
                "team_b": team_b,
                "tournament_type": "tier1",
            })
    
    return tier1_matches


def main():
    print("="*60)
    print("Tier 1 Collection via Discovery Parser")
    print("="*60)
    
    # Fetch HTML
    url = "https://dltv.org/matches"
    print(f"\n[FETCH] Getting {url}...")
    
    try:
        with urlopen(url, timeout=15.0) as resp:
            html = resp.read().decode("utf-8")
        
        print(f"[FETCH] Got {len(html)} bytes\n")
        
        # Parse
        tier1 = get_tier1_matches_from_html(html)
        
        print("\n" + "="*60)
        print(f"Found {len(tier1)} Tier 1 matches:")
        
        # DEBUG: Show all events found
        from collections import defaultdict
        by_event = defaultdict(int)
        for cls, attrs, body in blocks:
            if 'live' not in cls and 'upcoming' not in cls:
                continue
            event_m = re.search(r'<div class="match__head-event">.*?<span>([^<]{2,})</span>', body, re.DOTALL)
            if event_m:
                name = event_m.group(1).strip()
                by_event[name] += 1
        
        print(f"\nAll events found ({len(by_event)} unique):")
        for evt, count in sorted(by_event.items()):
            is_t1 = any(p.lower() in evt.lower() for p in TIER1_PATTERNS)
            marker = " [TIER1]" if is_t1 else ""
            print(f"   • {evt} x{count}{marker}")
        
        # Group by tournament
        from collections import defaultdict
        by_tournament = defaultdict(list)
        
        for m in tier1:
            by_tournament[m["event"]].append(m)
        
        for tournament, matches in sorted(by_tournament.items()):
            has_steam = sum(1 for m in matches if m.get("steam_id"))
            print(f"\n📊 {tournament}")
            print(f"   Total: {len(matches)} | With Steam ID: {has_steam}")
        
        # Save
        output_file = "ml_data/tier1_discovery.json"
        import json
        os.makedirs("ml_data", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(tier1, f, indent=2)
        
        print("\n" + "="*60)
        print(f"Saved to: {output_file}")
        print(f"Total unique matches: {len(tier1)}\n")
        
        return tier1
        
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    matches = main()
