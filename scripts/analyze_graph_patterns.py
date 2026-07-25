"""
Gold/XP Graph Shape Analysis
Classify match tempo from gold/xp graph shape.

Patterns:
- Snowball: One team dominates from start (monotonic increase)
- Comeback: One team behind, then overtakes (V-shape)
- Close: Teams stay even throughout (flat line)
- Late game: Even until 30 min, then one team spikes

Output: ml_data/graph_patterns.json
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List

MATCHES_DIR = Path("ml_data/full_matches")
OUTPUT_FILE = Path("ml_data/graph_patterns.json")


def classify_pattern(nw_series: List[float]) -> Dict:
    """Classify the networth advantage pattern."""
    if not nw_series or len(nw_series) < 5:
        return {'pattern': 'unknown', 'confidence': 0}
    
    # Normalize series
    arr = np.array(nw_series)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        return {'pattern': 'stalemate', 'confidence': 1.0}
    
    normalized = arr / max_abs
    
    # Calculate features
    start_val = normalized[0]
    end_val = normalized[-1]
    mid_idx = len(normalized) // 2
    mid_val = normalized[mid_idx]
    
    # Variance (how much the series fluctuates)
    variance = np.var(normalized)
    
    # Trend (linear regression slope)
    x = np.arange(len(normalized))
    slope = np.polyfit(x, normalized, 1)[0]
    
    # Check for V-shape (comeback): start and end have opposite signs, mid is extreme
    is_v_shape = (start_val * end_val < -0.3) and (abs(mid_val) > max(abs(start_val), abs(end_val)) * 0.7)
    
    # Check for snowball: monotonic increase/decrease
    diffs = np.diff(normalized)
    monotonic_ratio = np.sum(np.sign(diffs) == np.sign(diffs[0])) / len(diffs) if len(diffs) > 0 else 0
    is_snowball = monotonic_ratio > 0.7 and abs(end_val) > 0.5
    
    # Check for close game: low variance, stays near 0
    is_close = variance < 0.1 and abs(end_val) < 0.3
    
    # Check for late game spike: flat until 70%, then spike
    early_variance = np.var(normalized[:int(len(normalized)*0.7)])
    late_spike = abs(normalized[-1] - normalized[int(len(normalized)*0.7)]) > 0.5
    is_late_game = early_variance < 0.05 and late_spike
    
    # Determine pattern
    if is_close:
        return {'pattern': 'close', 'confidence': 1.0 - variance}
    elif is_snowball:
        winner = 'radiant' if end_val > 0 else 'dire'
        return {'pattern': 'snowball', 'winner': winner, 'confidence': monotonic_ratio}
    elif is_v_shape:
        comeback_team = 'radiant' if end_val > 0 else 'dire'
        return {'pattern': 'comeback', 'comeback_team': comeback_team, 'confidence': abs(start_val * end_val)}
    elif is_late_game:
        winner = 'radiant' if end_val > 0 else 'dire'
        return {'pattern': 'late_game', 'winner': winner, 'confidence': late_spike}
    else:
        # Default: classify by end state
        if abs(end_val) < 0.3:
            return {'pattern': 'close', 'confidence': 0.5}
        else:
            winner = 'radiant' if end_val > 0 else 'dire'
            return {'pattern': 'snowball', 'winner': winner, 'confidence': 0.5}


def analyze_matches():
    """Analyze all matches for graph patterns."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"Loaded {len(matches)} matches")
    
    patterns = []
    for i, match in enumerate(matches, 1):
        if i % 200 == 0:
            print(f"  Analyzed {i}/{len(matches)} matches")
        
        frames = match.get('frames')
        if not frames or not frames.get('radiant_networth_advantage'):
            continue
        
        nw_series = frames['radiant_networth_advantage']
        if not nw_series or len(nw_series) < 5:
            continue
        
        classification = classify_pattern(nw_series)
        classification['match_id'] = int(match['match_id'])
        classification['duration_min'] = round(match['duration'] / 60.0, 2)
        classification['radiant_victory'] = bool(match['radiant_victory'])
        classification['final_nw_advantage'] = int(nw_series[-1])
        # Convert numpy types to Python types
        if 'confidence' in classification:
            classification['confidence'] = float(classification['confidence'])
        
        patterns.append(classification)
    
    return patterns


def main():
    print("Analyzing match graph patterns...")
    patterns = analyze_matches()
    
    print(f"\nClassified {len(patterns)} matches")
    
    # Count patterns
    pattern_counts = Counter(p['pattern'] for p in patterns)
    print(f"\nPattern distribution:")
    for pattern, count in pattern_counts.most_common():
        pct = count / len(patterns) * 100
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    # Analyze by winner
    print(f"\nPattern by match winner:")
    for pattern in ['snowball', 'comeback', 'late_game', 'close']:
        pattern_matches = [p for p in patterns if p['pattern'] == pattern]
        if pattern_matches:
            radiant_wins = sum(1 for p in pattern_matches if p['radiant_victory'])
            print(f"  {pattern}: {len(pattern_matches)} matches, Radiant wins: {radiant_wins} ({radiant_wins/len(pattern_matches)*100:.1f}%)")
    
    # Save to JSON
    result = {
        'total_matches': len(patterns),
        'pattern_counts': dict(pattern_counts),
        'matches': patterns
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
