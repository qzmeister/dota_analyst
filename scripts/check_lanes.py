"""Check how laneInfo maps to player positions in our corpus."""
import json
import glob
from collections import Counter, defaultdict

# lane -> side -> [hero_id, ...]
lanes_data = defaultdict(lambda: defaultdict(list))
hero_lane_count = Counter()

for f in glob.glob('ml_data/full_matches/*.json')[:500]:
    m = json.load(open(f, encoding='utf-8'))
    if m.get('has_error'):
        continue
    for side in ('radiant', 'dire'):
        for p in (m.get(side) or {}).get('player_performances') or []:
            li = p.get('laneInfo') or {}
            lane = li.get('lane')
            vid = ((p.get('performance') or {}).get('hero') or {}).get('valve_id')
            if not (lane and vid):
                continue
            lanes_data[lane][side].append(vid)
            hero_lane_count[(lane, vid)] += 1

print("Per-lane hero counts by side:")
for lane in ('BOTTOM', 'TOP', 'MIDDLE', 'JUNGLE', 'ROAM'):
    print(f"\n  {lane}:")
    for side in ('radiant', 'dire'):
        heroes = lanes_data[lane][side]
        cnt = len(heroes)
        unique = len(set(heroes))
        print(f"    {side}: {cnt} occurrences, {unique} unique heroes")

# Also check: does ROAM always co-occur with BOTTOM for the same side?
print("\nROAM co-occurrence with BOTTOM (same side):")
import random
random.seed(0)
sample_files = random.sample(glob.glob('ml_data/full_matches/*.json'), 100)
for f in sample_files[:5]:
    m = json.load(open(f, encoding='utf-8'))
    if m.get('has_error'):
        continue
    for side in ('radiant', 'dire'):
        roam_heroes = []
        bot_heroes = []
        for p in (m.get(side) or {}).get('player_performances') or []:
            li = p.get('laneInfo') or {}
            vid = ((p.get('performance') or {}).get('hero') or {}).get('valve_id')
            if li.get('lane') == 'ROAM' and vid:
                roam_heroes.append(vid)
            elif li.get('lane') == 'BOTTOM' and vid:
                bot_heroes.append(vid)
        if roam_heroes:
            print(f"  {f.split(chr(92))[-1][:30]}... {side}: ROAM={roam_heroes}  BOTTOM={bot_heroes}")
