import json
from collections import Counter
import os

lanes = Counter()
meta_lanes = Counter()

files = os.listdir('ml_data/full_matches')[:100]
for f in files:
    m = json.load(open(f'ml_data/full_matches/{f}'))
    for side in ['radiant', 'dire']:
        for p in m[side]['player_performances']:
            li = p.get('laneInfo', {})
            lanes[li.get('lane')] += 1
            meta_lanes[li.get('metaLane')] += 1

print('Lanes:', lanes.most_common(20))
print('MetaLanes:', meta_lanes.most_common(20))
