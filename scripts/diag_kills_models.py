"""Compare kills_v1 vs kills_v11 metrics."""
import json
for v in ['1', '11']:
    m = json.load(open(f'ml_data/models/kills_v{v}/metadata.json'))
    print(f'kills_v{v}:')
    print(f'  metrics={m["metrics"]}')
    print(f'  feature_names: {m.get("feature_names", [])[:3]}... ({len(m.get("feature_names",[]))} total)')
    print(f'  train_data keys: {list(m.get("train_data",{}).keys())}')
    print(f'  n_matches: {m["train_data"].get("n_matches")}')
    print()
