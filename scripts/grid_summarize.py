"""Print a top-N summary of the grid results from grid_night_results.jsonl.

Run this while the grid is still going to see live best-so-far.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

f = Path("scripts/grid_night_results.jsonl")
if not f.exists():
    print("no results yet")
    sys.exit(0)

results = []
for line in f.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    try:
        results.append(json.loads(line))
    except Exception:
        pass

print(f"total records: {len(results)}", flush=True)

# Per target: best by mae
for target in ("kills", "duration_mean", "duration_p10", "duration_p90", "winner"):
    sub = [r for r in results if r["target"] == target]
    if not sub:
        print(f"\n--- {target}: NO DATA YET ---")
        continue
    if target == "winner":
        sub.sort(key=lambda r: r["log_loss_mean"])
    else:
        sub.sort(key=lambda r: r["mae_mean"])
    print(f"\n--- {target}: top 10 (n={len([r for r in results if r['target']==target])}) ---")
    if target == "winner":
        print(f"{'rank':>4} {'acc':>8} {'acc_std':>8} {'logloss':>8} {'auc':>8} {'model':<35} {'groups'}")
        for i, r in enumerate(sub[:10]):
            print(f"{i+1:>4} {r.get('acc_mean', 0)*100:>7.3f}% {r.get('acc_std', 0)*100:>7.3f}% "
                  f"{r.get('log_loss_mean', 0):>8.4f} {r.get('auc_mean', 0):>8.4f} "
                  f"{r['name']:<35} {r['groups']}")
    else:
        print(f"{'rank':>4} {'mae':>9} {'std':>6} {'metric':>9} {'model':<35} {'groups'}")
        for i, r in enumerate(sub[:10]):
            m = r.get("metric_mean", 0)
            print(f"{i+1:>4} {r['mae_mean']:>9.4f} {r.get('mae_std', 0):>6.3f} {m:>9.4f} "
                  f"{r['name']:<35} {r['groups']}")
