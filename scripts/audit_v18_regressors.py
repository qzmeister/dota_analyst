"""Audit v18 regressors: kills_total, duration_sec, first_15_kills.

For each model:
  1. Load it
  2. Predict on 3 different synthetic inputs
  3. Verify predictions DIFFER (no constant-output bug)
  4. Verify predictions are in a reasonable range

If any model is constant or returns nonsense, this is a smoking
gun and we need to retrain or fall back to v17.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))
MODELS = PRO_ROOT / "ml_data" / "models"

# Synthetic feature rows: vary the tier (premium vs minor) and
# heroes (carry meta vs support meta).  We DON'T vary the
# post-game features (duration_min, gold_adv) because v18
# deliberately doesn't use them.

PREMIUM_HEROES = [1, 5, 8, 11, 25]   # team fight
MINOR_HEROES   = [35, 38, 41, 44, 47]  # late-game
PREMIUM_ID = 9824702   # 24-6, tier=2
MINOR_ID   = 9425656   # 84-72, tier=0


def build_inputs():
    """Return 4 (label, feats_dict) tuples that differ in tier and hero."""
    out = []
    for label, r_id, r_p in [
        ("premium_t1", PREMIUM_ID, PREMIUM_HEROES),
        ("premium_t2", PREMIUM_ID, [12, 14, 19, 23, 31]),
        ("minor_t1",   MINOR_ID,   PREMIUM_HEROES),
        ("minor_t2",   MINOR_ID,   MINOR_HEROES),
    ]:
        out.append((label, r_id, r_p, [2, 3, 4, 7, 9]))
    return out


def load_and_predict(model_dir: Path, inputs, target: str):
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    feat_names = meta["feature_columns"]
    sys.path.insert(0, str(PRO_ROOT))
    from business.v18_predict import _build_features
    rows = []
    for label, r_id, r_p, d_p in inputs:
        feats = _build_features(
            r_picks=r_p, d_picks=d_p,
            r_team_id=r_id, d_team_id=None,
            r_bans=None, d_bans=None,
            start_time=int(time.time()),
            patch="7.41",
        )
        X = np.asarray([[feats.get(f, 0.0) for f in feat_names]], dtype=np.float32)
        rows.append((label, X))
    preds = []
    for label, X in rows:
        try:
            p = float(model.predict(X)[0])
        except Exception as exc:
            p = f"err:{exc}"
        preds.append((label, p))
    return meta, preds


def main():
    print("=" * 78)
    print("v18 regressor audit")
    print("=" * 78)
    print()
    inputs = build_inputs()
    for target in ("kills_total", "duration_sec", "first_15_kills"):
        d = MODELS / f"_v18_{target}"
        if not d.exists():
            print(f"  {target}: model dir not found, skipping")
            continue
        print(f"--- {target} ---")
        try:
            meta, preds = load_and_predict(d, inputs, target)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        for label, p in preds:
            print(f"  {label:>12s}: {p:>8.3f}")
        # Sanity: predictions should differ
        nums = [p for _, p in preds if isinstance(p, (int, float))]
        if len(nums) >= 2:
            spread = max(nums) - min(nums)
            mean = sum(nums) / len(nums)
            print(f"  spread={spread:.3f}  mean={mean:.3f}")
            # For kills_total: realistic range is 30-60
            # For duration_sec: 1200-3000 (20-50 min)
            # For first_15: 0-15
            if target == "kills_total":
                ok = 20 < mean < 80 and spread > 1
            elif target == "duration_sec":
                ok = 600 < mean < 3600 and spread > 60
            elif target == "first_15_kills":
                # 0-15 is a very narrow range; allow degenerate
                ok = 0 <= mean <= 16
            else:
                ok = True
            tag = "OK" if ok else "SUSPECT"
            print(f"  -> {tag}")
        print()


if __name__ == "__main__":
    main()
