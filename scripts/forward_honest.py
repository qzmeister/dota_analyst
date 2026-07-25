"""Honest apples-to-apples forward comparison: v1 vs v3-style XGBoost
on the SAME out-of-sample split.

v1 (HistGBR) was trained on 883 matches.  v3 (XGBoost) was
trained on 1904 matches.  The A/B harness on 2389 in-sample-
weighted for both, which masks the real forward delta.

This script trains both on 883 matches and reports the
forward MAE on the OTHER 1497.  That's apples-to-apples.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import HeroWinRateEncoder, extract_features
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
GROUPS = ("hero",)

def load():
    raw, targets = [], []
    for p in sorted(DATA_DIR.glob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except: continue
        t = extract_target(d)
        if t is None: continue
        raw.append(d); targets.append(t)
    return raw, targets

def winsorize(y, n=3.0):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    sigma = 1.4826 * mad
    if sigma == 0: return y.copy()
    lo, hi = med - n*sigma, med + n*sigma
    out = y.copy(); out[out < lo] = lo; out[out > hi] = hi
    return out

def main():
    raw, targets = load()
    print(f"loaded {len(raw)} matches")
    encoder = HeroWinRateEncoder().fit(raw)
    X = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, encoder, groups=GROUPS) for t in targets],
        dtype=float,
    )
    y_k = np.asarray([t.kills_total for t in targets], dtype=float)
    y_d = np.asarray([t.duration_minutes for t in targets], dtype=float)

    # Train on 883 (v1 setup) — fixed 80/20 from a 1104-match corpus
    # is no longer available; we'll use the same random_state=42
    # split that v1 used and pull the first 883 by train_test_split order.
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=np.asarray([t.winner for t in targets], dtype=int),
    )
    # Subsample train to first 883 (v1's n_train) and test to the rest
    v1_train = idx_train[:883]
    v1_test = np.concatenate([idx_train[883:], idx_test])
    print(f"v1-style: train on {len(v1_train)}, test on {len(v1_test)} (forward)")

    print()
    print("=" * 80)
    print("KILLS — forward honest comparison on 1497 out-of-sample")
    print("=" * 80)
    hgb = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=20, random_state=42,
    )
    hgb.fit(X[v1_train], winsorize(y_k[v1_train]))
    pred = np.clip(hgb.predict(X[v1_test]), 0, None)
    print(f"  histgb_v1  forward: MAE={mean_absolute_error(y_k[v1_test], pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_k[v1_test], pred)):.4f}")

    xgb_p = xgb.XGBRegressor(
        objective="count:poisson", n_estimators=50, max_depth=3,
        learning_rate=0.1, random_state=42, tree_method="hist", verbosity=0,
    )
    xgb_p.fit(X[v1_train], winsorize(y_k[v1_train]))
    pred = np.clip(xgb_p.predict(X[v1_test]), 0, None)
    print(f"  xgb_p50d3  forward: MAE={mean_absolute_error(y_k[v1_test], pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_k[v1_test], pred)):.4f}")

    print()
    print("=" * 80)
    print("DURATION — forward honest comparison on 1497 out-of-sample")
    print("=" * 80)
    hgb = HistGradientBoostingRegressor(
        loss="gamma", learning_rate=0.05, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=20, random_state=42,
    )
    hgb.fit(X[v1_train], winsorize(y_d[v1_train]))
    pred = np.clip(hgb.predict(X[v1_test]), 0, None)
    print(f"  histgb_v1  forward: MAE={mean_absolute_error(y_d[v1_test], pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_d[v1_test], pred)):.4f}")

    xgb_l = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=80, max_depth=4,
        learning_rate=0.05, random_state=42, tree_method="hist", verbosity=0,
    )
    xgb_l.fit(X[v1_train], winsorize(y_d[v1_train]))
    pred = np.clip(xgb_l.predict(X[v1_test]), 0, None)
    print(f"  xgb_p80d4  forward: MAE={mean_absolute_error(y_d[v1_test], pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_d[v1_test], pred)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
