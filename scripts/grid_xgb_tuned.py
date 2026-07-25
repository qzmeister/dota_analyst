"""Tighter XGBoost grid: match the v0.3.9 winner (n_est=50, lr=0.1, md=3)
and see if a similar conservative config beats HistGBR on kills/duration.

The earlier grid used n_estimators=300 / max_depth=6 / lr=0.05 which
overfit the 1904-match training set.  This grid tests 3 conservative
configs against the HistGBR v1 baseline.
"""
from __future__ import annotations

import json, sys, time
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

XGB_CONFIGS = [
    ("xgb_p50_d3_lr01",  dict(n_estimators=50,  max_depth=3, learning_rate=0.1)),
    ("xgb_p80_d4_lr05",  dict(n_estimators=80,  max_depth=4, learning_rate=0.05)),
    ("xgb_p100_d4_lr1",  dict(n_estimators=100, max_depth=4, learning_rate=0.1)),
    ("xgb_p150_d3_lr05", dict(n_estimators=150, max_depth=3, learning_rate=0.05)),
    ("xgb_p200_d3_lr05", dict(n_estimators=200, max_depth=3, learning_rate=0.05)),
]


def make_kills(name, kw):
    if "poisson" in name or "xgb" in name:
        return xgb.XGBRegressor(objective="count:poisson", **kw,
                                random_state=42, tree_method="hist", verbosity=0)
    return HistGradientBoostingRegressor(loss="poisson", **kw, random_state=42)


def make_duration(name, kw):
    return xgb.XGBRegressor(objective="reg:squarederror", **kw,
                            random_state=42, tree_method="hist", verbosity=0)


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
    out = y.copy()
    out[out < lo] = lo
    out[out > hi] = hi
    return out


def evaluate(m, X, y):
    pred = np.clip(np.asarray(m.predict(X), dtype=float), 0.0, None)
    return float(mean_absolute_error(y, pred)), float(np.sqrt(mean_squared_error(y, pred)))


def main():
    raw, targets = load()
    print(f"loaded {len(raw)} matches")
    encoder = HeroWinRateEncoder().fit(raw)
    X = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, encoder, groups=GROUPS) for t in targets],
        dtype=float,
    )
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=np.asarray([t.winner for t in targets], dtype=int),
    )
    y_k = np.asarray([t.kills_total for t in targets], dtype=float)
    y_d = np.asarray([t.duration_minutes for t in targets], dtype=float)

    print()
    print("KILLS — tuned XGBoost configs")
    print(f"  {'model':<25} {'honest_mae':>11} {'honest_rmse':>12} {'train_s':>8}")
    for name, kw in XGB_CONFIGS:
        m = make_kills(name, kw)
        m.fit(X[idx_train], winsorize(y_k[idx_train]))
        mae, rmse = evaluate(m, X[idx_test], y_k[idx_test])
        t0 = time.perf_counter(); m.fit(X[idx_train], winsorize(y_k[idx_train])); train_s = time.perf_counter() - t0
        print(f"  {name:<25} {mae:>11.4f} {rmse:>12.4f} {train_s:>8.2f}s")

    print()
    print("DURATION_MEAN — tuned XGBoost configs")
    print(f"  {'model':<25} {'honest_mae':>11} {'honest_rmse':>12} {'train_s':>8}")
    for name, kw in XGB_CONFIGS:
        m = make_duration(name, kw)
        m.fit(X[idx_train], winsorize(y_d[idx_train]))
        mae, rmse = evaluate(m, X[idx_test], y_d[idx_test])
        t0 = time.perf_counter(); m.fit(X[idx_train], winsorize(y_d[idx_train])); train_s = time.perf_counter() - t0
        print(f"  {name:<25} {mae:>11.4f} {rmse:>12.4f} {train_s:>8.2f}s")

    # HistGBR baseline for reference
    print()
    print("HistGBR baseline (v1 factory)")
    hgb_k = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=20, random_state=42,
    )
    hgb_k.fit(X[idx_train], winsorize(y_k[idx_train]))
    mae, rmse = evaluate(hgb_k, X[idx_test], y_k[idx_test])
    print(f"  {'histgb_poisson (v1)':<25} {mae:>11.4f} {rmse:>12.4f}")
    hgb_d = HistGradientBoostingRegressor(
        loss="gamma", learning_rate=0.05, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=20, random_state=42,
    )
    hgb_d.fit(X[idx_train], winsorize(y_d[idx_train]))
    mae, rmse = evaluate(hgb_d, X[idx_test], y_d[idx_test])
    print(f"  {'histgb_gamma (v1)':<25} {mae:>11.4f} {rmse:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
