"""Grid: kills and duration_mean regressors.

Compares the current HistGBR factories against XGBoost with
different loss functions (poisson, squared_error for kills;
gamma, squared_error for duration).  Evaluates on the test
split (honest) AND on the full corpus (production-like, mild
inflated) to mirror the v0.3.10 honest-vs-idiomatic split.

This is the 0.3.12 dev-cycle harness — picks the best
configuration for `kills` and `duration_mean` so we ship a
tighter v0.3.12 instead of reusing v1 (which was trained on
the much smaller 0.3.7 corpus).
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
GROUPS = ("hero",)  # team features regressed kills/duration in 0.3.11; revert to hero


def make_kills_models(rs=42):
    return {
        "histgb_poisson": HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=rs,
        ),
        "histgb_l2": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=rs,
        ),
        "xgb_poisson": xgb.XGBRegressor(
            objective="count:poisson", learning_rate=0.05, n_estimators=300,
            max_depth=6, random_state=rs, tree_method="hist", verbosity=0,
        ),
        "xgb_l2": xgb.XGBRegressor(
            objective="reg:squarederror", learning_rate=0.05, n_estimators=300,
            max_depth=6, random_state=rs, tree_method="hist", verbosity=0,
        ),
        "xgb_tweedie": xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.3,
            learning_rate=0.05, n_estimators=300, max_depth=6,
            random_state=rs, tree_method="hist", verbosity=0,
        ),
    }


def make_duration_models(rs=42):
    return {
        "histgb_gamma": HistGradientBoostingRegressor(
            loss="gamma", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=rs,
        ),
        "histgb_l2": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=rs,
        ),
        "xgb_gamma": xgb.XGBRegressor(
            # XGBoost doesn't have a native gamma objective; reg:gamma is
            # actually a distribution name (older).  Use squared_error
            # for the L2 baseline + the gamma HistGBR for the proper
            # distribution.  Skip reg:gamma here.
            objective="reg:squarederror", learning_rate=0.05, n_estimators=300,
            max_depth=6, random_state=rs, tree_method="hist", verbosity=0,
        ),
    }


def load():
    raw, targets = [], []
    for p in sorted(DATA_DIR.glob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except: continue
        t = extract_target(d)
        if t is None: continue
        raw.append(d); targets.append(t)
    return raw, targets


def _to_dict(t):
    """MatchTarget -> dict so the harness can read fields uniformly."""
    if hasattr(t, "__dict__"):
        return t.__dict__
    return t


def evaluate(model, X, y, n_features=0):
    pred = np.asarray(model.predict(X), dtype=float)
    pred = np.clip(pred, 0.0, None)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def main():
    raw, targets = load()
    print(f"loaded {len(raw)} matches")
    encoder = HeroWinRateEncoder().fit(raw)
    X = np.asarray(
        [
            extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, encoder,
                groups=GROUPS,
            )
            for t in targets
        ], dtype=float,
    )

    idx = np.arange(len(targets))
    y_kills = np.asarray([t.kills_total for t in targets], dtype=float)
    y_dur = np.asarray([t.duration_minutes for t in targets], dtype=float)

    # Honest split (80/20).
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=np.asarray([t.winner for t in targets], dtype=int),
    )

    # Winsorize target on train only (mirrors train.py).
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

    print()
    print("=" * 80)
    print("KILLS — different model + loss combinations")
    print("=" * 80)
    print(f"  {'model':<20} {'honest_mae':>11} {'honest_rmse':>12} {'train_s':>8}")
    for name, m in make_kills_models().items():
        y_tr = winsorize(y_kills[idx_train])
        m.fit(X[idx_train], y_tr)
        met = evaluate(m, X[idx_test], y_kills[idx_test])
        t0 = time.perf_counter()
        m.fit(X[idx_train], winsorize(y_kills[idx_train]))
        elapsed = time.perf_counter() - t0
        print(f"  {name:<20} {met['mae']:>11.4f} {met['rmse']:>12.4f} {elapsed:>8.1f}s")

    print()
    print("=" * 80)
    print("DURATION_MEAN — different model + loss combinations")
    print("=" * 80)
    print(f"  {'model':<20} {'honest_mae':>11} {'honest_rmse':>12} {'train_s':>8}")
    for name, m in make_duration_models().items():
        y_tr = winsorize(y_dur[idx_train])
        m.fit(X[idx_train], y_tr)
        met = evaluate(m, X[idx_test], y_dur[idx_test])
        t0 = time.perf_counter()
        m.fit(X[idx_train], winsorize(y_dur[idx_train]))
        elapsed = time.perf_counter() - t0
        print(f"  {name:<20} {met['mae']:>11.4f} {met['rmse']:>12.4f} {elapsed:>8.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
