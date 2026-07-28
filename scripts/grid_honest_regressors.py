"""Honest 5-fold CV for KILLS and DURATION_MEAN, top configs from leaky grid.

Refits the encoder on each train fold so the test rows' target encodings
are not contaminated with the test rows' own outcomes.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    FEATURE_GROUPS, HeroWinRateEncoder, PlayerWinRateEncoder, extract_features,
)
from business.ml.targets import extract_target

DATA = ROOT / "ml_data" / "full_matches"
ALL_GROUPS = ["hero", "team", "lane", "matchup", "patch", "player"]


def load():
    raw, targets = [], []
    for p in sorted(DATA.glob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except: continue
        t = extract_target(d)
        if t is None: continue
        raw.append(d); targets.append(t)
    return raw, targets


def build_features_honest(raw, targets, train_idx, test_idx, groups):
    train_raw = [raw[i] for i in train_idx]
    enc = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)
    enc.player_encoder = PlayerWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)
    F = sum(len(FEATURE_GROUPS[g]) for g in groups)
    X_tr = np.empty((len(train_idx), F), dtype=float)
    X_te = np.empty((len(test_idx), F), dtype=float)
    for k, i in enumerate(train_idx):
        t = targets[i]; m = raw[i]
        X_tr[k] = extract_features(
            t.radiant_hero_ids, t.dire_hero_ids, enc,
            radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
            match=m, groups=tuple(groups),
        )
    for k, i in enumerate(test_idx):
        t = targets[i]; m = raw[i]
        X_te[k] = extract_features(
            t.radiant_hero_ids, t.dire_hero_ids, enc,
            radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
            match=m, groups=tuple(groups),
        )
    return X_tr, X_te


def winsorize(y, n=3.0):
    med = np.median(y); mad = np.median(np.abs(y - med)); sigma = 1.4826 * mad
    if sigma == 0: return y.copy()
    out = y.copy()
    out[out < med - n*sigma] = med - n*sigma
    out[out > med + n*sigma] = med + n*sigma
    return out


def make_kills_model(name, kw):
    if "xgb" in name:
        if "poisson" in name:
            return xgb.XGBRegressor(objective="count:poisson", random_state=42, tree_method="hist", verbosity=0, **kw)
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=42, tree_method="hist", verbosity=0, **kw)
    if "hgb" in name:
        if "poisson" in name:
            return HistGradientBoostingRegressor(loss="poisson", random_state=42, **kw)
        return HistGradientBoostingRegressor(random_state=42, **kw)
    if "ridge" in name:
        return Ridge(**kw)
    raise ValueError(name)


def make_duration_model(name, kw):
    if "xgb" in name:
        if "tweedie" in name:
            return xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=2.0,
                                    random_state=42, tree_method="hist", verbosity=0, **kw)
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=42, tree_method="hist", verbosity=0, **kw)
    if "hgb" in name:
        if "gamma" in name:
            return HistGradientBoostingRegressor(loss="gamma", random_state=42, **kw)
        return HistGradientBoostingRegressor(random_state=42, **kw)
    if "ridge" in name:
        return Ridge(**kw)
    raise ValueError(name)


def cv_honest(name, kw, X, y, model_fn, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, rmses, times = [], [], []
    for tr, te in kf.split(np.arange(len(X))):
        t0 = time.perf_counter()
        m = model_fn(name, kw)
        m.fit(X[tr], winsorize(y[tr]))
        pred = np.clip(np.asarray(m.predict(X[te]), dtype=float), 0.0, None)
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        times.append(time.perf_counter() - t0)
    return {
        "n_folds": n_splits,
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "train_s_mean": float(np.mean(times)),
    }


def main():
    raw, targets = load()
    n = len(raw)
    y_k = np.asarray([t.kills_total for t in targets], dtype=float)
    y_d = np.asarray([t.duration_minutes for t in targets], dtype=float)
    print(f"loaded {n} matches", flush=True)

    # Group choices (top from leaky grid)
    group_choices = [
        list(ALL_GROUPS),  # 34
        ["hero", "team", "player", "matchup"],  # 24
        ["hero", "team", "player"],  # 21
        ["hero", "player"],  # 17 (current v15 baseline for winner, also v3 for kills)
    ]

    # Top configs from leaky grid
    kills_configs = [
        ("xgb_poisson_n50_d3_lr0.1",   dict(n_estimators=50, max_depth=3, learning_rate=0.1)),
        ("xgb_poisson_n100_d4_lr0.1",  dict(n_estimators=100, max_depth=4, learning_rate=0.1)),
        ("xgb_poisson_n50_d4_lr0.1",   dict(n_estimators=50, max_depth=4, learning_rate=0.1)),
        ("xgb_poisson_n100_d3_lr0.1",  dict(n_estimators=100, max_depth=3, learning_rate=0.1)),
        ("xgb_poisson_n200_d4_lr0.05", dict(n_estimators=200, max_depth=4, learning_rate=0.05)),
        ("xgb_sq_n100_d3_lr0.05",      dict(n_estimators=100, max_depth=3, learning_rate=0.05)),
        ("xgb_poisson_n50_d3_lr0.2",   dict(n_estimators=50, max_depth=3, learning_rate=0.2)),
        ("xgb_poisson_n80_d3_lr0.1",   dict(n_estimators=80, max_depth=3, learning_rate=0.1)),
        ("xgb_poisson_n50_d5_lr0.1",   dict(n_estimators=50, max_depth=5, learning_rate=0.1)),
        ("hgb_poisson_mi300_ml15_lr0.05", dict(max_iter=300, max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=20)),
        ("ridge_a1.0",                 dict(alpha=1.0)),
    ]
    duration_configs = [
        ("xgb_sq_n50_d2_lr0.1",  dict(n_estimators=50, max_depth=2, learning_rate=0.1)),
        ("xgb_sq_n50_d3_lr0.1",  dict(n_estimators=50, max_depth=3, learning_rate=0.1)),
        ("xgb_sq_n100_d3_lr0.05", dict(n_estimators=100, max_depth=3, learning_rate=0.05)),
        ("xgb_sq_n80_d4_lr0.05", dict(n_estimators=80, max_depth=4, learning_rate=0.05)),
        ("xgb_sq_n50_d4_lr0.1",  dict(n_estimators=50, max_depth=4, learning_rate=0.1)),
        ("xgb_sq_n100_d5_lr0.05", dict(n_estimators=100, max_depth=5, learning_rate=0.05)),
        ("xgb_tweedie_n50_d3_lr0.1", dict(n_estimators=50, max_depth=3, learning_rate=0.1)),
        ("xgb_tweedie_n100_d4_lr0.05", dict(n_estimators=100, max_depth=4, learning_rate=0.05)),
        ("hgb_gamma_mi300_ml31_lr0.05", dict(max_iter=300, max_leaf_nodes=31, learning_rate=0.05, min_samples_leaf=20)),
        ("hgb_gamma_mi200_ml31_lr0.05", dict(max_iter=200, max_leaf_nodes=31, learning_rate=0.05, min_samples_leaf=20)),
        ("ridge_a10.0", dict(alpha=10.0)),
    ]

    out_fh = (ROOT / "scripts" / "grid_honest_regressors.jsonl").open("w", encoding="utf-8")
    counter = 0
    t_start = time.perf_counter()
    results = []

    # KILLS
    print("\n=== KILLS HONEST ===", flush=True)
    for g in group_choices:
        for name, kw in kills_configs:
            try:
                # Build per-fold X (slow because of encoder refit)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                maes, rmses, times = [], [], []
                for tr, te in kf.split(np.arange(n)):
                    t0 = time.perf_counter()
                    X_tr, X_te = build_features_honest(raw, targets, tr, te, g)
                    m = make_kills_model(name, kw)
                    m.fit(X_tr, winsorize(y_k[tr]))
                    pred = np.clip(np.asarray(m.predict(X_te), dtype=float), 0.0, None)
                    maes.append(mean_absolute_error(y_k[te], pred))
                    rmses.append(np.sqrt(mean_squared_error(y_k[te], pred)))
                    times.append(time.perf_counter() - t0)
                r = {
                    "n_folds": 5,
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "train_s_mean": float(np.mean(times)),
                }
                rec = {
                    "target": "kills", "name": name, "kw": kw, "groups": list(g),
                    "n_features": sum(len(FEATURE_GROUPS[x]) for x in g),
                    **r,
                }
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                results.append(rec)
                counter += 1
                elapsed = time.perf_counter() - t_start
                print(f"  [{counter:>3}] kills {name:<28} {str(g):<55} mae={r['mae_mean']:.3f}±{r['mae_std']:.3f} rmse={r['rmse_mean']:.3f} t={r['train_s_mean']:.1f}s ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    # DURATION
    print("\n=== DURATION HONEST ===", flush=True)
    for g in group_choices:
        for name, kw in duration_configs:
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                maes, rmses, times = [], [], []
                for tr, te in kf.split(np.arange(n)):
                    t0 = time.perf_counter()
                    X_tr, X_te = build_features_honest(raw, targets, tr, te, g)
                    m = make_duration_model(name, kw)
                    m.fit(X_tr, winsorize(y_d[tr]))
                    pred = np.clip(np.asarray(m.predict(X_te), dtype=float), 0.0, None)
                    maes.append(mean_absolute_error(y_d[te], pred))
                    rmses.append(np.sqrt(mean_squared_error(y_d[te], pred)))
                    times.append(time.perf_counter() - t0)
                r = {
                    "n_folds": 5,
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "train_s_mean": float(np.mean(times)),
                }
                rec = {
                    "target": "duration", "name": name, "kw": kw, "groups": list(g),
                    "n_features": sum(len(FEATURE_GROUPS[x]) for x in g),
                    **r,
                }
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                results.append(rec)
                counter += 1
                elapsed = time.perf_counter() - t_start
                print(f"  [{counter:>3}] duration {name:<28} {str(g):<55} mae={r['mae_mean']:.3f}±{r['mae_std']:.3f} rmse={r['rmse_mean']:.3f} t={r['train_s_mean']:.1f}s ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    out_fh.close()

    print("\n\n=== KILLS TOP 5 ===", flush=True)
    sub = [r for r in results if r["target"] == "kills"]
    sub.sort(key=lambda r: r["mae_mean"])
    for r in sub[:5]:
        print(f"  mae={r['mae_mean']:.3f}±{r['mae_std']:.3f} {r['name']} {r['groups']}", flush=True)
    print("\n=== DURATION TOP 5 ===", flush=True)
    sub = [r for r in results if r["target"] == "duration"]
    sub.sort(key=lambda r: r["mae_mean"])
    for r in sub[:5]:
        print(f"  mae={r['mae_mean']:.3f}±{r['mae_std']:.3f} {r['name']} {r['groups']}", flush=True)


if __name__ == "__main__":
    main()
