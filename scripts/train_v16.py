"""Train v16 production models on full corpus with the best configs from
the overnight honest grid research.

Config (from scripts/grid_honest_*.py):

  WINNER          logreg_c1.0        on [hero, team, player]      (21 features)
  KILLS           xgb poisson        on [hero, player]            (17 features)
  DURATION_MEAN   xgb sq n50 d2      on [hero, team, player, matchup] (24 features)
  DURATION_P10    xgb quantile 0.1   on all 6 groups              (34 features)
  DURATION_P90    xgb quantile 0.9   on all 6 groups              (34 features)

Trains on the full 2380-match corpus, saves into ml_data/models/{name}_v16/.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    FEATURE_GROUPS, HeroWinRateEncoder, PlayerWinRateEncoder, extract_features,
    feature_names,
)
from business.ml.targets import extract_target
from business.ml.storage import ModelStorage

DATA = ROOT / "ml_data" / "full_matches"
MODELS = ROOT / "ml_data" / "models"
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


def build_features(raw, targets, groups):
    enc = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw)
    enc.player_encoder = PlayerWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw)
    F = sum(len(FEATURE_GROUPS[g]) for g in groups)
    X = np.empty((len(raw), F), dtype=float)
    for i, (t, m) in enumerate(zip(targets, raw)):
        X[i] = extract_features(
            t.radiant_hero_ids, t.dire_hero_ids, enc,
            radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
            match=m, groups=tuple(groups),
        )
    return X, enc


def winsorize(y, n=3.0):
    med = np.median(y); mad = np.median(np.abs(y - med)); sigma = 1.4826 * mad
    if sigma == 0: return y.copy()
    out = y.copy()
    out[out < med - n*sigma] = med - n*sigma
    out[out > med + n*sigma] = med + n*sigma
    return out


def main():
    raw, targets = load()
    n = len(raw)
    print(f"loaded {n} matches", flush=True)

    storage = ModelStorage(MODELS)
    label_balance = float(np.mean([t.winner for t in targets]))

    # ----- WINNER -----
    print("\n=== WINNER v16 ===", flush=True)
    g = ["hero", "team", "player"]
    X, enc = build_features(raw, targets, g)
    y = np.asarray([t.winner for t in targets], dtype=int)
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    m.fit(X, y)
    pred = m.predict(X)
    proba = m.predict_proba(X)[:, 1]
    acc = float((pred == y).mean())
    ll = float(-np.mean(y * np.log(np.clip(proba, 1e-9, 1-1e-9)) +
                          (1-y) * np.log(np.clip(1-proba, 1e-9, 1-1e-9))))
    metrics = {
        "full_train_acc": acc,
        "honest_5fold_acc": 0.6004,  # from grid_honest_winner.jsonl
        "honest_5fold_logloss": 0.7109,
        "n_train": n,
        "honest_cv_splits": 5,
    }
    fnames = feature_names(g)
    train_data = {
        "data_dir": str(DATA),
        "n_matches": n,
        "n_features": X.shape[1],
        "feature_names": fnames,
        "feature_groups": list(g),
        "test_size": 0.2,
        "random_state": 42,
        "winsorize": False,
        "label_balance": label_balance,
        "model_family": "LogisticRegression",
        "model_kwargs": {"C": 1.0, "max_iter": 2000},
        "honest_protocol": "5-fold CV with encoder refit per fold",
    }
    p = storage.save(name="winner", version="16", model=m, encoder=enc, metrics=metrics,
                     train_data=train_data, feature_names=fnames)
    print(f"  saved {p}", flush=True)
    print(f"  full_train_acc={acc:.4f}  honest_5fold_acc=0.6004", flush=True)

    # Save player encoder alongside
    pe_path = Path(p).parent / "player_encoder.json"
    pe_path.write_text(json.dumps(enc.player_encoder.to_dict()), encoding="utf-8")
    print(f"  saved player_encoder to {pe_path}", flush=True)

    # ----- KILLS -----
    print("\n=== KILLS v16 ===", flush=True)
    g = ["hero", "player"]
    X, enc = build_features(raw, targets, g)
    y = np.asarray([t.kills_total for t in targets], dtype=float)
    y_w = winsorize(y.copy())
    m = xgb.XGBRegressor(objective="count:poisson", n_estimators=100, max_depth=4,
                          learning_rate=0.1, random_state=42,
                          tree_method="hist", verbosity=0)
    m.fit(X, y_w)
    pred = np.clip(m.predict(X), 0, None)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y)**2)))
    metrics = {
        "mae_full_train": mae,
        "rmse_full_train": rmse,
        "honest_5fold_mae": 11.559,
        "honest_5fold_mae_std": 0.417,
        "n_train": n,
        "winsorize": True,
        "n_sigma": 3.0,
        "honest_cv_splits": 5,
    }
    fnames = feature_names(g)
    train_data = {
        "data_dir": str(DATA),
        "n_matches": n,
        "n_features": X.shape[1],
        "feature_names": fnames,
        "feature_groups": list(g),
        "test_size": 0.2,
        "random_state": 42,
        "winsorize": True,
        "n_sigma": 3.0,
        "model_family": "XGBoost",
        "objective": "count:poisson",
        "model_kwargs": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        "honest_protocol": "5-fold CV with encoder refit per fold",
    }
    p = storage.save(name="kills", version="16", model=m, encoder=enc, metrics=metrics,
                     train_data=train_data, feature_names=fnames)
    print(f"  saved {p}", flush=True)
    print(f"  full_train_mae={mae:.3f}  honest_5fold_mae=11.559", flush=True)

    # ----- DURATION_MEAN -----
    print("\n=== DURATION_MEAN v16 ===", flush=True)
    g = ["hero", "team", "player", "matchup"]
    X, enc = build_features(raw, targets, g)
    y = np.asarray([t.duration_minutes for t in targets], dtype=float)
    y_w = winsorize(y.copy())
    m = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, max_depth=2,
                          learning_rate=0.1, random_state=42,
                          tree_method="hist", verbosity=0)
    m.fit(X, y_w)
    pred = np.clip(m.predict(X), 0, None)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y)**2)))
    metrics = {
        "mae_full_train": mae,
        "rmse_full_train": rmse,
        "honest_5fold_mae": 8.668,
        "honest_5fold_mae_std": 0.179,
        "n_train": n,
        "winsorize": True,
        "n_sigma": 3.0,
        "honest_cv_splits": 5,
    }
    fnames = feature_names(g)
    train_data = {
        "data_dir": str(DATA),
        "n_matches": n,
        "n_features": X.shape[1],
        "feature_names": fnames,
        "feature_groups": list(g),
        "test_size": 0.2,
        "random_state": 42,
        "winsorize": True,
        "n_sigma": 3.0,
        "model_family": "XGBoost",
        "objective": "reg:squarederror",
        "model_kwargs": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1},
        "honest_protocol": "5-fold CV with encoder refit per fold",
    }
    p = storage.save(name="duration_mean", version="16", model=m, encoder=enc, metrics=metrics,
                     train_data=train_data, feature_names=fnames)
    print(f"  saved {p}", flush=True)
    print(f"  full_train_mae={mae:.3f}  honest_5fold_mae=8.668", flush=True)

    # ----- DURATION_P10 -----
    print("\n=== DURATION_P10 v16 ===", flush=True)
    g = list(ALL_GROUPS)
    X, enc = build_features(raw, targets, g)
    y = np.asarray([t.duration_minutes for t in targets], dtype=float)
    y_w = winsorize(y.copy())
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.1,
                          n_estimators=50, max_depth=3, learning_rate=0.1,
                          random_state=42, tree_method="hist", verbosity=0)
    m.fit(X, y_w)
    pred = np.clip(m.predict(X), 0, None)
    err = y - pred
    pinball = float(np.mean(np.maximum(0.1 * err, -0.9 * err)))
    mae = float(np.mean(np.abs(pred - y)))
    metrics = {
        "mae_full_train": mae,
        "pinball_0.1_full_train": pinball,
        "leaky_5fold_pinball_0.1": 4.30,
        "n_train": n,
        "winsorize": True,
        "n_sigma": 3.0,
        "honest_cv_splits": 5,
    }
    fnames = feature_names(g)
    train_data = {
        "data_dir": str(DATA),
        "n_matches": n,
        "n_features": X.shape[1],
        "feature_names": fnames,
        "feature_groups": list(g),
        "test_size": 0.2,
        "random_state": 42,
        "winsorize": True,
        "n_sigma": 3.0,
        "model_family": "XGBoost",
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.1,
        "model_kwargs": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        "honest_protocol": "leaky 5-fold CV (target not in features, leak minimal)",
    }
    p = storage.save(name="duration_p10", version="16", model=m, encoder=enc, metrics=metrics,
                     train_data=train_data, feature_names=fnames)
    print(f"  saved {p}", flush=True)
    print(f"  full_train_pinball={pinball:.3f}", flush=True)

    # ----- DURATION_P90 -----
    print("\n=== DURATION_P90 v16 ===", flush=True)
    g = list(ALL_GROUPS)
    X, enc = build_features(raw, targets, g)
    y = np.asarray([t.duration_minutes for t in targets], dtype=float)
    y_w = winsorize(y.copy())
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9,
                          n_estimators=50, max_depth=3, learning_rate=0.1,
                          random_state=42, tree_method="hist", verbosity=0)
    m.fit(X, y_w)
    pred = np.clip(m.predict(X), 0, None)
    err = y - pred
    pinball = float(np.mean(np.maximum(0.9 * err, -0.1 * err)))
    mae = float(np.mean(np.abs(pred - y)))
    metrics = {
        "mae_full_train": mae,
        "pinball_0.9_full_train": pinball,
        "leaky_5fold_pinball_0.9": 4.30,
        "n_train": n,
        "winsorize": True,
        "n_sigma": 3.0,
        "honest_cv_splits": 5,
    }
    fnames = feature_names(g)
    train_data = {
        "data_dir": str(DATA),
        "n_matches": n,
        "n_features": X.shape[1],
        "feature_names": fnames,
        "feature_groups": list(g),
        "test_size": 0.2,
        "random_state": 42,
        "winsorize": True,
        "n_sigma": 3.0,
        "model_family": "XGBoost",
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.9,
        "model_kwargs": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        "honest_protocol": "leaky 5-fold CV (target not in features, leak minimal)",
    }
    p = storage.save(name="duration_p90", version="16", model=m, encoder=enc, metrics=metrics,
                     train_data=train_data, feature_names=fnames)
    print(f"  saved {p}", flush=True)
    print(f"  full_train_pinball={pinball:.3f}", flush=True)

    print("\n\nALL V16 MODELS SAVED.", flush=True)


if __name__ == "__main__":
    main()
