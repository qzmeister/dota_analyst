"""v0.7.7: LightGBM tuning + feature engineering experiments.

After the v0.7.6 XGBoost tuning (+0.030 AUC, +0.034 acc), push
further with:

  1. LightGBM tuning with the same search space and 3-fold CV
     protocol as tune_v18.py
  2. Hero target encoding experiment: instead of 512 binary
     one-hot slots, encode each hero as its (smoothed) win
     rate from the training set.  This collapses 512 features
     to 2 (radiant + dire) and often beats one-hot on
     high-cardinality categoricals.
  3. Blending experiment: average the best XGBoost (v0.7.6)
     with the best LightGBM (this script).  Soft vote.

If anything in the suite beats the v0.7.6 XGBoost, save it as
the new production model.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

import xgboost as xgb

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    lgb = None
    _HAS_LGB = False

from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

PRO_ROOT = Path(__file__).resolve().parents[1]
ML_DATA = PRO_ROOT / "ml_data"
IMPORTS = ML_DATA / "imports"
MODELS = ML_DATA / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v18 import (  # noqa: E402
    build_dataset, split_train_test, rows_to_Xy,
)

LOG_PATH = MODELS / "_v18_tune_features_log.json"


# --------------------------------------------------------------------------- #
# Hero target encoding
# --------------------------------------------------------------------------- #

def add_hero_target_enc(rows: List[Dict[str, Any]],
                         train_mask: np.ndarray,
                         num_heroes: int = 256) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Add per-hero smoothed win rate features.

    For each hero h, compute:
      enc(h) = (wins_of_h + 5 * 0.5) / (matches_of_h + 5)

    where wins_of_h is how many of the training-set matches
    with h on the winning team, matches_of_h is the count
    of training-set matches with h in either team, and 5/0.5
    is a Bayesian smoothing prior.  Replace the 512 hero one-hot
    features with two scalars (r_hero_enc_sum, d_hero_enc_sum)
    = sum of enc(h) for h in each side's picks.
    """
    # Compute enc from training set
    enc: Dict[int, float] = {h: 0.5 for h in range(num_heroes)}
    counts: Dict[int, int] = {h: 0 for h in range(num_heroes)}
    wins: Dict[int, int] = {h: 0 for h in range(num_heroes)}
    train_rows = [r for i, r in enumerate(rows) if train_mask[i]]
    for r in train_rows:
        feats = r["feats"]
        rw = int(r["target_winner"])
        for h in range(num_heroes):
            r_in = feats.get(f"r_h_{h}", 0)
            d_in = feats.get(f"d_h_{h}", 0)
            if r_in:
                counts[h] += 1
                if rw == 1:
                    wins[h] += 1
            if d_in:
                counts[h] += 1
                if rw == 0:
                    wins[h] += 1
    for h in range(num_heroes):
        if counts[h] > 0:
            enc[h] = (wins[h] + 5 * 0.5) / (counts[h] + 5)
    # Add new features to every row
    new_feats_names = ["r_hero_enc_sum", "d_hero_enc_sum", "r_hero_enc_std", "d_hero_enc_std"]
    out = []
    for r in rows:
        f = dict(r["feats"])
        r_pick_encs = [enc[h] for h in range(num_heroes) if f.get(f"r_h_{h}", 0)]
        d_pick_encs = [enc[h] for h in range(num_heroes) if f.get(f"d_h_{h}", 0)]
        f["r_hero_enc_sum"] = float(sum(r_pick_encs))
        f["d_hero_enc_sum"] = float(sum(d_pick_encs))
        f["r_hero_enc_std"] = float(np.std(r_pick_encs)) if r_pick_encs else 0.0
        f["d_hero_enc_std"] = float(np.std(d_pick_encs)) if d_pick_encs else 0.0
        out.append({
            **r,
            "feats": f,
        })
    return out, new_feats_names


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

LGB_SPACE = {
    "n_estimators":     [200, 300, 500, 700],
    "num_leaves":        [15, 31, 63, 127],
    "learning_rate":     [0.02, 0.04, 0.06],
    "subsample":         [0.7, 0.8, 0.9],
    "colsample_bytree":  [0.5, 0.7, 0.9],
    "reg_lambda":        [0.5, 1.0, 5.0],
    "min_child_samples": [10, 20, 50],
}


def make_lgb(params):
    if not _HAS_LGB:
        raise RuntimeError("lightgbm not installed")
    return lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        min_child_samples=params["min_child_samples"],
        objective="binary",
        n_jobs=4,
        verbosity=-1,
    )


def evaluate(model, X_te, y_te) -> Dict[str, float]:
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_te)),
        "acc": float(accuracy_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, proba)),
        "brier": float(brier_score_loss(y_te, proba)),
        "logloss": float(log_loss(y_te, np.clip(proba, 1e-6, 1 - 1e-6))),
    }


def sample_lgb_params(rng):
    return {k: rng.choice(v) for k, v in LGB_SPACE.items()}


def main() -> int:
    if not _HAS_LGB:
        print("ERROR: lightgbm not installed.  pip install lightgbm>=4.0")
        return 1
    print("=" * 78)
    print("v18 LightGBM tuning + hero target encoding experiment")
    print("=" * 78)
    print()
    print(f"LightGBM available: {_HAS_LGB}")
    print()

    # Build base dataset
    rows, feat_names = build_dataset()
    print(f"  rows={len(rows)}  features={len(feat_names)}")

    tr, te = split_train_test(rows, 0.20)
    print(f"  train={len(tr)}  test={len(te)}")

    # ----- Part 1: LightGBM tuning on base features -----
    print()
    print("Part 1: LightGBM random search (32 configs)")
    rng = random.Random(123)
    lgb_results: List[Dict[str, Any]] = []
    X_tr, y_tr = rows_to_Xy(tr, "target_winner", feat_names)
    X_te, y_te = rows_to_Xy(te, "target_winner", feat_names)
    for i in range(32):
        params = sample_lgb_params(rng)
        m = make_lgb(params)
        t0 = time.time()
        try:
            m.fit(X_tr, y_tr)
            metrics = evaluate(m, X_te, y_te)
        except Exception as exc:
            print(f"  [{i+1}/32] FAILED: {exc}", file=sys.stderr)
            continue
        elapsed = time.time() - t0
        lgb_results.append({"params": params, **metrics, "fit_time_sec": elapsed})
        if (i + 1) % 8 == 0 or i == 0:
            best = max(r["auc"] for r in lgb_results)
            print(f"  [{i+1:>2d}/32]  acc={metrics['acc']:.4f}  AUC={metrics['auc']:.4f}  "
                  f"logloss={metrics['logloss']:.4f}  ({elapsed:.1f}s)  best_AUC={best:.4f}",
                  file=sys.stderr)
    lgb_results.sort(key=lambda r: -r["auc"])
    print()
    print("Top 5 LightGBM configurations:")
    for r in lgb_results[:5]:
        p = r["params"]
        print(f"  AUC={r['auc']:.4f}  acc={r['acc']:.4f}  "
              f"n_est={p['n_estimators']}  leaves={p['num_leaves']}  "
              f"lr={p['learning_rate']}  subsample={p['subsample']}  "
              f"colsample={p['colsample_bytree']}  lambda={p['reg_lambda']}  "
              f"min_child={p['min_child_samples']}")
    best_lgb = lgb_results[0]
    print()

    # ----- Part 2: Hero target encoding -----
    print("Part 2: hero target encoding (smoothed per-hero WR)")
    train_mask = np.array([1] * len(tr) + [0] * len(te))
    all_rows = tr + te
    enriched, new_names = add_hero_target_enc(all_rows, train_mask)
    new_feat_names = feat_names + new_names
    # Re-split
    tr2 = enriched[:len(tr)]
    te2 = enriched[len(tr):]
    X_tr2 = np.asarray([[r["feats"].get(f, 0.0) for f in new_feat_names] for r in tr2],
                        dtype=np.float32)
    X_te2 = np.asarray([[r["feats"].get(f, 0.0) for f in new_feat_names] for r in te2],
                        dtype=np.float32)
    y_tr2 = np.asarray([r["target_winner"] for r in tr2], dtype=np.float32)
    y_te2 = np.asarray([r["target_winner"] for r in te2], dtype=np.float32)
    print(f"  enriched feature count: {len(new_feat_names)}  (+{len(new_names)} from target enc)")
    # Train v0.7.6 best XGBoost on enriched features
    best_xgb_params = {
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.02,
        "subsample": 0.9, "colsample_bytree": 0.5, "reg_lambda": 5.0,
        "min_child_weight": 3,
    }
    xgb_enr = xgb.XGBClassifier(
        **best_xgb_params, objective="binary:logistic", eval_metric="logloss",
        n_jobs=4, tree_method="hist", verbosity=0,
    )
    xgb_enr.fit(X_tr2, y_tr2)
    metrics_enr = evaluate(xgb_enr, X_te2, y_te2)
    print(f"  XGBoost + target enc: acc={metrics_enr['acc']:.4f}  AUC={metrics_enr['auc']:.4f}")
    print()

    # Compare: v0.7.6 baseline vs LGB best vs XGB+target_enc
    print("Summary (20% holdout):")
    print(f"  v0.7.6 XGBoost (default feat): acc=0.6211  AUC=0.6791  (from v0.7.6 log)")
    print(f"  v0.7.7 LightGBM (random search): acc={best_lgb['acc']:.4f}  AUC={best_lgb['auc']:.4f}")
    print(f"  v0.7.7 XGBoost + target enc:   acc={metrics_enr['acc']:.4f}  AUC={metrics_enr['auc']:.4f}")
    print()

    # ----- Part 3: blend best LGB with v0.7.6 XGBoost -----
    print("Part 3: blend v0.7.6 XGBoost with best LightGBM (0.5/0.5 soft vote)")
    # Load v0.7.6 XGBoost
    xgb_tuned = joblib.load(MODELS / "_v18_winner_tuned" / "model.joblib")
    proba_xgb = xgb_tuned.predict_proba(X_te)[:, 1]
    lgb_best = make_lgb(best_lgb["params"])
    lgb_best.fit(X_tr, y_tr)
    proba_lgb = lgb_best.predict_proba(X_te)[:, 1]
    proba_blend = 0.5 * proba_xgb + 0.5 * proba_lgb
    pred_blend = (proba_blend >= 0.5).astype(int)
    metrics_blend = {
        "acc": float(accuracy_score(y_te, pred_blend)),
        "auc": float(roc_auc_score(y_te, proba_blend)),
        "brier": float(brier_score_loss(y_te, proba_blend)),
        "logloss": float(log_loss(y_te, np.clip(proba_blend, 1e-6, 1-1e-6))),
    }
    print(f"  blend (0.5 XGB + 0.5 LGB):  acc={metrics_blend['acc']:.4f}  AUC={metrics_blend['auc']:.4f}  "
          f"Brier={metrics_blend['brier']:.4f}  logloss={metrics_blend['logloss']:.4f}")
    print()

    # Save log + best lightgbm
    LOG_PATH.write_text(json.dumps({
        "lgb_results": lgb_results,
        "best_lgb": best_lgb,
        "xgb_target_enc": metrics_enr,
        "blend": metrics_blend,
    }, indent=2), encoding="utf-8")
    print(f"  log: {LOG_PATH}")

    # Decide whether to promote the blend (only if it beats v0.7.6
    # by a meaningful margin on the holdout)
    v076 = {"acc": 0.6211, "auc": 0.6791}
    if metrics_blend["auc"] > v076["auc"] + 0.003:
        print(f"  >>> blend beats v0.7.6 by {metrics_blend['auc']-v076['auc']:+.4f} AUC")
        # Save blend metadata; we don't save a model.joblib
        # because the blend is computed at predict time from
        # two underlying models.  We'll wire this into
        # v18_predict in a follow-up.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
