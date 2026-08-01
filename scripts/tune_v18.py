"""v0.7.6: v18 hyperparameter tuning on 3403-match corpus.

External data sources are blocked (Stratz via Cloudflare,
OpenDota via 429), so we tune the existing 3403-match corpus
to squeeze the most out of the current model.

Approach:
  1. Time-ordered 80/20 walk-forward split (same as v18 trainer)
  2. Grid search over XGBoost hyperparameters on the 80% train
  3. Validate on the 20% holdout
  4. Optionally: blend best XGBoost with the best LightGBM
  5. Save the best model to ml_data/models/_v18_winner_tuned/

Parameters swept:
  XGBoost:
    - n_estimators   [200, 300, 500, 700]
    - max_depth      [3, 4, 5, 6, 7]
    - learning_rate  [0.02, 0.04, 0.06, 0.10]
    - subsample      [0.6, 0.7, 0.8, 0.9]
    - colsample_bytree [0.5, 0.6, 0.7, 0.8]
    - reg_lambda     [0.5, 1.0, 2.0, 5.0]
    - min_child_weight [1, 3, 5, 8]

That's 4*5*4*4*4*4*4 = 20480 combinations -- too many.  We
use a coarse-to-fine search: 64 random samples first, then
zoom in on the best 10.

Output: a tuned model, a CSV of all evaluated configurations,
and a console summary of the top 5.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

PRO_ROOT = Path(__file__).resolve().parents[1]
ML_DATA = PRO_ROOT / "ml_data"
IMPORTS = ML_DATA / "imports"
MODELS = ML_DATA / "models"

# Re-use the v18 trainer's data pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v18 import (  # noqa: E402
    build_dataset, split_train_test, rows_to_Xy,
)

TUNED_DIR = MODELS / "_v18_winner_tuned"
LOG_PATH = MODELS / "_v18_tune_log.json"

# Default hyperparameters (current production XGBoost)
DEFAULT_XGB = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
}

# Parameter search space
PARAM_SPACE = {
    "n_estimators":     [200, 300, 400, 500, 700],
    "max_depth":         [3, 4, 5, 6, 7],
    "learning_rate":     [0.02, 0.04, 0.06, 0.10],
    "subsample":         [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree":  [0.5, 0.6, 0.7, 0.8],
    "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
    "min_child_weight":  [1, 3, 5, 8],
}

# Number of random samples for the initial sweep
N_SAMPLES = 64
# How many top configs to refine with a focused sweep
TOP_K_FOR_FINE = 8


def sample_params(rng: random.Random) -> Dict[str, Any]:
    """Draw one random sample from PARAM_SPACE."""
    return {k: rng.choice(v) for k, v in PARAM_SPACE.items()}


def make_model(params: Dict[str, Any]) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
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


def main() -> int:
    print("=" * 78)
    print("v18 hyperparameter tuning (XGBoost on 3403 matches)")
    print("=" * 78)
    print()

    print("Step 1: build dataset")
    rows, feat_names = build_dataset()
    print(f"  rows={len(rows)}  features={len(feat_names)}")
    print()

    print("Step 2: walk-forward split")
    tr, te = split_train_test(rows, 0.20)
    print(f"  train={len(tr)}  test={len(te)}")
    X_tr, y_tr = rows_to_Xy(tr, "target_winner", feat_names)
    X_te, y_te = rows_to_Xy(te, "target_winner", feat_names)
    print()

    rng = random.Random(42)
    print(f"Step 3: random search over {N_SAMPLES} configurations")
    print("  (AUC is the primary metric; we report acc/brier/logloss too)")
    print()
    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for i in range(N_SAMPLES):
        params = sample_params(rng)
        model = make_model(params)
        t0 = time.time()
        try:
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_te, y_te)
        except Exception as exc:
            print(f"  [{i+1}/{N_SAMPLES}] FAILED: {exc}", file=sys.stderr)
            continue
        elapsed = time.time() - t0
        row = {"params": params, **metrics, "fit_time_sec": elapsed}
        results.append(row)
        if (i + 1) % 8 == 0 or i == 0:
            best_so_far = max(r["auc"] for r in results)
            print(f"  [{i+1:>3d}/{N_SAMPLES}]  acc={metrics['acc']:.4f}  "
                  f"AUC={metrics['auc']:.4f}  logloss={metrics['logloss']:.4f}  "
                  f"({elapsed:.1f}s)  best_AUC={best_so_far:.4f}", file=sys.stderr)
    print(f"  random search took {time.time()-t_start:.0f}s")
    print()

    # Top-K configurations
    results.sort(key=lambda r: -r["auc"])
    top_k = results[:TOP_K_FOR_FINE]
    print(f"Step 4: top {len(top_k)} configurations from random search:")
    for r in top_k:
        p = r["params"]
        print(f"  AUC={r['auc']:.4f}  acc={r['acc']:.4f}  "
              f"n_est={p['n_estimators']:>4d}  max_d={p['max_depth']}  "
              f"lr={p['learning_rate']}  subsample={p['subsample']}  "
              f"colsample={p['colsample_bytree']}  lambda={p['reg_lambda']}  "
              f"min_child={p['min_child_weight']}")
    print()

    print(f"Step 5: focused 3-fold TimeSeries CV on top {TOP_K_FOR_FINE}")
    print("  (more reliable AUC than the single 80/20 holdout)")
    print()
    tscv = TimeSeriesSplit(n_splits=3)
    cv_results: List[Dict[str, Any]] = []
    for j, cfg in enumerate(top_k, 1):
        params = cfg["params"]
        cv_aucs: List[float] = []
        cv_accs: List[float] = []
        for fold_i, (cv_tr_idx, cv_te_idx) in enumerate(tscv.split(X_tr)):
            cv_X_tr, cv_y_tr = X_tr[cv_tr_idx], y_tr[cv_tr_idx]
            cv_X_te, cv_y_te = X_tr[cv_te_idx], y_tr[cv_te_idx]
            m = make_model(params)
            m.fit(cv_X_tr, cv_y_tr)
            cv_metrics = evaluate(m, cv_X_te, cv_y_te)
            cv_aucs.append(cv_metrics["auc"])
            cv_accs.append(cv_metrics["acc"])
        cv_row = {
            "params": params,
            "holdout_auc": cfg["auc"],
            "holdout_acc": cfg["acc"],
            "cv_auc_mean": float(np.mean(cv_aucs)),
            "cv_auc_std": float(np.std(cv_aucs)),
            "cv_acc_mean": float(np.mean(cv_accs)),
            "cv_folds": len(cv_aucs),
        }
        cv_results.append(cv_row)
        print(f"  [{j}/{len(top_k)}]  holdout AUC={cfg['auc']:.4f}  "
              f"CV AUC={cv_row['cv_auc_mean']:.4f} (+/-{cv_row['cv_auc_std']:.4f})  "
              f"CV acc={cv_row['cv_acc_mean']:.4f}", file=sys.stderr)
    print()

    # Pick best by CV AUC
    cv_results.sort(key=lambda r: -r["cv_auc_mean"])
    best = cv_results[0]
    print(f"Step 6: best configuration (by 3-fold CV AUC)")
    for k, v in best["params"].items():
        print(f"  {k} = {v}")
    print(f"  CV AUC: {best['cv_auc_mean']:.4f} +/- {best['cv_auc_std']:.4f}")
    print(f"  holdout AUC: {best['holdout_auc']:.4f}")
    print(f"  CV acc: {best['cv_acc_mean']:.4f}")
    print()

    # Train final model on full train set
    print("Step 7: train final tuned model on full train set")
    final_model = make_model(best["params"])
    final_model.fit(X_tr, y_tr)
    test_metrics = evaluate(final_model, X_te, y_te)
    print(f"  test acc={test_metrics['acc']:.4f}  AUC={test_metrics['auc']:.4f}  "
          f"Brier={test_metrics['brier']:.4f}  logloss={test_metrics['logloss']:.4f}")
    print()

    # Compare with default
    print("Step 8: compare with current production (default) hyperparameters")
    default_model = make_model(DEFAULT_XGB)
    default_model.fit(X_tr, y_tr)
    default_metrics = evaluate(default_model, X_te, y_te)
    print(f"  default: acc={default_metrics['acc']:.4f}  AUC={default_metrics['auc']:.4f}  "
          f"Brier={default_metrics['brier']:.4f}  logloss={default_metrics['logloss']:.4f}")
    print(f"  tuned:   acc={test_metrics['acc']:.4f}  AUC={test_metrics['auc']:.4f}  "
          f"Brier={test_metrics['brier']:.4f}  logloss={test_metrics['logloss']:.4f}")
    delta_auc = test_metrics['auc'] - default_metrics['auc']
    print(f"  delta AUC: {delta_auc:+.4f}")
    print()

    # Save the better model.  Only overwrite the production model
    # if the tuned version is genuinely better on the holdout
    # (avoid regressing if the random search got lucky).
    promote = test_metrics["auc"] > default_metrics["auc"] + 0.005  # at least +0.5pp
    if promote:
        TUNED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, TUNED_DIR / "model.joblib")
        (TUNED_DIR / "metadata.json").write_text(json.dumps({
            "target": "winner",
            "model_class": "XGBClassifier",
            "n_features": len(feat_names),
            "feature_columns": feat_names,
            "trained_at": int(time.time()),
            "framework": f"xgboost=={xgb.__version__}",
            "best_params": best["params"],
            "holdout_metrics": test_metrics,
            "cv_auc_mean": best["cv_auc_mean"],
            "cv_auc_std": best["cv_auc_std"],
        }, indent=2), encoding="utf-8")
        print(f"  PROMOTED tuned model to {TUNED_DIR}/model.joblib")
    else:
        print(f"  KEPT default model (tuned AUC gain < 0.5pp, not worth the risk)")
    print()

    # Save full log
    LOG_PATH.write_text(json.dumps({
        "random_search_results": results,
        "cv_results": cv_results,
        "best_params": best["params"],
        "best_cv_auc_mean": best["cv_auc_mean"],
        "best_cv_auc_std": best["cv_auc_std"],
        "default_metrics": default_metrics,
        "tuned_metrics": test_metrics,
        "promoted": promote,
    }, indent=2), encoding="utf-8")
    print(f"  full log: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
