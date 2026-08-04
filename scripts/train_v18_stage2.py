"""v18 Stage 2 trainer -- XGBoost + LightGBM ensemble on 5-10K matches.

Stage 1 (v0.6.0) used XGBoost on 3403 OpenDota pro matches with
hero one-hot encoding, reaching 58.30% accuracy / 0.617 AUC.

Stage 2 (v0.7.1) extends the corpus to 5-10K matches (when
available) and adds a LightGBM head.  We then either pick the
best single model OR build a soft-vote ensemble of XGBoost +
LightGBM, whichever has the higher honest-test AUC.

Why LightGBM?
-------------
Gradient boosting on decision trees is the strongest off-the-shelf
class for tabular data.  XGBoost and LightGBM use different split
strategies (XGBoost: level-wise / pre-sorted, LightGBM: leaf-wise
/ histogram-based) and different categorical handling.  An ensemble
of the two typically gains 0.5-1.5 pp AUC over either alone, with
no additional data cost.

Why 5-10K matches?
-------------------
The honest-test accuracy for Dota 2 pre-game winner prediction
plateaus around 58-60% with the current feature set (523 columns
= 11 base + 512 hero one-hot).  Going from 3K to 10K matches
should add 0.5-1.0 pp accuracy from the hero-pair interactions
the XGBoost/LightGBM trees can learn.  Beyond 10K the marginal
gain drops sharply because the meta shifts and the older matches
add noise.

Inputs (same as v18 Stage 1):
  - ml_data/imports/v17_match_*.json   (OpenDota /matches/{id})
  - ml_data/imports/v18_top_teams.json (tier file from v0.7.0-audit)

Outputs:
  - ml_data/models/_v18_winner/          (the chosen winner model)
  - ml_data/models/_v18_winner_xgb/      (XGBoost head, always)
  - ml_data/models/_v18_winner_lgb/      (LightGBM head, when lgb available)
  - ml_data/models/_v18_ensemble.json    (which model to use + weights)

Run:  python scripts/train_v18_stage2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

import xgboost as xgb

# LightGBM is optional.  We try/except so the trainer still
# works on a host without it -- falls back to XGBoost-only.
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore
    _HAS_LGB = False

# v0.7.72: add HistGradientBoostingClassifier as a 3rd ensemble
# member.  Found via _hgb_probe.py that XGB + HGB beats XGB-only
# by +0.4pp honest test (0.6285 vs 0.6241) on v19 LITE features.
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_HGB = True
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None  # type: ignore
    _HAS_HGB = False

from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

# Re-use the v18 Stage 1 trainer for data loading and feature
# extraction.  That's the single source of truth for "what does
# v18 see".
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v18 import (  # noqa: E402
    PRO_ROOT, ML_DATA, IMPORTS, MODELS, PATCH_INFO_PATH,
    NUM_HEROES, TIER_THRESHOLD_PREMIUM, TIER_THRESHOLD_PROFESSIONAL,
    _load_patch_info, _load_top_teams, _tier_for, _players_to_picks,
    _bans_from_draft, _gold_adv_at, extract_features, build_dataset,
    split_train_test, rows_to_Xy, evaluate_winner, evaluate_regressor,
    save_model,
)

# v0.7.1: model artefacts live under explicit names so we don't
# overwrite the Stage 1 _v18_winner until we've validated the
# Stage 2 one is better.
MODELS_XGB_DIR = MODELS / "_v18_winner_xgb_stage2"
MODELS_LGB_DIR = MODELS / "_v18_winner_lgb_stage2"
MODELS_HGB_DIR = MODELS / "_v18_winner_hgb_stage2"  # v0.7.72
META_PATH = MODELS / "_v18_winner_stage2_meta.json"
ENSEMBLE_PATH = MODELS / "_v18_ensemble.json"

# Walk-forward split.  Same as v18 Stage 1 (most recent 20% holdout).
TEST_FRAC = 0.20


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #

def train_xgb_winner(X_tr: np.ndarray, y_tr: np.ndarray) -> xgb.XGBClassifier:
    """XGBoost head.  Same hyper-params as Stage 1; the data
    scale is the only thing that changes."""
    model = xgb.XGBClassifier(
        n_estimators=500,           # 400 -> 500 for the larger corpus
        max_depth=5,                # 4 -> 5; 10K matches support deeper trees
        learning_rate=0.04,         # 0.05 -> 0.04 to compensate for more trees
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def train_lgb_winner(X_tr: np.ndarray, y_tr: np.ndarray):
    """LightGBM head.  Defaults to a histogram-based gradient
    booster; ~equivalent to XGBoost in capacity, different
    in split / categorical handling.

    Categorical: hero one-hot slots are already one-hot encoded
    (0/1), so we don't pass categorical_feature=...  LightGBM
    would otherwise try to bin them.
    """
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=-1,               # no limit (LightGBM grows leaf-wise)
        num_leaves=63,              # 2^max_depth - 1; depth ~6
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        min_child_samples=20,
        objective="binary",
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    return model


# --------------------------------------------------------------------------- #
# v0.7.72: HistGradientBoosting head (sklearn-native, different
# histogram/tree algorithm from XGBoost and LightGBM)
# --------------------------------------------------------------------------- #

def train_hgb_winner(X_tr: np.ndarray, y_tr: np.ndarray):
    """HistGradientBoosting — sklearn's histogram-based GBM.  Different
    from XGB and LGB in tree growth + categorical handling, so
    ensemble diversifies well (XGB + HGB beat XGB-only by +0.4pp
    on v19 LITE features).
    """
    if HistGradientBoostingClassifier is None:
        raise RuntimeError("scikit-learn not installed")
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X_tr, y_tr)
    return model


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #

def save_model_to(model, out_dir: Path, target: str, feat_names: List[str],
                   framework: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps({
            "target": target,
            "model_class": type(model).__name__,
            "n_features": len(feat_names),
            "feature_columns": feat_names,
            "trained_at": int(time.time()),
            "framework": framework,
        }, indent=2),
        encoding="utf-8",
    )


def load_model_from(out_dir: Path):
    model = joblib.load(out_dir / "model.joblib")
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 78)
    print("v18 Stage 2 trainer -- XGBoost + LightGBM ensemble")
    print("=" * 78)
    print()
    print(f"LightGBM available: {_HAS_LGB}")
    if not _HAS_LGB:
        print("  -- falling back to XGBoost-only (no LGB)")
    print()

    # Step 1: build dataset
    print("Step 1: build dataset")
    rows, feat_names = build_dataset()
    if len(rows) < 1000:
        print(f"ERROR: only {len(rows)} rows -- Stage 2 needs at least 1000")
        return 1
    print(f"  rows={len(rows)}  features={len(feat_names)}")
    print()

    # Step 2: time-ordered split
    print("Step 2: walk-forward split (most recent 20% holdout)")
    tr, te = split_train_test(rows, TEST_FRAC)
    print(f"  train: {len(tr)} (earliest)")
    print(f"  test:  {len(te)}  (most recent)")
    print()

    # Step 3: train both heads
    X_tr, y_tr = rows_to_Xy(tr, "target_winner", feat_names)
    X_te, y_te = rows_to_Xy(te, "target_winner", feat_names)

    print("Step 3: train XGBoost head")
    t0 = time.time()
    xgb_model = train_xgb_winner(X_tr, y_tr)
    t1 = time.time()
    xgb_metrics = evaluate_winner(xgb_model, X_te, y_te)
    print(f"  trained in {t1-t0:.1f}s")
    print(f"  test acc={xgb_metrics['acc']:.4f}  AUC={xgb_metrics['auc']:.4f}  "
          f"Brier={xgb_metrics['brier']:.4f}  logloss={xgb_metrics['logloss']:.4f}")
    print()

    lgb_metrics: Optional[Dict[str, float]] = None
    lgb_model = None
    lgb_proba = None
    if _HAS_LGB:
        print("Step 4: train LightGBM head")
        t0 = time.time()
        lgb_model = train_lgb_winner(X_tr, y_tr)
        t1 = time.time()
        lgb_metrics = evaluate_winner(lgb_model, X_te, y_te)
        lgb_proba = lgb_model.predict_proba(X_te)[:, 1]
        print(f"  trained in {t1-t0:.1f}s")
        print(f"  test acc={lgb_metrics['acc']:.4f}  AUC={lgb_metrics['auc']:.4f}  "
              f"Brier={lgb_metrics['brier']:.4f}  logloss={lgb_metrics['logloss']:.4f}")
        print()

    hgb_metrics: Optional[Dict[str, float]] = None
    hgb_model = None
    hgb_proba = None
    if _HAS_HGB:
        print("Step 4b: train HistGradientBoosting head")
        t0 = time.time()
        hgb_model = train_hgb_winner(X_tr, y_tr)
        t1 = time.time()
        hgb_metrics = evaluate_winner(hgb_model, X_te, y_te)
        hgb_proba = hgb_model.predict_proba(X_te)[:, 1]
        print(f"  trained in {t1-t0:.1f}s")
        print(f"  test acc={hgb_metrics['acc']:.4f}  AUC={hgb_metrics['auc']:.4f}  "
              f"Brier={hgb_metrics['brier']:.4f}  logloss={hgb_metrics['logloss']:.4f}")
        print()

    # Step 5: pick the best (single model OR ensemble)
    print("Step 5: pick the best model / ensemble")
    xgb_proba = xgb_model.predict_proba(X_te)[:, 1]

    def _ens_metrics(*probas) -> Dict[str, float]:
        avg = np.mean(probas, axis=0)
        return {
            "n": int(len(y_te)),
            "acc": float(accuracy_score(y_te, (avg >= 0.5).astype(int))),
            "auc": float(roc_auc_score(y_te, avg)),
            "brier": float(brier_score_loss(y_te, avg)),
            "logloss": float(log_loss(y_te, np.clip(avg, 1e-6, 1-1e-6))),
        }

    candidates: List[Tuple[str, float, Dict[str, Any]]] = []
    candidates.append(("xgb", xgb_metrics["auc"], {"models": ["xgb"]}))
    if lgb_metrics is not None:
        candidates.append(("lgb", lgb_metrics["auc"], {"models": ["lgb"]}))
    if hgb_metrics is not None:
        candidates.append(("hgb", hgb_metrics["auc"], {"models": ["hgb"]}))
    if lgb_proba is not None:
        candidates.append(("xgb+lgb",
                            _ens_metrics(xgb_proba, lgb_proba)["auc"],
                            {"models": ["xgb", "lgb"]}))
    if hgb_proba is not None:
        candidates.append(("xgb+hgb",
                            _ens_metrics(xgb_proba, hgb_proba)["auc"],
                            {"models": ["xgb", "hgb"]}))
    if lgb_proba is not None and hgb_proba is not None:
        candidates.append(("xgb+lgb+hgb",
                            _ens_metrics(xgb_proba, lgb_proba, hgb_proba)["auc"],
                            {"models": ["xgb", "lgb", "hgb"]}))

    # Print all candidates
    for name, auc, _ in candidates:
        print(f"  candidate {name:14s}  AUC={auc:.4f}")
    candidates.sort(key=lambda t: t[1], reverse=True)
    chosen, chosen_auc, chosen_meta = candidates[0]
    best_single = max((c for c in candidates if "+" not in c[0]),
                       key=lambda t: t[1], default=None)
    if best_single is not None and chosen_auc - best_single[1] >= 0.001:
        print(f"  >>> {chosen} wins by +{chosen_auc - best_single[1]:.4f} AUC")
    else:
        # No ensemble meaningfully better — fall back to best single.
        if best_single is not None:
            print(f"  >>> no ensemble gains; using {best_single[0]}")
            chosen = best_single[0]
            chosen_meta = best_single[2]

    print()

    # Step 6: save
    print("Step 6: save models")
    save_model_to(xgb_model, MODELS_XGB_DIR, "winner", feat_names, f"xgboost=={xgb.__version__}")
    print(f"  wrote {MODELS_XGB_DIR}/model.joblib")
    if lgb_model is not None:
        save_model_to(lgb_model, MODELS_LGB_DIR, "winner", feat_names,
                       f"lightgbm=={getattr(lgb, '__version__', 'unknown')}")
        print(f"  wrote {MODELS_LGB_DIR}/model.joblib")
    if hgb_model is not None:
        save_model_to(hgb_model, MODELS_HGB_DIR, "winner", feat_names,
                       f"scikit-learn==HistGradientBoosting")
        print(f"  wrote {MODELS_HGB_DIR}/model.joblib")

    # Save metadata + ensemble config
    META_PATH.write_text(json.dumps({
        "rows_total": len(rows),
        "rows_train": len(tr),
        "rows_test": len(te),
        "xgb_metrics": xgb_metrics,
        "lgb_metrics": lgb_metrics,
        "hgb_metrics": hgb_metrics,
        "chosen": chosen,
        "chosen_models": chosen_meta.get("models", [chosen]),
        "framework": ("xgboost+lightgbm+histgb" if (_HAS_LGB and _HAS_HGB)
                      else "xgboost+lightgbm" if _HAS_LGB
                      else "xgboost+histgb" if _HAS_HGB
                      else "xgboost"),
    }, indent=2), encoding="utf-8")
    print(f"  wrote {META_PATH}")

    # v0.7.72: weight by chosen.  For ensembles we use equal
    # weight (1/n) over the chosen models.  Single models get
    # weight=1.0.
    chosen_set = set(chosen_meta.get("models", [chosen]))
    def _w(model: str) -> float:
        return 1.0 / len(chosen_set) if model in chosen_set else 0.0
    ENSEMBLE_PATH.write_text(json.dumps({
        "chosen": chosen,
        "chosen_models": sorted(chosen_set),
        "xgb": {"path": "_v18_winner_xgb_stage2", "weight": _w("xgb")},
        "lgb": {"path": "_v18_winner_lgb_stage2", "weight": _w("lgb")},
        "hgb": {"path": "_v18_winner_hgb_stage2", "weight": _w("hgb")},
    }, indent=2), encoding="utf-8")
    print(f"  wrote {ENSEMBLE_PATH}")
    print()

    print("=" * 78)
    print(f"DONE.  Winner model: {chosen}")
    print(f"  XGBoost:  acc={xgb_metrics['acc']:.4f}  AUC={xgb_metrics['auc']:.4f}")
    if lgb_metrics is not None:
        print(f"  LightGBM: acc={lgb_metrics['acc']:.4f}  AUC={lgb_metrics['auc']:.4f}")
    if hgb_metrics is not None:
        print(f"  HistGB:   acc={hgb_metrics['acc']:.4f}  AUC={hgb_metrics['auc']:.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
