"""v17 trainer v3: per-target recursive feature elimination.

Использует grid_research() из train_v17_v2.py (тот же 21 признак,
тот же набор configs), но дополнительно:

1. **Per-target feature importance report** (уже есть в v2)
2. **Greedy RFE** per target: начиная с full feature set, удаляем
   по одному самый слабый признак и пересчитываем honest 5-fold CV.
   Останавливаемся когда score перестаёт расти.
3. **Final config retrain** на pruned feature subset.

Output:
  ml_data/imports/v17_rfe_<target>.json  — список dropped features per target
  ml_data/imports/v17_grid_v3.json        — final grid (на pruned features)
  ml_data/models/<target>_v17/             — перезаписаны pruned версией

Usage:
  python scripts/train_v17_v3_rfe.py all
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v17_v2 import (
    load_matches, load_import, build_arrays,
    make_configs, _make_model, _evaluate, _encode_categorical,
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"
MODELS = PRO_ROOT / "ml_data" / "models"
GRID_OUT = IMPORTS / "v17_grid_v3.json"
RFE_OUT = IMPORTS / "v17_rfe_v3.json"

N_SPLITS = 5
RANDOM_STATE = 42
TARGETS = ("kills_total", "duration_sec", "first_15_kills", "winner")

# Множитель для greedy RFE early-stop: если новый score хуже предыдущего
# более чем на этот % — stop.
RFE_TOL = 0.005  # 0.5%


def _eval_best_for_target(target: str, X: np.ndarray, y: np.ndarray,
                           w: np.ndarray, configs: List) -> Tuple[str, float, str]:
    """Honest 5-fold CV; returns (best_config_name, best_score, metric)."""
    if target != "winner":
        mask = (y > 0) & np.isfinite(y)
    else:
        mask = np.ones(X.shape[0], dtype=bool)
    Xt, yt, wt = X[mask], y[mask], w[mask]
    if len(Xt) < 30:
        return "", float("inf"), "mae"
    kf = KFold(n_splits=min(N_SPLITS, max(2, len(Xt) // 30)),
               shuffle=True, random_state=RANDOM_STATE)
    best = (None, float("inf") if target != "winner" else 0.0)
    metric = "mae" if target != "winner" else "accuracy"
    for cfg in configs:
        fold_scores = []
        for tr, va in kf.split(Xt):
            model = _make_model(target, cfg)
            try:
                model.fit(Xt[tr], yt[tr], sample_weight=wt[tr])
            except TypeError:
                model.fit(Xt[tr], yt[tr])
            if target == "winner":
                try:
                    p = model.predict_proba(Xt[va])[:, 1]
                except Exception:
                    p = model.predict(Xt[va]).astype(float)
                fold_scores.append(_evaluate(target, yt[va], p, wt[va]))
            else:
                p = model.predict(Xt[va])
                fold_scores.append(_evaluate(target, yt[va], p, wt[va]))
        agg = {}
        for k in fold_scores[0].keys():
            agg[k] = float(np.mean([fs[k] for fs in fold_scores]))
        score = agg.get(metric, float("inf"))
        if target == "winner":
            if score > best[1]:
                best = (cfg.name, score)
        else:
            if score < best[1]:
                best = (cfg.name, score)
    return best[0], best[1], metric


def greedy_rfe(target: str, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               feature_names: List[str]) -> Dict[str, Any]:
    """Drop weakest feature one at a time. Stop when score stops improving
    (or only improves less than RFE_TOL)."""
    configs = make_configs(target)
    # Baseline
    base_name, base_score, metric = _eval_best_for_target(target, X, y, w, configs)
    keep = list(range(X.shape[1]))
    history = [{"step": 0, "n_features": len(keep),
                "best_config": base_name, "score": base_score, "kept": list(feature_names)}]
    print(f"  [{target}] step 0: {len(keep)} feats, best={base_name}, {metric}={base_score:.4f}",
          file=sys.stderr)
    while len(keep) > 3:  # минимум 3 фичи
        # Quick importance: fit Ridge/LogReg baseline и взять |coef|
        try:
            Xt_sub = X[:, keep]
            if target == "winner":
                m = LogisticRegression(C=0.5, penalty="l1", solver="liblinear", max_iter=2000)
                m.fit(Xt_sub, y, sample_weight=w)
                imp = np.abs(m.coef_[0])
            else:
                m = Ridge(alpha=1.0)
                m.fit(Xt_sub, y, sample_weight=w)
                imp = np.abs(m.coef_)
            # Normalize
            if imp.sum() > 0:
                imp = imp / imp.sum()
            # Find weakest
            weakest_local = int(np.argmin(imp))
        except Exception as exc:
            print(f"  [{target}] importance step failed: {exc}", file=sys.stderr)
            break
        # Try dropping it
        trial = list(keep)
        trial.pop(weakest_local)
        trial_name, trial_score, _ = _eval_best_for_target(
            target, X[:, trial], y, w, configs)
        # Decision: drop only if it strictly improves
        if target == "winner":
            improved = trial_score - base_score  # accuracy up = good
        else:
            improved = base_score - trial_score  # mae down = good
        if improved > 0:
            # Drop helps
            dropped = feature_names[keep[weakest_local]]
            keep = trial
            base_name, base_score = trial_name, trial_score
            history.append({"step": len(history), "n_features": len(keep),
                            "dropped": dropped, "best_config": base_name,
                            "score": base_score, "improved_by": float(improved)})
            print(f"  [{target}] step {len(history)-1}: drop {dropped}, "
                  f"{len(keep)} feats, {metric}={base_score:.4f} (improved by {improved:.4f})",
                  file=sys.stderr)
        else:
            # Stopping criterion: drop doesn't improve
            print(f"  [{target}] stop: dropping {feature_names[keep[weakest_local]]} "
                  f"would not improve {metric} (Δ={improved:.4f})", file=sys.stderr)
            break
    return {
        "target": target,
        "kept_features": [feature_names[i] for i in keep],
        "kept_idx": keep,
        "final_score": base_score,
        "final_config": base_name,
        "metric": metric,
        "history": history,
    }


def train_pruned(rfe_results: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Retrain best config on pruned features per target; save to *_v17/."""
    import joblib
    X_full = arrays["X"]
    w = arrays["w"]
    out: Dict[str, Any] = {}
    for target in TARGETS:
        rfe = rfe_results[target]
        idxs = rfe["kept_idx"]
        X = X_full[:, idxs]
        y = arrays[f"y_{target}"]
        if target != "winner":
            mask = (y > 0) & np.isfinite(y)
        else:
            mask = np.ones(X.shape[0], dtype=bool)
        Xt, yt, wt = X[mask], y[mask], w[mask]
        cfg_name = rfe["final_config"]
        cfg = next(c for c in make_configs(target) if c.name == cfg_name)
        model = _make_model(target, cfg)
        try:
            model.fit(Xt, yt, sample_weight=wt)
        except TypeError:
            model.fit(Xt, yt)
        MODELS.mkdir(parents=True, exist_ok=True)
        target_dir = MODELS / f"{target}_v17"
        target_dir.mkdir(exist_ok=True)
        joblib.dump(model, target_dir / "model.joblib")
        meta = {
            "target": target,
            "config": {"name": cfg_name, "model": cfg.model, "params": cfg.params},
            "n_matches": int(len(Xt)),
            "n_features": len(idxs),
            "feature_columns": [arrays["feature_names"][i] for i in idxs],
            "metric": rfe["metric"],
            "score_honest": float(rfe["final_score"]),
            "trained_at": int(time.time()),
            "trainer_version": "v3-rfe",
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        out[target] = {"dir": str(target_dir), "metric": rfe["metric"],
                       "score": float(rfe["final_score"]),
                       "n_features": len(idxs)}
        print(f"  saved {target_dir} ({len(idxs)} features, {rfe['metric']}={rfe['final_score']:.4f})",
              file=sys.stderr)
    return out


def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    print("[train_v17_v3] loading data...", file=sys.stderr)
    matches = load_matches()
    print(f"  {len(matches)} normalised matches", file=sys.stderr)
    top_teams  = load_import("v17_phase1_top_teams.json")
    hero_stats = load_import("v17_phase4_hero_stats.json")
    patch_info = load_import("v17_phase7_patch_info.json")
    team_players = load_import("v17_phase6_team_players.json")
    arrays = build_arrays(matches, top_teams=top_teams, hero_stats=hero_stats,
                          patch_info=patch_info, team_players=team_players)
    print(f"  X shape: {arrays['X'].shape}", file=sys.stderr)
    print(f"  features: {arrays['feature_names']}", file=sys.stderr)

    print("[train_v17_v3] running greedy RFE per target...", file=sys.stderr)
    rfe_results: Dict[str, Any] = {}
    for target in TARGETS:
        y = arrays[f"y_{target}"]
        rfe_results[target] = greedy_rfe(target, arrays["X"], y, arrays["w"],
                                          arrays["feature_names"])
    RFE_OUT.write_text(json.dumps(rfe_results, indent=2, default=str))
    print(f"[train_v17_v3] wrote {RFE_OUT}", file=sys.stderr)

    print("[train_v17_v3] training pruned production models...", file=sys.stderr)
    out = train_pruned(rfe_results, arrays)
    GRID_OUT.write_text(json.dumps({"rfe": rfe_results, "production": out},
                                    indent=2, default=str))
    print(f"[train_v17_v3] wrote {GRID_OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
