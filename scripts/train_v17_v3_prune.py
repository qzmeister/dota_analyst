"""v17 trainer v3: threshold-based feature pruning.

Подход: используем v17_feature_importance_v2.json (permutation importance)
и дропаем все фичи с importance < threshold. Затем пересчитываем
grid research на pruned feature set, retrain production models.

Output:
  ml_data/imports/v17_grid_v3.json
  ml_data/models/<target>_v17/ (overwrite pruned version)

Usage:
  python scripts/train_v17_v3_prune.py all
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v17_v2 import (
    load_matches, load_import, build_arrays,
    make_configs, _make_model, _evaluate,
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
TRAIN_OUT = IMPORTS / "v17_train_results_v3.json"

N_SPLITS = 5
RANDOM_STATE = 42
TARGETS = ("kills_total", "duration_sec", "first_15_kills", "winner")

# Pruning thresholds per target (importance ratio, sum-normalised)
PRUNE_TOL = {
    "kills_total":    0.01,   # drop features with importance < 1%
    "duration_sec":   0.01,
    "first_15_kills": 0.005,
    "winner":         0.005,  # gentler for winner — pruning kills accuracy
}
# Minimum number of features to keep per target
PRUNE_MIN_KEEP = {
    "kills_total":    10,
    "duration_sec":   10,
    "first_15_kills": 12,
    "winner":         14,  # winner needs most signal
}


def _eval_target(target: str, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    """Honest 5-fold CV across all configs for one target."""
    if target != "winner":
        mask = (y > 0) & np.isfinite(y)
    else:
        mask = np.ones(X.shape[0], dtype=bool)
    Xt, yt, wt = X[mask], y[mask], w[mask]
    if len(Xt) < 30:
        return {"skipped": True, "n_samples": int(len(Xt))}
    kf = KFold(n_splits=min(N_SPLITS, max(2, len(Xt) // 30)),
               shuffle=True, random_state=RANDOM_STATE)
    per_cfg: Dict[str, Any] = {}
    for cfg in make_configs(target):
        fold_scores: List[Dict[str, float]] = []
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
        agg: Dict[str, float] = {}
        for k in fold_scores[0].keys():
            agg[k] = float(np.mean([fs[k] for fs in fold_scores]))
        per_cfg[cfg.name] = {"params": cfg.params, "model": cfg.model, **agg}
    if target == "winner":
        best = max(per_cfg.items(), key=lambda kv: kv[1].get("accuracy", 0))
        sort_key, best_score = "accuracy", best[1]["accuracy"]
    else:
        best = min(per_cfg.items(), key=lambda kv: kv[1].get("mae", 1e9))
        sort_key, best_score = "mae", best[1]["mae"]
    return {
        "configs": per_cfg,
        "best_config": best[0],
        f"best_{sort_key}": float(best_score),
        "n_samples": int(len(Xt)),
    }


def select_features(target: str, feature_names: List[str],
                    importance: List[Dict[str, float]], tol: float,
                    min_keep: int = 5) -> List[int]:
    """Pick features with importance > tol * max.  Always keep at least
    `min_keep` top-importance features."""
    by_name = {row["feature"]: row["score"] for row in importance}
    if not by_name:
        return list(range(len(feature_names)))
    max_score = max(by_name.values()) or 1.0
    keep = [i for i, n in enumerate(feature_names) if by_name.get(n, 0) >= tol * max_score]
    if len(keep) < min_keep:
        # Fallback: keep top-min_keep by importance
        order = sorted(range(len(feature_names)),
                       key=lambda i: -by_name.get(feature_names[i], 0))
        keep = order[:min_keep]
    return sorted(keep)


def train_pruned(arrays: Dict[str, np.ndarray], grid: Dict[str, Any]) -> Dict[str, Any]:
    import joblib
    X = arrays["X"]
    w = arrays["w"]
    out: Dict[str, Any] = {}
    for target in TARGETS:
        if target not in grid["results"]:
            continue
        cfg_name = grid["results"][target]["best_config"]
        cfg = next(c for c in make_configs(target) if c.name == cfg_name)
        idxs = grid["results"][target]["kept_idx"]
        kept_names = [arrays["feature_names"][i] for i in idxs]
        y = arrays[f"y_{target}"]
        if target != "winner":
            mask = (y > 0) & np.isfinite(y)
        else:
            mask = np.ones(X.shape[0], dtype=bool)
        Xt, yt, wt = X[mask][:, idxs], y[mask], w[mask]
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
            "feature_columns": kept_names,
            "metrics_honest": {k: v for k, v in grid["results"][target]["configs"][cfg_name].items()
                                  if k not in ("params", "model")},
            "trained_at": int(time.time()),
            "trainer_version": "v3-prune",
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        out[target] = {"dir": str(target_dir),
                       "n_features": len(idxs),
                       "kept_features": kept_names,
                       "metrics": meta["metrics_honest"]}
        print(f"  saved {target_dir} ({len(idxs)} features: {kept_names})", file=sys.stderr)
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

    # Load v2 importance report
    imp_path = IMPORTS / "v17_feature_importance_v2.json"
    if not imp_path.exists():
        print(f"  missing {imp_path}; run v2 first", file=sys.stderr)
        return 2
    imp = json.loads(imp_path.read_text(encoding="utf-8"))

    grid: Dict[str, Any] = {
        "n_matches": int(arrays["X"].shape[0]),
        "n_features_full": int(arrays["X"].shape[1]),
        "feature_names": list(arrays["feature_names"]),
        "results": {},
    }

    for target in TARGETS:
        if target not in imp["results"]:
            continue
        # Select features
        tol = PRUNE_TOL[target]
        min_keep = PRUNE_MIN_KEEP[target]
        idxs = select_features(target, arrays["feature_names"],
                                imp["results"][target]["importance"], tol, min_keep)
        kept_names = [arrays["feature_names"][i] for i in idxs]
        # Grid on pruned
        y = arrays[f"y_{target}"]
        X_pruned = arrays["X"][:, idxs]
        result = _eval_target(target, X_pruned, y, arrays["w"])
        if "skipped" in result:
            print(f"  [{target}] SKIP: {result}", file=sys.stderr)
            continue
        result["kept_features"] = kept_names
        result["kept_idx"] = idxs
        result["n_features_pruned"] = len(idxs)
        grid["results"][target] = result
        print(f"  [{target}] pruned {len(arrays['feature_names'])} -> {len(idxs)}: {kept_names}",
              file=sys.stderr)
        print(f"           best={result['best_config']}, "
              f"{'accuracy' if target == 'winner' else 'mae'}="
              f"{result.get('best_accuracy', result.get('best_mae')):.4f}",
              file=sys.stderr)

    GRID_OUT.write_text(json.dumps(grid, indent=2, default=str))
    print(f"[train_v17_v3] wrote {GRID_OUT}", file=sys.stderr)

    print("[train_v17_v3] training pruned production models...", file=sys.stderr)
    out = train_pruned(arrays, grid)
    TRAIN_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"[train_v17_v3] wrote {TRAIN_OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
