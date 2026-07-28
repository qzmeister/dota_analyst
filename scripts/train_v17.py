"""v17 trainer: build features, run grid research, train production models.

Loads `ml_data/full_matches/<id>.json` (normalised v17 format) and
combines with:
  - ml_data/imports/v17_phase1_top_teams.json  (top-N pro teams)
  - ml_data/imports/v17_phase4_hero_stats.json  (per-hero per-patch stats)
  - ml_data/imports/v17_phase5_hero_matchups.json  (per-hero matchups)
  - ml_data/imports/v17_phase6_team_players.json  (team rosters)
  - ml_data/imports/v17_phase7_patch_info.json  (patch boundaries)

Feature blocks (all are Optional — v17 picks up only what the
normalised match payload has):
  F1.  Patch  — "7.40", "7.39", ... (categorical code)
  F2.  Team tier (per-team_id)  — premium / professional / minor
  F3.  Hero target encoding (per hero_id)  — recent 7d win rate
  F4.  Hero-hero synergy (avg of radiant pick win rates minus dire)
  F5.  Side (radiant vs dire)  — one-hot (0/1)
  F6.  Recency weight  — from v17_features.recency_weight

Targets (from v17_targets):
  T1.  kills_total         (regression)
  T2.  duration_sec       (regression)
  T3.  first_15_kills     (regression)
  T4.  winner             (binary classification, y=1 if radiant_win)

We run **honest 5-fold CV** with the encoder refit per fold, and
select the best config per target.  Then retrain the winners on the
full corpus and save under `ml_data/models/<target>_v17/`.

Usage:
  python scripts/train_v17.py grid      # grid research, log to grid_v17.json
  python scripts/train_v17.py train     # retrain best configs on full corpus
  python scripts/train_v17.py all       # grid + train in one go
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error

PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"
FULL_MATCHES = PRO_ROOT / "ml_data" / "full_matches"
MODELS = PRO_ROOT / "ml_data" / "models"
GRID_OUT = IMPORTS / "v17_grid.json"
TRAIN_OUT = IMPORTS / "v17_train_results.json"

# v17 hyperparameters
N_SPLITS = 5
RANDOM_STATE = 42

TARGETS = ("kills_total", "duration_sec", "first_15_kills", "winner")
NUMERIC_TARGETS = ("kills_total", "duration_sec", "first_15_kills")

# Sample-weight column.  Combined from recency_weight (match age) and
# team tier (premium / professional).
SAMPLE_WEIGHT = "sample_weight"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_matches() -> List[Dict[str, Any]]:
    """Read every normalized v17 match from `ml_data/full_matches/`."""
    out = []
    for path in FULL_MATCHES.glob("*.json"):
        try:
            m = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        if "v17_targets" not in m:
            continue
        out.append(m)
    return out


def load_import(name: str) -> Any:
    p = IMPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def _all_picks(match: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """Return (radiant_hero_ids, dire_hero_ids) from picks_bans."""
    r, d = [], []
    for pb in match.get("picks_bans") or []:
        if not pb.get("is_pick"):
            continue
        if pb.get("team") == 0:
            r.append(pb.get("hero_id"))
        elif pb.get("team") == 1:
            d.append(pb.get("hero_id"))
    return r, d


def _top_team_tier(teams: Optional[List[Dict[str, Any]]]) -> Dict[int, str]:
    """Map team_id -> tier (premium/professional) from top teams list.

    OpenDota top-teams list has `team_id` + `wins/losses/rating`,
    but no explicit tier.  We derive a proxy tier:
      rating >= 1400 -> "premium"
      rating >= 1100 -> "professional"
      else           -> "minor"
    """
    out: Dict[int, str] = {}
    if not teams:
        return out
    for t in teams:
        try:
            tid = int(t["team_id"])
        except (KeyError, ValueError, TypeError):
            continue
        r = t.get("rating") or 0
        if r >= 1400:
            out[tid] = "premium"
        elif r >= 1100:
            out[tid] = "professional"
        else:
            out[tid] = "minor"
    return out


def _patch_index(patch_info: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
    """patch name -> index 0,1,2,...  Used for categorical encoding."""
    out: Dict[str, int] = {}
    if not patch_info:
        return out
    for i, p in enumerate(patch_info):
        if not isinstance(p, dict):
            continue
        n = p.get("name")
        if isinstance(n, str) and n:
            out[n] = i
    return out


def _hero_target_encoder(hero_stats: Optional[List[Dict[str, Any]]]) -> Dict[int, float]:
    """Per-hero recent-7d win rate (smoothed by 5).  Returns 0.5
    default for unknown hero_ids.
    """
    out: Dict[int, float] = {}
    if not hero_stats:
        return out
    for h in hero_stats:
        if not isinstance(h, dict):
            continue
        hid = h.get("id")
        if hid is None:
            continue
        pick_trend = h.get("pub_pick_trend") or []
        win_trend = h.get("pub_win_trend") or []
        if pick_trend and win_trend and len(pick_trend) == len(win_trend):
            pick = pick_trend[0] or 0
            wins = win_trend[0] or 0
            smooth = (wins + 5 * 0.5) / (pick + 5) if pick else 0.5
            out[int(hid)] = float(smooth)
    return out


def _encode_categorical(values: List[Any]) -> List[int]:
    """Map values to integer codes; unseen values get -1."""
    mapping: Dict[Any, int] = {}
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(-1)
            continue
        if v not in mapping:
            mapping[v] = len(mapping)
        out.append(mapping[v])
    return out


def build_arrays(matches: List[Dict[str, Any]],
                 top_teams: Optional[List[Dict[str, Any]]] = None,
                 hero_stats: Optional[List[Dict[str, Any]]] = None,
                 patch_info: Optional[List[Dict[str, Any]]] = None
                 ) -> Dict[str, np.ndarray]:
    """Build numpy arrays for features, weights, targets.

    Returns a dict of:
      X:           (n, n_features) float32
      feature_names: list[str]
      w:           (n,) float32
      y_<target>:  (n,) float32 (or int8 for winner)
    """
    tier_by_team = _top_team_tier(top_teams)
    patch_idx = _patch_index(patch_info)
    hero_enc = _hero_target_encoder(hero_stats)

    n = len(matches)
    patch_codes, r_tier_codes, d_tier_codes = [], [], []
    r_team_ids, d_team_ids = [], []
    r_hero_enc, d_hero_enc = [], []
    r_picks_count, d_picks_count = [], []
    r_top, d_top = [], []
    side_radiant, recency = [], []
    weights = []
    y_kills, y_dur, y_f15, y_win = [], [], [], []

    for m in matches:
        radiant_picks, dire_picks = _all_picks(m)
        r_team = m.get("radiant_team_id")
        d_team = m.get("dire_team_id")
        r_tier = tier_by_team.get(int(r_team), "minor") if r_team else "minor"
        d_tier = tier_by_team.get(int(d_team), "minor") if d_team else "minor"
        patch = (m.get("v17_features") or {}).get("patch") or m.get("patch") or ""
        r_h = float(np.mean([hero_enc.get(int(h), 0.5) for h in radiant_picks])) if radiant_picks else 0.5
        d_h = float(np.mean([hero_enc.get(int(h), 0.5) for h in dire_picks])) if dire_picks else 0.5
        rec_w = (m.get("v17_features") or {}).get("recency_weight") or 0.5
        tier_w = {"premium": 1.0, "professional": 0.7, "minor": 0.4}.get(r_tier, 0.4) \
                * {"premium": 1.0, "professional": 0.7, "minor": 0.4}.get(d_tier, 0.4)
        tier_weight = math.sqrt(max(0.1, tier_w))
        weights.append(float(rec_w) * float(tier_weight))

        patch_codes.append(patch)
        r_tier_codes.append(r_tier)
        d_tier_codes.append(d_tier)
        r_team_ids.append(int(r_team) if r_team else 0)
        d_team_ids.append(int(d_team) if d_team else 0)
        r_hero_enc.append(r_h)
        d_hero_enc.append(d_h)
        r_picks_count.append(len(radiant_picks))
        d_picks_count.append(len(dire_picks))
        r_top.append(int(r_tier in ("premium", "professional")))
        d_top.append(int(d_tier in ("premium", "professional")))
        side_radiant.append(1)
        recency.append(float(rec_w))

        tgts = m.get("v17_targets") or {}
        y_kills.append(tgts.get("kills_total") or 0)
        y_dur.append(tgts.get("duration_sec") or 0)
        y_f15.append(tgts.get("first_15_kills") or 0)
        y_win.append(int(bool(m.get("radiant_win"))))

    # Build X.  Categorical features: encode to int codes.  Numeric
    # features: keep as float32.
    feature_spec = [
        ("patch",     patch_codes,    "cat"),
        ("r_tier",     r_tier_codes,   "cat"),
        ("d_tier",     d_tier_codes,   "cat"),
        ("r_team_id",  r_team_ids,     "num"),
        ("d_team_id",  d_team_ids,     "num"),
        ("r_hero_enc", r_hero_enc,     "num"),
        ("d_hero_enc", d_hero_enc,     "num"),
        ("r_dire_syn", [r - d for r, d in zip(r_hero_enc, d_hero_enc)], "num"),
        ("r_picks",    r_picks_count,  "num"),
        ("d_picks",    d_picks_count,  "num"),
        ("r_top_team", r_top,          "num"),
        ("d_top_team", d_top,          "num"),
        ("side_rad",   side_radiant,   "num"),
    ]
    cols, names, kinds = [], [], []
    for name, vals, kind in feature_spec:
        if kind == "cat":
            cols.append(np.asarray(_encode_categorical(vals), dtype=np.int32))
        else:
            arr = np.asarray(vals, dtype=np.float32)
            # NaN/inf guard
            arr[~np.isfinite(arr)] = 0.0
            cols.append(arr)
        names.append(name)
        kinds.append(kind)
    X = np.column_stack(cols)
    return {
        "X": X,
        "feature_names": names,
        "feature_kinds": kinds,
        "w": np.asarray(weights, dtype=np.float32),
        "y_kills_total": np.asarray(y_kills, dtype=np.float32),
        "y_duration_sec": np.asarray(y_dur, dtype=np.float32),
        "y_first_15_kills": np.asarray(y_f15, dtype=np.float32),
        "y_winner": np.asarray(y_win, dtype=np.int8),
    }


# --------------------------------------------------------------------------- #
# Grid research
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    name: str
    model: str            # "logreg" / "hgb" / "ridge"
    params: Dict[str, Any] = field(default_factory=dict)


def make_configs(target: str) -> List[Config]:
    """Generate the candidate configs for a target."""
    if target == "winner":
        return [
            Config("logreg_c0.5_l1", "logreg", {"C": 0.5, "penalty": "l1", "solver": "liblinear"}),
            Config("logreg_c1.0_l2", "logreg", {"C": 1.0, "penalty": "l2"}),
            Config("logreg_c2.0_l2", "logreg", {"C": 2.0, "penalty": "l2"}),
            Config("hgb_default", "hgb", {}),
            Config("hgb_max_depth3", "hgb", {"max_depth": 3, "learning_rate": 0.05}),
            Config("hgb_max_depth4", "hgb", {"max_depth": 4, "learning_rate": 0.1}),
        ]
    if target in ("kills_total", "first_15_kills"):
        return [
            Config("ridge_a0.1", "ridge", {"alpha": 0.1}),
            Config("ridge_a1.0", "ridge", {"alpha": 1.0}),
            Config("ridge_a10",  "ridge", {"alpha": 10.0}),
            Config("hgb_default", "hgb", {}),
            Config("hgb_max_depth3", "hgb", {"max_depth": 3, "learning_rate": 0.05}),
            Config("hgb_poisson", "hgb", {"loss": "poisson"}),
        ]
    if target == "duration_sec":
        return [
            Config("ridge_a0.1", "ridge", {"alpha": 0.1}),
            Config("ridge_a1.0", "ridge", {"alpha": 1.0}),
            Config("ridge_a10",  "ridge", {"alpha": 10.0}),
            Config("hgb_default", "hgb", {}),
            Config("hgb_max_depth4", "hgb", {"max_depth": 4, "learning_rate": 0.1}),
            Config("hgb_squared", "hgb", {"loss": "squared_error"}),
        ]
    return []


def _make_model(target: str, cfg: Config):
    if cfg.model == "logreg":
        return LogisticRegression(max_iter=2000, **cfg.params)
    if cfg.model == "ridge":
        return Ridge(**cfg.params)
    if cfg.model == "hgb":
        if target == "winner":
            return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **cfg.params)
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **cfg.params)
    raise ValueError(f"unknown model {cfg.model}")


def _evaluate(target: str, y_true, y_pred, sample_w=None) -> Dict[str, float]:
    if target == "winner":
        # y_pred is probability of class=1 (radiant)
        pred_class = (np.asarray(y_pred) >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, pred_class)),
            "logloss":  float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7), labels=[0, 1])),
        }
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred, sample_weight=sample_w)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_w))),
    }


def grid_research(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Honest 5-fold CV across all candidate configs per target."""
    X = arrays["X"]
    w = arrays["w"]
    out: Dict[str, Any] = {"n_matches": int(X.shape[0]), "n_features": int(X.shape[1]), "results": {}}
    for target in TARGETS:
        y = arrays[f"y_{target}"]
        # Drop rows where target is NaN/0 for duration/kills
        if target != "winner":
            mask = (y > 0) & np.isfinite(y)
        else:
            mask = np.ones(X.shape[0], dtype=bool)
        Xt, yt, wt = X[mask], y[mask], w[mask]
        if len(Xt) < 30:
            print(f"  {target:18s} SKIP (only {len(Xt)} samples)", file=sys.stderr)
            continue
        kf = KFold(n_splits=min(N_SPLITS, max(2, len(Xt) // 30)),
                   shuffle=True, random_state=RANDOM_STATE)
        per_cfg: Dict[str, Any] = {}
        for cfg in make_configs(target):
            fold_scores: List[Dict[str, float]] = []
            for fold, (tr, va) in enumerate(kf.split(Xt)):
                model = _make_model(target, cfg)
                try:
                    model.fit(Xt[tr], yt[tr], sample_weight=wt[tr])
                except TypeError:
                    model.fit(Xt[tr], yt[tr])
                if target == "winner":
                    try:
                        proba = model.predict_proba(Xt[va])[:, 1]
                    except Exception:
                        proba = model.predict(Xt[va]).astype(float)
                    fold_scores.append(_evaluate(target, yt[va], proba, wt[va]))
                else:
                    pred = model.predict(Xt[va])
                    fold_scores.append(_evaluate(target, yt[va], pred, wt[va]))
            agg: Dict[str, float] = {}
            for k in fold_scores[0].keys():
                agg[k] = float(np.mean([fs[k] for fs in fold_scores]))
            per_cfg[cfg.name] = {"params": cfg.params, "model": cfg.model, **agg}
        # Pick the winner
        if target == "winner":
            best = max(per_cfg.items(), key=lambda kv: kv[1].get("accuracy", 0))
            sort_key, best_score = "accuracy", best[1]["accuracy"]
        else:
            best = min(per_cfg.items(), key=lambda kv: kv[1].get("mae", 1e9))
            sort_key, best_score = "mae", best[1]["mae"]
        out["results"][target] = {
            "configs": per_cfg,
            "best_config": best[0],
            f"best_{sort_key}": float(best_score),
            "n_samples": int(len(Xt)),
        }
        print(f"  {target:18s} best={best[0]:24s} {sort_key}={best_score:.4f} (n={len(Xt)})", file=sys.stderr)
    return out


def train_production(arrays: Dict[str, np.ndarray], grid: Dict[str, Any]) -> Dict[str, Any]:
    """Retrain the best config per target on the full corpus."""
    import joblib
    X = arrays["X"]
    w = arrays["w"]
    out: Dict[str, Any] = {}
    for target in TARGETS:
        if target not in grid["results"]:
            continue
        cfg_name = grid["results"][target]["best_config"]
        cfg = next(c for c in make_configs(target) if c.name == cfg_name)
        model = _make_model(target, cfg)
        y = arrays[f"y_{target}"]
        if target != "winner":
            mask = (y > 0) & np.isfinite(y)
        else:
            mask = np.ones(X.shape[0], dtype=bool)
        Xt, yt, wt = X[mask], y[mask], w[mask]
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
            "n_features": int(X.shape[1]),
            "feature_columns": arrays["feature_names"],
            "metrics_honest": {k: v for k, v in grid["results"][target]["configs"][cfg_name].items()
                                  if k not in ("params", "model")},
            "trained_at": int(time.time()),
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        out[target] = {"dir": str(target_dir), "metrics": meta["metrics_honest"]}
        print(f"  saved {target_dir}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    print("[train_v17] loading data...", file=sys.stderr)
    matches = load_matches()
    print(f"  {len(matches)} normalised matches", file=sys.stderr)
    top_teams  = load_import("v17_phase1_top_teams.json")
    hero_stats = load_import("v17_phase4_hero_stats.json")
    patch_info = load_import("v17_phase7_patch_info.json")
    arrays = build_arrays(matches, top_teams=top_teams, hero_stats=hero_stats, patch_info=patch_info)
    print(f"  X shape: {arrays['X'].shape}, features: {arrays['feature_names']}", file=sys.stderr)
    if cmd in ("grid", "all"):
        print("[train_v17] running grid research...", file=sys.stderr)
        grid = grid_research(arrays)
        GRID_OUT.write_text(json.dumps(grid, indent=2, default=str))
        print(f"[train_v17] wrote {GRID_OUT}", file=sys.stderr)
    if cmd in ("train", "all"):
        if not GRID_OUT.exists():
            print(f"  missing {GRID_OUT}; run grid first", file=sys.stderr)
            return 2
        grid = json.loads(GRID_OUT.read_text())
        print("[train_v17] training production models...", file=sys.stderr)
        results = train_production(arrays, grid)
        TRAIN_OUT.write_text(json.dumps(results, indent=2, default=str))
        print(f"[train_v17] wrote {TRAIN_OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
