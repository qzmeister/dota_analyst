"""v17 trainer v2: расширенный feature set + XGBoost + feature importance.

Улучшения v2:
  F1. patch (cat)             — 7.39/7.40/7.41
  F2. r_tier, d_tier (cat)    — premium/professional/minor
  F3. r_team_id, d_team_id    — pro team id (numeric embedding)
  F4. r_hero_enc, d_hero_enc  — avg per-team hero target encoding (recency-weighted winrate)
  F5. r_dire_syn              — radiant mean hero enc - dire mean hero enc
  F6. r_picks, d_picks        — number of picks
  F7. r_top_team, d_top_team  — 0/1 flag for premium/professional tier
  F8. side_rad                — 1 (always radiant for one-hot)
  ==== NEW v2 ====
  F9.  r_ban_enc, d_ban_enc   — avg target encoding забаненных героев
  F10. r_team_synergy         — stddev of hero enc (сигнатура-команда)
  F11. d_team_synergy         — same for dire
  F12. gold_adv_5min          — radiant_gold_adv[5]
  F13. gold_adv_10min         — radiant_gold_adv[10]
  F14. days_since_patch       — days from patch release to match start_time
  F15. n_player_accounts_top30 — сколько игроков в ростере топ-30 (из phase 6)

Models:
  - ridge_a0.1, a1.0, a10
  - hgb_default, max_depth3, poisson
  - logreg_c0.5_l1, c1.0_l2, c2.0_l2
  - xgboost_dart (gbtree), xgboost_poisson, xgboost_quantile_alpha0.5

Outputs:
  ml_data/imports/v17_grid_v2.json
  ml_data/models/<target>_v17/ (overwrites v17)
  ml_data/imports/v17_feature_importance.json

Usage:
  python scripts/train_v17_v2.py all
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

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

PRO_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = PRO_ROOT / "ml_data" / "imports"
FULL_MATCHES = PRO_ROOT / "ml_data" / "full_matches"
MODELS = PRO_ROOT / "ml_data" / "models"
GRID_OUT = IMPORTS / "v17_grid_v2.json"
TRAIN_OUT = IMPORTS / "v17_train_results_v2.json"
IMP_OUT = IMPORTS / "v17_feature_importance_v2.json"

N_SPLITS = 5
RANDOM_STATE = 42

TARGETS = ("kills_total", "duration_sec", "first_15_kills", "winner")
NUMERIC_TARGETS = ("kills_total", "duration_sec", "first_15_kills")

# Tier weights (user spec: выше уровень команды — больше влияние)
TIER_WEIGHT = {"premium": 1.0, "professional": 0.7, "minor": 0.4}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_matches() -> List[Dict[str, Any]]:
    out = []
    for path in FULL_MATCHES.glob("*.json"):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
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
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Feature engineering (v2)
# --------------------------------------------------------------------------- #

def _all_picks(match: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    r, d = [], []
    for pb in match.get("picks_bans") or []:
        if not pb.get("is_pick"):
            continue
        if pb.get("team") == 0:
            r.append(pb.get("hero_id"))
        elif pb.get("team") == 1:
            d.append(pb.get("hero_id"))
    return r, d


def _all_bans(match: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    r, d = [], []
    for pb in match.get("picks_bans") or []:
        if pb.get("is_pick"):
            continue
        if pb.get("team") == 0:
            r.append(pb.get("hero_id"))
        elif pb.get("team") == 1:
            d.append(pb.get("hero_id"))
    return r, d


def _top_team_tier(teams: Optional[List[Dict[str, Any]]]) -> Dict[int, str]:
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
    out: Dict[str, str] = {}
    if not patch_info:
        return out
    for p in patch_info:
        if not isinstance(p, dict):
            continue
        n = p.get("name")
        d = p.get("date") or p.get("start_date")
        if isinstance(n, str) and n:
            out[n] = d
    return out


def _hero_target_encoder(hero_stats: Optional[List[Dict[str, Any]]]) -> Dict[int, float]:
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


def _days_since_patch(patch_name: str, start_time: int,
                       patch_dates: Dict[str, str]) -> float:
    if not patch_name or patch_name not in patch_dates:
        return 90.0  # conservative default
    try:
        date_str = patch_dates[patch_name]
        if not date_str:
            return 90.0
        # OpenDota format: 2026-03-24T00:50:59.580Z
        ts = int(time.mktime(time.strptime(date_str[:10], "%Y-%m-%d")))
        return max(0.0, (start_time - ts) / 86400.0)
    except Exception:
        return 90.0


def _gold_advantage_at_minute(adv: List, minute: int) -> float:
    """Sample radiant_gold_adv at `minute` minutes into the match.
    OpenDota returns 1 sample per minute (0..duration_minutes).
    """
    if not adv:
        return 0.0
    if minute < len(adv):
        v = adv[minute]
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    # out-of-range -> last value
    if adv:
        v = adv[-1]
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return 0.0


def build_arrays(matches: List[Dict[str, Any]],
                 top_teams: Optional[List[Dict[str, Any]]] = None,
                 hero_stats: Optional[List[Dict[str, Any]]] = None,
                 patch_info: Optional[List[Dict[str, Any]]] = None,
                 team_players: Optional[Dict[int, Any]] = None
                 ) -> Dict[str, np.ndarray]:
    tier_by_team = _top_team_tier(top_teams)
    patch_dates = _patch_index(patch_info)
    hero_enc = _hero_target_encoder(hero_stats)

    # Build top-30 player set
    top_players: set = set()
    if team_players:
        for tid, roster in team_players.items():
            if int(tid) not in tier_by_team:
                continue
            for p in (roster or []):
                acc = p.get("account_id")
                if acc:
                    top_players.add(int(acc))

    n = len(matches)
    patch_codes, r_tier_codes, d_tier_codes = [], [], []
    r_team_ids, d_team_ids = [], []
    r_hero_enc, d_hero_enc = [], []
    r_dire_syn = []
    r_picks_count, d_picks_count = [], []
    r_top, d_top = [], []
    side_radiant, recency = [], []
    # v2 new
    r_ban_enc, d_ban_enc = [], []
    r_team_syn, d_team_syn = [], []
    gold_adv_5, gold_adv_10 = [], []
    days_since_patch_l = []
    n_top_players_l = []
    weights = []
    y_kills, y_dur, y_f15, y_win = [], [], [], []

    for m in matches:
        radiant_picks, dire_picks = _all_picks(m)
        radiant_bans, dire_bans = _all_bans(m)
        r_team = m.get("radiant_team_id")
        d_team = m.get("dire_team_id")
        r_tier = tier_by_team.get(int(r_team), "minor") if r_team else "minor"
        d_tier = tier_by_team.get(int(d_team), "minor") if d_team else "minor"
        patch = (m.get("v17_features") or {}).get("patch") or m.get("patch") or ""
        r_h = float(np.mean([hero_enc.get(int(h), 0.5) for h in radiant_picks])) if radiant_picks else 0.5
        d_h = float(np.mean([hero_enc.get(int(h), 0.5) for h in dire_picks])) if dire_picks else 0.5
        r_b = float(np.mean([hero_enc.get(int(h), 0.5) for h in radiant_bans])) if radiant_bans else 0.5
        d_b = float(np.mean([hero_enc.get(int(h), 0.5) for h in dire_bans])) if dire_bans else 0.5
        r_std = float(np.std([hero_enc.get(int(h), 0.5) for h in radiant_picks])) if radiant_picks else 0.0
        d_std = float(np.std([hero_enc.get(int(h), 0.5) for h in dire_picks])) if dire_picks else 0.0
        rec_w = (m.get("v17_features") or {}).get("recency_weight") or 0.5
        tier_w = TIER_WEIGHT.get(r_tier, 0.4) * TIER_WEIGHT.get(d_tier, 0.4)
        tier_weight = math.sqrt(max(0.1, tier_w))
        weights.append(float(rec_w) * float(tier_weight))

        # NEW v2
        start_time = int(m.get("start_time") or 0)
        gold_adv_5_l = _gold_advantage_at_minute(m.get("radiant_gold_adv") or [], 5)
        gold_adv_10_l = _gold_advantage_at_minute(m.get("radiant_gold_adv") or [], 10)
        days_p = _days_since_patch(patch, start_time, patch_dates)
        n_top_players = 0
        for p in (m.get("players") or []):
            acc = p.get("account_id")
            if acc and int(acc) in top_players:
                n_top_players += 1

        patch_codes.append(patch)
        r_tier_codes.append(r_tier)
        d_tier_codes.append(d_tier)
        r_team_ids.append(int(r_team) if r_team else 0)
        d_team_ids.append(int(d_team) if d_team else 0)
        r_hero_enc.append(r_h)
        d_hero_enc.append(d_h)
        r_dire_syn.append(r_h - d_h)
        r_picks_count.append(len(radiant_picks))
        d_picks_count.append(len(dire_picks))
        r_top.append(int(r_tier in ("premium", "professional")))
        d_top.append(int(d_tier in ("premium", "professional")))
        side_radiant.append(1)
        recency.append(float(rec_w))
        # v2
        r_ban_enc.append(r_b)
        d_ban_enc.append(d_b)
        r_team_syn.append(r_std)
        d_team_syn.append(d_std)
        gold_adv_5.append(gold_adv_5_l)
        gold_adv_10.append(gold_adv_10_l)
        days_since_patch_l.append(days_p)
        n_top_players_l.append(n_top_players)

        tgts = m.get("v17_targets") or {}
        y_kills.append(tgts.get("kills_total") or 0)
        y_dur.append(tgts.get("duration_sec") or 0)
        y_f15.append(tgts.get("first_15_kills") or 0)
        y_win.append(int(bool(m.get("radiant_win"))))

    feature_spec = [
        ("patch",            patch_codes,        "cat"),
        ("r_tier",            r_tier_codes,        "cat"),
        ("d_tier",            d_tier_codes,        "cat"),
        ("r_team_id",         r_team_ids,         "num"),
        ("d_team_id",         d_team_ids,         "num"),
        ("r_hero_enc",        r_hero_enc,         "num"),
        ("d_hero_enc",        d_hero_enc,         "num"),
        ("r_dire_syn",        r_dire_syn,         "num"),
        ("r_picks",           r_picks_count,      "num"),
        ("d_picks",           d_picks_count,      "num"),
        ("r_top_team",        r_top,              "num"),
        ("d_top_team",        d_top,              "num"),
        ("side_rad",          side_radiant,       "num"),
        # NEW v2
        ("r_ban_enc",         r_ban_enc,          "num"),
        ("d_ban_enc",         d_ban_enc,          "num"),
        ("r_team_syn",        r_team_syn,         "num"),
        ("d_team_syn",        d_team_syn,         "num"),
        ("gold_adv_5",        gold_adv_5,         "num"),
        ("gold_adv_10",       gold_adv_10,        "num"),
        ("days_since_patch",  days_since_patch_l, "num"),
        ("n_top_players",     n_top_players_l,    "num"),
    ]
    cols, names, kinds = [], [], []
    for name, vals, kind in feature_spec:
        if kind == "cat":
            cols.append(np.asarray(_encode_categorical(vals), dtype=np.int32))
        else:
            arr = np.asarray(vals, dtype=np.float32)
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
# Configs v2
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    name: str
    model: str            # "logreg" / "hgb" / "ridge" / "xgb"
    params: Dict[str, Any] = field(default_factory=dict)


def make_configs(target: str) -> List[Config]:
    cfgs: List[Config] = []
    if target == "winner":
        cfgs += [
            Config("logreg_c0.5_l1", "logreg", {"C": 0.5, "penalty": "l1", "solver": "liblinear"}),
            Config("logreg_c1.0_l2", "logreg", {"C": 1.0, "penalty": "l2"}),
            Config("logreg_c2.0_l2", "logreg", {"C": 2.0, "penalty": "l2"}),
            Config("hgb_default",     "hgb", {}),
            Config("hgb_max_depth3",  "hgb", {"max_depth": 3, "learning_rate": 0.05}),
            Config("hgb_max_depth4",  "hgb", {"max_depth": 4, "learning_rate": 0.1}),
        ]
        if XGB_AVAILABLE:
            cfgs += [
                Config("xgb_dart",  "xgb_clf", {"booster": "dart",  "max_depth": 4, "eta": 0.1, "n_estimators": 200, "rate_drop": 0.1}),
                Config("xgb_gbtree","xgb_clf", {"booster": "gbtree","max_depth": 4, "eta": 0.1, "n_estimators": 200}),
            ]
    else:  # regression
        cfgs += [
            Config("ridge_a0.1", "ridge", {"alpha": 0.1}),
            Config("ridge_a1.0", "ridge", {"alpha": 1.0}),
            Config("ridge_a10",  "ridge", {"alpha": 10.0}),
            Config("hgb_default",    "hgb", {}),
            Config("hgb_max_depth3", "hgb", {"max_depth": 3, "learning_rate": 0.05}),
            Config("hgb_poisson",    "hgb", {"loss": "poisson"}),
        ]
        if XGB_AVAILABLE:
            if target == "duration_sec":
                cfgs += [
                    Config("xgb_reg_sq", "xgb_reg", {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "n_estimators": 200}),
                    Config("xgb_reg_quantile_p50", "xgb_reg", {"objective": "reg:quantileerror", "quantile_alpha": 0.5, "max_depth": 4, "eta": 0.1, "n_estimators": 200}),
                ]
            elif target == "kills_total":
                cfgs += [
                    Config("xgb_reg_sq",     "xgb_reg", {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "n_estimators": 200}),
                    Config("xgb_reg_poisson", "xgb_reg", {"objective": "count:poisson", "max_depth": 4, "eta": 0.1, "n_estimators": 200}),
                ]
            elif target == "first_15_kills":
                cfgs += [
                    Config("xgb_reg_sq",      "xgb_reg", {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "n_estimators": 200}),
                    Config("xgb_reg_poisson", "xgb_reg", {"objective": "count:poisson", "max_depth": 4, "eta": 0.1, "n_estimators": 200}),
                ]
    return cfgs


def _make_model(target: str, cfg: Config):
    if cfg.model == "logreg":
        return LogisticRegression(max_iter=2000, **cfg.params)
    if cfg.model == "ridge":
        return Ridge(**cfg.params)
    if cfg.model == "hgb":
        if target == "winner":
            return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **cfg.params)
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **cfg.params)
    if cfg.model == "xgb_clf":
        if not XGB_AVAILABLE:
            return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **cfg.params)
        return xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                                  tree_method="hist", verbosity=0, **cfg.params)
    if cfg.model == "xgb_reg":
        if not XGB_AVAILABLE:
            return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **cfg.params)
        return xgb.XGBRegressor(random_state=RANDOM_STATE, tree_method="hist", verbosity=0,
                                 **cfg.params)
    raise ValueError(f"unknown model {cfg.model}")


def _evaluate(target: str, y_true, y_pred, sample_w=None) -> Dict[str, float]:
    if target == "winner":
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
    X = arrays["X"]
    w = arrays["w"]
    out: Dict[str, Any] = {"n_matches": int(X.shape[0]),
                            "n_features": int(X.shape[1]),
                            "feature_names": list(arrays["feature_names"]),
                            "results": {}}
    for target in TARGETS:
        y = arrays[f"y_{target}"]
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
        print(f"  {target:18s} best={best[0]:24s} {sort_key}={best_score:.4f} (n={len(Xt)})",
              file=sys.stderr)
    return out


def feature_importance(arrays: Dict[str, np.ndarray], grid: Dict[str, Any]) -> Dict[str, Any]:
    """Compute |coefficient| / permutation importance for best config per target."""
    X = arrays["X"]
    w = arrays["w"]
    out: Dict[str, Any] = {"method": "coefficient|permutation", "results": {}}
    rng = np.random.default_rng(RANDOM_STATE)
    for target in TARGETS:
        if target not in grid["results"]:
            continue
        y = arrays[f"y_{target}"]
        if target != "winner":
            mask = (y > 0) & np.isfinite(y)
        else:
            mask = np.ones(X.shape[0], dtype=bool)
        Xt, yt, wt = X[mask], y[mask], w[mask]
        cfg_name = grid["results"][target]["best_config"]
        cfg = next(c for c in make_configs(target) if c.name == cfg_name)
        model = _make_model(target, cfg)
        try:
            model.fit(Xt, yt, sample_weight=wt)
        except TypeError:
            model.fit(Xt, yt)
        imp = np.zeros(X.shape[1], dtype=np.float32)
        # Coef-based importance
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 1:
                imp = np.abs(coef).astype(np.float32)
            else:
                imp = np.abs(coef).mean(axis=0).astype(np.float32)
        # XGBoost importance
        elif hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=np.float32)
        # HGB
        elif hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=np.float32)
        else:
            # Permutation importance fallback
            base_pred = model.predict(Xt)
            if target == "winner":
                base = -log_loss(yt, np.clip(base_pred, 1e-7, 1-1e-7), labels=[0, 1])
            else:
                base = mean_absolute_error(yt, base_pred, sample_weight=wt)
            for j in range(X.shape[1]):
                Xp = Xt.copy()
                rng.shuffle(Xp[:, j])
                pred = model.predict(Xp)
                if target == "winner":
                    score = -log_loss(yt, np.clip(pred, 1e-7, 1-1e-7), labels=[0, 1])
                else:
                    score = mean_absolute_error(yt, pred, sample_weight=wt)
                imp[j] = max(0.0, score - base)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        order = np.argsort(imp)[::-1]
        out["results"][target] = {
            "best_config": cfg_name,
            "importance": [
                {"feature": arrays["feature_names"][int(j)], "score": float(imp[int(j)])}
                for j in order
            ],
        }
    return out


def train_production(arrays: Dict[str, np.ndarray], grid: Dict[str, Any]) -> Dict[str, Any]:
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
            "trainer_version": "v2",
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        out[target] = {"dir": str(target_dir), "metrics": meta["metrics_honest"]}
        print(f"  saved {target_dir}", file=sys.stderr)
    return out


def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    print("[train_v17_v2] loading data...", file=sys.stderr)
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
    if cmd in ("grid", "all"):
        print("[train_v17_v2] running grid research...", file=sys.stderr)
        grid = grid_research(arrays)
        GRID_OUT.write_text(json.dumps(grid, indent=2, default=str))
        print(f"[train_v17_v2] wrote {GRID_OUT}", file=sys.stderr)
        print("[train_v17_v2] computing feature importance...", file=sys.stderr)
        imp = feature_importance(arrays, grid)
        IMP_OUT.write_text(json.dumps(imp, indent=2, default=str))
        print(f"[train_v17_v2] wrote {IMP_OUT}", file=sys.stderr)
    if cmd in ("train", "all"):
        if not GRID_OUT.exists():
            print(f"  missing {GRID_OUT}; run grid first", file=sys.stderr)
            return 2
        grid = json.loads(GRID_OUT.read_text(encoding="utf-8"))
        print("[train_v17_v2] training production models...", file=sys.stderr)
        results = train_production(arrays, grid)
        TRAIN_OUT.write_text(json.dumps(results, indent=2, default=str))
        print(f"[train_v17_v2] wrote {TRAIN_OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
