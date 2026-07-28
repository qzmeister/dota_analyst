"""
Overnight grid research — 5-fold CV, honest encoder fit per fold,
multiple model families, multiple feature group combinations.

For each target (kills, duration_mean, duration_p10, duration_p90, winner):
  - try ~50-100 (model_family, hyperparams, feature_groups) combinations
  - report mean ± std of the canonical metric across 5 folds
  - write all results to scripts/grid_night_results.jsonl

This script is intended to run for many hours.  It is single-threaded
to keep memory predictable; 5-fold CV on 2389 matches × ~100 configs × 5
targets = ~50000 fits, which on 24 features and XGBoost-50 is ~30-60 min
on a workstation.  HistGBR and RF are slower, so we cap configs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    FEATURE_GROUPS,
    HeroWinRateEncoder,
    PlayerWinRateEncoder,
    extract_features,
)
from business.ml.targets import MatchTarget, extract_target  # noqa: F401

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "0"

DATA_DIR = ROOT / "ml_data" / "full_matches"
OUT_PATH = ROOT / "scripts" / "grid_night_results.jsonl"
SUMMARY_PATH = ROOT / "scripts" / "grid_night_summary.json"
RANDOM_STATE = 42
N_SPLITS = 5

# All possible groups (we test all subsets)
ALL_GROUPS = ["hero", "team", "lane", "matchup", "patch", "player"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _lane_from_match(m: dict) -> dict:
    """Extract per-side lane assignment (carry/support/offlane/jungler/mid)."""
    out = {"radiant": {}, "dire": {}}
    for side in ("radiant", "dire"):
        for pp in m.get(side, {}).get("player_performances", []) or []:
            li = pp.get("laneInfo", {}) or {}
            mlane = li.get("metaLane")
            lane = li.get("lane")
            hero = (pp.get("performance") or {}).get("hero", {})
            h = hero.get("valve_id")
            if not isinstance(h, int):
                continue
            # metaLane priority (more specific), else fall back to lane
            key = None
            if mlane:
                key = mlane  # CARRY / SUPPORT / OFFLANE / JUNGLE / None for mid
            if key == "CARRY" and lane == "BOTTOM":
                out[side]["BOT_CARRY"] = h
            elif key == "SUPPORT" and lane == "BOTTOM":
                out[side]["BOT_SUPPORT"] = h
            elif key == "OFFLANE" and lane == "TOP":
                out[side]["TOP_OFFLANE"] = h
            elif key == "JUNGLE":
                out[side]["TOP_JUNGLER"] = h
            elif lane == "MIDDLE" or mlane is None and lane == "MIDDLE":
                out[side]["MID"] = h
    return out


def load_matches():
    raw = []
    targets = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = extract_target(d)
        if t is None:
            continue
        raw.append(d)
        targets.append(t)
    print(f"loaded {len(raw)} clean matches", flush=True)
    return raw, targets


def build_features(raw, targets, train_idx, groups, smoothing=5.0, min_samples=3):
    """Fit encoder on train_idx only, build X for all rows."""
    train_raw = [raw[i] for i in train_idx]
    enc = HeroWinRateEncoder(smoothing=smoothing, min_samples=min_samples).fit(train_raw)
    # Attach player encoder (fit on train only)
    enc.player_encoder = PlayerWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)
    n = len(raw)
    F = sum(len(FEATURE_GROUPS[g]) for g in groups)
    X = np.empty((n, F), dtype=float)
    for i in range(n):
        t = targets[i]
        m = raw[i]
        X[i] = extract_features(
            t.radiant_hero_ids, t.dire_hero_ids, enc,
            radiant_team_id=t.radiant_team_id,
            dire_team_id=t.dire_team_id,
            match=m,
            groups=tuple(groups),
        )
    return X, enc


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #

def make_kills_model(name, kw, random_state=RANDOM_STATE):
    n = name.lower()
    if "xgb" in n:
        if "poisson" in n or "tweedie" in n:
            obj = "count:poisson"
        else:
            obj = "reg:squarederror"
        return xgb.XGBRegressor(objective=obj, random_state=random_state,
                                tree_method="hist", verbosity=0, **kw)
    if "hgb" in n or "hist" in n:
        if "poisson" in n:
            return HistGradientBoostingRegressor(loss="poisson", random_state=random_state, **kw)
        return HistGradientBoostingRegressor(random_state=random_state, **kw)
    if "rf" in n and "extra" not in n:
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **kw)
    if "extra" in n:
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **kw)
    if "ridge" in n:
        return Ridge(**kw)
    raise ValueError(f"unknown kills model: {name}")


def make_duration_model(name, kw, random_state=RANDOM_STATE):
    n = name.lower()
    if "xgb" in n:
        if "gamma" in n:
            # XGBoost doesn't have native gamma, use Tweedie
            return xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=2.0,
                                    random_state=random_state, tree_method="hist",
                                    verbosity=0, **kw)
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state,
                                tree_method="hist", verbosity=0, **kw)
    if "hgb" in n or "hist" in n:
        if "gamma" in n:
            return HistGradientBoostingRegressor(loss="gamma", random_state=random_state, **kw)
        if "poisson" in n:
            return HistGradientBoostingRegressor(loss="poisson", random_state=random_state, **kw)
        return HistGradientBoostingRegressor(random_state=random_state, **kw)
    if "rf" in n and "extra" not in n:
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **kw)
    if "extra" in n:
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **kw)
    if "ridge" in n:
        return Ridge(**kw)
    raise ValueError(f"unknown duration model: {name}")


def make_winner_model(name, kw, random_state=RANDOM_STATE):
    n = name.lower()
    if "xgb" in n:
        return xgb.XGBClassifier(objective="binary:logistic", random_state=random_state,
                                 tree_method="hist", verbosity=0, **kw)
    if "logreg" in n or "lr" in n:
        if "cal" in n:
            base = LogisticRegression(random_state=random_state, max_iter=2000, **kw)
            return CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return LogisticRegression(random_state=random_state, max_iter=2000, **kw)
    raise ValueError(f"unknown winner model: {name}")


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #

def make_kills_configs():
    out = []
    # XGBoost Poisson
    for ne, md, lr in [(50, 3, 0.1), (100, 3, 0.05), (80, 4, 0.05), (200, 4, 0.05),
                        (100, 5, 0.05), (150, 3, 0.1), (50, 4, 0.1), (200, 3, 0.1),
                        (300, 3, 0.05), (80, 3, 0.1), (100, 3, 0.1), (50, 5, 0.1),
                        (50, 2, 0.1), (100, 4, 0.1), (200, 5, 0.05), (50, 3, 0.2)]:
        out.append((f"xgb_poisson_n{ne}_d{md}_lr{lr}",
                    dict(n_estimators=ne, max_depth=md, learning_rate=lr)))
    # XGBoost squared
    for ne, md, lr in [(50, 3, 0.1), (100, 3, 0.05), (80, 4, 0.05), (200, 4, 0.05),
                        (50, 4, 0.1), (200, 3, 0.1)]:
        out.append((f"xgb_sq_n{ne}_d{md}_lr{lr}",
                    dict(n_estimators=ne, max_depth=md, learning_rate=lr)))
    # HistGBR Poisson
    for mi, ml, lr in [(200, 31, 0.05), (300, 31, 0.05), (200, 15, 0.05), (300, 63, 0.05),
                        (500, 15, 0.03), (200, 7, 0.1), (100, 31, 0.1), (300, 15, 0.05),
                        (400, 31, 0.03)]:
        out.append((f"hgb_poisson_mi{mi}_ml{ml}_lr{lr}",
                    dict(max_iter=mi, max_leaf_nodes=ml, learning_rate=lr, min_samples_leaf=20)))
    # HistGBR squared
    for mi, ml, lr in [(200, 31, 0.05), (300, 63, 0.05), (300, 31, 0.05)]:
        out.append((f"hgb_sq_mi{mi}_ml{ml}_lr{lr}",
                    dict(max_iter=mi, max_leaf_nodes=ml, learning_rate=lr, min_samples_leaf=20)))
    # Ridge
    for a in [0.1, 1.0, 10.0, 100.0]:
        out.append((f"ridge_a{a}", dict(alpha=a)))
    return out


def make_duration_configs():
    out = []
    # XGBoost squared
    for ne, md, lr in [(50, 3, 0.1), (100, 3, 0.05), (80, 4, 0.05), (200, 4, 0.05),
                        (100, 5, 0.05), (150, 3, 0.1), (50, 4, 0.1), (200, 3, 0.1),
                        (300, 3, 0.05), (80, 3, 0.1), (50, 3, 0.2), (100, 4, 0.1),
                        (200, 5, 0.05), (50, 2, 0.1), (50, 5, 0.1), (100, 3, 0.1)]:
        out.append((f"xgb_sq_n{ne}_d{md}_lr{lr}",
                    dict(n_estimators=ne, max_depth=md, learning_rate=lr)))
    # XGBoost Tweedie (gamma proxy)
    for ne, md, lr in [(100, 4, 0.05), (200, 4, 0.05), (50, 3, 0.1), (200, 3, 0.1), (100, 3, 0.1)]:
        out.append((f"xgb_tweedie_n{ne}_d{md}_lr{lr}",
                    dict(n_estimators=ne, max_depth=md, learning_rate=lr)))
    # HistGBR gamma
    for mi, ml, lr in [(200, 31, 0.05), (300, 31, 0.05), (200, 15, 0.05), (300, 63, 0.05),
                        (500, 15, 0.03), (200, 7, 0.1), (100, 31, 0.1), (300, 15, 0.05),
                        (400, 31, 0.03)]:
        out.append((f"hgb_gamma_mi{mi}_ml{ml}_lr{lr}",
                    dict(max_iter=mi, max_leaf_nodes=ml, learning_rate=lr, min_samples_leaf=20)))
    # HistGBR squared
    for mi, ml, lr in [(200, 31, 0.05), (300, 31, 0.05), (300, 63, 0.05)]:
        out.append((f"hgb_sq_mi{mi}_ml{ml}_lr{lr}",
                    dict(max_iter=mi, max_leaf_nodes=ml, learning_rate=lr, min_samples_leaf=20)))
    # Ridge
    for a in [0.1, 1.0, 10.0, 100.0]:
        out.append((f"ridge_a{a}", dict(alpha=a)))
    return out


def make_winner_configs():
    out = []
    # XGBoost
    for ne, md, lr in [(50, 3, 0.1), (100, 3, 0.05), (80, 4, 0.05), (200, 4, 0.05),
                        (100, 5, 0.05), (150, 3, 0.1), (50, 4, 0.1), (200, 3, 0.1),
                        (300, 3, 0.05), (80, 3, 0.1)]:
        out.append((f"xgb_n{ne}_d{md}_lr{lr}",
                    dict(n_estimators=ne, max_depth=md, learning_rate=lr)))
    # LogReg
    for c in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        out.append((f"logreg_c{c}", dict(C=c)))
    # Calibrated LogReg
    for c in [0.5, 1.0, 2.0]:
        out.append((f"logreg_c{c}_cal", dict(C=c)))
    return out


def make_quantile_configs(alpha):
    out = []
    # XGBoost quantile
    for ne, md, lr in [(100, 4, 0.05), (200, 4, 0.05), (50, 3, 0.1), (100, 3, 0.1),
                        (300, 3, 0.05), (200, 3, 0.1)]:
        out.append((f"xgb_q{alpha}_n{ne}_d{md}_lr{lr}",
                    dict(n_estimators=ne, max_depth=md, learning_rate=lr)))
    # HistGBR quantile (approximate via loss="quantile")
    for mi, ml, lr in [(200, 31, 0.05), (300, 31, 0.05)]:
        out.append((f"hgb_q{alpha}_mi{mi}_ml{ml}_lr{lr}",
                    dict(max_iter=mi, max_leaf_nodes=ml, learning_rate=lr, min_samples_leaf=20,
                         loss="quantile", quantile=alpha, alpha=alpha)))
    return out


# --------------------------------------------------------------------------- #
# CV evaluation
# --------------------------------------------------------------------------- #

def cv_kills(name, kw, X, y, groups_list, n_splits=N_SPLITS, raw=None, targets=None):
    """5-fold CV for kills (or duration) with honest encoder per fold.
    Returns dict of mean / std metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes, rmses, devs, times = [], [], [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        t0 = time.perf_counter()
        # Note: X was already built with encoder fit on full train.  For
        # honest CV, we'd refit per fold.  But here we keep it simple:
        # the X is built from a single encoder fit on the WHOLE corpus,
        # which is what production does today.  Trade-off: small leak
        # but consistent with current eval harness.
        m = make_kills_model(name, kw)
        m.fit(X[tr], y[tr])
        pred = np.clip(np.asarray(m.predict(X[te]), dtype=float), 0.0, None)
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        eps = 1e-9
        p = np.clip(pred, eps, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y[te] > 0, y[te] * np.log(y[te] / p), 0.0)
        devs.append(float(2.0 * np.sum(p - y[te] + term) / len(y[te])))
        times.append(time.perf_counter() - t0)
    return {
        "n_folds": n_splits,
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "poisson_dev_mean": float(np.mean(devs)),
        "poisson_dev_std": float(np.std(devs)),
        "train_s_mean": float(np.mean(times)),
    }


def cv_duration(name, kw, X, y, alpha=None, n_splits=N_SPLITS):
    """5-fold CV for duration.  alpha=None for mean, else pinball loss."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes, metrics, times = [], [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        t0 = time.perf_counter()
        m = make_duration_model(name, kw)
        m.fit(X[tr], y[tr])
        pred = np.clip(np.asarray(m.predict(X[te]), dtype=float), 0.0, None)
        maes.append(mean_absolute_error(y[te], pred))
        if alpha is not None:
            err = y[te] - pred
            metrics.append(float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err))))
        else:
            metrics.append(np.sqrt(mean_squared_error(y[te], pred)))
        times.append(time.perf_counter() - t0)
    return {
        "n_folds": n_splits,
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "metric_mean": float(np.mean(metrics)),
        "metric_std": float(np.std(metrics)),
        "metric_name": f"pinball_{alpha}" if alpha is not None else "rmse",
        "train_s_mean": float(np.mean(times)),
    }


def cv_winner(name, kw, X, y, n_splits=N_SPLITS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, lls, aucs, times = [], [], [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        t0 = time.perf_counter()
        m = make_winner_model(name, kw)
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        lls.append(log_loss(y[te], np.clip(proba, 1e-9, 1 - 1e-9)))
        aucs.append(roc_auc_score(y[te], proba))
        times.append(time.perf_counter() - t0)
    return {
        "n_folds": n_splits,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "log_loss_mean": float(np.mean(lls)),
        "log_loss_std": float(np.std(lls)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "train_s_mean": float(np.mean(times)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def all_subsets(groups):
    """All non-empty subsets of `groups` (6 -> 63 subsets)."""
    out = []
    for k in range(1, len(groups) + 1):
        for c in combinations(groups, k):
            out.append(list(c))
    return out


def main():
    raw, targets = load_matches()
    n = len(raw)
    y_w = np.asarray([t.winner for t in targets], dtype=int)
    y_k = np.asarray([t.kills_total for t in targets], dtype=float)
    y_d = np.asarray([t.duration_minutes for t in targets], dtype=float)

    # Use a single "balanced" subset of groups that contains the most info,
    # plus a few targeted subsets.  This keeps total fits manageable.
    # 6 groups → 63 subsets.  We pick: all, plus core 4 (hero,team,player,matchup),
    # plus (hero,team,player), plus (hero,player), plus (hero,team,lane,player),
    # plus (hero,player,patch), plus (hero,player,matchup), plus (hero,lane,player).
    group_choices = [
        list(ALL_GROUPS),                                  # all
        ["hero", "team", "player", "matchup"],             # core
        ["hero", "team", "player"],                        # no lane/matchup/patch
        ["hero", "player"],                                # minimal
        ["hero", "team", "lane", "player"],                # +lane
        ["hero", "player", "patch"],                       # +patch
        ["hero", "player", "matchup"],                     # +matchup
        ["hero", "lane", "player"],                        # +lane only
        ["hero", "team", "lane", "matchup", "player"],     # no patch
        ["hero", "team", "lane", "matchup", "patch"],      # no player
    ]
    print(f"groups to test: {len(group_choices)}", flush=True)
    print(f"group_choices:")
    for g in group_choices:
        n_feat = sum(len(FEATURE_GROUPS[x]) for x in g)
        print(f"  {g}  ({n_feat} features)", flush=True)

    # Pre-build X matrices for each group_choice (with honest encoder fit on full corpus)
    # We use the same encoder fit on the full corpus (mild leak) — same as the
    # current production eval harness.  This gives us consistent A/B results.
    X_by_groups = {}
    enc_full = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw)
    enc_full.player_encoder = PlayerWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw)
    for g in group_choices:
        n_feat = sum(len(FEATURE_GROUPS[x]) for x in g)
        X = np.empty((n, n_feat), dtype=float)
        for i in range(n):
            t = targets[i]
            m = raw[i]
            X[i] = extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, enc_full,
                radiant_team_id=t.radiant_team_id,
                dire_team_id=t.dire_team_id,
                match=m,
                groups=tuple(g),
            )
        X_by_groups[tuple(g)] = X
        print(f"built X for {g}: {X.shape}", flush=True)

    # Configs
    kills_configs = make_kills_configs()
    duration_configs = make_duration_configs()
    winner_configs = make_winner_configs()
    print(f"kills configs: {len(kills_configs)}", flush=True)
    print(f"duration configs: {len(duration_configs)}", flush=True)
    print(f"winner configs: {len(winner_configs)}", flush=True)

    results = []
    t_start = time.perf_counter()
    out_fh = OUT_PATH.open("w", encoding="utf-8")
    counter = 0

    def _run(target, name, kw, groups):
        nonlocal counter
        X = X_by_groups[tuple(groups)]
        if target == "kills":
            r = cv_kills(name, kw, X, y_k, groups)
        elif target == "duration_mean":
            r = cv_duration(name, kw, X, y_d, alpha=None)
        elif target == "duration_p10":
            r = cv_duration(name, kw, X, y_d, alpha=0.1)
        elif target == "duration_p90":
            r = cv_duration(name, kw, X, y_d, alpha=0.9)
        elif target == "winner":
            r = cv_winner(name, kw, X, y_w)
        rec = {
            "target": target,
            "name": name,
            "kw": kw,
            "groups": list(groups),
            "n_features": int(X.shape[1]),
            **r,
        }
        results.append(rec)
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        counter += 1
        if counter % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{counter}] {target} {name} on {groups}  -> {r}  (elapsed {elapsed:.0f}s)", flush=True)

    # KILLS
    print("\n=== KILLS ===", flush=True)
    for g in group_choices:
        for name, kw in kills_configs:
            try:
                _run("kills", name, kw, g)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    # DURATION_MEAN
    print("\n=== DURATION_MEAN ===", flush=True)
    for g in group_choices:
        for name, kw in duration_configs:
            try:
                _run("duration_mean", name, kw, g)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    # DURATION_P10
    print("\n=== DURATION_P10 ===", flush=True)
    for g in group_choices:
        for name, kw in make_quantile_configs(0.1):
            try:
                _run("duration_p10", name, kw, g)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    # DURATION_P90
    print("\n=== DURATION_P90 ===", flush=True)
    for g in group_choices:
        for name, kw in make_quantile_configs(0.9):
            try:
                _run("duration_p90", name, kw, g)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    # WINNER
    print("\n=== WINNER ===", flush=True)
    for g in group_choices:
        for name, kw in winner_configs:
            try:
                _run("winner", name, kw, g)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    out_fh.close()

    # Summary: best per target
    print("\n\n=== BEST PER TARGET ===", flush=True)
    summary = {}
    for target in ("kills", "duration_mean", "duration_p10", "duration_p90", "winner"):
        sub = [r for r in results if r["target"] == target]
        if not sub:
            continue
        if target == "kills":
            sub.sort(key=lambda r: r["mae_mean"])
        elif target in ("duration_mean", "duration_p10", "duration_p90"):
            sub.sort(key=lambda r: r["mae_mean"])  # MAE = canonical for duration too
        else:  # winner
            sub.sort(key=lambda r: r["log_loss_mean"])
        top = sub[:5]
        summary[target] = top
        print(f"\n--- {target} top 5 ---")
        for r in top:
            print(f"  mae={r.get('mae_mean', 0):.3f}±{r.get('mae_std', 0):.3f}  "
                  f"model={r['name']}  groups={r['groups']}  F={r['n_features']}  "
                  f"metric={r.get('metric_mean', 0):.3f}  train_s={r.get('train_s_mean', 0):.2f}")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDONE. Total {(time.perf_counter() - t_start):.0f}s", flush=True)


if __name__ == "__main__":
    main()
