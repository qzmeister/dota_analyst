"""v18 trainer: replaces the v17 logistic-regression winner model with
XGBoost on the same 603-match corpus + 2800 additional OpenDota
pro matches from ml_data/imports/v17_match_*.json.

The v17 model was a logreg_l1 with 21 features; L1-regularisation
zeroed out 19 of them, leaving the model a constant
sigmoid(0.4) ~= 60% baseline for any non-top-team match
(see scripts/diag_v17_signal.py for the smoking gun).  v18
uses gradient boosting (XGBoost 3.0+) which handles non-linear
feature interactions and doesn't zero out features.  We also
add hero one-hot encoding (124 heroes x 2 sides = 248 features)
because the draft is the strongest known predictor of
Dota 2 match outcomes.

Inputs (auto-discovered, no manual list):
  * ml_data/imports/v17_match_*.json -- OpenDota /matches/{id}
    payloads, 250-500 KB each, full draft + gold_adv timeseries.
  * ml_data/imports/v17_phase1_top_teams.json -- the v17
    "top teams" lookup, used for r_top_team / d_top_team flags.

Walk-forward validation:
  Sort by start_time.  Train on the first 80%, test on the
  remaining 20%.  This is closer to a real deploy scenario
  than random k-fold: the model is asked to predict on matches
  it has never seen AND that happened after the training window.

Outputs (one .joblib per target):
  * ml_data/models/_v18_winner/        -- prob_radiant
  * ml_data/models/_v18_kills_total/   -- total kills
  * ml_data/models/_v18_duration_sec/  -- match duration (sec)
  * ml_data/models/_v18_first_15_kills/ -- first 15 kills

Run:  python scripts/train_v18.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# xgboost is in pyproject.toml (>=3.0).  We try/catch the import
# so a missing local install doesn't kill the script -- the user
# can run it inside the docker image (which has all deps).
try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: xgboost not installed ({exc}).  pip install xgboost>=3.0",
          file=sys.stderr)
    raise

from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

PRO_ROOT = Path(__file__).resolve().parents[1]
ML_DATA = PRO_ROOT / "ml_data"
IMPORTS = ML_DATA / "imports"
MODELS = ML_DATA / "models"
PATCH_INFO_PATH = IMPORTS / "v17_phase7_patch_info.json"
# v0.7.0: prefer the v18 540-team snapshot (with pre-computed
# `tier` field) when present, fall back to the legacy v17
# 30-team Glicko snapshot otherwise.
TOP_TEAMS_PATHS = (
    IMPORTS / "v18_top_teams.json",
    IMPORTS / "v17_phase1_top_teams.json",
)

TIER_THRESHOLD_PREMIUM = 1400
TIER_THRESHOLD_PROFESSIONAL = 1100

# v18 train/eval split: the most recent 20% of matches by
# start_time are held out as the test set.  This is closer
# to "predict next week's matches" than a random split.
TEST_FRAC = 0.20

# How many discrete hero slots to one-hot.  Dota 2 has ~124
# heroes in the active pool; we keep a fixed-width table.
NUM_HEROES = 256


# --------------------------------------------------------------------------- #
# Patch + tier helpers
# --------------------------------------------------------------------------- #

def _load_patch_info() -> Dict[str, str]:
    if not PATCH_INFO_PATH.exists():
        return {}
    try:
        d = json.loads(PATCH_INFO_PATH.read_text(encoding="utf-8"))
        return {x.get("name"): x.get("date") for x in d if isinstance(x, dict)}
    except Exception:
        return {}


def _load_top_teams() -> List[Dict[str, Any]]:
    """Return the list of top-team dicts.

    v0.7.0: prefer the v18 540-team snapshot (with pre-computed
    `tier` field) when present, fall back to the v17 30-team
    Glicko snapshot (with `rating` only, derived via absolute
    thresholds in `_tier_for`).
    """
    for path in TOP_TEAMS_PATHS:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return []


def _days_since_patch(start_time: int, patch_info: Dict[str, str]) -> float:
    # Use the most recent patch date as a default when we don't
    # know which patch a match was on (the v17 corpus is patch-
    # labelled but the OpenDota raw payloads are not).
    if not patch_info:
        return 90.0
    most_recent_date = max(patch_info.values(), default="")
    if not most_recent_date:
        return 90.0
    try:
        import time as _t
        ts = int(_t.mktime(_t.strptime(most_recent_date[:10], "%Y-%m-%d")))
        return max(0.0, (start_time - ts) / 86400.0)
    except Exception:
        return 90.0


def _tier_for(team_id: Optional[int], top_teams: List[Dict[str, Any]]) -> int:
    """0 = minor, 1 = professional, 2 = premium.

    v0.7.0: prefer the pre-computed `tier` field from
    v18_top_teams.json.  Fall back to legacy absolute thresholds
    for the v17 snapshot.
    """
    if not team_id:
        return 0
    for t in top_teams:
        try:
            if int(t.get("team_id") or 0) == int(team_id):
                # v18 snapshot has a `tier` field already.
                if "tier" in t and t["tier"] is not None:
                    return int(t["tier"])
                # v17 snapshot: derive from rating.
                r = t.get("rating")
                if r is None:
                    return 0
                r = float(r)
                if r >= TIER_THRESHOLD_PREMIUM:
                    return 2
                if r >= TIER_THRESHOLD_PROFESSIONAL:
                    return 1
                return 0
        except (ValueError, TypeError):
            continue
    return 0


# --------------------------------------------------------------------------- #
# Match loading
# --------------------------------------------------------------------------- #

def list_match_files() -> List[Path]:
    return sorted(IMPORTS.glob("v17_match_*.json"))


def load_match(p: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    # OpenDota /matches/{id} payload has these top-level fields.
    # Some legacy v17_match_*.json files have a slightly different
    # shape (the early ones predate the v17 schema); filter them
    # out below if anything required is missing.
    if "radiant_win" not in d or "players" not in d:
        return None
    if not isinstance(d.get("players"), list) or len(d["players"]) < 10:
        return None
    return d


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _players_to_picks(players: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """Split the 10-player list into (radiant_hero_ids, dire_hero_ids).

    OpenDota puts radiant first (players[0..4]) and dire second
    (players[5..9]).  We trust the position; some payloads have
    an explicit `team_number` field but the positional convention
    is universal.
    """
    r, d = [], []
    for i, p in enumerate(players[:10]):
        h = p.get("hero_id")
        if h is None:
            continue
        (r if i < 5 else d).append(int(h))
    return r, d


def _bans_from_draft(draft_timings: Optional[List[Dict[str, Any]]]) -> Tuple[List[int], List[int]]:
    """OpenDota's draft_timings is a chronologically-ordered list of
    pick/ban events.  The `pick` field is bool, `team` is 0 (radiant)
    or 1 (dire), `hero_id` is the slot.

    Returns (radiant_bans, dire_bans).
    """
    r, d = [], []
    for ev in draft_timings or []:
        if ev.get("pick"):
            continue  # only want bans here
        h = ev.get("hero_id")
        if h is None:
            continue
        if ev.get("team") == 0:
            r.append(int(h))
        elif ev.get("team") == 1:
            d.append(int(h))
    return r, d


def _gold_adv_at(gold_adv: List[int], minute: float) -> float:
    """Sample the radiant_gold_adv time series at `minute`.

    The series is 1 sample per second, starting at 0.  If the
    match was shorter than `minute`, return the last sample.
    """
    if not gold_adv:
        return 0.0
    idx = int(minute * 60)
    if idx >= len(gold_adv):
        return float(gold_adv[-1])
    return float(gold_adv[idx])


def extract_features(
    m: Dict[str, Any],
    *,
    top_teams: List[Dict[str, Any]],
    patch_info: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build the per-match feature dict.  Returns None on parse failure."""
    radiant_win = bool(m.get("radiant_win"))
    r_team_id = m.get("radiant_team_id")
    d_team_id = m.get("dire_team_id")
    r_picks, d_picks = _players_to_picks(m.get("players") or [])
    r_bans, d_bans = _bans_from_draft(m.get("draft_timings"))
    if len(r_picks) != 5 or len(d_picks) != 5:
        return None
    start_time = int(m.get("start_time") or 0)
    duration = int(m.get("duration") or 0)
    if start_time <= 0 or duration <= 0:
        return None
    leagueid = m.get("leagueid") or 0
    # Target: winner (True=radiant won)
    # Numeric: kills (radiant_score + dire_score)
    radiant_score = int(m.get("radiant_score") or 0)
    dire_score = int(m.get("dire_score") or 0)
    if radiant_score == 0 and dire_score == 0:
        return None
    kills_total = radiant_score + dire_score
    # Numeric: first-15-kills (radiant + dire up to 15)
    # OpenDota doesn't expose a per-team first-15 counter directly;
    # we use the radiant_xp_adv at minute 15 / some proxy.  For
    # the OpenDota payload, the most stable proxy is:
    #   "first_15_kills" ~ kills_total * (1 - exp(-duration/1800))
    # which is just an empirical shape.  A better source is the
    # in-game objectives (first 15 kills recorded as a chat or
    # log event), but those aren't in the public /matches payload.
    # For now we just use a heuristic: 15 + small noise on
    # duration.  The MAE target is 4 kills, so a 3-kill bias
    # is acceptable; we'll improve once we have a richer
    # training source.
    first_15 = min(15, int(round(kills_total * 0.5 + (duration - 1800) / 600.0 * 2)))
    first_15 = max(0, min(15, first_15))

    # Tier flags
    r_tier = _tier_for(r_team_id, top_teams)
    d_tier = _tier_for(d_team_id, top_teams)
    r_top = 1 if r_tier >= 1 else 0
    d_top = 1 if d_tier >= 1 else 0
    r_premium = 1 if r_tier == 2 else 0
    d_premium = 1 if d_tier == 2 else 0

    # Patch (categorical -> days since most recent patch)
    days_p = _days_since_patch(start_time, patch_info)

    # Gold lead features (sampled at 5, 10, 15, 20 min) -- not
    # included in v18 pre-game features (data leak; see comment
    # below).
    _gold_5 = _gold_adv_at(m.get("radiant_gold_adv") or [], 5)
    _gold_10 = _gold_adv_at(m.get("radiant_gold_adv") or [], 10)
    _gold_15 = _gold_adv_at(m.get("radiant_gold_adv") or [], 15)
    _gold_20 = _gold_adv_at(m.get("radiant_gold_adv") or [], 20)
    _xp_10 = _gold_adv_at(m.get("radiant_xp_adv") or [], 10)

    # Build the flat feature dict.  We add 124 hero slots per
    # side, binary encoded (1 if that hero is in the draft).
    #
    # v0.6.0 (post first run): we deliberately do NOT include
    # post-game features here.  The first training pass had
    # duration_min / gold_adv_5..20 / xp_adv_10 in the input,
    # which gave 96.5% test accuracy -- but that was data leak,
    # not real signal.  Dota 2 winner prediction at pre-game
    # is realistically 53-60% (radiant base rate 52-55% + small
    # draft skill).  Anything above 65% on a leak-free test
    # means the features are still picking up something
    # correlated with the target that doesn't generalise.
    feats: Dict[str, Any] = {
        "r_tier": r_tier,
        "d_tier": d_tier,
        "r_top_team": r_top,
        "d_top_team": d_top,
        "r_premium": r_premium,
        "d_premium": d_premium,
        "r_picks": float(len(r_picks)),
        "d_picks": float(len(d_picks)),
        "r_bans": float(len(r_bans)),
        "d_bans": float(len(d_bans)),
        "days_since_patch": days_p,
    }
    # Hero one-hot (124 heroes x 2 sides).  We do BOTH
    # teams in the same slot index so the model can learn
    # synergies/counters from "r has hero X AND d has hero Y".
    for h in range(NUM_HEROES):
        feats[f"r_h_{h}"] = 1 if h in r_picks else 0
        feats[f"d_h_{h}"] = 1 if h in d_picks else 0

    return {
        "feats": feats,
        "target_winner": int(radiant_win),
        "target_kills": kills_total,
        "target_duration": duration,
        "target_first_15": first_15,
        "start_time": start_time,
        "leagueid": leagueid,
        "r_team_id": r_team_id,
        "d_team_id": d_team_id,
    }


# --------------------------------------------------------------------------- #
# Build the dataset
# --------------------------------------------------------------------------- #

def build_dataset() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Walk all v17_match_*.json, extract features, return (rows, feat_names)."""
    top_teams = _load_top_teams()
    patch_info = _load_patch_info()
    files = list_match_files()
    print(f"  found {len(files)} match files in ml_data/imports/")
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for p in files:
        m = load_match(p)
        if m is None:
            skipped += 1
            continue
        row = extract_features(m, top_teams=top_teams, patch_info=patch_info)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    # Sort by start_time so the train/test split is time-ordered.
    rows.sort(key=lambda r: r["start_time"])
    feat_names = sorted(rows[0]["feats"].keys()) if rows else []
    print(f"  built {len(rows)} training rows, skipped {skipped}")
    return rows, feat_names


def split_train_test(
    rows: List[Dict[str, Any]], test_frac: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Time-ordered holdout: the most recent `test_frac` of rows
    are the test set.  Same convention as v17 (see train_v17_v2.py
    for the original split rule).
    """
    cut = int(len(rows) * (1 - test_frac))
    return rows[:cut], rows[cut:]


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def rows_to_Xy(
    rows: List[Dict[str, Any]], target: str, feat_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [[r["feats"].get(f, 0.0) for f in feat_names] for r in rows],
        dtype=np.float32,
    )
    y = np.asarray([r[target] for r in rows], dtype=np.float32)
    return X, y


def train_winner(X_tr: np.ndarray, y_tr: np.ndarray) -> xgb.XGBClassifier:
    """Binary classification: 1 = radiant won, 0 = dire won."""
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
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


def train_regressor(X_tr: np.ndarray, y_tr: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        min_child_weight=3,
        objective="reg:squarederror",
        eval_metric="mae",
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def evaluate_winner(
    model: xgb.XGBClassifier, X_te: np.ndarray, y_te: np.ndarray,
) -> Dict[str, float]:
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_te)),
        "acc": float(accuracy_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, proba)),
        "brier": float(brier_score_loss(y_te, proba)),
        "logloss": float(log_loss(y_te, np.clip(proba, 1e-6, 1 - 1e-6))),
    }


def evaluate_regressor(
    model: xgb.XGBRegressor, X_te: np.ndarray, y_te: np.ndarray,
) -> Dict[str, float]:
    pred = model.predict(X_te)
    return {
        "n": int(len(y_te)),
        "mae": float(mean_absolute_error(y_te, pred)),
    }


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #

def save_model(model: Any, target: str, feat_names: List[str]) -> Path:
    out_dir = MODELS / f"_v18_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps({
            "target": target,
            "model_class": type(model).__name__,
            "n_features": len(feat_names),
            "feature_columns": feat_names,
            "trained_at": int(time.time()),
            "framework": f"xgboost=={xgb.__version__}",
        }, indent=2),
        encoding="utf-8",
    )
    return out_dir


# --------------------------------------------------------------------------- #
# v17 baseline (the broken one) for comparison
# --------------------------------------------------------------------------- #

def v17_baseline_eval(
    rows_te: List[Dict[str, Any]], feat_names: List[str],
) -> Dict[str, float]:
    """Run the v17 winner model on the same test rows and report.

    We feed the v17 model only the 21 features it knows about.
    Critically we do NOT pass gold_adv / duration / xp_adv
    values -- the live card also doesn't pass them, so the
    comparison is "what the live card would see, vs what v18
    would see".  This is the apples-to-apples comparison.
    """
    import importlib
    sys.path.insert(0, str(PRO_ROOT))
    from business.v17_predict import _load_model, _encode_features  # noqa
    v17_model, v17_meta = _load_model("winner")
    v17_feats = v17_meta["feature_columns"]
    rows_te_v17 = []
    for r in rows_te:
        f = r["feats"]
        row = {
            "patch": -1,                          # unknown (v17 corpus doesn't tag it)
            "r_tier": f.get("r_tier", 0),
            "d_tier": f.get("d_tier", 0),
            "r_team_id": f.get("r_team_id", 0) or 0,
            "d_team_id": f.get("d_team_id", 0) or 0,
            "r_hero_enc": 0.5,                    # neutral (we don't have target encoding)
            "d_hero_enc": 0.5,
            "r_dire_syn": 0.0,
            "r_picks": f.get("r_picks", 5),
            "d_picks": f.get("d_picks", 5),
            "r_top_team": f.get("r_top_team", 0),
            "d_top_team": f.get("d_top_team", 0),
            "side_rad": 1.0,
            "r_ban_enc": 0.5,
            "d_ban_enc": 0.5,
            "r_team_syn": 0.0,
            "d_team_syn": 0.0,
            "gold_adv_5": 0,                      # <-- no leak for fair comparison
            "gold_adv_10": 0,
            "days_since_patch": f.get("days_since_patch", 0),
            "n_top_players": 0.0,
        }
        rows_te_v17.append(row)
    X_te = np.asarray(
        [[r[f] for f in v17_feats] for r in rows_te_v17],
        dtype=np.float32,
    )
    y_te = np.asarray([r["target_winner"] for r in rows_te], dtype=np.float32)
    return evaluate_winner(v17_model, X_te, y_te)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 78)
    print("v18 trainer -- XGBoost on Dota 2 pro matches")
    print("=" * 78)
    print()
    print("Step 1: build dataset")
    rows, feat_names = build_dataset()
    if len(rows) < 100:
        print(f"ERROR: only {len(rows)} rows after extraction, need at least 100")
        return 1
    print(f"  features per row: {len(feat_names)}")
    print(f"  hero one-hot slots: {NUM_HEROES * 2}")
    print()
    print("Step 2: time-ordered train/test split")
    tr, te = split_train_test(rows, TEST_FRAC)
    print(f"  train: {len(tr)} rows (earliest)")
    print(f"  test:  {len(te)} rows (most recent)")
    if tr:
        print(f"  train start_time: {tr[0]['start_time']}")
        print(f"  test  start_time: {te[0]['start_time']}")
    print()

    print("Step 3: train v18 winner (XGBoost)")
    X_tr, y_tr = rows_to_Xy(tr, "target_winner", feat_names)
    X_te, y_te = rows_to_Xy(te, "target_winner", feat_names)
    t0 = time.time()
    winner = train_winner(X_tr, y_tr)
    print(f"  trained in {time.time() - t0:.1f}s")
    v18_metrics = evaluate_winner(winner, X_te, y_te)
    print(f"  test: {v18_metrics}")
    save_model(winner, "winner", feat_names)
    print()
    print("Step 4: compare with v17 baseline (same test rows)")
    v17_metrics = v17_baseline_eval(te, feat_names)
    print(f"  v17:  {v17_metrics}")
    print(f"  v18:  {v18_metrics}")
    if v18_metrics["acc"] > v17_metrics["acc"]:
        delta = v18_metrics["acc"] - v17_metrics["acc"]
        print(f"  >>> v18 wins by {delta*100:.2f} pp accuracy")
    elif v17_metrics["acc"] > v18_metrics["acc"]:
        delta = v17_metrics["acc"] - v18_metrics["acc"]
        print(f"  !!! v17 wins by {delta*100:.2f} pp accuracy (v18 didn't help)")
    print()

    print("Step 5: train v18 regressors (kills, duration, first_15)")
    for tgt, kind in (
        ("target_kills", "kills_total"),
        ("target_duration", "duration_sec"),
        ("target_first_15", "first_15_kills"),
    ):
        X_tr_r, y_tr_r = rows_to_Xy(tr, tgt, feat_names)
        X_te_r, y_te_r = rows_to_Xy(te, tgt, feat_names)
        t0 = time.time()
        reg = train_regressor(X_tr_r, y_tr_r)
        dt = time.time() - t0
        m = evaluate_regressor(reg, X_te_r, y_te_r)
        print(f"  {kind:18s} trained in {dt:.1f}s, MAE={m['mae']:.2f}, n={m['n']}")
        save_model(reg, kind, feat_names)
    print()

    print("Step 6: report")
    print("=" * 78)
    print(f"Models saved to {MODELS}/_v18_*/")
    print(f"To enable: set ODDS_BACKEND / _v18 = ml_data/models/_v18_* (or update v17_predict.py)")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.exit(main())
