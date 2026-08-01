"""v18 winner predictor -- XGBoost on 3403 OpenDota pro matches.

The v17 model (business/v17_predict.py) was a logreg_l1 that
zeroed out 19 of 21 features, leaving a constant sigmoid(0.4)
baseline for any non-top-team match.  See
scripts/diag_v17_signal.py and the v0.6.0 commit message for
the smoking gun.

v18 is a gradient-boosted decision tree (XGBoost 3.0+) on
the same 21 v17 features + 512 hero one-hot slots (256
heroes x 2 sides).  It actually uses the hero features --
different drafts give different predictions instead of the
"59% on every team" we saw with v17.

Conforms to the v17 module's interface (predict_winner_v18
returns a dict in the v17 shape) so the live/postmatch cards
can call it as a drop-in replacement.

Loading: v18 model files live at
  ml_data/models/_v18_winner/{model.joblib, metadata.json}
The leading underscore on _v18_winner is the same convention
v17 uses (keeps the legacy ModelStorage scan in
business/ml/storage.py from picking it up -- the legacy
feature_groups schema doesn't match v18's 523-feature
schema).

Failure modes: if the model files are missing, or joblib
fails to load, or the input doesn't pass basic sanity
checks, we raise v18_unavailable.  The caller (v17_predict)
catches that and falls back to the v17 model so a v18
problem never kills the live card.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

PRO_ROOT = Path(__file__).resolve().parents[1]
ML_DATA = PRO_ROOT / "ml_data"
MODELS_DIR = ML_DATA / "models"
TOP_TEAMS_PATH = ML_DATA / "imports" / "v17_phase1_top_teams.json"
PATCH_INFO_PATH = ML_DATA / "imports" / "v17_phase7_patch_info.json"

TIER_THRESHOLD_PREMIUM = 1400
TIER_THRESHOLD_PROFESSIONAL = 1100
NUM_HEROES = 256

# Optional one-time load cache (re-built on first call per process).
_MODEL_CACHE: Any = None
_META_CACHE: Optional[Dict[str, Any]] = None
_TOP_TEAMS_CACHE: Optional[Dict[int, float]] = None
_PATCH_INFO_CACHE: Optional[Dict[str, str]] = None


class v18_unavailable(RuntimeError):
    """Raised when v18 model can't be loaded or run.  Caller
    should fall back to v17."""


def _load_v18() -> Tuple[Any, Dict[str, Any]]:
    global _MODEL_CACHE, _META_CACHE
    if _MODEL_CACHE is not None and _META_CACHE is not None:
        return _MODEL_CACHE, _META_CACHE
    if joblib is None:
        raise v18_unavailable("joblib is not installed")
    path = MODELS_DIR / "_v18_winner"
    model_file = path / "model.joblib"
    meta_file = path / "metadata.json"
    if not (model_file.exists() and meta_file.exists()):
        raise v18_unavailable(
            f"v18 model not found at {path}.  Run scripts/train_v18.py first."
        )
    try:
        model = joblib.load(model_file)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise v18_unavailable(f"failed to load v18 model: {exc}") from exc
    _MODEL_CACHE = model
    _META_CACHE = meta
    return model, meta


def _load_top_teams() -> Dict[int, float]:
    """team_id -> rating, used to compute the r/d_top_team flags."""
    global _TOP_TEAMS_CACHE
    if _TOP_TEAMS_CACHE is not None:
        return _TOP_TEAMS_CACHE
    out: Dict[int, float] = {}
    if TOP_TEAMS_PATH.exists():
        try:
            data = json.loads(TOP_TEAMS_PATH.read_text(encoding="utf-8"))
            for t in data:
                tid = t.get("team_id")
                if tid is None:
                    continue
                try:
                    out[int(tid)] = float(t.get("rating") or 0)
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
    _TOP_TEAMS_CACHE = out
    return out


def _load_patch_info() -> Dict[str, str]:
    global _PATCH_INFO_CACHE
    if _PATCH_INFO_CACHE is not None:
        return _PATCH_INFO_CACHE
    out: Dict[str, str] = {}
    if PATCH_INFO_PATH.exists():
        try:
            data = json.loads(PATCH_INFO_PATH.read_text(encoding="utf-8"))
            out = {x.get("name"): x.get("date") for x in data if isinstance(x, dict)}
        except Exception:
            pass
    _PATCH_INFO_CACHE = out
    return out


def _days_since_patch(start_time: Optional[int],
                       patch: Optional[str],
                       patch_info: Dict[str, str]) -> float:
    if not patch or patch not in patch_info:
        # Unknown patch.  Use the most recent known patch as a
        # default so the model sees a reasonable number rather
        # than the v17-era 90.0 default.  This matters because
        # the XGBoost tree was trained with feature values in
        # the [0..180]-day range, not the 0..365+ range.
        if patch_info:
            try:
                import time as _t
                most_recent = max(patch_info.values(), default="")
                if most_recent:
                    ts = int(_t.mktime(_t.strptime(most_recent[:10], "%Y-%m-%d")))
                    if start_time:
                        return max(0.0, (start_time - ts) / 86400.0)
            except Exception:
                pass
        return 30.0
    try:
        import time as _t
        ts = int(_t.mktime(_t.strptime(patch_info[patch][:10], "%Y-%m-%d")))
        if start_time:
            return max(0.0, (start_time - ts) / 86400.0)
    except Exception:
        pass
    return 30.0


def _tier_for(team_id: Optional[int], top_teams: Dict[int, float]) -> int:
    if not team_id:
        return 0
    r = top_teams.get(int(team_id), 0)
    if r >= TIER_THRESHOLD_PREMIUM:
        return 2
    if r >= TIER_THRESHOLD_PROFESSIONAL:
        return 1
    return 0


def _build_features(
    r_picks: List[int],
    d_picks: List[int],
    r_team_id: Optional[int],
    d_team_id: Optional[int],
    r_bans: Optional[List[int]],
    d_bans: Optional[List[int]],
    start_time: Optional[int],
    patch: Optional[str],
) -> Dict[str, Any]:
    top_teams = _load_top_teams()
    patch_info = _load_patch_info()
    r_tier = _tier_for(r_team_id, top_teams)
    d_tier = _tier_for(d_team_id, top_teams)
    feats: Dict[str, Any] = {
        "r_tier": r_tier,
        "d_tier": d_tier,
        "r_top_team": 1 if r_tier >= 1 else 0,
        "d_top_team": 1 if d_tier >= 1 else 0,
        "r_premium": 1 if r_tier == 2 else 0,
        "d_premium": 1 if d_tier == 2 else 0,
        "r_picks": float(len(r_picks or [])),
        "d_picks": float(len(d_picks or [])),
        "r_bans": float(len(r_bans or [])),
        "d_bans": float(len(d_bans or [])),
        "days_since_patch": _days_since_patch(start_time, patch, patch_info),
    }
    r_set = set(int(h) for h in (r_picks or []))
    d_set = set(int(h) for h in (d_picks or []))
    for h in range(NUM_HEROES):
        feats[f"r_h_{h}"] = 1 if h in r_set else 0
        feats[f"d_h_{h}"] = 1 if h in d_set else 0
    return feats


def predict_winner_v18(
    radiant_picks: List[int],
    dire_picks: List[int],
    radiant_team_id: Optional[int] = None,
    dire_team_id: Optional[int] = None,
    radiant_bans: Optional[List[int]] = None,
    dire_bans: Optional[List[int]] = None,
    start_time: Optional[int] = None,
    patch: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the v18 winner model.  Returns the same dict shape
    as v17_predict.predict()'s "winner" sub-dict, plus a
    "source" field for downstream observability.

    Raises v18_unavailable if the model isn't installed (caller
    falls back to v17).
    """
    model, meta = _load_v18()
    feat_names = meta["feature_columns"]
    feats = _build_features(
        radiant_picks, dire_picks,
        radiant_team_id, dire_team_id,
        radiant_bans, dire_bans,
        start_time, patch,
    )
    try:
        import numpy as np
        X = np.asarray(
            [[feats.get(f, 0.0) for f in feat_names]],
            dtype=np.float32,
        )
    except Exception as exc:
        raise v18_unavailable(f"numpy conversion failed: {exc}") from exc
    try:
        proba = model.predict_proba(X)
        prob_radiant = float(proba[0, 1]) if proba.shape[1] > 1 else 0.5
    except Exception as exc:
        raise v18_unavailable(f"v18 predict_proba failed: {exc}") from exc
    if not (0.0 <= prob_radiant <= 1.0):
        prob_radiant = 0.5
    winner_team = "radiant" if prob_radiant >= 0.5 else "dire"
    return {
        "team": winner_team,
        "prob_radiant": prob_radiant,
        "probability": max(prob_radiant, 1.0 - prob_radiant),
        "source": "v18",
    }


def is_available() -> bool:
    """Quick check: are the v18 model files present?"""
    try:
        _load_v18()
        return True
    except v18_unavailable:
        return False
