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
# v0.7.0: prefer the v18 top-teams snapshot (540 teams with
# percentile-based tier field) when present, fall back to the v17
# 30-team Glicko snapshot if the new file doesn't exist yet.
TOP_TEAMS_PATHS = (
    ML_DATA / "imports" / "v18_top_teams.json",
    ML_DATA / "imports" / "v17_phase1_top_teams.json",
)
PATCH_INFO_PATH = ML_DATA / "imports" / "v17_phase7_patch_info.json"

# v0.7.0: tier thresholds are now percentile-based inside
# v18_top_teams.json.  We keep these constants as a last-resort
# fallback for the legacy 30-team snapshot (where the Glicko
# rating has mean 1500 and std 350).
TIER_THRESHOLD_PREMIUM = 1400
TIER_THRESHOLD_PROFESSIONAL = 1100
NUM_HEROES = 256

# Optional one-time load cache (re-built on first call per process).
_MODEL_CACHE: Any = None
_META_CACHE: Optional[Dict[str, Any]] = None
_TOP_TEAMS_CACHE: Optional[Dict[int, int]] = None
_PATCH_INFO_CACHE: Optional[Dict[str, str]] = None
# v0.7.2: DLTV uses an internal hero-id namespace that's NOT
# the same as the OpenDota/Steam one.  Out of 127 DLTV heroes,
# 104 have id != steam_id (DLTV 24 = Lina / Steam 25, etc).
# The v18 model was trained on OpenDota hero_ids (Steam), so
# if the live card passes a DLTV id (which it does for v1 API
# sources and most browser cache entries), the model sees the
# wrong hero.  We build the mapping lazily on first call and
# convert any incoming hero_id in `_normalise_hero_id`.
_DLTV_TO_STEAM: Optional[Dict[int, int]] = None


class v18_unavailable(RuntimeError):
    """Raised when v18 model can't be loaded or run.  Caller
    should fall back to v17."""


def _load_v18() -> Tuple[List[Any], Dict[str, Any]]:
    """Load the v18 winner model(s).  Returns a list of models
    (for ensemble averaging) plus the shared metadata.

    v0.7.72: prefer the stage-2 ensemble (scripts/train_v18_stage2.py
    writes _v18_ensemble.json describing which sub-models to load).
    Falls back to the single-model _v18_winner/ directory for
    backward compatibility.

    Returns (models_list, metadata_dict) where metadata_dict
    has the shared `feature_columns`.  The caller averages
    predict_proba across `models_list` for the final probability.
    """
    global _MODEL_CACHE, _META_CACHE
    if _MODEL_CACHE is not None and _META_CACHE is not None:
        return _MODEL_CACHE, _META_CACHE
    if joblib is None:
        raise v18_unavailable("joblib is not installed")

    # v0.7.72: try stage-2 ensemble first.  _v18_ensemble.json
    # names the chosen models with paths and weights.
    ensemble_path = MODELS_DIR / "_v18_ensemble.json"
    paths: List[Path] = []
    if ensemble_path.exists():
        try:
            ens = json.loads(ensemble_path.read_text(encoding="utf-8"))
            chosen = ens.get("chosen_models") or [ens.get("chosen", "xgb")]
            for m in chosen:
                entry = ens.get(m)
                if not entry:
                    continue
                w = entry.get("weight", 0.0)
                if w <= 0:
                    continue
                p = MODELS_DIR / entry["path"]
                if (p / "model.joblib").exists() and (p / "metadata.json").exists():
                    paths.append(p)
        except Exception:
            paths = []

    # Fallback: legacy single-model candidates.
    if not paths:
        for c in (MODELS_DIR / "_v18_winner_tuned",
                  MODELS_DIR / "_v18_winner"):
            if (c / "model.joblib").exists() and (c / "metadata.json").exists():
                paths.append(c)

    if not paths:
        raise v18_unavailable(
            f"v18 model not found at {MODELS_DIR}; run scripts/train_v18.py "
            "or scripts/train_v18_stage2.py first."
        )

    models: List[Any] = []
    meta: Optional[Dict[str, Any]] = None
    for p in paths:
        try:
            models.append(joblib.load(p / "model.joblib"))
            m = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
            if meta is None:
                meta = m
        except Exception as exc:
            raise v18_unavailable(f"failed to load {p}: {exc}") from exc
    if meta is None:
        raise v18_unavailable("no metadata found")
    _MODEL_CACHE = models
    _META_CACHE = meta
    # Eagerly build the DLTV->Steam map so the first predict
    # call doesn't pay the cost.
    _get_dltv_to_steam_map()
    return models, meta


# --------------------------------------------------------------------------- #
# Hero-id namespace translation
# --------------------------------------------------------------------------- #

def _get_dltv_to_steam_map() -> Dict[int, int]:
    """Build (lazily) and return the DLTV internal hero-id ->
    OpenDota/Steam hero-id map.

    The DLTV index has 127 heroes, of which 104 use a different
    id than Steam (mostly offset +1, but the newer heroes
    114-127 use a remap that goes up to offset +28).  The v18
    model was trained on OpenDota /matches/{id} payloads which
    use Steam ids, so the live card must convert DLTV -> Steam
    before passing to v18 -- otherwise the model treats the
    picked hero as a different one (DLTV 120 = Hoodwink but
    Steam 120 = Pangolier; the model would predict the
    Pangolier matchup when the team actually picked Hoodwink).

    The map is cached at module scope after the first build.
    """
    global _DLTV_TO_STEAM
    if _DLTV_TO_STEAM is not None:
        return _DLTV_TO_STEAM
    out: Dict[int, int] = {}
    try:
        # Local import to avoid a hard dep on dltv_client at
        # module load (the v18 trainer doesn't have dltv_client
        # available in all contexts).
        from .dltv_client import client
        heroes = client.get_heroes() or []
        for h in heroes:
            did = h.get("id")
            sid = h.get("steam_id")
            if did is None or sid is None:
                continue
            try:
                out[int(did)] = int(sid)
            except (TypeError, ValueError):
                continue
    except Exception:
        # If dltv_client isn't importable (test contexts), the
        # caller will fall through to the pass-through branch
        # in `_normalise_hero_id`.
        pass
    _DLTV_TO_STEAM = out
    return out


def _dltv_to_steam(hero_id: int) -> int:
    """Force-translate a single hero_id from DLTV internal to
    Steam (OpenDota/Valve) namespace.

    IMPORTANT: the two namespaces are ambiguous in [1..23]
    (both use the same ids for the first 23 heroes) and
    deliberately collide at higher ids too (e.g. both use
    24 for Lina, 120 for Hoodwink, but with completely
    different heroes).  Use this function ONLY when you know
    the input is a DLTV internal id.  For callers that pass
    Steam ids, use the no-op pass-through (the model expects
    Steam).

    Heroes that are not in the DLTV map (i.e. the input is
    already Steam, or the input is a new hero not yet indexed)
    pass through unchanged.
    """
    if hero_id is None:
        return 0
    try:
        hid = int(hero_id)
    except (TypeError, ValueError):
        return 0
    if hid <= 0:
        return 0
    m = _DLTV_TO_STEAM
    if m is None:
        m = _get_dltv_to_steam_map()
    return int(m.get(hid, hid))


def _load_top_teams() -> Dict[int, int]:
    """team_id -> tier (0/1/2), used to compute r_tier/d_tier and
    r_top_team/d_top_team.

    v0.7.0: the v18 snapshot has a pre-computed `tier` field
    (percentile-based, 60% minor / 30% professional / 10%
    premium).  We prefer that.  For the v17 fallback (30-team
    Glicko snapshot without `tier` field), we apply the legacy
    absolute thresholds (1400/1100).
    """
    global _TOP_TEAMS_CACHE
    if _TOP_TEAMS_CACHE is not None:
        return _TOP_TEAMS_CACHE
    out: Dict[int, int] = {}
    for path in TOP_TEAMS_PATHS:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # v0.7.0: v18 snapshot has a pre-computed `tier` field.
        if data and isinstance(data[0], dict) and "tier" in data[0]:
            for t in data:
                tid = t.get("team_id")
                if tid is None:
                    continue
                try:
                    out[int(tid)] = int(t.get("tier") or 0)
                except (TypeError, ValueError):
                    continue
            break
        # v17 snapshot: derive tier from rating with absolute
        # thresholds (legacy 30-team Glicko mean 1500 std 350).
        for t in data:
            tid = t.get("team_id")
            if tid is None:
                continue
            try:
                r = float(t.get("rating") or 0)
            except (TypeError, ValueError):
                continue
            if r >= TIER_THRESHOLD_PREMIUM:
                out[int(tid)] = 2
            elif r >= TIER_THRESHOLD_PROFESSIONAL:
                out[int(tid)] = 1
            else:
                out[int(tid)] = 0
        break
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


def _tier_for(team_id: Optional[int], top_teams: Dict[int, int]) -> int:
    if not team_id:
        return 0
    return int(top_teams.get(int(team_id), 0))


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
        # v0.7.66: v19 LITE player-level WR placeholders.  At
        # predict time we don't have the rolling-lookup
        # infrastructure, so we default to 0.5 (the neutral
        # prior that ~80% of training rows have — the min-games
        # gate is 30 so most player-hero pairs are below
        # threshold).  The model was trained with this same
        # distribution, so 0.5 at predict time is in-distribution
        # and doesn't break the model's learned splits.
        "r_player_wr_avg": 0.5,
        "d_player_wr_avg": 0.5,
        "r_player_wr_max": 0.5,
        "d_player_wr_max": 0.5,
        "player_wr_diff": 0.0,
    }
    # The v18 model was trained on OpenDota /matches/{id} payloads
    # which use Steam hero ids.  _build_features always treats
    # the input as Steam (the public predict_winner_v18 pre-
    # translates DLTV->Steam when the caller asks for that
    # namespace).  We deliberately do NOT translate again here
    # because the two namespaces are ambiguous for ids in
    # [1..23] (both use the same ids for the first 23 heroes)
    # and for some specific ids in [24..127] (e.g. 25 = Lion
    # in DLTV but 25 = Lina in Steam).  Re-translating an
    # already-Steam id would corrupt the feature vector.
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
    hero_id_namespace: str = "steam",
) -> Dict[str, Any]:
    """Run the v18 winner model.  Returns the same dict shape
    as v17_predict.predict()'s "winner" sub-dict, plus a
    "source" field for downstream observability.

    `hero_id_namespace` (v0.7.2): the namespace of the incoming
    hero ids.  Must be one of:
      - "steam"  : OpenDota / Valve / Steam ids (default).
                   Pass this for watchlist JSON, OpenDota
                   payloads, v18 trainer data, anything that's
                   already in the OpenDota namespace.
      - "dltv"   : DLTV internal ids.  v18 will force-translate
                   to Steam via the dltv_client map.  Use this
                   for v1 API maps[].picks, the dltv_browser
                   cache overlay, and any other source that
                   exposes DLTV internal ids.
    The two namespaces are NOT identical (104 of 127 DLTV
    heroes use a different id than Steam), so getting this
    wrong biases the model toward the wrong hero.

    Raises v18_unavailable if the model isn't installed (caller
    falls back to v17).
    """
    models, meta = _load_v18()
    feat_names = meta["feature_columns"]
    # v0.7.2: pre-translate the picks/bans if the caller said
    # they're DLTV.  We do this once here so _build_features
    # can stay Steam-only.
    if hero_id_namespace == "dltv":
        radiant_picks = [_dltv_to_steam(int(h)) for h in (radiant_picks or [])]
        dire_picks = [_dltv_to_steam(int(h)) for h in (dire_picks or [])]
        if radiant_bans:
            radiant_bans = [_dltv_to_steam(int(h)) for h in radiant_bans]
        if dire_bans:
            dire_bans = [_dltv_to_steam(int(h)) for h in dire_bans]
    elif hero_id_namespace != "steam":
        raise ValueError(
            f"hero_id_namespace must be 'steam' or 'dltv', got {hero_id_namespace!r}"
        )
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
    # v0.7.72: average predict_proba across the loaded ensemble.
    # Each sub-model (XGB / LGB / HGB) sees the same feature row
    # and produces a probability; we soft-vote them.
    try:
        probas = []
        for m in models:
            p = m.predict_proba(X)
            probas.append(float(p[0, 1]) if p.shape[1] > 1 else 0.5)
        prob_radiant = sum(probas) / len(probas)
        if len(models) > 1:
            src = f"v18-ensemble({len(models)})"
        else:
            src = "v18"
    except Exception as exc:
        raise v18_unavailable(f"v18 predict_proba failed: {exc}") from exc
    if not (0.0 <= prob_radiant <= 1.0):
        prob_radiant = 0.5
    winner_team = "radiant" if prob_radiant >= 0.5 else "dire"
    return {
        "team": winner_team,
        "prob_radiant": prob_radiant,
        "probability": max(prob_radiant, 1.0 - prob_radiant),
        "source": src,
    }


def is_available() -> bool:
    """Quick check: are the v18 model files present?"""
    try:
        _load_v18()
        return True
    except v18_unavailable:
        return False
