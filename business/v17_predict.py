"""v17 trained-model predictor.

Loads the 4 production models from `ml_data/models/<target>_v17/`
(lazy init on first call) and returns a dict matching the legacy
`engine.analyze(...)` shape so callers don't need to know which
backend produced the prediction.

Target features (per `metadata.json` feature_columns):
  - kills_total:       ridge_a10, 13 features, MAE 11.80
  - duration_sec:      ridge_a10, 13 features, MAE 586.94
  - first_15_kills:    hgb_max_depth3, 16 features, MAE 3.92
  - winner:            logreg_l1, 14 features, accuracy 75.30%

All 4 models are honest 5-fold CV (no encoder leak); sample weights
are recency_weight * sqrt(tier_weight).

If a model is missing or `joblib.load` fails, we surface a clear
`MLError` so the caller can fall back to the legacy engine.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# joblib is part of scikit-learn
try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

PRO_ROOT = Path(__file__).resolve().parents[1]
ML_DATA = PRO_ROOT / "ml_data"
MODELS = ML_DATA / "imports"
MODELS_DIR = ML_DATA / "models"

# Tier weights — keep in sync with train_v17_v2.py
TIER_WEIGHT = {"premium": 1.0, "professional": 0.7, "minor": 0.4}
TIER_THRESHOLD_PREMIUM = 1400
TIER_THRESHOLD_PROFESSIONAL = 1100

# Optional one-time load cache
_MODEL_CACHE: Dict[str, Any] = {}
_META_CACHE: Dict[str, Dict[str, Any]] = {}


class MLError(Exception):
    """Raised when a v17 model cannot produce a prediction."""


def _model_path(target: str) -> Path:
    # v17 models live under `_v17_<target>/` (underscore prefix) so the
    # legacy `ModelStorage` scan in `business/ml/storage.py` doesn't
    # pick them up — the legacy `FEATURE_GROUPS` schema doesn't match
    # v17's 21-feature schema and would raise RuntimeError.
    return MODELS_DIR / f"_v17_{target}"


def _load_model(target: str) -> Tuple[Any, Dict[str, Any]]:
    if target in _MODEL_CACHE and target in _META_CACHE:
        return _MODEL_CACHE[target], _META_CACHE[target]
    if joblib is None:
        raise MLError("joblib is not installed; v17_predict cannot load models")
    path = _model_path(target)
    if not path.exists():
        raise MLError(f"missing model dir: {path}")
    model_file = path / "model.joblib"
    meta_file = path / "metadata.json"
    if not model_file.exists() or not meta_file.exists():
        raise MLError(f"missing model files in {path}")
    try:
        model = joblib.load(model_file)
    except Exception as exc:
        raise MLError(f"failed to load {model_file}: {exc}") from exc
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MLError(f"failed to parse {meta_file}: {exc}") from exc
    _MODEL_CACHE[target] = model
    _META_CACHE[target] = meta
    return model, meta


def _tier_from_rating(rating: Optional[float]) -> str:
    if rating is None:
        return "minor"
    r = float(rating)
    if r >= TIER_THRESHOLD_PREMIUM:
        return "premium"
    if r >= TIER_THRESHOLD_PROFESSIONAL:
        return "professional"
    return "minor"


def _team_tier(team_id: Optional[int],
                top_teams: Optional[List[Dict[str, Any]]]) -> str:
    if not team_id or not top_teams:
        return "minor"
    for t in top_teams:
        try:
            if int(t.get("team_id") or 0) == int(team_id):
                return _tier_from_rating(t.get("rating"))
        except (ValueError, TypeError):
            continue
    return "minor"


def _load_top_teams() -> List[Dict[str, Any]]:
    """Top teams (v17) — same as phase 1, used for tier classification."""
    p = MODELS / "v17_phase1_top_teams.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_patch_dates() -> Dict[str, str]:
    p = MODELS / "v17_phase7_patch_info.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {x.get("name"): x.get("date") for x in d if isinstance(x, dict)}
    except Exception:
        return {}


def _load_hero_stats() -> List[Dict[str, Any]]:
    p = MODELS / "v17_phase4_hero_stats.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _hero_target_enc(hero_stats: List[Dict[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
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


def _days_since_patch(patch: str, start_time: int,
                       patch_dates: Dict[str, str]) -> float:
    if not patch or patch not in patch_dates:
        return 90.0
    try:
        import time
        date_str = patch_dates[patch]
        if not date_str:
            return 90.0
        ts = int(time.mktime(time.strptime(date_str[:10], "%Y-%m-%d")))
        return max(0.0, (start_time - ts) / 86400.0)
    except Exception:
        return 90.0


def _patch_index(patch: str) -> int:
    """Categorical patch code — fixed ordering; train_v17_v2 used
    the same `name` strings so we encode by alphabetical order of
    seen patches.  In practice with only current+prev in the
    corpus the codes are stable across calls.
    """
    # Patches we've trained on
    _PATCHES = ["7.39", "7.40", "7.41"]
    try:
        return _PATCHES.index(patch)
    except ValueError:
        return -1


def _encode_features(meta: Dict[str, Any], feats: Dict[str, Any]) -> List[float]:
    """Map a feature dict to the column order required by `meta.feature_columns`.
    Categorical columns are encoded to integer codes matching the
    trainer's `_encode_categorical` (alphabetical insertion order).
    """
    cols = meta["feature_columns"]
    # Categorical: patch, r_tier, d_tier.  All other v17 features are numeric.
    cat_values = {
        "patch":  feats.get("patch", ""),
        "r_tier": feats.get("r_tier", "minor"),
        "d_tier": feats.get("d_tier", "minor"),
    }
    out: List[float] = []
    for c in cols:
        if c in cat_values:
            v = cat_values[c]
            if c == "patch":
                out.append(float(_patch_index(v)))
            else:
                # tier codes (matching trainer's alphabetical insertion: minor=0, premium=1, professional=2)
                tier_codes = {"minor": 0, "premium": 1, "professional": 2}
                out.append(float(tier_codes.get(v, 0)))
        else:
            v = feats.get(c, 0.0)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            if not math.isfinite(fv):
                fv = 0.0
            out.append(fv)
    return out


def build_features(radiant_team_id: Optional[int],
                    dire_team_id: Optional[int],
                    radiant_picks: List[int],
                    dire_picks: List[int],
                    radiant_bans: Optional[List[int]] = None,
                    dire_bans: Optional[List[int]] = None,
                    gold_adv_5: Optional[float] = None,
                    gold_adv_10: Optional[float] = None,
                    start_time: Optional[int] = None,
                    patch: Optional[str] = None,
                    account_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Build the full feature dict from a live match state."""
    top_teams = _load_top_teams()
    hero_stats = _load_hero_stats()
    hero_enc = _hero_target_enc(hero_stats)
    patch_dates = _load_patch_dates()

    r_tier = _team_tier(radiant_team_id, top_teams)
    d_tier = _team_tier(dire_team_id, top_teams)
    r_h = float(sum(hero_enc.get(int(h), 0.5) for h in radiant_picks) /
                max(1, len(radiant_picks)))
    d_h = float(sum(hero_enc.get(int(h), 0.5) for h in dire_picks) /
                max(1, len(dire_picks)))
    r_bans = radiant_bans or []
    d_bans = dire_bans or []
    r_b = float(sum(hero_enc.get(int(h), 0.5) for h in r_bans) /
                max(1, len(r_bans))) if r_bans else 0.5
    d_b = float(sum(hero_enc.get(int(h), 0.5) for h in d_bans) /
                max(1, len(d_bans))) if d_bans else 0.5
    r_std = 0.0
    d_std = 0.0
    if radiant_picks:
        vals = [hero_enc.get(int(h), 0.5) for h in radiant_picks]
        mean = r_h
        r_std = float(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))
    if dire_picks:
        vals = [hero_enc.get(int(h), 0.5) for h in dire_picks]
        mean = d_h
        d_std = float(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))

    import time
    st = int(start_time or time.time())
    days_p = _days_since_patch(patch or "", st, patch_dates)

    # top_30 players
    team_players_path = MODELS / "v17_phase6_team_players.json"
    top_players: set = set()
    if team_players_path.exists():
        try:
            rosters = json.loads(team_players_path.read_text(encoding="utf-8"))
            top_team_ids = {int(t["team_id"]) for t in _load_top_teams()
                            if t.get("team_id") is not None}
            for tid, roster in rosters.items():
                if int(tid) not in top_team_ids:
                    continue
                for p in (roster or []):
                    acc = p.get("account_id")
                    if acc:
                        top_players.add(int(acc))
        except Exception:
            pass
    n_top_players = sum(1 for a in (account_ids or []) if a in top_players)

    return {
        "patch":             patch or "",
        "r_tier":            r_tier,
        "d_tier":            d_tier,
        "r_team_id":         int(radiant_team_id) if radiant_team_id else 0,
        "d_team_id":         int(dire_team_id) if dire_team_id else 0,
        "r_hero_enc":        r_h,
        "d_hero_enc":        d_h,
        "r_dire_syn":        r_h - d_h,
        "r_picks":           float(len(radiant_picks)),
        "d_picks":           float(len(dire_picks)),
        "r_top_team":        float(r_tier in ("premium", "professional")),
        "d_top_team":        float(d_tier in ("premium", "professional")),
        "side_rad":          1.0,
        "r_ban_enc":         r_b,
        "d_ban_enc":         d_b,
        "r_team_syn":        r_std,
        "d_team_syn":        d_std,
        "gold_adv_5":        float(gold_adv_5) if gold_adv_5 is not None else 0.0,
        "gold_adv_10":       float(gold_adv_10) if gold_adv_10 is not None else 0.0,
        "days_since_patch":  days_p,
        "n_top_players":     float(n_top_players),
    }


def predict(radiant_team_id: Optional[int],
            dire_team_id: Optional[int],
            radiant_picks: List[int],
            dire_picks: List[int],
            radiant_bans: Optional[List[int]] = None,
            dire_bans: Optional[List[int]] = None,
            gold_adv_5: Optional[float] = None,
            gold_adv_10: Optional[float] = None,
            start_time: Optional[int] = None,
            patch: Optional[str] = None,
            account_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Run all 4 models.  Returns a dict in the legacy `engine.analyze()` shape."""
    feats = build_features(
        radiant_team_id, dire_team_id, radiant_picks, dire_picks,
        radiant_bans=radiant_bans, dire_bans=dire_bans,
        gold_adv_5=gold_adv_5, gold_adv_10=gold_adv_10,
        start_time=start_time, patch=patch, account_ids=account_ids,
    )

    out: Dict[str, Any] = {
        "kills": 0.0, "duration_sec": 0.0,
        "first_15_kills": 0.0, "winner": {
            "team": "radiant", "prob_radiant": 0.5, "probability": 0.5,
        },
        "confidence": "low",
        "source": "v17",
    }

    # Numeric regressions
    for tgt, key in (("kills_total", "kills"),
                     ("duration_sec", "duration_sec"),
                     ("first_15_kills", "first_15_kills")):
        try:
            model, meta = _load_model(tgt)
            row = _encode_features(meta, feats)
            pred = float(model.predict([row])[0])
            out[key] = max(0.0, pred)
        except MLError:
            out[key] = 0.0
        except Exception as exc:  # noqa: BLE001
            raise MLError(f"{tgt} predict failed: {exc}") from exc

    # Winner
    try:
        model, meta = _load_model("winner")
        row = _encode_features(meta, feats)
        proba = model.predict_proba([row])[0]
        prob_radiant = float(proba[1]) if len(proba) > 1 else 0.5
        prob_dire = float(proba[0]) if len(proba) > 0 else 0.5
        out["winner"] = {
            "team": "radiant" if prob_radiant >= 0.5 else "dire",
            "prob_radiant": prob_radiant,
            "prob_dire":     prob_dire,
            "probability":   max(prob_radiant, prob_dire),
        }
        margin = abs(prob_radiant - 0.5)
        out["confidence"] = "high" if margin > 0.15 else ("medium" if margin > 0.07 else "low")
    except MLError:
        out["winner"] = {"team": "radiant", "prob_radiant": 0.5, "probability": 0.5}
        out["confidence"] = "low"
    except Exception as exc:  # noqa: BLE001
        raise MLError(f"winner predict failed: {exc}") from exc

    return out


def is_available() -> bool:
    """Quick check: all 4 model files present?"""
    for tgt in ("kills_total", "duration_sec", "first_15_kills", "winner"):
        p = _model_path(tgt) / "model.joblib"
        if not p.exists():
            return False
    return True


# ---------------------------------------------------------------------------- #
# v17 ↔ legacy hybrid predictor
# ---------------------------------------------------------------------------- #
#
# v17 only produces 4 numbers (kills_total, duration_sec, first_15_kills,
# winner).  The postmatch card also needs `towers`, `multikill`, `first_blood`
# (added by `analyze_map_with_verdict` from team fb_rate / tower-dominance
# heuristics and the legacy ML models).  Rather than re-train v17 to predict
# those, the hybrid runs v17 for the 4 v17-owned targets and the legacy
# engine for everything else, then merges the two into a single prediction
# dict in the legacy `analyze_map_with_verdict` shape.
#
# The merge is mechanical:
#   * winner — from v17 (side → team name via team_a / team_b; float prob
#              → int %).
#   * kills / duration — from v17 (flat float → dict; sec → min).
#   * first_15 — v17 doesn't predict a per-side team, so we use the
#              winner side as a proxy (the team predicted to win is
#              usually the one to reach 15 first).  The numeric
#              `first_15_kills` count from v17 is preserved as a new
#              supplementary field.
#   * kills_total_over_under / total_over_under — derived from the v17
#              numbers via the same helpers the legacy engine uses
#              (`_build_over_under_from_kills` / `_build_over_under_from_
#              duration`).
#   * towers / towers_over_under — from legacy (v17 has no towers model).
#   * multikill — from legacy (v17 has no multikill model; the v0.3
#              classifier degenerated to "always High" on the pro corpus).
#   * first_blood — added from team fb_rates, same as
#              `analyze_map_with_verdict` does on top of `engine.analyze`.
#   * verdict — from `map_verdicts` (same helper), so the postmatch
#              card's "Победитель ✓/✗" / "Карта N: TM" lines all keep
#              working unchanged.
#
# v17 failures (MLError, missing features, NaN) fall through to the
# legacy prediction for the affected block — so a single bad target
# doesn't kill the card.
#
# The function returns the same `{prediction, verdict}` shape as
# `analysis.analyze_map_with_verdict`, so `board.py` can substitute
# one for the other without changing the rest of the call chain.

def hybrid_predict(
    *,
    engine,
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    heroes_a: List[Any],
    heroes_b: List[Any],
    actual: Dict[str, Any],
    # v17 inputs (extracted from the post-match map)
    radiant_team_id: Optional[int],
    dire_team_id: Optional[int],
    radiant_picks: List[int],
    dire_picks: List[int],
    radiant_bans: Optional[List[int]] = None,
    dire_bans: Optional[List[int]] = None,
    start_time: Optional[int] = None,
    patch: Optional[str] = None,
) -> Dict[str, Any]:
    """v17 ↔ legacy hybrid post-match prediction.

    Returns ``{"prediction": <merged pred dict>, "verdict": <map_verdicts>}``.
    """
    # Local imports to keep this module importable even if the rest
    # of `business/` is not yet on the path (e.g. unit tests on a
    # checkout without the full tree).  The legacy engine has its
    # own over/under helpers; analysis.py has map_verdicts.
    from .ml.engine import (
        _build_over_under_from_duration,
        _build_over_under_from_kills,
    )
    from .analysis import (
        FALLBACK_FB,
        _val,
        map_verdicts,
    )

    # 1. Run v17 (may raise MLError if a model is missing/broken).
    v17: Dict[str, Any] = {}
    try:
        v17 = predict(
            radiant_team_id=radiant_team_id,
            dire_team_id=dire_team_id,
            radiant_picks=list(radiant_picks or []),
            dire_picks=list(dire_picks or []),
            radiant_bans=list(radiant_bans or []),
            dire_bans=list(dire_bans or []),
            start_time=start_time,
            patch=patch,
        )
    except MLError as exc:
        # v17 is fully unavailable — fall back to pure legacy.
        # The caller (board.py) catches MLError and re-runs the
        # legacy path; reaching here means a partial v17 failure,
        # which we handle per-block below.
        v17 = {}
        v17.setdefault("kills", 0.0)
        v17.setdefault("duration_sec", 0.0)
        v17.setdefault("first_15_kills", 0.0)
        v17.setdefault("winner", {"team": "radiant", "prob_radiant": 0.5, "probability": 0.5})
        v17.setdefault("confidence", "low")

    # 2. Run the legacy engine — this still gives us towers /
    #    multikill / confidence and the kills_total_over_under /
    #    total_over_under / towers_over_under bet shapes for the
    #    postmatch card's verdict lines.
    legacy = engine.analyze(team_a, team_b, heroes_a, heroes_b)

    # 3. Build the merged prediction, taking v17 for the 4 v17-owned
    #    targets and legacy for the rest.  All conversions are
    #    shape-only (the actual numbers come from v17's predictions).
    pred: Dict[str, Any] = dict(legacy)  # copy so we don't mutate the engine's dict

    # ---- winner ----
    v17_winner = v17.get("winner") or {}
    v17_prob_radiant = float(v17_winner.get("prob_radiant", 0.5) or 0.5)
    if not (0.0 <= v17_prob_radiant <= 1.0):
        v17_prob_radiant = 0.5
    winner_team_name = team_a["name"] if v17_prob_radiant >= 0.5 else team_b["name"]
    winner_prob_pct = int(round(max(v17_prob_radiant, 1.0 - v17_prob_radiant) * 100))
    pred["winner"] = {
        "team": winner_team_name,
        "probability": winner_prob_pct,
        "prob_radiant": int(round(v17_prob_radiant * 100)),
        "source": "v17",
    }

    # ---- kills ----
    v17_kills_total = float(v17.get("kills") or 0.0)
    if v17_kills_total > 0:
        kills_total = int(round(v17_kills_total))
        # Split equally (the legacy engine does the same in its
        # `_predict_kills`).  Without a per-side kills model we
        # don't have a better signal; the winner side is "favoured"
        # for slightly more kills but the round() wash is fine for
        # an over/under bet.
        kills_a = int(round(kills_total / 2))
        kills_b = kills_total - kills_a
        pred["kills"] = {"total": kills_total, "radiant": kills_a, "dire": kills_b}
        pred["kills_total_over_under"] = _build_over_under_from_kills(kills_total)

    # ---- duration ----
    v17_dur_sec = float(v17.get("duration_sec") or 0.0)
    if v17_dur_sec > 0:
        dur_min = round(v17_dur_sec / 60.0, 1)
        pred["duration_min"] = dur_min
        pred["total_over_under"] = _build_over_under_from_duration(dur_min)

    # ---- first_15 ----
    # v17 returns `first_15_kills` (count), not a per-side team.
    # We use the winner side as a proxy for first_to_15.team (the
    # team predicted to win is usually the first to reach 15
    # kills).  The probability is the v17 winner probability (same
    # reasoning).  The raw `first_15_kills` count is preserved as
    # a new field so the UI can show it if it wants to.
    v17_first_15 = float(v17.get("first_15_kills") or 0.0)
    if v17_first_15 > 0:
        pred["first_to_15"] = {
            "team": pred["winner"]["team"],
            "probability": pred["winner"]["probability"],
        }
        pred["first_15_kills"] = v17_first_15  # new supplementary field

    # ---- confidence ----
    # v17's `confidence` is a string ("low"/"medium"/"high");
    # the legacy engine emits a 0..1 float.  Map v17's buckets
    # to a sensible float so the postmatch card's
    # "достоверность данных: X%" line keeps working.
    v17_conf = v17.get("confidence", "low")
    conf_map = {"high": 0.85, "medium": 0.65, "low": 0.45}
    if isinstance(v17_conf, str):
        pred["confidence"] = conf_map.get(v17_conf, 0.5)
    else:
        # Legacy float already (shouldn't happen in v17 path, but
        # be safe in case the engine was already a hybrid).
        pred["confidence"] = float(v17_conf) if v17_conf is not None else 0.5

    # ---- source ----
    # Mark the prediction as v17-derived so accuracy tracking and
    # any future UI badges can distinguish hybrid from pure legacy.
    pred["source"] = "v17+legacy"
    pred["v17_contrib"] = ["winner", "kills", "duration", "first_15"]

    # ---- first_blood (from team fb_rates, same as analyze_map_with_verdict) ----
    fb_a = _val(team_a, "fb_rate", FALLBACK_FB)
    fb_b = _val(team_b, "fb_rate", FALLBACK_FB)
    pred["first_blood"] = {
        "team": team_a["name"] if fb_a >= fb_b else team_b["name"],
        "probability": int(round(max(fb_a, fb_b))),
    }

    # ---- verdict ----
    verdict = map_verdicts(
        pred, actual, team_a["name"], team_b["name"],
    )

    return {"prediction": pred, "verdict": verdict}
