"""
Prediction engine Strategy.

`IPredictionEngine` is the abstract contract every prediction backend
must satisfy. The two concrete implementations are:

  - `HeuristicEngine` — a thin wrapper around the existing
    `business.analysis.analyze()` function. The output is byte-for-byte
    identical to what the rest of the system has been emitting since
    v0.0.0, so it's the safe default.

  - `MLEngine` — loads one or more trained sub-models from
    `ModelStorage` (one per target: `winner`, `kills`, `duration_mean`,
    `duration_p10`, `duration_p90`) and overrides the matching blocks
    in the heuristic result.  Blocks whose sub-model is missing
    stay heuristic — so a half-trained `MLEngine` is still useful,
    it just doesn't predict what it doesn't know.

The two are interchangeable from `board.py`'s point of view: same
input, same output dict shape.  Selection is made once at process
start through `make_engine()` driven by the `PREDICTION_ENGINE` env
var.

Feature groups at predict time (0.3.10)
---------------------------------------
The trained model may include features from three groups:
`hero` (always available at predict time), `team` (depends on
`team_a`/`team_b` having a stable id) and `lane` (depends on
the per-hero lane role being known).  At predict time the engine
passes whatever it has — for `lane` it passes an empty lane dict
(decoder returns `global_rate` for every missing cell, which is
the neutral 0.5 prior).  The same model is therefore usable both
at train time (with full lane info) and predict time (without).

Why per-target sub-models?
---------------------------
0.2.0 shipped a winner-only MLEngine.  0.2.1 extends it to four
regression targets (kills, duration mean, P10, P90) plus the
classifier.  Training each head independently:

  - lets us swap the loss (Poisson for kills, Tweedie for duration)
    without retraining the others;
  - lets a model be "promoted" independently — promote the kills
    regressor without touching the winner classifier;
  - keeps each training run under a few seconds, which matters
    for the eval harness that needs to retrain often.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .._logging import get_logger
from .features import (
    LANE_KEYS,
    HeroWinRateEncoder,
    extract_features,
    feature_names,
)
from .storage import LoadedModel, ModelStorage

log = get_logger(__name__)


# Sub-models we know how to use.  Order matters: the engine checks
# them in this order so that a more-specific head (e.g. duration_p10)
# shadows a less-specific one (e.g. duration_mean) when both are
# present.  In practice the trainer saves both, and we use both.
KNOWN_TARGETS: tuple = (
    "winner",
    "kills",
    "duration_mean",
    "duration_p10",
    "duration_p90",
    "towers",     # 0.2.2 — disabled until tower data is available
    # NOTE (0.3.10): "multikill" was removed.  The pro-only corpus
    # has 100 % High matches (every pro player rampages), so the
    # classifier degenerated to "always High" in 0.3.0 and was
    # never useful.  The bins are still used by `analysis.analyze`
    # for the heuristic, but no trained model is loaded.  See
    # `train.HEAD_REGISTRY` for the full rationale.
)


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def _hero_ids_from_metas(heroes: List[Optional[Dict]]) -> List[Optional[int]]:
    """Pull a Valve hero id from each hero-meta dict.

    DLTV metadata has both `id` (DLTV internal) and `steam_id` (the
    Valve hero id we trained on).  We prefer `steam_id` because
    that matches the `valve_id` we use during training (DatDota
    full_matches call it the same thing).

    Returns one id per element in `heroes` (None if missing).
    Order is preserved — important because `extract_features`
    treats the first five as a positional list.
    """
    out: List[Optional[int]] = []
    for h in heroes:
        if not h:
            out.append(None)
            continue
        sid = h.get("steam_id")
        if isinstance(sid, int):
            out.append(sid)
            continue
        out.append(None)
    return out


def _valid_hero_id_list(
    radiant_metas: List[Optional[Dict]],
    dire_metas: List[Optional[Dict]],
) -> Optional[tuple]:
    """Return (radiant_ids, dire_ids) only if every position has an id.

    Returns None if any hero is missing a Valve id, which signals
    the engine to fall back to the heuristic.  We require all 5+5
    to be resolvable because `extract_features` is positional.
    """
    r = _hero_ids_from_metas(radiant_metas)
    d = _hero_ids_from_metas(dire_metas)
    if any(x is None for x in r) or any(x is None for x in d):
        return None
    if len(r) != 5 or len(d) != 5:
        return None
    return (r, d)  # type: ignore[return-value]


def _team_id(team: Optional[Dict[str, Any]]) -> Optional[int]:
    """Pull a stable team id from a team dict, or None.

    Sources in priority order:
      - `team["team_id"]`  (int) — set by `eval_engines.synth_team` for tests
      - `team["valve_id"]` (int) — set by DLTV team metadata
    Both spellings are accepted because we don't have a single
    producer for `team` objects yet; once we do, the loser can be
    deleted.

    Used by the `team` feature group (0.3.10 C retry).  When the
    team id is missing, the encoder returns its `global_rate`
    (0.5 prior) for that side — no NaNs leak into the feature
    vector.
    """
    if not team:
        return None
    for key in ("team_id", "valve_id"):
        v = team.get(key)
        if isinstance(v, int):
            return int(v)
    return None


def _empty_lane_dict() -> Dict[str, Optional[int]]:
    """A lane dict with all roles set to None.

    Used as a fallback at predict time when the upstream match
    draft doesn't carry per-hero lane assignments.  The lane
    encoder returns `global_rate` for any pair with a missing
    hero, so the resulting feature values are the neutral 0.5
    prior — well-defined and bias-free.
    """
    return {k: None for k in LANE_KEYS}


def _model_feature_groups(loaded: LoadedModel) -> Tuple[str, ...]:
    """Return the feature groups a model was trained on.

    Reads the `feature_groups` field saved into the model
    metadata (0.3.10+).  Falls back to the 0.3.9 default
    `("hero",)` for older models that predate this field.
    """
    td = loaded.metadata.train_data or {}
    groups = td.get("feature_groups")
    if groups:
        return tuple(groups)
    return ("hero",)


def _predict_features(
    loaded: LoadedModel,
    radiant_hero_ids: List[int],
    dire_hero_ids: List[int],
    *,
    radiant_team_id: Optional[int] = None,
    dire_team_id: Optional[int] = None,
    radiant_lane: Optional[Dict[str, Optional[int]]] = None,
    dire_lane: Optional[Dict[str, Optional[int]]] = None,
) -> List[float]:
    """Build the feature vector for `loaded` at predict time.

    Always uses the same `groups` the model was trained on so
    the column order matches what the classifier / regressor
    saw in train.  If a group requires data the engine doesn't
    have (e.g. `lane` at predict time), pass a fallback dict —
    the encoder's per-cell `global_rate` is the neutral prior.
    """
    groups = _model_feature_groups(loaded)
    return extract_features(
        radiant_hero_ids, dire_hero_ids, loaded.encoder,
        radiant_team_id=radiant_team_id,
        dire_team_id=dire_team_id,
        radiant_lane=radiant_lane,
        dire_lane=dire_lane,
        groups=groups,
    )


def _build_over_under_from_duration(
    duration_min: float,
    threshold_high: float = 40.0,
    offset: int = 1,
) -> Dict[str, Any]:
    """Replicate the heuristic's duration over/under bet format.

    Long game (>= threshold_high) → bet under at the predicted
    duration.  Short game → bet over at predicted + offset.
    """
    if duration_min >= threshold_high:
        side = "under"
        threshold = int(round(duration_min))
    else:
        side = "over"
        threshold = int(round(duration_min)) + offset
    pred_dur_int = int(round(duration_min))
    return {
        "side": side,
        "threshold": threshold,
        "formatted": f"{pred_dur_int // 60}:{str(pred_dur_int % 60).zfill(2)}",
    }


def _build_over_under_from_kills(total: int, threshold_high: int = 50, offset: int = 1) -> Dict[str, Any]:
    """Replicate the heuristic's kills over/under bet format."""
    if total >= threshold_high:
        return {"side": "under", "threshold": int(round(total))}
    return {"side": "over", "threshold": int(round(total)) + offset}


# ---------------------------------------------------------------------------- #
# Abstract base
# ---------------------------------------------------------------------------- #

class IPredictionEngine(ABC):
    """Strategy contract for prediction backends.

    `analyze()` MUST return a dict with the same shape as
    `business.analysis.analyze()`.  Downstream code in `board.py`
    assumes this shape and does not know which engine produced it.
    """

    #: Short, human-readable name.  Used in logs and the `/api/board`
    #: response so we can tell which engine produced a given
    #: prediction.
    name: str = "abstract"

    @abstractmethod
    def analyze(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Run the full multi-metric prediction.  Returns a `dict`."""


# ---------------------------------------------------------------------------- #
# Heuristic engine — wraps analysis.analyze()
# ---------------------------------------------------------------------------- #

class HeuristicEngine(IPredictionEngine):
    """Pure heuristic.  This is the 0.0.x behaviour preserved 1:1."""

    name = "heuristic"

    def __init__(self) -> None:
        # Local import to keep this module importable even if the
        # rest of `business/` is not yet on the path (e.g. unit
        # tests on a checkout without the full tree).
        from ..analysis import analyze
        self._analyze = analyze

    def analyze(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return self._analyze(team_a, team_b, heroes_a, heroes_b)


# ---------------------------------------------------------------------------- #
# ML engine — multi-target overrides
# ---------------------------------------------------------------------------- #

class MLEngine(IPredictionEngine):
    """Loads one or more sub-models and overrides the matching blocks.

    A sub-model is a `LoadedModel` from `ModelStorage`.  The engine
    knows how to use any of:

      - `winner`            — binary classifier; overrides `result["winner"]`
      - `kills`             — count regressor; overrides
                              `result["kills"]["total"]` /
                              `["kills"]["radiant"]` /
                              `["kills"]["dire"]` and
                              `result["kills_total_over_under"]`
      - `duration_mean`     — duration regressor in minutes; overrides
                              `result["duration_min"]` and
                              `result["total_over_under"]`
      - `duration_p10`/`p90` — quantile heads (used to widen the
                              over/under bet when both are present)
      - `towers`            — 0.2.2 (deferred)

    Sub-models that are missing or that fail to predict are silently
    skipped — the corresponding block stays heuristic.  This keeps
    a half-trained `MLEngine` useful and makes upgrades safe.
    """

    name = "ml"

    def __init__(
        self,
        sub_models: Dict[str, LoadedModel],
        fallback: Optional[IPredictionEngine] = None,
    ) -> None:
        if not sub_models:
            raise ValueError("MLEngine requires at least one sub-model")
        self._sub_models: Dict[str, LoadedModel] = dict(sub_models)
        self._fallback = fallback or HeuristicEngine()
        self._versions: Dict[str, str] = {
            k: v.version for k, v in sub_models.items()
        }

    # ---- per-block predictors ---------------------------------------- #

    def _predict_winner(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        loaded = self._sub_models.get("winner")
        if loaded is None:
            return None
        ids = _valid_hero_id_list(heroes_a, heroes_b)
        if ids is None:
            return None
        r_ids, d_ids = ids
        feats = _predict_features(
            loaded, r_ids, d_ids,
            radiant_team_id=_team_id(team_a),
            dire_team_id=_team_id(team_b),
            radiant_lane=_empty_lane_dict(),
            dire_lane=_empty_lane_dict(),
        )
        # predict_proba returns [[p_dire, p_radiant]] for a binary
        # classifier trained with class 1 == radiant win
        # (see features.py + train.py).
        proba = loaded.model.predict_proba([feats])
        p_radiant = float(proba[0][1])
        return {
            "team": team_a["name"] if p_radiant >= 0.5 else team_b["name"],
            "probability": int(round(max(p_radiant, 1 - p_radiant) * 100)),
            "prob_radiant": int(round(p_radiant * 100)),
            "source": f"ml:{loaded.version}",
        }

    def _predict_kills(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        loaded = self._sub_models.get("kills")
        if loaded is None:
            return None
        ids = _valid_hero_id_list(heroes_a, heroes_b)
        if ids is None:
            return None
        r_ids, d_ids = ids
        feats = _predict_features(
            loaded, r_ids, d_ids,
            radiant_team_id=_team_id(team_a),
            dire_team_id=_team_id(team_b),
            radiant_lane=_empty_lane_dict(),
            dire_lane=_empty_lane_dict(),
        )
        total = float(loaded.model.predict([feats])[0])
        total = max(0.0, total)
        return {
            "total": int(round(total)),
            "radiant": int(round(total / 2)),
            "dire": int(round(total / 2)),
            "source": f"ml:{loaded.version}",
        }

    def _predict_duration(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Return (mean_min, p10, p90) — any may be None.

        `p10`/`p90` are only returned if the corresponding quantile
        sub-models are loaded.  `mean_min` uses `duration_mean`.
        """
        ids = _valid_hero_id_list(heroes_a, heroes_b)
        if ids is None:
            return None
        r_ids, d_ids = ids

        mean_loaded = self._sub_models.get("duration_mean")
        p10_loaded = self._sub_models.get("duration_p10")
        p90_loaded = self._sub_models.get("duration_p90")

        # We need at least the mean to return anything.  Quantiles
        # are bonus — when both quantiles are present the engine
        # uses the wider spread to choose the bet side.
        if mean_loaded is None and p10_loaded is None and p90_loaded is None:
            return None

        def _predict_with(loaded: LoadedModel) -> float:
            feats = _predict_features(
                loaded, r_ids, d_ids,
                radiant_team_id=_team_id(team_a),
                dire_team_id=_team_id(team_b),
                radiant_lane=_empty_lane_dict(),
                dire_lane=_empty_lane_dict(),
            )
            return float(loaded.model.predict([feats])[0])

        mean = _predict_with(mean_loaded) if mean_loaded is not None else None
        p10 = _predict_with(p10_loaded) if p10_loaded is not None else None
        p90 = _predict_with(p90_loaded) if p90_loaded is not None else None
        return {"mean": mean, "p10": p10, "p90": p90}

    def _predict_towers(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Return a `towers` block, or None if prediction isn't possible.

        Splitting the prediction into radiant/dire 50/50 is a
        placeholder — the 0.2.1 heuristic uses the dominance
        factor (abs(p_a - 0.5) * 2) to bias the split, but the
        trained regressor only knows the *total*.  Splitting is
        left to the engine so we can swap in a per-side model
        later without touching storage.
        """
        loaded = self._sub_models.get("towers")
        if loaded is None:
            return None
        ids = _valid_hero_id_list(heroes_a, heroes_b)
        if ids is None:
            return None
        r_ids, d_ids = ids
        feats = _predict_features(
            loaded, r_ids, d_ids,
            radiant_team_id=_team_id(team_a),
            dire_team_id=_team_id(team_b),
            radiant_lane=_empty_lane_dict(),
            dire_lane=_empty_lane_dict(),
        )
        total = max(0, int(round(float(loaded.model.predict([feats])[0]))))
        return {
            "total": total,
            "radiant": total // 2,
            "dire": total - total // 2,
            "source": f"ml:{loaded.version}",
        }

    def _predict_multikill(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Return a `multikill` block, or None if prediction isn't possible.

        Output mirrors the heuristic shape: `{"level": "Low"|"Medium"|"High",
        "likely_side": <team name>}`.  The `source` tag carries the
        model version for the eval harness.
        """
        loaded = self._sub_models.get("multikill")
        if loaded is None:
            return None
        ids = _valid_hero_id_list(heroes_a, heroes_b)
        if ids is None:
            return None
        r_ids, d_ids = ids
        feats = _predict_features(
            loaded, r_ids, d_ids,
            radiant_team_id=_team_id(team_a),
            dire_team_id=_team_id(team_b),
            radiant_lane=_empty_lane_dict(),
            dire_lane=_empty_lane_dict(),
        )
        # predict returns the predicted class label directly.
        # The model was trained on strings ("Low", "Medium",
        # "High") so the prediction is also a string.
        level = str(loaded.model.predict([feats])[0])
        # Which side is more likely to host the rampage?  Without
        # a per-side model we use the same heuristic the analysis
        # engine uses: tied → winning team; otherwise the side
        # with the higher win probability.
        proba = loaded.model.predict_proba([feats])[0]
        classes = list(loaded.model.classes_)
        # The "High" class probability is the natural signal for
        # which side tends to pop off; if High is unlikely we
        # don't bother picking a side.  Otherwise default to the
        # winning side (mirrors `analysis.analyze`).
        if "High" in classes:
            high_p = proba[classes.index("High")]
        else:
            high_p = 0.0
        # The full feature vector doesn't tell us which side
        # the rampage happened on; we leave the side to the
        # heuristic via the `likely_side` field that already
        # exists.  ML only contributes the *level* in 0.3.0.
        return {
            "level": level,
            "likely_side": None,  # let the heuristic fill this in
            "source": f"ml:{loaded.version}",
            "p_high": float(high_p),
        }

    # ---- top-level analyze -------------------------------------------- #

    def analyze(
        self,
        team_a: Dict[str, Any],
        team_b: Dict[str, Any],
        heroes_a: List[Optional[Dict[str, Any]]],
        heroes_b: List[Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        # 1. Always start from the heuristic so we get all the
        #    metrics and the UI fields the user expects.
        result = self._fallback.analyze(team_a, team_b, heroes_a, heroes_b)

        # 2. Override blocks.  Each call is best-effort: any
        #    failure (missing hero id, model exception) leaves
        #    the heuristic block in place and logs a warning.
        #
        #    NOTE: these five catches deliberately use `except Exception`
        #    rather than `except MLPredictError`.  A bug in our own
        #    _predict_* helpers (KeyError on a missing field, TypeError
        #    on a None result) would otherwise crash the entire board
        #    request.  Per-target fallback is the safer contract here.
        try:
            new_winner = self._predict_winner(team_a, team_b, heroes_a, heroes_b)
            if new_winner is not None:
                result["winner"] = new_winner
        except Exception as exc:  # noqa: BLE001 — keep the board alive
            log.warning("ml winner prediction failed: %s", exc)

        try:
            new_kills = self._predict_kills(team_a, team_b, heroes_a, heroes_b)
            if new_kills is not None:
                total = new_kills["total"]
                result["kills"] = {
                    "total": total,
                    "radiant": new_kills["radiant"],
                    "dire": new_kills["dire"],
                }
                result["kills_total_over_under"] = _build_over_under_from_kills(total)
        except Exception as exc:  # noqa: BLE001
            log.warning("ml kills prediction failed: %s", exc)

        try:
            new_dur = self._predict_duration(team_a, team_b, heroes_a, heroes_b)
            if new_dur is not None and new_dur["mean"] is not None:
                dur = float(new_dur["mean"])
                result["duration_min"] = round(dur, 1)
                result["total_over_under"] = _build_over_under_from_duration(dur)
        except Exception as exc:  # noqa: BLE001
            log.warning("ml duration prediction failed: %s", exc)

        try:
            new_towers = self._predict_towers(team_a, team_b, heroes_a, heroes_b)
            if new_towers is not None:
                result["towers"] = {
                    "total": new_towers["total"],
                    "radiant": new_towers["radiant"],
                    "dire": new_towers["dire"],
                }
        except Exception as exc:  # noqa: BLE001
            log.warning("ml towers prediction failed: %s", exc)

        try:
            new_mk = self._predict_multikill(team_a, team_b, heroes_a, heroes_b)
            if new_mk is not None:
                # ML only owns the `level`; merge with the
                # heuristic's `likely_side` so the UI keeps a
                # consistent shape.
                old_mk = result.get("multikill") or {}
                result["multikill"] = {
                    "level": new_mk["level"],
                    "likely_side": old_mk.get("likely_side") or team_a["name"],
                    "source": new_mk["source"],
                }
        except Exception as exc:  # noqa: BLE001
            log.warning("ml multikill prediction failed: %s", exc)

        return result


# ---------------------------------------------------------------------------- #
# Factory + module singleton
# ---------------------------------------------------------------------------- #

def make_engine(
    name: Optional[str] = None,
    model_dir: Optional[Path] = None,
) -> IPredictionEngine:
    """Build the engine the rest of the app should use.

    `name` defaults to the `PREDICTION_ENGINE` env var, then to
    `"heuristic"`.  Unknown names raise — we do not silently fall
    back, because that would mask a typo in the env file.

    For `name == "ml"` we scan `MODEL_DIR` for every known sub-model
    and load whatever is present.  An empty result degrades to
    `HeuristicEngine` with a warning.
    """
    name = (name or os.environ.get("PREDICTION_ENGINE", "heuristic")).strip().lower()

    if name == "heuristic":
        log.info("prediction engine: heuristic")
        return HeuristicEngine()

    if name == "ml":
        if model_dir is None:
            model_dir = Path(os.environ.get("MODEL_DIR", "ml_data/models"))
        storage = ModelStorage(Path(model_dir))

        sub: Dict[str, LoadedModel] = {}
        for target in KNOWN_TARGETS:
            loaded = storage.load(target)
            if loaded is not None:
                sub[target] = loaded

        if not sub:
            log.warning(
                "PREDICTION_ENGINE=ml but no trained sub-models found at %s; "
                "falling back to heuristic. Run `python -m business.ml.train` first.",
                model_dir,
            )
            return HeuristicEngine()

        log.info(
            "prediction engine: ml (loaded sub-models: %s)",
            sorted(sub.keys()),
        )
        return MLEngine(sub_models=sub, fallback=HeuristicEngine())

    raise ValueError(
        f"unknown PREDICTION_ENGINE={name!r}; expected 'heuristic' or 'ml'"
    )


# Module-level lazy singleton.  Built on first access;
# `reset_default_engine()` is provided so tests (and `app.py`
# startup hooks) can force a rebuild after the env changes.
_default_engine: Optional[IPredictionEngine] = None


def get_default_engine() -> IPredictionEngine:
    """Return the process-wide prediction engine, building it on first use."""
    global _default_engine
    if _default_engine is None:
        _default_engine = make_engine()
    return _default_engine


def reset_default_engine() -> None:
    """Drop the cached engine so the next `get_default_engine()` rebuilds it."""
    global _default_engine
    _default_engine = None
