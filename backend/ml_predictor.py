"""Runtime adapter for the trained pre-match winner model.

The board supplies teams and completed drafts but not player line-ups.  This
adapter uses the model's historical team/draft features and leaves unavailable
player-specific fields at their neutral values.  It never prevents the board
from rendering: callers receive ``None`` when the model is unavailable.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models"
_model = None
_feature_names: Optional[List[str]] = None
_snapshot: Optional[Dict] = None
_load_attempted = False
_loaded_marker = None
_model_source = "temporal_xgboost"


def _rate(record: Dict) -> float:
    return (float(record.get("wins", 0.0)) + 1.0) / (float(record.get("matches", 0.0)) + 2.0)


def _entity_features(prefix: str, keys: Iterable[str], records: Dict) -> Dict[str, float]:
    values = [records.get(key, {"wins": 0.0, "matches": 0.0}) for key in keys if key]
    if not values:
        return {f"{prefix}_wr": 0.5, f"{prefix}_matches": 0.0}
    return {
        f"{prefix}_wr": float(np.mean([_rate(value) for value in values])),
        f"{prefix}_matches": float(np.mean([float(value.get("matches", 0.0)) for value in values])),
    }


def _pair_keys(hero_ids: List[int]) -> List[str]:
    return [f"{first}:{second}" for first, second in combinations(sorted(hero_ids), 2)]


def _recent_rate(record: Dict) -> float:
    results = (record.get("recent") or [])[-10:]
    return float(np.mean(results)) if results else 0.5


def _team_record(team_name: str) -> Dict:
    """Find a historical team record despite harmless name formatting changes."""
    if not _load() or _snapshot is None:
        return {}
    teams = _snapshot.get("teams", {})
    if team_name in teams:
        return teams[team_name]
    normalized = "".join(character for character in team_name.lower() if character.isalnum())
    for name, record in teams.items():
        if "".join(character for character in name.lower() if character.isalnum()) == normalized:
            return record
    return {}


def predict_prematch_winner(team_a: Dict, team_b: Dict) -> Optional[Dict]:
    """Predict a series favourite from historical ML snapshot data only.

    Unlike ``predict_winner``, this works before a draft and deliberately uses
    no hero/player data from the future map.
    """
    a_record, b_record = _team_record(team_a.get("name") or ""), _team_record(team_b.get("name") or "")
    if not a_record and not b_record:
        return None
    a_elo, b_elo = float(a_record.get("elo", 1500.0)), float(b_record.get("elo", 1500.0))
    elo_probability = 1.0 / (1.0 + 10.0 ** ((b_elo - a_elo) / 400.0))
    a_probability = 0.70 * elo_probability + 0.20 * _rate(a_record) + 0.10 * _recent_rate(a_record)
    b_probability = 0.70 * (1.0 - elo_probability) + 0.20 * _rate(b_record) + 0.10 * _recent_rate(b_record)
    probability_a = a_probability / (a_probability + b_probability)
    winner = team_a if probability_a >= 0.5 else team_b
    return {
        "source": "historical_elo_form",
        "team": winner.get("name"),
        "probability": int(round(max(probability_a, 1.0 - probability_a) * 100)),
        "prob_team_a": int(round(probability_a * 100)),
    }


def team_form_context(team_a: Dict, team_b: Dict) -> Dict:
    """Expose only pre-match historical context used by the prediction."""
    a_record = _team_record(team_a.get("name") or "")
    b_record = _team_record(team_b.get("name") or "")
    if not a_record and not b_record:
        return {}

    def profile(record: Dict) -> Dict:
        matches = int(record.get("matches", 0))
        recent = record.get("recent") or []
        return {
            "elo": int(round(float(record.get("elo", 1500)))),
            "maps": matches,
            "win_rate": int(round(_rate(record) * 100)) if matches else None,
            "recent_win_rate": int(round(_recent_rate(record) * 100)) if recent else None,
        }

    a_name, b_name = team_a.get("name") or "", team_b.get("name") or ""
    h2h = (_snapshot or {}).get("head_to_head", {}).get(":".join(sorted((a_name, b_name))), {})
    return {
        "team_a": profile(a_record),
        "team_b": profile(b_record),
        "h2h_maps": int(h2h.get("matches", 0)),
    }


def draft_context(radiant_hero_ids: List[int], dire_hero_ids: List[int]) -> Dict:
    """Summarise the historical draft signal without pretending it is live gold data."""
    if not _load() or _snapshot is None:
        return {}

    def side(ids: List[int]) -> Dict:
        records = [_snapshot.get("heroes", {}).get(str(hero_id), {}) for hero_id in ids]
        records = [record for record in records if record.get("matches")]
        if not records:
            return {"maps": 0, "win_rate": None}
        maps = sum(float(record.get("matches", 0)) for record in records)
        weighted_wins = sum(float(record.get("wins", 0)) for record in records)
        return {
            "maps": int(round(maps / len(records))),
            "win_rate": int(round(100 * weighted_wins / maps)) if maps else None,
        }

    return {"radiant": side(radiant_hero_ids), "dire": side(dire_hero_ids)}


def _load() -> bool:
    """Load model artifacts once; return False when optional ML dependencies are absent."""
    global _feature_names, _load_attempted, _loaded_marker, _model, _model_source, _snapshot
    metadata_path = MODEL_DIR / "model_metadata_prematch.json"
    try:
        marker = metadata_path.stat().st_mtime_ns
    except OSError:
        marker = None
    if _load_attempted and marker == _loaded_marker:
        return _model is not None
    _load_attempted = True
    _loaded_marker = marker
    _model = None
    try:
        model_path = MODEL_DIR / "xgb_prematch.json"
        features_path = MODEL_DIR / "feature_cols_prematch.json"
        snapshot_path = MODEL_DIR / "prematch_snapshot.json"
        if not all(path.exists() for path in (model_path, features_path, snapshot_path)):
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("type") == "prematch_random_forest":
            import joblib
            model_path = MODEL_DIR / metadata.get("model_path", "prematch_model.joblib")
            if not model_path.exists():
                return False
            _model = joblib.load(model_path)
            _model_source = "random_forest"
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(model_path))
            _model = model
            _model_source = "temporal_xgboost"
        _feature_names = json.loads(features_path.read_text(encoding="utf-8"))
        _snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def predict_winner(radiant_team: Dict, dire_team: Dict, radiant_hero_ids: List[int], dire_hero_ids: List[int]) -> Optional[Dict]:
    """Return the ML winner prediction for a complete draft, or ``None`` as fallback."""
    if len(radiant_hero_ids) != 5 or len(dire_hero_ids) != 5 or not _load():
        return None
    assert _feature_names is not None and _model is not None and _snapshot is not None
    r_team = radiant_team.get("name") or ""
    d_team = dire_team.get("name") or ""
    state = _snapshot
    row: Dict[str, float] = {}
    for index, hero_id in enumerate(sorted(radiant_hero_ids)):
        row[f"r_hero_{index}"] = float(hero_id)
    for index, hero_id in enumerate(sorted(dire_hero_ids)):
        row[f"d_hero_{index}"] = float(hero_id)
    row.update(_entity_features("r_team", [r_team], state.get("teams", {})))
    row.update(_entity_features("d_team", [d_team], state.get("teams", {})))
    row.update(_entity_features("r_players", [], state.get("players", {})))
    row.update(_entity_features("d_players", [], state.get("players", {})))
    row.update(_entity_features("r_heroes", [str(hero) for hero in radiant_hero_ids], state.get("heroes", {})))
    row.update(_entity_features("d_heroes", [str(hero) for hero in dire_hero_ids], state.get("heroes", {})))
    row.update(_entity_features("r_patch_heroes", [], state.get("patch_heroes", {})))
    row.update(_entity_features("d_patch_heroes", [], state.get("patch_heroes", {})))
    row.update(_entity_features("r_pairs", _pair_keys(radiant_hero_ids), state.get("hero_pairs", {})))
    row.update(_entity_features("d_pairs", _pair_keys(dire_hero_ids), state.get("hero_pairs", {})))
    row.update(_entity_features("r_roster", [], state.get("rosters", {})))
    row.update(_entity_features("d_roster", [], state.get("rosters", {})))
    r_record = state.get("teams", {}).get(r_team, {})
    d_record = state.get("teams", {}).get(d_team, {})
    row["r_team_elo"] = float(r_record.get("elo", 1500.0))
    row["d_team_elo"] = float(d_record.get("elo", 1500.0))
    row["team_elo_diff"] = row["r_team_elo"] - row["d_team_elo"]
    row["r_team_recent_wr"] = _recent_rate(r_record)
    row["d_team_recent_wr"] = _recent_rate(d_record)
    row["team_recent_wr_diff"] = row["r_team_recent_wr"] - row["d_team_recent_wr"]
    h2h_key = ":".join(sorted((r_team, d_team)))
    h2h = state.get("head_to_head", {}).get(h2h_key, {"wins": 0.0, "matches": 0.0})
    matches = float(h2h.get("matches", 0.0))
    wins = float(h2h.get("wins", 0.0))
    row["r_h2h_wr"] = ((wins if r_team <= d_team else matches - wins) + 1.0) / (matches + 2.0)
    row["h2h_matches"] = matches
    for metric in ("wr", "matches"):
        for prefix in ("team", "player", "hero", "pair", "roster", "patch_hero"):
            source = {"player": "players", "hero": "heroes", "patch_hero": "patch_heroes"}.get(prefix, f"{prefix}s")
            row[f"{prefix}_{metric}_diff"] = row.get(f"r_{source}_{metric}", 0.0) - row.get(f"d_{source}_{metric}", 0.0)
    vector = np.array([[row.get(name, 0.0) for name in _feature_names]], dtype=float)
    radiant_probability = float(_model.predict_proba(vector)[0][1])
    return {
        "source": _model_source,
        "prob_radiant": int(round(radiant_probability * 100)),
        "winner": radiant_team.get("name") if radiant_probability >= 0.5 else dire_team.get("name"),
        "probability": int(round(max(radiant_probability, 1.0 - radiant_probability) * 100)),
    }
