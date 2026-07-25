"""Train a leakage-resistant pre-match outcome model.

Features for a match are calculated solely from matches that started earlier.
The final statistics snapshot is saved alongside the model for inference on new
drafts.  A chronological holdout, rather than shuffled cross-validation, is
used for evaluation.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


MATCHES_DIR = Path("ml_data/full_matches")
MODEL_DIR = Path("ml_models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_prematch.json"
FEATURES_PATH = MODEL_DIR / "feature_cols_prematch.json"
SNAPSHOT_PATH = MODEL_DIR / "prematch_snapshot.json"
METADATA_PATH = MODEL_DIR / "model_metadata_prematch.json"


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rate(record: Dict[str, float], default: float = 0.5) -> float:
    """Return a smoothed historical win rate for one entity."""
    return (record["wins"] + 2.0 * default) / (record["matches"] + 2.0)


def entity_features(prefix: str, names: Iterable[str], records: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Summarise historical performance of a team, roster, or hero pool."""
    values = [records.get(name, {"wins": 0.0, "matches": 0.0}) for name in names if name]
    if not values:
        return {f"{prefix}_wr": 0.5, f"{prefix}_matches": 0.0}
    return {
        f"{prefix}_wr": float(np.mean([rate(value) for value in values])),
        f"{prefix}_matches": float(np.mean([value['matches'] for value in values])),
    }


def recent_rate(record: Dict[str, Any], window: int = 10) -> float:
    """Return the team's result over its most recent completed maps."""
    results = record.get("recent", [])[-window:]
    return float(np.mean(results)) if results else 0.5


def pair_keys(hero_ids: List[int]) -> List[str]:
    """Create canonical keys for every two-hero combination in a draft."""
    return [f"{first}:{second}" for first, second in combinations(sorted(hero_ids), 2)]


def roster_key(players: List[str]) -> str:
    """Create a stable key for the five players currently in a lineup."""
    return ":".join(sorted(players))


def head_to_head_key(first_team: str, second_team: str) -> str:
    return ":".join(sorted((first_team, second_team)))


def expected_win_probability(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def match_entities(match: Dict[str, Any]) -> Tuple[str, str, List[str], List[str], List[int], List[int]]:
    """Extract draft-visible entities; no lane or post-game performance is used."""
    radiant = match["radiant"]
    dire = match["dire"]
    r_players = radiant["player_performances"]
    d_players = dire["player_performances"]
    r_heroes = sorted(player["performance"]["hero"]["valve_id"] for player in r_players)
    d_heroes = sorted(player["performance"]["hero"]["valve_id"] for player in d_players)
    r_names = [str(player["player"].get("steam32") or player["player"].get("nickname", "")) for player in r_players]
    d_names = [str(player["player"].get("steam32") or player["player"].get("nickname", "")) for player in d_players]
    return (radiant["team"].get("name", ""), dire["team"].get("name", ""), r_names, d_names, r_heroes, d_heroes)


def build_features(match: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, float]:
    """Build draft-time features from the state known before this match."""
    r_team, d_team, r_players, d_players, r_heroes, d_heroes = match_entities(match)
    row: Dict[str, float] = {}
    for index, hero_id in enumerate(r_heroes):
        row[f"r_hero_{index}"] = float(hero_id)
    for index, hero_id in enumerate(d_heroes):
        row[f"d_hero_{index}"] = float(hero_id)
    row.update(entity_features("r_team", [r_team], state["teams"]))
    row.update(entity_features("d_team", [d_team], state["teams"]))
    row.update(entity_features("r_players", r_players, state["players"]))
    row.update(entity_features("d_players", d_players, state["players"]))
    row.update(entity_features("r_heroes", [str(hero) for hero in r_heroes], state["heroes"]))
    row.update(entity_features("d_heroes", [str(hero) for hero in d_heroes], state["heroes"]))
    patch = str(match.get("patch") or "unknown")
    row.update(entity_features("r_patch_heroes", [f"{patch}:{hero}" for hero in r_heroes], state["patch_heroes"]))
    row.update(entity_features("d_patch_heroes", [f"{patch}:{hero}" for hero in d_heroes], state["patch_heroes"]))
    row.update(entity_features("r_pairs", pair_keys(r_heroes), state["hero_pairs"]))
    row.update(entity_features("d_pairs", pair_keys(d_heroes), state["hero_pairs"]))
    row.update(entity_features("r_roster", [roster_key(r_players)], state["rosters"]))
    row.update(entity_features("d_roster", [roster_key(d_players)], state["rosters"]))
    r_team_record = state["teams"].get(r_team, {})
    d_team_record = state["teams"].get(d_team, {})
    r_elo = float(r_team_record.get("elo", 1500.0))
    d_elo = float(d_team_record.get("elo", 1500.0))
    row["r_team_elo"] = r_elo
    row["d_team_elo"] = d_elo
    row["team_elo_diff"] = r_elo - d_elo
    row["r_team_recent_wr"] = recent_rate(r_team_record)
    row["d_team_recent_wr"] = recent_rate(d_team_record)
    row["team_recent_wr_diff"] = row["r_team_recent_wr"] - row["d_team_recent_wr"]
    h2h = state["head_to_head"].get(head_to_head_key(r_team, d_team), {"wins": 0.0, "matches": 0.0})
    r_is_canonical = r_team <= d_team
    r_h2h_wins = h2h["wins"] if r_is_canonical else h2h["matches"] - h2h["wins"]
    row["r_h2h_wr"] = (r_h2h_wins + 1.0) / (h2h["matches"] + 2.0)
    row["h2h_matches"] = h2h["matches"]
    for metric in ("wr", "matches"):
        row[f"team_{metric}_diff"] = row[f"r_team_{metric}"] - row[f"d_team_{metric}"]
        row[f"player_{metric}_diff"] = row[f"r_players_{metric}"] - row[f"d_players_{metric}"]
        row[f"hero_{metric}_diff"] = row[f"r_heroes_{metric}"] - row[f"d_heroes_{metric}"]
        row[f"pair_{metric}_diff"] = row[f"r_pairs_{metric}"] - row[f"d_pairs_{metric}"]
        row[f"roster_{metric}_diff"] = row[f"r_roster_{metric}"] - row[f"d_roster_{metric}"]
        row[f"patch_hero_{metric}_diff"] = row[f"r_patch_heroes_{metric}"] - row[f"d_patch_heroes_{metric}"]
    return row


def update_state(match: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Record the result only after features for its entire timestamp batch exist."""
    r_team, d_team, r_players, d_players, r_heroes, d_heroes = match_entities(match)
    radiant_won = bool(match["radiant_victory"])
    for collection, names, won in (
        (state["teams"], [r_team], radiant_won),
        (state["teams"], [d_team], not radiant_won),
        (state["players"], r_players, radiant_won),
        (state["players"], d_players, not radiant_won),
        (state["heroes"], [str(hero) for hero in r_heroes], radiant_won),
        (state["heroes"], [str(hero) for hero in d_heroes], not radiant_won),
        (state["patch_heroes"], [f"{match.get('patch') or 'unknown'}:{hero}" for hero in r_heroes], radiant_won),
        (state["patch_heroes"], [f"{match.get('patch') or 'unknown'}:{hero}" for hero in d_heroes], not radiant_won),
        (state["hero_pairs"], pair_keys(r_heroes), radiant_won),
        (state["hero_pairs"], pair_keys(d_heroes), not radiant_won),
        (state["rosters"], [roster_key(r_players)], radiant_won),
        (state["rosters"], [roster_key(d_players)], not radiant_won),
    ):
        for name in names:
            collection[name]["matches"] += 1.0
            collection[name]["wins"] += float(won)
    for team, won in ((r_team, radiant_won), (d_team, not radiant_won)):
        state["teams"][team].setdefault("recent", []).append(float(won))
        state["teams"][team]["recent"] = state["teams"][team]["recent"][-10:]
    r_record = state["teams"][r_team]
    d_record = state["teams"][d_team]
    r_expected = expected_win_probability(r_record["elo"], d_record["elo"])
    adjustment = 24.0 * (float(radiant_won) - r_expected)
    r_record["elo"] += adjustment
    d_record["elo"] -= adjustment
    h2h = state["head_to_head"][head_to_head_key(r_team, d_team)]
    h2h["matches"] += 1.0
    h2h["wins"] += float(radiant_won if r_team <= d_team else not radiant_won)


def serialise_state(state: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {kind: dict(values) for kind, values in state.items()}


def main() -> None:
    matches: List[Dict[str, Any]] = []
    for path in MATCHES_DIR.glob("*.json"):
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
            if match.get("start_date") and len(match["radiant"].get("player_performances", [])) == 5:
                matches.append(match)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    matches.sort(key=lambda item: (item["start_date"], item["match_id"]))
    if len(matches) < 100:
        raise RuntimeError("Need at least 100 valid timestamped matches for temporal training")

    def new_record() -> Dict[str, Any]:
        return {"wins": 0.0, "matches": 0.0, "elo": 1500.0, "recent": []}

    state = {kind: defaultdict(new_record) for kind in (
        "teams", "players", "heroes", "patch_heroes", "hero_pairs", "rosters", "head_to_head"
    )}
    rows: List[Dict[str, float]] = []
    targets: List[int] = []
    index = 0
    while index < len(matches):
        timestamp = matches[index]["start_date"]
        end = index
        while end < len(matches) and matches[end]["start_date"] == timestamp:
            end += 1
        for match in matches[index:end]:
            rows.append(build_features(match, state))
            targets.append(int(bool(match["radiant_victory"])))
        for match in matches[index:end]:
            update_state(match, state)
        index = end

    feature_names = sorted(rows[0])
    X = np.array([[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=float)
    y = np.array(targets, dtype=int)
    split = int(len(X) * 0.8)
    model_args = dict(n_estimators=260, max_depth=3, learning_rate=0.03, subsample=0.85,
                      colsample_bytree=0.9, min_child_weight=10, reg_alpha=0.3, reg_lambda=3.0,
                      random_state=42, eval_metric="logloss")
    holdout_model = XGBClassifier(**model_args)
    holdout_model.fit(X[:split], y[:split])
    probabilities = holdout_model.predict_proba(X[split:])[:, 1]
    accuracy = accuracy_score(y[split:], probabilities >= 0.5)
    auc = roc_auc_score(y[split:], probabilities)
    print(f"Chronological holdout: {len(y) - split} newest matches")
    print(f"Accuracy: {accuracy:.4f}; ROC-AUC: {auc:.4f}")

    final_model = XGBClassifier(**model_args)
    final_model.fit(X, y)
    staged_model = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.tmp.json")
    final_model.save_model(str(staged_model))
    os.replace(staged_model, MODEL_PATH)
    atomic_write_json(FEATURES_PATH, feature_names)
    atomic_write_json(SNAPSHOT_PATH, serialise_state(state))
    atomic_write_json(METADATA_PATH, {
        "type": "prematch_temporal", "n_samples": len(y), "n_features": len(feature_names),
        "holdout_fraction": 0.2, "holdout_samples": len(y) - split,
        "chronological_holdout_accuracy": float(accuracy), "chronological_holdout_roc_auc": float(auc),
        "training_end_timestamp": matches[-1]["start_date"],
        "feature_names": feature_names,
    })
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
