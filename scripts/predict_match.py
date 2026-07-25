"""
ML Inference Pipeline
Predict match outcomes for live/upcoming matches.

Usage:
  python scripts/predict_match.py --match_id 8910909413
  python scripts/predict_match.py --match_json ml_data/full_matches/8910909413.json
"""
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

# Paths
ML_DATA = Path("ml_data")
MODEL_DIR = Path("ml_models")


# ─── Load model and feature columns ────────────────────────────────
def load_model(model_type: str = "prematch"):
    """Load trained model and feature columns."""
    if model_type in {"prematch", "temporal"}:
        model_path = MODEL_DIR / "xgb_prematch.json"
        cols_path = MODEL_DIR / "feature_cols_prematch.json"
    else:
        model_path = MODEL_DIR / "xgb_phase1.json"
        cols_path = MODEL_DIR / "feature_cols_prematch.json"
    
    model = XGBClassifier()
    model.load_model(str(model_path))
    
    with open(cols_path, 'r') as f:
        feature_cols = json.load(f)
    
    return model, feature_cols


def extract_temporal_features(match: dict) -> dict:
    """Extract leakage-resistant features using the final historical snapshot."""
    from train_temporal_prematch import build_features

    snapshot_path = MODEL_DIR / "prematch_snapshot.json"
    with open(snapshot_path, encoding="utf-8") as file:
        snapshot = json.load(file)
    return build_features(match, snapshot)


def load_regression_models():
    """Load regression models for duration, kills, towers."""
    from xgboost import XGBRegressor
    
    models = {}
    cols_path = MODEL_DIR / "feature_cols_regression.json"
    
    with open(cols_path, 'r') as f:
        feature_cols = json.load(f)
    
    for target in ['duration', 'kills', 'towers']:
        model_path = MODEL_DIR / f"xgb_{target}.json"
        if model_path.exists():
            model = XGBRegressor()
            model.load_model(str(model_path))
            models[target] = model
    
    return models, feature_cols


# ─── Load pre-computed feature files ────────────────────────────────
def load_json(name: str) -> dict:
    path = ML_DATA / name
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# Load once at module level
team_stats = load_json("team_stats.json")
mid_matchups = load_json("mid_matchups.json")
player_form = load_json("player_form.json")
patch_meta = load_json("patch_meta.json")
tournament_tiers = load_json("tournament_tiers.json")
team_laning = load_json("team_laning_profiles.json")

tier_lookup = {}
if tournament_tiers and 'leagues' in tournament_tiers:
    for league_name, data in tournament_tiers['leagues'].items():
        tier_lookup[league_name] = data.get('weight', 0.4)


# ─── Helper functions ───────────────────────────────────────────────
def get_team_stat(team_name: str, stat: str, default=0.0) -> float:
    if not team_name or not team_stats:
        return default
    if team_name in team_stats:
        return team_stats[team_name].get(stat, default)
    norm = team_name.lower().replace(' ', '')
    for t in team_stats:
        if t.lower().replace(' ', '') == norm:
            return team_stats[t].get(stat, default)
    return default


def get_player_form_stat(player_name: str, stat: str, default=0.0) -> float:
    if not player_name or not player_form:
        return default
    if player_name in player_form:
        return player_form[player_name].get(stat, default)
    return default


def get_mid_hero_wr(hero_name: str) -> float:
    if not hero_name or not mid_matchups:
        return 0.5
    if hero_name in mid_matchups:
        return mid_matchups[hero_name].get('win_rate', 0.5)
    return 0.5


def get_hero_patch_wr(hero_name: str, patch: str) -> float:
    if not hero_name or not patch_meta:
        return 0.5
    patch_data = patch_meta.get(patch, {})
    heroes = patch_data.get('heroes', {})
    if hero_name in heroes:
        return heroes[hero_name].get('win_rate', 50.0) / 100.0
    return 0.5


def get_laning_stat(team_name: str, stat: str, default=0.0) -> float:
    if not team_name or not team_laning:
        return default
    profiles = team_laning if isinstance(team_laning, dict) else {}
    if team_name in profiles:
        return profiles[team_name].get(stat, default)
    norm = team_name.lower().replace(' ', '')
    for t in profiles:
        if t.lower().replace(' ', '') == norm:
            return profiles[t].get(stat, default)
    return default


# ─── Feature extraction ────────────────────────────────────────────
def extract_features(match: dict) -> dict:
    """Extract pre-match features from a match JSON."""
    row = {}
    
    patch = match.get('patch', 'unknown')
    league_name = match.get('league', {}).get('name', '')
    
    radiant_team = match['radiant']['team']['name']
    dire_team = match['dire']['team']['name']
    r_players = match['radiant']['player_performances']
    d_players = match['dire']['player_performances']
    
    # Hero picks (10)
    r_heroes = sorted([p['performance']['hero']['valve_id'] for p in r_players])
    d_heroes = sorted([p['performance']['hero']['valve_id'] for p in d_players])
    for j, h in enumerate(r_heroes):
        row[f'r_hero_{j}'] = h
    for j, h in enumerate(d_heroes):
        row[f'd_hero_{j}'] = h
    
    # Hero meta win rates (3)
    r_hero_wrs = [get_hero_patch_wr(p['performance']['hero']['short_name'], patch) for p in r_players]
    d_hero_wrs = [get_hero_patch_wr(p['performance']['hero']['short_name'], patch) for p in d_players]
    row['r_avg_hero_wr'] = np.mean(r_hero_wrs)
    row['d_avg_hero_wr'] = np.mean(d_hero_wrs)
    row['hero_wr_diff'] = row['r_avg_hero_wr'] - row['d_avg_hero_wr']
    
    # Team stats (12)
    for prefix, team_name in [('r', radiant_team), ('d', dire_team)]:
        row[f'{prefix}_team_wr'] = get_team_stat(team_name, 'win_rate')
        row[f'{prefix}_team_avg_kills'] = get_team_stat(team_name, 'avg_kills')
        row[f'{prefix}_team_avg_gpm'] = get_team_stat(team_name, 'avg_gpm')
        row[f'{prefix}_team_avg_xpm'] = get_team_stat(team_name, 'avg_xpm')
        row[f'{prefix}_team_avg_duration'] = get_team_stat(team_name, 'avg_duration_min')
        row[f'{prefix}_team_nw_adv'] = get_team_stat(team_name, 'avg_nw_advantage')
    
    row['team_wr_diff'] = row['r_team_wr'] - row['d_team_wr']
    row['team_gpm_diff'] = row['r_team_avg_gpm'] - row['d_team_avg_gpm']
    
    # Player form (18)
    r_by_gpm = sorted(r_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)
    d_by_gpm = sorted(d_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)
    
    for prefix, players in [('r', r_by_gpm[:3]), ('d', d_by_gpm[:3])]:
        for j, p in enumerate(players):
            pname = p['player']['nickname']
            row[f'{prefix}_p{j}_wr'] = get_player_form_stat(pname, 'recent_win_rate', 0.5)
            row[f'{prefix}_p{j}_gpm'] = get_player_form_stat(pname, 'recent_avg_gpm', 500)
            row[f'{prefix}_p{j}_kda'] = get_player_form_stat(pname, 'recent_avg_kda', 0)
            row[f'{prefix}_p{j}_form_delta'] = get_player_form_stat(pname, 'form_delta.win_rate_delta', 0)
    
    for j in range(3):
        row[f'p{j}_wr_diff'] = row.get(f'r_p{j}_wr', 0.5) - row.get(f'd_p{j}_wr', 0.5)
        row[f'p{j}_gpm_diff'] = row.get(f'r_p{j}_gpm', 500) - row.get(f'd_p{j}_gpm', 500)
    
    # Mid matchup hero win rates (3)
    r_mid = next((p for p in r_players if p.get('laneInfo', {}).get('lane') == 'MIDDLE'), None)
    d_mid = next((p for p in d_players if p.get('laneInfo', {}).get('lane') == 'MIDDLE'), None)
    
    if r_mid and d_mid:
        r_mid_hero = r_mid['performance']['hero']['short_name']
        d_mid_hero = d_mid['performance']['hero']['short_name']
        row['r_mid_hero_wr'] = get_mid_hero_wr(r_mid_hero)
        row['d_mid_hero_wr'] = get_mid_hero_wr(d_mid_hero)
        row['mid_hero_wr_diff'] = row['r_mid_hero_wr'] - row['d_mid_hero_wr']
    else:
        row['r_mid_hero_wr'] = row['d_mid_hero_wr'] = row['mid_hero_wr_diff'] = 0.5
    
    # Laning stats (6)
    for prefix, team_name in [('r', radiant_team), ('d', dire_team)]:
        row[f'{prefix}_lanes_won'] = get_laning_stat(team_name, 'lanes_won_pct', 50)
        row[f'{prefix}_nw_adv_laning'] = get_laning_stat(team_name, 'nw_advantage', 0)
        row[f'{prefix}_fb_pct'] = get_laning_stat(team_name, 'fb_pct', 50)
    
    # Tournament tier (1)
    row['tier_weight'] = tier_lookup.get(league_name, 0.4)
    
    # Draft diversity (2)
    row['r_hero_diversity'] = len(set(r_heroes)) / 5.0
    row['d_hero_diversity'] = len(set(d_heroes)) / 5.0
    
    return row


# ─── Prediction ─────────────────────────────────────────────────────
def predict_match(match: dict, model_type: str = "prematch", include_regression: bool = True) -> dict:
    """Predict match outcome and optional regression targets."""
    model, feature_cols = load_model(model_type)
    
    # Extract features
    features = extract_temporal_features(match) if model_type in {"prematch", "temporal"} else extract_features(match)
    
    # Build feature vector
    feature_vector = [features.get(col, 0.0) for col in feature_cols]
    X = np.array([feature_vector])
    
    # Predict win probability
    proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    
    radiant_team = match['radiant']['team']['name']
    dire_team = match['dire']['team']['name']
    
    result = {
        'match_id': match.get('match_id'),
        'radiant_team': radiant_team,
        'dire_team': dire_team,
        'prediction': 'Radiant' if prediction == 1 else 'Dire',
        'radiant_win_prob': float(proba[1]),
        'dire_win_prob': float(proba[0]),
        'confidence': float(max(proba)),
        'league': match.get('league', {}).get('name', 'Unknown'),
        'top_features': {}
    }
    
    # Show top contributing features
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[::-1][:10]
    for idx in top_indices:
        col_name = feature_cols[idx]
        feat_val = features.get(col_name, 0.0)
        result['top_features'][col_name] = {
            'value': round(feat_val, 4),
            'importance': round(float(feature_importance[idx]), 4)
        }
    
    # Add regression predictions
    if include_regression:
        reg_models, reg_cols = load_regression_models()
        
        # Build regression feature vector (may have different columns)
        reg_vector = [features.get(col, 0.0) for col in reg_cols]
        X_reg = np.array([reg_vector])
        
        if 'duration' in reg_models:
            duration_pred = reg_models['duration'].predict(X_reg)[0]
            result['predicted_duration_min'] = round(float(duration_pred), 1)
        
        if 'kills' in reg_models:
            kills_pred = reg_models['kills'].predict(X_reg)[0]
            result['predicted_total_kills'] = round(float(kills_pred), 0)
        
        if 'towers' in reg_models:
            towers_pred = reg_models['towers'].predict(X_reg)[0]
            result['predicted_towers_destroyed'] = round(float(towers_pred), 1)
    
    return result


# ─── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Predict match outcome')
    parser.add_argument('--match_id', type=int, help='Match ID from ml_data/full_matches/')
    parser.add_argument('--match_json', type=str, help='Path to match JSON file')
    parser.add_argument('--model', type=str, default='prematch', choices=['prematch', 'temporal', 'full'],
                        help='Model type: prematch (default), temporal alias, or full')
    
    args = parser.parse_args()
    
    if args.match_json:
        match_path = Path(args.match_json)
    elif args.match_id:
        match_path = Path(f"ml_data/full_matches/{args.match_id}.json")
    else:
        print("Error: Provide --match_id or --match_json")
        return
    
    if not match_path.exists():
        print(f"Error: File not found: {match_path}")
        return
    
    with open(match_path, 'r', encoding='utf-8') as f:
        match = json.load(f)
    
    print("=" * 60)
    print("Match Outcome Prediction")
    print("=" * 60)
    
    result = predict_match(match, args.model)
    
    print(f"\nMatch: {result['radiant_team']} vs {result['dire_team']}")
    print(f"League: {result['league']}")
    print(f"\nPrediction: {result['prediction']} wins")
    print(f"  Radiant win probability: {result['radiant_win_prob']:.1%}")
    print(f"  Dire win probability: {result['dire_win_prob']:.1%}")
    print(f"  Confidence: {result['confidence']:.1%}")
    
    if 'predicted_duration_min' in result:
        print(f"\nPredicted Match Stats:")
        print(f"  Duration: {result['predicted_duration_min']:.1f} min")
        print(f"  Total Kills: {result['predicted_total_kills']:.0f}")
        print(f"  Towers Destroyed: {result['predicted_towers_destroyed']:.1f}")
    
    print(f"\nTop contributing features:")
    for feat, data in result['top_features'].items():
        print(f"  {feat:30s} = {data['value']:8.4f} (imp: {data['importance']:.4f})")
    
    print(f"\n{'=' * 60}")
    
    # Save prediction
    output_path = Path(f"ml_models/prediction_{result['match_id']}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    main()
