"""
Train models with Advanced Hero Features
Integrate hero-specific stats, player-hero combos, matchups, and synergy.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Paths
MATCHES_DIR = Path("ml_data/full_matches")
ML_DATA = Path("ml_data")
MODEL_DIR = Path("ml_models")
MODEL_DIR.mkdir(exist_ok=True)

from train_prematch_model import build_training_data
from train_regression_models import add_regression_targets
from tune_and_improve import add_advanced_features


def load_hero_features() -> dict:
    """Load advanced hero features."""
    path = ML_DATA / "advanced_hero_features.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_hero_specific_features(df: pd.DataFrame, matches: list, hero_features: dict) -> pd.DataFrame:
    """Add hero-specific, player-hero, matchup, and synergy features."""
    print("\nAdding hero-specific features...")
    
    hero_stats = hero_features['hero_stats']
    player_hero_stats = hero_features['player_hero_stats']
    hero_matchups = hero_features['hero_matchups']
    hero_pair_synergy = hero_features['hero_pair_synergy']
    
    # Initialize new columns
    df['r_avg_hero_win_rate'] = 0.5
    df['d_avg_hero_win_rate'] = 0.5
    df['hero_wr_diff_advanced'] = 0.0
    df['r_avg_hero_kda'] = 0.0
    df['d_avg_hero_kda'] = 0.0
    df['r_avg_hero_gpm'] = 500.0
    df['d_avg_hero_gpm'] = 500.0
    df['r_player_hero_wr'] = 0.5
    df['d_player_hero_wr'] = 0.5
    df['r_avg_matchup_wr'] = 0.5
    df['d_avg_matchup_wr'] = 0.5
    df['r_avg_synergy_wr'] = 0.5
    df['d_avg_synergy_wr'] = 0.5
    
    for idx, match in enumerate(matches):
        if idx >= len(df):
            break
        
        r_players = match['radiant']['player_performances']
        d_players = match['dire']['player_performances']
        
        # Hero-specific stats
        r_hero_wrs = []
        d_hero_wrs = []
        r_hero_kdas = []
        d_hero_kdas = []
        r_hero_gpms = []
        d_hero_gpms = []
        
        for p in r_players:
            hero = p['performance']['hero']['short_name']
            if hero in hero_stats:
                r_hero_wrs.append(hero_stats[hero]['win_rate'])
                r_hero_kdas.append(hero_stats[hero]['avg_kda'])
                r_hero_gpms.append(hero_stats[hero]['avg_gpm'])
        
        for p in d_players:
            hero = p['performance']['hero']['short_name']
            if hero in hero_stats:
                d_hero_wrs.append(hero_stats[hero]['win_rate'])
                d_hero_kdas.append(hero_stats[hero]['avg_kda'])
                d_hero_gpms.append(hero_stats[hero]['avg_gpm'])
        
        if r_hero_wrs:
            df.at[idx, 'r_avg_hero_win_rate'] = np.mean(r_hero_wrs)
            df.at[idx, 'r_avg_hero_kda'] = np.mean(r_hero_kdas)
            df.at[idx, 'r_avg_hero_gpm'] = np.mean(r_hero_gpms)
        
        if d_hero_wrs:
            df.at[idx, 'd_avg_hero_win_rate'] = np.mean(d_hero_wrs)
            df.at[idx, 'd_avg_hero_kda'] = np.mean(d_hero_kdas)
            df.at[idx, 'd_avg_hero_gpm'] = np.mean(d_hero_gpms)
        
        df.at[idx, 'hero_wr_diff_advanced'] = df.at[idx, 'r_avg_hero_win_rate'] - df.at[idx, 'd_avg_hero_win_rate']
        
        # Player-hero combinations (top 3 players by GPM)
        r_by_gpm = sorted(r_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)[:3]
        d_by_gpm = sorted(d_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)[:3]
        
        r_ph_wrs = []
        d_ph_wrs = []
        
        for p in r_by_gpm:
            player = p['player']['nickname']
            hero = p['performance']['hero']['short_name']
            if player in player_hero_stats and hero in player_hero_stats[player]:
                r_ph_wrs.append(player_hero_stats[player][hero]['win_rate'])
        
        for p in d_by_gpm:
            player = p['player']['nickname']
            hero = p['performance']['hero']['short_name']
            if player in player_hero_stats and hero in player_hero_stats[player]:
                d_ph_wrs.append(player_hero_stats[player][hero]['win_rate'])
        
        if r_ph_wrs:
            df.at[idx, 'r_player_hero_wr'] = np.mean(r_ph_wrs)
        if d_ph_wrs:
            df.at[idx, 'd_player_hero_wr'] = np.mean(d_ph_wrs)
        
        # Hero matchups (simplified: average win rate vs opponent heroes)
        r_matchup_wrs = []
        d_matchup_wrs = []
        
        r_heroes = [p['performance']['hero']['short_name'] for p in r_players]
        d_heroes = [p['performance']['hero']['short_name'] for p in d_players]
        
        for r_hero in r_heroes:
            if r_hero in hero_matchups:
                for d_hero in d_heroes:
                    if d_hero in hero_matchups[r_hero]:
                        r_matchup_wrs.append(hero_matchups[r_hero][d_hero]['win_rate'])
        
        for d_hero in d_heroes:
            if d_hero in hero_matchups:
                for r_hero in r_heroes:
                    if r_hero in hero_matchups[d_hero]:
                        d_matchup_wrs.append(hero_matchups[d_hero][r_hero]['win_rate'])
        
        if r_matchup_wrs:
            df.at[idx, 'r_avg_matchup_wr'] = np.mean(r_matchup_wrs)
        if d_matchup_wrs:
            df.at[idx, 'd_avg_matchup_wr'] = np.mean(d_matchup_wrs)
        
        # Hero pair synergy (average synergy of all pairs in team)
        r_synergy_wrs = []
        d_synergy_wrs = []
        
        for i in range(len(r_heroes)):
            for j in range(i+1, len(r_heroes)):
                pair_key = '_'.join(sorted([r_heroes[i], r_heroes[j]]))
                if pair_key in hero_pair_synergy:
                    r_synergy_wrs.append(hero_pair_synergy[pair_key]['win_rate'])
        
        for i in range(len(d_heroes)):
            for j in range(i+1, len(d_heroes)):
                pair_key = '_'.join(sorted([d_heroes[i], d_heroes[j]]))
                if pair_key in hero_pair_synergy:
                    d_synergy_wrs.append(hero_pair_synergy[pair_key]['win_rate'])
        
        if r_synergy_wrs:
            df.at[idx, 'r_avg_synergy_wr'] = np.mean(r_synergy_wrs)
        if d_synergy_wrs:
            df.at[idx, 'd_avg_synergy_wr'] = np.mean(d_synergy_wrs)
    
    print(f"  Added 15 hero-specific features")
    return df


def main():
    print("=" * 60)
    print("Training with Advanced Hero Features")
    print("=" * 60)
    
    # Load data
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"\nLoaded {len(matches)} matches")
    
    # Build base training data
    df = build_training_data()
    df = add_regression_targets(df)
    df = add_advanced_features(df)
    
    # Load hero features
    hero_features = load_hero_features()
    
    # Add hero-specific features
    df = add_hero_specific_features(df, matches, hero_features)
    
    # Drop rows with missing targets
    df = df.dropna(subset=['duration_min', 'total_kills', 'estimated_towers'])
    
    # Define features
    exclude_cols = {'target', 'match_id', 'duration_min', 'total_kills', 
                    'total_building_damage', 'estimated_towers'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Save feature columns
    cols_path = MODEL_DIR / "feature_cols_hero.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Best params from exhaustive search
    best_params = {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'min_child_weight': 3,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    # Train models
    results = {}
    
    # 1. Duration
    y_duration = df['duration_min'].values
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    model_duration = XGBRegressor(**best_params)
    mae_duration = -cross_val_score(model_duration, X, y_duration, cv=cv, scoring=mae_scorer).mean()
    model_duration.fit(X, y_duration)
    model_duration.save_model(str(MODEL_DIR / "xgb_duration_hero.json"))
    results['duration'] = {'mae': float(mae_duration)}
    
    print(f"\n  Duration MAE: {mae_duration:.2f} min")
    
    # 2. Kills
    y_kills = df['total_kills'].values
    model_kills = XGBRegressor(**best_params)
    mae_kills = -cross_val_score(model_kills, X, y_kills, cv=cv, scoring=mae_scorer).mean()
    model_kills.fit(X, y_kills)
    model_kills.save_model(str(MODEL_DIR / "xgb_kills_hero.json"))
    results['kills'] = {'mae': float(mae_kills)}
    
    print(f"  Kills MAE: {mae_kills:.2f}")
    
    # 3. Towers
    y_towers = df['estimated_towers'].values
    model_towers = XGBRegressor(**best_params)
    mae_towers = -cross_val_score(model_towers, X, y_towers, cv=cv, scoring=mae_scorer).mean()
    model_towers.fit(X, y_towers)
    model_towers.save_model(str(MODEL_DIR / "xgb_towers_hero.json"))
    results['towers'] = {'mae': float(mae_towers)}
    
    print(f"  Towers MAE: {mae_towers:.2f}")
    
    # Save metadata
    metadata = {
        'approach': 'hero_features',
        'models': results,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols
    }
    
    meta_path = MODEL_DIR / "model_metadata_hero.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("TRAINING WITH HERO FEATURES COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nMAE with hero features:")
    print(f"  Duration: {results['duration']['mae']:.2f} min")
    print(f"  Kills: {results['kills']['mae']:.2f}")
    print(f"  Towers: {results['towers']['mae']:.2f}")


if __name__ == "__main__":
    main()
