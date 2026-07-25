"""
Advanced Feature Engineering + Hyperparameter Tuning
Improve MAE by adding interaction features and optimizing model parameters.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Paths
MATCHES_DIR = Path("ml_data/full_matches")
ML_DATA = Path("ml_data")
MODEL_DIR = Path("ml_models")
MODEL_DIR.mkdir(exist_ok=True)

# Import from existing scripts
from train_prematch_model import build_training_data
from train_regression_models import add_regression_targets


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and polynomial features."""
    print("\nAdding advanced features...")
    
    # ── Interaction features ──
    # Team strength × Hero strength
    df['team_wr_x_hero_wr'] = df['team_wr_diff'] * df['hero_wr_diff']
    df['team_gpm_x_hero_wr'] = df['team_gpm_diff'] * df['hero_wr_diff']
    
    # Player form momentum
    df['form_momentum'] = df['p0_wr_diff'] + df['p1_wr_diff'] + df['p2_wr_diff']
    df['form_momentum_abs'] = df['form_momentum'].abs()
    
    # Team consistency (lower diff = more consistent)
    df['team_consistency'] = 1.0 - (df['team_wr_diff'].abs() / 2.0)
    
    # Player synergy (all 3 players in good form)
    df['player_synergy'] = (df['p0_wr_diff'] > 0).astype(int) + \
                           (df['p1_wr_diff'] > 0).astype(int) + \
                           (df['p2_wr_diff'] > 0).astype(int)
    
    # ── Polynomial features ──
    df['team_wr_diff_sq'] = df['team_wr_diff'] ** 2
    df['hero_wr_diff_sq'] = df['hero_wr_diff'] ** 2
    df['team_gpm_diff_sq'] = df['team_gpm_diff'] ** 2
    
    # ── Ratio features ──
    df['r_team_wr_ratio'] = df['r_team_wr'] / (df['d_team_wr'] + 0.01)
    df['d_team_wr_ratio'] = df['d_team_wr'] / (df['r_team_wr'] + 0.01)
    
    # ── Difference features ──
    df['r_vs_d_kills'] = df['r_team_avg_kills'] - df['d_team_avg_kills']
    df['r_vs_d_duration'] = df['r_team_avg_duration'] - df['d_team_avg_duration']
    df['r_vs_d_nw'] = df['r_team_nw_adv'] - df['d_team_nw_adv']
    
    # ── Laning advantage combined ──
    df['lanes_won_diff'] = df['r_lanes_won'] - df['d_lanes_won']
    df['fb_pct_diff'] = df['r_fb_pct'] - df['d_fb_pct']
    df['nw_adv_laning_diff'] = df['r_nw_adv_laning'] - df['d_nw_adv_laning']
    
    # ── Mid lane dominance ──
    df['mid_dominance'] = df['mid_hero_wr_diff'] * df['team_wr_diff']
    
    # ── Draft quality score ──
    df['r_draft_quality'] = df['r_avg_hero_wr'] * df['r_hero_diversity']
    df['d_draft_quality'] = df['d_avg_hero_wr'] * df['d_hero_diversity']
    df['draft_quality_diff'] = df['r_draft_quality'] - df['d_draft_quality']
    
    # ── Tier weight interactions ──
    df['tier_x_team_wr'] = df['tier_weight'] * df['team_wr_diff']
    df['tier_x_hero_wr'] = df['tier_weight'] * df['hero_wr_diff']
    
    print(f"  Added {len(df.columns) - 71} new features")
    print(f"  Total features: {len(df.columns)}")
    
    return df


def tune_hyperparameters(X, y, target_name: str):
    """Find optimal hyperparameters using GridSearchCV."""
    print(f"\n{'=' * 60}")
    print(f"Hyperparameter tuning for {target_name}")
    print(f"{'=' * 60}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [3, 5, 7],
        'reg_alpha': [0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5]
    }
    
    # Use RandomizedSearchCV for faster search (samples 50 combinations)
    from sklearn.model_selection import RandomizedSearchCV
    
    base_model = XGBRegressor(random_state=42)
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n  Searching {total_combos} total combinations...")
    print(f"  (Using RandomizedSearchCV with 50 iterations)")
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=50,
        scoring=mae_scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X, y)
    
    print(f"\n  Best MAE: {-search.best_score_:.2f}")
    print(f"  Best parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")
    
    return search.best_params_, -search.best_score_


def main():
    print("=" * 60)
    print("Advanced Feature Engineering + Hyperparameter Tuning")
    print("=" * 60)
    
    # Build training data
    df = build_training_data()
    df = add_regression_targets(df)
    
    # Drop rows with missing targets
    df = df.dropna(subset=['duration_min', 'total_kills', 'estimated_towers'])
    
    # Add advanced features
    df = add_advanced_features(df)
    
    # Define features (exclude targets and metadata)
    exclude_cols = {'target', 'match_id', 'duration_min', 'total_kills', 
                    'total_building_damage', 'estimated_towers'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Save feature columns
    cols_path = MODEL_DIR / "feature_cols_advanced.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Tune and train for each target
    results = {}
    
    # 1. Duration
    y_duration = df['duration_min'].values
    best_params, best_mae = tune_hyperparameters(X, y_duration, "Duration")
    results['duration'] = {'params': best_params, 'mae': best_mae}
    
    # Train final model with best params
    model_duration = XGBRegressor(**best_params, random_state=42)
    model_duration.fit(X, y_duration)
    model_duration.save_model(str(MODEL_DIR / "xgb_duration_advanced.json"))
    
    # 2. Kills
    y_kills = df['total_kills'].values
    best_params, best_mae = tune_hyperparameters(X, y_kills, "Kills")
    results['kills'] = {'params': best_params, 'mae': best_mae}
    
    model_kills = XGBRegressor(**best_params, random_state=42)
    model_kills.fit(X, y_kills)
    model_kills.save_model(str(MODEL_DIR / "xgb_kills_advanced.json"))
    
    # 3. Towers
    y_towers = df['estimated_towers'].values
    best_params, best_mae = tune_hyperparameters(X, y_towers, "Towers")
    results['towers'] = {'params': best_params, 'mae': best_mae}
    
    model_towers = XGBRegressor(**best_params, random_state=42)
    model_towers.fit(X, y_towers)
    model_towers.save_model(str(MODEL_DIR / "xgb_towers_advanced.json"))
    
    # Save metadata
    metadata = {
        'models': results,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols
    }
    
    meta_path = MODEL_DIR / "model_metadata_advanced.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("TUNING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nImproved MAE:")
    print(f"  Duration: {results['duration']['mae']:.2f} min")
    print(f"  Kills: {results['kills']['mae']:.2f}")
    print(f"  Towers: {results['towers']['mae']:.2f}")
    print(f"\nModels saved with '_advanced' suffix")


if __name__ == "__main__":
    main()
