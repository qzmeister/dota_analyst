"""
Multi-Target ML Training Pipeline
Train models for:
  1. Match duration (minutes)
  2. Total kills
  3. Total towers destroyed (building damage proxy)

Uses same pre-match features as win prediction.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from train_prematch_model import build_training_data, build_match_features

# Paths
MATCHES_DIR = Path("ml_data/full_matches")
ML_DATA = Path("ml_data")
MODEL_DIR = Path("ml_models")
MODEL_DIR.mkdir(exist_ok=True)


def add_regression_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add regression targets from actual match data."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            m = json.load(f)
        
        match_id = m['match_id']
        duration_min = m['duration'] / 60.0
        
        r_players = m['radiant']['player_performances']
        d_players = m['dire']['player_performances']
        
        total_kills = sum(p['performance']['kills'] or 0 for p in r_players + d_players)
        total_building_damage = sum(p['performance']['building_damage'] or 0 for p in r_players + d_players)
        
        # Proxy for towers: building_damage / 2000 (approximate tower HP)
        estimated_towers = total_building_damage / 2000
        
        matches.append({
            'match_id': match_id,
            'duration_min': round(duration_min, 2),
            'total_kills': total_kills,
            'total_building_damage': total_building_damage,
            'estimated_towers': round(estimated_towers, 1)
        })
    
    targets_df = pd.DataFrame(matches)
    
    # Merge with existing df
    df = df.merge(targets_df, on='match_id', how='left')
    
    return df


def train_regression_model(X, y, target_name: str, model_path: Path):
    """Train a regression model."""
    print(f"\n{'=' * 60}")
    print(f"Training {target_name} prediction model")
    print(f"{'=' * 60}")
    
    print(f"\nTarget stats:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}, Max: {y.max():.2f}")
    
    # XGBoost Regressor
    model = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    # Cross-validation (MAE and RMSE)
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring=mae_scorer)
    rmse_scores = cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer)
    
    # Negate because sklearn returns negative scores for error metrics
    mae_scores = -mae_scores
    rmse_scores = -rmse_scores
    
    print(f"\n  CV MAE: {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")
    print(f"  CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
    
    # Train final model
    model.fit(X, y)
    
    # Feature importance
    importance = model.feature_importances_
    feature_cols = [f'f{i}' for i in range(len(importance))]
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 10 features:")
    for i, (name, imp) in enumerate(feat_imp[:10], 1):
        print(f"    {i}. feature_{i}: {imp:.4f}")
    
    # Save model
    model.save_model(str(model_path))
    print(f"\n  Model saved to {model_path}")
    
    return {
        'mae': float(mae_scores.mean()),
        'rmse': float(rmse_scores.mean()),
        'target_mean': float(y.mean()),
        'target_std': float(y.std())
    }


def main():
    print("=" * 60)
    print("Multi-Target ML Training Pipeline")
    print("Predict: duration, kills, towers")
    print("=" * 60)
    
    # Build training data (reuse from pre-match)
    df = build_training_data()
    
    # Add regression targets
    df = add_regression_targets(df)
    
    # Check for missing targets
    if df['duration_min'].isna().sum() > 0:
        print(f"\nWarning: {df['duration_min'].isna().sum()} matches missing targets — dropping")
        df = df.dropna(subset=['duration_min', 'total_kills', 'estimated_towers'])
    
    # Define features (exclude targets and metadata)
    exclude_cols = {'target', 'match_id', 'duration_min', 'total_kills', 
                    'total_building_damage', 'estimated_towers'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Save feature columns
    cols_path = MODEL_DIR / "feature_cols_regression.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Train models
    results = {}
    
    # 1. Duration
    y_duration = df['duration_min'].values
    duration_path = MODEL_DIR / "xgb_duration.json"
    results['duration'] = train_regression_model(X, y_duration, "Duration (min)", duration_path)
    
    # 2. Total kills
    y_kills = df['total_kills'].values
    kills_path = MODEL_DIR / "xgb_kills.json"
    results['kills'] = train_regression_model(X, y_kills, "Total Kills", kills_path)
    
    # 3. Towers (estimated)
    y_towers = df['estimated_towers'].values
    towers_path = MODEL_DIR / "xgb_towers.json"
    results['towers'] = train_regression_model(X, y_towers, "Estimated Towers", towers_path)
    
    # Save metadata
    metadata = {
        'models': results,
        'n_samples': len(X),
        'n_features': len(feature_cols)
    }
    
    meta_path = MODEL_DIR / "model_metadata_regression.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("ALL REGRESSION MODELS TRAINED!")
    print(f"{'=' * 60}")
    print(f"\nModels saved:")
    print(f"  - xgb_duration.json (MAE: {results['duration']['mae']:.2f} min)")
    print(f"  - xgb_kills.json (MAE: {results['kills']['mae']:.2f} kills)")
    print(f"  - xgb_towers.json (MAE: {results['towers']['mae']:.2f} towers)")


if __name__ == "__main__":
    main()
