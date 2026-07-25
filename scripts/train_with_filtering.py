"""
Train models with outlier filtering
Remove extreme values (very short/long matches, anomalous kills)
Keep only "normal" matches for better prediction accuracy.
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


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme values from dataset."""
    print("\nFiltering outliers...")
    print(f"  Original dataset: {len(df)} matches")
    
    # Duration filters (remove very short and very long games)
    duration_p5 = df['duration_min'].quantile(0.05)  # 5th percentile
    duration_p95 = df['duration_min'].quantile(0.95)  # 95th percentile
    
    print(f"\n  Duration distribution:")
    print(f"    Min: {df['duration_min'].min():.1f} min")
    print(f"    5th percentile: {duration_p5:.1f} min")
    print(f"    Median: {df['duration_min'].median():.1f} min")
    print(f"    95th percentile: {duration_p95:.1f} min")
    print(f"    Max: {df['duration_min'].max():.1f} min")
    
    # Kill filters
    kills_p5 = df['total_kills'].quantile(0.05)
    kills_p95 = df['total_kills'].quantile(0.95)
    
    print(f"\n  Kills distribution:")
    print(f"    Min: {df['total_kills'].min():.0f}")
    print(f"    5th percentile: {kills_p5:.0f}")
    print(f"    Median: {df['total_kills'].median():.0f}")
    print(f"    95th percentile: {kills_p95:.0f}")
    print(f"    Max: {df['total_kills'].max():.0f}")
    
    # Tower filters
    towers_p5 = df['estimated_towers'].quantile(0.05)
    towers_p95 = df['estimated_towers'].quantile(0.95)
    
    print(f"\n  Towers distribution:")
    print(f"    Min: {df['estimated_towers'].min():.1f}")
    print(f"    5th percentile: {towers_p5:.1f}")
    print(f"    Median: {df['estimated_towers'].median():.1f}")
    print(f"    95th percentile: {towers_p95:.1f}")
    print(f"    Max: {df['estimated_towers'].max():.1f}")
    
    # Apply filters (keep 5th-95th percentile)
    df_filtered = df[
        (df['duration_min'] >= duration_p5) & 
        (df['duration_min'] <= duration_p95) &
        (df['total_kills'] >= kills_p5) & 
        (df['total_kills'] <= kills_p95) &
        (df['estimated_towers'] >= towers_p5) & 
        (df['estimated_towers'] <= towers_p95)
    ].copy()
    
    print(f"\n  After filtering: {len(df_filtered)} matches")
    print(f"  Removed: {len(df) - len(df_filtered)} outliers ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")
    
    return df_filtered


def train_with_filtering(X, y, target_name: str, best_params: dict):
    """Train model with cross-validation."""
    print(f"\n{'=' * 60}")
    print(f"Training {target_name} on filtered data")
    print(f"{'=' * 60}")
    
    print(f"\n  Samples: {len(X)}")
    print(f"  Target stats:")
    print(f"    Mean: {y.mean():.2f}")
    print(f"    Std: {y.std():.2f}")
    print(f"    Min: {y.min():.2f}, Max: {y.max():.2f}")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    model = XGBRegressor(**best_params)
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring=mae_scorer)
    mae_scores = -mae_scores
    
    print(f"\n  CV MAE: {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")
    print(f"  Fold scores: {[f'{s:.2f}' for s in mae_scores]}")
    
    # Train final model
    model.fit(X, y)
    
    return model, mae_scores.mean()


def main():
    print("=" * 60)
    print("Training with Outlier Filtering")
    print("=" * 60)
    
    # Build training data
    df = build_training_data()
    df = add_regression_targets(df)
    df = add_advanced_features(df)
    
    # Drop rows with missing targets
    df = df.dropna(subset=['duration_min', 'total_kills', 'estimated_towers'])
    
    # Filter outliers
    df_filtered = filter_outliers(df)
    
    # Define features
    exclude_cols = {'target', 'match_id', 'duration_min', 'total_kills', 
                    'total_building_damage', 'estimated_towers'}
    feature_cols = [c for c in df_filtered.columns if c not in exclude_cols]
    
    X = df_filtered[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples (filtered): {len(X)}")
    
    # Save feature columns
    cols_path = MODEL_DIR / "feature_cols_filtered.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Best params from previous tuning
    best_params_duration = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.9,
        'min_child_weight': 3,
        'reg_alpha': 1.0,
        'reg_lambda': 1.5,
        'random_state': 42
    }
    
    best_params_kills = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    best_params_towers = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.01,
        'reg_lambda': 1.5,
        'random_state': 42
    }
    
    # Train models
    results = {}
    
    # 1. Duration
    y_duration = df_filtered['duration_min'].values
    model_duration, mae_duration = train_with_filtering(
        X, y_duration, "Duration", best_params_duration
    )
    model_duration.save_model(str(MODEL_DIR / "xgb_duration_filtered.json"))
    results['duration'] = {'mae': mae_duration}
    
    # 2. Kills
    y_kills = df_filtered['total_kills'].values
    model_kills, mae_kills = train_with_filtering(
        X, y_kills, "Kills", best_params_kills
    )
    model_kills.save_model(str(MODEL_DIR / "xgb_kills_filtered.json"))
    results['kills'] = {'mae': mae_kills}
    
    # 3. Towers
    y_towers = df_filtered['estimated_towers'].values
    model_towers, mae_towers = train_with_filtering(
        X, y_towers, "Towers", best_params_towers
    )
    model_towers.save_model(str(MODEL_DIR / "xgb_towers_filtered.json"))
    results['towers'] = {'mae': mae_towers}
    
    # Save metadata
    metadata = {
        'approach': 'outlier_filtering',
        'filter_percentiles': [5, 95],
        'models': results,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'original_samples': len(df)
    }
    
    meta_path = MODEL_DIR / "model_metadata_filtered.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("FILTERED TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nMAE on filtered dataset ({len(X)} matches):")
    print(f"  Duration: {results['duration']['mae']:.2f} min")
    print(f"  Kills: {results['kills']['mae']:.2f}")
    print(f"  Towers: {results['towers']['mae']:.2f}")
    print(f"\nModels saved with '_filtered' suffix")


if __name__ == "__main__":
    main()
