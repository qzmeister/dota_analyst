"""
Batch Training with Rotation (Ensemble Approach)
Train 10 models on different batches of 100 matches each.
Rotate test/train splits and ensemble predictions.

Approach:
  - Split 1,111 matches into ~11 batches of 100
  - Train 10 models, each on 9 batches (900 matches)
  - Validate on 1 batch (100 matches)
  - Rotate: each batch is used as test set once
  - Ensemble: average predictions from all 10 models
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
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


def batch_train_with_rotation(X, y, target_name: str, n_batches: int = 10):
    """Train models with batch rotation and ensemble."""
    print(f"\n{'=' * 60}")
    print(f"Batch Training with Rotation for {target_name}")
    print(f"{'=' * 60}")
    
    n_samples = len(X)
    batch_size = n_samples // n_batches
    
    print(f"\n  Total samples: {n_samples}")
    print(f"  Batch size: ~{batch_size}")
    print(f"  Number of batches: {n_batches}")
    
    # Use KFold for rotation
    kf = KFold(n_splits=n_batches, shuffle=True, random_state=42)
    
    models = []
    test_predictions = []
    test_actuals = []
    fold_maes = []
    
    # Best params from previous tuning
    best_params = {
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
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"\n  Fold {fold}/{n_batches}:")
        print(f"    Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        
        # Train model
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        fold_mae = mean_absolute_error(y_test, y_pred)
        
        print(f"    MAE: {fold_mae:.2f}")
        
        models.append(model)
        test_predictions.extend(y_pred)
        test_actuals.extend(y_test)
        fold_maes.append(fold_mae)
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(test_actuals, test_predictions)
    
    print(f"\n  {'=' * 40}")
    print(f"  Overall Results:")
    print(f"    Mean MAE across folds: {np.mean(fold_maes):.2f} (+/- {np.std(fold_maes):.2f})")
    print(f"    Ensemble MAE: {overall_mae:.2f}")
    print(f"  {'=' * 40}")
    
    return models, overall_mae, fold_maes


def ensemble_predict(models, X):
    """Average predictions from all models."""
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)


def main():
    print("=" * 60)
    print("Batch Training with Rotation (Ensemble)")
    print("=" * 60)
    
    # Build training data
    df = build_training_data()
    df = add_regression_targets(df)
    df = add_advanced_features(df)
    
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
    cols_path = MODEL_DIR / "feature_cols_ensemble.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Train ensemble models for each target
    results = {}
    
    # 1. Duration
    y_duration = df['duration_min'].values
    models_duration, mae_duration, fold_maes_duration = batch_train_with_rotation(
        X, y_duration, "Duration", n_batches=10
    )
    results['duration'] = {
        'mae': mae_duration,
        'fold_maes': [float(m) for m in fold_maes_duration],
        'n_models': len(models_duration)
    }
    
    # Save ensemble models
    for i, model in enumerate(models_duration):
        model.save_model(str(MODEL_DIR / f"xgb_duration_ensemble_fold{i+1}.json"))
    
    # 2. Kills
    y_kills = df['total_kills'].values
    models_kills, mae_kills, fold_maes_kills = batch_train_with_rotation(
        X, y_kills, "Kills", n_batches=10
    )
    results['kills'] = {
        'mae': mae_kills,
        'fold_maes': [float(m) for m in fold_maes_kills],
        'n_models': len(models_kills)
    }
    
    for i, model in enumerate(models_kills):
        model.save_model(str(MODEL_DIR / f"xgb_kills_ensemble_fold{i+1}.json"))
    
    # 3. Towers
    y_towers = df['estimated_towers'].values
    models_towers, mae_towers, fold_maes_towers = batch_train_with_rotation(
        X, y_towers, "Towers", n_batches=10
    )
    results['towers'] = {
        'mae': mae_towers,
        'fold_maes': [float(m) for m in fold_maes_towers],
        'n_models': len(models_towers)
    }
    
    for i, model in enumerate(models_towers):
        model.save_model(str(MODEL_DIR / f"xgb_towers_ensemble_fold{i+1}.json"))
    
    # Save metadata
    metadata = {
        'approach': 'batch_rotation_ensemble',
        'n_batches': 10,
        'models': results,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols
    }
    
    meta_path = MODEL_DIR / "model_metadata_ensemble.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("BATCH TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nEnsemble MAE (10 models averaged):")
    print(f"  Duration: {results['duration']['mae']:.2f} min")
    print(f"  Kills: {results['kills']['mae']:.2f}")
    print(f"  Towers: {results['towers']['mae']:.2f}")
    print(f"\n30 models saved (10 folds × 3 targets)")


if __name__ == "__main__":
    main()
