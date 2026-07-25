"""
Full Hyperparameter Search on Complete Dataset (1,111 matches)
Exhaustive search for optimal parameters since training is fast.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, RandomizedSearchCV
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


def exhaustive_search(X, y, target_name: str, n_iter=100):
    """Exhaustive hyperparameter search."""
    print(f"\n{'=' * 60}")
    print(f"Exhaustive Search for {target_name}")
    print(f"{'=' * 60}")
    
    # Wider parameter grid
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 5.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],
        'scale_pos_weight': [1.0]  # Not used for regression
    }
    
    total_combos = 1
    for v in param_dist.values():
        total_combos *= len(v)
    
    print(f"\n  Parameter space: {total_combos:,} combinations")
    print(f"  Random search iterations: {n_iter}")
    print(f"  Samples: {len(X)}")
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    base_model = XGBRegressor(random_state=42, tree_method='hist')
    
    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\n  Searching...")
    search.fit(X, y)
    
    best_mae = -search.best_score_
    best_params = search.best_params_
    
    print(f"\n  Best MAE: {best_mae:.2f}")
    print(f"  Best parameters:")
    for param, value in sorted(best_params.items()):
        print(f"    {param:20s}: {value}")
    
    # Train final model with best params
    final_model = XGBRegressor(**best_params, random_state=42, tree_method='hist')
    final_model.fit(X, y)
    
    return final_model, best_mae, best_params


def main():
    print("=" * 60)
    print("Exhaustive Hyperparameter Search")
    print("Full dataset (1,111 matches)")
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
    cols_path = MODEL_DIR / "feature_cols_exhaustive.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Search and train for each target
    results = {}
    
    # 1. Duration
    y_duration = df['duration_min'].values
    model_duration, mae_duration, params_duration = exhaustive_search(
        X, y_duration, "Duration", n_iter=150
    )
    model_duration.save_model(str(MODEL_DIR / "xgb_duration_exhaustive.json"))
    results['duration'] = {
        'mae': float(mae_duration),
        'params': {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                   for k, v in params_duration.items()}
    }
    
    # 2. Kills
    y_kills = df['total_kills'].values
    model_kills, mae_kills, params_kills = exhaustive_search(
        X, y_kills, "Kills", n_iter=150
    )
    model_kills.save_model(str(MODEL_DIR / "xgb_kills_exhaustive.json"))
    results['kills'] = {
        'mae': float(mae_kills),
        'params': {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                   for k, v in params_kills.items()}
    }
    
    # 3. Towers
    y_towers = df['estimated_towers'].values
    model_towers, mae_towers, params_towers = exhaustive_search(
        X, y_towers, "Towers", n_iter=150
    )
    model_towers.save_model(str(MODEL_DIR / "xgb_towers_exhaustive.json"))
    results['towers'] = {
        'mae': float(mae_towers),
        'params': {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                   for k, v in params_towers.items()}
    }
    
    # Save metadata
    metadata = {
        'approach': 'exhaustive_search',
        'n_iterations': 150,
        'models': results,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols
    }
    
    meta_path = MODEL_DIR / "model_metadata_exhaustive.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("EXHAUSTIVE SEARCH COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nBest MAE found:")
    print(f"  Duration: {results['duration']['mae']:.2f} min")
    print(f"  Kills: {results['kills']['mae']:.2f}")
    print(f"  Towers: {results['towers']['mae']:.2f}")
    print(f"\nModels saved with '_exhaustive' suffix")


if __name__ == "__main__":
    main()
