"""Train the selected Random Forest prematch model on all collected maps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from benchmark_prematch_models import load_dataset


MODEL_DIR = Path("ml_models")
MODEL_PATH = MODEL_DIR / "prematch_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata_prematch.json"
FEATURES_PATH = MODEL_DIR / "feature_cols_prematch.json"
SNAPSHOT_PATH = MODEL_DIR / "prematch_snapshot.json"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    X, y, features, snapshot = load_dataset(include_state=True)
    split = int(len(X) * 0.8)
    params = dict(n_estimators=700, max_depth=7, min_samples_leaf=10, max_features=0.7, random_state=42, n_jobs=-1)

    holdout = RandomForestClassifier(**params)
    holdout.fit(X[:split], y[:split])
    probabilities = holdout.predict_proba(X[split:])[:, 1]
    accuracy = float(accuracy_score(y[split:], probabilities >= 0.5))
    auc = float(roc_auc_score(y[split:], probabilities))
    brier = float(brier_score_loss(y[split:], probabilities))
    print(f"Chronological holdout: accuracy={accuracy:.4f}; ROC-AUC={auc:.4f}; Brier={brier:.4f}")

    final_model = RandomForestClassifier(**params)
    final_model.fit(X, y)
    temporary = MODEL_PATH.with_suffix(".tmp.joblib")
    joblib.dump(final_model, temporary)
    os.replace(temporary, MODEL_PATH)
    atomic_json(FEATURES_PATH, features)
    atomic_json(SNAPSHOT_PATH, snapshot)
    atomic_json(METADATA_PATH, {
        "type": "prematch_random_forest", "model_path": MODEL_PATH.name,
        "n_samples": int(len(y)), "n_features": len(features),
        "holdout_fraction": 0.2, "holdout_samples": int(len(y) - split),
        "chronological_holdout_accuracy": accuracy,
        "chronological_holdout_roc_auc": auc,
        "chronological_holdout_brier": brier,
    })
    print(f"Saved production model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
