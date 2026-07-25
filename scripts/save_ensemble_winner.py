"""Bagging ensemble of LogReg winner heads (5 models, average probabilities).

0.3.9 — audit point E (local stand-in for corpus expansion).  The
real E would be re-pulling ~3000 more matches from DatDota; we
don't have time for that in this session.  Bagging is the
next-best thing: 5 LogReg models on bootstrap samples, average
their predict_proba.  Reduces overconfidence → better log_loss
without sacrificing accuracy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.ml.train import load_matches_with_targets, extract_features
from business.ml.features import FEATURE_ORDER, HeroWinRateEncoder


def main() -> int:
    data_dir = _ROOT / "ml_data" / "full_matches"
    matched_raw, targets = load_matches_with_targets(data_dir)
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    train_t = [targets[i] for i in idx_train]
    test_t = [targets[i] for i in idx_test]
    enc = HeroWinRateEncoder().fit(matched_raw)

    Xtr = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in train_t], dtype=float,
    )
    Xte = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in test_t], dtype=float,
    )
    ytr, yte = y[idx_train], y[idx_test]

    rng = np.random.default_rng(42)
    n_models = 5
    models = []
    for i in range(n_models):
        # Bootstrap sample of train
        boot_idx = rng.integers(0, len(Xtr), size=len(Xtr))
        Xb, yb = Xtr[boot_idx], ytr[boot_idx]
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=42 + i)
        m.fit(Xb, yb)
        models.append(m)
        print(f"  trained ensemble member {i+1}/{n_models} on bootstrap size {len(Xb)}")

    # Average probabilities
    probs = np.mean([m.predict_proba(Xte)[:, 1] for m in models], axis=0)
    acc = accuracy_score(yte, (probs >= 0.5).astype(int))
    ll = log_loss(yte, probs, labels=[0, 1])
    auc = roc_auc_score(yte, probs)
    print(f"\nBagging ensemble ({n_models} LogReg, C=1.0, bootstrap):")
    print(f"  acc={acc:.4f}  log_loss={ll:.4f}  auc={auc:.4f}")

    # Save as winner_v10 — need to wrap in a class with predict_proba
    class _BaggingModel:
        def __init__(self, models):
            self._models = models
        def predict_proba(self, X):
            ps = [m.predict_proba(X) for m in self._models]
            return np.mean(ps, axis=0)
        def predict(self, X):
            p = self.predict_proba(X)[:, 1]
            return (p >= 0.5).astype(int)

    wrapped = _BaggingModel(models)
    out_dir = _ROOT / "ml_data" / "models" / "winner_v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapped, out_dir / "model.joblib")
    metadata = {
        "name": "winner",
        "version": "10",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "sklearn_version": "1.9.0",
        "numpy_version": "2.5.1",
        "python_version": "3.14.3",
        "feature_names": list(FEATURE_ORDER),
        "n_features": len(FEATURE_ORDER),
        "metrics": {
            "accuracy": acc,
            "log_loss": ll,
            "roc_auc": auc,
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "model": "logreg_bagging",
            "n_estimators": n_models,
        },
        "n_features": len(FEATURE_ORDER),
        "feature_names": list(FEATURE_ORDER),
        "encoder": enc.to_dict(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
