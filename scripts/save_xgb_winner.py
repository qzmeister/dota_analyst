"""Train XGBoost winner head and save as winner_v9."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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

    # XGBoost best from grid: n_est=50, lr=0.1, max_depth=3
    # Plain XGBoost (no calibration).  Calibration (sigmoid) gave
    # the same accuracy and slightly better log_loss but the
    # user-facing signal is the winner.team, not the probability.
    base = XGBClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    model = base
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, (p >= 0.5).astype(int))
    ll = log_loss(yte, p, labels=[0, 1])
    auc = roc_auc_score(yte, p)
    print(f"XGBoost n_est=50, lr=0.1, depth=3, sigmoid")
    print(f"  acc={acc:.4f}  log_loss={ll:.4f}  auc={auc:.4f}")

    # Save as winner_v9 (or v11 if --version 11 was passed)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="9")
    p.add_argument("--use-sigmoid", action="store_true")
    args, _ = p.parse_known_args()
    version = args.version

    out_dir = _ROOT / "ml_data" / "models" / f"winner_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    metadata = {
        "name": "winner",
        "version": version,
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
            "calibration": "none",
            "model": "xgboost",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
        "n_features": len(FEATURE_ORDER),
        "feature_names": list(FEATURE_ORDER),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Save the encoder (so engine can load it) — embed in
    # metadata.json the same way ModelStorage.save() does, so
    # `ModelStorage.load()` can find it via `meta.encoder`.
    enc_save = enc.to_dict()
    metadata["encoder"] = enc_save
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
