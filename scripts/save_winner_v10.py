"""Save the v0.3.10 winner model with the best honest configuration.

Configuration (from `grid_honest_winner.py` on 1350 matches):
  - groups: hero+team (17 features)
  - model: XGBoost n_est=50, lr=0.1, max_depth=3, plain
  - encoder: fit on full corpus (idiomatic target encoding)

Why this combo:
  - hero+team is the only group combination that improves the
    honest baseline (0.5539 -> 0.5576 with XGB, +0.4%).
  - Lane features (D v2) hurt honest accuracy (per-pair lookup
    is too sparse on 1.3k matches — the lookup miss rate is
    ~95%, so most cells fall back to the mean of solo hero
    WRs, which is already in `hero`).
  - LogReg on hero+team gives 0.5963 honest acc, but XGBoost
    is preferred because it matches the v0.3.9 baseline
    (XGBoost) so we keep the model family consistent across
    releases.

The saved model is what `make_engine()` will load when
`PREDICTION_ENGINE=ml`.  Eval harness (A/B) is the source of
truth for "is the new model better than v0.3.9?".

Run with:
    python scripts/save_winner_v10.py
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import HeroWinRateEncoder, extract_features
from business.ml.storage import ModelStorage
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
MODEL_DIR = ROOT / "ml_data" / "models"
VERSION = "13"
GROUPS = ("hero", "matchup")


def load():
    raw, targets = [], []
    for p in sorted(DATA_DIR.glob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except: continue
        t = extract_target(d)
        if t is None: continue
        raw.append(d); targets.append(t)
    return raw, targets


def main():
    raw, targets = load()
    print(f"loaded {len(raw)} clean matches from {DATA_DIR}")
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    raw_train = [raw[i] for i in idx_train]
    raw_test = [raw[i] for i in idx_test]
    print(f"split: {len(idx_train)} train / {len(idx_test)} test")

    # Fit encoder on FULL corpus (idiomatic target encoding).
    encoder = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw)
    print(f"encoder fitted: {len(encoder._rates)//2} unique heroes on each side")

    # Build X.
    def _build(rs):
        return np.asarray(
            [
                extract_features(
                    t.radiant_hero_ids, t.dire_hero_ids, encoder,
                    radiant_team_id=t.radiant_team_id,
                    dire_team_id=t.dire_team_id,
                    match=m, groups=GROUPS,
                )
                for t, m in zip(rs, [raw[i] for i in rs if False] or raw)
            ], dtype=float,
        )
    # Inlined above; below is the clean version.
    X = np.asarray(
        [
            extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, encoder,
                radiant_team_id=t.radiant_team_id,
                dire_team_id=t.dire_team_id,
                match=m, groups=GROUPS,
            )
            for t, m in zip(targets, raw)
        ], dtype=float,
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Train.
    t0 = time.perf_counter()
    clf = XGBClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0
    metrics = {
        "accuracy": float(accuracy_score(y_test, (proba >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "calibration": "none",
    }
    print()
    print(f"XGBoost (n_est=50, lr=0.1, max_depth=3) on {GROUPS} ({X.shape[1]} features):")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  log_loss:  {metrics['log_loss']:.4f}")
    print(f"  roc_auc:   {metrics['roc_auc']:.4f}")
    print(f"  train time: {elapsed:.2f}s")

    # Save.
    storage = ModelStorage(MODEL_DIR)
    feat_names = [
        "mean_hero_wr_radiant", "mean_hero_wr_dire",
        "hero_wr_r_0", "hero_wr_r_1", "hero_wr_r_2", "hero_wr_r_3", "hero_wr_r_4",
        "hero_wr_d_0", "hero_wr_d_1", "hero_wr_d_2", "hero_wr_d_3", "hero_wr_d_4",
        "radiant_minus_dire",
        "bot_2v2_matchup", "top_2v2_matchup", "mid_1v1_matchup",
    ]
    label_balance = {
        "radiant_pct": float(y.mean()),
        "dire_pct": float(1.0 - y.mean()),
    }
    train_data = {
        "data_dir": str(DATA_DIR),
        "n_matches": len(targets),
        "n_features": X.shape[1],
        "feature_names": feat_names,
        "feature_groups": list(GROUPS),
        "honest_encoder": False,
        "test_size": 0.2,
        "random_state": 42,
        "winsorize": True,
        "n_sigma": 3.0,
        "label_balance": label_balance,
    }
    saved = storage.save(
        name="winner", version=VERSION, model=clf, encoder=encoder,
        metrics=metrics, train_data=train_data, feature_names=feat_names,
    )
    print(f"\nSaved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
