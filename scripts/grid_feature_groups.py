"""A/B grid: same XGBoost config, four feature-group combinations.

This is the 0.3.10 dev-cycle harness — reproduces the v0.3.9 XGBoost
winner (n_est=50, lr=0.1, max_depth=3, plain) on 1275 matches
with the four possible feature-group combinations:

  F=13:  hero          (0.3.9 baseline)
  F=17:  hero + team   (C retry)
  F=20:  hero + lane   (D v2)
  F=24:  hero + team + lane  (C + D)

We use the same train/test split (random_state=42) and the same
encoder (fit on the FULL corpus, idiomatic target encoding) so
the comparison isolates the effect of the feature groups.

Reports accuracy and log_loss for each variant so we can pick the
best one to retrain + save in the next step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from business.ml.features import (  # noqa: E402
    FEATURE_GROUPS,
    HeroWinRateEncoder,
    extract_features,
    feature_names,
)
from business.ml.targets import extract_target  # noqa: E402

DATA_DIR = ROOT / "ml_data" / "full_matches"

GROUPS = {
    "hero":           ("hero",),
    "hero+team":      ("hero", "team"),
    "hero+lane":      ("hero", "lane"),
    "hero+team+lane": ("hero", "team", "lane"),
}

XGB_PARAMS = dict(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)


def load_data() -> tuple[list[dict], list]:
    raw, targets = [], []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = extract_target(d)
        if t is None:
            continue
        raw.append(d)
        targets.append(t)
    print(f"loaded {len(raw)} clean matches")
    return raw, targets


def main() -> int:
    raw, targets = load_data()
    encoder = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw)

    # Build X for every group combination once.
    X_by_group: dict[tuple, np.ndarray] = {}
    for name, groups in GROUPS.items():
        X = np.asarray(
            [
                extract_features(
                    t.radiant_hero_ids, t.dire_hero_ids, encoder,
                    radiant_team_id=t.radiant_team_id,
                    dire_team_id=t.dire_team_id,
                    match=m,
                    groups=groups,
                )
                for t, m in zip(targets, raw)
            ],
            dtype=float,
        )
        X_by_group[groups] = X
        print(f"  {name:<22}  F={X.shape[1]:2d}  names={feature_names(groups)}")

    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"split: {len(idx_train)} train / {len(idx_test)} test, "
          f"train radiant_pct={y[idx_train].mean():.3f}, "
          f"test radiant_pct={y[idx_test].mean():.3f}")

    results = []
    for name, groups in GROUPS.items():
        X = X_by_group[groups]
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        t0 = time.perf_counter()
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - t0
        acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
        ll = log_loss(y_test, proba, labels=[0, 1])
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = float("nan")
        results.append({
            "groups": name,
            "n_features": X.shape[1],
            "accuracy": float(acc),
            "log_loss": float(ll),
            "roc_auc": float(auc),
            "train_seconds": float(elapsed),
        })
        print(f"  {name:<22}  acc={acc:.4f}  log_loss={ll:.4f}  "
              f"auc={auc:.4f}  ({elapsed:.1f}s)")

    print()
    print("=" * 80)
    print(f"  {'groups':<22} {'F':>3} {'acc':>8} {'log_loss':>10} {'auc':>8}   time")
    print("=" * 80)
    base = next(r for r in results if r["groups"] == "hero")
    for r in results:
        d_acc = r["accuracy"] - base["accuracy"]
        d_ll = r["log_loss"] - base["log_loss"]
        print(f"  {r['groups']:<22} {r['n_features']:>3} "
              f"{r['accuracy']:>8.4f} {r['log_loss']:>10.4f} {r['roc_auc']:>8.4f}   "
              f"{r['train_seconds']:.1f}s   d_acc={d_acc:+.4f}  d_ll={d_ll:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
