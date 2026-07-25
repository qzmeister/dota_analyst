"""Honest grid: encoder fit on TRAIN only (no test leakage).

The previous grid used `encoder.fit(full_corpus)`, which means
test labels were visible to the pair-lookup tables in
`LanePairEncoder`.  This version re-fits the encoder on the
train split and reports accuracy on the same test split — the
honest comparison.
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
    n_estimators=50, learning_rate=0.1, max_depth=3,
    random_state=42, eval_metric="logloss", verbosity=0,
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

    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    raw_train = [raw[i] for i in idx_train]
    targets_train = [targets[i] for i in idx_train]
    raw_test = [raw[i] for i in idx_test]
    targets_test = [targets[i] for i in idx_test]
    print(f"split: {len(idx_train)} train / {len(idx_test)} test")

    # 1) Encoder fit on TRAIN only.
    encoder = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw_train)

    # 2) Build X for train (using train-only encoder) and test (same encoder).
    results = []
    for name, groups in GROUPS.items():
        X_train = np.asarray(
            [
                extract_features(
                    t.radiant_hero_ids, t.dire_hero_ids, encoder,
                    radiant_team_id=t.radiant_team_id,
                    dire_team_id=t.dire_team_id,
                    match=m,
                    groups=groups,
                )
                for t, m in zip(targets_train, raw_train)
            ],
            dtype=float,
        )
        X_test = np.asarray(
            [
                extract_features(
                    t.radiant_hero_ids, t.dire_hero_ids, encoder,
                    radiant_team_id=t.radiant_team_id,
                    dire_team_id=t.dire_team_id,
                    match=m,
                    groups=groups,
                )
                for t, m in zip(targets_test, raw_test)
            ],
            dtype=float,
        )
        t0 = time.perf_counter()
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_train, y[idx_train])
        proba = clf.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - t0
        acc = accuracy_score(y[idx_test], (proba >= 0.5).astype(int))
        ll = log_loss(y[idx_test], proba, labels=[0, 1])
        try:
            auc = roc_auc_score(y[idx_test], proba)
        except Exception:
            auc = float("nan")
        results.append({
            "groups": name,
            "n_features": X_train.shape[1],
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
