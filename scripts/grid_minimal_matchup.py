"""Minimal matchup grid: mid_1v1 only, vs full 3-feature matchup.

Coverage diagnostic showed:
  - bot  2v2:  1.4% OOS hit rate
  - top  2v2:  1.0% OOS hit rate
  - mid  1v1: 81.9% OOS hit rate

The bot/top pairs almost always fall back to `global_rate`.
This grid tests whether a 1-feature "mid_1v1" group is better
than the 3-feature "matchup" group — it has the same signal
without the 2 noisy bot/top features.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    FEATURE_GROUPS, HeroWinRateEncoder, extract_features, feature_names,
)
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
XGB = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                    random_state=42, eval_metric="logloss", verbosity=0)


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
    print(f"loaded {len(raw)} matches")
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    v1_train = idx_train[:883]
    v1_test = np.concatenate([idx_train[883:], idx_test])
    raw_train = [raw[i] for i in v1_train]
    enc = HeroWinRateEncoder().fit(raw_train)

    VARIANTS = {
        "hero+matchup(bot+top+mid)": ("hero", "matchup"),
        "hero+mid_1v1":              ("hero", "matchup_mid_only"),
    }
    print()
    print("=" * 80)
    print(f"  {'variant':<30} {'F':>3} {'acc':>8} {'log_loss':>10} {'auc':>8}")
    print("=" * 80)
    for name, groups in VARIANTS.items():
        # Build X for the full 3-feature matchup, then optionally
        # collapse to the mid column.
        X_full = np.asarray(
            [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc, match=m, groups=("hero", "matchup")) for t, m in zip(targets, raw)],
            dtype=float,
        )
        n_hero = 13  # the canonical hero group has 13 features
        if name.endswith("mid_1v1"):
            # Keep hero + the LAST column of matchup (mid_1v1).
            X = np.hstack([X_full[:, :n_hero], X_full[:, -1:]])
        else:
            X = X_full
        Xtr, Xte = X[v1_train], X[v1_test]
        ytr, yte = y[v1_train], y[v1_test]
        m = XGB.__class__(**XGB.get_params())
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, (proba >= 0.5).astype(int))
        ll = log_loss(yte, proba, labels=[0, 1])
        try: auc = roc_auc_score(yte, proba)
        except: auc = float("nan")
        print(f"  {name:<30} {X.shape[1]:>3} {acc:>8.4f} {ll:>10.4f} {auc:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
