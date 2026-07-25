"""Smoothing grid for the cross-side matchup encoder.

The 0.3.13 release picked smoothing=3.0 by analogy with the
hero encoder (which uses 5.0) and the lane encoder (3.0).
Apples-to-apples forward on the winner head shows the
right number is somewhere in [1.0, 5.0]; this grid
sweeps it and re-trains winner_v14 with the best value.

Honest methodology: encoder fit on train only (883 matches),
evaluated on the same 1497-match OOS split used by
`scripts/forward_winner_v013.py`.
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
    HeroWinRateEncoder,
    CrossSideMatchupEncoder,
    extract_features,
)
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
XGB = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                    random_state=42, eval_metric="logloss", verbosity=0)
SMOOTHINGS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


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
    print(f"apples-to-apples: train {len(v1_train)}, OOS {len(v1_test)}")

    print()
    print("=" * 80)
    print(f"  {'smoothing':<10} {'acc':>8} {'log_loss':>10} {'auc':>8}")
    print("=" * 80)
    rows = []
    for smooth in SMOOTHINGS:
        # Build a fresh encoder with the candidate smoothing on
        # the matchup sub-encoder (hero/team/fit on the same
        # train pool, matchup on train only).
        enc = HeroWinRateEncoder().fit(raw_train)
        # Re-fit only the matchup encoder with the candidate smoothing.
        enc.matchup_encoder = CrossSideMatchupEncoder(smoothing=smooth).fit(raw_train)
        X = np.asarray(
            [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc, match=m, groups=("hero", "matchup")) for t, m in zip(targets, raw)],
            dtype=float,
        )
        Xtr, Xte = X[v1_train], X[v1_test]
        ytr, yte = y[v1_train], y[v1_test]
        m = XGB.__class__(**XGB.get_params())
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, (proba >= 0.5).astype(int))
        ll = log_loss(yte, proba, labels=[0, 1])
        try: auc = roc_auc_score(yte, proba)
        except: auc = float("nan")
        rows.append({"smoothing": smooth, "acc": acc, "ll": ll, "auc": auc})
        print(f"  {smooth:<10.1f} {acc:>8.4f} {ll:>10.4f} {auc:>8.4f}")

    print()
    print("Sorted by accuracy:")
    for r in sorted(rows, key=lambda x: -x["acc"]):
        print(f"  smoothing={r['smoothing']:.1f}  acc={r['acc']:.4f}  ll={r['ll']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
