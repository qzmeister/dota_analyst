"""v0.3.13 grid: hero + (team) + (lane) + (matchup) + (patch) on the
apples-to-apples forward split (883 train, 1497 OOS).

Tests each feature combination for honest forward improvements
on the winner head.  XGBoost n_est=50, md=3, lr=0.1 (the
v0.3.9 winner_v9 config that we know generalises well).
"""
from __future__ import annotations

import json, sys, time
from itertools import product
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import HeroWinRateEncoder, extract_features
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
GROUPS_TO_TRY = {
    "hero":                     ("hero",),
    "hero+team":                ("hero", "team"),
    "hero+lane":                ("hero", "lane"),
    "hero+matchup":             ("hero", "matchup"),
    "hero+patch":               ("hero", "patch"),
    "hero+team+matchup":        ("hero", "team", "matchup"),
    "hero+team+patch":          ("hero", "team", "patch"),
    "hero+matchup+patch":       ("hero", "matchup", "patch"),
    "hero+team+matchup+patch":  ("hero", "team", "matchup", "patch"),
    "hero+lane+matchup":        ("hero", "lane", "matchup"),
    "hero+lane+matchup+patch":  ("hero", "lane", "matchup", "patch"),
    "all":                      ("hero", "team", "lane", "matchup", "patch"),
}

XGB = XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    random_state=42, eval_metric="logloss", verbosity=0,
)


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
    encoder = HeroWinRateEncoder().fit(raw)
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))

    # v1-style split: first 883 for train, rest for OOS forward
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    v1_train = idx_train[:883]
    v1_test = np.concatenate([idx_train[883:], idx_test])
    print(f"v1-style: train {len(v1_train)}, OOS forward {len(v1_test)}")

    print()
    print("=" * 80)
    print(f"  {'groups':<28} {'F':>3} {'acc':>8} {'log_loss':>10} {'auc':>8}")
    print("=" * 80)
    rows = []
    for name, groups in GROUPS_TO_TRY.items():
        X = np.asarray(
            [extract_features(t.radiant_hero_ids, t.dire_hero_ids, encoder, match=m, groups=groups) for t, m in zip(targets, raw)],
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
        rows.append({"groups": name, "F": X.shape[1], "acc": acc, "ll": ll, "auc": auc})
        print(f"  {name:<28} {X.shape[1]:>3} {acc:>8.4f} {ll:>10.4f} {auc:>8.4f}")

    print()
    print("=" * 80)
    print("Sorted by accuracy:")
    print("=" * 80)
    for r in sorted(rows, key=lambda x: -x["acc"]):
        print(f"  {r['groups']:<28} F={r['F']:>3}  acc={r['acc']:.4f}  ll={r['ll']:.4f}  auc={r['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
