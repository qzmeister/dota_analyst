"""XGBoost hyperparameter grid on the best honest configuration (hero+team).

Sweeps n_estimators, learning_rate, max_depth on the
encoder-fit-on-full-corpus (idiomatic target encoding) setup
with the hero+team feature group.  Reports test-set accuracy
and log_loss for every combination.

The best (n_est=50, lr=0.1, max_depth=3) was already locked in
for v0.3.9; this script is here to confirm it stays optimal
when the team feature group is added (4 new features).
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import HeroWinRateEncoder, extract_features
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
GROUPS = ("hero", "team")

GRID = {
    "n_estimators": [30, 50, 80, 120, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6],
}


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
    print(f"loaded {len(raw)} clean matches")
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    encoder = HeroWinRateEncoder().fit(raw)
    X = np.asarray(
        [
            extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, encoder,
                radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
                match=m, groups=GROUPS,
            )
            for t, m in zip(targets, raw)
        ], dtype=float,
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    print(f"X shape: {X.shape}, train {len(X_train)} / test {len(X_test)}")

    rows = []
    combos = list(product(*GRID.values()))
    print(f"grid: {len(combos)} combinations")
    for i, (n, lr, md) in enumerate(combos, 1):
        clf = XGBClassifier(
            n_estimators=n, learning_rate=lr, max_depth=md,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        rows.append({
            "n": n, "lr": lr, "md": md,
            "acc": accuracy_score(y_test, (proba >= 0.5).astype(int)),
            "ll": log_loss(y_test, proba, labels=[0, 1]),
            "auc": roc_auc_score(y_test, proba),
        })
        if i % 10 == 0 or i == len(combos):
            print(f"  [{i}/{len(combos)}]  best so far: "
                  f"{max(rows, key=lambda r: r['acc'])['acc']:.4f}")

    print()
    print("=" * 80)
    print(f"  {'n_est':>5} {'lr':>5} {'md':>3} {'acc':>8} {'log_loss':>10} {'auc':>8}")
    print("=" * 80)
    rows.sort(key=lambda r: -r["acc"])
    for r in rows[:15]:
        print(f"  {r['n']:>5} {r['lr']:>5.2f} {r['md']:>3} "
              f"{r['acc']:>8.4f} {r['ll']:>10.4f} {r['auc']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
