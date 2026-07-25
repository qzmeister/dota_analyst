"""Comprehensive honest grid: model x feature group, encoder fit on train only.

This is the 0.3.10 dev-cycle harness for picking the best
honest baseline.  We test:

  - 4 feature groups: hero, hero+team, hero+lane, hero+team+lane
  - 2 model families: XGBoost (n_est=50, lr=0.1, md=3, plain)
                      and LogReg (C=1.0, lbfgs)
  - 2 encoder regimes: full corpus (mild leak, what 0.3.9 used)
                        and train-only (honest)

We report accuracy / log_loss / AUC for each combination.  The
final pick is whatever wins on the honest / hero+team row.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import HeroWinRateEncoder, extract_features
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"

GROUPS = {
    "hero":           ("hero",),
    "hero+team":      ("hero", "team"),
    "hero+lane":      ("hero", "lane"),
    "hero+team+lane": ("hero", "team", "lane"),
}

MODELS = {
    "xgb": lambda: XGBClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        random_state=42, eval_metric="logloss", verbosity=0,
    ),
    "lr":  lambda: LogisticRegression(
        C=1.0, max_iter=2000, random_state=42, solver="lbfgs",
    ),
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


def build_X(raw, targets, encoder, groups):
    return np.asarray(
        [
            extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, encoder,
                radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
                match=m, groups=groups,
            )
            for t, m in zip(targets, raw)
        ], dtype=float,
    )


def main():
    raw, targets = load()
    print(f"loaded {len(raw)} clean matches")
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    raw_train = [raw[i] for i in idx_train]
    raw_test = [raw[i] for i in idx_test]
    targets_train = [targets[i] for i in idx_train]
    targets_test = [targets[i] for i in idx_test]
    print(f"split: {len(idx_train)} train / {len(idx_test)} test")

    # Pre-build X for all combos of (encoder, group).
    enc_full = HeroWinRateEncoder().fit(raw)
    enc_train = HeroWinRateEncoder().fit(raw_train)
    print(f"hero count: full={len(enc_full._rates)//2} train={len(enc_train._rates)//2}")

    rows = []
    for enc_name, enc in (("full", enc_full), ("train", enc_train)):
        for grp_name, groups in GROUPS.items():
            X = build_X(raw, targets, enc, groups)
            Xtr, Xte = X[idx_train], X[idx_test]
            ytr, yte = y[idx_train], y[idx_test]
            for model_name, mk in MODELS.items():
                t0 = time.perf_counter()
                clf = mk()
                clf.fit(Xtr, ytr)
                proba = clf.predict_proba(Xte)[:, 1]
                elapsed = time.perf_counter() - t0
                rows.append({
                    "encoder": enc_name, "groups": grp_name, "F": X.shape[1],
                    "model": model_name,
                    "acc": accuracy_score(yte, (proba >= 0.5).astype(int)),
                    "ll": log_loss(yte, proba, labels=[0, 1]),
                    "auc": roc_auc_score(yte, proba),
                    "sec": elapsed,
                })

    print()
    print("=" * 90)
    print(f"  {'encoder':<7} {'groups':<16} {'F':>3} {'model':<5} {'acc':>8} {'log_loss':>10} {'auc':>8}   time")
    print("=" * 90)
    for r in rows:
        print(f"  {r['encoder']:<7} {r['groups']:<16} {r['F']:>3} {r['model']:<5} "
              f"{r['acc']:>8.4f} {r['ll']:>10.4f} {r['auc']:>8.4f}   {r['sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
