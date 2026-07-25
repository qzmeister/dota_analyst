"""Pin down: does team encoding actually help honestly?

Two questions:
  1) Full corpus encoder + team:   is the team signal real,
     or just hero-side encoding?
  2) Train-only encoder + team:     is the team signal still
     there when the encoder has not seen the test set?
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
from business.ml.features import HeroWinRateEncoder, extract_features
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
XGB_PARAMS = dict(n_estimators=50, learning_rate=0.1, max_depth=3,
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
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    # Fit encoder on FULL corpus (the v0.3.9 default).
    enc_full = HeroWinRateEncoder().fit(raw)
    # Fit encoder on TRAIN only (honest).
    enc_train = HeroWinRateEncoder().fit([raw[i] for i in idx_train])

    rows = []
    for enc_name, enc in (("full", enc_full), ("train", enc_train)):
        for grp_name, groups in (("hero", ("hero",)), ("hero+team", ("hero", "team"))):
            X = build_X(raw, targets, enc, groups)
            Xtr, Xte = X[idx_train], X[idx_test]
            ytr, yte = y[idx_train], y[idx_test]
            clf = XGBClassifier(**XGB_PARAMS)
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
            rows.append({
                "encoder": enc_name, "groups": grp_name, "F": X.shape[1],
                "acc": accuracy_score(yte, (proba >= 0.5).astype(int)),
                "ll": log_loss(yte, proba, labels=[0, 1]),
                "auc": roc_auc_score(yte, proba),
            })
    print(f"  {'encoder':<8} {'groups':<10} {'F':>3} {'acc':>8} {'log_loss':>10} {'auc':>8}")
    for r in rows:
        print(f"  {r['encoder']:<8} {r['groups']:<10} {r['F']:>3} "
              f"{r['acc']:>8.4f} {r['ll']:>10.4f} {r['auc']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
