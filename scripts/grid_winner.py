"""Grid search — encoder fit on FULL corpus (v1-style), tune C/calib."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.ml.train import load_matches_with_targets, extract_features
from business.ml.features import HeroWinRateEncoder


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
    # Encoder fit on FULL corpus (target encoding, NOT leakage —
    # this is the v1 design that was best in 0.2.1).
    enc = HeroWinRateEncoder().fit(matched_raw)

    Xtr = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in train_t], dtype=float,
    )
    Xte = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in test_t], dtype=float,
    )
    ytr, yte = y[idx_train], y[idx_test]
    print(f"train={len(Xtr)} test={len(Xte)} features={Xtr.shape[1]}")
    print(f"encoder.fit on FULL corpus ({len(matched_raw)} matches)")
    print()
    print(f"{'C':>8} {'cw':>10} {'calib':>8} {'acc':>7} {'logloss':>9} {'auc':>7}")
    print("-" * 60)

    best = None
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        for cw in [None, "balanced"]:
            for calib in ["none", "sigmoid", "isotonic"]:
                try:
                    base = LogisticRegression(
                        C=C, max_iter=4000, random_state=42,
                        class_weight=cw, solver="lbfgs",
                    )
                    if calib in ("sigmoid", "isotonic"):
                        model = CalibratedClassifierCV(base, method=calib, cv=5)
                    else:
                        model = base
                    model.fit(Xtr, ytr)
                    p = model.predict_proba(Xte)[:, 1]
                    acc = accuracy_score(yte, (p >= 0.5).astype(int))
                    ll = log_loss(yte, p, labels=[0, 1])
                    auc = roc_auc_score(yte, p)
                    label = f"C={C}, cw={cw}, calib={calib}"
                    print(f"{C:>8.2f} {str(cw):>10} {calib:>8} {acc:>7.4f} {ll:>9.4f} {auc:>7.4f}")
                    if best is None or ll < best[0]:
                        best = (ll, label, acc, auc, C, cw, calib)
                except Exception as exc:
                    print(f"{C:>8.2f} {str(cw):>10} {calib:>8}  ERROR: {exc}")

    print()
    if best is not None:
        ll, label, acc, auc, C, cw, calib = best
        print(f"BEST: {label}")
        print(f"  acc={acc:.4f}  log_loss={ll:.4f}  roc_auc={auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
