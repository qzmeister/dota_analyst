"""Grid search: XGBoost vs LogReg for the winner head."""
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
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed", file=sys.stderr)
        return 1

    data_dir = _ROOT / "ml_data" / "full_matches"
    matched_raw, targets = load_matches_with_targets(data_dir)
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    train_t = [targets[i] for i in idx_train]
    test_t = [targets[i] for i in idx_test]
    enc = HeroWinRateEncoder().fit(matched_raw)

    Xtr = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in train_t], dtype=float,
    )
    Xte = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in test_t], dtype=float,
    )
    ytr, yte = y[idx_train], y[idx_test]
    print(f"train={len(Xtr)} test={len(Xte)} features={Xtr.shape[1]}")
    print()

    # 1) LogReg baseline (v1 design)
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    print(f"  LogReg C=1.0 (v1 baseline)        acc={accuracy_score(yte,(p>=0.5).astype(int)):.4f}  "
          f"log_loss={log_loss(yte,p,labels=[0,1]):.4f}  auc={roc_auc_score(yte,p):.4f}")
    print()

    # 2) XGBoost grid
    print(f"{'n_est':>6} {'lr':>6} {'md':>4} {'calib':>8} {'acc':>7} {'logloss':>9} {'auc':>7}")
    print("-" * 60)
    best = None
    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.2]:
            for md in [3, 5, 7]:
                for calib in ["none", "sigmoid"]:
                    try:
                        base = XGBClassifier(
                            n_estimators=n_est, learning_rate=lr, max_depth=md,
                            random_state=42, eval_metric="logloss",
                            use_label_encoder=False, verbosity=0,
                        )
                        if calib == "sigmoid":
                            model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
                        else:
                            model = base
                        model.fit(Xtr, ytr)
                        p = model.predict_proba(Xte)[:, 1]
                        acc = accuracy_score(yte, (p >= 0.5).astype(int))
                        ll = log_loss(yte, p, labels=[0, 1])
                        auc = roc_auc_score(yte, p)
                        print(f"{n_est:>6} {lr:>6.2f} {md:>4} {calib:>8} {acc:>7.4f} {ll:>9.4f} {auc:>7.4f}")
                        if best is None or ll < best[0]:
                            best = (ll, n_est, lr, md, calib, acc, auc)
                    except Exception as exc:
                        print(f"{n_est:>6} {lr:>6.2f} {md:>4} {calib:>8}  ERROR: {exc}")

    print()
    if best is not None:
        ll, n_est, lr, md, calib, acc, auc = best
        print(f"BEST: n_est={n_est}, lr={lr}, max_depth={md}, calib={calib}")
        print(f"  acc={acc:.4f}  log_loss={ll:.4f}  auc={auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
