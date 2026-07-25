"""Debug script: inspect what features look like for train vs test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    print(f"total matches: {len(targets)}")

    y = np.asarray([t.winner for t in targets], dtype=int)
    print(f"label balance: radiant={y.mean():.3f}")

    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    train_raw = [matched_raw[i] for i in idx_train]
    train_t = [targets[i] for i in idx_train]
    test_t = [targets[i] for i in idx_test]
    test_raw = [matched_raw[i] for i in idx_test]

    enc = HeroWinRateEncoder().fit(train_raw)
    print(f"team_encoder._rates has {len(enc.team_encoder._rates)} teams")
    print(f"team_encoder._global_rate = {enc.team_encoder._global_rate:.4f}")

    # How many test matches have a team_id known to the encoder?
    test_team_ids = [t.radiant_team_id for t in test_t] + [t.dire_team_id for t in test_t]
    known = sum(1 for tid in test_team_ids if tid is not None and tid in enc.team_encoder._rates)
    print(f"test team_ids: {sum(1 for x in test_team_ids if x is not None)} non-null, "
          f"{known} known to encoder")

    # X stats
    Xtr = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc,
                          t.radiant_team_id, t.dire_team_id) for t in train_t],
        dtype=float,
    )
    Xte = np.asarray(
        [extract_features(t.radiant_hero_ids, t.dire_hero_ids, enc,
                          t.radiant_team_id, t.dire_team_id) for t in test_t],
        dtype=float,
    )
    ytr, yte = y[idx_train], y[idx_test]
    print(f"Xtr shape: {Xtr.shape}, Xte shape: {Xte.shape}")
    print(f"Xtr mean: {Xtr.mean(axis=0)[:5]}")
    print(f"Xte mean: {Xte.mean(axis=0)[:5]}")
    print(f"Xtr std:  {Xtr.std(axis=0)[:5]}")
    print(f"Xte std:  {Xte.std(axis=0)[:5]}")

    # Train + score
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    m.fit(Xtr, ytr)
    p_te = m.predict_proba(Xte)[:, 1]
    print(f"  acc={accuracy_score(yte, (p_te >= 0.5).astype(int)):.4f}  "
          f"log_loss={log_loss(yte, p_te, labels=[0,1]):.4f}  "
          f"auc={roc_auc_score(yte, p_te):.4f}")

    # Sanity: does the model work on TRAIN (in-sample)?
    p_tr = m.predict_proba(Xtr)[:, 1]
    print(f"  in-sample acc={accuracy_score(ytr, (p_tr >= 0.5).astype(int)):.4f}  "
          f"log_loss={log_loss(ytr, p_tr, labels=[0,1]):.4f}")

    # Try training without team features (13 features only)
    from business.ml.features import extract_features as _ef
    Xtr13 = np.asarray(
        [_ef(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in train_t],
        dtype=float,
    )
    Xte13 = np.asarray(
        [_ef(t.radiant_hero_ids, t.dire_hero_ids, enc) for t in test_t],
        dtype=float,
    )
    m13 = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    m13.fit(Xtr13, ytr)
    p_te13 = m13.predict_proba(Xte13)[:, 1]
    print(f"\n  13 features only (no team): acc={accuracy_score(yte, (p_te13 >= 0.5).astype(int)):.4f}  "
          f"log_loss={log_loss(yte, p_te13, labels=[0,1]):.4f}  "
          f"auc={roc_auc_score(yte, p_te13):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
