"""Honest 5-fold CV for the winner target.

Refits the encoder on the train fold ONLY (not the test fold), so the
target-encoded features don't carry the test row's outcome.  This is
the only way to measure "what production would see on a new match".

For each top config from the leaky grid, report the honest metric.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    FEATURE_GROUPS, HeroWinRateEncoder, PlayerWinRateEncoder, extract_features,
)
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"
OUT_PATH = ROOT / "scripts" / "grid_honest_winner.jsonl"
RANDOM_STATE = 42
N_SPLITS = 5

ALL_GROUPS = ["hero", "team", "lane", "matchup", "patch", "player"]


def load():
    raw, targets = [], []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = extract_target(d)
        if t is None:
            continue
        raw.append(d); targets.append(t)
    return raw, targets


def build_features_honest(raw, targets, train_idx, test_idx, groups):
    train_raw = [raw[i] for i in train_idx]
    enc = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)
    enc.player_encoder = PlayerWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)
    F = sum(len(FEATURE_GROUPS[g]) for g in groups)
    X_tr = np.empty((len(train_idx), F), dtype=float)
    X_te = np.empty((len(test_idx), F), dtype=float)
    for k, i in enumerate(train_idx):
        t = targets[i]
        m = raw[i]
        X_tr[k] = extract_features(
            t.radiant_hero_ids, t.dire_hero_ids, enc,
            radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
            match=m, groups=tuple(groups),
        )
    for k, i in enumerate(test_idx):
        t = targets[i]
        m = raw[i]
        X_te[k] = extract_features(
            t.radiant_hero_ids, t.dire_hero_ids, enc,
            radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
            match=m, groups=tuple(groups),
        )
    return X_tr, X_te


def make_model(name, kw):
    n = name.lower()
    if "xgb" in n:
        return xgb.XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE,
                                 tree_method="hist", verbosity=0, **kw)
    if "cal" in n:
        base = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, **kw)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if "logreg" in n or "lr" in n:
        return LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, **kw)
    raise ValueError(f"unknown: {name}")


def cv_honest(name, kw, raw, targets, y, groups, n_splits=N_SPLITS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, lls, aucs, times = [], [], [], []
    for fold, (tr, te) in enumerate(kf.split(np.arange(len(raw)))):
        t0 = time.perf_counter()
        X_tr, X_te = build_features_honest(raw, targets, tr, te, groups)
        m = make_model(name, kw)
        m.fit(X_tr, y[tr])
        proba = m.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        lls.append(log_loss(y[te], np.clip(proba, 1e-9, 1 - 1e-9)))
        aucs.append(roc_auc_score(y[te], proba))
        times.append(time.perf_counter() - t0)
    return {
        "n_folds": n_splits,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "log_loss_mean": float(np.mean(lls)),
        "log_loss_std": float(np.std(lls)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "train_s_mean": float(np.mean(times)),
    }


def main():
    raw, targets = load()
    n = len(raw)
    y = np.asarray([t.winner for t in targets], dtype=int)
    print(f"loaded {n} matches", flush=True)

    # Top configs from the leaky grid
    configs = [
        # (name, kwargs)
        ("xgb_n50_d3_lr0.1",  dict(n_estimators=50, max_depth=3, learning_rate=0.1)),
        ("xgb_n80_d3_lr0.1",  dict(n_estimators=80, max_depth=3, learning_rate=0.1)),
        ("xgb_n100_d3_lr0.05", dict(n_estimators=100, max_depth=3, learning_rate=0.05)),
        ("xgb_n80_d4_lr0.05", dict(n_estimators=80, max_depth=4, learning_rate=0.05)),
        ("xgb_n150_d3_lr0.1", dict(n_estimators=150, max_depth=3, learning_rate=0.1)),
        ("xgb_n200_d3_lr0.05", dict(n_estimators=200, max_depth=3, learning_rate=0.05)),
        ("xgb_n300_d3_lr0.05", dict(n_estimators=300, max_depth=3, learning_rate=0.05)),
        ("xgb_n100_d4_lr0.1", dict(n_estimators=100, max_depth=4, learning_rate=0.1)),
        ("xgb_n200_d4_lr0.05", dict(n_estimators=200, max_depth=4, learning_rate=0.05)),
        ("logreg_c1.0_cal",  dict(C=1.0)),
        ("logreg_c0.5_cal",  dict(C=0.5)),
        ("logreg_c2.0_cal",  dict(C=2.0)),
        ("logreg_c0.5",      dict(C=0.5)),
        ("logreg_c1.0",      dict(C=1.0)),
        ("logreg_c2.0",      dict(C=2.0)),
    ]
    # Top 4 group choices from the leaky grid
    group_choices = [
        list(ALL_GROUPS),  # all 34
        ["hero", "team", "player", "matchup"],  # 24
        ["hero", "player", "matchup"],  # 20
        ["hero", "team", "lane", "matchup", "player"],  # 31
        ["hero", "team", "player"],  # 21
    ]
    out_fh = OUT_PATH.open("w", encoding="utf-8")
    counter = 0
    t_start = time.perf_counter()
    results = []
    for g in group_choices:
        for name, kw in configs:
            try:
                r = cv_honest(name, kw, raw, targets, y, g)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)
                continue
            rec = {
                "target": "winner",
                "name": name, "kw": kw, "groups": list(g),
                "n_features": sum(len(FEATURE_GROUPS[x]) for x in g),
                **r,
            }
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            results.append(rec)
            counter += 1
            elapsed = time.perf_counter() - t_start
            print(f"  [{counter:>3}] {name:<22} {g} -> acc={r['acc_mean']*100:.2f}%±{r['acc_std']*100:.2f}% "
                  f"ll={r['log_loss_mean']:.4f} auc={r['auc_mean']:.4f} t={r['train_s_mean']:.2f}s  "
                  f"({elapsed:.0f}s)", flush=True)
    out_fh.close()

    print("\n=== TOP 5 BY ACCURACY ===", flush=True)
    results.sort(key=lambda r: -r["acc_mean"])
    for r in results[:5]:
        print(f"  acc={r['acc_mean']*100:.2f}%  ll={r['log_loss_mean']:.4f}  auc={r['auc_mean']:.4f}  "
              f"{r['name']}  {r['groups']}", flush=True)
    print("\n=== TOP 5 BY LOG_LOSS ===", flush=True)
    results.sort(key=lambda r: r["log_loss_mean"])
    for r in results[:5]:
        print(f"  acc={r['acc_mean']*100:.2f}%  ll={r['log_loss_mean']:.4f}  auc={r['auc_mean']:.4f}  "
              f"{r['name']}  {r['groups']}", flush=True)


if __name__ == "__main__":
    main()
