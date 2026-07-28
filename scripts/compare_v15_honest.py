"""Single 80/20 split with HONEST encoder fit on train (mirrors v15's protocol).

Use to compare apples-to-apples with v15's reported honest_80_20_acc_target_enc.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    FEATURE_GROUPS, HeroWinRateEncoder, PlayerWinRateEncoder, extract_features,
)
from business.ml.targets import extract_target

DATA = ROOT / "ml_data" / "full_matches"

ALL_GROUPS = ["hero", "team", "lane", "matchup", "patch", "player"]


def load():
    raw, targets = [], []
    for p in sorted(DATA.glob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except: continue
        t = extract_target(d)
        if t is None: continue
        raw.append(d); targets.append(t)
    return raw, targets


def main():
    raw, targets = load()
    n = len(raw)
    y = np.asarray([t.winner for t in targets], dtype=int)
    print(f"loaded {n} matches", flush=True)

    # v15 protocol: stratified 80/20
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    # Honest encoder: fit on train ONLY
    train_raw = [raw[i] for i in idx_train]
    enc = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)
    enc.player_encoder = PlayerWinRateEncoder(smoothing=5.0, min_samples=3).fit(train_raw)

    def X_for(groups, idxs):
        F = sum(len(FEATURE_GROUPS[g]) for g in groups)
        X = np.empty((len(idxs), F), dtype=float)
        for k, i in enumerate(idxs):
            t = targets[i]; m = raw[i]
            X[k] = extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, enc,
                radiant_team_id=t.radiant_team_id, dire_team_id=t.dire_team_id,
                match=m, groups=tuple(groups),
            )
        return X

    # Test ALL feature group combinations
    from itertools import combinations
    group_choices = []
    for k in range(1, 7):
        for c in combinations(ALL_GROUPS, k):
            group_choices.append(list(c))
    # Always include [hero, player] (v15 baseline)
    if ["hero", "player"] not in group_choices:
        group_choices.append(["hero", "player"])

    # Test top model configs
    model_configs = [
        ("xgb_n50_d3_lr0.1",   lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=50, max_depth=3, learning_rate=0.1,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("xgb_n80_d3_lr0.1",   lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=80, max_depth=3, learning_rate=0.1,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("xgb_n100_d3_lr0.05", lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=100, max_depth=3, learning_rate=0.05,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("xgb_n80_d4_lr0.05",  lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=80, max_depth=4, learning_rate=0.05,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("xgb_n200_d3_lr0.05", lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=200, max_depth=3, learning_rate=0.05,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("xgb_n300_d3_lr0.05", lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=300, max_depth=3, learning_rate=0.05,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("xgb_n500_d3_lr0.05", lambda: xgb.XGBClassifier(objective="binary:logistic",
                                                        n_estimators=500, max_depth=3, learning_rate=0.05,
                                                        tree_method="hist", verbosity=0, random_state=42)),
        ("logreg_c0.5",        lambda: LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
        ("logreg_c1.0",        lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        ("logreg_c2.0",        lambda: LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
        ("logreg_c0.5_cal",    lambda: CalibratedClassifierCV(LogisticRegression(C=0.5, max_iter=2000, random_state=42), method="sigmoid", cv=3)),
        ("logreg_c1.0_cal",    lambda: CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=2000, random_state=42), method="sigmoid", cv=3)),
        ("logreg_c2.0_cal",    lambda: CalibratedClassifierCV(LogisticRegression(C=2.0, max_iter=2000, random_state=42), method="sigmoid", cv=3)),
    ]

    results = []
    for g in group_choices:
        X_tr = X_for(g, idx_train)
        X_te = X_for(g, idx_test)
        for name, mk in model_configs:
            try:
                t0 = time.perf_counter()
                m = mk()
                m.fit(X_tr, y[idx_train])
                proba = m.predict_proba(X_te)[:, 1]
                pred = (proba >= 0.5).astype(int)
                acc = accuracy_score(y[idx_test], pred)
                ll = log_loss(y[idx_test], np.clip(proba, 1e-9, 1-1e-9))
                auc = roc_auc_score(y[idx_test], proba)
                t = time.perf_counter() - t0
                results.append({"name": name, "groups": g, "acc": acc, "ll": ll, "auc": auc, "t": t,
                                "n_features": X_tr.shape[1]})
                print(f"  {name:<22} {str(g):<60} acc={acc*100:.2f}% ll={ll:.4f} auc={auc:.4f} F={X_tr.shape[1]} t={t:.1f}s", flush=True)
            except Exception as e:
                print(f"  FAIL {name} {g}: {e}", flush=True)

    print("\n=== TOP 10 BY ACCURACY ===", flush=True)
    results.sort(key=lambda r: -r["acc"])
    for r in results[:10]:
        print(f"  acc={r['acc']*100:.2f}% ll={r['ll']:.4f} auc={r['auc']:.4f} {r['name']} {r['groups']}", flush=True)


if __name__ == "__main__":
    main()
