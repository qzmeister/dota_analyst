"""Compare leakage-safe prematch models on the same chronological holdout.

This script never overwrites the production model.  It is intended to decide
whether a candidate is genuinely better before promotion.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_temporal_prematch as temporal


REPORT_PATH = Path("ml_models/benchmark_report.json")


def load_dataset(include_state=False):
    matches = []
    for path in temporal.MATCHES_DIR.glob("*.json"):
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
            if match.get("start_date") and len(match["radiant"].get("player_performances", [])) == 5:
                matches.append(match)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    matches.sort(key=lambda item: (item["start_date"], item["match_id"]))

    def new_record():
        return {"wins": 0.0, "matches": 0.0, "elo": 1500.0, "recent": []}

    state = {kind: defaultdict(new_record) for kind in (
        "teams", "players", "heroes", "patch_heroes", "hero_pairs", "rosters", "head_to_head"
    )}
    rows, targets = [], []
    index = 0
    while index < len(matches):
        timestamp = matches[index]["start_date"]
        end = index
        while end < len(matches) and matches[end]["start_date"] == timestamp:
            end += 1
        for match in matches[index:end]:
            rows.append(temporal.build_features(match, state))
            targets.append(int(bool(match["radiant_victory"])))
        for match in matches[index:end]:
            temporal.update_state(match, state)
        index = end
    names = sorted(rows[0])
    dataset = (np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float), np.array(targets), names)
    return (*dataset, temporal.serialise_state(state)) if include_state else dataset


def metrics(y, probabilities):
    return {
        "accuracy": round(float(accuracy_score(y, probabilities >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        "brier": round(float(brier_score_loss(y, probabilities)), 4),
    }


def main():
    X, y, names = load_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.15, max_iter=2000)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=10, max_features=0.7, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=12, l2_regularization=4.0, random_state=42
        ),
        "xgboost_current": XGBClassifier(
            n_estimators=260, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.9,
            min_child_weight=10, reg_alpha=0.3, reg_lambda=3.0, random_state=42, eval_metric="logloss"
        ),
        "xgboost_regularized": XGBClassifier(
            n_estimators=450, max_depth=2, learning_rate=0.025, subsample=0.9, colsample_bytree=0.75,
            min_child_weight=16, reg_alpha=0.8, reg_lambda=7.0, random_state=42, eval_metric="logloss"
        ),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        results[name] = metrics(y_test, model.predict_proba(X_test)[:, 1])
        print(f"{name}: {results[name]}")
    report = {
        "samples": int(len(y)), "holdout_samples": int(len(y_test)), "features": len(names),
        "split": "chronological final 20%", "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
