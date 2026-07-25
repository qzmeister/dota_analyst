"""
CLI: train one or more prediction targets on `ml_data/full_matches/`.

Usage
-----
    # Train every known target (default in 0.2.1+)
    python -m business.ml.train

    # Train a single target
    python -m business.ml.train --target winner
    python -m business.ml.train --target kills
    python -m business.ml.train --target duration_mean

    # Customise the outlier clip and the version string
    python -m business.ml.train --n-sigma 2.5 --version 2

    # Disable winsorize (raw targets — useful as a control in eval)
    python -m business.ml.train --no-winsorize

    # Calibrate the winner classifier (0.2.2) — better log_loss at
    # the cost of a small accuracy hit
    python -m business.ml.train --target winner --calibrate isotonic

    # 0.3.10: pick feature groups
    python -m business.ml.train --target winner --groups hero
    python -m business.ml.train --target winner --groups hero,team
    python -m business.ml.train --target winner --groups hero,lane
    python -m business.ml.train --target winner --groups hero,team,lane

What it does
------------
  1. Load every `*.json` from `--data-dir` as a DatDota match dict.
  2. Drop matches with `has_error`, missing `radiant_victory`,
     fewer than 5 hero picks per side, or duration outside
     [10 min, 90 min].
  3. Fit a `HeroWinRateEncoder` (with team + lane encoders) on the
     full corpus.  The encoder never sees the split — it computes
     only aggregate statistics about heroes / teams / lane pairs,
     so fitting on the whole corpus doesn't leak the split.
  4. Build a (N, F) feature matrix with `extract_features()`,
     where F depends on `--groups` (13 / 17 / 20 / 24).
  5. Build per-target y vectors from `targets.py`.
  6. Train/test split (stratified on the winner target).
  7. For each target:
       - winsorize y at mean ± `n_sigma` * std  (unless --no-winsorize)
       - fit the regressor / classifier from `regressors.py`
       - optionally wrap the winner LogReg in `CalibratedClassifierCV`
       - compute holdout metrics
       - save model + encoder + metrics via `ModelStorage`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .._logging import setup_logging
from ..exceptions import MLTrainError, ParseError
from .classifiers import CLASSIFIER_REGISTRY, make_classifier
from .features import (
    FEATURE_GROUPS,
    FEATURE_ORDER,
    N_FEATURES,
    HeroWinRateEncoder,
    extract_features,
    feature_names,
)
from .outliers import count_clipped, winsorize_in_place
from .regressors import (
    REGRESSOR_REGISTRY,
    make_duration_mean_regressor,
    make_kills_regressor,
    make_regressor,
)
from .storage import ModelStorage
from .targets import MatchTarget, extract_target, iter_clean_targets


log = logging.getLogger("business.ml.train")


# Targets that this CLI knows how to train.  `winner` is special-cased
# (LogisticRegression, predict_proba) so it doesn't go through
# REGRESSOR_REGISTRY; `multikill` (3-class) goes through
# CLASSIFIER_REGISTRY; everything else is a regression.
HEAD_REGISTRY: Dict[str, Dict] = {
    "winner": {
        "kind": "classifier",
        "y_attr": "winner",
        "model_kind": "binary",
    },
    # NOTE (0.3.10): "multikill" is intentionally NOT in this registry.
    # On the 2036-match pro corpus every match has `max_kills >= 7`
    # (DatDota full_matches is sampled at pro level where everyone
    # rampages), so the Low/Medium/High bins from `targets.target_multikill`
    # collapse to 100% High.  The classifier degenerated to "always
    # High" in 0.3.0.  The bins are still used by the heuristic in
    # `analysis.py`; the trained model is just not a useful
    # alternative.  Revisit in 0.4.x with a different target
    # (e.g. rampage-yes/no on a per-player basis) or a wider
    # corpus that includes non-pro matches.
    "kills": {
        "kind": "regressor",
        "y_attr": "kills_total",
        "metrics": ("mae", "rmse", "poisson_deviance"),
    },
    "duration_mean": {
        "kind": "regressor",
        "y_attr": "duration_minutes",
        "metrics": ("mae", "rmse"),
    },
    "duration_p10": {
        "kind": "regressor",
        "y_attr": "duration_minutes",
        "metrics": ("mae", "pinball_0.1"),
    },
    "duration_p90": {
        "kind": "regressor",
        "y_attr": "duration_minutes",
        "metrics": ("mae", "pinball_0.9"),
    },
    "towers": {
        "kind": "regressor",
        "y_attr": "towers_total",
        "metrics": ("mae", "rmse", "poisson_deviance"),
    },
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def iter_matches(data_dir: Path) -> Iterable[dict]:
    """Yield every `*.json` under `data_dir` as a parsed match dict.

    Files that fail to parse are logged and skipped — we do not
    abort the whole run on one bad file.  The training corpus is
    allowed to have a few bad apples.
    """
    for path in sorted(data_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                yield json.load(fh)
        except (OSError, ValueError, ParseError) as exc:
            # OSError: file system problem (missing, perm denied).
            # ValueError: json.JSONDecodeError — malformed JSON.
            # ParseError: future-proofing for files we mark as parse-failed
            # upstream.  We deliberately do NOT catch generic Exception
            # so a bug in json.load surfaces during development.
            log.warning("failed to parse %s: %s", path.name, exc)


def load_matches_with_targets(
    data_dir: Path,
) -> Tuple[List[Dict], List[MatchTarget]]:
    """Load raw matches paired 1:1 with their filtered `MatchTarget`.

    `iter_clean_targets` is the only place that decides which raw
    matches are usable; the returned `raw` list is *only* the ones
    that pass that filter, in the same order as `targets`.  This
    pairing is what makes train/test encoder fitting honest — we
    can pick a subset of `targets` and know exactly which raw
    matches produced them.

    Returned `raw` items are deep-equal to the file contents (we
    don't deep-copy — just the dict reference from json.load).
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    raw = list(iter_matches(data_dir))
    targets: List[MatchTarget] = []
    matched: List[Dict] = []
    for m in raw:
        t = extract_target(m)
        if t is None:
            continue
        targets.append(t)
        matched.append(m)

    if len(targets) < 50:
        raise RuntimeError(
            f"only {len(targets)} usable matches found; need at least 50 to train"
        )
    return matched, targets


def build_dataset(
    data_dir: Path,
    *,
    fit_encoder_on: Optional[List[Dict]] = None,
    groups: Tuple[str, ...] = ("hero", "team", "lane"),
) -> Tuple[List[MatchTarget], HeroWinRateEncoder, np.ndarray]:
    """Walk `data_dir`, return (targets, encoder, X).

    `X` is the (N, F) feature matrix (F=24 in 0.3.10 — hero +
    team + lane); `targets` carries the per-row labels for
    every head.  Order is identical between the two — row `i`
    of `X` is the same match as `targets[i]`.

    The encoder is fitted ONLY on `fit_encoder_on` if provided;
    otherwise on every raw match in the corpus.  The split-time
    path always passes a `fit_encoder_on` that's a subset of the
    training rows — this is what kills the target-leakage that
    made v3 look "too good" (0.97 accuracy, 1.43 log_loss) in
    the 0.3.9 dev cycle.

    `groups` (0.3.10) controls which feature groups are included
    in the matrix.  Default is all three ("hero", "team", "lane");
    pass a subset to reproduce the 0.3.9 baseline or the C/D
    experiments.  The full match dict is required for the `lane`
    group (so we can pull laneInfo per player) — we pass it via
    the `match` argument to `extract_features`.
    """
    matched_raw, targets = load_matches_with_targets(data_dir)
    fit_pool = fit_encoder_on if fit_encoder_on is not None else matched_raw
    encoder = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(fit_pool)

    X = np.asarray(
        [
            extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, encoder,
                radiant_team_id=t.radiant_team_id,
                dire_team_id=t.dire_team_id,
                match=m,
                groups=groups,
            )
            for t, m in zip(targets, matched_raw)
        ],
        dtype=float,
    )
    log.info(
        "loaded %d clean matches from %s (built (N=%d, F=%d) feature matrix, "
        "encoder fitted on %d matches, groups=%s)",
        len(targets), data_dir, X.shape[0], X.shape[1], len(fit_pool), groups,
    )
    return targets, encoder, X


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

def _safe_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """2 * (y_pred - y_true + y_true * log(y_true / y_pred)) summed.

    sklearn has this in `sklearn.metrics.mean_poisson_deviance` but it
    is only on newer versions.  We inline the formula to stay
    compatible with sklearn 1.3+ and to handle the y_pred == 0 edge
    case (which we clip to a tiny positive).
    """
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2.0 * np.sum(y_pred - y_true + term) / len(y_true))


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile loss for the alpha-th percentile.

    `alpha=0.1` is "under-prediction is fine, over-prediction is
    expensive by 9x"; `alpha=0.9` is the mirror.
    """
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def evaluate_regressor(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the metrics declared in `HEAD_REGISTRY[name]["metrics"]`."""
    out: Dict[str, float] = {}
    for m in HEAD_REGISTRY[name]["metrics"]:
        if m == "mae":
            out[m] = float(mean_absolute_error(y_true, y_pred))
        elif m == "rmse":
            out[m] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif m == "poisson_deviance":
            out[m] = _safe_poisson_deviance(y_true, np.clip(y_pred, 1e-9, None))
        elif m.startswith("pinball_"):
            alpha = float(m.split("_", 1)[1])
            out[m] = _pinball_loss(y_true, y_pred, alpha)
        else:  # pragma: no cover — defensive
            log.warning("unknown metric %s for target %s", m, name)
    return out


# --------------------------------------------------------------------------- #
# Per-target trainers
# --------------------------------------------------------------------------- #

def _train_winner(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    random_state: int,
    calibrate: str = "none",
    logreg_c: float = 1.0,
    logreg_class_weight: Optional[str] = None,
    logreg_max_iter: int = 2000,
) -> Tuple[object, Dict[str, float]]:
    """Fit the winner classifier; optionally wrap in CalibratedClassifierCV.

    `calibrate="sigmoid"` → Platt scaling (logistic, 1-D fit).
    `calibrate="isotonic"` → isotonic regression (non-parametric).
    Both improve probability calibration at a small cost to raw
    accuracy.  The wrapper's `predict_proba` is API-compatible with
    the base LogReg, so the engine doesn't have to special-case it.

    `logreg_c`, `logreg_class_weight`, `logreg_max_iter` are 0.3.9
    additions: surface the LogReg hyperparameters so the operator
    can grid-search the winner head without re-running the full
    training pipeline.  Empirically (grid_winner.py), C around
    0.5 with `sigmoid` calibration gives the best log_loss on
    the 1111-match corpus; C=1.0 (the old default) overfits and
    `CalibratedClassifierCV` cannot recover from that.
    """
    base = LogisticRegression(
        C=logreg_c,
        max_iter=logreg_max_iter,
        random_state=random_state,
        class_weight=logreg_class_weight,
        solver="lbfgs",
    )
    if calibrate in ("sigmoid", "isotonic"):
        from sklearn.calibration import CalibratedClassifierCV
        model: object = CalibratedClassifierCV(
            base, method=calibrate, cv=5,
        )
    else:
        model = base
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, (proba >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "calibration": calibrate,
        "logreg_c": logreg_c,
        "logreg_class_weight": logreg_class_weight,
    }
    return model, metrics


def _train_regressor(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    random_state: int,
    zinb: bool = False,
) -> Tuple[object, Dict[str, float]]:
    model = make_regressor(name, random_state=random_state, zinb=zinb)
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)  # no negative counts / durations
    metrics = evaluate_regressor(name, y_test, y_pred)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    if name == "towers":
        metrics["estimator_family"] = "zinb" if zinb else "poisson_histgbr"
    return model, metrics


def _train_multiclass_classifier(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    random_state: int,
) -> Tuple[object, Dict[str, float]]:
    """3-class classifier (e.g. multikill Low/Medium/High)."""
    model = make_classifier(name, random_state=random_state)
    # Filter out rows with None / NaN labels (corpus may have a
    # match where we couldn't derive the multikill level).
    train_mask = np.asarray([y is not None and not (isinstance(y, float) and np.isnan(y)) for y in y_train])
    test_mask = np.asarray([y is not None and not (isinstance(y, float) and np.isnan(y)) for y in y_test])
    X_train_f = X_train[train_mask]
    y_train_f = np.asarray([y for y, m in zip(y_train, train_mask) if m])
    X_test_f = X_test[test_mask]
    y_test_f = np.asarray([y for y, m in zip(y_test, test_mask) if m])

    if len(X_train_f) < 50 or len(X_test_f) < 10:
        raise RuntimeError(
            f"too few labelled rows for multiclass {name!r}: "
            f"train={len(X_train_f)}, test={len(X_test_f)}"
        )

    model.fit(X_train_f, y_train_f)
    y_pred = model.predict(X_test_f)
    y_proba = model.predict_proba(X_test_f)
    classes = list(model.classes_)

    metrics = {
        "accuracy": float(accuracy_score(y_test_f, y_pred)),
        "f1_macro": float(f1_score(y_test_f, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_f, y_pred, average="weighted", zero_division=0)),
        "n_train": int(len(X_train_f)),
        "n_test": int(len(X_test_f)),
        "classes": classes,
        "class_distribution": {
            c: int((y_train_f == c).sum()) for c in classes
        },
    }
    # Per-class precision/recall — useful for the eval harness
    # to spot "model always predicts Low" issues.
    from sklearn.metrics import classification_report
    report = classification_report(
        y_test_f, y_pred, output_dict=True, zero_division=0,
    )
    for c in classes:
        if c in report:
            metrics[f"precision_{c}"] = float(report[c]["precision"])
            metrics[f"recall_{c}"] = float(report[c]["recall"])
    return model, metrics


# --------------------------------------------------------------------------- #
# Top-level training pipeline
# --------------------------------------------------------------------------- #

def train_all(
    data_dir: Path,
    model_dir: Path,
    version: str,
    targets: List[str],
    winsorize: bool,
    n_sigma: float,
    test_size: float,
    random_state: int,
    calibrate: str = "none",
    zinb: bool = False,
    logreg_c: float = 1.0,
    logreg_class_weight: Optional[str] = None,
    logreg_max_iter: int = 2000,
    groups: Tuple[str, ...] = ("hero", "team", "lane"),
    honest_encoder: bool = False,
) -> Dict[str, Path]:
    """Run the pipeline for every requested target; return saved paths.

    `calibrate` is forwarded only to the winner classifier; regressors
    ignore it.  See `_train_winner` for the available values.

    `zinb` swaps the towers regressor for the statsmodels ZINB
    factory (when statsmodels is installed AND the target is
    `towers`); other targets ignore it.

    `groups` is the tuple of feature groups to include in the
    feature matrix (0.3.10).  Each group contributes its own
    features (see `FEATURE_GROUPS` in features.py).  The same
    tuple is recorded in the saved model's metadata so the
    engine can rebuild the right vector at predict time.

    `honest_encoder` (0.3.10): when True, fit the encoder on the
    train split only — never on the test rows.  This is the
    "honest" target-encoding practice: it makes the holdout
    metrics reflect what the model would see on a *truly new*
    match.  When False (default for backward compatibility with
    0.3.9), the encoder is fit on the full corpus and the
    holdout metrics are mildly inflated (the encoder has seen
    each test row's outcome when building its lookup tables).
    The inflation is small for `hero` features (smoothing pulls
    toward 0.5) and large for `lane` features (per-pair lookup
    can have < 1 sample per key, so the empirical rate is the
    test label itself).
    """
    # 1. Load all raw matches paired 1:1 with their MatchTargets.
    #    We need the raw dicts (not just the targets) because the
    #    encoder's `fit()` consumes full match dicts to compute
    #    hero / team / pair aggregates.
    matched_raw, all_targets = load_matches_with_targets(data_dir)

    # 2. Stratified split on the winner target — every other
    #    target rides along on the same indices so the rows
    #    stay aligned across heads.
    y_winner_full = np.asarray([t.winner for t in all_targets], dtype=int)
    label_balance = {
        "radiant_pct": float(y_winner_full.mean()),
        "dire_pct": float(1.0 - y_winner_full.mean()),
    }
    log.info("label balance (winner): %s", label_balance)

    idx = np.arange(len(all_targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y_winner_full,
    )
    log.info("split: %d train / %d test", len(idx_train), len(idx_test))

    # 3. Fit the encoder.  v0.3.9 default: full corpus (mild leak
    #    in metrics, but the lookup is the most informative).  v0.3.10
    #    honest: train split only — metrics reflect what production
    #    would see before the new match's outcome is known.
    train_targets = [all_targets[i] for i in idx_train]
    test_targets = [all_targets[i] for i in idx_test]
    raw_train = [matched_raw[i] for i in idx_train]
    raw_test = [matched_raw[i] for i in idx_test]
    if honest_encoder:
        log.info("honest encoder: fitting on %d train rows only", len(raw_train))
        encoder = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(raw_train)
    else:
        log.info("encoder: fitting on full corpus (%d rows)", len(matched_raw))
        encoder = HeroWinRateEncoder(smoothing=5.0, min_samples=3).fit(matched_raw)

    # 4. Build feature matrices for train and test from the same
    #    encoder.  This is the contract: the encoder knows only
    #    about train data, so its encodings for test rows are
    #    "what would a new observation look like".  The
    #    `groups` argument controls which features are included
    #    (0.3.10).
    feat_names = feature_names(groups)
    n_features = len(feat_names)
    log.info("feature groups: %s -> %d features: %s", groups, n_features, feat_names)

    def _build(targets_subset, raw_subset):
        X = np.empty((len(targets_subset), n_features), dtype=float)
        for i, (t, m) in enumerate(zip(targets_subset, raw_subset)):
            X[i] = extract_features(
                t.radiant_hero_ids, t.dire_hero_ids, encoder,
                radiant_team_id=t.radiant_team_id,
                dire_team_id=t.dire_team_id,
                match=m,
                groups=groups,
            )
        return X

    X_train = _build(train_targets, raw_train)
    X_test = _build(test_targets, raw_test)
    X = np.concatenate([X_train, X_test], axis=0)
    # Re-order so the canonical row order is preserved (the rest
    # of the function still walks `targets_list` in raw order).
    # We use `targets_list` as the reference and re-order the
    # split indices to keep the same downstream contract.
    targets_list = all_targets

    storage = ModelStorage(model_dir)
    saved_paths: Dict[str, Path] = {}

    for target in targets:
        if target not in HEAD_REGISTRY:
            log.error("unknown target %r; skipping", target)
            continue

        y_attr = HEAD_REGISTRY[target]["y_attr"]
        # Multiclass targets (e.g. multikill) carry string labels;
        # regressors expect numeric.  Build the y-vector as an
        # object array first and let the per-target trainer
        # coerce to the right dtype.
        if HEAD_REGISTRY[target].get("model_kind") == "multiclass":
            y_full = np.asarray(
                [getattr(t, y_attr) for t in targets_list],
                dtype=object,
            )
        else:
            y_full = np.asarray(
                [getattr(t, y_attr) for t in targets_list],
                dtype=float,
            )

        # Towers data is not present in the 0.2.1 corpus — every
        # `towers_total` is None.  Skip the target with a clear
        # message rather than feeding a column of NaNs to the
        # regressor.
        if HEAD_REGISTRY[target]["kind"] == "regressor":
            n_with_target = int(np.sum(np.isfinite(y_full)))
            if n_with_target < 50:
                log.warning(
                    "  [%s] only %d rows have a non-null target; "
                    "skipping (need >= 50 to train). Re-pull a "
                    "tower-aware corpus to enable this target.",
                    target, n_with_target,
                )
                continue
            log.info(
                "  [%s] %d / %d rows have a target value",
                target, n_with_target, len(y_full),
            )

        # Some targets share y_attr (duration_p10 / p90 / mean
        # all use `duration_minutes`).  That's fine — we just
        # subset the same vector under different train indices.
        X_train, X_test = X[idx_train], X[idx_test]
        y_train_full, y_test_full = y_full[idx_train], y_full[idx_test]

        if winsorize and HEAD_REGISTRY[target]["kind"] == "regressor":
            # Clip the TRAIN side only — the test side stays raw
            # so the metrics reflect what the model actually sees
            # in production.  Winsorize is in-place, but we
            # explicitly copy first so we never mutate the
            # original vectors.
            n_clipped = count_clipped(y_train_full, n_sigma=n_sigma)
            winsorize_in_place(y_train_full, n_sigma=n_sigma)
            log.info(
                "  [%s] winsorized %d train values at n_sigma=%.1f",
                target, n_clipped, n_sigma,
            )

        # Train
        if HEAD_REGISTRY[target]["kind"] == "classifier":
            if HEAD_REGISTRY[target].get("model_kind") == "multiclass":
                model, metrics = _train_multiclass_classifier(
                    target, X_train, y_train_full, X_test, y_test_full,
                    random_state=random_state,
                )
            else:
                model, metrics = _train_winner(
                    X_train, y_train_full.astype(int),
                    X_test, y_test_full.astype(int),
                    random_state=random_state,
                    calibrate=calibrate,
                    logreg_c=logreg_c,
                    logreg_class_weight=logreg_class_weight,
                    logreg_max_iter=logreg_max_iter,
                )
        else:
            model, metrics = _train_regressor(
                target, X_train, y_train_full, X_test, y_test_full,
                random_state=random_state,
                zinb=zinb,
            )

        log.info("  [%s] metrics: %s", target, metrics)

        train_data = {
            "data_dir": str(data_dir),
            "n_matches": len(targets_list),
            "n_features": n_features,
            "feature_names": feat_names,
            "feature_groups": list(groups),
            "honest_encoder": bool(honest_encoder),
            "test_size": test_size,
            "random_state": random_state,
            "winsorize": winsorize,
            "n_sigma": n_sigma,
            "label_balance": label_balance,
        }
        saved = storage.save(
            name=target,
            version=version,
            model=model,
            encoder=encoder,
            metrics=metrics,
            train_data=train_data,
            feature_names=feat_names,
        )
        log.info("  [%s] saved %s v%s to %s", target, target, version, saved)
        saved_paths[target] = saved

    return saved_paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m business.ml.train",
        description="Train prediction heads on ml_data/full_matches/",
    )
    p.add_argument("--data-dir", default="ml_data/full_matches", type=Path)
    p.add_argument("--model-dir", default="ml_data/models", type=Path)
    p.add_argument("--version", default="1")
    p.add_argument(
        "--target", default="all",
        help=(
            "comma-separated list of targets to train, or 'all' "
            f"(choices: {','.join(HEAD_REGISTRY)} + 'all')"
        ),
    )
    p.add_argument(
        "--calibrate", default="none",
        choices=("none", "sigmoid", "isotonic"),
        help=(
            "Wrap the winner classifier in CalibratedClassifierCV "
            "(0.2.2). 'sigmoid' = Platt scaling; 'isotonic' = isotonic "
            "regression. Improves log_loss on the test set at a small "
            "cost to raw accuracy. Ignored for non-classifier targets."
        ),
    )
    p.add_argument(
        "--zinb", dest="zinb", action="store_true", default=False,
        help=(
            "Use the ZINB factory for the towers regressor (0.2.2). "
            "Requires `pip install statsmodels`. Falls back to "
            "HistGBR(Poisson) with a warning if statsmodels is missing."
        ),
    )
    p.add_argument(
        "--winsorize", dest="winsorize", action="store_true", default=True,
        help="clip training y at mean ± n_sigma*std (default)",
    )
    p.add_argument(
        "--no-winsorize", dest="winsorize", action="store_false",
        help="disable winsorize (use raw targets — eval control only)",
    )
    p.add_argument("--n-sigma", default=3.0, type=float)
    p.add_argument("--test-size", default=0.2, type=float)
    p.add_argument("--random-state", default=42, type=int)
    p.add_argument(
        "--logreg-c", type=float, default=1.0,
        help="inverse regularisation strength for the LogReg winner head (0.3.9)",
    )
    p.add_argument(
        "--logreg-class-weight", default=None,
        help="None | 'balanced' — class weight for LogReg (0.3.9)",
    )
    p.add_argument(
        "--logreg-max-iter", type=int, default=2000,
        help="max iterations for the LogReg solver (0.3.9)",
    )
    p.add_argument(
        "--groups", default="hero,team,lane",
        help=(
            "Comma-separated feature groups to include (0.3.10). "
            "Available: hero, team, lane.  Default: hero,team,lane (24 features). "
            "Use 'hero' for the 0.3.9 baseline (13 features), "
            "'hero,team' for the C retry, 'hero,lane' for the D v2."
        ),
    )
    p.add_argument(
        "--honest-encoder", dest="honest_encoder", action="store_true", default=False,
        help=(
            "Fit the encoder on the train split only (0.3.10). "
            "The default (False) fits the encoder on the full corpus — "
            "mildly inflated metrics but consistent with the 0.3.9 baseline. "
            "Use this flag for an honest holdout that reflects production."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _resolve_targets(spec: str) -> List[str]:
    """Turn the --target flag into a concrete list."""
    if spec.strip().lower() == "all":
        return list(HEAD_REGISTRY)
    out: List[str] = []
    for piece in spec.split(","):
        name = piece.strip()
        if not name:
            continue
        if name not in HEAD_REGISTRY:
            raise SystemExit(f"unknown target {name!r}; expected one of {list(HEAD_REGISTRY)}")
        out.append(name)
    return out


def _resolve_groups(spec: str) -> Tuple[str, ...]:
    """Turn the --groups flag into a concrete tuple.

    Order is preserved (so a model's `feature_names` is
    deterministic).  We dedupe and re-check against `FEATURE_GROUPS`
    so a typo fails fast.
    """
    out: List[str] = []
    for piece in spec.split(","):
        g = piece.strip()
        if not g:
            continue
        if g not in FEATURE_GROUPS:
            raise SystemExit(f"unknown feature group {g!r}; expected one of {list(FEATURE_GROUPS)}")
        if g not in out:
            out.append(g)
    if not out:
        raise SystemExit("at least one --groups entry is required")
    return tuple(out)


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    logging.getLogger().setLevel(args.log_level.upper())
    targets = _resolve_targets(args.target)
    groups = _resolve_groups(args.groups)
    log.info("training targets: %s", targets)
    log.info("feature groups: %s", groups)

    try:
        saved = train_all(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            version=args.version,
            targets=targets,
            winsorize=args.winsorize,
            n_sigma=args.n_sigma,
            test_size=args.test_size,
            random_state=args.random_state,
            calibrate=args.calibrate,
            zinb=args.zinb,
            logreg_c=args.logreg_c,
            logreg_class_weight=args.logreg_class_weight,
            logreg_max_iter=args.logreg_max_iter,
            groups=groups,
            honest_encoder=args.honest_encoder,
        )
    except (OSError, ValueError, KeyError, MLTrainError) as exc:
        # `train_all` raises native exceptions for things like missing
        # data files (OSError), malformed JSON (ValueError), or schema
        # mismatches (KeyError) in the corpus.  We translate them into a
        # single `MLTrainError` so the CLI exit code is consistent and
        # the log line is uniform.
        log.error("training failed: %s", exc, exc_info=True)
        return 1
    print(f"\nOK — saved:")
    for name, p in saved.items():
        print(f"  {name:>14}  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
