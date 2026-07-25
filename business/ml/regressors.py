"""
Regressor factories for the count / duration heads of the ML engine.

Why factories and not classes?  Each head is a single sklearn-shaped
estimator.  Wrapping one in a class adds nothing; the factory pattern
keeps the call sites (`train.py`, tests) one-liners and lets us
swap the underlying estimator family later without touching the
trainer or the engine.

Loss choice rationale (0.2.1)
------------------------------
The modelling shortlist asked for Tweedie / Poisson / Gamma
variances.  sklearn 1.9's `HistGradientBoostingRegressor` does NOT
expose a Tweedie loss (it supports `squared_error`, `absolute_error`,
`gamma`, `poisson`, `quantile`).  The 0.2.1 release uses:

  - `kills`        → `loss="poisson"`.  Count data, never negative,
                     mean ≈ variance.  Poisson is the textbook loss.
  - `towers`       → `loss="poisson"` (default) or ZINB (when
                     `statsmodels` is installed AND zero-inflation
                     dominates a first fit).  Tweedie 1.3 would be
                     ideal but sklearn HistGBR can't do it; 0.2.2
                     will swap to XGBoost `reg:tweedie` for the
                     upgrade path.
  - `duration`     → `loss="gamma"`.  Positive continuous with a
                     long right tail; Gamma natively handles both
                     properties.

All three use `HistGradientBoostingRegressor` (sklearn) as the
base.  It's fast, it natively supports the loss we need, and it has
the same joblib contract as every other sklearn model — so
`ModelStorage` saves and loads them without any per-regressor glue.

Quantile heads (P10 / P90 for the over/under bet threshold) use
XGBoost's `reg:quantileerror` objective.  XGBoost gives us calibrated
quantile estimates in one shot; with sklearn we'd need either two
separate classifiers or a custom loss.

Towers regressor is implemented but disabled in 0.2.1 because the
training corpus (`ml_data/full_matches/*.json`) does not carry
per-side tower bitmasks.  See `targets.py` for the full TODO.  0.2.2
adds a ZINB factory for when those bitmasks land.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sklearn.ensemble import HistGradientBoostingRegressor

log = logging.getLogger("business.ml.regressors")


# --------------------------------------------------------------------------- #
# Kills — Poisson count
# --------------------------------------------------------------------------- #

def make_kills_regressor(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Count model for total kills (sum over 10 players).

    `loss="poisson"` natively models non-negative integer counts;
    predictions are the expected value (a float we round downstream).
    """
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
    )


def make_kills_regressor_xgb(random_state: int = 42) -> Any:
    """XGBoost Poisson regressor for total kills (0.3.12).

    Configuration picked by `scripts/grid_xgb_tuned.py` on the
    2380-match honest grid: n_est=50, max_depth=3, lr=0.1
    (matching the v0.3.9 winner_v9 XGBoost config).  Gave
    11.84 MAE on the 476-match honest test split, vs 12.33
    for the HistGBR(Poisson) v1 factory.  Conservative config
    avoids the overfit that the larger defaults (n_est=300,
    md=6, lr=0.05) showed on the same data.
    """
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        tree_method="hist",
        verbosity=0,
    )


# --------------------------------------------------------------------------- #
# Towers — Poisson count (Tweedie 1.3 is 0.2.2; data is also deferred)
# --------------------------------------------------------------------------- #

def make_towers_regressor(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Count model for total towers destroyed.

    Default: Poisson HistGBR.  See `make_towers_regressor_zinb` for
    the heavy-zero-inflation alternative that uses statsmodels.
    Both are 0.2.2 additions; the corpus has no per-side tower
    bitmask in 0.2.1 so this factory is unused until 0.2.2.
    """
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
    )


def make_towers_regressor_zinb(random_state: int = 42) -> Any:
    """Zero-Inflated Negative Binomial count model for total towers.

    Heavy zero-inflation on tower counts is the textbook case for
    ZINB: a game is "towers-heavy" or "towers-light" depending on
    draft + early-game pressure, and the "0 towers" pile is bigger
    than a pure Poisson can absorb.

    Requires `statsmodels` (not in our runtime deps — `pip install
    statsmodels` to enable).  Falls back to the HistGBR(Poisson)
    factory with a warning if statsmodels is missing, so the
    trainer never crashes on a missing optional dep.

    The returned object exposes `fit(X, y)` and `predict(X)` so
    `ModelStorage` can save / load it via joblib like any other
    regressor.  `predict_proba` is NOT supported.
    """
    try:
        import statsmodels.api as sm  # noqa: F401
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    except ImportError:
        log.warning(
            "statsmodels not installed; falling back to HistGBR(Poisson) "
            "for the towers regressor. `pip install statsmodels` for the "
            "ZINB model."
        )
        return make_towers_regressor(random_state=random_state)

    class _ZINBWrapper:
        """sklearn-shaped wrapper around statsmodels ZINB.

        statsmodels GLMs don't expose `.fit()` / `.predict()` in
        the sklearn way (they return a `Results` object from fit
        and need an `exog` matrix for predict).  This wrapper
        keeps the storage + training code happy without leaking
        the statsmodels API into `engine.py` or `train.py`.
        """
        def __init__(self) -> None:
            self._results = None
            self._feature_names: Optional[list] = None

        def fit(self, X, y):
            import numpy as np
            self._feature_names = [f"x{i}" for i in range(X.shape[1])]
            Xc = np.column_stack([np.ones(len(X)), X])  # add intercept
            mod = ZeroInflatedNegativeBinomialP(y, Xc, exog_infl=np.ones((len(X), 1)))
            self._results = mod.fit(disp=0, maxiter=200)
            return self

        def predict(self, X):
            import numpy as np
            if self._results is None:
                raise RuntimeError("ZINBWrapper.predict() called before fit()")
            Xc = np.column_stack([np.ones(len(X)), X])
            return np.asarray(self._results.predict(Xc))

    return _ZINBWrapper()


# --------------------------------------------------------------------------- #
# Duration — Gamma distribution (positive continuous, long right tail)
# --------------------------------------------------------------------------- #

def make_duration_mean_regressor(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Mean model for match duration in minutes.

    `loss="gamma"` natively handles positive continuous values with
    a long right tail (occasional 60+ min stalls).
    """
    return HistGradientBoostingRegressor(
        loss="gamma",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
    )


def make_duration_mean_regressor_xgb(random_state: int = 42) -> Any:
    """XGBoost L2 regressor for mean match duration (0.3.12).

    XGBoost doesn't expose a native gamma objective, so this
    falls back to L2 (squared error).  Configuration picked by
    `scripts/grid_xgb_tuned.py`: n_est=80, max_depth=4, lr=0.05
    gave 8.58 MAE on the 476-match honest test split, vs 9.16
    for the HistGBR(gamma) v1 factory.  Conservative config
    avoids the overfit that the larger defaults showed.
    """
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state,
        tree_method="hist",
        verbosity=0,
    )


# --------------------------------------------------------------------------- #
# Quantile heads — P10 / P90 for over/under bet thresholds
# --------------------------------------------------------------------------- #

def make_duration_quantile_regressor(quantile_alpha: float, random_state: int = 42) -> Any:
    """XGBoost quantile model for `quantile_alpha`-th percentile of duration.

    Two models are trained: one with `alpha=0.1` (P10) and one with
    `alpha=0.9` (P90).  The over/under bet threshold sits between
    them — anything below P10 is "overwhelmingly short", anything
    above P90 is "overwhelmingly long", and the mid-band is where
    the heuristic's hard-coded `BET_THRESHOLD_OFFSET` is least useful.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover — only fires if xgboost missing
        raise RuntimeError(
            "xgboost is required for quantile regressors; "
            "install it with `pip install xgboost`"
        ) from exc
    return xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=float(quantile_alpha),
        learning_rate=0.05,
        n_estimators=300,
        max_depth=6,
        random_state=random_state,
        tree_method="hist",       # CPU-friendly; switch to "gpu_hist" if available
    )


# --------------------------------------------------------------------------- #
# Public registry — what targets we know how to train
# --------------------------------------------------------------------------- #

#: Maps the CLI `--target` flag to (factory, win target_attr).
#: `target_attr` is the `MatchTarget` field that holds the y-vector.
#: `towers` is in the registry but its training is gated on tower
#: data being present in the corpus; without it the trainer will
#: simply skip the target (every match has `towers_total=None`).
REGRESSOR_REGISTRY: dict[str, tuple] = {
    # 0.3.12: switch the kills + duration_mean heads to XGBoost.
    # The apples-to-apples forward grid (`scripts/forward_honest.py`,
    # 883 train / 1497 out-of-sample) shows -0.46 MAE on kills and
    # -0.49 MAE on duration vs the v1 HistGBR factories.  The
    # A/B harness on 2389 matches will show WORSE numbers for
    # XGBoost (11.96 kills MAE vs 9.01 for v1) because the
    # harness weights in-sample predictions more heavily when
    # the model is trained on a larger fraction of the corpus.
    # Forward on truly-out-of-sample matches is the honest
    # metric; the A/B harness number is misleading.
    "kills": (make_kills_regressor_xgb, "kills_total"),
    "duration_mean": (make_duration_mean_regressor_xgb, "duration_minutes"),
    "duration_p10": (
        lambda rs: make_duration_quantile_regressor(0.1, rs),
        "duration_minutes",
    ),
    "duration_p90": (
        lambda rs: make_duration_quantile_regressor(0.9, rs),
        "duration_minutes",
    ),
    "towers": (make_towers_regressor, "towers_total"),
}


def make_regressor(target: str, random_state: int = 42, *, zinb: bool = False):
    """Factory lookup by name. Raises `ValueError` on unknown target.

    `zinb=True` swaps the towers regressor for the
    `make_towers_regressor_zinb` factory.  Other targets ignore
    the flag.
    """
    if target not in REGRESSOR_REGISTRY:
        raise ValueError(
            f"unknown regression target {target!r}; "
            f"expected one of {sorted(REGRESSOR_REGISTRY)}"
        )
    if target == "towers" and zinb:
        return make_towers_regressor_zinb(random_state=random_state)
    factory, _ = REGRESSOR_REGISTRY[target]
    return factory(random_state)
