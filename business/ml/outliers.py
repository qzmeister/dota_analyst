"""
Outlier handling for the ML training pipeline.

Why winsorize?
--------------
Some of our regression targets (kills, towers, duration) have long
tails — a handful of stomps (~80 kills) or super-long games (~70 min)
sit well outside the bulk.  A Poisson / Gamma / XGBoost regressor
trained on the raw distribution spends a non-trivial fraction of its
capacity modelling the tail at the expense of the median, where most
of the prediction error actually lives.

We do NOT drop tail rows.  We **clip** them to `median ± n_sigma * MAD`
so they still contribute (just not as outliers).  The resulting
estimator is slightly conservative on extreme games and noticeably
better on the median — exactly what the betting market cares about.

Why MAD instead of std?
-----------------------
Naïve `mean ± 3*std` has a known failure mode on heavy-tailed data:
the outliers inflate `std`, which inflates the clip bounds, which
fails to clip the very outliers we wanted to clip.  With [1, 2, 3,
4, 100] the standard deviation is ~39, so `mean ± 3*std = [-95, 139]`
— 100 falls well inside and is NOT clipped.

MAD (Median Absolute Deviation) is the textbook robust scale
estimator: it is unaffected by up to 50% of the data being outliers.
We use the Gaussian-consistency constant `1.4826` so that on a
genuinely normal distribution, `1.4826 * MAD ≈ std`.

Binary targets (winner) are NEVER winsorized — a 0/1 label has no
distance metric to clip.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


# Gaussian-consistency constant: on a normal distribution,
# std == 1.4826 * MAD.  Using this lets us talk in "sigma" units
# while staying robust to outliers.
MAD_TO_STD = 1.4826


def _mad_bounds(values: np.ndarray, n_sigma: float) -> Tuple[float, float]:
    """Compute `median ± n_sigma * 1.4826 * MAD` as a robust 3σ bound.

    Falls back to a degenerate `(median, median)` range when MAD is 0
    (a constant array); in that case every value is its own median,
    and clipping is a no-op anyway.
    """
    if values.size == 0:
        return (0.0, 0.0)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad == 0.0:
        return (med, med)
    std_robust = MAD_TO_STD * mad
    return (med - n_sigma * std_robust, med + n_sigma * std_robust)


def winsorize_values(values: Iterable[float], n_sigma: float = 3.0) -> np.ndarray:
    """Clip `values` to `[median - n_sigma*1.4826*MAD, median + ...]`.

    The output is a fresh numpy array; the input is left untouched.
    Non-finite values (NaN, +inf, -inf) are replaced with the
    median of the finite portion — they would otherwise survive
    winsorization and poison the regressor.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr

    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        finite_median = float(np.median(arr[finite_mask])) if finite_mask.any() else 0.0
        arr = np.where(finite_mask, arr, finite_median)

    lo, hi = _mad_bounds(arr, n_sigma=n_sigma)
    return np.clip(arr, lo, hi)


def winsorize_in_place(arr: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """In-place version for callers that already own the array.

    Returns the same array (so it can be used fluently in a pipeline
    without `arr = winsorize_in_place(arr, ...)`).
    """
    if arr.size == 0:
        return arr

    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        finite_median = float(np.median(arr[finite_mask])) if finite_mask.any() else 0.0
        arr[~finite_mask] = finite_median

    lo, hi = _mad_bounds(arr, n_sigma=n_sigma)
    np.clip(arr, lo, hi, out=arr)
    return arr


def count_clipped(values: np.ndarray, n_sigma: float = 3.0) -> int:
    """How many entries would be clipped at this sigma level?

    Used by the trainer for logging — we want to see, e.g.
    "winsorized 12/1111 kills values (1.1%)" without having to
    re-compute the bounds in two places.
    """
    if values.size == 0:
        return 0
    lo, hi = _mad_bounds(values, n_sigma=n_sigma)
    return int(np.sum((values < lo) | (values > hi)))
