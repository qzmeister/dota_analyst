"""
Unit tests for `business.ml.outliers` — the winsorize utility that
clips long-tail training targets to `median ± n_sigma * 1.4826 * MAD`.

Outlier handling is the silent foundation of every regressor in
the package; these tests pin the contract so a future refactor
can't silently change the clipping behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from business.ml.outliers import (
    MAD_TO_STD,
    _mad_bounds,
    count_clipped,
    winsorize_in_place,
    winsorize_values,
)


# --------------------------------------------------------------------------- #
# Realistic sample — 100 values with a long tail, what winsorize is for
# --------------------------------------------------------------------------- #

def _realistic_sample() -> np.ndarray:
    """100 floats: 95 in [20, 50], 5 in [200, 1000] (long tail)."""
    bulk = np.linspace(20, 50, 95)
    tail = np.array([200, 300, 500, 750, 1000])
    return np.concatenate([bulk, tail])  # not shuffled — kept ordered


# --------------------------------------------------------------------------- #
# _mad_bounds
# --------------------------------------------------------------------------- #

class TestMadBounds:
    def test_empty_array(self):
        assert _mad_bounds(np.array([], dtype=float), 3.0) == (0.0, 0.0)

    def test_constant_array_degenerate_range(self):
        # MAD = 0 → degenerate range around the value.
        assert _mad_bounds(np.array([5.0, 5.0, 5.0]), 3.0) == (5.0, 5.0)

    def test_robust_to_long_tail(self):
        # The 1000 in the tail must NOT inflate the bounds — that's
        # the whole point of using MAD instead of std.
        a = _realistic_sample()
        lo, hi = _mad_bounds(a, 3.0)
        # Median is ~50 (the 50/51-th element of the bulk), MAD is
        # small (most of the data is in [20, 50]).  Bounds must be
        # tight enough to clip the 200..1000 tail.
        assert hi < 200.0, f"expected upper bound < 200 (tail clipped), got {hi}"
        assert lo > 0.0

    def test_classic_3sigma_clips_at_3_mad(self):
        # The textbook example: [1, 2, 3, 4, 100].  Median = 3,
        # MAD = |1-3|, |2-3|, |3-3|, |4-3|, |100-3| sorted → 1, 1, 0, 1, 97 → median = 1.
        # std_robust = 1.4826 * 1 = 1.4826.
        # 3 * std_robust = ~4.45.  Bounds = [3-4.45, 3+4.45] = [-1.45, 7.45].
        a = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        lo, hi = _mad_bounds(a, 3.0)
        assert lo < 0.0  # 1 is above the lower bound, no clip
        assert hi < 10.0  # 100 is well above 7.45 → clipped
        assert 100.0 > hi

    def test_tighter_sigma_means_tighter_bounds(self):
        a = _realistic_sample()
        lo1, hi1 = _mad_bounds(a, 1.0)
        lo3, hi3 = _mad_bounds(a, 3.0)
        assert (hi1 - lo1) < (hi3 - lo3)


# --------------------------------------------------------------------------- #
# winsorize_values
# --------------------------------------------------------------------------- #

class TestWinsorizeValues:
    def test_realistic_long_tail_is_clipped(self):
        # The user story: 100-row training set with a few extreme
        # values that need to come down.  After winsorize the max
        # should be close to the upper bound, not 1000.
        a = _realistic_sample()
        out = winsorize_values(a, n_sigma=3.0)
        assert float(np.max(out)) < 200.0

    def test_5_element_extreme_clipped(self):
        # The textbook heavy-tail example — 100 must be clipped.
        a = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        out = winsorize_values(a, n_sigma=3.0)
        # 100 must be brought down to the upper bound (~7.45).
        assert float(out[-1]) < 10.0
        # 1..4 stay put.
        assert float(out[0]) == 1.0
        assert float(out[1]) == 2.0

    def test_no_clip_for_tight_distribution(self):
        a = np.linspace(10, 20, 100)
        out = winsorize_values(a, n_sigma=3.0)
        np.testing.assert_array_equal(out, a)

    def test_replaces_nan_with_median(self):
        a = [1.0, 2.0, 3.0, 4.0, math.nan]
        out = winsorize_values(a, n_sigma=3.0)
        assert not np.isnan(out).any()
        # Median of [1, 2, 3, 4] = 2.5.
        assert out[-1] == pytest.approx(2.5)

    def test_replaces_inf(self):
        a = [1.0, 2.0, 3.0, math.inf]
        out = winsorize_values(a, n_sigma=3.0)
        assert np.isfinite(out).all()

    def test_empty_input_returns_empty(self):
        out = winsorize_values([], n_sigma=3.0)
        assert out.size == 0

    def test_does_not_mutate_input(self):
        a = _realistic_sample()
        original = a.copy()
        _ = winsorize_values(a, n_sigma=3.0)
        np.testing.assert_array_equal(a, original)

    def test_iterable_input_accepted(self):
        out = winsorize_values((1.0, 2.0, 3.0), n_sigma=3.0)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_mad_to_std_constant(self):
        # 1.4826 is the Gaussian-consistency constant.  Pin it so
        # a "let's use 1.5 instead" refactor doesn't slip through.
        assert MAD_TO_STD == pytest.approx(1.4826, abs=1e-4)


# --------------------------------------------------------------------------- #
# winsorize_in_place
# --------------------------------------------------------------------------- #

class TestWinsorizeInPlace:
    def test_mutates_caller_array(self):
        a = _realistic_sample()
        out = winsorize_in_place(a, n_sigma=3.0)
        assert out is a
        # Tail values should be pulled in.
        assert float(np.max(a)) < 200.0

    def test_empty_array(self):
        a = np.array([], dtype=float)
        out = winsorize_in_place(a, n_sigma=3.0)
        assert out is a
        assert out.size == 0


# --------------------------------------------------------------------------- #
# count_clipped
# --------------------------------------------------------------------------- #

class TestCountClipped:
    def test_zero_when_no_outliers(self):
        a = np.linspace(10, 20, 100)
        assert count_clipped(a, n_sigma=3.0) == 0

    def test_counts_long_tail_entries(self):
        a = _realistic_sample()
        # The 5 long-tail entries (200..1000) should be clipped.
        clipped = count_clipped(a, n_sigma=3.0)
        assert clipped >= 1

    def test_empty_array(self):
        assert count_clipped(np.array([], dtype=float), n_sigma=3.0) == 0

    def test_tighter_sigma_clips_more(self):
        a = _realistic_sample()
        n_1sigma = count_clipped(a, n_sigma=1.0)
        n_3sigma = count_clipped(a, n_sigma=3.0)
        # Tighter sigma → more outliers.
        assert n_1sigma >= n_3sigma
