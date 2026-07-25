"""
Unit tests for `business.ml.regressors` — the factory functions that
build sklearn / xgboost estimators for each numeric head.

These tests do not check whether the estimators are *good* at their
job (that's the eval harness's job); they check the *shape* of what
comes out of the factory — does it have `.predict()`, does it
return the right shape, does it respect `random_state`.
"""

from __future__ import annotations

import numpy as np
import pytest

from business.ml.regressors import (
    REGRESSOR_REGISTRY,
    make_duration_mean_regressor,
    make_duration_quantile_regressor,
    make_kills_regressor,
    make_regressor,
    make_towers_regressor,
    make_towers_regressor_zinb,
)


# --------------------------------------------------------------------------- #
# Smoke — every factory returns something that looks like an estimator
# --------------------------------------------------------------------------- #

class TestFactories:
    def test_kills_factory(self):
        m = make_kills_regressor()
        # sklearn HistGradientBoostingRegressor has `loss` and `predict`.
        assert hasattr(m, "loss")
        assert m.loss == "poisson"
        assert hasattr(m, "predict")

    def test_towers_factory(self):
        m = make_towers_regressor()
        # 0.2.1 uses Poisson while Tweedie support lands in 0.2.2.
        assert m.loss == "poisson"
        assert hasattr(m, "predict")

    def test_towers_zinb_factory_falls_back_when_statsmodels_missing(self):
        # When statsmodels is not installed, the ZINB factory
        # should silently fall back to the HistGBR(Poisson)
        # factory.  We can't easily simulate "statsmodels missing"
        # at runtime; the import is already done at module load.
        # We just verify the factory returns something with .predict.
        m = make_towers_regressor_zinb()
        assert hasattr(m, "predict")
        assert hasattr(m, "fit")

    def test_duration_mean_factory(self):
        m = make_duration_mean_regressor()
        # Gamma handles the long right tail of match durations.
        assert m.loss == "gamma"
        assert hasattr(m, "predict")

    def test_duration_quantile_p10(self):
        m = make_duration_quantile_regressor(0.1)
        # XGBoost 3.x stores the constructor params via get_params(),
        # not as instance attributes.  We use the same channel the
        # rest of sklearn uses.
        params = m.get_params()
        assert params["objective"] == "reg:quantileerror"
        assert params["quantile_alpha"] == pytest.approx(0.1)

    def test_duration_quantile_p90(self):
        m = make_duration_quantile_regressor(0.9)
        params = m.get_params()
        assert params["quantile_alpha"] == pytest.approx(0.9)

    def test_make_regressor_lookup_known(self):
        for name in REGRESSOR_REGISTRY:
            m = make_regressor(name)
            assert m is not None

    def test_make_regressor_unknown_raises(self):
        with pytest.raises(ValueError):
            make_regressor("killzzz")


# --------------------------------------------------------------------------- #
# Behavioural — the estimators actually fit and predict
# --------------------------------------------------------------------------- #

class TestFitPredict:
    @pytest.fixture
    def toy_regression_data(self):
        """A tiny synthetic dataset: y = 2 * x1 + noise."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = 2.0 * X[:, 0] + 0.1 * rng.normal(size=200)
        return X, y

    def test_kills_fits_and_predicts(self, toy_regression_data):
        X, y = toy_regression_data
        # We feed in non-negative counts; the model handles Poisson.
        y_count = np.maximum(0, y + 5).astype(int)
        m = make_kills_regressor()
        m.fit(X, y_count)
        preds = m.predict(X[:3])
        assert preds.shape == (3,)
        # Predictions on the training set should be in the same ballpark
        # as the labels.
        assert float(np.mean(preds)) > 0

    def test_towers_fits_and_predicts(self, toy_regression_data):
        X, y = toy_regression_data
        y_count = np.maximum(0, y + 5).astype(int)
        m = make_towers_regressor()
        m.fit(X, y_count)
        preds = m.predict(X[:3])
        assert preds.shape == (3,)
        assert float(np.mean(preds)) > 0

    def test_duration_mean_fits_and_predicts(self, toy_regression_data):
        X, y = toy_regression_data
        # Duration is positive — add a constant offset.
        y_pos = y + 30
        m = make_duration_mean_regressor()
        m.fit(X, y_pos)
        preds = m.predict(X[:3])
        assert preds.shape == (3,)
        # Predictions should be in a reasonable range, not negative.
        assert float(np.min(preds)) > 0

    def test_duration_p10_fits_and_predicts(self, toy_regression_data):
        X, y = toy_regression_data
        y_pos = y + 30
        m = make_duration_quantile_regressor(0.1)
        m.fit(X, y_pos)
        preds = m.predict(X[:3])
        assert preds.shape == (3,)
        assert float(np.min(preds)) > 0

    def test_duration_p90_fits_and_predicts(self, toy_regression_data):
        X, y = toy_regression_data
        y_pos = y + 30
        m = make_duration_quantile_regressor(0.9)
        m.fit(X, y_pos)
        preds = m.predict(X[:3])
        assert preds.shape == (3,)
        assert float(np.min(preds)) > 0


# --------------------------------------------------------------------------- #
# Registry shape
# --------------------------------------------------------------------------- #

class TestRegistry:
    def test_registry_keys_match_documented_targets(self):
        # 0.2.2 added `towers` to the registry.  Training is
        # gated on the corpus having tower data; the registry
        # entry exists regardless.
        expected = {"kills", "duration_mean", "duration_p10", "duration_p90", "towers"}
        assert set(REGRESSOR_REGISTRY.keys()) == expected

    def test_registry_values_are_callable(self):
        for name, (factory, _attr) in REGRESSOR_REGISTRY.items():
            assert callable(factory), f"{name} factory is not callable"
