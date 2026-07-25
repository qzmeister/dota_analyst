"""
Tests for `business.exceptions` — the project's exception hierarchy.

These tests pin the *contract* of the hierarchy so accidental
refactors (e.g. someone detaching `BoardBuildError` from
`DotaAnalystError`) get caught immediately.  They don't exercise
production code; that's covered by the integration tests.
"""

from __future__ import annotations

import pytest

from business.exceptions import (
    BoardBuildError,
    DLTVError,
    DiscoveryError,
    DotaAnalystError,
    GatewayError,
    HTTPClientError,
    InfraError,
    MLPredictError,
    MLTrainError,
    MLError,
    ParseError,
    ScrapeError,
    SteamAPIError,
    SteamFetchError,
    StreamError,
    UpstreamError,
)


# ========================================================================== #
# Root
# ========================================================================== #

class TestRoot:
    def test_dota_analyst_error_is_exception(self):
        # All our errors must be catchable by `except Exception` —
        # otherwise they bypass stdlib machinery like traceback printing.
        assert issubclass(DotaAnalystError, Exception)

    def test_dota_analyst_error_can_be_instantiated_with_message(self):
        e = DotaAnalystError("boom")
        assert str(e) == "boom"


# ========================================================================== #
# ML hierarchy
# ========================================================================== #

class TestMLHierarchy:
    def test_ml_error_inherits_from_root(self):
        assert issubclass(MLError, DotaAnalystError)

    def test_predict_error_inherits_from_ml_error(self):
        assert issubclass(MLPredictError, MLError)

    def test_train_error_inherits_from_ml_error(self):
        assert issubclass(MLTrainError, MLError)

    def test_predict_and_train_are_siblings(self):
        # Predict and train failures should not be conflated — a caller
        # catching MLPredictError to "fall back to heuristic" must not
        # accidentally swallow an MLTrainError.
        assert MLPredictError is not MLTrainError
        assert not issubclass(MLPredictError, MLTrainError)
        assert not issubclass(MLTrainError, MLPredictError)

    @pytest.mark.parametrize("cls", [MLPredictError, MLTrainError])
    def test_ml_subclasses_catchable_by_root(self, cls):
        # `except DotaAnalystError` should catch all of these.
        try:
            raise cls("test")
        except DotaAnalystError as e:
            assert isinstance(e, cls)


# ========================================================================== #
# Board hierarchy
# ========================================================================== #

class TestBoardHierarchy:
    def test_board_build_error_inherits_from_root(self):
        assert issubclass(BoardBuildError, DotaAnalystError)

    def test_board_build_error_not_ml_error(self):
        # Board assembly is not an ML concern; an ML-flavoured
        # `except MLError` must not catch a BoardBuildError.
        assert not issubclass(BoardBuildError, MLError)


# ========================================================================== #
# Discovery hierarchy
# ========================================================================== #

class TestDiscoveryHierarchy:
    def test_discovery_error_inherits_from_root(self):
        assert issubclass(DiscoveryError, DotaAnalystError)

    @pytest.mark.parametrize("cls", [ScrapeError, SteamFetchError, ParseError])
    def test_discovery_subclasses_inherit(self, cls):
        assert issubclass(cls, DiscoveryError)

    def test_discovery_subclasses_are_distinct(self):
        # Three siblings — different operational meanings.
        assert ScrapeError is not SteamFetchError
        assert ScrapeError is not ParseError
        assert SteamFetchError is not ParseError


# ========================================================================== #
# Upstream hierarchy
# ========================================================================== #

class TestUpstreamHierarchy:
    def test_upstream_error_inherits_from_root(self):
        assert issubclass(UpstreamError, DotaAnalystError)

    @pytest.mark.parametrize("cls", [DLTVError, SteamAPIError])
    def test_upstream_subclasses_inherit(self, cls):
        assert issubclass(cls, UpstreamError)

    def test_upstream_and_discovery_are_separate_trees(self):
        # A "scrape failed because DLTV is down" is conceptually a
        # discovery issue, but the typing puts it under UpstreamError
        # (HTTP call failed).  These trees must NOT overlap — a
        # caller writing `except DiscoveryError` to mean "any
        # discovery-time error" must NOT silently catch a generic
        # DLTVError.
        assert not issubclass(DLTVError, DiscoveryError)
        assert not issubclass(ScrapeError, UpstreamError)


# ========================================================================== #
# Infra hierarchy
# ========================================================================== #

class TestInfraHierarchy:
    def test_infra_error_inherits_from_root(self):
        assert issubclass(InfraError, DotaAnalystError)

    @pytest.mark.parametrize("cls", [HTTPClientError, StreamError, GatewayError])
    def test_infra_subclasses_inherit(self, cls):
        assert issubclass(cls, InfraError)

    def test_infra_subclasses_are_distinct(self):
        assert HTTPClientError is not StreamError
        assert HTTPClientError is not GatewayError
        assert StreamError is not GatewayError


# ========================================================================== #
# Catching semantics
# ========================================================================== #

class TestCatchingSemantics:
    """End-to-end catching: verify the documented catch sites work."""

    def test_ml_predict_caught_by_ml_error(self):
        with pytest.raises(MLError):
            raise MLPredictError("model missing")

    def test_ml_train_caught_by_ml_error(self):
        with pytest.raises(MLError):
            raise MLTrainError("bad data")

    def test_dltv_caught_by_upstream(self):
        with pytest.raises(UpstreamError):
            raise DLTVError("dltv 500")

    def test_scrape_caught_by_discovery(self):
        with pytest.raises(DiscoveryError):
            raise ScrapeError("scrape timeout")

    def test_board_build_caught_by_root(self):
        with pytest.raises(DotaAnalystError):
            raise BoardBuildError("series broken")

    def test_stream_caught_by_infra(self):
        with pytest.raises(InfraError):
            raise StreamError("queue overflow")

    def test_generic_exception_does_not_match_our_hierarchy(self):
        # A plain `ValueError` is NOT one of ours — callers who want
        # only our errors should catch `DotaAnalystError`, not the
        # stdlib root.
        with pytest.raises(ValueError):
            try:
                raise ValueError("stdlib")
            except DotaAnalystError:
                pytest.fail("DotaAnalystError caught a stdlib ValueError")

    def test_keyboard_interrupt_not_caught_by_root(self):
        # KeyboardInterrupt inherits from BaseException, not Exception.
        # Our hierarchy must not shadow that.
        assert not issubclass(DotaAnalystError, KeyboardInterrupt)
        assert not issubclass(BoardBuildError, KeyboardInterrupt)


# ========================================================================== #
# __all__ surface
# ========================================================================== #

class TestExports:
    def test_all_public_names_are_importable(self):
        # Pin the public surface — anything in `__all__` must be
        # importable by name from the module.
        from business import exceptions
        for name in exceptions.__all__:
            assert hasattr(exceptions, name), f"missing: {name}"

    def test_all_exports_are_classes_or_root(self):
        # Everything in `__all__` is a class (root + 14 subclasses).
        # Catches typos like accidentally exporting a constant.
        from business import exceptions
        for name in exceptions.__all__:
            obj = getattr(exceptions, name)
            assert isinstance(obj, type), f"{name} is not a class"
