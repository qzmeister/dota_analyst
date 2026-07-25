"""
Unit tests for `business.ml.engine` — the Strategy pattern that swaps
`HeuristicEngine` and `MLEngine` behind a single `IPredictionEngine`
contract.

The strategy itself is a thin wrapper; the interesting failure modes
are:
  - MLEngine must not crash the request when hero IDs are missing
  - MLEngine must override ONLY the blocks whose sub-model is loaded
  - the module singleton in `get_default_engine()` must reset
    cleanly between tests
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from business.ml.engine import (
    KNOWN_TARGETS,
    HeuristicEngine,
    MLEngine,
    _valid_hero_id_list,
    get_default_engine,
    make_engine,
    reset_default_engine,
)
from business.ml.features import HeroWinRateEncoder
from business.ml.storage import LoadedModel, ModelMetadata, ModelStorage


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #

class _FakeClassifier:
    """A sklearn-shaped classifier that always predicts the same class."""

    def __init__(self, p_radiant: float) -> None:
        self._p = float(p_radiant)

    def predict_proba(self, X):
        n = len(X)
        return np.tile([1.0 - self._p, self._p], (n, 1))


class _FakeRegressor:
    """A sklearn-shaped regressor that always returns a constant."""

    def __init__(self, value: float) -> None:
        self._v = float(value)

    def predict(self, X):
        return np.full(len(X), self._v)


class _ExplodingModel:
    """A model that raises — used to test fallback behaviour."""

    def predict_proba(self, X):
        raise RuntimeError("kaboom")

    def predict(self, X):
        raise RuntimeError("kaboom")


def _make_loaded(name: str, model, version: str = "1") -> LoadedModel:
    """Wrap a fake model in a `LoadedModel` so MLEngine accepts it."""
    meta = ModelMetadata(
        name=name,
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sklearn_version="0.0.0-fake",
        numpy_version="0.0.0-fake",
        python_version="0.0.0-fake",
        feature_names=list(("a", "b", "c")),  # not used by MLEngine
        n_features=3,
        metrics={},
        train_data={},
        encoder={},
    )
    return LoadedModel(
        model=model,
        encoder=HeroWinRateEncoder(),
        metadata=meta,
        path=Path(f"/fake/{name}_v{version}"),
    )


@pytest.fixture
def sample_match_payload(sample_heroes_balanced, sample_team_a, sample_team_b):
    """A bundle of (team_a, team_b, heroes_a, heroes_b) shaped for analyze()."""
    heroes_a, heroes_b = sample_heroes_balanced
    return sample_team_a, sample_team_b, heroes_a, heroes_b


@pytest.fixture(autouse=True)
def _clean_singleton(monkeypatch):
    """Make sure no test inherits a cached engine from a previous one."""
    reset_default_engine()
    # Strip the env var so tests don't leak it.
    monkeypatch.delenv("PREDICTION_ENGINE", raising=False)
    yield
    reset_default_engine()


# --------------------------------------------------------------------------- #
# HeuristicEngine
# --------------------------------------------------------------------------- #

class TestHeuristicEngine:
    def test_name(self):
        assert HeuristicEngine().name == "heuristic"

    def test_analyze_returns_full_dict(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        result = HeuristicEngine().analyze(team_a, team_b, heroes_a, heroes_b)
        for key in ("winner", "kills", "kills_total_over_under",
                    "duration_min", "total_over_under", "towers",
                    "first_to_15", "multikill", "confidence"):
            assert key in result


# --------------------------------------------------------------------------- #
# MLEngine — empty / no sub-models
# --------------------------------------------------------------------------- #

class TestMLEngineConstruction:
    def test_requires_at_least_one_sub_model(self):
        with pytest.raises(ValueError):
            MLEngine(sub_models={})

    def test_name(self):
        eng = MLEngine(sub_models={"winner": _make_loaded("winner", _FakeClassifier(0.7))})
        assert eng.name == "ml"

    def test_known_targets_constant(self):
        # The factory scans this list — keep it honest.
        for name in ("winner", "kills", "duration_mean"):
            assert name in KNOWN_TARGETS


# --------------------------------------------------------------------------- #
# MLEngine — winner sub-model
# --------------------------------------------------------------------------- #

class TestMLEngineWinner:
    def test_overrides_winner_block(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "winner": _make_loaded("winner", _FakeClassifier(0.8), version="1"),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        w = out["winner"]
        assert w["team"] == team_a["name"]
        assert w["prob_radiant"] == 80
        assert w["probability"] == 80
        assert w["source"] == "ml:1"

    def test_keeps_other_blocks_heuristic(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "winner": _make_loaded("winner", _FakeClassifier(0.6)),
        })
        ml = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        heu = HeuristicEngine().analyze(team_a, team_b, heroes_a, heroes_b)
        # winner is overridden
        assert ml["winner"].get("source") == "ml:1"
        # everything else is identical to the heuristic
        for k in ("first_to_15", "multikill", "confidence"):
            assert ml[k] == heu[k]

    def test_keeps_heuristic_winner_when_any_hero_id_missing(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        heroes_b = list(heroes_b)
        heroes_b[0] = {**heroes_b[0]}
        heroes_b[0].pop("steam_id", None)
        heroes_b[0].pop("id", None)
        eng = MLEngine(sub_models={
            "winner": _make_loaded("winner", _FakeClassifier(0.9)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert "source" not in (out.get("winner") or {})

    def test_keeps_heuristic_winner_when_count_wrong(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        heroes_b = heroes_b[:4]
        eng = MLEngine(sub_models={
            "winner": _make_loaded("winner", _FakeClassifier(0.9)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert "source" not in (out.get("winner") or {})

    def test_swallows_model_exception(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "winner": _make_loaded("winner", _ExplodingModel()),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert "source" not in (out.get("winner") or {})


# --------------------------------------------------------------------------- #
# MLEngine — kills + duration sub-models
# --------------------------------------------------------------------------- #

class TestMLEngineNumericBlocks:
    def test_kills_override_total(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "kills": _make_loaded("kills", _FakeRegressor(42.0)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert out["kills"]["total"] == 42
        # over/under should follow the heuristic's rules applied to 42
        assert out["kills_total_over_under"]["side"] == "over"
        assert out["kills_total_over_under"]["threshold"] == 43

    def test_kills_high_means_under(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "kills": _make_loaded("kills", _FakeRegressor(55.0)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert out["kills_total_over_under"]["side"] == "under"

    def test_duration_mean_override(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "duration_mean": _make_loaded("duration_mean", _FakeRegressor(45.0)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert out["duration_min"] == 45.0
        assert out["total_over_under"]["side"] == "under"
        assert out["total_over_under"]["threshold"] == 45
        # 45 minutes in MM:SS format = 0 hours 45 minutes = "0:45".
        assert out["total_over_under"]["formatted"] == "0:45"

    def test_duration_short_means_over(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "duration_mean": _make_loaded("duration_mean", _FakeRegressor(32.0)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert out["total_over_under"]["side"] == "over"
        assert out["total_over_under"]["threshold"] == 33

    def test_all_blocks_overridden_together(self, sample_match_payload):
        team_a, team_b, heroes_a, heroes_b = sample_match_payload
        eng = MLEngine(sub_models={
            "winner": _make_loaded("winner", _FakeClassifier(0.6)),
            "kills": _make_loaded("kills", _FakeRegressor(48.0)),
            "duration_mean": _make_loaded("duration_mean", _FakeRegressor(40.0)),
        })
        out = eng.analyze(team_a, team_b, heroes_a, heroes_b)
        assert out["winner"].get("source") == "ml:1"
        assert out["kills"]["total"] == 48
        assert out["duration_min"] == 40.0
        # first_to_15 / multikill stay heuristic
        assert "source" not in out["first_to_15"]


# --------------------------------------------------------------------------- #
# _valid_hero_id_list
# --------------------------------------------------------------------------- #

class TestValidHeroIdList:
    def test_returns_none_when_any_id_missing(self, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        heroes_b = list(heroes_b)
        heroes_b[0] = {**heroes_b[0]}
        heroes_b[0].pop("steam_id", None)
        heroes_b[0].pop("id", None)
        assert _valid_hero_id_list(heroes_a, heroes_b) is None

    def test_returns_none_when_count_wrong(self, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        assert _valid_hero_id_list(heroes_a, heroes_b[:4]) is None
        assert _valid_hero_id_list(heroes_a[:4], heroes_b) is None

    def test_returns_ids_when_all_present(self, sample_heroes_balanced):
        heroes_a, heroes_b = sample_heroes_balanced
        r, d = _valid_hero_id_list(heroes_a, heroes_b)
        # Conftest heroes all have steam_id=1; we get [1, 1, 1, 1, 1].
        assert r == [1, 1, 1, 1, 1]
        assert d == [1, 1, 1, 1, 1]


# --------------------------------------------------------------------------- #
# make_engine + get_default_engine
# --------------------------------------------------------------------------- #

class TestMakeEngine:
    def test_heuristic_default(self):
        eng = make_engine("heuristic")
        assert isinstance(eng, HeuristicEngine)
        assert eng.name == "heuristic"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            make_engine("quantum-tunneling")

    def test_ml_falls_back_to_heuristic_when_no_model(self, tmp_path):
        eng = make_engine("ml", model_dir=tmp_path)
        assert isinstance(eng, HeuristicEngine)

    def test_ml_uses_loaded_sub_models(self, tmp_path, monkeypatch):
        # Build and save a tiny winner sub-model.
        enc = HeroWinRateEncoder()
        ModelStorage(tmp_path).save(
            name="winner",
            version="1",
            model=_FakeClassifier(0.65),
            encoder=enc,
            metrics={"accuracy": 0.55, "n_train": 100, "n_test": 25},
            train_data={"n_matches": 125, "data_dir": "fake"},
        )
        eng = make_engine("ml", model_dir=tmp_path)
        assert isinstance(eng, MLEngine)
        assert eng.name == "ml"
        assert "winner" in eng._sub_models

    def test_ml_partial_sub_models_still_runs(self, tmp_path, monkeypatch):
        # Only winner trained, no kills/duration — engine should still
        # come up and just leave those blocks heuristic.
        enc = HeroWinRateEncoder()
        ModelStorage(tmp_path).save(
            name="winner", version="1", model=_FakeClassifier(0.5),
            encoder=enc, metrics={"n_train": 1, "n_test": 1},
            train_data={"n_matches": 2},
        )
        eng = make_engine("ml", model_dir=tmp_path)
        assert isinstance(eng, MLEngine)
        assert "winner" in eng._sub_models
        assert "kills" not in eng._sub_models

    def test_ml_uses_env_model_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        enc = HeroWinRateEncoder()
        ModelStorage(tmp_path).save(
            name="winner", version="1", model=_FakeClassifier(0.5),
            encoder=enc, metrics={"n_train": 1, "n_test": 1},
            train_data={"n_matches": 2},
        )
        eng = make_engine("ml")
        assert isinstance(eng, MLEngine)


class TestDefaultEngineSingleton:
    def test_first_call_builds_engine(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_ENGINE", "heuristic")
        eng1 = get_default_engine()
        assert isinstance(eng1, HeuristicEngine)

    def test_second_call_returns_same_instance(self):
        eng1 = get_default_engine()
        eng2 = get_default_engine()
        assert eng1 is eng2

    def test_reset_drops_cache(self):
        eng1 = get_default_engine()
        reset_default_engine()
        eng2 = get_default_engine()
        assert eng1 is not eng2
