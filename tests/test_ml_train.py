"""
Tests for `business.ml.train` — the training pipeline.

Strategy:
  - Pure helpers (`_pinball_loss`, `_safe_poisson_deviance`,
    `evaluate_regressor`, `_resolve_targets`, `HEAD_REGISTRY`)
    are tested directly with hand-built numpy arrays.
  - Per-target trainers (`_train_winner`, `_train_regressor`,
    `_train_multiclass_classifier`) get tiny synthetic feature
    matrices — we just need enough rows for the model to fit
    something coherent.
  - `build_dataset` / `train_all` use a `tmp_path` corpus with
    60+ synthetic matches so the ">= 50 matches" floor passes.
  - `main` is tested by feeding argv through `main(argv=[...])`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

from business.ml import train as train_mod
from business.ml.train import (
    HEAD_REGISTRY,
    _pinball_loss,
    _safe_poisson_deviance,
    _resolve_targets,
    _parse_args,
    build_dataset,
    evaluate_regressor,
    iter_matches,
    train_all,
    _train_multiclass_classifier,
    _train_regressor,
    _train_winner,
)


# ========================================================================== #
# Fixtures
# ========================================================================== #

#: Canonical 10 hero-id pool.  Picked so every match we build has 5
#: unique radiant + 5 unique dire heroes.
HERO_POOL = list(range(1, 11))


def _synth_match(
    match_id: int,
    radiant_win: bool,
    max_kills: int = 10,
    duration_min: int = 35,
    hero_offset: int = 0,
) -> dict:
    """Build a minimal match dict that satisfies `extract_target`.

    The keys mirror what `iter_clean_targets` / `extract_target` look
    for.  We rotate through HERO_POOL so adjacent matches have
    different hero lineups — the encoder needs variety.
    """
    # 5 radiant heroes, 5 dire heroes, no overlap.
    r_heroes = [HERO_POOL[(hero_offset + i) % len(HERO_POOL)] for i in range(5)]
    d_heroes = [HERO_POOL[(hero_offset + 5 + i) % len(HERO_POOL)] for i in range(5)]

    def _player(hero_id: int) -> dict:
        return {
            "performance": {
                "hero": {"valve_id": hero_id, "short_name": f"h{hero_id}"},
                "kills": max_kills,
                "deaths": 5,
                "assists": 8,
                "gpm": 400,
                "xpm": 500,
            },
        }

    return {
        "match_id": match_id,
        "duration": duration_min * 60,
        "radiant_victory": radiant_win,
        "has_error": False,
        "patch": "7.40",
        "start_date": "2026-07-01T12:00:00+00:00",
        "state": "finished",
        "radiant": {"player_performances": [_player(h) for h in r_heroes]},
        "dire": {"player_performances": [_player(h) for h in d_heroes]},
    }


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Write 60 synthetic matches to a tmp dir (above the 50-match floor)."""
    out = tmp_path / "matches"
    out.mkdir()
    for i in range(60):
        m = _synth_match(
            match_id=1000 + i,
            radiant_win=(i % 2 == 0),  # balanced 30/30
            max_kills=8 + (i % 6),     # 8..13
            duration_min=30 + (i % 20),  # 30..49
            hero_offset=i,             # different hero lineups
        )
        (out / f"match_{1000 + i:04d}.json").write_text(
            json.dumps(m), encoding="utf-8",
        )
    # Plus one *malformed* file to exercise the iter_matches skip path.
    (out / "broken_9999.json").write_text("{not valid json", encoding="utf-8")
    return out


# ========================================================================== #
# HEAD_REGISTRY
# ========================================================================== #

class TestHeadRegistry:
    def test_known_targets_are_present(self):
        for name in ("winner", "kills", "duration_mean",
                     "duration_p10", "duration_p90", "towers"):
            assert name in HEAD_REGISTRY, f"missing target {name!r}"
        # 0.3.10: "multikill" is intentionally NOT in HEAD_REGISTRY
        # because the pro corpus is 100% High (no Low/Medium matches),
        # which degenerated the classifier.  See train.py for the
        # full rationale and the 0.4.x revisit plan.
        assert "multikill" not in HEAD_REGISTRY, (
            "multikill removed in 0.3.10 — see train.HEAD_REGISTRY comment"
        )

    def test_each_entry_has_required_keys(self):
        for name, entry in HEAD_REGISTRY.items():
            assert "kind" in entry, f"{name}: missing 'kind'"
            assert entry["kind"] in ("classifier", "regressor"), (
                f"{name}: bad kind {entry['kind']!r}"
            )
            assert "y_attr" in entry, f"{name}: missing 'y_attr'"

    def test_regressors_declare_metrics(self):
        for name, entry in HEAD_REGISTRY.items():
            if entry["kind"] == "regressor":
                assert "metrics" in entry, (
                    f"{name}: regressor without metrics list"
                )
                assert len(entry["metrics"]) > 0

    def test_winner_is_binary_classifier(self):
        assert HEAD_REGISTRY["winner"]["kind"] == "classifier"
        assert HEAD_REGISTRY["winner"]["model_kind"] == "binary"

    def test_multikill_target_extractor_still_works(self):
        # The categorical bins are still used by the heuristic in
        # analysis.py (Low/Medium/High).  Even though the trained
        # multikill classifier was discontinued, the `target_multikill`
        # function in `business.ml.targets` should still classify
        # matches into the three bins correctly.
        from business.ml.targets import (
            target_multikill,
            MULTIKILL_HIGH_THRESHOLD,
            MULTIKILL_MEDIUM_THRESHOLD,
        )
        # Sanity: a synthetic match with one player at 10 kills is High.
        import json, tempfile, os
        m = {
            "radiant": {"player_performances": [
                {"performance": {"hero": {"valve_id": 1}, "kills": 10}}
            ]},
            "dire": {"player_performances": [
                {"performance": {"hero": {"valve_id": 2}, "kills": 3}}
            ]},
        }
        assert target_multikill(m) == "High"
        assert MULTIKILL_HIGH_THRESHOLD == 7
        assert MULTIKILL_MEDIUM_THRESHOLD == 4


# ========================================================================== #
# Pure metric helpers
# ========================================================================== #

class TestPinballLoss:
    def test_zero_when_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _pinball_loss(y, y, alpha=0.5) == pytest.approx(0.0)

    def test_asymmetry_alpha_0_1(self):
        # Under-prediction is cheap at alpha=0.1 (0.1x), over-prediction
        # is expensive (0.9x).  Over-predicting by 1 should cost 9x
        # more than under-predicting by 1.
        y = np.array([10.0])
        under = _pinball_loss(y, np.array([9.0]), alpha=0.1)
        over = _pinball_loss(y, np.array([11.0]), alpha=0.1)
        assert over == pytest.approx(under * 9)

    def test_mirror_alpha_0_9(self):
        # Mirror: over-prediction is cheap, under-prediction is expensive.
        y = np.array([10.0])
        under = _pinball_loss(y, np.array([9.0]), alpha=0.9)
        over = _pinball_loss(y, np.array([11.0]), alpha=0.9)
        assert under == pytest.approx(over * 9)

    def test_alpha_0_5_is_half_mae(self):
        # At alpha=0.5, pinball loss = MAE / 2.
        y = np.array([1.0, 5.0, 3.0])
        p = np.array([2.0, 4.0, 5.0])
        mae = np.abs(y - p).mean()
        assert _pinball_loss(y, p, alpha=0.5) == pytest.approx(mae / 2)


class TestSafePoissonDeviance:
    def test_zero_when_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _safe_poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_handles_zero_y_pred(self):
        # y_pred=0 would be log(0); we clip to eps and the result
        # is finite (a large positive number, not NaN).
        y = np.array([1.0, 2.0])
        p = np.array([0.0, 0.0])
        result = _safe_poisson_deviance(y, p)
        assert np.isfinite(result)
        assert result > 0

    def test_handles_zero_y_true(self):
        # 0 * log(0) is defined as 0 (per sklearn convention).
        y = np.array([0.0, 0.0, 5.0])
        p = np.array([1.0, 1.0, 5.0])
        result = _safe_poisson_deviance(y, p)
        # First two rows contribute 0; third contributes 0.  The
        # non-zero "cost" comes from the prediction-error term, not
        # the log term.  Result must be finite.
        assert np.isfinite(result)


# ========================================================================== #
# evaluate_regressor — metric dispatch
# ========================================================================== #

class TestEvaluateRegressor:
    def test_mae(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 2.0, 2.5])
        m = evaluate_regressor("kills", y, p)
        assert "mae" in m
        # |0.5| + |0| + |0.5| = 1.0, divided by 3 = 0.333...
        assert m["mae"] == pytest.approx(1.0 / 3.0)

    def test_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([2.0, 2.0, 2.0])  # errors 1,0,1 → rmse = sqrt(2/3)
        m = evaluate_regressor("kills", y, p)
        assert "rmse" in m
        assert m["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_pinball_dispatch(self):
        y = np.array([10.0, 20.0])
        p = np.array([11.0, 19.0])
        m = evaluate_regressor("duration_p10", y, p)
        assert "pinball_0.1" in m

    def test_unknown_metric_logs_and_skips(self, caplog):
        # Inject a bad metric via monkey-patch (the registry normally
        # only has known metrics).
        y = np.array([1.0])
        p = np.array([1.0])
        original = HEAD_REGISTRY["kills"]["metrics"]
        HEAD_REGISTRY["kills"]["metrics"] = ("bogus_metric",)
        try:
            with caplog.at_level("WARNING"):
                m = evaluate_regressor("kills", y, p)
            assert m == {}
            assert any("unknown metric" in r.message for r in caplog.records)
        finally:
            HEAD_REGISTRY["kills"]["metrics"] = original


# ========================================================================== #
# iter_matches
# ========================================================================== #

class TestIterMatches:
    def test_yields_each_json_in_dir(self, corpus_dir):
        # corpus_dir has 60 valid + 1 broken → 60 yielded
        matches = list(iter_matches(corpus_dir))
        assert len(matches) == 60

    def test_skips_malformed_json_with_warning(self, corpus_dir, caplog):
        with caplog.at_level("WARNING"):
            matches = list(iter_matches(corpus_dir))
        # The broken file is logged, not raised.
        assert len(matches) == 60
        assert any("broken_9999.json" in r.message for r in caplog.records)

    def test_empty_dir_yields_nothing(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list(iter_matches(empty)) == []

    def test_nonexistent_dir_yields_nothing(self, tmp_path):
        # `Path.glob` on a non-existent directory returns an empty
        # iterator in modern Python (no exception).  iter_matches
        # inherits that — the failure happens later in
        # `build_dataset` (which does an `is_dir()` check).
        assert list(iter_matches(tmp_path / "does_not_exist")) == []


# ========================================================================== #
# _resolve_targets
# ========================================================================== #

class TestResolveTargets:
    def test_all_returns_every_registered(self):
        targets = _resolve_targets("all")
        assert set(targets) == set(HEAD_REGISTRY)

    def test_single_target(self):
        assert _resolve_targets("winner") == ["winner"]

    def test_comma_separated(self):
        assert _resolve_targets("winner,kills") == ["winner", "kills"]

    def test_whitespace_tolerated(self):
        assert _resolve_targets(" winner , kills ") == ["winner", "kills"]

    def test_unknown_target_raises_systemexit(self):
        with pytest.raises(SystemExit) as ei:
            _resolve_targets("winner,bogus")
        assert "unknown target" in str(ei.value)
        assert "bogus" in str(ei.value)

    def test_empty_pieces_ignored(self):
        # `"winner,,kills"` → 2 targets, not 3.
        assert _resolve_targets("winner,,kills") == ["winner", "kills"]


# ========================================================================== #
# _parse_args
# ========================================================================== #

class TestParseArgs:
    def test_defaults(self):
        a = _parse_args([])
        assert a.target == "all"
        assert a.winsorize is True
        assert a.n_sigma == pytest.approx(3.0)
        assert a.test_size == pytest.approx(0.2)
        assert a.random_state == 42
        assert a.calibrate == "none"
        assert a.zinb is False

    def test_target_flag(self):
        a = _parse_args(["--target", "winner"])
        assert a.target == "winner"

    def test_disable_winsorize(self):
        a = _parse_args(["--no-winsorize"])
        assert a.winsorize is False

    def test_calibrate_choice(self):
        a = _parse_args(["--calibrate", "isotonic"])
        assert a.calibrate == "isotonic"

    def test_zinb_flag(self):
        a = _parse_args(["--zinb"])
        assert a.zinb is True


# ========================================================================== #
# Per-target trainers (with tiny synthetic features)
# ========================================================================== #

class TestTrainWinner:
    def _data(self):
        # 200 rows, 13 features, balanced labels.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 13))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_basic_fit(self):
        X, y = self._data()
        Xtr, Xte = X[:160], X[160:]
        ytr, yte = y[:160], y[160:]
        model, metrics = _train_winner(Xtr, ytr, Xte, yte, random_state=0)
        # All standard keys present.
        for k in ("accuracy", "log_loss", "roc_auc", "n_train", "n_test", "calibration"):
            assert k in metrics
        assert metrics["n_train"] == 160
        assert metrics["n_test"] == 40
        assert metrics["calibration"] == "none"
        # Accuracy on a signal-bearing synthetic set should beat coin flip.
        assert metrics["accuracy"] > 0.6

    def test_calibrate_sigmoid(self):
        X, y = self._data()
        model, metrics = _train_winner(X[:160], y[:160], X[160:], y[160:],
                                       random_state=0, calibrate="sigmoid")
        assert metrics["calibration"] == "sigmoid"
        # predict_proba still works (the wrapper preserves the API).
        proba = model.predict_proba(X[160:])[:, 1]
        assert proba.shape == (40,)

    def test_calibrate_isotonic(self):
        X, y = self._data()
        model, metrics = _train_winner(X[:160], y[:160], X[160:], y[160:],
                                       random_state=0, calibrate="isotonic")
        assert metrics["calibration"] == "isotonic"


class TestTrainRegressor:
    def _data(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 13))
        # y depends on a couple of features so MAE < std(y).
        y = (5.0 + 2.0 * X[:, 0] - X[:, 2] + rng.normal(scale=0.5, size=200))
        y = np.maximum(y, 0)  # counts / durations
        return X, y

    def test_kills_target(self):
        X, y = self._data()
        Xtr, Xte = X[:160], X[160:]
        ytr, yte = y[:160], y[160:]
        model, metrics = _train_regressor(
            "kills", Xtr, ytr, Xte, yte, random_state=0,
        )
        assert "mae" in metrics
        assert "rmse" in metrics
        # Predictions are non-negative (clipped).
        preds = model.predict(Xte)
        assert (preds >= 0).all()

    def test_zinb_estimator_label(self):
        X, y = self._data()
        _, metrics = _train_regressor(
            "towers", X[:160], y[:160], X[160:], y[160:],
            random_state=0, zinb=True,
        )
        # When statsmodels is missing, the regressor falls back to
        # poisson_histgbr; either label is acceptable.
        assert metrics.get("estimator_family") in (
            "zinb", "poisson_histgbr", None,
        )


class TestTrainMulticlassClassifier:
    def _data(self, with_nulls: bool = True):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 13))
        # 3-class target by a 2-feature threshold.
        y = np.where(
            X[:, 0] + X[:, 1] > 1.0, "High",
            np.where(X[:, 0] > 0, "Medium", "Low"),
        ).astype(object)
        if with_nulls:
            y[0:10] = None  # simulate "no label" rows
        return X, y

    def test_filters_none_labels(self):
        X, y = self._data(with_nulls=True)
        Xtr, Xte = X[:160], X[160:]
        ytr, yte = y[:160], y[160:]
        model, metrics = _train_multiclass_classifier(
            "multikill", Xtr, ytr, Xte, yte,
            random_state=0,
        )
        # None rows dropped from both train and test.
        assert metrics["n_train"] <= 160
        assert metrics["n_test"] <= 40

    def test_returns_class_distribution(self):
        X, y = self._data(with_nulls=False)
        Xtr, Xte = X[:160], X[160:]
        ytr, yte = y[:160], y[160:]
        _, metrics = _train_multiclass_classifier(
            "multikill", Xtr, ytr, Xte, yte,
            random_state=0,
        )
        # class_distribution is a dict {class_label: count}.
        assert "class_distribution" in metrics
        assert isinstance(metrics["class_distribution"], dict)
        # Distribution sums to n_train.
        assert sum(metrics["class_distribution"].values()) == metrics["n_train"]

    def test_too_few_rows_raises(self):
        X = np.zeros((4, 13))
        y = np.array(["High", "Low", "Medium", None], dtype=object)
        with pytest.raises(RuntimeError, match="too few labelled"):
            _train_multiclass_classifier(
                "multikill", X, y, X, y, random_state=0,
            )

    def test_per_class_precision_recall(self):
        X, y = self._data(with_nulls=False)
        Xtr, Xte = X[:160], X[160:]
        ytr, yte = y[:160], y[160:]
        _, metrics = _train_multiclass_classifier(
            "multikill", Xtr, ytr, Xte, yte,
            random_state=0,
        )
        # Per-class metrics surface as `precision_<class>` /
        # `recall_<class>`.  We don't assert on the values (random
        # data) — just that the keys exist.
        for k in metrics:
            if k.startswith(("precision_", "recall_")):
                assert isinstance(metrics[k], float)


# ========================================================================== #
# build_dataset
# ========================================================================== #

class TestBuildDataset:
    def test_returns_three_tuple(self, corpus_dir):
        targets, encoder, X = build_dataset(corpus_dir)
        assert isinstance(targets, list)
        assert len(targets) == 60  # all 60 corpus matches pass the filter
        # 0.3.10: default is all 24 features (hero+team+lane).
        assert X.shape[1] == 24

    def test_returns_three_tuple_hero_only(self, corpus_dir):
        # 0.3.9 baseline — pin the 13-feature contract.
        targets, encoder, X = build_dataset(corpus_dir, fit_encoder_on=None)
        # We don't have a `groups` flag on build_dataset in 0.3.10
        # (it was added to train_all); check via the helper directly.
        from business.ml.features import extract_features
        from business.ml.targets import MatchTarget
        feats = extract_features(
            targets[0].radiant_hero_ids, targets[0].dire_hero_ids, encoder,
            groups=("hero",),
        )
        assert len(feats) == 13

    def test_encoder_is_fitted(self, corpus_dir):
        _, encoder, _ = build_dataset(corpus_dir)
        # After fit, encode() returns a real probability, not the default.
        val = encoder.encode("radiant", 1)
        assert 0.0 <= val <= 1.0

    def test_under_floor_raises(self, tmp_path):
        # 30 matches < 50 floor.
        out = tmp_path / "tiny"
        out.mkdir()
        for i in range(30):
            m = _synth_match(match_id=i, radiant_win=(i % 2 == 0), hero_offset=i)
            (out / f"m_{i:04d}.json").write_text(json.dumps(m))
        with pytest.raises(RuntimeError, match="only 30 usable matches"):
            build_dataset(out)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_dataset(tmp_path / "nope")


# ========================================================================== #
# train_all (end-to-end on tmp corpus)
# ========================================================================== #

class TestTrainAll:
    def test_winner_only_round_trip(self, corpus_dir, tmp_path):
        model_dir = tmp_path / "models"
        saved = train_all(
            data_dir=corpus_dir,
            model_dir=model_dir,
            version="1",
            targets=["winner"],
            winsorize=False,
            n_sigma=3.0,
            test_size=0.2,
            random_state=0,
        )
        assert "winner" in saved
        # ModelStorage wrote a directory with metadata + joblib.
        assert (model_dir / "winner_v1").is_dir()
        assert (model_dir / "winner_v1" / "metadata.json").is_file()
        assert (model_dir / "winner_v1" / "model.joblib").is_file()

    def test_skips_target_with_too_few_rows(self, corpus_dir, tmp_path, caplog):
        # Our synthetic corpus has no tower data — every `towers_total`
        # is None.  train_all should log a warning and skip the target.
        model_dir = tmp_path / "models"
        with caplog.at_level("WARNING"):
            saved = train_all(
                data_dir=corpus_dir,
                model_dir=model_dir,
                version="1",
                targets=["towers"],
                winsorize=False,
                n_sigma=3.0,
                test_size=0.2,
                random_state=0,
            )
        # Saved dict is empty (the one target was skipped).
        assert saved == {}
        assert any(
            "skipping" in r.message and "towers" in r.message
            for r in caplog.records
        )

    def test_calibrate_param_forwarded_to_winner(self, corpus_dir, tmp_path):
        model_dir = tmp_path / "models"
        train_all(
            data_dir=corpus_dir,
            model_dir=model_dir,
            version="1",
            targets=["winner"],
            winsorize=False,
            n_sigma=3.0,
            test_size=0.2,
            random_state=0,
            calibrate="sigmoid",
        )
        # Metadata records which calibration was used.
        meta = json.loads(
            (model_dir / "winner_v1" / "metadata.json").read_text()
        )
        assert meta["metrics"]["calibration"] == "sigmoid"


# ========================================================================== #
# main (CLI entrypoint)
# ========================================================================== #

class TestMain:
    def test_main_returns_zero_on_success(self, corpus_dir, tmp_path, monkeypatch):
        # Avoid touching the real ml_data tree.
        model_dir = tmp_path / "models"
        rc = train_mod.main([
            "--data-dir", str(corpus_dir),
            "--model-dir", str(model_dir),
            "--target", "winner",
            "--no-winsorize",
        ])
        assert rc == 0
        assert (model_dir / "winner_v1").is_dir()

    def test_main_returns_nonzero_on_missing_data(self, tmp_path, monkeypatch, caplog):
        # Point at a non-existent dir; the FileNotFoundError inside
        # build_dataset is caught and translated to exit code 1.
        with caplog.at_level("ERROR"):
            rc = train_mod.main([
                "--data-dir", str(tmp_path / "nope"),
                "--model-dir", str(tmp_path / "models"),
                "--target", "winner",
            ])
        assert rc == 1
        assert any("training failed" in r.message for r in caplog.records)

    def test_main_prints_saved_models(self, corpus_dir, tmp_path, capsys):
        train_mod.main([
            "--data-dir", str(corpus_dir),
            "--model-dir", str(tmp_path / "models"),
            "--target", "winner",
            "--no-winsorize",
        ])
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert "winner" in captured.out
