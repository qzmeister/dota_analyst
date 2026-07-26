"""
Unit tests for live accuracy tracking (v0.3.15+).

The accuracy module persists predictions in `ml_data/live_predictions.jsonl`
and computes aggregate stats.  These tests exercise the round-trip on a
temp directory so we don't pollute the real log.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_ml_data(monkeypatch, tmp_path):
    """Redirect ML_DATA_DIR to a temp dir so tests don't touch ml_data/."""
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path))
    # Re-import the module so its module-level constants pick up
    # the new env var (ML_DATA_DIR is read at import time).
    import importlib
    from business import accuracy
    importlib.reload(accuracy)
    yield tmp_path
    importlib.reload(accuracy)


class TestRecordPrediction:
    def test_returns_full_record(self, tmp_ml_data):
        from business.accuracy import record_prediction
        r = record_prediction(
            match_id=12345, series_id=678,
            predicted_winner="team_a", predicted_probability=0.7,
            engine="ml", extra={"team_a_id": 100, "team_b_id": 200,
                                "team_a_name": "Dire", "team_b_name": "Radi",
                                "game_no": 1},
        )
        assert r["match_id"] == 12345
        assert r["series_id"] == 678
        assert r["predicted_winner"] == "team_a"
        assert r["predicted_probability"] == 0.7
        assert r["engine"] == "ml"
        assert r["scored"] is None
        # extra fields are flattened into the record
        assert r["team_a_name"] == "Dire"
        assert r["game_no"] == 1

    def test_dedup_per_match_game(self, tmp_ml_data):
        """Second call for same (match_id, game_no) is a no-op."""
        from business.accuracy import record_prediction
        kwargs = dict(
            match_id=12345, series_id=678,
            predicted_winner="team_a", predicted_probability=0.7,
            engine="ml", extra={"game_no": 1},
        )
        r1 = record_prediction(**kwargs)
        r2 = record_prediction(**kwargs)
        assert "_skipped" in r2
        assert r2["_skipped"] == "already_recorded"
        # Only one row was actually written
        log = list((Path(tmp_ml_data) / "live_predictions.jsonl").read_text().splitlines())
        assert len(log) == 1

    def test_different_game_no_records_again(self, tmp_ml_data):
        """Same match, but a new game (game 2 of a Bo3) is a new record."""
        from business.accuracy import record_prediction
        r1 = record_prediction(match_id=12345, series_id=678,
                               predicted_winner="team_a", predicted_probability=0.7,
                               engine="ml", extra={"game_no": 1})
        r2 = record_prediction(match_id=12345, series_id=678,
                               predicted_winner="team_b", predicted_probability=0.6,
                               engine="ml", extra={"game_no": 2})
        assert "_skipped" not in r1
        assert "_skipped" not in r2

    def test_needs_match_or_series(self, tmp_ml_data):
        from business.accuracy import record_prediction
        from business.exceptions import AccuracyError
        with pytest.raises(AccuracyError):
            record_prediction(match_id=None, series_id=None,
                              predicted_winner="team_a", predicted_probability=0.5,
                              engine="ml")


class TestComputeStats:
    def test_empty_log_returns_zeros(self, tmp_ml_data):
        from business.accuracy import accuracy_summary
        s = accuracy_summary()
        assert s["total"] == 0
        assert s["scored"] == 0
        assert s["pending"] == 0
        assert s["accuracy"] is None

    def test_engine_breakdown(self, tmp_ml_data):
        from business.accuracy import record_prediction, accuracy_summary, _dedup_seen, _dedup_loaded
        # Force a clean dedup cache (reload may have left it populated)
        _dedup_seen.clear()
        _dedup_loaded = False
        # Three predictions, two engines
        for i, eng in enumerate(["ml", "ml", "heuristic"]):
            _dedup_seen.clear()  # allow multiple writes by bypassing dedup
            record_prediction(
                match_id=10000 + i, series_id=20000 + i,
                predicted_winner="team_a", predicted_probability=0.6,
                engine=eng, extra={"game_no": 1, "team_a_id": 1, "team_b_id": 2,
                                   "team_a_name": "A", "team_b_name": "B"},
            )
        s = accuracy_summary()
        # Without a real DLTV match, we can't auto-score.  So all 3
        # remain pending.
        assert s["total"] == 3
        assert s["pending"] == 3


class TestCompare:
    """The verdict logic — pure function, no I/O."""

    def test_correct_when_side_matches(self, tmp_ml_data):
        from business.accuracy import _compare
        rec = {
            "predicted_winner": "team_a",
            "extra": {"team_a_id": 100, "team_b_id": 200,
                      "team_a_name": "Dire", "team_b_name": "Radi"},
        }
        info = {"winner": 100, "ended_at": "2026-07-26T10:00:00+00:00"}
        assert _compare(rec, info) == "correct"

    def test_wrong_when_side_differs(self, tmp_ml_data):
        from business.accuracy import _compare
        rec = {
            "predicted_winner": "team_b",
            "extra": {"team_a_id": 100, "team_b_id": 200,
                      "team_a_name": "Dire", "team_b_name": "Radi"},
        }
        info = {"winner": 100, "ended_at": "2026-07-26T10:00:00+00:00"}
        assert _compare(rec, info) == "wrong"

    def test_push_on_draw(self, tmp_ml_data):
        from business.accuracy import _compare
        rec = {
            "predicted_winner": "team_a",
            "extra": {"team_a_id": 100, "team_b_id": 200,
                      "team_a_name": "Dire", "team_b_name": "Radi"},
        }
        assert _compare(rec, {"winner": "draw", "ended_at": "2026-07-26T10:00:00+00:00"}) == "push"

    def test_no_teams_marker(self, tmp_ml_data):
        from business.accuracy import _compare
        rec = {"predicted_winner": "team_a", "extra": {}}
        # No team_a_id/team_b_id → can't decide, mark as "no_teams"
        assert _compare(rec, {"winner": 100, "ended_at": "x"}) == "no_teams"


class TestEndpoint:
    """Smoke test: GET /api/accuracy returns the summary shape."""

    def test_accuracy_endpoint_shape(self, tmp_ml_data, monkeypatch):
        from business import app as app_module
        from business import stream as stream_module
        from fastapi.testclient import TestClient

        async def _no_publisher(*_a, **_k):
            import asyncio
            await asyncio.Event().wait()
        async def _fake_event_stream(*_a, **_k):
            yield b": keepalive\n\n"
        monkeypatch.setattr(app_module, "board_publisher_loop", _no_publisher)
        monkeypatch.setattr(app_module, "event_stream", _fake_event_stream)
        monkeypatch.setenv("PREDICTION_ENGINE", "heuristic")
        from business.ml.engine import reset_default_engine
        reset_default_engine()

        with TestClient(app_module.app) as c:
            r = c.get("/api/accuracy")
            assert r.status_code == 200
            body = r.json()
            assert "total" in body
            assert "scored" in body
            assert "accuracy" in body
            assert "last24h" in body
            assert "engine_breakdown" in body
        reset_default_engine()
