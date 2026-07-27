"""
Smoke tests for `business.app` — the FastAPI service endpoints.

Strategy:
  - Use `fastapi.testclient.TestClient` to drive the real app
    (so middleware, exception handlers and the lifespan
    startup hook all run).
  - Mock the external collaborators: `client.get_events`,
    `client.get_heroes`, `build_board`, the SSE publisher.
  - We don't go deep on response shape here — the unit tests
    for `board.py` and the A/B harness in `scripts/eval_engines.py`
    cover that.  These are the "does it boot and answer 200"
    smoke tests.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Module-level fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_collaborators(monkeypatch):
    """Stub out the heavy collaborators `business.app` calls.

    `client.get_events` and `client.get_heroes` are wired into
    the readiness check and the build_board path.  `build_board`
    is stubbed to return a minimal valid payload so /api/board
    doesn't try to talk to DLTV.
    """
    fake_client = MagicMock()
    fake_client.get_events.return_value = []
    fake_client.get_heroes.return_value = [{"id": 1, "name": "axe"}]

    monkeypatch.setattr("business.app.client", fake_client)
    # `app` is imported lazily so the TestClient startup picks
    # up the patched client.
    return fake_client


@pytest.fixture
def client(mock_collaborators, monkeypatch):
    """Build a TestClient around the FastAPI app.

    Two collaborators are stubbed so the lifespan hook and the
    SSE endpoint don't try to talk to anything:
      - `business.stream.board_publisher_loop` → sleeps forever
        (cancelled by the lifespan's `finally` block)
      - `business.app.event_stream` → yields a single keepalive
        byte and returns, so the SSE endpoint closes immediately
        instead of blocking the test on a long-lived queue.
    """
    # Import inside the fixture so `monkeypatch` is in scope.
    from business import app as app_module

    async def _no_publisher(*_args, **_kwargs):
        import asyncio
        await asyncio.Event().wait()

    # Both `board_publisher_loop` and `event_stream` are imported
    # into `app.py` via `from .stream import ...`, so the binding
    # the lifespan / endpoint sees lives in `business.app.__dict__`,
    # not `business.stream.__dict__`.  Patching the *stream* module
    # would miss the import shadowing and the real function would
    # still run.  Patch `app_module` instead.
    monkeypatch.setattr(app_module, "board_publisher_loop", _no_publisher)

    async def _fake_event_stream(*_args, **_kwargs):
        yield b": test keepalive\n\n"
    monkeypatch.setattr(app_module, "event_stream", _fake_event_stream)

    monkeypatch.setenv("PREDICTION_ENGINE", "heuristic")
    from business.ml.engine import reset_default_engine
    reset_default_engine()

    with TestClient(app_module.app) as c:
        yield c

    reset_default_engine()


# --------------------------------------------------------------------------- #
# Health endpoints
# --------------------------------------------------------------------------- #

class TestHealth:
    def test_healthz_returns_ok(self, client):
        r = client.get("/api/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readyz_returns_ready_when_client_ok(self, client, mock_collaborators):
        # `get_heroes` is set up by the mock_collaborators fixture.
        r = client.get("/api/readyz")
        # The readyz endpoint pings `client.get_heroes()` which our
        # mock returns a list for — that path is the "ready" branch.
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_readyz_returns_not_ready_on_client_failure(self, client, mock_collaborators):
        # The endpoint catches `DLTVError` (and the broader upstream family)
        # — any error from the DLTV client stack translates to a 503.  We
        # raise a DLTVError directly so the test exercises the real
        # exception type, not a generic RuntimeError.
        from business.exceptions import DLTVError
        mock_collaborators.get_heroes.side_effect = DLTVError("dltv down")
        r = client.get("/api/readyz")
        assert r.status_code == 503
        assert r.json()["status"] == "not-ready"
        assert "dltv down" in r.json()["error"]


# --------------------------------------------------------------------------- #
# Read endpoints (board / leagues)
# --------------------------------------------------------------------------- #

class TestLeagues:
    def test_leagues_returns_list(self, client, mock_collaborators):
        # v0.3.21+: /api/leagues reads the v1 events list
        # directly (we don't go through leagues_with_status()
        # any more — that was too slow under cold cache).
        # mock_collaborators is the fixture that already stubs
        # business.app.client; we just add get_events().
        from business import app as app_module
        app_module.client.get_events.return_value = [
            {"id": 42, "title": "DreamLeague", "is_active": 1},
            {"id": 43, "title": "ESL One", "is_active": 1},
        ]
        r = client.get("/api/leagues")
        assert r.status_code == 200
        body = r.json()
        assert "leagues" in body
        # Sorted by match_count desc, then title asc.  Both have
        # count=0 (empty auto-board), so we sort alphabetically.
        ids = [L["id"] for L in body["leagues"]]
        assert ids == [42, 43]
        for L in body["leagues"]:
            assert L["match_count"] == 0
            assert "status" not in L  # the legacy status field is gone


class TestBoard:
    def test_board_returns_full_dict(self, client, monkeypatch):
        # Stub `build_board` so we don't need real DLTV data.
        async def _stub():
            return {"prematch": [], "live": [], "postmatch": []}
        # In app.py, `build_board` is called as a function — patch
        # the symbol on the app module.
        monkeypatch.setattr(
            "business.app.build_board",
            lambda *a, **kw: {
                "prematch": [{"stage": "prematch", "series_id": 1}],
                "live": [],
                "postmatch": [],
            },
        )
        r = client.get("/api/board")
        assert r.status_code == 200
        body = r.json()
        assert "prematch" in body
        assert "live" in body
        assert "postmatch" in body
        assert "engine" in body  # the engine name surfaces here

    def test_board_includes_engine_field(self, client, monkeypatch):
        monkeypatch.setattr(
            "business.app.build_board",
            lambda *a, **kw: {"prematch": [], "live": [], "postmatch": []},
        )
        r = client.get("/api/board")
        body = r.json()
        # heuristic default in test env.
        assert body["engine"] in ("heuristic", "ml")

    def test_board_event_ids_dedup_and_parse(self, client, monkeypatch):
        # v0.3.22 cont 4: filtered requests are served from the
        # auto-board (filtered server-side), so build_board is NOT
        # called for them.  We seed the auto-board with a known
        # shape and assert the dedup + parse logic happens on the
        # `events=` query string.
        from business import app as _app
        _app._latest_auto_board = {
            "prematch": [
                {"event_id": 1, "event": "A", "team_a": {"name": "a"}, "team_b": {"name": "b"}},
                {"event_id": 2, "event": "B", "team_a": {"name": "a"}, "team_b": {"name": "b"}},
                {"event_id": 3, "event": "C", "team_a": {"name": "a"}, "team_b": {"name": "b"}},
                # duplicate of eid=1 (should pass filter; not in user filter though)
                {"event_id": 1, "event": "A", "team_a": {"name": "a"}, "team_b": {"name": "b"}},
                # unmapped (eid=None) — should be dropped when there's a filter
                {"event_id": None, "event": "Steam league 99", "team_a": {"name": "a"}, "team_b": {"name": "b"}},
            ],
            "live": [],
            "postmatch": [],
            "engine": "ml",
        }
        import time as _t
        _app._latest_auto_board_ts = _t.monotonic()

        r = client.get("/api/board?events=1,2,2,3&watch=4,4,5")
        assert r.status_code == 200
        body = r.json()
        # The user's `events=1,2,2,3` is deduped to [1, 2, 3] and
        # reflected in `selected`.  Each requested event_id is
        # present at least once in the cards (the auto-board
        # itself has 4 in-set cards + 1 unmapped).
        eids = [c["event_id"] for c in body["prematch"]]
        for want in (1, 2, 3):
            assert want in eids, f"missing eid={want}: {eids}"
        # Unmapped cards (eid=None) are dropped when the user has
        # narrowed the board — that's the strict live filter.
        assert None not in eids
        assert body["selected"] == [1, 2, 3]
        assert body["watch"] == [4, 5]


# --------------------------------------------------------------------------- #
# SSE endpoint
# --------------------------------------------------------------------------- #

class TestSse:
    def test_stream_endpoint_is_text_event_stream(self, client):
        # We use `client.stream(...)` so the streaming response is
        # read incrementally and we can verify the headers.
        with client.stream("GET", "/api/stream/matches") as r:
            assert r.status_code == 200
            # media_type is set on the StreamingResponse in app.py.
            assert r.headers["content-type"].startswith("text/event-stream")
            # The polling publisher is stubbed to sleep forever, so
            # the first SSE frame we'd see is the keepalive comment.
            # We don't need to drain it; the headers are enough.
            # Read at most a tiny chunk to confirm the response is open.
            for _ in r.iter_bytes():
                break
