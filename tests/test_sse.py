"""
Unit tests for `business.stream.MatchStream` — the in-process pub-sub
that backs the SSE endpoint.

The async generator (`event_stream`) is exercised end-to-end
through the FastAPI test client in `test_app_streaming.py`;
these tests focus on the bucket math + lifecycle.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from business.stream import (
    MatchStream,
    StreamEvent,
    _hash_board,
    _keepalive_comment,
    _summary,
    get_stream,
    reset_stream,
)


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

def _board(live: int = 0, prematch: int = 0, postmatch: int = 0) -> dict:
    return {
        "live": [{"id": i} for i in range(live)],
        "prematch": [{"id": i} for i in range(prematch)],
        "postmatch": [{"id": i} for i in range(postmatch)],
    }


# --------------------------------------------------------------------------- #
# StreamEvent
# --------------------------------------------------------------------------- #

class TestStreamEvent:
    def test_render_includes_event_id_data(self):
        e = StreamEvent(event="board_update", data={"x": 1}, id=42)
        out = e.render().decode("utf-8")
        assert out.startswith("event: board_update\n")
        assert "id: 42\n" in out
        assert 'data: {"x": 1}\n' in out
        # SSE frame ends with a blank line.
        assert out.endswith("\n\n")

    def test_render_escapes_unicode_safely(self):
        e = StreamEvent(event="x", data={"name": "ТБ"})
        # ensure_ascii=False so the JSON keeps the Cyrillic as-is.
        assert "ТБ" in e.render().decode("utf-8")

    def test_keepalive_is_a_comment(self):
        # SSE comments start with ':' and are invisible to onmessage.
        out = _keepalive_comment()
        assert out.startswith(b": ping\n\n")


# --------------------------------------------------------------------------- #
# MatchStream — subscribe / publish / unsubscribe
# --------------------------------------------------------------------------- #

class TestMatchStreamLifecycle:
    @pytest.mark.asyncio
    async def test_subscribe_increments_count(self):
        s = MatchStream()
        assert s.subscriber_count == 0
        q1 = await s.subscribe()
        q2 = await s.subscribe()
        assert s.subscriber_count == 2
        await s.unsubscribe(q1)
        await s.unsubscribe(q2)
        assert s.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_publish_puts_event_on_every_queue(self):
        s = MatchStream()
        q1 = await s.subscribe()
        q2 = await s.subscribe()
        board = _board(live=2)
        delivered = await s.publish_if_changed(board)
        assert delivered == 2
        e1 = q1.get_nowait()
        e2 = q2.get_nowait()
        assert isinstance(e1, StreamEvent)
        assert e1.event == "board_update"
        assert e1.data["summary"]["live"] == 2
        assert e2.data["summary"]["live"] == 2

    @pytest.mark.asyncio
    async def test_unchanged_board_is_not_republished(self):
        s = MatchStream()
        q = await s.subscribe()
        b = _board(live=1)
        assert await s.publish_if_changed(b) == 1
        # Drain the first event so we can detect whether a second
        # publish sneaks through.
        q.get_nowait()
        # Second publish with the same board → no-op.
        assert await s.publish_if_changed(b) == 0
        assert q.empty()

    @pytest.mark.asyncio
    async def test_changed_board_is_republished(self):
        s = MatchStream()
        q = await s.subscribe()
        b1 = _board(live=1)
        b2 = _board(live=1, prematch=3)  # different
        assert await s.publish_if_changed(b1) == 1
        q.get_nowait()  # drain
        assert await s.publish_if_changed(b2) == 1
        e = q.get_nowait()
        assert e.data["summary"]["prematch"] == 3


class TestMatchStreamBackpressure:
    @pytest.mark.asyncio
    async def test_slow_subscriber_is_dropped(self):
        # With max_queue=1 each subscriber can buffer exactly one
        # event before the next put_nowait raises QueueFull.  We
        # drain the fast subscriber between publishes so it never
        # gets full, while the slow subscriber piles up.
        s = MatchStream(max_queue=1)
        slow_q = await s.subscribe()
        fast_q = await s.subscribe()

        b1 = _board(live=1)
        b2 = _board(live=2)
        b3 = _board(live=3)
        b4 = _board(live=4)

        # Publish #1 — both subscribers' queues accept (1/1 each).
        assert await s.publish_if_changed(b1) == 2
        # Drain fast_q; slow_q stays at 1/1 (full).
        fast_q.get_nowait()

        # Publish #2 — slow_q QueueFull → drop; fast_q 1/1 → OK.
        assert await s.publish_if_changed(b2) == 1
        assert s.subscriber_count == 1
        fast_q.get_nowait()

        # Publish #3 — slow_q already dropped; fast_q 1/1 → OK.
        delivered = await s.publish_if_changed(b3)
        assert delivered == 1
        fast_q.get_nowait()

        # Publish #4 — same; the dropped slow_q stays gone.
        delivered = await s.publish_if_changed(b4)
        assert delivered == 1


# --------------------------------------------------------------------------- #
# Hashing — what counts as "changed"
# --------------------------------------------------------------------------- #

class TestHash:
    def test_hash_ignores_engine_label(self):
        # Two boards that differ ONLY in `engine` should hash equal
        # so a heuristic↔ml toggle doesn't spam the client.
        a = _board(live=1)
        a["engine"] = "heuristic"
        b = _board(live=1)
        b["engine"] = "ml"
        assert _hash_board(a) == _hash_board(b)

    def test_hash_changes_when_live_count_changes(self):
        a = _board(live=1)
        b = _board(live=2)
        assert _hash_board(a) != _hash_board(b)

    def test_hash_changes_when_live_score_changes(self):
        a = _board(live=1)
        a["live"] = [{"score": 5}]
        b = _board(live=1)
        b["live"] = [{"score": 6}]
        assert _hash_board(a) != _hash_board(b)

    def test_summary_counts_cards(self):
        s = _summary(_board(live=2, prematch=3, postmatch=1))
        assert s == {"live": 2, "prematch": 3, "postmatch": 1}


# --------------------------------------------------------------------------- #
# Module singleton
# --------------------------------------------------------------------------- #

class TestModuleSingleton:
    def test_get_stream_returns_singleton(self):
        reset_stream()
        a = get_stream()
        b = get_stream()
        assert a is b

    def test_reset_drops_singleton(self):
        reset_stream()
        a = get_stream()
        reset_stream()
        b = get_stream()
        assert a is not b
