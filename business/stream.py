"""
SSE (Server-Sent Events) pub-sub for live match updates.

Why polling?  A 5-second polling loop calling `build_board()` is
good enough for 0.1.1 — the discovery and DLTV layers are
already cached (`_TTLCache` on events / series), so the cost of
"is anything new?" is one hash compare per tick.  The Redis
pub-sub upgrade in 0.2.x will swap the publisher without
touching the subscriber interface.

Protocol
--------
Each SSE message has:

    event: board_update       (or "ping" for the keepalive)
    id: <monotonic>
    data: <json>

Clients consume the `board_update` event to repaint the Kanban.
The `ping` event arrives every ~30s as a comment (": ping\\n\\n")
to keep proxies from killing the connection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from ._logging import get_logger
from .exceptions import (
    BoardBuildError,
    DiscoveryError,
    InfraError,
    MLError,
    UpstreamError,
)


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #

@dataclass
class StreamEvent:
    """A single SSE-bound event.

    `data` is serialised as JSON.  `event` defaults to "board_update"
    but may be set to anything the protocol allows (we use "ping"
    for keepalives).
    """
    event: str
    data: Dict[str, Any]
    id: int = 0

    def render(self) -> bytes:
        """Encode as an SSE frame (`event: ...\\nid: ...\\ndata: ...\\n\\n`)."""
        lines: List[str] = [f"event: {self.event}", f"id: {self.id}"]
        # `data:` accepts a multi-line value; for our use case the
        # payload is single-line JSON.
        lines.append("data: " + json.dumps(self.data, ensure_ascii=False))
        return ("\n".join(lines) + "\n\n").encode("utf-8")


def _keepalive_comment() -> bytes:
    """SSE comment (starts with ':') — invisible to `onmessage`."""
    return b": ping\n\n"


# --------------------------------------------------------------------------- #
# MatchStream — the in-process pub-sub
# --------------------------------------------------------------------------- #

class MatchStream:
    """In-process broadcast hub for board updates.

    Subscribers get an `asyncio.Queue`; the publisher puts the
    latest `build_board()` payload into every queue whenever it
    changes.  When a subscriber is closed (its queue is no longer
    being drained) we drop it on the next publish — the queue
    becomes full and the publish is a no-op for that subscriber.
    """

    def __init__(self, max_queue: int = 8) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._max_queue = max_queue
        self._last_hash: Optional[str] = None
        self._tick_id = 0
        self._lock = asyncio.Lock()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def subscribe(self) -> asyncio.Queue:
        """Open a new subscription.  Returns the queue to `get()` from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        async with self._lock:
            self._subscribers.add(q)
        log.info("sse subscribe; total=%d", len(self._subscribers))
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)
        log.info("sse unsubscribe; total=%d", len(self._subscribers))

    async def publish_if_changed(self, board: Dict[str, Any]) -> int:
        """Push `board` to every subscriber iff its hash changed.

        Returns the number of subscribers that actually got the
        payload (others had full queues and were dropped).
        """
        h = _hash_board(board)
        async with self._lock:
            if h == self._last_hash:
                return 0
            self._last_hash = h
            self._tick_id += 1
            event = StreamEvent(
                event="board_update",
                data={"engine": board.get("engine", "?"), "summary": _summary(board)},
                id=self._tick_id,
            )
            delivered = 0
            stale: List[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                    delivered += 1
                except asyncio.QueueFull:
                    stale.append(q)
            # A full queue means the client isn't draining — drop
            # it so we don't leak memory on a slow / dead consumer.
            for q in stale:
                self._subscribers.discard(q)
            return delivered


# --------------------------------------------------------------------------- #
# Hash + summary helpers
# --------------------------------------------------------------------------- #

def _hash_board(board: Dict[str, Any]) -> str:
    """Stable hash of the parts of the board that matter for "is it new?".

    We deliberately ignore `selected` / `watch` (caller-controlled)
    and `engine` (label, not data).  We DO include the live / prematch
    / postmatch payloads so a score change shows up.
    """
    blob = json.dumps(
        {
            "live": board.get("live", []),
            "prematch": board.get("prematch", []),
            "postmatch": board.get("postmatch", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _summary(board: Dict[str, Any]) -> Dict[str, int]:
    """A tiny per-stage count so the client can show "X live, Y prematch"."""
    return {
        "live": len(board.get("live") or []),
        "prematch": len(board.get("prematch") or []),
        "postmatch": len(board.get("postmatch") or []),
    }


# --------------------------------------------------------------------------- #
# Module singleton
# --------------------------------------------------------------------------- #

_default_stream: Optional[MatchStream] = None


def get_stream() -> MatchStream:
    """Return the process-wide `MatchStream` instance."""
    global _default_stream
    if _default_stream is None:
        _default_stream = MatchStream()
    return _default_stream


def reset_stream() -> None:
    """Drop the cached stream (used by tests)."""
    global _default_stream
    _default_stream = None


# --------------------------------------------------------------------------- #
# Background poller — wired in `app.py` startup
# --------------------------------------------------------------------------- #

async def board_publisher_loop(
    stream: MatchStream,
    interval_sec: float = 5.0,
) -> None:
    """Poll `build_board()` and publish to the stream.

    Runs forever (until the event loop is cancelled by shutdown).
    The function lives in `stream.py` so `app.py` can register
    it as a `lifespan` task without taking on the implementation.
    """
    from .board import build_board  # local import — avoids cycle on app startup
    from .ml.engine import get_default_engine

    log.info("sse publisher loop started; interval=%.1fs", interval_sec)
    try:
        while True:
            try:
                engine = get_default_engine()
                board = build_board([], watch_ids=[], engine=engine)
                delivered = await stream.publish_if_changed(board)
                if delivered:
                    log.debug("sse publish delivered to %d subscribers", delivered)
            except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
                # Poller is the safety net for the whole data pipeline —
                # a network blip or a single bad scrape must not kill the
                # background task.  We catch our full exception tree but
                # deliberately let `KeyboardInterrupt`/`SystemExit` and
                # coding bugs in our own modules surface.
                log.warning("sse publisher tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        log.info("sse publisher loop cancelled")
        raise


async def event_stream(
    stream: MatchStream,
    keepalive_sec: float = 30.0,
) -> AsyncIterator[bytes]:
    """Async generator for the SSE endpoint.

    Yields SSE frames: `board_update` payloads when the publisher
    has new data, and a `: ping` comment every `keepalive_sec`
    seconds so the connection doesn't get culled by an idle
    proxy.
    """
    q = await stream.subscribe()
    last_ping = time.monotonic()
    try:
        while True:
            try:
                # Wait up to `keepalive_sec` for the next event.
                event: StreamEvent = await asyncio.wait_for(
                    q.get(), timeout=keepalive_sec,
                )
                yield event.render()
            except asyncio.TimeoutError:
                last_ping = time.monotonic()
                yield _keepalive_comment()
    except asyncio.CancelledError:
        # Client disconnected — drop the subscription.
        pass
    finally:
        await stream.unsubscribe(q)
