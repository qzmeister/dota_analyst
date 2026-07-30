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

    v0.3.25t-patch: split `build_board` (CPU/IO bound, blocking) from
    the asyncio tick.  The previous version used `asyncio.to_thread()`
    + `await asyncio.sleep(interval)`, but in our setup the thread
    blocks forever (chromium / urllib deadlock, or some other in-
    process contention with the asyncio default executor) — and
    `asyncio.to_thread` does NOT cancel the actual running thread on
    cancellation, so the publisher's `build_board` calls pile up in
    the default executor and the loop never produces a fresh board.

    The fix: a plain `threading.Thread` daemon runs `build_board` in
    a loop, updating `_app._latest_auto_board` in place.  The
    asyncio half of this function just polls the cache every
    `interval_sec` and pushes SSE.  Pure `asyncio.sleep` — no
    chance of an asyncio-side deadlock cascading into a stuck
    thread.

    Only this publisher fix is applied; the v0.3.25l (socket.io
    hook), v0.3.25m (game_time normalisation), v0.3.25n (duration
    fallback), v0.3.25r (TBD filter) and v0.3.25s (0-picks filter)
    changes are NOT included — that whole block caused live matches
    to disappear from the UI and was rolled back by the user.
    """
    from . import app as _app  # for the module-level board cache (v0.3.14+)
    from .board import build_board  # local import — avoids cycle on app startup
    from .ml.engine import get_default_engine

    import threading as _threading
    import time as _t

    log.info("sse publisher loop started; interval=%.1fs", interval_sec)

    # v0.4.0-cache: keep the last NON-EMPTY board as a fallback so the UI
    # never blanks out when the publisher build times out / hangs / returns
    # an empty payload.  The publisher thread can take 5-9 minutes per cycle
    # (DLTV scrape + /live/{id}.json enrichment for 30+ live matches); without
    # this fallback the SSE clients would receive 0/0/0 every 5 seconds
    # during that window.  We only count a board as "good" if it has at
    # least one item in `live + prematch + postmatch`.
    _BOARD_CACHE_TTL_SEC = 300.0
    _last_good_board: Optional[Dict[str, Any]] = None
    _last_good_ts: float = 0.0

    def _has_content(b: Dict[str, Any]) -> bool:
        return bool(
            (b.get("live") or [])
            or (b.get("prematch") or [])
            or (b.get("postmatch") or [])
        )

    _stop = _threading.Event()

    def _build_loop() -> None:
        log.info("sse publisher: board-builder thread started")
        while not _stop.is_set():
            t0 = _t.monotonic()
            # v0.4.0.2: heartbeat so the watchdog can detect a frozen
            # or silently-killed thread.  Updated every cycle (the
            # try/except below runs once per loop iteration).
            try:
                _publisher_last_heartbeat[0] = _t.monotonic()
            except NameError:
                pass  # before the watchdog initialised; safe to skip
            try:
                board = build_board([], [])
                if "engine" not in board:
                    board["engine"] = get_default_engine().name
                if _has_content(board):
                    _app._latest_auto_board = board
                    _app._latest_auto_board_ts = _t.monotonic()
                    # Only refresh the fallback when we have something
                    # non-empty to fall back to.  Stale-but-non-empty is
                    # better than empty.
                    global _last_good_board, _last_good_ts
                    _last_good_board = board
                    _last_good_ts = _t.monotonic()
                else:
                    # Empty build — try the fallback if it's fresh enough.
                    if (
                        _last_good_board is not None
                        and (_t.monotonic() - _last_good_ts) < _BOARD_CACHE_TTL_SEC
                    ):
                        age = _t.monotonic() - _last_good_ts
                        log.info(
                            "sse publisher: empty build, serving cached board "
                            "(age=%.1fs, live=%d, prematch=%d, postmatch=%d)",
                            age,
                            len(_last_good_board.get("live") or []),
                            len(_last_good_board.get("prematch") or []),
                            len(_last_good_board.get("postmatch") or []),
                        )
                        # Bump the timestamp so the UI's "обновлено HH:MM"
                        # counter keeps moving — better UX than a stuck clock.
                        _app._latest_auto_board = _last_good_board
                        _app._latest_auto_board_ts = _t.monotonic()
                    else:
                        # No fallback (or it's too old) — publish the empty
                        # board so the UI at least knows the publisher is alive.
                        _app._latest_auto_board = board
                        _app._latest_auto_board_ts = _t.monotonic()
                log.info(
                    "sse publisher: build_board done in %.2fs (live=%d)",
                    _t.monotonic() - t0,
                    len(board.get("live", [])),
                )
                # v0.4.0: subscribe every live match's steam_id to
                # the direct socket.io client.  This replaces the
                # Playwright `dltv_browser` scrape as the source of
                # real-time live state (see `dltv_socket` for the
                # full rationale).  We pull steam_ids from the
                # freshly-built board (it carries the resolution
                # the discovery tracker has done), plus from the
                # watchlist (in case a watchlist pin is the only
                # way the user follows that match).
                try:
                    from . import dltv_socket as _ds
                    _prev_subs = _ds.get_subscriptions()
                    for col in ("live", "prematch"):
                        for c in (board.get(col) or []):
                            mid = c.get("match_id")
                            if mid is not None:
                                _ds.subscribe(int(mid))
                    # v0.4.0-fix: DLTV rejects mid-session SUBSCRIBE
                    # packets.  If the subscription set grew since the
                    # last reconnect (a new live match appeared, or an
                    # old one left the board), the new steam_ids will
                    # not get any events until the socket reconnects
                    # naturally (~45-90 s).  Request a cooperative
                    # reconnect so the new picks/bans/lead show up on
                    # the live card within ~5 s.  `force_reconnect()`
                    # throttles itself to once per 30 s.
                    _new_subs = _ds.get_subscriptions()
                    if _new_subs - _prev_subs:
                        _ds.force_reconnect()
                        log.info(
                            "sse publisher: %d new live match(es), requested socket reconnect",
                            len(_new_subs - _prev_subs),
                        )
                except Exception as exc:  # never let socket bugs kill the publisher
                    log.debug("dltv_socket subscribe failed: %s", exc)
            except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
                log.warning("sse publisher build failed: %s", exc, exc_info=False)
            except Exception as exc:
                # catch-all so a coding bug in build_board (or any
                # unanticipated exception) cannot silently kill the
                # publisher.  Without this the SSE clients would keep
                # getting the same stale auto-board for hours.
                log.exception("sse publisher build crashed (recovered): %s", exc)
            # Tick at interval_sec; break early if cancelled.
            _stop.wait(interval_sec)
        log.info("sse publisher: board-builder thread stopped")

    # v0.4.0.2: watchdog for the board-builder thread.  The thread
    # was observed dead (still in process) after a long stretch of
    # DLTV timeouts — its outer try/except recovered from individual
    # `build_board` exceptions but something else (we suspect a
    # garbage-collected closure or a daemon-thread teardown race)
    # killed the whole `while not _stop.is_set()` loop.  Symptoms:
    # dltv_socket has 0 subscriptions (no `dltv_socket.subscribe()`
    # calls from the dead publisher), `live=0` everywhere, but the
    # container is still healthy.  Hard to repro — the only reliable
    # mitigation is a watchdog that re-starts the thread if it dies.
    # Heartbeat: `_publisher_last_heartbeat` is updated every cycle
    # in the build loop.  The watchdog compares it to now() and
    # considers the thread dead if it hasn't ticked in 3× the
    # configured interval.
    _publisher_last_heartbeat = [_t.monotonic()]
    _publisher_thread_ref = [None]  # filled in below; replaced on each restart

    def _heartbeat_loop() -> None:
        """Watch the board-builder thread.  If it stops heart-beating
        for > 3× the build interval, log loudly and re-spawn it.
        """
        _watchdog_interval = max(interval_sec * 3, 15.0)
        while not _stop.is_set():
            _t.sleep(_watchdog_interval)
            last = _publisher_last_heartbeat[0]
            age = _t.monotonic() - last
            thr = _publisher_thread_ref[0]
            if thr is not None and not thr.is_alive():
                log.warning(
                    "sse publisher: board-builder thread is DEAD "
                    "(last heartbeat %.1fs ago) — restarting",
                    age,
                )
                new_t = _threading.Thread(
                    target=_build_loop, name="board-builder", daemon=True
                )
                new_t.start()
                _publisher_thread_ref[0] = new_t
            elif age > _watchdog_interval:
                log.warning(
                    "sse publisher: board-builder thread heartbeat "
                    "stale (%.1fs since last tick, watchdog_interval=%.1fs)",
                    age, _watchdog_interval,
                )

    _board_builder_thread = _threading.Thread(
        target=_build_loop, name="board-builder", daemon=True
    )
    _board_builder_thread.start()
    _publisher_thread_ref[0] = _board_builder_thread
    _threading.Thread(
        target=_heartbeat_loop, name="publisher-watchdog", daemon=True
    ).start()

    # The asyncio half just watches the cache and pushes SSE
    # when something changed.  Sleep `interval_sec` between
    # checks; the publisher's own throttle (publish_if_changed)
    # de-dupes unchanged boards.
    try:
        while True:
            try:
                await asyncio.sleep(interval_sec)
                board = _app._latest_auto_board
                if not board:
                    continue
                delivered = await stream.publish_if_changed(board)
                if delivered:
                    log.debug("sse publish delivered to %d subscribers", delivered)
            except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
                log.warning("sse publisher tick failed: %s", exc, exc_info=False)
            except Exception as exc:
                log.exception("sse publisher tick crashed (recovered): %s", exc)
    except asyncio.CancelledError:
        _stop.set()
        log.info("sse publisher loop cancelled")
        raise


# --------------------------------------------------------------------------- #
# Player.win_rate refresh — Playwright-driven, 5 s cadence
# --------------------------------------------------------------------------- #
#
# Phase 3: the DLTV v1 API doesn't ship live player.win_rate; only the
# rendered HTML at /matches/{series_id}/{slug} does.  We poll the
# Playwright-based scraper every 5 s and write the result to a JSON
# cache so the next predict call (which doesn't have time to spin
# up chromium) can read it.  Caching is essential because chromium
# launch is ~2 s — the board's 5 s publisher cycle would never
# fit if we did this inline.
#
# v0.3.24e: dropped from 30s to 5s so the match-state overlay
# (`dltv_browser.get_cached_match_state`) stays close to the
# socket.io cadence DLTV's own page uses.  At 30s the live card
# showed scores that lagged the DLTV page by 5-30s, defeating
# the point of the cache.  MATCH_STATE_TTL_SEC was bumped in
# v0.3.24d to 30s specifically to survive a 5s publisher tick
# — the original design intent — so bringing the publisher back
# to 5s completes that round of fixes.  Heavy work (chromium
# fetch) is bounded by `MATCH_STATE_TTL_SEC`: the publisher
# only re-fetches when the cache has expired OR the cache is
# empty, so a 5s tick on a 30s TTL means ~1 fetch per match
# per 30s, not per tick.

PLAYER_WR_POLL_INTERVAL_SEC = 5.0


async def player_wr_browser_loop(interval_sec: float = PLAYER_WR_POLL_INTERVAL_SEC) -> None:
    """Refresh the player.win_rate + live match-state cache.

    v0.3.20+: in addition to the career-win_rate scrape, this loop
    also pulls the live picks/score/time for every live match with
    a URL.  DLTV's v1 API hides in-progress series, so the only
    source for picks during a best-of-N is the rendered HTML
    page — and that's the same Playwright fetch we already do
    for win rates, so we reuse the connection.

    Runs alongside the SSE publisher; uses `asyncio.to_thread` so a
    slow dltv.org page doesn't stall the event loop.  We deliberately
    run fetches serially — chromium has its own event loop and a
    single page per match is the simplest invariant.
    """
    import asyncio as _aio
    from . import dltv_browser
    from .discovery import tracker as _tracker
    log.info("player_wr browser loop started; interval=%.1fs", interval_sec)
    try:
        while True:
            try:
                # Snapshot the live series with a URL — these are the
                # ones the scraper found on /matches.  We need the
                # (series_id, url, steam_id) triple: the dltv series id
                # is the primary cache key (the publisher writes it);
                # the steam id is an alias key so the watchlist path
                # can find the same data after the tracker is pruned
                # (which happens immediately when a match ends).
                #
                # v0.4.0-bans: also iterate `_by_steam` (Steam-only
                # matches that never had a dltv URL — usually Chinese
                # amateur / minor-league games).  For these we
                # construct a synthetic URL `dltv.org/matches/{steam_id}`
                # which DLTV usually redirects to the real series
                # page; if it doesn't, chromium will 404 and we just
                # skip the row.  Without this branch the live card's
                # bans row stays empty for the whole game because no
                # source ever queries the dltv page for the draft.
                with _tracker._lock:
                    targets = []
                    for sid, m in _tracker._by_series.items():
                        if m.get("stage") != "live" or not m.get("url"):
                            continue
                        # Tracker rows have either top-level `steam_id`
                        # (scraper format) or `maps[0].steam_id`
                        # (watchlist format).  Check both.
                        steam_id = m.get("steam_id")
                        if not steam_id:
                            maps = m.get("maps") or []
                            if maps and isinstance(maps[0], dict):
                                steam_id = maps[0].get("steam_id")
                        try:
                            steam_id = int(steam_id) if steam_id is not None else None
                        except (TypeError, ValueError):
                            steam_id = None
                        targets.append((int(sid), m.get("url"), steam_id))
                    # Steam-only matches: no dltv URL, but we can try
                    # a redirect-style URL `dltv.org/matches/{steam_id}`.
                    for steam_id, m in _tracker._by_steam.items():
                        if m.get("stage") != "live":
                            continue
                        try:
                            sid_int = int(steam_id)
                        except (TypeError, ValueError):
                            continue
                        # Skip if the dltv cache is already populated
                        # for this steam id (the dltv_browser lookup
                        # by steam alias covers this case).
                        if dltv_browser.get_cached_match_state_by_steam(sid_int) is not None:
                            continue
                        url = f"https://dltv.org/matches/{sid_int}"
                        # Use the steam id as a "series id" stand-in
                        # so the cache write is keyed consistently;
                        # the publisher also writes the steam-id
                        # alias key, so a subsequent board.py lookup
                        # by either key will find the data.
                        targets.append((sid_int, url, sid_int))
                for sid_int, url, steam_id in targets:
                    # v0.3.20: match state is the more important
                    # thing — without it the live card shows empty
                    # picks.  Try it first; the WR fetch piggybacks
                    # on the same page load (separate call to the
                    # module, but the cache write is shared).
                    if dltv_browser.get_cached_match_state(sid_int) is None:
                        await _aio.to_thread(dltv_browser.update_match_state_cache,
                                             sid_int, url, steam_id)
                    if dltv_browser.get_cached_player_winrates(sid_int) is None:
                        await _aio.to_thread(dltv_browser.update_player_wr_cache,
                                             sid_int, url)
            except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
                log.warning("player_wr loop tick failed: %s", exc, exc_info=False)
            except Exception as exc:
                # Anything else (e.g. dltv_browser not imported
                # because the worker thread is the wrong one) — log
                # and keep going.  This is a "best effort" task.
                log.warning("player_wr loop unexpected: %s", exc, exc_info=False)
            await _aio.sleep(interval_sec)
    except _aio.CancelledError:
        log.info("player_wr browser loop cancelled")
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
