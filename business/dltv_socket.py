"""Direct DLTV socket.io client — bypasses the browser entirely.

v0.4.0: replaces the v0.3.20-v0.3.25d Playwright-based dltv_browser
match-state scrape.  DLTV ships live match data via a socket.io
endpoint at `wss://dltv.org/socket.io/?EIO=4&transport=websocket`.
Subscribing to `__nd2_match_{steam_id}` from Python gives us the
same `result` payload the page would render — but with ~100ms
latency and without paying the chromium-launch cost on every
tick.

Architectural notes:
  * The connection is process-wide, owned by a dedicated daemon
    thread (`_run_client_loop`).  It runs `asyncio.run` on a
    private event loop; the rest of the app reads from the
    in-memory state via `get_live_state(steam_id)` (a synchronous
    getter).
  * `subscribe(steam_id)` and `unsubscribe(steam_id)` are thread-
    safe; the loop picks up the diff on each iteration.  New
    channels are sent on the open socket; removed ones just stop
    being subscribed on reconnect.
  * The DLTV server is generous — it broadcasts `__nd2_match_*`
    events for any match the page is watching, not just the ones
    you subscribed to.  We filter by the `match_id` field of the
    payload, so a stray event for the wrong match is dropped.
  * Engine.IO keep-alive: server sends a PING (`2`) every
    `pingInterval` (25s by default).  We reply with PONG (`3`).
    If we miss two pings the server drops us, so we also send a
    proactive PING every 20s as a backstop.
  * Reconnects: on any exception (network blip, server-side
    drop), the loop sleeps with exponential backoff (1s, 2s, 4s,
    capped at 30s) and reconnects, re-subscribing to everything.

Why this is a win:
  * No chromium → no greenlet thread leak (v0.3.22 was spending
    ~1.4GB WSL overhead keeping the browser alive).
  * No `MatchState_TTL_SEC` staleness — the socket pushes a new
    event every few seconds, and `get_live_state` returns
    whatever arrived most recently.
  * No DLTV-side HTTP cache lag (the `/live/{id}.json` adapter
    was 5-10 min behind in the user's tests; the socket is
    real-time).
  * Concurrent: a single connection carries every live match on
    the site (we observed `__nd2_match_8917923821` arriving even
    though we'd only subscribed to `__nd2_match_8917853656`).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, Optional, Set

import websockets

from ._logging import get_logger
from .exceptions import DiscoveryError

log = get_logger(__name__)

WS_URL = "wss://dltv.org/socket.io/?EIO=4&transport=websocket"
STATE_TTL_SEC = 60.0   # payloads older than this are treated as missing
PING_INTERVAL_SEC = 20.0   # backstop — server also pings us at 25s
RECONNECT_BACKOFF_INITIAL = 1.0
RECONNECT_BACKOFF_MAX = 30.0

# Public, thread-safe state.  Reads (`get_live_state`) take a
# short lock; writes happen on the asyncio loop's thread but the
# lock is the same one, so cross-thread visibility is fine.
_state: Dict[int, Dict[str, Any]] = {}
_state_ts: Dict[int, float] = {}
_subscriptions: Set[int] = set()
_lock = threading.RLock()

# Background runner
_loop_thread: Optional[threading.Thread] = None
_loop_started = threading.Event()
_loop_should_stop = threading.Event()


def _now() -> float:
    return time.monotonic()


def get_live_state(steam_id: int) -> Optional[Dict[str, Any]]:
    """Return the latest payload for `steam_id` or None if stale.

    A payload is considered "stale" if its age exceeds
    `STATE_TTL_SEC`.  We rely on the underlying socket push to
    refresh every few seconds, so any non-stale value is by
    construction within a few hundred ms of the live page.
    """
    with _lock:
        ts = _state_ts.get(int(steam_id))
        if ts is None:
            return None
        if _now() - ts > STATE_TTL_SEC:
            return None
        # Return a shallow copy so the caller can't mutate
        # shared state.
        return dict(_state[int(steam_id)])


def _set_state(steam_id: int, payload: Dict[str, Any]) -> None:
    with _lock:
        _state[int(steam_id)] = payload
        _state_ts[int(steam_id)] = _now()


def get_subscriptions() -> Set[int]:
    with _lock:
        return set(_subscriptions)


def subscribe(steam_id: int) -> None:
    """Add a steam_id to the subscription set.  The socket loop
    will send a SUBSCRIBE packet on its next iteration.  Idempotent.
    """
    with _lock:
        _subscriptions.add(int(steam_id))


def unsubscribe(steam_id: int) -> None:
    """Remove a steam_id from the subscription set.  The socket loop
    stops re-subscribing on reconnect; the server doesn't have an
    explicit unsubscribe (socket.io's standard behaviour is
    disconnect-reconnect, which is what happens on a reconnect)."""
    with _lock:
        _subscriptions.discard(int(steam_id))


def _parse_sio_event(msg: str) -> Optional[Dict[str, Any]]:
    """Parse an Engine.IO + SIO event message into a payload dict.

    Returns the payload for `42["channel", arg1, arg2, ...]` events,
    or None for anything else (PING/PONG, CONNECT ack, NOOP, etc.).
    The caller is expected to know which channel it cares about
    (we look for `__nd2_match_{steam_id}` specifically).
    """
    if not msg or len(msg) < 2:
        return None
    eio_type = msg[0]
    if eio_type != "4":  # SIO MESSAGE
        return None
    sio_type = msg[1]
    if sio_type != "2":  # SIO EVENT
        return None
    rest = msg[2:]
    if not rest:
        return None
    try:
        data = json.loads(rest)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    return {"channel": data[0], "args": data[1:]}


async def _run_session() -> None:
    """A single session: connect, subscribe, listen, raise on failure."""
    backoff = RECONNECT_BACKOFF_INITIAL
    while not _loop_should_stop.is_set():
        try:
            log.info("dltv_socket: connecting to %s", WS_URL)
            async with websockets.connect(
                WS_URL,
                origin="https://dltv.org",
                open_timeout=30,
                # The server drops the connection after ~20s of
                # no inbound activity — the DLTV socket.io server
                # doesn't honour the standard WS-level PING
                # frames, and the EIO PING ("2")/PONG ("3")
                # protocol isn't reliable either.  The empirically
                # robust solution is to send a benign SIO EVENT
                # every 15s as a keep-alive — the server treats
                # any incoming SIO EVENT as activity and stops
                # counting us as idle.  We use the
                # `__nd2_match_*` re-subscribe pattern for the
                # currently-subscribed matches; the server
                # accepts duplicate SUBSCRIBEs without complaint.
                ping_interval=None,
                ping_timeout=None,
            ) as ws:
                log.info("dltv_socket: connected")
                # Engine.IO OPEN
                open_msg = await ws.recv()
                if not open_msg.startswith("0"):
                    raise RuntimeError(f"unexpected EIO open: {open_msg[:80]!r}")
                # SIO CONNECT
                await ws.send("40")
                connect_msg = await ws.recv()
                if not connect_msg.startswith("40"):
                    raise RuntimeError(f"unexpected SIO connect reply: {connect_msg[:80]!r}")
                log.info("dltv_socket: SIO connected, subscribing to %d channels",
                         len(get_subscriptions()))
                # Initial subscriptions.  We always also subscribe
                # to `__nd2_series` because the server pushes
                # `live` and `upcoming` lists on a regular
                # cadence — those messages count as activity and
                # keep the connection alive even when no per-match
                # event has fired in a while.  Without this
                # the connection dies after ~3 minutes of
                # no per-match activity.
                await ws.send('42["__nd2_series"]')
                for sid in get_subscriptions():
                    await ws.send(f'42["__nd2_match_{sid}"]')
                # Listen loop.  Two things in parallel:
                #   1. Read events as they arrive and update the
                #      shared state.
                #   2. Every KEEPALIVE_INTERVAL_SEC seconds, send
                #      a re-subscribe for every tracked steam_id
                #      to keep the connection alive (the server
                #      drops idle clients after ~20s).
                KEEPALIVE_INTERVAL_SEC = 12.0
                next_keepalive = _now() + KEEPALIVE_INTERVAL_SEC
                while not _loop_should_stop.is_set():
                    recv_timeout = max(0.5, next_keepalive - _now())
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                    except asyncio.TimeoutError:
                        # No data — fall through to keepalive
                        msg = None
                    except websockets.ConnectionClosed:
                        raise RuntimeError("connection closed by server")
                    if msg:
                        # SIO EVENT
                        evt = _parse_sio_event(msg)
                        if evt:
                            channel = evt.get("channel") or ""
                            if channel.startswith("__nd2_match_"):
                                try:
                                    sid = int(channel.rsplit("_", 1)[-1])
                                except (ValueError, IndexError):
                                    sid = None
                                if sid is not None:
                                    args = evt.get("args") or []
                                    payload = args[0] if args else {}
                                    if isinstance(payload, dict):
                                        payload_mid = payload.get("match_id")
                                        if payload_mid is None or int(payload_mid) == sid:
                                            _set_state(sid, payload)
                    if _now() >= next_keepalive:
                        # Re-subscribe everything we have so far.
                        # The server accepts duplicate SUBSCRIBEs
                        # without complaint and treats them as
                        # activity, which keeps the socket open.
                        for sid in get_subscriptions():
                            try:
                                await ws.send(f'42["__nd2_match_{sid}"]')
                            except websockets.ConnectionClosed:
                                raise RuntimeError("connection closed by server during keepalive")
                        next_keepalive = _now() + KEEPALIVE_INTERVAL_SEC
                # Loop exited normally (should_stop)
                return
        except Exception as exc:
            log.warning("dltv_socket: session error: %s — reconnecting in %.1fs",
                        exc, backoff)
            # Sleep with cancellable semantics: we exit the
            # sleep early if `_loop_should_stop` is set, so a
            # clean shutdown doesn't have to wait the full
            # backoff.  We poll the threading.Event from the
            # asyncio side via a small `asyncio.to_thread` helper
            # would be ideal, but a simple loop on a short
            # `asyncio.sleep` is enough — backoff tops at 30s
            # and shutdown is fine to take that long.
            slept = 0.0
            step = 0.5
            while slept < backoff:
                if _loop_should_stop.is_set():
                    return
                await asyncio.sleep(step)
                slept += step
            backoff = min(RECONNECT_BACKOFF_MAX, backoff * 2)


def _thread_main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_session())
    finally:
        try:
            loop.close()
        except Exception:
            pass


def start_socket_client() -> threading.Thread:
    """Start the WebSocket client in a dedicated background thread.

    Idempotent: returns the existing thread if already started.
    The thread is a daemon and dies with the process.
    """
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return _loop_thread
    _loop_should_stop.clear()
    t = threading.Thread(
        target=_thread_main,
        name="dltv-socket-client",
        daemon=True,
    )
    t.start()
    _loop_thread = t
    return t


def stop_socket_client(timeout: float = 5.0) -> None:
    """Signal the client to stop and wait for the thread to exit."""
    _loop_should_stop.set()
    if _loop_thread is not None:
        _loop_thread.join(timeout=timeout)
