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
# v0.4.0-ttl: bumped 60s -> 5min.  DLTV stops sending fresh
# events for a match once nothing in it is changing (e.g. mid-
# game dead-time, or all picks/bans done and only the gold
# trickle updating).  The publisher build cadence is 5s, but
# if we lose 3-4 ticks in a row because the server went quiet
# the previous 60s window dropped the last good values and the
# live card snapped back to "no data".  5 minutes is still
# well under "this match is over" so the data stays useful.
STATE_TTL_SEC = 300.0
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

# v0.4.0.3: series-level state pushed by the `__nd2_series`
# channel.  This is the fallback discovery source we use when the
# HTTP scraper is unreachable (DNS, Cloudflare, etc.): DLTV's
# socket.io broadcast gives us the live + upcoming + results
# lists in a single dict, so the board can keep showing cards
# even when the scraper is dark.  Layout (verified on 2026-07-30):
#   {
#     "live":     {"<steam_id>": <series_id>, ...},    # live matches
#     "upcoming": [ {id, event_id, status, slug, type,
#                    first_team_id, second_team_id,
#                    started_at, ended_at, is_active, ...}, ... ],
#     "results":  [ <same shape as upcoming> ]         # recently finished
#   }
# The `live` dict is the most valuable: it carries the
# steam_id ↔ series_id mapping that the HTML scraper frequently
# omits for live cards (DLTV doesn't render `data-match="..."` on
# the matches page until the game is well underway).
_series_live: Dict[int, int] = {}            # steam_id -> series_id
_series_upcoming: List[Dict[str, Any]] = []  # upcoming series dicts
_series_results: List[Dict[str, Any]] = []   # finished series dicts
_series_ts: float = 0.0                      # monotonic stamp of last push
_SERIES_TTL_SEC = 600.0                      # 10 min — list refreshes are rare

# v0.4.0-bans: persistent bans cache.  DLTV's live socket.io
# payload carries `db.{first,second}_team.bans` only during the
# draft phase.  Once the game starts, the `bans` array goes
# empty in every subsequent payload (the draft is "done", DLTV
# doesn't ship historical draft data on the live channel).  We
# snapshot the last non-empty bans into this dict so the live
# card can show the full draft (picks + bans) throughout the
# game.  TTL is much longer than the main `_state` because bans
# never change mid-game — 4h covers the longest pro match (Bo5
# with 1+ hour pauses) plus a comfortable margin.
_BANS_CACHE_TTL_SEC = 4 * 3600.0
_bans_cache: Dict[int, Dict[str, Any]] = {}
_bans_ts: Dict[int, float] = {}

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


def get_series_state() -> Dict[str, Any]:
    """Snapshot of the `__nd2_series` channel state.

    v0.4.0.3: this is the second-socket fallback for discovery.
    Returns a dict with three lists (deep-copied) and the
    monotonic timestamp of the last push.  When the data is
    older than `_SERIES_TTL_SEC` the lists are still returned
    (the caller may want them as a "last known" snapshot) but
    the `stale` flag is set so the caller can decide.

    Shape:
        {
            "live":     {steam_id(int): series_id(int), ...},
            "upcoming": [series dict, ...],
            "results":  [series dict, ...],
            "ts":       <monotonic float>,
            "stale":    bool,
        }
    """
    with _lock:
        ts = _series_ts
        stale = (ts == 0.0) or ((_now() - ts) > _SERIES_TTL_SEC)
        return {
            "live":     {int(k): int(v) for k, v in _series_live.items()},
            "upcoming": [dict(s) for s in _series_upcoming],
            "results":  [dict(s) for s in _series_results],
            "ts":       ts,
            "stale":    bool(stale),
        }


def get_steam_id_for_series(series_id: int) -> Optional[int]:
    """Reverse lookup: given a DLTV series_id, return the
    live steam_id (if any) the `__nd2_series` channel knows about.

    v0.4.0.3: this is the bridge between the scraper's series_id
    and the socket-only steam_id.  When a live card is missing
    its steam_id (because the HTML didn't render `data-match="…"`)
    but the socket broadcast has the pair, this function returns
    it so `_live_card` can hydrate picks/teams from the live
    payload.
    """
    with _lock:
        for steam_id, sid in _series_live.items():
            if int(sid) == int(series_id):
                return int(steam_id)
    return None


def get_cached_bans(steam_id: int) -> Optional[Dict[str, Any]]:
    """Return the last non-empty bans for `steam_id` (or None).

    v0.4.0-bans: the live socket payload drops `bans` once the
    draft is over, but the UI needs to keep showing them.  This
    cache preserves the last draft-phase payload so the live card
    can render picks AND bans throughout the game.  Returns
    `{"first_bans": [...], "second_bans": [...],
    "first_is_radiant": bool}` or None if we never saw a non-empty
    draft for this match (e.g. live card rendered between
    subscription and the first event).
    """
    with _lock:
        ts = _bans_ts.get(int(steam_id))
        if ts is None:
            return None
        if _now() - ts > _BANS_CACHE_TTL_SEC:
            return None
        snap = _bans_cache.get(int(steam_id))
        if snap is None:
            return None
        # Return a shallow copy so the caller can't mutate shared state.
        return {
            "first_bans": list(snap.get("first_bans") or []),
            "second_bans": list(snap.get("second_bans") or []),
            "first_is_radiant": bool(snap.get("first_is_radiant", False)),
        }


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
    global _force_reconnect_flag, _last_force_reconnect_ts
    backoff = RECONNECT_BACKOFF_INITIAL
    # v0.4.0.2: heartbeat.  The watchdog re-spawns the whole thread
    # if it stops heart-beating for >90s.  We update this on every
    # successful connection AND on every received message so a
    # frozen recv() doesn't go unnoticed.
    while not _loop_should_stop.is_set():
        try:
            log.info("dltv_socket: connecting to %s", WS_URL)
            async with websockets.connect(
                WS_URL,
                origin="https://dltv.org",
                open_timeout=30,
                # v0.4.0 dev: WS-level PINGs cause the DLTV
                # server to close the connection quickly (within
                # 30-60s in the app, even though the standalone
                # test survives 2+ minutes).  We disable the
                # library's PING and rely on the EIO PING/PONG
                # that the server initiates every 25s — that
                # flow IS handled correctly.  The bottleneck
                # is something else in the app's network path
                # (likely the build's heavy HTTP fetches
                # saturating the container's outbound
                # bandwidth); we mitigate by reconnecting on
                # any close.
                ping_interval=None,
                ping_timeout=None,
            ) as ws:
                log.info("dltv_socket: connected")
                # v0.4.0.2: heartbeat stamp on every successful
                # connection.  The watchdog compares this to
                # `now()` and re-spawns the thread if it goes
                # stale.
                _heartbeat()
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
                # Listen loop.  Empirical finding (v0.4.0 dev): the
                # DLTV server does NOT want unsolicited re-
                # SUBSCRIBEs while the connection is alive — it
                # treats them as junk and closes the socket
                # after a few minutes.  The server pushes events
                # for every subscribed match on its own cadence
                # (multiple per second during a live game), so
                # we don't need to keep the connection alive
                # ourselves.  We just block on recv() and
                # process events as they arrive; the connection
                # dies naturally when the server decides to
                # rotate us (typically 1-2 minutes per session
                # with 44 subscriptions) and the outer
                # `except` re-connects.
                while not _loop_should_stop.is_set():
                    # Cooperative force-reconnect: the publisher
                    # may have added new live matches to the
                    # subscription set after we connected.  DLTV
                    # rejects mid-session SUBSCRIBE packets so we
                    # have to drop and re-open to pick them up.
                    # We do this lazily: we don't poll aggressively
                    # (would burn CPU), we check on every recv (≤1
                    # per second) and at most every `_FORCE_RECONNECT_MIN_SEC`.
                    if _force_reconnect_flag and (now := _now()) - _last_force_reconnect_ts > _FORCE_RECONNECT_MIN_SEC:
                        with _lock:
                            _force_reconnect_flag = False
                            _last_force_reconnect_ts = now
                        log.info("dltv_socket: force_reconnect requested — dropping session")
                        raise RuntimeError("force_reconnect requested")
                    try:
                        msg = await ws.recv()
                    except websockets.ConnectionClosed:
                        raise RuntimeError("connection closed by server")
                    # v0.4.0.2: heartbeat on every received message
                    # (PING, SIO EVENT, anything).  Combined with the
                    # connection-stamp above, this guarantees the
                    # watchdog sees a live thread as long as the
                    # socket is alive.  If recv() hangs for >90s
                    # the watchdog will assume the thread is stuck
                    # and start a new one.
                    _heartbeat()
                    if not msg:
                        continue
                    # v0.4.0.1: EIO PING (server → client) → reply with
                    # EIO PONG (`"3"`).  The previous code passed `"2"`
                    # through `_parse_sio_event` which returns None for
                    # non-SIO frames, so we silently dropped the ping.
                    # After 2 missed pings (~50s with the default
                    # `pingInterval=25`) the server closes the WS, which
                    # is why our sessions used to die every 30-90 s and
                    # the publisher had to wait for the 5-min
                    # `_BOARD_CACHE_TTL` to recover.  The docstring at
                    # the top of this module already promised this
                    # behaviour ("We reply with PONG (`3`)") — this is
                    # the code change that makes it true.
                    if msg == "2":
                        try:
                            await ws.send("3")
                        except websockets.ConnectionClosed:
                            raise RuntimeError("connection closed by server")
                        continue
                    # SIO EVENT
                    evt = _parse_sio_event(msg)
                    if not evt:
                        continue
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
                                    # v0.4.0-bans: cache the draft-phase
                                    # bans for as long as the match stays
                                    # in _state.  The live socket payload
                                    # drops `db.{first,second}_team.bans`
                                    # once the draft is over (we observed
                                    # the array goes empty the moment the
                                    # game starts), so without this cache
                                    # the live card's bans row would
                                    # disappear right when the match is
                                    # most interesting to watch.
                                    db = payload.get("db")
                                    if isinstance(db, dict):
                                        first = db.get("first_team") or {}
                                        second = db.get("second_team") or {}
                                        first_bans = first.get("bans") or []
                                        second_bans = second.get("bans") or []
                                        if first_bans or second_bans:
                                            with _lock:
                                                _bans_cache[sid] = {
                                                    "first_bans": list(first_bans),
                                                    "second_bans": list(second_bans),
                                                    "first_is_radiant": bool(first.get("is_radiant")),
                                                }
                                                _bans_ts[sid] = _now()
                                        # v0.7.48: signal the board-builder
                                        # that live state changed.  Throttled
                                        # in `_maybe_wake_build` so a chatty
                                        # match (DLTV pushes every 1-2s)
                                        # doesn't pile up rebuilds.
                                        _maybe_wake_build()
                    elif channel == "__nd2_series":
                        # v0.4.0.3: the broadcast channel that pushes
                        # the live / upcoming / results lists.  We
                        # were already subscribed to this for
                        # keepalive (the comment above called it out
                        # as "DLTV pushes live+upcoming lists on a
                        # regular cadence") — now we actually parse
                        # the payload and store it for the discovery
                        # tracker to use as a fallback when the HTTP
                        # scraper is dark.
                        #
                        # Layout:
                        #   {"live":     {<steam_id>: <series_id>, ...},
                        #    "upcoming": [ {id, event_id, ...}, ... ],
                        #    "results":  [ {id, event_id, ...}, ... ]}
                        #
                        # The `live` dict is the prize — it bridges
                        # the scraper's series_id (which it gets from
                        # the HTML) to the socket's steam_id (which
                        # the scraper often misses on live cards).
                        args = evt.get("args") or []
                        # v0.4.0.3: the `__nd2_series` payload
                        # arrives as `42["__nd2_series", {...}]`, but
                        # during testing we also saw `42["__nd2_series"]`
                        # with no args (a refresh nudge).  We only
                        # update state when there's a dict to read.
                        payload = args[0] if args else None
                        if not isinstance(payload, dict):
                            continue
                        live = payload.get("live") or {}
                        upcoming = payload.get("upcoming") or []
                        results = payload.get("results") or []
                        new_live: Dict[int, int] = {}
                        if isinstance(live, dict):
                            for k, v in live.items():
                                try:
                                    new_live[int(k)] = int(v)  # type: ignore[arg-type]
                                except (TypeError, ValueError):
                                    continue
                        new_upcoming: List[Dict[str, Any]] = []
                        if isinstance(upcoming, list):
                            for s in upcoming:
                                if isinstance(s, dict):
                                    new_upcoming.append(dict(s))
                        new_results: List[Dict[str, Any]] = []
                        if isinstance(results, list):
                            for s in results:
                                if isinstance(s, dict):
                                    new_results.append(dict(s))
                        with _lock:
                            _series_live = new_live
                            _series_upcoming = new_upcoming
                            _series_results = new_results
                            _series_ts = _now()
                        # Heartbeat already stamped above; log the
                        # update for visibility on a noisy channel.
                        if new_live or new_upcoming or new_results:
                            log.info(
                                "dltv_socket: __nd2_series push — "
                                "live=%d upcoming=%d results=%d",
                                len(new_live), len(new_upcoming), len(new_results),
                            )
                            # v0.7.48: the series list (live + upcoming +
                            # results) is the cheapest authoritative source
                            # for "is there a new match in town?".  Wake
                            # the board-builder so a brand-new live match
                            # shows up in <1s instead of waiting for the
                            # 5s periodic tick.
                            _maybe_wake_build()
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


_force_reconnect_flag: bool = False
_last_force_reconnect_ts: float = 0.0
# v0.4.0.2: heartbeat timestamp (monotonic seconds since epoch).
# The watchdog re-spawns `_run_session` if this goes stale.
_socket_heartbeat: float = 0.0
# Watchdog tuning
_SOCKET_WATCHDOG_INTERVAL_SEC = 30.0
_SOCKET_WATCHDOG_DEAD_AFTER_SEC = 90.0  # 3× the poll interval
_socket_watchdog_thread: Optional[threading.Thread] = None


def _heartbeat() -> None:
    """Stamp the heartbeat timestamp — called by `_run_session`
    on every successful connection AND on every received message
    so a frozen recv() doesn't go unnoticed.
    """
    global _socket_heartbeat
    _socket_heartbeat = time.monotonic()


def _socket_watchdog_loop() -> None:
    """Re-spawn the dltv-socket-client thread if it dies.

    v0.4.0.2: We observed the socket thread silently going to
    `thread: None` after a long stretch of DLTV DNS failures.  The
    inner reconnect loop caught the connection errors but
    something else (we suspect a daemon-thread / atexit race)
    killed the outer `while not _loop_should_stop.is_set()` loop.
    This watchdog keeps the publisher's data flow alive even when
    DLTV is misbehaving.

    Two restart triggers:
      1. `thr is None or not thr.is_alive()` — the thread died
         outright.  Re-spawn.
      2. Heartbeat older than `_SOCKET_WATCHDOG_DEAD_AFTER_SEC`
         but the thread object is still alive — the recv() loop
         is stuck.  We can't safely kill it from another thread;
         we let the OS-level GC handle it and start a new one.
    """
    global _loop_thread
    while True:
        time.sleep(_SOCKET_WATCHDOG_INTERVAL_SEC)
        thr = _loop_thread
        age = (
            time.monotonic() - _socket_heartbeat
            if _socket_heartbeat
            else None
        )
        if thr is None or not thr.is_alive():
            log.warning(
                "dltv-socket-watchdog: socket thread is DEAD — restarting"
            )
            try:
                start_socket_client()
            except Exception as exc:  # noqa: BLE001
                log.exception("dltv-socket-watchdog: restart failed: %s", exc)
        elif age is not None and age > _SOCKET_WATCHDOG_DEAD_AFTER_SEC:
            log.warning(
                "dltv-socket-watchdog: socket thread alive but heartbeat "
                "stale (%.1fs since last tick, threshold=%.1fs) — restarting",
                age, _SOCKET_WATCHDOG_DEAD_AFTER_SEC,
            )
            # The thread is alive but its loop is stuck.  We can't
            # safely kill it from another thread, so we let the
            # OS-level GC eventually clean it up.  The new socket
            # will run on a fresh thread.
            _loop_thread = None
            try:
                start_socket_client()
            except Exception as exc:  # noqa: BLE001
                log.exception("dltv-socket-watchdog: restart failed: %s", exc)
# Throttle: don't honor `force_reconnect()` more than once per N
# seconds.  Reconnect takes ~3-5s (TCP+TLS+SIO CONNECT), so a
# 30s minimum is a safe lower bound — fast enough to keep
# subscriptions fresh, slow enough that a chatty publisher
# doesn't pin us in reconnect loops.
_FORCE_RECONNECT_MIN_SEC = 30.0

# v0.7.48: throttle the board-builder wake-up.  DLTV pushes
# `__nd2_match_*` updates every 1-2s during a live game; the
# publisher's `build_board()` takes ~1-1.5s per call, so firing
# the wake on every payload would pile up builds.  We cap the
# wake cadence to once per `WAKE_BUILD_MIN_GAP_SEC`; the wake
# itself is a single Event.set() so dropping it is free.
_WAKE_BUILD_MIN_GAP_SEC = 1.0
_last_wake_build_ts: float = 0.0
_wake_build_lock = threading.Lock()


def _maybe_wake_build() -> None:
    """Thread-safe wake-up for the board-builder (v0.7.48).

    Called from the socket recv loop after every data-bearing
    payload.  No-ops if we already fired within the last
    `_WAKE_BUILD_MIN_GAP_SEC` so a chatty match doesn't pile
    up builds.  Imports are local so this file remains
    importable even when `stream.py` is mid-startup.
    """
    global _last_wake_build_ts
    now = _now()
    with _wake_build_lock:
        if now - _last_wake_build_ts < _WAKE_BUILD_MIN_GAP_SEC:
            return
        _last_wake_build_ts = now
    try:
        from .stream import wake_build
        wake_build()
    except Exception as exc:
        # Don't let a missing/broken stream module kill the socket loop.
        log.debug("dltv_socket: wake_build failed: %s", exc)


def force_reconnect() -> None:
    """Request the socket loop to drop the current session and reconnect.

    The DLTV server is picky: it treats unsolicited SUBSCRIBE packets
    on an alive connection as junk and closes the socket after a few
    minutes.  So the only way to refresh the subscription set is to
    drop and re-open the WebSocket — which we do on demand from
    `stream.board_publisher_loop` after the live card set changes.
    """
    global _force_reconnect_flag
    _force_reconnect_flag = True


def start_socket_client() -> threading.Thread:
    """Start the WebSocket client in a dedicated background thread.

    Idempotent: returns the existing thread if already started.
    The thread is a daemon and dies with the process.

    v0.4.0.2: also kicks a watchdog thread that re-spawns this one
    if the loop dies.  We observed the socket thread silently going
    to `thread: None` after a long stretch of DLTV DNS failures —
    the inner reconnect loop caught the connection errors but
    something else (we suspect a daemon-thread / atexit race) killed
    the outer `while not _loop_should_stop.is_set()` loop.  The
    watchdog keeps the publisher's data flow alive even when DLTV
    is misbehaving.
    """
    global _loop_thread, _socket_watchdog_thread
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
    # Start the watchdog if it isn't running yet.  The watchdog
    # also uses a daemon thread; it stays alive for the process
    # lifetime and re-spawns the socket loop as needed.
    if _socket_watchdog_thread is None or not _socket_watchdog_thread.is_alive():
        _socket_watchdog_thread = threading.Thread(
            target=_socket_watchdog_loop,
            name="dltv-socket-watchdog",
            daemon=True,
        )
        _socket_watchdog_thread.start()
    return t


def stop_socket_client(timeout: float = 5.0) -> None:
    """Signal the client to stop and wait for the thread to exit."""
    _loop_should_stop.set()
    if _loop_thread is not None:
        _loop_thread.join(timeout=timeout)
