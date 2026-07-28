"""
FastAPI app for the Dota Analyst business service.

This service is **internal only** — the gateway in front of it is the
only thing the outside world talks to. Don't expose :8000 publicly.

Endpoints (called by the gateway, not by browsers):
  GET  /api/leagues            -> available leagues (DLTV events)
  GET  /api/board?events=1,2   -> {prematch, live, postmatch} for selected leagues
  GET  /api/stream/matches     -> Server-Sent Events (live updates)
  GET  /api/healthz            -> liveness
  GET  /api/readyz             -> readiness (DB + cache ping)
  POST /internal/...           -> internal-only routes (gated by gateway HMAC)
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ._logging import get_logger, setup_logging
from .accuracy import accuracy_summary
from .board import build_board, leagues_with_status
from .dltv_client import client
from .exceptions import (
    BoardBuildError,
    DiscoveryError,
    DLTVError,
    HTTPClientError,
    InfraError,
    MLError,
    UpstreamError,
)
from .ml.engine import get_default_engine, reset_default_engine
from .stream import board_publisher_loop, event_stream, get_stream

# Configure logging once on import. `setup_logging` is idempotent.
setup_logging()
log = get_logger(__name__)

# v0.3.24: hide live matches that are synthesised from Steam raw
# data without any DLTV coverage (`_steam_only: True`).  These are
# almost always minor Chinese amateur leagues where the scoreboard
# is empty, team names are TBD, and the user has no way to click
# through to a DLTV page.  Set to "0" to keep them.
LIVE_HIDE_STEAM_ONLY = os.environ.get("LIVE_HIDE_STEAM_ONLY", "1") == "1"

# --------------------------------------------------------------------------- #
# Board cache (v0.3.14+, v0.3.22 cont 4: filter-on-auto-board)
# --------------------------------------------------------------------------- #
# `build_board()` walks DLTV events + the discovery scraper + Steam live
# feed, which on cold cache can take 1-3 minutes (many sequential HTTP
# calls each bounded by a 12s timeout).  We don't want a browser
# request to wait that long.
#
# Strategy:
#   1. SSE publisher calls `build_board()` every 5s with NO filter
#      (auto-includes all active leagues) and writes the result to
#      `_latest_auto_board`.  The board is per-tick; if the publisher
#      is stuck (e.g. upstream dltv.org is slow), the timestamp stops
#      updating.
#   2. `/api/board` reads from `_latest_auto_board` and FILTERS it
#      server-side based on the `events=` query param.  No rebuild is
#      triggered for filtered requests — the response is instant.
#   3. Only if the auto-board is missing or too stale do we fall back
#      to a single-flight `build_board()` (cold-start case).
#
# v0.3.22 cont 4 motivation: earlier we triggered a fresh build for
# every distinct `?events=...` selection.  When the publisher loop is
# slow (chromium greenlet + dltv.org latency), this meant filtered
# requests timed out at 25s and the user saw an empty board even
# though the auto-board was full of relevant cards.  Filtering on
# the already-built auto-board fixes that without changing the data
# path.
# --------------------------------------------------------------------------- #
_BOARD_CACHE_TTL = 30.0   # if a board is older than this, refetch
_board_cache: dict = {}    # cache_key -> (board, ts) — see /api/board
_latest_auto_board: dict = {}   # board produced by SSE publisher (events=[], watch=[])
_latest_auto_board_ts: float = 0.0
# How long we consider the auto-board "fresh enough" to filter on.  If
# the publisher loop is mid-build, the auto-board is still the LAST
# successful build — usually 5-30s old, but can stretch to 60-120s
# when dltv.org is slow (v0.3.22 cont 4: observed 60-90s per build
# when DLTV's HTML scraper hangs at 12s × 30 leagues).  We accept up
# to 5 minutes; beyond that, we'd rather try a fresh build than
# serve cards that no longer match reality (a finished game is
# still a finished game, but a 5-minute-stale live score is
# actively misleading).
_AUTO_BOARD_FILTER_MAX_AGE_SEC = 300.0

# Single-flight for /api/board: when several requests miss the cache at
# once, we want them to share one build_board() call rather than stampede
# the upstream APIs.  We key the in-flight set on cache_key; the Future
# stores the eventual result dict.
#
# Why a dict of Futures and not a lock?  Because the value the requester
# is waiting on is *the board itself* — not just "I may go now".  Using
# a Future means each waiting request gets the SAME dict and only
# `build_board` is called once.
#
# We deliberately cap the in-flight set: a long-stuck build (e.g.
# upstream is hard-down) would otherwise leak a Future forever.
_inflight: dict = {}        # cache_key -> asyncio.Future


async def _build_board_singletrip(
    cache_key: tuple,
    events: List[int],
    watch_ids: List[int],
) -> Dict:
    """Run `build_board` exactly once for `cache_key`, even under
    concurrent misses.

    Returns the freshly built board dict.  Any other request waiting
    on the same key gets the same value.
    """
    import asyncio as _aio
    existing = _inflight.get(cache_key)
    if existing is not None and not existing.done():
        return await existing
    fut: _aio.Future = _aio.get_event_loop().create_future()
    _inflight[cache_key] = fut
    try:
        # build_board is synchronous and walks many upstream HTTP calls.
        # We run it in a thread so the event loop stays free for other
        # endpoints (and for the SSE keepalive).  A second request that
        # arrives while this thread is busy awaits the same Future.
        board = await _aio.to_thread(build_board, events, watch_ids)
        fut.set_result(board)
        return board
    except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
        fut.set_exception(exc)
        raise
    finally:
        # Give other waiters a chance to grab the result before we
        # remove the entry.  `fut.done()` is True here, so any future
        # caller will rebuild — but the cache is hot by then.
        _inflight.pop(cache_key, None)


async def _accuracy_loop(score_fn, interval_sec: float = 60.0) -> None:
    """Background loop: re-score un-scored predictions on a timer.

    Runs the synchronous `score_pending` in a worker thread so a slow
    DLTV lookup doesn't stall the event loop.  We deliberately don't
    fan out to multiple workers — the JSONL log is the source of truth
    and a single writer is the simplest invariant.
    """
    import asyncio as _aio
    try:
        while True:
            try:
                await _aio.to_thread(score_fn)
            except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
                log.warning("accuracy tick failed: %s", exc, exc_info=False)
            await _aio.sleep(interval_sec)
    except _aio.CancelledError:
        log.info("accuracy loop cancelled")
        raise


# --------------------------------------------------------------------------- #
# Lifespan: start the SSE publisher poller alongside the FastAPI app
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: warm the engine + start the SSE poller."""
    # 1. Engine (fail fast on missing model artifacts).
    engine = get_default_engine()
    log.info("prediction engine ready: %s", engine.name)

    # 2. SSE poller — background task that calls build_board() every
    #    few seconds and pushes deltas to subscribers.
    stream = get_stream()
    publisher = asyncio.create_task(
        board_publisher_loop(stream, interval_sec=5.0),
        name="sse-publisher",
    )
    log.info("sse publisher task scheduled")

    # 3. Accuracy scorer — re-scores un-scored predictions every
    #    60 seconds.  Lives in its own task so a slow DLTV lookup
    #    (cold cache, 12s timeout) doesn't starve the SSE poller.
    from .accuracy import score_pending  # local import to avoid cycles
    accuracy_task = asyncio.create_task(
        _accuracy_loop(score_pending, interval_sec=60.0),
        name="accuracy-scorer",
    )
    log.info("accuracy scorer task scheduled")

    # 4. Player.win_rate browser loop — polls dltv.org live match
    #    pages via Playwright (Phase 3) and writes the result to a
    #    JSON cache the predict path can read.
    from .stream import player_wr_browser_loop
    browser_task = asyncio.create_task(
        player_wr_browser_loop(),
        name="player-wr-browser",
    )
    log.info("player_wr browser task scheduled")

    # 5. v0.4.0: direct socket.io client — bypasses chromium and
    #    the DLTV-side /live/{id}.json cache.  Pushes the latest
    #    match-state payload per steam_id to an in-memory state the
    #    board layer reads.  Started in its own thread (see
    #    `dltv_socket.start_socket_client`).
    from . import dltv_socket
    dltv_socket.start_socket_client()
    log.info("dltv socket client started (background thread)")

    try:
        yield
    finally:
        publisher.cancel()
        accuracy_task.cancel()
        browser_task.cancel()
        for t in (publisher, accuracy_task, browser_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        dltv_socket.stop_socket_client(timeout=3.0)
        log.info("sse publisher + accuracy scorer + player_wr browser + dltv socket stopped")


app = FastAPI(title="Dota Analyst — business", version="0.4.0", lifespan=_lifespan)

# CORS: read allowed origins from env. Default to the gateway only.
# The gateway terminates browser-facing CORS; this is just for direct
# service-to-service calls (e.g. local debugging).
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8000").strip()
allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] or ["http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-Id"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------- #
# Startup — pre-build the prediction engine so the first request doesn't pay
# the load cost (and we fail fast if the model artifacts are missing).
# ---------------------------------------------------------------------------- #

# (engine warm-up moved to the `_lifespan` context manager above so
# it runs in the same event loop as the SSE publisher)


# ---------------------------------------------------------------------------- #
# Public-ish read endpoints (called by the gateway; no business logic here)
# ---------------------------------------------------------------------------- #

@app.get("/api/leagues")
def get_leagues():
    """Return leagues with a `match_count` per league.

    v0.3.21+: this endpoint used to call `leagues_with_status()`,
    which itself walks the discovery tracker (Steam + scraper +
    DLTV events).  On a cold cache that took >25s and the request
    504'd.  Now we return whatever the publisher already has:
      - the league list from `client.get_events()` (one cached
        DLTV call, fast),
      - match counts derived from `_latest_auto_board` (no extra
        upstream work — the publisher did it for us).

    Leagues that have zero live/prematch/postmatch cards in the
    current auto-board still appear in the response with
    `match_count: 0` — the UI greys them out so the user knows
    they exist but have nothing to show right now.
    """
    # Pull the static league list from DLTV's cached v1 events.
    try:
        events = client.get_events() or []
    except (DLTVError, HTTPClientError, UpstreamError):
        events = []

    # Per-league match count from the publisher's last board.
    counts: Dict[int, int] = {}
    board = _latest_auto_board if _latest_auto_board else {}
    for col in ("live", "prematch", "postmatch"):
        for card in board.get(col) or []:
            eid = card.get("event_id")
            if eid is None:
                continue
            counts[int(eid)] = counts.get(int(eid), 0) + 1

    leagues = [
        {
            "id": e.get("id"),
            "title": e.get("title"),
            "is_active": bool(e.get("is_active")),
            "match_count": counts.get(int(e["id"]), 0) if e.get("id") else 0,
        }
        for e in events
        if e.get("id")
    ]
    # Sort by activity (descending) so the most relevant leagues
    # appear at the top of the picker.
    leagues.sort(key=lambda L: (-L["match_count"], (L["title"] or "").lower()))
    return {"leagues": leagues}


@app.get("/api/board")
async def get_board(
    events: List[str] = Query([], description="event ids (repeated or comma-separated)"),
    watch: List[str] = Query([], description="steam match ids (repeated or comma-separated; watchlist, bypasses v1 API)"),
):
    """Return the Kanban board for selected events / watchlist.

    v0.3.15 rewrite: the handler is `async` and the synchronous
    `build_board()` runs in a worker thread.  We also add a single-flight
    Future so concurrent misses share one upstream call.  And we
    always serve the publisher's `auto-board` for unfiltered requests,
    even if it's slightly stale — better stale data than a 504.

    v0.3.22 cont 4: also serve the auto-board for FILTERED requests
    by filtering it server-side.  The publisher's auto-board is built
    without a filter (it includes every active league), so applying
    the user's `events=` selection to it is a simple in-memory
    `card.event_id in allowed_set` check per card.  This eliminates
    the 25s "build timed out, empty board" failure mode when the
    publisher is slow (chromium greenlet / dltv.org latency) — the
    user still sees their selected leagues' cards, just from a
    slightly older auto-board snapshot.
    """
    global _latest_auto_board, _latest_auto_board_ts
    ids: List[int] = []
    for group in events:
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    # stable order, dedup
    seen: set = set()
    deduped: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    ids = deduped

    watch_ids: List[int] = []
    seen_w: set = set()
    for group in watch:
        for part in group.split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in seen_w:
                seen_w.add(int(part))
                watch_ids.append(int(part))

    cache_key = (tuple(ids), tuple(watch_ids))
    now = time.monotonic()

    # 1. Hot cache — return instantly.
    cached = _board_cache.get(cache_key)
    if cached and (now - cached[1]) < _BOARD_CACHE_TTL:
        return JSONResponse(cached[0])

    # 2. Auto-board path — instant for both filtered and unfiltered
    #    requests.  The publisher's auto-board is built without a
    #    filter; we apply the user's selection server-side.
    auto = _latest_auto_board
    if auto and (now - _latest_auto_board_ts) < _AUTO_BOARD_FILTER_MAX_AGE_SEC:
        filtered = _filter_auto_board(auto, ids, watch_ids)
        filtered["selected"] = ids
        filtered["watch"] = watch_ids
        # Cache the filtered result so the next request (within TTL)
        # is a pure dict lookup.
        _board_cache[cache_key] = (filtered, now)
        return JSONResponse(filtered)

    # 3. No auto-board (cold start) or auto-board is too stale — fall
    #    through to a single-flight build.  This is the path that
    #    used to handle ALL filtered requests and that's where the
    #    25s timeouts came from.  We now hit this path only on the
    #    very first request after process start, or when the
    #    publisher loop is wedged.
    try:
        board = await asyncio.wait_for(
            _build_board_singletrip(cache_key, ids, watch_ids),
            timeout=25.0,  # under the nginx 30s proxy_read_timeout
        )
    except asyncio.TimeoutError:
        log.warning("/api/board build timed out (key=%s)", cache_key)
        if not ids and not watch_ids and _latest_auto_board:
            return JSONResponse(_latest_auto_board)
        return JSONResponse(
            {
                "prematch": [], "live": [], "postmatch": [],
                "selected": ids, "watch": watch_ids,
                "engine": get_default_engine().name,
                "stale": True,
            },
            status_code=200,
        )
    except (BoardBuildError, MLError, DiscoveryError, UpstreamError, InfraError) as exc:
        log.warning("/api/board build failed: %s", exc, exc_info=True)
        if not ids and not watch_ids and _latest_auto_board:
            return JSONResponse(_latest_auto_board)
        return JSONResponse(
            {
                "prematch": [], "live": [], "postmatch": [],
                "selected": ids, "watch": watch_ids,
                "engine": get_default_engine().name,
                "error": str(exc),
            },
            status_code=200,
        )

    board["selected"] = ids
    board["watch"] = watch_ids
    board["engine"] = get_default_engine().name
    _board_cache[cache_key] = (board, now)
    if not ids and not watch_ids:
        _latest_auto_board = board
        _latest_auto_board_ts = now
    return JSONResponse(board)


def _filter_auto_board(
    auto: Dict[str, Any],
    event_ids: List[int],
    watch_ids: List[int],
) -> Dict[str, Any]:
    """Apply the user's `events=` / `watch=` filter to the auto-board.

    Mirrors the strict filter in `build_board()`:
      - cards with `event_id` in `event_ids` always pass;
      - cards with `event_id is None` (e.g. "Steam league 19479" with
        no DLTV mapping) are dropped when the user has narrowed the
        board to a specific set of leagues; they pass when the
        request is unfiltered (no `event_ids`, no `watch_ids`);
      - watchlist cards (matched by `match_id`) are kept
        regardless of `event_id` — the user explicitly pinned them.

    v0.3.22 cont 4: the entire filter is in-memory; no upstream calls
    are made.  The auto-board is rebuilt every 5s by the publisher
    loop, so the filter result is at most ~5s old.
    """
    allowed = set(int(x) for x in (event_ids or []))
    watch_set = set(int(x) for x in (watch_ids or []))
    has_filter = bool(allowed) or bool(watch_set)
    # If the user has no filter at all, return the auto-board as-is
    # (already a public-shaped dict).
    if not has_filter:
        return dict(auto)

    def _keep(card: Dict[str, Any]) -> bool:
        mid = card.get("match_id")
        eid = card.get("event_id")
        # Watchlist pins: always keep — user explicitly requested.
        if mid is not None and int(mid) in watch_set:
            return True
        # v0.3.24: drop steam-only live cards when the env flag is on.
        # These come from the `_steam_game_to_series` fallback when DLTV
        # has no /live/{id}.json entry — typically minor amateur leagues
        # (Chinese "Steam league 17599" etc.) with empty scoreboards.
        if LIVE_HIDE_STEAM_ONLY and card.get("_steam_only"):
            return False
        # v0.3.25r (re-applied on top of v0.3.25k): drop "TBD vs TBD"
        # live cards.  These come from DLTV `/live/{id}.json` payloads
        # that haven't resolved the team slugs yet (pre-pick phase).
        # The card has real score/time/networth but the UI can't tell
        # the user *which match* it is — pure noise.  The watchlist
        # path above keeps it visible to anyone who pinned it.
        if card.get("stage") == "live":
            rn = (card.get("radiant_team") or {}).get("name")
            dn = (card.get("dire_team") or {}).get("name")
            if (not rn or rn == "TBD") and (not dn or dn == "TBD"):
                return False
        # No event_id (steam-only / unmapped scraper card): drop when
        # the user has narrowed the board.  This matches the strict
        # live filter in build_board() so server-side filtering and
        # a fresh build behave identically.
        if eid is None:
            return False
        return int(eid) in allowed

    return {
        "prematch": [c for c in (auto.get("prematch") or []) if _keep(c)],
        "live":     [c for c in (auto.get("live") or [])     if _keep(c)],
        "postmatch":[c for c in (auto.get("postmatch") or [])if _keep(c)],
        "engine":   auto.get("engine"),
        "filtered_from_auto": True,
    }


# ---------------------------------------------------------------------------- #
# SSE — live updates push (0.1.1)
# ---------------------------------------------------------------------------- #

@app.get("/api/stream/matches")
async def stream_matches() -> StreamingResponse:
    """Server-Sent Events stream of board updates.

    Emits `event: board_update` whenever the publisher loop detects
    a change, and an SSE comment (`: ping`) every 30s to keep the
    connection warm.  The client should `EventSource(...)` against
    this URL; the browser handles auto-reconnect on its own.

    Auth: the gateway already validates `X-API-Key` before the
    request reaches us, so we trust the connection here.  For
    direct curl access during dev, the gateway middleware still
    gates the URL.
    """
    stream = get_stream()
    return StreamingResponse(
        event_stream(stream, keepalive_sec=30.0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable nginx response buffering so events reach the
            # client as soon as the publisher pushes them.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------- #
# Health endpoints
# ---------------------------------------------------------------------------- #

@app.get("/api/healthz")
def healthz():
    """Liveness probe — process is alive."""
    return {"status": "ok"}


@app.get("/api/readyz")
def readyz():
    """Readiness probe — the service can handle traffic.

    For 0.1.0 with JsonFileRepository there is no DB to ping, so we
    just confirm the DLTV client can serve cached hero metadata.
    """
    try:
        client.get_heroes()  # populates hero index from cache
        return {"status": "ready"}
    except (DLTVError, HTTPClientError, UpstreamError) as exc:
        # Catches anything from the upstream API stack. We deliberately
        # do NOT catch `Exception` here — a coding bug in the readiness
        # path should crash the process so the orchestrator restarts it.
        log.warning("readiness check failed: %s", exc, exc_info=True)
        return JSONResponse({"status": "not-ready", "error": str(exc)}, status_code=503)


# ---------------------------------------------------------------------------- #
# Accuracy tracking (v0.3.15+) — see business/accuracy.py for the model.
# ---------------------------------------------------------------------------- #

@app.get("/api/accuracy")
def get_accuracy():
    """Return live accuracy stats.

    Reads the JSONL log, scores any un-scored predictions that have
    since completed, and returns aggregate stats.  The response is
    small (a few hundred bytes), so we don't bother caching.
    """
    try:
        return JSONResponse(accuracy_summary())
    except (DLTVError, HTTPClientError, UpstreamError) as exc:
        # The accuracy log is best-effort — even a partial read is
        # better than a 500.
        log.warning("accuracy summary failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc), "scored": 0, "accuracy": None}, status_code=200)
