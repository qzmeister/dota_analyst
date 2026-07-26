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
from typing import List

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

# --------------------------------------------------------------------------- #
# Board cache (v0.3.14+)
# --------------------------------------------------------------------------- #
# `build_board()` walks DLTV events + the discovery scraper + Steam live
# feed, which on cold cache can take 1-3 minutes (many sequential HTTP
# calls each bounded by a 12s timeout).  We don't want a browser
# request to wait that long.
#
# Strategy: SSE publisher calls `build_board()` every 5s and writes the
# result here.  `/api/board` reads from this cache — if it's fresh, the
# response is instant.  If it's stale (publisher loop is mid-build or
# the process just started), we still trigger a fresh build but the
# browser waits.  Subsequent requests always get the cached version.
#
# The publisher is the single owner of the cache; multiple workers
# would race.  We use a simple dict + monotonic timestamp; a lock is
# not needed because the writes happen from one coroutine and the
# reads are tolerant of a 1-tick stale value.
# --------------------------------------------------------------------------- #
_BOARD_CACHE_TTL = 30.0   # if a board is older than this, refetch
_board_cache: dict = {}    # cache_key -> (board, ts) — see /api/board
_latest_auto_board: dict = {}   # board produced by SSE publisher (events=[], watch=[])
_latest_auto_board_ts: float = 0.0

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

    try:
        yield
    finally:
        publisher.cancel()
        accuracy_task.cancel()
        for t in (publisher, accuracy_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        log.info("sse publisher + accuracy scorer stopped")


app = FastAPI(title="Dota Analyst — business", version="0.3.16", lifespan=_lifespan)

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
    """Return leagues with a status tag (live | upcoming | finished)."""
    return {"leagues": leagues_with_status()}


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

    # 2. Unfiltered request — always serve the publisher's auto-board
    #    if we have one, even if it's slightly older than _BOARD_CACHE_TTL.
    #    Stale data is better than a 504; the next publisher tick will
    #    refresh it within 5 seconds.
    if not ids and not watch_ids and _latest_auto_board:
        return JSONResponse(_latest_auto_board)

    # 3. Cold path — single-flight build in a thread.
    try:
        board = await asyncio.wait_for(
            _build_board_singletrip(cache_key, ids, watch_ids),
            timeout=25.0,  # under the nginx 30s proxy_read_timeout
        )
    except asyncio.TimeoutError:
        # Build hung — fall back to whatever we have.  For an
        # unfiltered request without any auto-board yet, return an
        # empty board so the UI renders "no matches" instead of
        # nginx returning 504.
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
        # Last-resort fallback: empty board so the UI shows "no matches"
        # rather than a 500.
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
