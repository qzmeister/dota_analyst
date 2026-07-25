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
from .board import build_board, leagues_with_status
from .dltv_client import client
from .exceptions import DLTVError, HTTPClientError, UpstreamError
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

    try:
        yield
    finally:
        publisher.cancel()
        try:
            await publisher
        except asyncio.CancelledError:
            pass
        log.info("sse publisher task stopped")


app = FastAPI(title="Dota Analyst — business", version="0.3.14", lifespan=_lifespan)

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
def get_board(
    events: List[str] = Query([], description="event ids (repeated or comma-separated)"),
    watch: List[str] = Query([], description="steam match ids (repeated or comma-separated; watchlist, bypasses v1 API)"),
):
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

    # Stale-while-revalidate cache: serialise (ids, watch_ids) as the key.
    cache_key = (tuple(ids), tuple(watch_ids))
    now = time.monotonic()
    cached = _board_cache.get(cache_key)
    if cached and (now - cached[1]) < _BOARD_CACHE_TTL:
        return JSONResponse(cached[0])

    # If the SSE publisher has a fresh auto-board (events=[]) and
    # this is a "show everything" request, reuse it to avoid a
    # potentially 1-3 minute cold-cache rebuild.
    if not ids and not watch_ids and _latest_auto_board:
        age = now - _latest_auto_board_ts
        if age < _BOARD_CACHE_TTL:
            return JSONResponse(_latest_auto_board)

    # Stale or missing — compute a fresh board.  The SSE publisher
    # updates _latest_auto_board every 5s; an unfiltered request that
    # finds that cache fresh returns immediately (see above).  If we
    # reach this branch it's a filtered request (specific events or
    # watchlist) that the auto-board doesn't cover; build it inline.
    # If that turns out to be slow, future work would dispatch to
    # `asyncio.to_thread` here too.
    engine = get_default_engine()
    board = build_board(ids, watch_ids=watch_ids)
    board["selected"] = ids
    board["watch"] = watch_ids
    # Surface which engine produced the predictions — useful for debugging
    # the ml vs heuristic switch from the browser / curl.
    board["engine"] = engine.name
    _board_cache[cache_key] = (board, now)
    if not ids and not watch_ids:
        # Update the auto-board so other in-flight requests can reuse it.
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
