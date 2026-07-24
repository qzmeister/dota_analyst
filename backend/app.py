"""
FastAPI app for the Dota Analyst MVP.

Endpoints:
  GET /api/leagues            -> available leagues (DLTV events)
  GET /api/board?events=1,2   -> {prematch, live, postmatch} for selected leagues
  GET /                       -> Kanban board UI (static frontend)
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .board import build_board, leagues_with_status
from .dltv_client import client

app = FastAPI(title="Dota Analyst", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/api/leagues")
def get_leagues():
    """Return leagues with a status tag (live | upcoming | finished).

    Includes finished leagues too (so users can review past events),
    sorted: live first, upcoming second, finished last.
    """
    return {"leagues": leagues_with_status()}


@app.get("/api/board")
def get_board(
    events: List[str] = Query([], description="event ids (repeated or comma-separated)"),
    watch: List[str] = Query([], description="steam match ids (repeated or comma-separated; watchlist, bypasses v1 API)"),
):
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
    # Always build board — even with empty events/watch, the discovery
    # scraper (dltv.org/matches) auto-populates prematch + live.
    board = build_board(ids, watch_ids=watch_ids)
    board["selected"] = ids
    board["watch"] = watch_ids
    return JSONResponse(board)


# ---- static frontend ---- #
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
