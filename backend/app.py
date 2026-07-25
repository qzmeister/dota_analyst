"""
FastAPI app for the Dota Analyst MVP.

Endpoints:
  GET /api/leagues            -> available leagues (DLTV events)
  GET /api/board?events=1,2   -> {prematch, live, postmatch} for selected leagues
  GET /                       -> Kanban board UI (static frontend)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .board import build_board, leagues_with_status
from .dltv_client import client
from .prediction_audit import prediction_audit

app = FastAPI(title="Dota Analyst", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
ROOT_DIR = Path(FRONTEND_DIR).parent


@app.get("/api/leagues")
def get_leagues():
    """Return leagues with a status tag (live | upcoming | finished).

    Includes finished leagues too (so users can review past events),
    sorted: live first, upcoming second, finished last.
    """
    return {"leagues": leagues_with_status()}


@app.get("/api/model-status")
def get_model_status():
    """Return model quality plus historical collection status for the UI."""
    def read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    metadata = read_json(ROOT_DIR / "ml_models" / "model_metadata_prematch.json")
    learning = read_json(ROOT_DIR / "ml_data" / "continuous_learning_status.json")
    manifest = read_json(ROOT_DIR / "ml_data" / "target_tournaments_matches.json")
    collection_progress = read_json(ROOT_DIR / "ml_data" / "target_tournaments_progress.json")
    post_collection_training = read_json(ROOT_DIR / "ml_data" / "post_collection_training_status.json")
    observed = ROOT_DIR / "ml_data" / "observed_live"
    return {
        "model": {
            "samples": metadata.get("n_samples", 0),
            "accuracy": metadata.get("chronological_holdout_accuracy"),
            "roc_auc": metadata.get("chronological_holdout_roc_auc"),
            "training_end_timestamp": metadata.get("training_end_timestamp"),
        },
        "learning": {**learning, "observed_maps": len(list(observed.glob("*.json"))) if observed.exists() else 0},
        "collection": {
            "total": len(manifest) if isinstance(manifest, list) else 0,
            "fetched": len(collection_progress.get("fetched_ids", [])),
            "failed": len(collection_progress.get("failed_ids", [])),
            "state": post_collection_training.get("state"),
            "remaining": post_collection_training.get("remaining"),
        },
    }


@app.get("/api/analytics")
def get_analytics():
    """Return audit metrics for predictions that were visible on the board."""
    return {"prediction_audit": prediction_audit.summary()}


@app.get("/api/data-quality")
def get_data_quality():
    """Compact data-health panel: collection coverage and records ready for ML."""
    full_matches = ROOT_DIR / "ml_data" / "full_matches"
    observed = ROOT_DIR / "ml_data" / "observed_live"
    target_progress_path = ROOT_DIR / "ml_data" / "target_tournaments_progress.json"
    try:
        progress = json.loads(target_progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        progress = {}
    return {
        "full_maps": len(list(full_matches.glob("*.json"))) if full_matches.exists() else 0,
        "observed_live_maps": len(list(observed.glob("*.json"))) if observed.exists() else 0,
        "target_fetched": len(progress.get("fetched_ids", [])),
        "target_failed": len(progress.get("failed_ids", [])),
        "audited_predictions": prediction_audit.summary().get("settled", 0),
    }


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
