"""Persist analysed maps and continuously add completed maps to ML training data."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable

from .datdota_client import get_match_details


ROOT = Path(__file__).resolve().parent.parent
FULL_MATCHES_DIR = ROOT / "ml_data" / "full_matches"
OBSERVED_DIR = ROOT / "ml_data" / "observed_live"
STATUS_PATH = ROOT / "ml_data" / "continuous_learning_status.json"
TRAINER = ROOT / "scripts" / "train_temporal_prematch.py"
RETRAIN_AFTER_NEW_MAPS = 5


class LearningPipeline:
    """Background collector; failures never interrupt board rendering."""

    def __init__(self) -> None:
        self._pending: set[int] = set()
        self._queue: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()
        self._new_since_train = 0
        self._worker = threading.Thread(target=self._run, name="ml-map-collector", daemon=True)
        self._worker.start()

    @staticmethod
    def _write_json(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def observe_live_map(self, match_id: int | None, payload: Dict) -> None:
        """Save the exact draft/prediction that was shown during a live map."""
        if not match_id:
            return
        self._write_json(OBSERVED_DIR / f"{match_id}.json", {"observed_at": time.time(), **payload})

    def queue_completed_maps(self, maps: Iterable[Dict]) -> None:
        """Queue completed maps that do not yet have a full DatDota payload."""
        for game_map in maps:
            match_id = game_map.get("steam_id")
            if not isinstance(match_id, int) or not game_map.get("winner"):
                continue
            if (FULL_MATCHES_DIR / f"{match_id}.json").exists():
                continue
            with self._lock:
                if match_id in self._pending:
                    continue
                self._pending.add(match_id)
            self._queue.put(match_id)

    def _save_status(self, last_match_id: int | None = None, error: str | None = None) -> None:
        self._write_json(STATUS_PATH, {
            "last_match_id": last_match_id,
            "last_error": error,
            "queued_maps": self._queue.qsize(),
            "new_maps_since_train": self._new_since_train,
            "updated_at": time.time(),
        })

    def _train(self) -> None:
        try:
            subprocess.run([sys.executable, str(TRAINER)], cwd=ROOT, check=True, timeout=180)
            self._new_since_train = 0
        except Exception as exc:
            self._save_status(error=f"training: {exc}")

    def _run(self) -> None:
        while True:
            match_id = self._queue.get()
            try:
                payload = get_match_details(match_id)
                details = (payload or {}).get("data")
                if not details:
                    self._save_status(match_id, "DatDota returned no match details")
                    continue
                self._write_json(FULL_MATCHES_DIR / f"{match_id}.json", details)
                self._new_since_train += 1
                self._save_status(match_id)
                if self._new_since_train >= RETRAIN_AFTER_NEW_MAPS:
                    self._train()
            except Exception as exc:
                self._save_status(match_id, str(exc))
            finally:
                with self._lock:
                    self._pending.discard(match_id)
                self._queue.task_done()
                time.sleep(3.0)  # DatDota rate limit


learning_pipeline = LearningPipeline()
