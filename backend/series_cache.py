"""Persistent short-term cache of completed maps in an active series."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List


PATH = Path(__file__).resolve().parent.parent / "ml_data" / "active_series_maps.json"


def _key(event: str, first: str, second: str) -> str:
    normalise = lambda value: re.sub(r"[^a-z0-9]+", "", (value or "").lower())
    return "|".join((normalise(event), *sorted((normalise(first), normalise(second)))))


class SeriesMapCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _read() -> Dict:
        try:
            value = json.loads(PATH.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write(value: Dict) -> None:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(PATH)

    def merge(self, event: str, first: str, second: str, maps: List[Dict]) -> List[Dict]:
        """Merge fresh completed maps with the saved maps and return their union."""
        key = _key(event, first, second)
        with self._lock:
            payload = self._read()
            previous = payload.get(key, {}).get("maps", [])
            merged: Dict[str, Dict] = {}
            for index, game_map in enumerate(previous + maps, start=1):
                identity = str(game_map.get("map_id") or f"game:{game_map.get('game', index)}")
                merged[identity] = game_map
            result = sorted(merged.values(), key=lambda game_map: game_map.get("game", 0))
            if result:
                payload[key] = {"updated_at": time.time(), "maps": result}
                self._write(payload)
            return result

    def clear(self, event: str, first: str, second: str) -> None:
        key = _key(event, first, second)
        with self._lock:
            payload = self._read()
            if key in payload:
                payload.pop(key, None)
                self._write(payload)


series_map_cache = SeriesMapCache()
