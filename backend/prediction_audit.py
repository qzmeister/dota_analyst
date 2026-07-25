"""Durable audit trail for predictions actually shown on the live board."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = ROOT / "ml_data" / "prediction_audit.json"
MAX_RECORDS = 5000


class PredictionAudit:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _read() -> List[Dict]:
        try:
            data = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def _write(records: List[Dict]) -> None:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = AUDIT_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(records[-MAX_RECORDS:], ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(AUDIT_PATH)

    def record_live(self, match_id: int | None, payload: Dict) -> None:
        prediction = (payload.get("prediction") or {}).get("winner") or {}
        team = prediction.get("team")
        probability = prediction.get("probability")
        if not match_id or not team or not isinstance(probability, (int, float)):
            return
        with self._lock:
            records = self._read()
            key = str(match_id)
            existing = next((record for record in records if str(record.get("match_id")) == key), None)
            observation = {
                "match_id": match_id,
                "shown_at": time.time(),
                "event": payload.get("event"),
                "radiant_team": (payload.get("radiant_team") or {}).get("name"),
                "dire_team": (payload.get("dire_team") or {}).get("name"),
                "predicted_team": team,
                "probability": float(probability) / 100.0,
                "source": ((payload.get("prediction") or {}).get("ml_winner") or {}).get("source", "heuristic"),
            }
            if existing is None:
                records.append(observation)
            elif not existing.get("actual_winner"):
                existing.update(observation)
            self._write(records)

    def settle(self, match_id: int | None, actual_winner: Optional[str]) -> None:
        if not match_id or not actual_winner:
            return
        with self._lock:
            records = self._read()
            for record in records:
                if str(record.get("match_id")) == str(match_id):
                    record["actual_winner"] = actual_winner
                    record["settled_at"] = time.time()
                    record["correct"] = record.get("predicted_team") == actual_winner
                    break
            else:
                return
            self._write(records)

    def summary(self) -> Dict:
        with self._lock:
            records = self._read()
        settled = [record for record in records if isinstance(record.get("correct"), bool)]
        if not settled:
            return {"shown": len(records), "settled": 0, "accuracy": None, "brier_score": None, "calibration": []}
        calibration = []
        for lower in range(50, 100, 10):
            upper = lower + 10
            bucket = [record for record in settled if lower <= record["probability"] * 100 < upper]
            if bucket:
                calibration.append({
                    "range": f"{lower}–{upper - 1}%",
                    "samples": len(bucket),
                    "predicted": round(sum(record["probability"] for record in bucket) / len(bucket), 3),
                    "actual": round(sum(1.0 if record["correct"] else 0.0 for record in bucket) / len(bucket), 3),
                })
        accuracy = sum(1.0 if record["correct"] else 0.0 for record in settled) / len(settled)
        brier = sum((record["probability"] - (1.0 if record["correct"] else 0.0)) ** 2 for record in settled) / len(settled)
        return {
            "shown": len(records),
            "settled": len(settled),
            "accuracy": accuracy,
            "brier_score": brier,
            "calibration": calibration,
        }


prediction_audit = PredictionAudit()
