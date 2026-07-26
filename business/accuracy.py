"""
Live accuracy tracking (v0.3.15+).

Persists every winner prediction we emit for a live match, then scores
each one when the match completes.  The result is an evolving accuracy
metric the user can watch to know whether the model is actually
delivering value (vs. just feeling right in offline tests).

Files (under `ml_data/`, mounted from host into the container):
  live_predictions.jsonl     — append-only log of every prediction
  live_accuracy.json          — current aggregate stats, recomputed on demand

Why a JSONL log and not "append to a list in one big JSON"?  Because
the file is written on every prediction and re-scored on a timer.
Append-only is the only safe write pattern that survives concurrent
publishers or a crash mid-write.

The accuracy loop runs on a single background task that:
  1. Loads pending (un-scored) predictions.
  2. For each, asks the DLTV v1 series endpoint for the actual
     winner (only available after the match's `ended_at` is set).
  3. Records a verdict (correct / wrong / push).
  4. Updates the aggregate.

This is intentionally simple — no DB, no migration, no fancy feature
group.  Just one JSONL log + one cache file.  When accuracy tracking
becomes a product surface we'll promote this to a proper store.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._logging import get_logger
from .exceptions import AccuracyError
from .dltv_client import client

log = get_logger(__name__)

# Default location, overridable by env for tests
ML_DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "ml_data"))
PREDICTIONS_FILE = ML_DATA_DIR / "live_predictions.jsonl"
ACCURACY_CACHE_FILE = ML_DATA_DIR / "live_accuracy.json"

# Predictions are auto-pruned after this many days — we don't need
# forever-old predictions piling up.
PREDICTION_TTL_DAYS = 30


def _ensure_dir() -> None:
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a single JSON object as one line.  Thread-safe via flock."""
    _ensure_dir()
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    # We use a process-level lock to keep two writers from interleaving
    # bytes on the same file.  A real DB would do better; for a JSONL
    # log this is enough on a single-process service.
    if not hasattr(_append_jsonl, "_lock"):
        _append_jsonl._lock = threading.Lock()  # type: ignore[attr-defined]
    with _append_jsonl._lock:  # type: ignore[attr-defined]
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)


# In-memory dedup: we only record a prediction once per
# (match_id, game_no) tuple.  The board rebuilds every 5s — without
# this we'd log 12 identical rows/min/match.  Loaded once from the
# existing log; kept consistent across calls in a single process.
_dedup_seen: set = set()
_dedup_loaded = False


def _ensure_dedup_loaded() -> None:
    global _dedup_loaded
    if _dedup_loaded:
        return
    for r in _read_jsonl(PREDICTIONS_FILE):
        # Use (match_id, game_no) as the dedup key.  game_no comes
        # from the engine's "current game of the series" counter; for
        # series that aren't best-of-N it's just 1.
        extra = r.get("extra") or {}
        key = (r.get("match_id"), extra.get("game_no"))
        if key != (None, None):
            _dedup_seen.add(key)
    _dedup_loaded = True


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError as exc:
                # Don't let one bad line take the whole log down.
                log.warning("accuracy: bad line in %s: %s", path, exc)
    return out


def _write_jsonl_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Rewrite the file atomically (write to temp + os.replace)."""
    _ensure_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def record_prediction(
    *,
    match_id: Optional[int],
    series_id: Optional[int],
    predicted_winner: str,
    predicted_probability: Optional[float],
    engine: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a winner prediction to the live log.

    `match_id` is the Steam match id; `series_id` is the DLTV series id
    (we try to set both; whichever is present is enough to look the
    match up later).  Returns the record that was written.
    """
    if not match_id and not series_id:
        raise AccuracyError("record_prediction needs at least one of match_id/series_id")
    _ensure_dedup_loaded()
    game_no = (extra or {}).get("game_no") if extra else None
    dedup_key = (int(match_id) if match_id else None, game_no)
    if dedup_key in _dedup_seen:
        # Already recorded for this (match, game).  Skip — the board
        # rebuilds every 5s and would otherwise log 12 rows/min/match.
        return {"_skipped": "already_recorded", "match_id": match_id, "game_no": game_no}
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "match_id": int(match_id) if match_id else None,
        "series_id": int(series_id) if series_id else None,
        "predicted_winner": predicted_winner,
        "predicted_probability": predicted_probability,
        "engine": engine,
        "scored": None,        # None / "correct" / "wrong" / "push"
        "actual_winner": None,
        "actual_ended_at": None,
    }
    if extra:
        rec.update(extra)
    _append_jsonl(PREDICTIONS_FILE, rec)
    _dedup_seen.add(dedup_key)
    return rec


def load_predictions() -> List[Dict[str, Any]]:
    """Return the full prediction log (newest last)."""
    return _read_jsonl(PREDICTIONS_FILE)


def _match_actual_winner(series_id: int) -> Optional[Dict[str, Any]]:
    """Look up the actual winner of a DLTV series, or None if not finished yet.

    Returns a dict {"winner": "teamA|teamB|draw", "ended_at": iso} on success.
    """
    try:
        series_list = client.get_series(series_id) or []
    except Exception as exc:
        log.debug("accuracy: get_series(%s) failed: %s", series_id, exc)
        return None
    if not series_list:
        return None
    s = series_list[0]
    ended_at = s.get("ended_at")
    first_id = s.get("first_team_id")
    second_id = s.get("second_team_id")
    if not (ended_at and first_id and second_id):
        return None
    # Score by series wins
    score = {first_id: 0, second_id: 0}
    for m in (s.get("maps") or []):
        if not (m.get("duration") and m.get("winner")):
            continue
        wid = m.get("radiant_team_id") if m.get("winner") == "radiant" else m.get("dire_team_id")
        if wid in score:
            score[wid] += 1
    if score[first_id] == score[second_id]:
        return {"winner": "draw", "ended_at": ended_at}
    winner_id = first_id if score[first_id] > score[second_id] else second_id
    return {"winner": winner_id, "ended_at": ended_at}


def score_pending() -> Dict[str, Any]:
    """Scan un-scored predictions, look up the actual winner, update log.

    Returns aggregate stats.  Caller (e.g. the accuracy loop) decides
    how often to call this.
    """
    rows = load_predictions()
    if not rows:
        return _empty_stats()

    # Idempotency: only score rows that haven't been scored yet.
    changed = False
    for r in rows:
        if r.get("scored") is not None:
            continue
        sid = r.get("series_id")
        if not sid:
            # We can't score without a series_id — leave as None
            # and add a marker so the next call skips it cheaply.
            r["scored"] = "no_series"
            changed = True
            continue
        info = _match_actual_winner(int(sid))
        if info is None:
            # match not finished yet — leave for next tick
            continue
        r["actual_winner"] = info["winner"]
        r["actual_ended_at"] = info["ended_at"]
        # Compare predicted team name to the actual team id
        # The ML engine emits "team_a" / "team_b" labels, so we map
        # the actual winner_id back to that label by looking at the
        # recorded extra payload if present.
        r["scored"] = _compare(r, info)
        changed = True

    if changed:
        _write_jsonl_atomic(PREDICTIONS_FILE, rows)
        # Prune rows older than PREDICTION_TTL_DAYS
        _prune_old(rows)

    return compute_stats(rows)


def _compare(rec: Dict[str, Any], info: Dict[str, Any]) -> str:
    """Return 'correct' | 'wrong' | 'push'.

    The ML engine stores predicted_winner as a team name (e.g. "Direborn")
    in `record_prediction`.  The actual winner is a team_id in `info`,
    so we need a side-channel mapping.  We stash the team A/B labels
    in the `extra` payload at record time and use them here.
    """
    if info["winner"] == "draw":
        return "push"
    extra = rec.get("extra") or {}
    team_a_id = extra.get("team_a_id")
    team_b_id = extra.get("team_b_id")
    if not (team_a_id and team_b_id):
        return "no_teams"
    winner_side = "team_a" if int(info["winner"]) == int(team_a_id) else "team_b"
    pred = (rec.get("predicted_winner") or "").lower()
    if pred == winner_side:
        return "correct"
    if pred in ("team_a", "team_b"):
        return "wrong"
    # Predicted by team name (heuristic engine) — best-effort match
    team_a_name = (extra.get("team_a_name") or "").lower()
    team_b_name = (extra.get("team_b_name") or "").lower()
    if pred == (team_a_name if winner_side == "team_a" else team_b_name).lower():
        return "correct"
    return "wrong"


def _prune_old(rows: List[Dict[str, Any]]) -> None:
    """Drop rows older than PREDICTION_TTL_DAYS (we keep the cache fresh)."""
    cutoff = time.time() - PREDICTION_TTL_DAYS * 86400
    kept: List[Dict[str, Any]] = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["ts"]).timestamp()
        except (KeyError, ValueError, TypeError):
            kept.append(r)
            continue
        if ts >= cutoff:
            kept.append(r)
    if len(kept) != len(rows):
        _write_jsonl_atomic(PREDICTIONS_FILE, kept)


def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate accuracy stats from the prediction log."""
    scored = [r for r in rows if r.get("scored") in ("correct", "wrong", "push")]
    correct = sum(1 for r in scored if r["scored"] == "correct")
    wrong = sum(1 for r in scored if r["scored"] == "wrong")
    push = sum(1 for r in scored if r["scored"] == "push")
    pending = sum(1 for r in rows if r.get("scored") is None)
    no_series = sum(1 for r in rows if r.get("scored") == "no_series")
    no_teams = sum(1 for r in rows if r.get("scored") == "no_teams")
    total = len(rows)
    decided = correct + wrong
    accuracy = (correct / decided) if decided else None
    # Last 24h rolling window
    now = time.time()
    last24 = [r for r in scored if (ts := _try_ts(r)) and (now - ts) < 86400]
    last24_correct = sum(1 for r in last24 if r["scored"] == "correct")
    last24_wrong = sum(1 for r in last24 if r["scored"] == "wrong")
    last24_decided = last24_correct + last24_wrong
    last24_acc = (last24_correct / last24_decided) if last24_decided else None
    return {
        "total": total,
        "scored": len(scored),
        "pending": pending,
        "no_series": no_series,
        "no_teams": no_teams,
        "correct": correct,
        "wrong": wrong,
        "push": push,
        "accuracy": accuracy,
        "last24h": {
            "scored": len(last24),
            "accuracy": last24_acc,
        },
        "engine_breakdown": _engine_breakdown(scored),
    }


def _try_ts(rec: Dict[str, Any]) -> Optional[float]:
    try:
        return datetime.fromisoformat(rec["ts"]).timestamp()
    except (KeyError, ValueError, TypeError):
        return None


def _engine_breakdown(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_engine: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        eng = r.get("engine") or "unknown"
        by_engine.setdefault(eng, []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for eng, items in by_engine.items():
        c = sum(1 for r in items if r["scored"] == "correct")
        w = sum(1 for r in items if r["scored"] == "wrong")
        decided = c + w
        out[eng] = {
            "scored": len(items),
            "correct": c,
            "wrong": w,
            "accuracy": (c / decided) if decided else None,
        }
    return out


def _empty_stats() -> Dict[str, Any]:
    return {
        "total": 0, "scored": 0, "pending": 0,
        "no_series": 0, "no_teams": 0,
        "correct": 0, "wrong": 0, "push": 0,
        "accuracy": None,
        "last24h": {"scored": 0, "accuracy": None},
        "engine_breakdown": {},
    }


def accuracy_summary() -> Dict[str, Any]:
    """Return current accuracy stats, refreshing pending verdicts first."""
    return score_pending()
