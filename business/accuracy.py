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
        # Use (match_id, game_no) as the dedup key for live rows.
        # Snapshot rows use (match_id, "snapshot_3_5min") as their
        # own bucket — without loading those here, a process
        # restart would happily re-write the snapshot on the next
        # rebuild in the 3-5 min window.  We load both shapes.
        extra = r.get("extra") or {}
        match_id_val = r.get("match_id")
        if match_id_val is None:
            match_id_val = r.get("synthetic_match_id")
        if r.get("is_snapshot"):
            _dedup_seen.add((match_id_val, "snapshot_3_5min"))
        else:
            key = (match_id_val, extra.get("game_no"))
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
    game_time: Optional[int] = None,
    predicted_kills: Optional[float] = None,
    predicted_duration: Optional[float] = None,
    predicted_first_15: Optional[float] = None,
    game_state: Optional[str] = None,
    snapshot: bool = False,
) -> Dict[str, Any]:
    """Append a winner prediction to the live log.

    `match_id` is the Steam match id; `series_id` is the DLTV series id
    (we try to set both; whichever is present is enough to look the
    match up later).  Returns the record that was written.

    v0.4.0.3: extended with live tracking.  Pass `game_time`,
    `predicted_kills`, `predicted_duration`, `predicted_first_15`,
    `game_state` to record a per-tick prediction (not just winner).
    Pass `snapshot=True` to mark a frozen row that the 3-5 min
    "frozen for post-match comparison" logic uses; snapshot rows
    have their own dedup key and are scored separately.
    """
    if not match_id and not series_id:
        raise AccuracyError("record_prediction needs at least one of match_id/series_id")
    # v0.3.17+: series_id may be a synthetic string like
    # "steam-8914601438" or "watch-8914601438" when the live match
    # didn't come through the DLTV v1 series list (e.g. Steam-only
    # match or watchlist pin).  We can't look those up later, but
    # the row is still useful for stats; just stash the synthetic
    # id as a string and skip the int conversion.  Scoring code
    # treats non-int series_id as "never scoreable" (it will mark
    # the row `no_series` and move on).
    try:
        series_id_int = int(series_id) if series_id is not None else None
    except (TypeError, ValueError):
        series_id_int = None
    try:
        match_id_int = int(match_id) if match_id is not None else None
    except (TypeError, ValueError):
        match_id_int = None
    # We accept non-numeric ids — they just become un-scoreable
    # rows in the log.  The "at least one provided" check above is
    # the only mandatory guard.
    _ensure_dedup_loaded()
    game_no = (extra or {}).get("game_no") if extra else None
    # v0.3.17+: use the raw match_id string for dedup when the
    # int form is missing.  Steam-only / watchlist rows would
    # otherwise all collapse to the (None, game_no) key and the
    # second synthetic match would silently overwrite the first.
    dedup_match_key = match_id_int if match_id_int is not None else (
        str(match_id) if match_id is not None else None
    )
    # v0.4.0.3: dedup key shape depends on whether this is a
    # snapshot row or a live update row.
    #   * snapshot=True       → (match_id, "snapshot_3_5min")
    #     one frozen row per match (the 3-5 min window).  Multiple
    #     board rebuilds in that window coalesce.
    #   * live update (default) → (match_id, game_no, game_time_bucket)
    #     where bucket = game_time // 30.  A 30-min match would
    #     produce up to 60 rows, not the 360 it would naively log
    #     at 5s cadence.  This still gives us a 2-row-per-minute
    #     trace of how the prediction evolved.
    if snapshot:
        dedup_key = (dedup_match_key, "snapshot_3_5min")
    else:
        gt = int(game_time) if game_time is not None else 0
        bucket = gt // 30
        dedup_key = (dedup_match_key, game_no, bucket)
    if dedup_key in _dedup_seen:
        # Already recorded for this bucket.  Skip — the board
        # rebuilds every 5s and would otherwise log 12 rows/min/match.
        return {"_skipped": "already_recorded", "match_id": match_id,
                "game_no": game_no, "snapshot": snapshot, "game_time": game_time}
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "match_id": match_id_int,
        "series_id": series_id_int,
        "predicted_winner": predicted_winner,
        "predicted_probability": predicted_probability,
        "engine": engine,
        "scored": None,        # None / "correct" / "wrong" / "push"
        "actual_winner": None,
        "actual_ended_at": None,
    }
    if extra:
        rec.update(extra)
    # v0.4.0.3: live tracking fields.  All optional, so existing
    # callers that don't pass them get rows without these fields
    # (backwards compat).
    if game_time is not None:
        rec["game_time"] = int(game_time)
    if predicted_kills is not None:
        rec["predicted_kills"] = float(predicted_kills)
    if predicted_duration is not None:
        rec["predicted_duration"] = float(predicted_duration)
    if predicted_first_15 is not None:
        rec["predicted_first_15"] = float(predicted_first_15)
    if game_state is not None:
        rec["game_state"] = str(game_state)
    if snapshot:
        rec["is_snapshot"] = True
        rec["scored_snapshot"] = None  # separate verdict for snapshot rows
    # v0.3.17+: stash the raw synthetic id so the row is still
    # traceable when series_id is non-numeric.
    if series_id is not None and series_id_int is None:
        rec["synthetic_id"] = str(series_id)
    if match_id is not None and match_id_int is None:
        rec["synthetic_match_id"] = str(match_id)
    _append_jsonl(PREDICTIONS_FILE, rec)
    _dedup_seen.add(dedup_key)
    return rec


def load_predictions() -> List[Dict[str, Any]]:
    """Return the full prediction log (newest last)."""
    return _read_jsonl(PREDICTIONS_FILE)


def _match_actual_winner(series_id: int, game_no: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Look up the actual winner + per-game stats of a DLTV series, or None.

    v0.4.0.3: now also returns per-game `duration_sec`, `kills_total`,
    `first_15_kills` for the snapshot scoring path.  When `game_no`
    is provided we look at that specific map; otherwise we return
    series-level aggregates.

    Returns a dict {"winner": "teamA|teamB|draw", "ended_at": iso,
    "duration_sec": int|None, "kills_total": int|None,
    "first_15_kills": int|None, "map_winner": "radiant"|"dire"|None}
    on success.
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
    maps = s.get("maps") or []
    for m in maps:
        if not (m.get("duration") and m.get("winner")):
            continue
        wid = m.get("radiant_team_id") if m.get("winner") == "radiant" else m.get("dire_team_id")
        if wid in score:
            score[wid] += 1
    # Series-level outcome
    if score[first_id] == score[second_id]:
        winner_id: Any = "draw"
    else:
        winner_id = first_id if score[first_id] > score[second_id] else second_id
    out: Dict[str, Any] = {"winner": winner_id, "ended_at": ended_at}
    # v0.4.0.3: per-game stats.  Walk the maps in order; the
    # `game_no` recorded at prediction time is 1-based.  We pair
    # `game_no - 1` with the Nth played map (i.e. a map that has
    # both duration and winner set — unpicked / unplayed maps are
    # skipped).  If the specific map can't be found we leave the
    # per-game fields None and the snapshot scoring will treat
    # them as `no_data` rather than punish the model.
    played = [m for m in maps if m.get("duration") and m.get("winner")]
    target_map: Optional[Dict[str, Any]] = None
    if game_no is not None and 1 <= game_no <= len(played):
        target_map = played[game_no - 1]
    elif len(played) == 1:
        target_map = played[0]
    if target_map is not None:
        try:
            duration = int(target_map.get("duration") or 0) or None
        except (TypeError, ValueError):
            duration = None
        try:
            rs = int(target_map.get("radiant_score") or 0)
            ds = int(target_map.get("dire_score") or 0)
            kills = rs + ds if (rs or ds) else None
        except (TypeError, ValueError):
            kills = None
        out["duration_sec"] = duration
        out["kills_total"] = kills
        out["map_winner"] = target_map.get("winner")
        # v0.4.0.3: first_15_kills.  We can approximate from the
        # gold-advantage time-series: a positive gold lead at the
        # 5-min mark is almost always the result of a kill (rare
        # other sources matter at the pro tier).  Multiply by a
        # 0.7 constant to convert "gold lead" → "kills lead at
        # 15min" empirically; this is a very rough proxy.  When
        # the time series is missing we leave None.
        gold_adv = target_map.get("radiant_gold_adv") or []
        f15 = None
        try:
            if isinstance(gold_adv, list) and gold_adv:
                # Sample at 5min (index 5) and 15min (index 15).
                # Use the difference as a kill count proxy.
                if len(gold_adv) > 15:
                    g5 = int(gold_adv[5] or 0)
                    g15 = int(gold_adv[15] or 0)
                    # The "kills lead at 15" is roughly
                    # (g15 - g5) / 350 (avg gold/kill ≈ 350).
                    f15 = max(0, int((g15 - g5) / 350))
        except (TypeError, ValueError):
            f15 = None
        out["first_15_kills"] = f15
    return out


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
        # v0.4.0.3: pass game_no so the per-map stats are looked
        # up correctly.  game_no is on the row's `extra` payload.
        game_no = (r.get("extra") or {}).get("game_no")
        try:
            game_no_int = int(game_no) if game_no is not None else None
        except (TypeError, ValueError):
            game_no_int = None
        info = _match_actual_winner(int(sid), game_no=game_no_int)
        if info is None:
            # match not finished yet — leave for next tick
            continue
        r["actual_winner"] = info["winner"]
        r["actual_ended_at"] = info["ended_at"]
        # v0.4.0.3: per-game actual values (kills / duration /
        # first_15).  Stash them on the row so compute_stats can
        # report them and snapshot scoring has them in scope.
        r["actual_kills"] = info.get("kills_total")
        r["actual_duration"] = info.get("duration_sec")
        r["actual_first_15"] = info.get("first_15_kills")
        # Compare predicted team name to the actual team id
        # The ML engine emits "team_a" / "team_b" labels, so we map
        # the actual winner_id back to that label by looking at the
        # recorded extra payload if present.
        r["scored"] = _compare(r, info)
        # v0.4.0.3: snapshot rows get a separate verdict per
        # target.  We only mark `scored_snapshot` if the row has
        # the live-tracking fields; otherwise we leave it None.
        if r.get("is_snapshot"):
            r["scored_snapshot"] = _compare_snapshot(r, info)
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


# v0.4.0.3: tolerance for "close enough" predictions on numeric
# targets.  The v17 model isn't precise to the kill; an off-by-3
# prediction is fine for a 30-min game.  These tolerances match
# the bet-market brackets the front-end surfaces (kills over/under
# 2.5, duration over/under 5min, etc.).
KILLS_TOL = 3
DURATION_TOL_SEC = 300  # 5 min
FIRST_15_TOL = 2


def _compare_snapshot(rec: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """Score the 3-5 min snapshot row against the actual per-game stats.

    Returns a per-target verdict dict:
      {
        "winner":     "correct" | "wrong" | "push" | "no_data",
        "kills":      "ok" | "off" | "no_data",
        "duration":   "ok" | "off" | "no_data",
        "first_15":   "ok" | "off" | "no_data",
        "deltas": {
            "kills":    int (predicted - actual),
            "duration": int (predicted - actual, in seconds),
            "first_15": int (predicted - actual),
        }
      }

    "ok" = within tolerance; "off" = outside tolerance; "no_data"
    = we couldn't compute the actual value (e.g. the map data
    didn't include radiant_gold_adv for first_15).
    """
    out: Dict[str, Any] = {"winner": "no_data", "kills": "no_data",
                           "duration": "no_data", "first_15": "no_data",
                           "deltas": {}}
    # Winner — reuse the legacy comparator for the side label.
    if info.get("winner") is not None and info.get("winner") != "draw" or info.get("winner") == "draw":
        out["winner"] = _compare(rec, info)
    # Kills
    pk = rec.get("predicted_kills")
    ak = info.get("kills_total")
    if pk is not None and ak is not None:
        try:
            delta = int(round(float(pk) - int(ak)))
        except (TypeError, ValueError):
            delta = None
        if delta is not None:
            out["deltas"]["kills"] = delta
            out["kills"] = "ok" if abs(delta) <= KILLS_TOL else "off"
    # Duration
    pd_ = rec.get("predicted_duration")
    ad = info.get("duration_sec")
    if pd_ is not None and ad is not None:
        try:
            # prediction may be in minutes (legacy) or seconds (v17);
            # if the magnitude is < 1000 we assume minutes and convert.
            pd_val = float(pd_)
            if pd_val < 1000:
                pd_val = pd_val * 60.0
            delta = int(round(pd_val - int(ad)))
        except (TypeError, ValueError):
            delta = None
        if delta is not None:
            out["deltas"]["duration"] = delta
            out["duration"] = "ok" if abs(delta) <= DURATION_TOL_SEC else "off"
    # First-15 kills
    pf15 = rec.get("predicted_first_15")
    af15 = info.get("first_15_kills")
    if pf15 is not None and af15 is not None:
        try:
            delta = int(round(float(pf15) - int(af15)))
        except (TypeError, ValueError):
            delta = None
        if delta is not None:
            out["deltas"]["first_15"] = delta
            out["first_15"] = "ok" if abs(delta) <= FIRST_15_TOL else "off"
    return out


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
