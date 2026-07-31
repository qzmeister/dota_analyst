"""Background poller for the configured odds backend.

The live card asks for a single match's quotes (Steam match_id)
but most bookmakers key quotes by team-pair.  To bridge, this
module periodically calls `backend.get_all_live_quotes()` and
caches the result in a `Dict[str, List[OddsQuote]]` keyed by
`f"{team_a}|{team_b}"`.  The live card's `_compute_odds` then
looks up by team-pair.

Threading model: one background thread (mirrors opendota_live).
Module-scope state with an RLock for read/write safety.

Configuration: env-driven, no other wiring needed.
  * ODDS_BACKEND — fully-qualified module.class (loaded lazily)
  * ODDS_LIVE_POLL_SEC — interval between polls, default 60

Behaviour:
  * When no backend is configured (or stub), the poller still
    runs but does nothing — `get_all_live_quotes` returns {}.
  * When the configured backend raises (e.g. expired session),
    the exception is logged once per minute and the poller
    continues.  No backoff stack: the next poll tries again.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .odds import OddsBackend, OddsQuote
from .odds_match import team_pair_key
from ._logging import get_logger

log = get_logger(__name__)

# How often the poller refreshes the live-quotes cache.
POLL_INTERVAL_SEC = float(os.environ.get("ODDS_LIVE_POLL_SEC", "60"))

# Live-quotes entries older than this are considered stale.
LIVE_TTL_SEC = 90.0

# team_a|team_b  (lowercased) -> {
#   "ts":  monotonic timestamp of when we stored,
#   "quotes": List[OddsQuote],
# }
_state: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()

# Poller state
_loop_thread: Optional[threading.Thread] = None
_loop_started = threading.Event()
_loop_should_stop = threading.Event()
_last_warn: float = 0.0


def _now() -> float:
    return time.monotonic()


def _key(team_a: str, team_b: str) -> str:
    # v0.4.2: route through `team_pair_key` so the live card's
    # lookup uses the same normalisation as the backend's storage
    # (lowercase + strip + replace [.,_-] + drop common suffixes
    # like "team", "esports").  Without this, a name divergence
    # like "Team.Liquid" (DLTV) vs "Team Liquid" (odds-api.io)
    # would silently miss the cache.
    return team_pair_key(team_a, team_b)


def get_quotes_for_teams(team_a: str, team_b: str) -> List[OddsQuote]:
    """Read-side: return quotes for the (team_a, team_b) pair, or [].

    Empty if the pair isn't in the cache or the entry is older
    than `LIVE_TTL_SEC`.
    """
    k = _key(team_a, team_b)
    with _lock:
        entry = _state.get(k)
        if entry is None:
            return []
        if _now() - float(entry.get("ts") or 0) > LIVE_TTL_SEC:
            return []
        return list(entry.get("quotes") or [])


def get_all_live_quotes() -> Dict[str, List[OddsQuote]]:
    """Read-side: dump the entire live-quotes cache (post-TTL)."""
    now = _now()
    out: Dict[str, List[OddsQuote]] = {}
    with _lock:
        for k, entry in _state.items():
            if now - float(entry.get("ts") or 0) > LIVE_TTL_SEC:
                continue
            out[k] = list(entry.get("quotes") or [])
    return out


def _refresh_once(backend: OddsBackend) -> int:
    """One poll cycle: pull all live quotes, replace the cache.

    Returns the number of keys that were updated.
    """
    global _last_warn
    try:
        snap = backend.get_all_live_quotes() or {}
    except Exception as exc:
        now = time.monotonic()
        if now - _last_warn > 60:
            log.warning("odds poller: backend %r raised: %s", backend.name, exc)
            _last_warn = now
        return 0
    if not isinstance(snap, dict):
        return 0
    now = _now()
    updated = 0
    with _lock:
        # Wholesale replace with the latest snapshot.  Anything
        # not in the new snapshot falls out (matches that ended
        # since the last poll).
        new_keys = set()
        for k, quotes in snap.items():
            # Normalise key to lowercase
            nk = k.strip().lower()
            new_keys.add(nk)
            _state[nk] = {"ts": now, "quotes": list(quotes)}
            updated += 1
        # Drop keys not seen in the new snapshot.  These are
        # matches that left the bookmaker's live feed.
        for k in list(_state.keys()):
            if k not in new_keys:
                _state.pop(k, None)
    return updated


def _poller_loop() -> None:
    log.info("odds_live: poller started, interval=%.1fs", POLL_INTERVAL_SEC)
    # Lazy backend import so importing this module doesn't pull
    # the user's chosen backend's deps at process start.
    from .odds import get_backend
    while not _loop_should_stop.is_set():
        try:
            backend = get_backend()
            n = _refresh_once(backend)
            if n:
                log.info("odds_live: refreshed %d live-quotes keys", n)
        except Exception as exc:
            log.warning("odds_live: poll cycle failed: %s", exc)
        # Cancellable sleep.
        slept = 0.0
        while slept < POLL_INTERVAL_SEC and not _loop_should_stop.is_set():
            time.sleep(0.5)
            slept += 0.5
    log.info("odds_live: poller stopped")


def start_poller() -> Optional[threading.Thread]:
    """Start the background poller.  Idempotent."""
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return _loop_thread
    _loop_should_stop.clear()
    t = threading.Thread(target=_poller_loop, name="odds-live-poller", daemon=True)
    t.start()
    _loop_thread = t
    return t


def stop_poller(timeout: float = 5.0) -> None:
    _loop_should_stop.set()
    if _loop_thread is not None:
        _loop_thread.join(timeout=timeout)
