"""OpenDota live API as a backup / primary source for live match state.

v0.4.0.3: until now, the live card's gold_lead, destroyed_towers and
player_name fields depended on either the DLTV socket payload (only
sent for matches DLTV knows about) or a Playwright DOM extractor
(Cloudflare-blocked on the match page in the container).  Both
sources fail for Steam-only minor/pro matches, which is most of the
live pool we actually have.  The user reported 'cards go blank
when the socket switches' and 'no gold lead / no player nickname
on the live card' — the underlying cause was the same: no source
that works for our typical match set.

OpenDota's public `/api/live` endpoint solves this.  It is:
  * not behind Cloudflare (no Playwright needed)
  * a single GET for the entire live pool (~100 matches)
  * rich per-match fields:
      - radiant_lead, radiant_score, dire_score, game_time
      - players[].{account_id, hero_id, team_slot, team}
      - building_state (bitmask → destroyed towers)
      - team_name_radiant, team_name_dire
  * rate-limited to ~60 req/min/IP — fine for 5s polling

For player nicknames we hit `/api/players/<account_id>` which
returns `profile.personaname` + `profile.loccountrycode`.  Cached
for 1h on disk because nicknames are stable.

The data layout in `_live_state[match_id]` is a normalised dict:
  {
    "radiant_score":      int | None,
    "dire_score":         int | None,
    "game_time":          int | None,         # seconds (OpenDota's "game_time" is already in seconds)
    "radiant_lead":       int | None,         # signed: + = radiant ahead
    "destroyed_towers":   {"radiant": int, "dire": int} | None,
    "players": [
        {"account_id": int, "hero_id": int, "team": 0|1, "team_slot": int},
        ...
    ],
    "team_name_radiant":  str | None,
    "team_name_dire":     str | None,
    "league_id":          int | None,
    "spectators":         int | None,
    "ts":                 float,              # monotonic when we stored
    "source_ts":          int | None,         # OpenDota's last_update_time (unix seconds)
  }

Read access via `get_live_state(match_id)` is thread-safe; the
state is updated in-place from the background poller thread.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ._logging import get_logger
from .exceptions import UpstreamError

log = get_logger(__name__)

# v0.4.0.3: persistent on-disk cache for the per-account_id
# player_info dict.  The in-memory cache evaporates on every
# process restart; the on-disk one survives, so the long path
# of "fetch -> rate limit -> 5min backoff -> fetch again"
# doesn't lose us everything we already learned.  Nicknames
# don't change (Steam doesn't allow them to anymore), so a
# 30-day TTL is effectively forever.
ML_DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "ml_data"))
PLAYER_INFO_CACHE_FILE = ML_DATA_DIR / "opendota_player_info.json"
PLAYER_INFO_DISK_TTL_SEC = 30 * 24 * 3600.0  # 30 days

OPENDOTA_LIVE_URL = "https://api.opendota.com/api/live"
OPENDOTA_PLAYER_URL = "https://api.opendota.com/api/players/{account_id}"
HTTP_TIMEOUT = 4.0
LIVE_TTL_SEC = 15.0     # /api/live entries are stale after 15s
# v0.4.0.3: bumped from 10s to 30s.  OpenDota's anonymous
# rate limit is tighter than the documented 60/min in
# practice — we got 429 even at 6 req/min.  A 30s interval
# means 2 req/min for the live feed alone, which gives the
# player-info fetches (1 every 0.6s, capped at 30 per cycle)
# room to breathe inside the per-minute budget.
POLL_INTERVAL_SEC = 30.0
PLAYER_TTL_SEC = 24 * 3600.0  # nicknames don't change; cache for 24h
# OpenDota's anonymous rate limit is ~60 requests / minute / IP.
# The live feed is one call per poll; the per-player lookups
# are N calls.  We do the player lookups at PLAYER_FETCH_DELAY_SEC
# spacing and only for matches that are actually in-game
# (a real `radiant_score > 0` or `game_time > 0`), so a typical
# active board of 20 live matches adds ~20 player fetches per
# cycle, well under the limit.  Cold start (no cache at all)
# bumps against the limit briefly but recovers within a minute.
PLAYER_FETCH_DELAY_SEC = 0.6
PLAYER_FETCH_BATCH_LIMIT = 30  # max new account_ids per cycle

# Tower bitmask layout (OpenDota /api/live building_state).
# Bits 0-10 = radiant towers, 11-15 = radiant barracks (ancient +
# top/bot melee/ranged), 16-26 = dire towers, 27-31 = dire barracks.
# Source: https://github.com/odota/core/blob/master/constants/match.js
# (BUILDING_STATE_* constants).  A 0 bit = standing, 1 = destroyed.
# We only care about the tower subset (bits 0-10 and 16-26).
_TOWER_BIT_RANGES = {
    # (lo, hi_inclusive)  ->  (side, label, count)
    "radiant_t1_towers":  (0, 2),    # bottom/mid/top tier-1 — but layout varies; use 0-10
    "radiant_towers":     (0, 10),   # 11 radiant towers (3 T1 + 2 mid + 2 T2 + 2 T3 + 4 racks? no — racks are separate)
    "dire_towers":        (16, 26),  # 11 dire towers
}
# Real layout per odota/core (BUILDING_STATE_*):
#   bits 0-2   : radiant tier-1 (3 towers)
#   bits 3-4   : radiant tier-2 (2 towers)
#   bits 5-6   : radiant tier-3 (2 towers)
#   bits 7-8   : radiant ancient / throne (2)
#   bits 9-10  : radiant barracks melee (2)
#   bit  11    : radiant barracks ranged (1)
#   bits 16-26 : mirror for dire
# So the *tower* count is just 7 per side (T1 + T2 + T3 + ancient×2).
# The bit pattern is the same on both sides, shifted by 16 for dire.
# Source: odota-core `BUILDING_STATE_*` constants.
#
# We sum the bits set in the relevant ranges to get destroyed count.
_RADIANT_TOWER_LO, _RADIANT_TOWER_HI = 0, 8   # bits 0..8 — towers + ancient (no barracks)
_DIRE_TOWER_LO, _DIRE_TOWER_HI = 16, 24       # same shifted by 16


# Process-wide state.  Reads take a short RLock; writes happen on
# the background poller's thread.
_state: Dict[int, Dict[str, Any]] = {}
_state_ts: Dict[int, float] = {}
# account_id -> {personaname, loccountrycode, ts}
_player_cache: Dict[int, Dict[str, Any]] = {}
_lock = threading.RLock()

# Poller thread handle
_loop_thread: Optional[threading.Thread] = None
_loop_started = threading.Event()
_loop_should_stop = threading.Event()
# v0.4.0.3: backoff after HTTP 429.  OpenDota's anonymous
# rate limit is per-IP and tighter than the documented 60/min
# in practice (we hit 429 with a steady 6 req/min poll + 30
# player fetches per cycle).  When we get a 429 we stop
# polling for BACKOFF_BASE_SEC and double the backoff on
# each subsequent 429, capping at BACKOFF_MAX_SEC.
_backoff_sec: float = 0.0
_BACKOFF_BASE_SEC = 60.0
_BACKOFF_MAX_SEC = 600.0
_lock_backoff = threading.Lock()


def _now() -> float:
    return time.monotonic()


def get_live_state(match_id: int) -> Optional[Dict[str, Any]]:
    """Return the OpenDota live state for `match_id`, or None if stale/missing."""
    with _lock:
        ts = _state_ts.get(int(match_id))
        if ts is None:
            return None
        if _now() - ts > LIVE_TTL_SEC:
            return None
        return dict(_state[int(match_id)])


def get_player_info(account_id: int) -> Optional[Dict[str, str]]:
    """Return cached `{personaname, loccountrycode}` for a Steam account_id."""
    with _lock:
        e = _player_cache.get(int(account_id))
        if e is None:
            return None
        if _now() - float(e.get("ts", 0)) > PLAYER_TTL_SEC:
            return None
        return {
            "personaname": e.get("personaname") or "",
            "loccountrycode": e.get("loccountrycode") or "",
        }


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def parse_building_state(building_state: int) -> Dict[str, int]:
    """Convert OpenDota's `building_state` bitmask to destroyed counts.

    Returns a dict with `radiant` and `dire` counts of destroyed
    towers + ancient.  Bits outside the tower range are ignored
    (barracks, fountain, etc.).  When `building_state` is 0 (game
    hasn't started, or the bitmask isn't tracked), we return
    `{"radiant": 0, "dire": 0}` so the live card still renders
    cleanly.
    """
    if not building_state:
        return {"radiant": 0, "dire": 0}
    rad = 0
    for bit in range(_RADIANT_TOWER_LO, _RADIANT_TOWER_HI + 1):
        if building_state & (1 << bit):
            rad += 1
    dire = 0
    for bit in range(_DIRE_TOWER_LO, _DIRE_TOWER_HI + 1):
        if building_state & (1 << bit):
            dire += 1
    return {"radiant": rad, "dire": dire}


def _normalize_match(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one OpenDota `/api/live` row into our cached shape."""
    try:
        match_id = int(raw.get("match_id") or 0)
    except (TypeError, ValueError):
        return None
    if not match_id:
        return None
    try:
        radiant_lead = int(raw["radiant_lead"]) if raw.get("radiant_lead") is not None else None
    except (TypeError, ValueError):
        radiant_lead = None
    try:
        radiant_score = int(raw.get("radiant_score") or 0)
    except (TypeError, ValueError):
        radiant_score = None
    try:
        dire_score = int(raw.get("dire_score") or 0)
    except (TypeError, ValueError):
        dire_score = None
    try:
        game_time = int(raw.get("game_time") or 0) or None
    except (TypeError, ValueError):
        game_time = None
    try:
        building_state = int(raw.get("building_state") or 0)
    except (TypeError, ValueError):
        building_state = 0
    destroyed = parse_building_state(building_state) if building_state else None
    # Players: each is {account_id, hero_id, team, team_slot}.
    players = []
    for p in (raw.get("players") or []):
        if not isinstance(p, dict):
            continue
        try:
            aid = int(p.get("account_id") or 0)
            hid = int(p.get("hero_id") or 0)
            slot = int(p.get("team_slot") or 0)
        except (TypeError, ValueError):
            continue
        # team_slot: 0..127 = radiant, 128..255 = dire (Dota's
        # standard encoding).  Convert to 0/1.
        team = 0 if slot < 128 else 1
        if not aid or not hid:
            continue
        players.append({
            "account_id": aid,
            "hero_id": hid,
            "team": team,
            "team_slot": slot,
        })
    try:
        source_ts = int(raw.get("last_update_time") or 0) or None
    except (TypeError, ValueError):
        source_ts = None
    return {
        "match_id": match_id,
        "radiant_score": radiant_score,
        "dire_score": dire_score,
        "game_time": game_time,
        "radiant_lead": radiant_lead,
        "destroyed_towers": destroyed,
        "players": players,
        "team_name_radiant": raw.get("team_name_radiant") or None,
        "team_name_dire": raw.get("team_name_dire") or None,
        "league_id": int(raw["league_id"]) if raw.get("league_id") is not None else None,
        "spectators": int(raw["spectators"]) if raw.get("spectators") is not None else None,
        "source_ts": source_ts,
        "ts": _now(),
    }


def fetch_live() -> List[Dict[str, Any]]:
    """GET /api/live and return a list of normalised live matches.

    v0.4.0.3: detects HTTP 429 and increments a backoff so the
    poller doesn't keep hammering OpenDota while we're being
    rate-limited.  Successful fetches clear the backoff.
    """
    global _backoff_sec
    req = urllib.request.Request(
        OPENDOTA_LIVE_URL,
        headers={"User-Agent": "dota-analyst/0.4 (research; contact via GitHub)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            with _lock_backoff:
                _backoff_sec = min(_backoff_sec * 2 if _backoff_sec else _BACKOFF_BASE_SEC,
                                   _BACKOFF_MAX_SEC)
            log.warning(
                "opendota_live: /api/live -> 429, backing off %.0fs",
                _backoff_sec,
            )
            return []
        log.warning("opendota_live: /api/live fetch failed: %s", exc)
        return []
    except (OSError, UpstreamError) as exc:
        log.warning("opendota_live: /api/live fetch failed: %s", exc)
        return []
    if not isinstance(data, list):
        log.warning("opendota_live: /api/live returned non-list payload: %r", type(data).__name__)
        return []
    # Successful fetch — clear any backoff.
    if _backoff_sec:
        with _lock_backoff:
            log.info("opendota_live: /api/live recovered from 429, clearing backoff")
            _backoff_sec = 0.0
    out: List[Dict[str, Any]] = []
    for raw in data:
        norm = _normalize_match(raw)
        if norm is not None:
            out.append(norm)
    log.info("opendota_live: /api/live -> %d live matches", len(out))
    return out


def fetch_player_info(account_id: int) -> Optional[Dict[str, str]]:
    """GET /api/players/<id> and return {personaname, loccountrycode}."""
    url = OPENDOTA_PLAYER_URL.format(account_id=int(account_id))
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "dota-analyst/0.4 (research; contact via GitHub)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            data = json.loads(r.read())
    except (OSError, UpstreamError) as exc:
        log.debug("opendota_live: /api/players/%s fetch failed: %s", account_id, exc)
        return None
    if not isinstance(data, dict):
        return None
    profile = data.get("profile") or {}
    if not isinstance(profile, dict):
        return None
    return {
        "personaname": profile.get("personaname") or "",
        "loccountrycode": profile.get("loccountrycode") or "",
    }


def _ingest_live(rows: List[Dict[str, Any]]) -> Set[int]:
    """Replace `_state` with the latest snapshot, returning the
    set of match_ids we just refreshed.

    v0.4.0.3: critical race fix.  When `fetch_live` returns an
    empty list (e.g. OpenDota rate-limited us with HTTP 429),
    the previous code did `_state.clear(); _state.update({})`
    which dropped the entire 100-match snapshot we had
    accumulated in the last 10 seconds.  The poller would
    then keep retrying every 10s and the state would oscillate
    between "full" (after a successful poll) and "empty" (after
    a 429), giving the board ~50% MISS rate on every match.

    We now treat an empty `rows` as a transient failure: keep
    the prior state in place.  The TTL layer (LIVE_TTL_SEC)
    still expires stale entries naturally, so matches that
    genuinely ended will fall out of the cache within 15s of
    a successful poll confirming the drop.
    """
    if not rows:
        return set()
    seen: Set[int] = set()
    new_state: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        mid = row.get("match_id")
        if mid is None:
            continue
        new_state[int(mid)] = row
        seen.add(int(mid))
    with _lock:
        # Replace state wholesale.  Anything not in `new_state`
        # is now stale (the OpenDota poll didn't include it, so
        # the match is no longer live on OpenDota either).
        _state.clear()
        _state.update(new_state)
        now = _now()
        _state_ts.clear()
        for mid in new_state:
            _state_ts[mid] = now
    return seen


def _load_player_info_from_disk() -> None:
    """Populate the in-memory `_player_cache` from disk on startup.

    v0.4.0.3: the player_info dict (personaname, loccountrycode)
    is small (~120 bytes per entry, ~600 bytes per match with 5
    unique players per match) and very stable.  We persist it to
    `ml_data/opendota_player_info.json` so a process restart
    doesn't have to re-fetch the same 1000 account_ids over a
    5-minute backoff cycle.  The on-disk entries are timestamped
    per fetch; we drop anything older than `PLAYER_INFO_DISK_TTL_SEC`
    (30 days, well past Steam's nickname-change cooldown).
    """
    if not PLAYER_INFO_CACHE_FILE.exists():
        return
    try:
        with open(PLAYER_INFO_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    now = _now()
    loaded = 0
    skipped_expired = 0
    with _lock:
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            try:
                aid = int(k)
                ts = float(v.get("ts") or 0)
            except (TypeError, ValueError):
                continue
            if now - ts > PLAYER_INFO_DISK_TTL_SEC:
                skipped_expired += 1
                continue
            _player_cache[aid] = v
            loaded += 1
    if loaded or skipped_expired:
        log.info(
            "opendota_live: loaded %d player_info entries from disk (skipped %d expired)",
            loaded, skipped_expired,
        )


def _save_player_info_to_disk() -> None:
    """Persist the current `_player_cache` to disk.

    Called opportunistically from `_refresh_player_info` after
    a successful batch.  Atomic write (write to .tmp + os.replace)
    so a crash mid-write doesn't corrupt the cache.
    """
    if not _player_cache:
        return
    try:
        ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    with _lock:
        # Don't snapshot the live dict — we want a value copy
        # so concurrent writes from the poller don't race us.
        snapshot: Dict[int, Dict[str, Any]] = {aid: dict(v) for aid, v in _player_cache.items()}
    # JSON keys must be strings.
    serialised = {str(aid): v for aid, v in snapshot.items()}
    tmp = PLAYER_INFO_CACHE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serialised, f, ensure_ascii=False)
        os.replace(tmp, PLAYER_INFO_CACHE_FILE)
    except OSError as exc:
        log.debug("opendota_live: player_info disk write failed: %s", exc)


def _refresh_player_info(account_ids: Set[int]) -> None:
    """For each account_id we don't have cached, fetch it.

    OpenDota's anonymous rate limit is ~60 requests/minute/IP.
    The live feed costs 1 request; per-player lookups are N
    requests.  We:
      * Skip ids we already have in the cache (within TTL).
      * Cap the per-cycle batch to PLAYER_FETCH_BATCH_LIMIT.
      * Pace at PLAYER_FETCH_DELAY_SEC between calls.
      * Cache empty placeholders for failed fetches so a flaky
        API doesn't make us retry the same id every 10s.

    Cold start (the first poll after a fresh process boot)
    processes up to 30 new account_ids per cycle, ~18 seconds
    of fetching — well under the next poll's tick.  Within 2-3
    cycles (30-60s) the active match set is fully cached.
    """
    if not account_ids:
        return
    with _lock:
        already = {aid for aid in account_ids
                   if aid in _player_cache
                   and _now() - float(_player_cache[aid].get("ts", 0)) < PLAYER_TTL_SEC}
    todo = [aid for aid in account_ids if aid not in already]
    if not todo:
        return
    # Cap the per-cycle batch so a sudden influx of live matches
    # (e.g. a tournament with 50 fresh pros) doesn't blow the
    # rate limit.  Remaining ids are picked up on the next cycle.
    todo = todo[:PLAYER_FETCH_BATCH_LIMIT]
    log.info("opendota_live: refreshing player info for %d account_ids", len(todo))
    new_entries = 0
    for aid in todo:
        info = fetch_player_info(aid)
        if info is None:
            # Cache a placeholder so we don't keep retrying
            # the same id within TTL (negative caching).
            info = {"personaname": "", "loccountrycode": ""}
        with _lock:
            _player_cache[int(aid)] = {
                **info,
                "ts": _now(),
            }
        new_entries += 1
        time.sleep(PLAYER_FETCH_DELAY_SEC)
    # v0.4.0.3: persist the (possibly expanded) cache to disk
    # so the next process restart doesn't have to re-fetch the
    # same account_ids.  Cheap (JSON write of ~120 bytes per
    # entry) and amortised across the cycle.
    if new_entries:
        _save_player_info_to_disk()


def _poll_once() -> Set[int]:
    """One poll cycle: fetch /api/live, refresh player info for
    any new account_ids we saw.

    v0.4.0.3: only refresh player info for matches that are
    actually in-game (a real `radiant_score > 0` or
    `game_time > 0`).  The 100-ish matches OpenDota reports
    include freshly-loaded lobby screens with no players in
    their final positions — fetching nicknames for those is
    pure waste and pushes us against the 60-req/min rate
    limit.  Restricting to in-game matches keeps us well
    under the limit on a typical evening.
    """
    rows = fetch_live()
    seen = _ingest_live(rows)
    # Collect account_ids across the IN-GAME matches only.
    account_ids: Set[int] = set()
    for row in rows:
        # A match is "in-game" when either side has at least one
        # kill, or the game clock is past 0.  Lobby / pre-game
        # matches report `radiant_score: 0, dire_score: 0,
        # game_time: 0` and have placeholder player slots
        # that don't reflect the real roster.
        in_game = (
            (row.get("radiant_score") or 0) > 0
            or (row.get("dire_score") or 0) > 0
            or (row.get("game_time") or 0) > 0
        )
        if not in_game:
            continue
        for p in row.get("players") or []:
            aid = p.get("account_id")
            if aid:
                account_ids.add(int(aid))
    _refresh_player_info(account_ids)
    return seen


def _poller_loop() -> None:
    """Background poller: runs until `_loop_should_stop` is set.

    v0.4.0.3: when `_backoff_sec > 0` (we hit 429), we wait
    that long before the next poll instead of the regular
    `POLL_INTERVAL_SEC`.  A successful `_poll_once` clears
    the backoff via the success path in `fetch_live`.
    """
    log.info("opendota_live: poller started, interval=%.1fs", POLL_INTERVAL_SEC)
    while not _loop_should_stop.is_set():
        try:
            _poll_once()
        except Exception as exc:  # noqa: BLE001
            log.warning("opendota_live: poll cycle failed: %s", exc, exc_info=False)
        # Sleep with cancellable semantics.  Length depends on
        # whether we're in 429 backoff or not.
        with _lock_backoff:
            sleep_for = _backoff_sec if _backoff_sec > 0 else POLL_INTERVAL_SEC
        slept = 0.0
        while slept < sleep_for and not _loop_should_stop.is_set():
            time.sleep(0.5)
            slept += 0.5
    log.info("opendota_live: poller stopped")


def start_poller() -> threading.Thread:
    """Start the background poller.  Idempotent."""
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return _loop_thread
    # v0.4.0.3: warm the player_info cache from disk before
    # the first poll.  Otherwise a fresh process has zero
    # entries and has to re-fetch every active account_id
    # from scratch, which (a) burns rate-limit budget, and
    # (b) means the live card shows `player_name: null`
    # for the first 60-90s of uptime.
    _load_player_info_from_disk()
    _loop_should_stop.clear()
    t = threading.Thread(target=_poller_loop, name="opendota-live-poller", daemon=True)
    t.start()
    _loop_thread = t
    return t


def stop_poller(timeout: float = 5.0) -> None:
    _loop_should_stop.set()
    if _loop_thread is not None:
        _loop_thread.join(timeout=timeout)
    # v0.4.0.3: final flush of the player_info cache to disk.
    # The poller already saves opportunistically, but a clean
    # shutdown is a guaranteed save point that doesn't depend
    # on which batch happened to finish last.
    _save_player_info_to_disk()


def populate_player_info_for_live() -> int:
    """Synchronous one-shot poll: refreshes live matches + the
    account_ids they reference.  Returns the number of matches
    we now have state for.  Used by tests and the startup path
    (so a fresh process doesn't have to wait 10s for the first
    data to land).

    Same in-game filter as `_poll_once` — only fetch player
    info for matches that have actually started.
    """
    rows = fetch_live()
    _ingest_live(rows)
    account_ids: Set[int] = set()
    for row in rows:
        in_game = (
            (row.get("radiant_score") or 0) > 0
            or (row.get("dire_score") or 0) > 0
            or (row.get("game_time") or 0) > 0
        )
        if not in_game:
            continue
        for p in row.get("players") or []:
            aid = p.get("account_id")
            if aid:
                account_ids.add(int(aid))
    _refresh_player_info(account_ids)
    return len(rows)
