"""
DLTV v1 API client.

Endpoints used (base https://dltv.org/api/v1):
  GET /events                     -> {items: [{id, title, is_active}]}
  GET /events/{id}/series         -> {series: [ series{...maps[...]} ]}
  GET /heroes                     -> {items: [{id, steam_id, title, win_rate, avg_duration, kda, roles, ...}]}
  GET /teams                      -> {items: [{id, title, win_rate, fb_rate, f10_rate, rank, ...}]}

Plus the static live JSON (rich live draft + per-hero team win-rates):
  GET https://dltv.org/live/{match_id}.json

Hero id namespaces (IMPORTANT):
  - v1 maps[].picks/bans use DLTV internal hero id  (field `id`,  range 1..127)
  - /live/{match_id}.json db picks use STEAM hero id (field `steam_id`, range 1..155)
  We therefore index heroes by BOTH.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._http import request_json
from ._logging import get_logger
from .exceptions import DLTVError, ParseError

log = get_logger(__name__)

BASE = "https://dltv.org/api/v1"
LIVE_BASE = "https://dltv.org/live"
SITE = "https://dltv.org"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


# --------------------------------------------------------------------------- #
# Small TTL cache
# --------------------------------------------------------------------------- #

class _TTLCache:
    """Thread-safe TTL cache with optional LRU eviction.

    Without `maxsize` the cache grows unbounded; set it to bound memory
    in long-running processes. Eviction is LRU-on-set: when `set` pushes
    size over `maxsize`, the oldest entry is dropped first.
    """

    def __init__(self, maxsize: int = 0):
        self._data: Dict[str, Any] = {}
        self._exp: Dict[str, float] = {}
        self._lock = threading.Lock()
        # 0 = unbounded. Otherwise evict oldest entry when len > maxsize.
        self._maxsize = int(maxsize)

    def get(self, key: str):
        with self._lock:
            if key in self._data and time.time() < self._exp.get(key, 0):
                return self._data[key]
        return None

    def set(self, key: str, value: Any, ttl: float):
        with self._lock:
            self._data[key] = value
            self._exp[key] = time.time() + ttl
            if self._maxsize and len(self._data) > self._maxsize:
                # Drop the oldest-inserted key. dict preserves insertion
                # order in Python 3.7+, so next(iter(...)) is the LRU head.
                oldest = next(iter(self._data))
                self._data.pop(oldest, None)
                self._exp.pop(oldest, None)


def _http_json(url: str, timeout: float = 10.0) -> Optional[Any]:
    """Fetch JSON with retry + exponential backoff.

    Per RULES.md §1: 3 attempts, exponential backoff, fallback on error.
    Wraps the shared `request_json` so dltv_client respects the same retry
    policy as the DatDota client.
    """
    return request_json(
        url=url,
        headers=HEADERS,
        timeout=timeout,
        retries=3,
        backoff_base=1.0,
        backoff_cap=20.0,
    )


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _abs_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if path.startswith("http"):
        return path
    return SITE + path


def _parse_dt(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        s = v.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        # fromisoformat raises ValueError for malformed input, TypeError
        # if `v` isn't a string at all. Both → unparseable, return None.
        return None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class DLTVClient:
    def __init__(self, series_ttl: float = 45.0, static_ttl: float = 900.0, cache_maxsize: int = 128):
        self._cache = _TTLCache(maxsize=cache_maxsize)
        self.series_ttl = series_ttl
        self.static_ttl = static_ttl
        self._hero_by_id: Dict[int, Dict] = {}
        self._hero_by_steam: Dict[int, Dict] = {}
        # Guards one-time hero index load across threads.
        self._hero_lock = threading.Lock()
        self._heroes_loaded = False

    # ---- raw resources (cached) ---- #

    def get_events(self) -> List[Dict]:
        cached = self._cache.get("events")
        if cached is not None:
            return cached
        d = _http_json(f"{BASE}/events") or {}
        items = d.get("items", []) if isinstance(d, dict) else []
        self._cache.set("events", items, self.static_ttl)
        return items

    def get_heroes(self) -> List[Dict]:
        cached = self._cache.get("heroes")
        if cached is not None:
            return cached
        d = _http_json(f"{BASE}/heroes") or {}
        items = d.get("items", []) if isinstance(d, dict) else []
        self._cache.set("heroes", items, self.static_ttl)
        self._build_hero_index(items)
        return items

    def get_series(self, event_id: int) -> List[Dict]:
        key = f"series:{event_id}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # v0.3.15+: bound the per-league fetch at 6s.  Cold cache
        # with 12 leagues × 5 workers would otherwise be 12s+6×0s
        # ~ 14s of in-flight work plus the queue, and we want the
        # board to finish under nginx's 30s proxy_read_timeout.
        d = _http_json(f"{BASE}/events/{event_id}/series", timeout=6.0) or {}
        series = d.get("series", []) if isinstance(d, dict) else []
        self._cache.set(key, series, self.series_ttl)
        return series

    def get_live_json(self, match_id: int) -> Optional[Dict]:
        """Rich live/finished match JSON (draft + per-hero team win-rates).

        v0.3.14: timeout cut from 6s -> 3s.  Called in a loop for every
        live match by `discovery.get_live_and_prematch()`; with 30+
        live matches on the wire a 6s timeout made the cold-cache
        /api/board take 3+ minutes.

        v0.4.0-perf: dropped retries to 1 and timeout to 1.5s.  The
        default _http_json has retries=3, backoff_base=1.0 which means
        a timed-out /live/{id}.json took 3*1.5 + 1 + 2 + 4 = ~12s of
        waiting per match.  Across 20+ live matches that was 3-5
        minutes of build time.  Since the live data is only useful for
        in-progress games, a single fast attempt is enough — if it
        times out, the next 5s TTL window will retry.  Combined with
        parallel enrichment in `tracker.get_live_and_prematch()` this
        drops build time to under 5s.
        """
        # Bypass _http_json: it has a fixed retries=3 policy.  We want
        # a single fast attempt (no backoff) so parallel enrichment
        # across 30+ matches finishes in seconds, not minutes.
        from ._http import request_json
        return request_json(
            url=f"{LIVE_BASE}/{match_id}.json",
            headers=HEADERS,
            timeout=1.5,
            retries=1,
            backoff_base=0.0,
            backoff_cap=0.0,
        )

    # ---- hero index ---- #

    def _build_hero_index(self, heroes: List[Dict]):
        by_id, by_steam = {}, {}
        for h in heroes:
            norm = self._normalize_hero(h)
            if h.get("id") is not None:
                by_id[h["id"]] = norm
            if h.get("steam_id") is not None:
                by_steam[h["steam_id"]] = norm
        self._hero_by_id = by_id
        self._hero_by_steam = by_steam

    @staticmethod
    def _normalize_hero(h: Dict) -> Dict:
        roles: List[str] = []
        raw_roles = h.get("roles")
        if isinstance(raw_roles, str):
            try:
                roles = json.loads(raw_roles)
            except ValueError:
                # json.JSONDecodeError inherits from ValueError.
                # Malformed role string → empty list, no role badges.
                roles = []
        elif isinstance(raw_roles, list):
            roles = raw_roles
        return {
            "id": h.get("id"),
            "steam_id": h.get("steam_id"),
            "name": h.get("title"),
            "image": _abs_url(h.get("image")),
            "icon": _abs_url(h.get("icon")),
            "attribute": h.get("primary_attribute"),
            "attack_type": h.get("attack_type"),
            "roles": roles,
            "win_rate": _to_float(h.get("win_rate")),        # percent 0..100
            "pick_rate": _to_float(h.get("pick_rate")),
            "avg_duration": _to_float(h.get("avg_duration")),  # seconds
            "kda": _to_float(h.get("kda")),
        }

    def hero_by_dltv_id(self, hid: Optional[int]) -> Optional[Dict]:
        if not self._heroes_loaded:
            with self._hero_lock:
                if not self._heroes_loaded:
                    self.get_heroes()
                    self._heroes_loaded = True
        return self._hero_by_id.get(hid)

    def hero_by_steam_id(self, sid: Optional[int]) -> Optional[Dict]:
        if not self._heroes_loaded:
            with self._hero_lock:
                if not self._heroes_loaded:
                    self.get_heroes()
                    self._heroes_loaded = True
        return self._hero_by_steam.get(sid)

    # ---- normalization helpers ---- #

    @staticmethod
    def normalize_team(team: Optional[Dict]) -> Dict:
        team = team or {}
        return {
            "id": team.get("id"),
            "name": team.get("title") or "TBD",
            "tag": team.get("tag"),
            "logo": _abs_url(team.get("image")),
            "rank": team.get("rank"),
            "win_rate": _to_float(team.get("win_rate")),
            "fb_rate": _to_float(team.get("fb_rate")),
            "f10_rate": _to_float(team.get("f10_rate")),
            "maps_total": team.get("maps_total"),
        }

    def classify_stage(self, series: Dict) -> str:
        """prematch | live | postmatch, robust to unknown status codes."""
        status = series.get("status")
        ended = _parse_dt(series.get("ended_at"))
        started = _parse_dt(series.get("started_at")) or _parse_dt(series.get("liquipedia_date"))
        now = datetime.now(timezone.utc)

        if status == 2 or ended is not None:
            return "postmatch"
        # a played/in-progress map present but series not ended -> live
        if self._has_active_map(series):
            return "live"
        if started is not None and started <= now:
            return "live"
        return "prematch"

    @staticmethod
    def _has_active_map(series: Dict) -> bool:
        for m in series.get("maps") or []:
            # map with a steam_id and started but no winner yet == in progress
            if m.get("steam_id") and not m.get("winner") and m.get("started_at"):
                return True
        return False


# module-level singleton
client = DLTVClient()


# --------------------------------------------------------------------------- #
# Watchlist: steam_ids manually tracked when DLTV v1 API omits a series
# --------------------------------------------------------------------------- #

WATCHLIST_TTL = 20.0  # seconds


def _live_json_to_series(match_id: int, lj: Dict) -> Optional[Dict]:
    """Convert a /live/{match_id}.json payload into a v1-compatible series dict.

    Returns None if the JSON has no usable db.first_team / db.second_team data.

    v0.4.0.1: DLTV's live JSON now carries the picks in TWO places:
      * `db.first_team.picks[*]` — old location, sometimes empty
      * `fast_picks.first_team[*]` — new location, has the player
        nickname throughout the game (not just during draft)

    The fast_picks shape also has `country.image` (the player's
    country flag) and the player slug.  We prefer fast_picks
    when it has entries; fall back to db.*.picks for the older
    endpoint shape.
    """
    db = lj.get("db") or {}
    first = db.get("first_team") or {}
    second = db.get("second_team") or {}
    series_meta = db.get("series") or {}
    if not first or not second:
        return None

    first_id = first.get("id")
    second_id = second.get("id")
    radiant_is_first = bool(first.get("is_radiant"))

    # v0.4.0.1: try fast_picks first (has player nicknames),
    # fall back to db.{first,second}_team.picks.  Either way,
    # each pick is a dict with at least `hero_id` / `_steam_id`
    # and (if from fast_picks) `_name` / `player_slug` /
    # `player_country`.
    fast = lj.get("fast_picks") or {}
    fast_first = fast.get("first_team") or []
    fast_second = fast.get("second_team") or []

    # v0.4.0.1: fast_picks entries are at the SAME level as
    # db.{first,second}_team.picks (i.e. they're already
    # split by the side that picked first/second, NOT by
    # radiant/dire).  Use the same is_radiant heuristic to
    # map first_team/second_team -> radiant/dire.
    def _map_fast_pick(p: Dict) -> Dict:
        # fast_picks shape: {hero_id, player: {title, slug},
        #                   country: {image: <flag_url>},
        #                   stats, player_stats}
        player = p.get("player") or {}
        country = p.get("country") or {}
        flag = country.get("image") if isinstance(country, dict) else None
        # country.image looks like /assets/plugins/flag-icon/flags/4x3/ru.svg;
        # the last path component before .svg is the country code.
        country_code = None
        if isinstance(flag, str) and "/" in flag:
            stem = flag.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if len(stem) <= 3:
                country_code = stem.upper()
        return {
            "hero_id":     p.get("hero_id"),
            "_steam_id":   p.get("hero_id"),
            "order":       0,  # order is index-based; set after the loop
            "_name":       player.get("title") if isinstance(player, dict) else None,
            "player_slug":  player.get("slug") if isinstance(player, dict) else None,
            "player_country": country_code,
        }

    def _map_legacy_pick(p: Dict) -> Dict:
        # db.*.picks shape: {hero: {id, steam_id, title, slug, image, ...}}
        # No player nickname here.
        hero = p.get("hero") or {}
        return {
            "hero_id":     hero.get("steam_id"),
            "_steam_id":   hero.get("steam_id"),
            "order":       0,
        }

    # Prefer fast_picks when present, fall back to db.*.picks.
    if fast_first or fast_second:
        radiant_src = fast_first if radiant_is_first else fast_second
        dire_src    = fast_second if radiant_is_first else fast_first
        radiant_picks = [_map_fast_pick(p) for p in radiant_src]
        dire_picks    = [_map_fast_pick(p) for p in dire_src]
    else:
        radiant_picks = [_map_legacy_pick(p) for p in
                         (first.get("picks", []) if radiant_is_first else second.get("picks", []))]
        dire_picks    = [_map_legacy_pick(p) for p in
                         (second.get("picks", []) if radiant_is_first else first.get("picks", []))]
    # Order is index-based (fast_picks is ordered; db.picks is not,
    # so legacy picks land in whatever order they were stored).
    for i, p in enumerate(radiant_picks):
        p["order"] = i
    for i, p in enumerate(dire_picks):
        p["order"] = i

    # v0.4.0.1: bans are still only in `db.*.bans` (fast_picks
    # doesn't carry them post-draft).  No change to the
    # legacy _map_ban path.
    def _map_ban(p: Dict) -> Dict:
        hero = p.get("hero") or {}
        return {"hero_id": hero.get("steam_id"), "order": 0, "_steam_id": hero.get("steam_id")}

    radiant_bans  = [_map_ban(p) for p in first.get("bans", [])] if radiant_is_first else [_map_ban(p) for p in second.get("bans", [])]
    dire_bans     = [_map_ban(p) for p in second.get("bans", [])] if radiant_is_first else [_map_ban(p) for p in first.get("bans", [])]
    for i, b in enumerate(radiant_bans): b["order"] = i
    for i, b in enumerate(dire_bans): b["order"] = i

    radiant_team_id = first_id if radiant_is_first else second_id
    dire_team_id    = second_id if radiant_is_first else first_id

    winner = lj.get("winner")  # None / "radiant" / "dire"
    is_deactivated = bool(lj.get("is_deactivated"))
    is_live_now = (not winner and not is_deactivated)

    # series-level score (games won so far)
    scores = db.get("scores") or {}
    first_score = scores.get("first_team", 0) or 0
    second_score = scores.get("second_team", 0) or 0

    # If the match is live (draft ended, no winner), make sure the synthetic map
    # looks active even if series_meta.started_at is missing.
    map_started = series_meta.get("started_at")
    if not map_started and is_live_now:
        map_started = datetime.now(timezone.utc).isoformat()

    m = {
        "id": match_id,
        "steam_id": match_id,
        "status": 1 if is_live_now else 2,
        "started_at": map_started,
        "radiant_team_id": radiant_team_id,
        "dire_team_id": dire_team_id,
        "radiant_score": lj.get("radiant_score") or 0,
        "dire_score": lj.get("dire_score") or 0,
        "winner": winner,
        "duration": lj.get("game_time"),
        "fb": lj.get("first_blood"),
        "f10": lj.get("first_ten"),
        "radiant_picks": radiant_picks,
        "dire_picks": dire_picks,
        "radiant_bans": radiant_bans,
        "dire_bans": dire_bans,
    }

    return {
        "id": f"watch-{match_id}",
        "event_id": series_meta.get("event_id"),
        "status": 1 if is_live_now else 2,
        "type": series_meta.get("type") or 3,
        "slug": series_meta.get("slug"),
        "first_team_id": first_id,
        "second_team_id": second_id,
        "started_at": series_meta.get("started_at") or map_started,
        "ended_at": None if is_live_now else series_meta.get("started_at"),
        "first_team": first,
        "second_team": second,
        "maps": [m],
        # extra, for UI labeling
        "_watchlist": True,
        # v0.3.24h: the `/live/{id}.json` response embeds the DLTV
        # series id in `db.series.id`.  The watchlist path would
        # otherwise only know the steam match id, which means it
        # can't look up the dltv_browser cache (the publisher
        # writes the cache under the DLTV id).  Carrying it here
        # lets `_live_card` find the cache directly, even after
        # the discovery tracker has pruned the row.
        "_dltv_series_id": series_meta.get("id"),
        "_series_score_first": first_score,
        "_series_score_second": second_score,
    }


def fetch_watchlist_series(match_ids: List[int]) -> List[Dict]:
    """Fetch /live/{id}.json for each match_id and return synthetic series dicts."""
    out: List[Dict] = []
    for mid in match_ids:
        lj = client.get_live_json(mid)
        if not lj:
            continue
        try:
            s = _live_json_to_series(mid, lj)
            if s:
                out.append(s)
        except (DLTVError, ParseError, AttributeError, TypeError) as exc:
            # Watchlist tolerates per-match failure: one bad /live payload
            # must not poison the rest. AttributeError/TypeError are the
            # realistic fallout of a malformed `lj` (e.g. not a dict).
            # Other Exception types (bugs in our own code) are NOT caught
            # here so they surface in tests.
            log.warning("watchlist skip %s: %s", mid, exc, exc_info=True)
    return out


def _steam_game_to_series(game: Dict, match_id: int) -> Optional[Dict]:
    """Convert a Steam GetLiveLeagueGames entry to a v1-compatible series dict.

    Used as a fallback when DLTV /live/{id}.json doesn't cover a Steam match.
    """
    radiant_team = game.get("radiant_team") or {}
    dire_team = game.get("dire_team") or {}
    scoreboard = game.get("scoreboard") or {}

    first_id = radiant_team.get("team_id")
    second_id = dire_team.get("team_id")

    radiant_picks_raw = (scoreboard.get("radiant") or {}).get("picks") or []
    dire_picks_raw    = (scoreboard.get("dire") or {}).get("picks") or []
    radiant_bans_raw  = (scoreboard.get("radiant") or {}).get("bans") or []
    dire_bans_raw     = (scoreboard.get("dire") or {}).get("bans") or []

    def _map_pick(p: Dict) -> Dict:
        return {"hero_id": p.get("hero_id"), "order": 0, "_steam_id": p.get("hero_id")}

    radiant_picks = [_map_pick(p) for p in radiant_picks_raw]
    dire_picks    = [_map_pick(p) for p in dire_picks_raw]
    radiant_bans  = [_map_pick(p) for p in radiant_bans_raw]
    dire_bans     = [_map_pick(p) for p in dire_bans_raw]

    rad_score = (scoreboard.get("radiant") or {}).get("score") or 0
    dire_score = (scoreboard.get("dire") or {}).get("score") or 0
    duration_s = int((scoreboard.get("duration") or 0))

    first = {
        "id": first_id,
        "title": radiant_team.get("team_name") or "TBD",
        "tag": None,
        "image": None,
        "rank": None,
        "is_radiant": True,
    }
    second = {
        "id": second_id,
        "title": dire_team.get("team_name") or "TBD",
        "tag": None,
        "image": None,
        "rank": None,
        "is_radiant": False,
    }

    map_started = datetime.now(timezone.utc).isoformat()
    m = {
        "id": match_id,
        "steam_id": match_id,
        "status": 1,
        "started_at": map_started,
        "radiant_team_id": first_id,
        "dire_team_id": second_id,
        "radiant_score": rad_score,
        "dire_score": dire_score,
        "winner": None,
        "duration": duration_s if duration_s > 0 else None,
        "radiant_picks": radiant_picks,
        "dire_picks": dire_picks,
        "radiant_bans": radiant_bans,
        "dire_bans": dire_bans,
    }

    # series_type: 0=none, 1=bo3, 2=bo5 (Steam enum); map to our int (1/2/3/5)
    series_type_raw = game.get("series_type") or 0
    type_map = {0: 3, 1: 3, 2: 5}  # default to BO3 when unknown
    bo = type_map.get(series_type_raw, 3)

    return {
        "id": f"steam-{match_id}",
        "event_id": None,
        "status": 1,  # live
        "type": bo,
        "slug": None,
        "first_team_id": first_id,
        "second_team_id": second_id,
        "started_at": map_started,
        "ended_at": None,
        "first_team": first,
        "second_team": second,
        "maps": [m],
        "_watchlist": True,
        "_steam_only": True,
        "_steam_league_id": game.get("league_id"),
        "_series_score_first": game.get("radiant_series_wins") or 0,
        "_series_score_second": game.get("dire_series_wins") or 0,
    }
