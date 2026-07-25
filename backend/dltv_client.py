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
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE = "https://dltv.org/api/v1"
LIVE_BASE = "https://dltv.org/live"
SITE = "https://dltv.org"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


# --------------------------------------------------------------------------- #
# Small TTL cache
# --------------------------------------------------------------------------- #

class _TTLCache:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._exp: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._data and time.time() < self._exp.get(key, 0):
                return self._data[key]
        return None

    def set(self, key: str, value: Any, ttl: float):
        with self._lock:
            self._data[key] = value
            self._exp[key] = time.time() + ttl


def _http_json(url: str, timeout: float = 10.0) -> Optional[Any]:
    """Fetch JSON with stdlib (no extra deps beyond requests elsewhere)."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


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
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class DLTVClient:
    def __init__(self, series_ttl: float = 45.0, static_ttl: float = 900.0):
        self._cache = _TTLCache()
        self.series_ttl = series_ttl
        self.static_ttl = static_ttl
        self._hero_by_id: Dict[int, Dict] = {}
        self._hero_by_steam: Dict[int, Dict] = {}

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
        d = _http_json(f"{BASE}/events/{event_id}/series") or {}
        series = d.get("series", []) if isinstance(d, dict) else []
        self._cache.set(key, series, self.series_ttl)
        return series

    def get_live_json(self, match_id: int) -> Optional[Dict]:
        """Rich live/finished match JSON (draft + per-hero team win-rates)."""
        return _http_json(f"{LIVE_BASE}/{match_id}.json", timeout=6.0)

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
            except Exception:
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
    
    @staticmethod
    def _val(dct: Optional[Dict], key: str, fallback: float) -> float:
        """Get value from dict or fallback."""
        return dct.get(key) if dct is not None else fallback

    def hero_by_dltv_id(self, hid: Optional[int]) -> Optional[Dict]:
        if not self._hero_by_id:
            self.get_heroes()
        return self._hero_by_id.get(hid)

    def hero_by_steam_id(self, sid: Optional[int]) -> Optional[Dict]:
        if not self._hero_by_steam:
            self.get_heroes()
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
        """Classify a series, keeping it live between maps until it is complete.

        Some upstream responses mark an individual map as finished before the
        BO series is over.  Score/played-map completion therefore takes
        precedence over a stale ``status`` or ``ended_at`` field.
        """
        status = series.get("status")
        ended = _parse_dt(series.get("ended_at"))
        started = _parse_dt(series.get("started_at")) or _parse_dt(series.get("liquipedia_date"))
        now = datetime.now(timezone.utc)

        complete = self._is_series_complete(series)
        if complete:
            return "postmatch"
        # A finished first/second map still means that BO3/BO5 is live while
        # the next map is being drafted or is about to start.
        if any(m.get("winner") for m in (series.get("maps") or [])) or self._has_active_map(series):
            return "live"
        if started is not None and started <= now:
            return "live"
        # Without a usable BO/maps payload, preserve the original upstream
        # terminal signal as the fallback.
        if status == 2 or ended is not None:
            return "postmatch"
        return "prematch"

    @staticmethod
    def _series_bo(series: Dict) -> Optional[int]:
        value = series.get("type") or series.get("_scraper_bo")
        if isinstance(value, int) and value in {1, 2, 3, 5}:
            return value
        if isinstance(value, str):
            value = value.lower().strip()
            if value.startswith("bo") and value[2:].isdigit():
                parsed = int(value[2:])
                return parsed if parsed in {1, 2, 3, 5} else None
        return None

    @classmethod
    def _is_series_complete(cls, series: Dict) -> bool:
        """Return True only when the BO format's required maps are complete."""
        bo = cls._series_bo(series)
        if bo is None:
            return False
        played = [game_map for game_map in (series.get("maps") or []) if game_map.get("winner")]
        if bo == 2:
            return len(played) >= 2
        required_wins = bo // 2 + 1
        score: Dict[Any, int] = {}
        for game_map in played:
            winner_id = game_map.get("radiant_team_id") if game_map.get("winner") == "radiant" else game_map.get("dire_team_id")
            if winner_id is not None:
                score[winner_id] = score.get(winner_id, 0) + 1
        return max(score.values(), default=0) >= required_wins or len(played) >= bo

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

    def _map_pick(p: Dict) -> Dict:
        # /live picks carry hero.steam_id; we need a hero id usable via hero_by_steam_id
        hero = p.get("hero") or {}
        return {"hero_id": hero.get("steam_id"), "order": 0, "_steam_id": hero.get("steam_id")}

    def _map_ban(p: Dict) -> Dict:
        hero = p.get("hero") or {}
        return {"hero_id": hero.get("steam_id"), "order": 0, "_steam_id": hero.get("steam_id")}

    radiant_picks = [_map_pick(p) for p in first.get("picks", [])] if radiant_is_first else [_map_pick(p) for p in second.get("picks", [])]
    dire_picks    = [_map_pick(p) for p in second.get("picks", [])] if radiant_is_first else [_map_pick(p) for p in first.get("picks", [])]
    radiant_bans  = [_map_ban(p) for p in first.get("bans", [])] if radiant_is_first else [_map_ban(p) for p in second.get("bans", [])]
    dire_bans     = [_map_ban(p) for p in second.get("bans", [])] if radiant_is_first else [_map_ban(p) for p in first.get("bans", [])]

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
        except Exception as exc:
            print(f"[watchlist] skip {mid}: {exc}")
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
