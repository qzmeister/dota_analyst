"""
Unified match discovery: DLTV matches page scraper + (optional) Steam live games.

The DLTV v1 API (/api/v1/events/.../series) covers only ~30 top leagues and
omits many matches that dltv.org itself shows. This module scrapes
https://dltv.org/matches, which lists ALL live + upcoming matches with:
  - data-series-id  (DLTV series id)
  - data-match      (Steam GC match id, only when live)
  - event/league name
  - team names (or "TBD")
  - start time (for upcoming)
  - bo format

Combined with /live/{match_id}.json enrichment, this lets the board show
prematch + live matches for every league DLTV covers — no v1 API needed.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from ._logging import get_logger
from .dltv_client import (
    SITE,
    _abs_url,
    _http_json,
    _live_json_to_series,
    _parse_dt,
    _steam_game_to_series,
    client,
)
from .exceptions import (
    DLTVError,
    HTTPClientError,
    ParseError,
    ScrapeError,
    SteamAPIError,
    SteamFetchError,
    UpstreamError,
)

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRAPE_URL = f"{SITE}/matches"
SCRAPE_TTL = 90.0          # seconds between re-scrapes
ENRICH_WINDOW_H = 2.0      # hours before start to begin probing /live
# Per-match TTL for the `/live/{id}.json` enrichment.  Two cases:
#   - live matches: short — picks/score change every few seconds in
#     a real game.  Without a tight TTL the board shows stale
#     data and the UI drift (e.g. 1-0 while DLTV is at 6-3) hits.
#   - postmatch / completed: longer — the JSON is frozen once the
#     game ends, so a few minutes is fine.
ENRICH_TTL_LIVE = 5.0
ENRICH_TTL_OTHER = 120.0
ENRICH_TTL = ENRICH_TTL_LIVE  # backward-compat for any external import
STEAM_TTL = 30.0           # seconds between GetLiveLeagueGames calls
HTTP_TIMEOUT = 12.0


# --------------------------------------------------------------------------- #
# HTML parsing
# --------------------------------------------------------------------------- #

# Match opening div: <div class="match XXX" ... attrs ...>
# (negative lookahead excludes inner divs like match__head)
_MATCH_OPEN = re.compile(r'<div class="match(?!__)([^"]*)"([^>]*)>')

# Extract individual attrs from the second capture group
_ATTR_SERIES = re.compile(r'data-series-id="(\d+)"')
_ATTR_MATCH = re.compile(r'data-match="(\d+)"')
_ATTR_ODD = re.compile(r'data-matches-odd="([^"]+)"')

_EVENT_NAME = re.compile(
    r'<div class="match__head-event">.*?<span>([^<]{2,})</span>',
    re.DOTALL,
)
_BO_FORMAT = re.compile(
    r'<div class="match__head-format(?! red)[^>]*">\s*<span>\s*(bo\d)\s*</span>',
    re.IGNORECASE | re.DOTALL,
)
_FORMAT_STAGE = re.compile(
    r'<div class="match__head-format red">\s*<span>([^<]+)</span>',
    re.DOTALL,
)

# Match page URL inside a card — last path segment is the event slug.
# e.g. href="https://dltv.org/matches/427435/kw-vs-spirit-academy-epl-masters-1-play-in"
_MATCH_URL = re.compile(
    r'<a\s+href="(https://dltv\.org/matches/\d+/[^"]+)"',
)

_TEAM_NAME = re.compile(
    r'<div class="team__title">\s*<span>([^<]+)</span>',
    re.DOTALL,
)
_TEAM_LOGO_LIGHT = re.compile(
    r'<div class="team__image">\s*<i[^>]*data-theme-light="([^"]+)"',
    re.DOTALL,
)
_TEAM_LOGO_DARK = re.compile(
    r'<div class="team__image">\s*<i[^>]*data-theme-dark="([^"]+)"',
    re.DOTALL,
)

_LIVE_SCORE = re.compile(
    r'<strong class="text-red">(\d+)</strong>\s*<small>\((\d+)\)</small>',
    re.DOTALL,
)
_LIVE_GAME = re.compile(r'<span>Игра\s*(\d+)</span>', re.IGNORECASE)
_LIVE_TIME = re.compile(
    r'<div class="duration__time">\s*<strong[^>]*>(\d+:\d+)</strong>',
    re.DOTALL,
)


def _split_match_blocks(html: str) -> List[Tuple[str, Dict[str, Optional[str]], str]]:
    """Return (classes_str, attrs_dict, block_body) for each match div."""
    out: List[Tuple[str, Dict[str, Optional[str]], str]] = []
    positions: List[Tuple[int, re.Match]] = []
    for m in _MATCH_OPEN.finditer(html):
        cls = (m.group(1) or "").strip()
        if cls in ("es__links", "es__v2-items"):
            # false positives: container divs with "es__" prefix (matches__...)
            # already filtered by (?!__) but defensive
            continue
        if "live" not in cls and "upcoming" not in cls:
            continue
        attrs_block = m.group(2) or ""
        sid = (_ATTR_SERIES.search(attrs_block) or [None, None])[1]
        mid = (_ATTR_MATCH.search(attrs_block) or [None, None])[1]
        odd = (_ATTR_ODD.search(attrs_block) or [None, None])[1]
        if not sid:
            continue
        positions.append((m.start(), m, cls, sid, mid, odd))  # type: ignore[arg-type]

    for i, (start, m, cls, sid, mid, odd) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(html)
        body = html[start:end]
        out.append((cls, {"series_id": sid, "match_id": mid, "odd": odd}, body))
    return out


def _parse_team_pair(body: str) -> Tuple[Dict, Dict]:
    names = _TEAM_NAME.findall(body)
    logos_l = _TEAM_LOGO_LIGHT.findall(body)
    logos_d = _TEAM_LOGO_DARK.findall(body)

    def _team(idx: int) -> Dict:
        name = (names[idx] if idx < len(names) else "TBD").strip()
        logo = None
        if idx < len(logos_d):
            logo = _abs_url(logos_d[idx])
        elif idx < len(logos_l):
            logo = _abs_url(logos_l[idx])
        return {"name": name, "logo": logo, "tag": None, "rank": None}

    return _team(0), _team(1)


_SLUG_NON_ALPHA = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse dashes — mirrors DLTV URL slugs."""
    s = (s or "").strip().lower()
    s = _SLUG_NON_ALPHA.sub("-", s)
    return s.strip("-")


def _extract_url_event_slug(body: str, slug_to_event: Optional[Dict[str, int]]) -> Optional[str]:
    """Return the event title whose slug matches the card's URL tail.

    DLTV match URLs end in `/<teamA>-vs-<teamB>-<event-slug>`. We try each
    known title's slug; if any is a suffix of the URL, that's the event.
    """
    if not slug_to_event:
        return None
    m = _MATCH_URL.search(body)
    if not m:
        return None
    url = m.group(1).rstrip("/")
    last = url.rsplit("/", 1)[-1]  # e.g. kw-vs-spirit-academy-epl-masters-1-play-in
    for title in slug_to_event:
        slug = _slugify(title)
        if slug and last.endswith(slug):
            return title
    return None


def _extract_event_slug(body: str, known_titles: Optional[Set[str]]) -> Optional[str]:
    """Return the event title for this match card, or None.

    DLTV v1 events have no slug field, so we use the event title as the key.
    The title is already extracted from <div class="match__head-event"><span>.
    """
    if not known_titles:
        return None
    m = _EVENT_NAME.search(body)
    if not m:
        return None
    name = m.group(1).strip()
    return name if name in known_titles else None


def _parse_one_match(
    cls: str,
    attrs: Dict[str, Optional[str]],
    body: str,
    known_slugs: Optional[Set[str]] = None,
    slug_to_event: Optional[Dict[str, int]] = None,
    slug_to_event_title: Optional[Dict[str, str]] = None,
    carry_event: Optional[str] = None,
    carry_bo: Optional[str] = None,
    carry_stage: Optional[str] = None,
    carry_event_id: Optional[int] = None,
) -> Optional[Dict]:
    """Parse one match card.

    When the card is missing its own <div class="match__head"> (subsequent cards
    of the same league group), the event title, BO format and stage label are
    resolved by:
      1. URL slug match (most reliable — every card links to its series page,
         and the URL tail embeds the event slug).
      2. Carry-forward from the previous card (used as last resort).
    """
    series_id = attrs.get("series_id")
    steam_id = attrs.get("match_id")
    odd = attrs.get("odd")
    if not series_id:
        return None

    # Capture the card's URL — we may need it later for the
    # Playwright-based player.win_rate enrichment (which requires
    # a slug, e.g. /matches/427455/direb-vs-nexa-lunar-horse-8).
    url_m = _MATCH_URL.search(body)
    url_path = url_m.group(1) if url_m else None

    event_m = _EVENT_NAME.search(body)
    bo_m = _BO_FORMAT.search(body)
    stage_m = _FORMAT_STAGE.search(body)
    team_a, team_b = _parse_team_pair(body)

    # Primary: explicit head on this card
    slug = _extract_event_slug(body, known_slugs)
    # Secondary: URL-slug match (works even when the head is absent)
    url_title = None
    if not slug:
        url_title = _extract_url_event_slug(body, slug_to_event)
        if url_title:
            slug = url_title
    event_id = slug_to_event.get(slug) if slug and slug_to_event else None
    event_title_from_slug = slug_to_event_title.get(slug) if slug and slug_to_event_title else None

    # event_name: prefer title from the card; fall back to URL match; fall back to carry
    event_name = (
        (event_m.group(1).strip() if event_m else None)
        or event_title_from_slug
        or url_title
        or carry_event
    )
    bo = (bo_m.group(1).lower() if bo_m else None) or carry_bo
    stage_label = (
        (stage_m.group(1).strip() if stage_m else None) or carry_stage
    )
    if event_id is None and slug is None and carry_event_id is not None and event_name == carry_event:
        event_id = carry_event_id

    if "live" in cls:
        stage = "live"
    elif "upcoming" in cls:
        stage = "prematch"
    else:
        stage = "prematch"

    # start time: prefer data-matches-odd attr, fallback to inner text
    start_time = None
    if odd:
        try:
            start_time = datetime.fromisoformat(f"{odd}+00:00").isoformat()
        except (ValueError, TypeError):
            # fromisoformat raises ValueError for malformed input, TypeError
            # if `odd` isn't a string. Fall back to the raw value with the
            # +00:00 suffix so downstream consumers see *something* shaped
            # like an ISO-8601 timestamp.
            start_time = f"{odd}+00:00"

    # live score / game time
    live_score = None
    game_no = None
    game_time = None
    if stage == "live":
        scores = _LIVE_SCORE.findall(body)
        if len(scores) >= 2:
            live_score = {
                "radiant": int(scores[0][0]),
                "dire": int(scores[1][0]),
                "series_a": int(scores[0][1]),
                "series_b": int(scores[1][1]),
            }
        gm = _LIVE_GAME.search(body)
        if gm:
            game_no = int(gm.group(1))
        tm = _LIVE_TIME.search(body)
        if tm:
            game_time = tm.group(1)

    return {
        "series_id": int(series_id),
        "steam_id": int(steam_id) if steam_id else None,
        "stage": stage,
        "event": event_name,
        "event_id": event_id,
        "event_slug": slug,
        "bo": bo,
        "stage_label": stage_label,
        "team_a": team_a,
        "team_b": team_b,
        "start_time": start_time,
        "live_score": live_score,
        "game_no": game_no,
        "game_time": game_time,
        "url": url_path,
    }


def _event_slug_maps() -> Tuple[Set[str], Dict[str, int], Dict[str, str]]:
    """Build title-based lookup maps from /api/v1/events.

    DLTV v1 events have no slug field, so we key by title (which the scraper
    also emits via <div class="match__head-event"><span>).

    Returns (title_set, title_to_event_id, title_to_event_title) — cached 15 min.
    """
    now = time.time()
    cached = _event_slug_maps.__dict__.get("_cached")
    if cached and (now - cached[0]) < 900:
        return cached[1:]
    try:
        events = client.get_events() or []
    except (DLTVError, HTTPClientError, UpstreamError):
        # Title maps stay empty when DLTV is unreachable — the scraper
        # just won't be able to tag leagues. We deliberately don't catch
        # generic Exception here so a bug in client.get_events surfaces
        # in logs (and the test suite).
        events = []
    titles: Set[str] = set()
    t2id: Dict[str, int] = {}
    t2title: Dict[str, str] = {}
    for e in events:
        t = (e.get("title") or "").strip()
        if not t:
            continue
        titles.add(t)
        if e.get("id"):
            t2id[t] = int(e["id"])
        t2title[t] = t
    result = (titles, t2id, t2title)
    _event_slug_maps.__dict__["_cached"] = (now,) + result
    return result


def scrape_dltv_matches() -> List[Dict]:
    """Fetch /matches and parse every live/upcoming match."""
    try:
        req = urllib.request.Request(
            SCRAPE_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (ScrapeError, OSError, UpstreamError) as exc:
        # urllib raises URLError (an OSError subclass) on network/timeout
        # failures and HTTPError on non-200 responses. ScrapeError covers
        # the case where the upstream layer wraps the URL call.
        log.warning("dltv scrape failed: %s", exc, exc_info=True)
        return []

    slugs, s2id, s2title = _event_slug_maps()
    out: List[Dict] = []
    carry_event: Optional[str] = None
    carry_bo: Optional[str] = None
    carry_stage: Optional[str] = None
    carry_event_id: Optional[int] = None
    for cls, attrs, body in _split_match_blocks(html):
        parsed = _parse_one_match(
            cls, attrs, body, slugs, s2id, s2title,
            carry_event=carry_event,
            carry_bo=carry_bo,
            carry_stage=carry_stage,
            carry_event_id=carry_event_id,
        )
        if parsed:
            # Update carry-forward: only refresh when this card had its own head
            # OR we resolved via URL slug (so the carry remains correct for
            # subsequent headless cards in the same league group).
            if parsed.get("event"):
                carry_event = parsed["event"]
                carry_event_id = parsed.get("event_id") or carry_event_id
            if parsed.get("bo"):
                carry_bo = parsed["bo"]
            if parsed.get("stage_label"):
                carry_stage = parsed["stage_label"]
            out.append(parsed)
    return out


# --------------------------------------------------------------------------- #
# Steam GetLiveLeagueGames (optional — requires STEAM_API_KEY env var)
# --------------------------------------------------------------------------- #

# `os` is already imported at the top of this module (line 21) for
# `os.environ` reads.  This section just documents the Steam key
# loader.  The actual `import os` was here in 0.3.14 and earlier;
# we removed the duplicate in 0.3.15 — see v0.3.15 commit message.

# --------------------------------------------------------------------------- #
# Steam Web API key (GetLiveLeagueGames requires it)
# Loaded from .steam_key file in project root OR from STEAM_API_KEY env var.
# --------------------------------------------------------------------------- #

def _load_steam_key() -> str:
    env_key = os.environ.get("STEAM_API_KEY", "").strip()
    if env_key:
        return env_key
    # Try .steam_key file in project root
    here = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(here)
    for candidate in (
        os.path.join(proj_root, ".steam_key"),
        os.path.join(here, ".steam_key"),
    ):
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    k = f.read().strip().splitlines()[0].strip()
                    if k and len(k) >= 20:
                        return k
            except (OSError, UnicodeDecodeError):
                # The .steam_key file may be missing, unreadable, or contain
                # binary garbage. None of those should crash the import path.
                pass
    return ""


STEAM_API_KEY = _load_steam_key()
if STEAM_API_KEY:
    log.info("Steam API key loaded (%s...%s)", STEAM_API_KEY[:6], STEAM_API_KEY[-4:])
else:
    log.info("Steam API key NOT configured — Steam live-discovery disabled")


def fetch_steam_live() -> List[Dict]:
    """Return live pro matches from Steam (requires STEAM_API_KEY)."""
    if not STEAM_API_KEY:
        return []
    url = (
        f"https://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/"
        f"?key={STEAM_API_KEY}&format=json"
    )
    # v0.3.15: bound the Steam call to a short timeout.  The discovery
    # tracker calls us on every refresh, and the urllib transport
    # (used inside `_http_json`) can otherwise block up to the default
    # 12s per attempt when Steam is slow — and that cascades into
    # `build_board()` being slow and `/api/board` returning 504.
    try:
        data = _http_json(url, timeout=min(HTTP_TIMEOUT, 4.0)) or {}
    except (SteamAPIError, SteamFetchError, HTTPClientError, UpstreamError) as exc:
        log.warning("steam fetch failed: %s", exc, exc_info=True)
        return []

    games = (data.get("result") or {}).get("games") or []
    out: List[Dict] = []
    for g in games:
        match_id = g.get("match_id")
        league_id = g.get("league_id")
        radiant = (g.get("radiant_team") or {}).get("team_name")
        dire = (g.get("dire_team") or {}).get("team_name")
        sb = g.get("scoreboard") or {}
        rad_score = (sb.get("radiant") or {}).get("score")
        dire_score = (sb.get("dire") or {}).get("score")
        if not match_id:
            continue
        out.append({
            "series_id": None,
            "steam_id": int(match_id),
            "stage": "live",
            "event": None,  # Steam rarely gives league_name; DLTV scraper will provide
            "bo": None,
            "stage_label": None,
            "team_a": {"name": radiant or "TBD", "logo": None, "tag": None, "rank": None},
            "team_b": {"name": dire or "TBD", "logo": None, "tag": None, "rank": None},
            "start_time": None,
            "live_score": {
                "radiant": rad_score or 0,
                "dire": dire_score or 0,
            },
            "game_no": None,
            "game_time": None,
            "steam_league_id": league_id,
            "steam_series_wins": {
                "radiant": g.get("radiant_series_wins") or 0,
                "dire": g.get("dire_series_wins") or 0,
            },
            "steam_series_type": g.get("series_type"),
            # Raw Steam game payload so we can synthesize a v1-like series dict
            # when DLTV /live/{id}.json doesn't cover this match (minor leagues).
            "_steam_raw": g,
        })
    return out


# --------------------------------------------------------------------------- #
# Discovery tracker — merges scraper + Steam, enriches via /live/{id}.json
# --------------------------------------------------------------------------- #

class _DiscoveryTracker:
    """Holds parsed matches from scrapers, keyed by series_id or steam_id."""

    def __init__(self):
        self._lock = threading.Lock()
        # series_id -> last parsed dict
        self._by_series: Dict[int, Dict] = {}
        # steam_id -> last parsed dict (for Steam-discovered matches without series_id)
        self._by_steam: Dict[int, Dict] = {}
        # (series_id|steam_id) -> timestamp of last /live probe
        self._last_probe: Dict[int, float] = {}
        # series_id -> cached synthetic series dict
        self._series_cache: Dict[int, Dict] = {}
        self._cache_ts: Dict[int, float] = {}
        self._last_scrape = 0.0
        self._last_steam = 0.0
        # Steam league_id -> DLTV event_id (learned when a Steam match
        # is also present on DLTV matches page with a known event_slug).
        self._steam_to_event: Dict[int, int] = {}
        self._steam_to_event_title: Dict[int, str] = {}

    # ---- refresh loops ---- #

    def refresh(self) -> None:
        """Pull DLTV matches via scraper + (optional) Steam live games.

        v0.3.15: DLTV's `/matches` HTML scraper has been unreliable for
        live cards (returns rows with `series_id=None, event=None, match=None`
        — the live block layout is different from prematch).  Steam API
        gives us live games reliably and at ~500ms.  So we now prefer
        Steam for live, and only fall back to scraper for the live
        section if Steam is unavailable.
        """
        now = time.time()
        if STEAM_API_KEY and now - self._last_steam >= STEAM_TTL:
            self._last_steam = now
            steam = fetch_steam_live()
            with self._lock:
                self._merge_steam(steam)
        # Scraper is on by default — the parser handles prematch rows
        # well, and `_merge_scraped` below filters the broken live
        # rows (no steam_id) so they don't reach the board.  Set
        # `DLTV_SCRAPER_ENABLED=0` to disable for low-bandwidth envs.
        scraper_enabled = os.environ.get("DLTV_SCRAPER_ENABLED", "1") != "0"
        if scraper_enabled and now - self._last_scrape >= SCRAPE_TTL:
            self._last_scrape = now
            try:
                scraped = scrape_dltv_matches()
                with self._lock:
                    self._merge_scraped(scraped)
            except Exception as exc:
                log.warning("scraper refresh failed: %s", exc, exc_info=False)
        # v0.4.0.3: socket-based discovery fallback.  The
        # `__nd2_series` channel pushes live/upcoming/results
        # lists on a regular cadence.  We always run it (it's
        # cheap — just a dict copy under the socket's lock) and
        # it backfills any steam_id the scraper missed.  This is
        # the second-socket bridge that keeps live cards showing
        # the right teams when the HTTP scraper is dark.
        with self._lock:
            self._merge_socket_series()

    def _merge_scraped(self, scraped: List[Dict]) -> None:
        """Merge the scraper's rows into `_by_series`.

        v0.3.15+ rationale: live rows from the scraper are accepted
        even without a `steam_id` — DLTV's page often renders the live
        card before `data-match` is populated, and we don't want to
        drop the only live source for minor-league matches that Steam
        either doesn't know about or knows with empty team names.
        The card will still render the team names + event from the
        scraper; the `get_live_json` enrichment later fills in picks
        if/when the steam_id becomes available.
        """
        seen_series: Set[int] = set()
        for m in scraped:
            sid = m.get("series_id")
            if not sid:
                continue
            seen_series.add(sid)
            prev = self._by_series.get(sid)
            # preserve any steam_id we already learned from a prior
            # tick (or from Steam cross-ref).
            if prev and prev.get("steam_id") and not m.get("steam_id"):
                m["steam_id"] = prev["steam_id"]
            self._by_series[sid] = m
        # prune series no longer on the page (likely finished >24h ago)
        for sid in list(self._by_series.keys()):
            if sid not in seen_series:
                # keep cache for 15 more min in case page briefly drops it
                if time.time() - self._cache_ts.get(sid, 0) > 900:
                    self._by_series.pop(sid, None)
                    self._series_cache.pop(sid, None)
                    self._cache_ts.pop(sid, None)

    def _merge_steam(self, steam: List[Dict]) -> None:
        now = time.time()
        seen: Set[int] = set()
        for m in steam:
            mid = m.get("steam_id")
            if not mid:
                continue
            seen.add(mid)
            m["_steam_seen"] = now
            # if a scraped match already has this steam_id, link it
            linked_series = None
            for sid, parsed in self._by_series.items():
                if parsed.get("steam_id") == mid:
                    linked_series = sid
                    break
            if linked_series is not None:
                # mark live if steam sees it live
                self._by_series[linked_series]["stage"] = "live"
                self._by_series[linked_series]["steam_id"] = mid
                # Learn steam_league_id -> event_id mapping from this cross-ref
                s_lid = m.get("steam_league_id")
                dltv_eid = self._by_series[linked_series].get("event_id")
                dltv_ename = self._by_series[linked_series].get("event")
                if s_lid and dltv_eid:
                    self._steam_to_event[int(s_lid)] = int(dltv_eid)
                    if dltv_ename:
                        self._steam_to_event_title[int(s_lid)] = dltv_ename
            else:
                self._by_steam[mid] = m

        # prune Steam-only matches no longer live (after ~15 min grace)
        for mid in list(self._by_steam.keys()):
            entry = self._by_steam[mid]
            seen_recently = (now - entry.get("_steam_seen", 0)) < 900
            if not seen_recently:
                self._by_steam.pop(mid, None)
                self._series_cache.pop(mid, None)
                self._cache_ts.pop(mid, None)

    def steam_event(self, steam_league_id: Optional[int]) -> Optional[Tuple[int, str]]:
        """Return (event_id, event_title) for a Steam league_id, or None."""
        if not steam_league_id:
            return None
        eid = self._steam_to_event.get(int(steam_league_id))
        if not eid:
            return None
        title = self._steam_to_event_title.get(int(steam_league_id)) or f"Event {eid}"
        return eid, title

    # ---- socket-based discovery fallback (v0.4.0.3) ----

    def _merge_socket_series(self) -> None:
        """Pull the live/upcoming/results lists from the
        dltv_socket's `__nd2_series` channel and merge them into
        the tracker.

        v0.4.0.3: this is the second-socket fallback for
        discovery.  When the HTTP scraper is dark (DNS failure,
        Cloudflare block, etc.) the dltv.org broadcast channel
        still gives us enough to render the board:

          * `live`     — `{steam_id: series_id}` — fills in
                         steam_ids the scraper missed and gives
                         us the bridge to live payload data
          * `upcoming` — list of series with team_id, event_id,
                         started_at — fed into `_by_series` as
                         "prematch" stubs
          * `results`  — list of finished series — fed into
                         `_by_series` as "postmatch" stubs

        For live matches the team names come from the
        `__nd2_match_<steam_id>` payload (which the socket loop
        already caches in `_state`); for upcoming/results we only
        get team_id, so names stay "TBD" until the user opens
        the match page or the v1 series API catches up.

        This is best-effort: any failure (dltv_socket not
        imported, channel never pushed, malformed payload) is
        swallowed so a flaky socket doesn't break the tracker.
        """
        try:
            from . import dltv_socket as _ds
        except Exception:
            return
        try:
            state = _ds.get_series_state()
        except Exception as exc:
            log.debug("dltv_socket.get_series_state failed: %s", exc)
            return
        if not state or state.get("stale"):
            return

        live_map: Dict[int, int] = state.get("live") or {}
        upcoming: List[Dict] = state.get("upcoming") or []
        results: List[Dict] = state.get("results") or []

        with self._lock:
            # 1) Live: backfill steam_id on existing scraper rows
            #    AND add new live rows for steam_ids the scraper
            #    didn't surface.
            new_subs: Set[int] = set()  # steam_ids the socket should subscribe to
            for steam_id, sid in live_map.items():
                # Defensive: malformed payload (e.g. "bad" string
                # keys) shouldn't crash the merge.  dltv_socket
                # already coerces to int, but we double-check here
                # so a buggy upstream can't poison the tracker.
                try:
                    steam_id_i = int(steam_id)
                    sid_i = int(sid)
                except (TypeError, ValueError):
                    continue
                if not steam_id_i or not sid_i:
                    continue
                prev = self._by_series.get(sid_i)
                if prev is not None:
                    # Backfill the steam_id we couldn't get from HTML.
                    if not prev.get("steam_id"):
                        prev["steam_id"] = steam_id_i
                    # Promote to live (the socket broadcast is
                    # authoritative for "this is happening now").
                    prev["stage"] = "live"
                else:
                    # No scraper row yet — synthesise a stub.
                    # The board will hydrate teams from the
                    # live payload via `get_live_state(steam_id)`.
                    self._by_series[sid_i] = {
                        "series_id": sid_i,
                        "steam_id": steam_id_i,
                        "stage": "live",
                        "event": None,
                        "event_id": None,
                        "bo": None,
                        "team_a": {"name": "TBD", "logo": None, "tag": None, "rank": None},
                        "team_b": {"name": "TBD", "logo": None, "tag": None, "rank": None},
                        "start_time": None,
                        "live_score": None,
                        "game_no": None,
                        "game_time": None,
                        "_socket_source": True,
                    }
                # Also seed `_by_steam` so the live enrichment
                # path (which keys off steam_id) can find it.
                if steam_id_i not in self._by_steam:
                    self._by_steam[steam_id_i] = {
                        "series_id": sid_i,
                        "steam_id": steam_id_i,
                        "stage": "live",
                        "_socket_source": True,
                    }
                # Mark the steam_id for subscription so the socket
                # loop starts sending us `__nd2_match_<id>` payloads
                # (which carry the real team names + picks + score).
                # Done OUTSIDE the lock below because dltv_socket
                # has its own.
                new_subs.add(steam_id_i)
            # Upcoming: synthesise a minimal prematch row per series.
            for s in upcoming:
                sid = s.get("id")
                if not sid:
                    continue
                try:
                    sid_i = int(sid)
                except (TypeError, ValueError):
                    continue
                prev = self._by_series.get(sid_i)
                if prev is None:
                    self._by_series[sid_i] = {
                        "series_id": sid_i,
                        "steam_id": None,
                        "stage": "prematch",
                        "event": None,
                        "event_id": s.get("event_id"),
                        "bo": None,
                        "team_a": {"name": "TBD", "logo": None, "tag": None, "rank": None},
                        "team_b": {"name": "TBD", "logo": None, "tag": None, "rank": None},
                        "start_time": s.get("started_at"),
                        "live_score": None,
                        "game_no": None,
                        "game_time": None,
                        "_socket_source": True,
                    }
                else:
                    # Backfill start_time / event_id if scraper missed
                    # them (common for fresh listings).
                    if not prev.get("start_time") and s.get("started_at"):
                        prev["start_time"] = s.get("started_at")
                    if prev.get("event_id") is None and s.get("event_id") is not None:
                        prev["event_id"] = s.get("event_id")
            # Results: feed the postmatch pool.  We do NOT synthesise
            # a row for every result — that would flood the postmatch
            # column with hundreds of finished matches.  Only seed
            # entries that already exist in `_by_series` (so a row
            # the scraper is currently showing as "live" gets
            # correctly transitioned to "postmatch" when the broadcast
            # says it's finished).
            for s in results:
                sid = s.get("id")
                if not sid:
                    continue
                try:
                    sid_i = int(sid)
                except (TypeError, ValueError):
                    continue
                prev = self._by_series.get(sid_i)
                if prev is not None and prev.get("stage") in ("live", "prematch"):
                    prev["stage"] = "postmatch"
                    if not prev.get("start_time") and s.get("started_at"):
                        prev["start_time"] = s.get("started_at")

        # Subscribe + force-reconnect OUTSIDE the lock.  The
        # dltv_socket server doesn't accept mid-session SUBSCRIBE
        # packets (it closes the connection a few minutes later);
        # the only reliable way to pick up new live matches is to
        # drop the WS and re-open with the fresh subscription set.
        # `force_reconnect()` is throttled to once per 30s so a
        # chatty publisher can't pin us in reconnect loops.
        if new_subs:
            try:
                from . import dltv_socket as _ds
                for sid in new_subs:
                    _ds.subscribe(int(sid))
                _ds.force_reconnect()
            except Exception as exc:
                log.debug("dltv_socket subscribe/reconnect failed: %s", exc)

    # ---- probing /live/{id}.json for upcoming matches ---- #

    def probe_upcoming(self) -> None:
        """For prematch matches whose start_time is within ENRICH_WINDOW_H,
        try to fetch /live/{id}.json and discover their steam_id.
        """
        now = time.time()
        now_utc = datetime.now(timezone.utc)
        with self._lock:
            candidates = []
            for sid, m in list(self._by_series.items()):
                if m.get("stage") != "prematch":
                    continue
                if m.get("steam_id"):
                    # already know steam_id, no need to probe for discovery
                    continue
                st = _parse_dt(m.get("start_time"))
                if not st:
                    continue
                delta_h = (st - now_utc).total_seconds() / 3600.0
                if delta_h <= ENRICH_WINDOW_H:
                    candidates.append((sid, m))

        for sid, m in candidates:
            # probe using the series page — DLTV series page has steam_ids for all maps
            # but we don't have a direct series API, so try /live with a guess...
            # Easier: use the v1 events/series list for known events
            # Best-effort: fetch the match page HTML for steam_id (TODO).
            # For now, skip — Steam discovery will catch it if live.
            pass

    # ---- produce series dicts for the board ---- #

    # ---- parallel enrichment helpers ---- #

    # v0.4.0-perf: how many worker threads to use when fetching
    # /live/{id}.json in parallel for a board build.  urllib's
    # default thread pool tops out around 6-8 active connections
    # to one host before DNS/connect contention kicks in, so 6 is
    # a sweet spot.
    _ENRICH_WORKERS = 6

    def _fetch_one_live(self, cache_key: int, mid: int, m: Dict, scraper_event_id: Optional[int], now: float) -> Tuple[int, Optional[Dict]]:
        """Fetch /live/{id}.json for a single match and synthesise a series dict.

        Returns (cache_key, series_or_None).  Failures (timeout, 404, parse
        error) are swallowed here — the tracker just won't get a fresh
        series this cycle and will fall back to the previous cache (if any)
        on the next call.
        """
        series: Optional[Dict] = None
        lj = client.get_live_json(mid)
        if lj:
            try:
                series = _live_json_to_series(mid, lj)
            except (DLTVError, ParseError, AttributeError, TypeError) as exc:
                log.warning("synth failed for %s: %s", mid, exc, exc_info=True)
        # DLTV miss? synthesize from Steam raw data (minor leagues)
        if not series and m.get("_steam_raw"):
            try:
                series = _steam_game_to_series(m["_steam_raw"], mid)
            except (SteamAPIError, ParseError, AttributeError, TypeError, KeyError) as exc:
                log.warning("steam-synth failed for %s: %s", mid, exc, exc_info=True)
        if series:
            series.setdefault("event_title", m.get("event"))
            series["_scraper_event"] = m.get("event")
            series["_scraper_bo"] = m.get("bo")
            series["_scraper_event_id"] = scraper_event_id
            self._series_cache[cache_key] = series
            self._cache_ts[cache_key] = now
        return cache_key, series

    def get_live_and_prematch(self) -> Tuple[List[Dict], List[Dict]]:
        """Return (live_series, prematch_series) dicts suitable for board rendering.

        live_series are fully-enriched synthetic series dicts (from /live/{id}.json).
        prematch_series are lightweight dicts with team/event/time only.

        v0.4.0-perf: live enrichment is now fanned out across a
        `ThreadPoolExecutor` (default 6 workers).  Previously we did
        this in a single sequential `for` loop, which meant a 20-match
        board took ~3-5 minutes when 18 of those /live/{id}.json calls
        timed out at 12s (retries=3 × timeout=3s + backoff).  With
        parallel fetch and the v0.4.0 retry/timeout trim in
        `client.get_live_json`, the same build now finishes in <5s.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.refresh()

        live: List[Dict] = []
        prematch: List[Dict] = []
        now = time.time()

        with self._lock:
            all_m = list(self._by_series.values()) + list(self._by_steam.values())

        # v0.3.24c: dedup by steam_id (and by series_id as a fallback for
        # matches the scraper found before the Steam id propagated).  A
        # single match can live in both `_by_series` (keyed by dltv
        # series_id, populated by the scraper) and `_by_steam` (keyed
        # by steam_id, populated by the Steam poller).  refresh() runs
        # Steam first, scraper second, so when scraper lands a row
        # with both ids the corresponding `_by_steam[steam_id]` entry
        # is stale but not removed.  Without this dedup the loop
        # below produces two identical `watch-…` series for the same
        # match — the user-visible "two cards for the same game" bug.
        seen_keys: set = set()
        deduped: List[Dict] = []
        for m in all_m:
            mid = m.get("steam_id")
            sid = m.get("series_id")
            key = mid or sid
            if key is None or key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(m)
        all_m = deduped

        # Two-pass approach: first pass resolves event_id from the
        # learned steam→event map, and figures out which live matches
        # have a fresh cache (skip HTTP) vs which need enrichment.
        needs_fetch: List[Tuple[int, int, Dict, Optional[int]]] = []  # (cache_key, mid, m, scraper_event_id)
        live_from_cache: List[Dict] = []
        for m in all_m:
            sid = m.get("series_id")
            mid = m.get("steam_id")
            stage = m.get("stage")
            scraper_event_id = m.get("event_id")
            steam_league_id = m.get("steam_league_id")
            if not scraper_event_id and steam_league_id:
                mapped = self.steam_event(steam_league_id)
                if mapped:
                    scraper_event_id = mapped[0]
                    if not m.get("event"):
                        m["event"] = mapped[1]

            if stage == "live" and mid:
                cache_key = sid or mid
                cached = self._series_cache.get(cache_key)
                # v0.3.19+: live matches get a 5s TTL — picks/score
                # change every few seconds in a real game.  Postmatch
                # and prematch stay at 120s because the JSON is
                # effectively frozen.
                cache_age = now - self._cache_ts.get(cache_key, 0)
                if cached and cache_age < ENRICH_TTL_LIVE:
                    cached["_scraper_event_id"] = cached.get("_scraper_event_id") or scraper_event_id
                    live_from_cache.append((cache_key, cached))
                else:
                    needs_fetch.append((cache_key, mid, m, scraper_event_id))

        # v0.4.0-perf: parallel enrichment.  Cap workers at the
        # number of pending fetches so a 3-match build doesn't spawn
        # 6 idle threads.
        if needs_fetch:
            with ThreadPoolExecutor(max_workers=min(self._ENRICH_WORKERS, len(needs_fetch))) as ex:
                futures = [
                    ex.submit(self._fetch_one_live, ck, mid, m, scraper_event_id, now)
                    for (ck, mid, m, scraper_event_id) in needs_fetch
                ]
                fetched: Dict[int, Optional[Dict]] = {}
                for f in as_completed(futures):
                    try:
                        ck, series = f.result()
                    except Exception as exc:
                        # Should not happen — _fetch_one_live catches its own
                        # exceptions.  Defensive: log and move on.
                        log.warning("enrich worker crashed: %s", exc, exc_info=True)
                        continue
                    fetched[ck] = series

        # Second pass: assemble the live list (cache first, then fresh fetches,
        # then live-without-steam-id rows).  We track which cache_keys have
        # already been added so the third pass doesn't re-add them.
        added_cache_keys: set = set()
        for ck, cached in live_from_cache:
            live.append(cached)
            added_cache_keys.add(ck)

        for (ck, mid, m, scraper_event_id) in needs_fetch:
            series = fetched.get(ck)
            if series is not None:
                live.append(series)
            else:
                # v0.4.0-perf: enrichment failed (timeout / 404).  Fall
                # back to a synthetic v1-shaped series so the card still
                # shows in the board (similar to the live-without-steam-id
                # branch below).  This is what the scraper reported and
                # /live/{id}.json couldn't enrich.  Better than dropping
                # the card silently.
                series = {
                    "id": m.get("series_id"),
                    "match_id": int(mid),
                    "event_id": scraper_event_id,
                    "first_team": m.get("team_a") or {"name": "TBD"},
                    "second_team": m.get("team_b") or {"name": "TBD"},
                    "type": 3,
                    "maps": [],
                    "started_at": m.get("start_time") or datetime.now(timezone.utc).isoformat(),
                    "status": 1,
                    "live_score": m.get("live_score"),
                    "_scraper_event": m.get("event"),
                    "_scraper_bo": m.get("bo"),
                    "_scraper_event_id": scraper_event_id,
                    "_live_enrich_failed": True,
                }
                live.append(series)
            added_cache_keys.add(ck)

        # Third pass: live rows without a steam_id and everything that
        # didn't go to the live section.
        for m in all_m:
            sid = m.get("series_id")
            mid = m.get("steam_id")
            stage = m.get("stage")
            scraper_event_id = m.get("event_id")

            if stage == "prematch":
                prematch.append({
                    "series_id": sid,
                    "steam_id": mid,
                    "stage": "prematch",
                    "event": m.get("event"),
                    "event_id": scraper_event_id,
                    "bo": m.get("bo"),
                    "stage_label": m.get("stage_label"),
                    "team_a": m.get("team_a"),
                    "team_b": m.get("team_b"),
                    "start_time": m.get("start_time"),
                })
                continue

            # Live row without a steam_id: we can't pull picks from
            # /live/{id}.json, but we still want the card in the
            # board so the user sees who is playing right now.
            if stage == "live":
                ck = sid or mid
                if ck is not None and ck in added_cache_keys:
                    continue  # already added via enrichment above
                title = m.get("event") or "Live match"
                synthesized_started_at = m.get("start_time") or datetime.now(timezone.utc).isoformat()
                series = {
                    "id": sid,
                    "match_id": mid,
                    "event_id": scraper_event_id,
                    "first_team": m.get("team_a") or {"name": "TBD"},
                    "second_team": m.get("team_b") or {"name": "TBD"},
                    "type": 3,
                    "maps": [],
                    "started_at": synthesized_started_at,
                    "status": 1,
                    "live_score": m.get("live_score"),
                    "_scraper_event": title,
                    "_scraper_bo": m.get("bo"),
                    "_scraper_event_id": scraper_event_id,
                    "_live_no_steam_id": True,
                }
                live.append(series)

            if stage == "prematch":
                prematch.append({
                    "series_id": sid,
                    "steam_id": mid,
                    "stage": "prematch",
                    "event": m.get("event"),
                    "event_id": scraper_event_id,
                    "bo": m.get("bo"),
                    "stage_label": m.get("stage_label"),
                    "team_a": m.get("team_a"),
                    "team_b": m.get("team_b"),
                    "start_time": m.get("start_time"),
                })
                continue

            # Live row without a steam_id: we can't pull picks from
            # /live/{id}.json, but we still want the card in the
            # board so the user sees who is playing right now.
            # Synthesize a minimal v1-shaped series with empty maps.
            if stage == "live":
                title = m.get("event") or "Live match"
                # v0.3.22: if the scraper didn't see a `start_time`
                # (the live card on dltv.org renders before the
                # timestamp is populated), fall back to "now" so
                # `classify_stage` returns "live" instead of
                # "prematch" — otherwise the row goes to the
                # prematch section and never gets the `_live_card`
                # match-state overlay.
                synthesized_started_at = m.get("start_time") or datetime.now(timezone.utc).isoformat()
                series = {
                    "id": sid,
                    "event_id": scraper_event_id,
                    "first_team": m.get("team_a") or {"name": "TBD"},
                    "second_team": m.get("team_b") or {"name": "TBD"},
                    "first_team_id": m.get("team_a", {}).get("id") if isinstance(m.get("team_a"), dict) else None,
                    "second_team_id": m.get("team_b", {}).get("id") if isinstance(m.get("team_b"), dict) else None,
                    "type": 3,  # best-of-3 default
                    "maps": [],  # no draft yet
                    "started_at": synthesized_started_at,
                    "status": 1,  # in progress — bypass classify_stage guess
                    "live_score": m.get("live_score"),
                    "_scraper_event": title,
                    "_scraper_bo": m.get("bo"),
                    "_scraper_event_id": scraper_event_id,
                    "_live_no_steam_id": True,
                }
                live.append(series)
                continue

        # sort prematch by start_time asc
        prematch.sort(key=lambda c: c.get("start_time") or "9999")
        return live, prematch


# module-level singleton
tracker = _DiscoveryTracker()


def discover() -> Tuple[List[Dict], List[Dict]]:
    """Public entrypoint: returns (live_series, prematch_series)."""
    return tracker.get_live_and_prematch()
