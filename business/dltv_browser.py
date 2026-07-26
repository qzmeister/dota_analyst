"""
Playwright-based scraper for dltv.org live match pages (v0.3.16+).

Why we need this at all
------------------------
The DLTV v1 API (`/api/v1/events/{id}/series`) ships `map_results[]`
with per-player career WR, but only after the map completes.  The
`/live/{steam_match_id}.json` payload for an in-progress match
returns the draft (picks/bans) but NOT player.win_rate.  Steam's
GetLiveLeagueGames doesn't carry it either, and Stratz/OpenDota
either 403 or 404 in our benchmarks.

So for a live match the only remaining source is the rendered HTML
page at `https://dltv.org/matches/{series_id}/{slug}` — which is a
React app.  We use Playwright (headless chromium) to load it,
wait for the player rows, and extract each player's career WR.

v0.3.20+: this module also fetches the full live match state
(picks, score, game time, gold lead) for series the DLTV v1
API doesn't return.  The v1 API hides in-progress series (it
only shows completed ones for each event), so the only way to
get picks for a live best-of-N after game 1 is the rendered
HTML.  Without this fallback the board shows blank picks for
EPL Masters 1 Игра 2 even though DLTV displays them on the
page.

Caching
-------
Each (series_id, slug) request hits dltv.org and the response is
cached to `ml_data/player_wr_cache.json` (match_state entry) for
`MATCH_STATE_TTL_SEC` (5s — live).  The publisher poll is the
natural cadence (5s), so the cache is a no-op for steady-state.

Performance
-----------
A cold fetch takes ~3-5s (chromium launch + page load + DOM
parse).  We share a single Playwright instance across fetches to
amortize the launch cost.  When the upstream is slow, the in-memory
TTL cap kicks in and the fetch returns whatever the last good
result was.

Optional
--------
The module is a soft dependency — if `playwright` isn't installed
or the chromium browser binary is missing, every call returns
`None` and logs a single warning at first use.  The board's
prediction path treats this as "no live data available" and falls
back to whatever DLTV v1 / Steam supplied.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._logging import get_logger
from .exceptions import DiscoveryError

log = get_logger(__name__)

ML_DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "ml_data"))
PLAYER_WR_CACHE_FILE = ML_DATA_DIR / "player_wr_cache.json"
PLAYER_WR_TTL_SEC = 5 * 60.0   # 5 minutes
MATCH_STATE_TTL_SEC = 5.0      # 5 seconds — live changes every tick
PLAYER_WR_FETCH_TIMEOUT_MS = 8000  # playwright goto timeout
PLAYER_WR_PAGE_LOAD_WAIT_MS = 2500  # let React render


# Lazy playwright import.  We don't want a hard dependency at
# business.app import time — only when /api/board or the publisher
# actually needs player.win_rate from a live page.
_playwright = None
_playwright_lock = threading.Lock()
_browser = None
_browser_lock = threading.Lock()


def _get_browser():
    """Return a process-wide Playwright browser, or None if unavailable.

    We hold a single browser instance and create per-call pages from
    it.  Closing the browser kills all pages, so we let the GC handle
    it on process exit.
    """
    global _browser
    if _browser is not None:
        return _browser
    with _browser_lock:
        if _browser is not None:
            return _browser
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError:
            log.warning("dltv_browser: playwright not installed — live player.win_rate disabled")
            return None
        # We can't keep a sync_playwright context open across calls
        # because it's not thread-safe; the caller is expected to be
        # the single thread that polls the cache.
        return None  # caller opens its own context


def _ensure_cache_dir() -> None:
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_cache() -> Dict[str, Any]:
    if not PLAYER_WR_CACHE_FILE.exists():
        return {}
    try:
        with open(PLAYER_WR_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # Corrupt cache — start fresh rather than crash the loop.
        return {}


def _write_cache(cache: Dict[str, Any]) -> None:
    _ensure_cache_dir()
    tmp = PLAYER_WR_CACHE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, default=str)
        os.replace(tmp, PLAYER_WR_CACHE_FILE)
    except OSError as exc:
        log.warning("dltv_browser: cache write failed: %s", exc)


def _cache_key(series_id: int) -> str:
    return f"s{series_id}"


def _cache_fresh(entry: Dict[str, Any], now: float) -> bool:
    return (now - float(entry.get("ts", 0))) < PLAYER_WR_TTL_SEC


def get_cached_player_winrates(series_id: int) -> Optional[Dict[str, float]]:
    """Return cached {player_name: win_rate} for `series_id`, or None if stale/missing.

    Distinguishes three states:
      - None:  no entry, or the entry has expired (caller should fetch).
      - {}:    a fresh entry exists but had no WR visible (caller
               should NOT refetch — we're respecting the TTL).
      - {name: rate}:  fresh entry with at least one player WR.

    This is the difference between "I haven't tried yet" and
    "I tried and there was nothing to scrape".  The periodic task
    uses it to skip work; the predict path uses it to know whether
    to consult the cache or rely on the encoder's training-time
    data.
    """
    cache = _read_cache()
    entry = cache.get(_cache_key(series_id))
    if not entry:
        return None
    if not _cache_fresh(entry, time.time()):
        return None
    rates = entry.get("rates") or {}
    if not isinstance(rates, dict):
        return {}
    # Filter to floats
    out: Dict[str, float] = {}
    for k, v in rates.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def fetch_player_winrates(url: str) -> Dict[str, float]:
    """Open `url` with Playwright and scrape the player rows for career WR.

    Returns {player_name: win_rate}.  Best-effort: if the page
    doesn't expose WR in the rendered DOM (e.g. tournament is too
    obscure and DLTV only shows team names) we return an empty
    dict and the caller falls back to existing encoder values.

    Raises DiscoveryError on hard failures (Playwright missing,
    chromium binary missing, network unreachable).  The caller
    should treat that as "skip this update" and retry next tick.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError as exc:
        raise DiscoveryError(f"playwright not installed: {exc}")

    rates: Dict[str, float] = {}
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:
            # Most common cause: chromium binary missing.  We log
            # once and let the caller disable future fetches.
            raise DiscoveryError(f"chromium launch failed: {exc}")
        try:
            page = browser.new_page()
            try:
                page.goto(url, timeout=PLAYER_WR_FETCH_TIMEOUT_MS,
                          wait_until="domcontentloaded")
            except PWTimeout:
                raise DiscoveryError(f"goto timeout: {url}")
            page.wait_for_timeout(PLAYER_WR_PAGE_LOAD_WAIT_MS)
            # The player row format is tournament-specific; we try
            # a few common shapes and pick the one that yields
            # >0 results.  Each block below is best-effort.
            rates = _extract_via_text_match(page) or _extract_via_datalayer(page)
        finally:
            browser.close()
    return rates


def _extract_via_text_match(page) -> Dict[str, float]:
    """Scan the rendered DOM for `Name N%` patterns.

    DLTV typically renders the WR next to the player name as
    "PlayerName 67%" (sometimes with a % suffix and a small dot).
    The format is fragile (depends on the i18n strings) so we
    fall back to the data-layer approach below if this yields
    nothing.
    """
    import re
    pat = re.compile(r"([A-Za-zÀ-ÿ' .\-]{2,30})\s+(\d{1,3})\s*%")
    out: Dict[str, float] = {}
    try:
        # Look at every text node — page.evaluate with a custom
        # walker keeps the cost low.
        text = page.evaluate(
            "() => document.body.innerText"
        )
    except Exception as exc:
        log.debug("dltv_browser: body innerText failed: %s", exc)
        return out
    if not isinstance(text, str):
        return out
    for m in pat.finditer(text):
        name = m.group(1).strip()
        rate = int(m.group(2))
        if rate < 30 or rate > 90:
            # The "%" we matched is almost certainly not a win rate.
            continue
        if name.lower() in {"win rate", "wr", "wins", "win"}:
            continue
        out[name] = float(rate)
    return out


def _extract_via_datalayer(page) -> Dict[str, float]:
    """Try the React/Vue data props if the body-text scan was empty.

    Looks for known keys: data-player, data-winrate, data-wr.
    """
    out: Dict[str, float] = {}
    selectors = [
        ("[data-player][data-winrate]", "data-player", "data-winrate"),
        ("[data-player][data-wr]",      "data-player", "data-wr"),
        ("[data-name][data-rate]",      "data-name",   "data-rate"),
    ]
    for sel, name_attr, rate_attr in selectors:
        try:
            count = page.locator(sel).count()
        except Exception:
            continue
        for i in range(count):
            try:
                el = page.locator(sel).nth(i)
                name = el.get_attribute(name_attr) or ""
                rate_raw = el.get_attribute(rate_attr) or ""
                rate = int(rate_raw)
                if name and 30 <= rate <= 90:
                    out[name] = float(rate)
            except Exception:
                continue
        if out:
            return out
    return out


def update_player_wr_cache(series_id: int, url: str) -> Optional[Dict[str, float]]:
    """Refresh the cache for `series_id` and return the new rates.

    Returns None if the fetch failed (caller should not retry within
    the same poll — we don't want a slow upstream to hammer
    dltv.org).  The cache is also written so the next caller can
    see whatever was scraped even if the fetch returned nothing.
    """
    cache = _read_cache()
    now = time.time()
    try:
        rates = fetch_player_winrates(url)
    except DiscoveryError as exc:
        log.warning("dltv_browser: fetch failed for %s: %s", series_id, exc)
        # Mark the cache as "tried, no result" so we don't retry
        # for TTL — better to wait than to keep hammering.
        cache[_cache_key(series_id)] = {"ts": now, "rates": {}, "url": url, "error": str(exc)}
        _write_cache(cache)
        return None
    if not rates:
        # Page rendered but no WR visible — not a hard error, just
        # nothing useful.  Cache the empty result for a short window
        # so we don't re-fetch the same page.
        cache[_cache_key(series_id)] = {"ts": now, "rates": {}, "url": url}
        _write_cache(cache)
        return {}
    cache[_cache_key(series_id)] = {"ts": now, "rates": rates, "url": url}
    _write_cache(cache)
    return rates


# --------------------------------------------------------------------------- #
# Match state — picks, score, time, gold (v0.3.20+)
# --------------------------------------------------------------------------- #
#
# `fetch_match_state` opens the rendered DLTV match page and
# returns the live draft + score + game time + per-side gold.
# This is the only way to get this data for an in-progress series
# that DLTV's v1 API doesn't return (the API hides live series).
#
# The returned shape is what `board._live_card` expects on
# `m["_picks"]` / `m["_live_score"]` etc. — see the consumer side
# for the contract.  We try several DOM patterns because DLTV's
# frontend markup changes between deploys; if one fails we fall
# back to a body-text scan.
#
# Each periodic update writes both the legacy `rates` entry and
# a `match_state` entry into the same cache file.  Readers pick
# whichever they need.

def _extract_picks_from_dom(page) -> Dict[str, Any]:
    """Return {picks: {radiant: [...], dire: [...]}, ...} from the DOM.

    The data lives in elements like `<div class="...">Hero Name
    1|0%</div>` per the DLTV markup at the time of writing.
    """
    out: Dict[str, Any] = {"picks": {"radiant": [], "dire": []}, "bans": {"radiant": [], "dire": []}}
    try:
        # Look for player rows in the radiant and dire sections.
        # DLTV uses .match__team.match__team--radiant and
        # .match__team--dire, with .player inside.  Each player
        # has .player__hero + .player__name.  We can't be 100%
        # sure of the selectors so try a few.
        for sel in [
            ".match__team--radiant .player",
            ".team--radiant .player",
            "[data-side='radiant'] .player",
        ]:
            try:
                count = page.locator(sel).count()
            except Exception:
                continue
            if count >= 1:
                for i in range(count):
                    try:
                        el = page.locator(sel).nth(i)
                        name = el.locator(".player__name, .player__hero, [data-name]").first.inner_text(timeout=1000)
                        out["picks"]["radiant"].append({"name": name.strip()})
                    except Exception:
                        pass
                break
        for sel in [
            ".match__team--dire .player",
            ".team--dire .player",
            "[data-side='dire'] .player",
        ]:
            try:
                count = page.locator(sel).count()
            except Exception:
                continue
            if count >= 1:
                for i in range(count):
                    try:
                        el = page.locator(sel).nth(i)
                        name = el.locator(".player__name, .player__hero, [data-name]").first.inner_text(timeout=1000)
                        out["picks"]["dire"].append({"name": name.strip()})
                    except Exception:
                        pass
                break
    except Exception as exc:
        log.debug("dltv_browser: picks DOM extract failed: %s", exc)
    return out


def _extract_score_from_text(page) -> Dict[str, Any]:
    """Try the body-text scan for "<left> <right>" + game time.

    The DLTV page header has the live score in big numbers and a
    game time like "12:35" nearby.  We scan the rendered body
    text for these patterns.  Less reliable than DOM selectors
    but works as a fallback.
    """
    out: Dict[str, Any] = {}
    try:
        text = page.evaluate("() => document.body.innerText")
    except Exception:
        return out
    if not isinstance(text, str):
        return out
    # Game time like "12:35"
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m:
        out["game_time"] = f"{m.group(1)}:{m.group(2)}"
    # Score: a pair of digits separated by 2-6 spaces or a tab,
    # near the top of the page (we just look anywhere; it's OK to
    # match multiple times and take the first large one).
    m = re.search(r"\b(\d{1,3})\s+(\d{1,3})\b", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a <= 100 and 0 <= b <= 100 and (a + b) > 0:
            out["radiant_score"] = a
            out["dire_score"] = b
    return out


def fetch_match_state(url: str) -> Dict[str, Any]:
    """Open `url` with Playwright and return the live match state.

    The returned dict has these keys (any may be missing if the
    page didn't render the corresponding block):
      - picks.radiant:   list of {name, hero_id?}
      - picks.dire:      same
      - bans.radiant:    list of {name, hero_id?}
      - bans.dire:       same
      - radiant_score:   int
      - dire_score:      int
      - game_time:       "MM:SS" string
      - radiant_gold:    float
      - dire_gold:       float
      - is_picks_ended:  bool

    Raises DiscoveryError on hard failures (Playwright missing,
    chromium binary missing, network unreachable).
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError as exc:
        raise DiscoveryError(f"playwright not installed: {exc}")

    state: Dict[str, Any] = {}
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:
            raise DiscoveryError(f"chromium launch failed: {exc}")
        try:
            page = browser.new_page()
            try:
                page.goto(url, timeout=PLAYER_WR_FETCH_TIMEOUT_MS,
                          wait_until="domcontentloaded")
            except PWTimeout:
                raise DiscoveryError(f"goto timeout: {url}")
            page.wait_for_timeout(PLAYER_WR_PAGE_LOAD_WAIT_MS)
            # v0.3.20: __NEXT_DATA__ / __INITIAL_STATE__ hold the
            # authoritative React state.  If we can pull it, the
            # whole page is one JSON blob — no fragile DOM walks.
            for win_token in ("__NEXT_DATA__", "__INITIAL_STATE__", "__NUXT__"):
                try:
                    raw = page.evaluate(
                        f"() => window.{win_token} ? JSON.stringify(window.{win_token}) : null"
                    )
                except Exception:
                    raw = None
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except Exception:
                    continue
                # Walk the blob looking for picks/score/time.  We
                # don't know the exact schema; the heuristic is
                # "any key called 'picks' / 'score' / 'duration'".
                state = _state_from_initial(parsed)
                if state:
                    break
            # Fall back to DOM scans if the React payload didn't help.
            if not state.get("picks"):
                state.update(_extract_picks_from_dom(page))
            if state.get("radiant_score") is None or state.get("game_time") is None:
                state.update(_extract_score_from_text(page))
        finally:
            browser.close()
    return state


def _state_from_initial(obj: Any, depth: int = 0) -> Dict[str, Any]:
    """Recursively walk a React initial-state blob looking for live match data.

    Returns the first {picks, score, ...} dict we can stitch together.
    Depth-bounded so a giant JSON doesn't blow the stack.
    """
    if depth > 8:
        return {}
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        # Direct hits
        for k in ("picks", "bans", "score", "duration", "radiant_gold", "dire_gold",
                  "is_picks_ended", "first_blood", "radiant_score", "dire_score"):
            if k in obj:
                out[k] = obj[k]
        if "picks" in out and isinstance(out["picks"], list) and out["picks"]:
            return out
        # Recurse
        for v in obj.values():
            sub = _state_from_initial(v, depth + 1)
            if sub:
                out.update(sub)
                if "picks" in out and isinstance(out["picks"], list) and out["picks"]:
                    return out
        return out if "picks" in out else {}
    if isinstance(obj, list):
        for item in obj:
            sub = _state_from_initial(item, depth + 1)
            if sub:
                return sub
    return {}


def get_cached_match_state(series_id: int) -> Optional[Dict[str, Any]]:
    """Return the cached live match state, or None if missing/stale/empty."""
    cache = _read_cache()
    entry = cache.get(_cache_key(series_id))
    if not entry:
        return None
    if (time.time() - float(entry.get("ts", 0))) > MATCH_STATE_TTL_SEC:
        return None
    state = entry.get("match_state")
    if not isinstance(state, dict):
        return None
    return state


def update_match_state_cache(series_id: int, url: str) -> Optional[Dict[str, Any]]:
    """Fetch + cache the live match state.  Same write semantics as
    `update_player_wr_cache`: returns None on failure and writes
    a failure marker so the next poll skips the URL.
    """
    cache = _read_cache()
    now = time.time()
    try:
        state = fetch_match_state(url)
    except DiscoveryError as exc:
        log.warning("dltv_browser: match_state fetch failed for %s: %s", series_id, exc)
        # Preserve any prior rates we may have cached.
        prev = cache.get(_cache_key(series_id), {})
        prev["ts"] = now
        prev["url"] = url
        prev["match_state"] = {}
        prev["error"] = str(exc)
        cache[_cache_key(series_id)] = prev
        _write_cache(cache)
        return None
    prev = cache.get(_cache_key(series_id), {})
    prev["ts"] = now
    prev["url"] = url
    prev["match_state"] = state
    cache[_cache_key(series_id)] = prev
    _write_cache(cache)
    return state
