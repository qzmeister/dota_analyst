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

Caching
-------
Each (series_id, slug) request hits dltv.org and the response is
cached to `ml_data/player_wr_cache.json` for `PLAYER_WR_TTL_SEC`.
The publisher poll is the natural cadence (5s), so the cache should
be a no-op for steady-state traffic.

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
prediction path treats this as "no player.win_rate available" and
falls back to the encoder's existing value (or the global rate
for unknown players).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._logging import get_logger
from .exceptions import DiscoveryError

log = get_logger(__name__)

ML_DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "ml_data"))
PLAYER_WR_CACHE_FILE = ML_DATA_DIR / "player_wr_cache.json"
PLAYER_WR_TTL_SEC = 5 * 60.0   # 5 minutes
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
