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
`MATCH_STATE_TTL_SEC` (8s as of v0.3.24f).  The publisher poll
runs every 5s and the fetch itself takes 2-3s with the
`wait_for_function` predicate (was 3.5-10s with a fixed
`wait_for_timeout`).  In steady state the cache age at the SSE
rebuild moment is therefore 0-5s + SSE roundtrip, well within
the user's "feel real-time" threshold.  v0.3.24d had bumped
the TTL from 5s to 30s to survive the publisher's 30s tick,
but the publisher was reduced to 5s in v0.3.24e — so 8s
leaves one tick of headroom for jitter while keeping the
cache fresh enough for live predictions.

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

import atexit
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
# v0.3.21: dltv.org cold load is slow when the chromium
# binary has to JIT its V8 on a small container.  8s was
# too tight — the first goto of a brand-new live match
# timed out before React could render the picks.  Bump to
# 20s (the publisher will skip the round entirely if the
# network is fully down) and let the page settle for an
# extra second after DOMContentLoaded.
PLAYER_WR_FETCH_TIMEOUT_MS = 20000
# v0.3.24f: the live state's `radiant_picks` / `dire_picks`
# globals are populated by socket.io events AFTER React
# hydrates.  A fixed 3.5s wait worked but made every fetch
# at least 3.5s long; with a 5s publisher tick + 5-10s
# fetch the live card lagged DLTV by 5-15s.  Replaced the
# fixed `page.wait_for_timeout(PLAYER_WR_PAGE_LOAD_WAIT_MS)`
# with `page.wait_for_function(predicate, timeout=...)` so
# we return as soon as the data is ready (0.5-2s in steady
# state).  The constant is now the MAX wait (hard upper
# bound), not the actual wait.  See `_wait_for_live_state`.
PLAYER_WR_PAGE_LOAD_WAIT_MS = 3500
# v0.3.24f: with faster fetches (1-3s) the 30s cache TTL was
# overkill — the live card was up to 30s old between fetches,
# which the user perceived as a steady 15-25s lag.  Drop to
# 8s: the publisher tick (5s) ensures a fetch is in flight
# before the cache expires, and a single missed tick is
# bounded by the fetch duration (~3s).  Worst-case cache
# age: TTL + one tick + fetch = ~16s, average ~6-8s.
MATCH_STATE_TTL_SEC = 3600.0   # v0.3.24h: bumped from 8s to 1h.
                                # The publisher writes the cache
                                # every 5s during a live match
                                # (so the live card stays fresh),
                                # but a live card for a recently
                                # finished match still wants to
                                # show the final picks/score/gold
                                # from the cache — the discovery
                                # tracker prunes the match the
                                # moment it ends, and the watchlist
                                # path would otherwise render an
                                # empty card for the next hour
                                # until someone deletes the file.
                                # 1h gives the user plenty of time
                                # to scroll back through finished
                                # games; the file is small
                                # (~10KB per entry) so unbounded
                                # growth is not a concern.


# Lazy playwright import.  We don't want a hard dependency at
# business.app import time — only when /api/board or the publisher
# actually needs player.win_rate from a live page.
#
# v0.3.22: we keep a single long-lived Playwright context + browser
# instance for the whole process.  Earlier code created a new
# `sync_playwright()` context (and a fresh chromium with ~20 helper
# processes) for every fetch — the publisher loop runs every 30s,
# so after a few hours we had 2-3k `chrome-headless` PIDs
# accumulating in the container and WSL ballooned to 16GB while
# `docker stats` still showed 540MB (cgroup memory hides orphaned
# subprocesses).  Sharing the browser fixes that.
#
# IMPORTANT: `sync_playwright` uses greenlet-based coroutines that
# are bound to the thread that called `.start()`.  If a different
# thread (e.g. another `asyncio.to_thread` worker) tries to use the
# shared browser, it fails with `greenlet.error: Cannot switch to
# a different thread` — even with a lock, because Playwright's
# internal callbacks (e.g. response handlers) can fire on a
# different thread than the one that called the API.
#
# Solution: every browser call goes through `_browser_executor` —
# a single-worker ThreadPoolExecutor.  That guarantees the call
# (and all of Playwright's internal callbacks) runs in the same
# thread that started playwright.
from concurrent.futures import ThreadPoolExecutor
_playwright = None
_browser = None
_browser_lock = threading.Lock()       # protects browser creation
_browser_executor: ThreadPoolExecutor = None  # single-worker executor
_browser_executor_lock = threading.Lock()
_browser_lock_pid = None  # last pid that touched the lock; for diagnostics


def _get_playwright():
    """Return a process-wide Playwright + Chromium, or None if unavailable.

    The first caller pays the launch cost (~3s for cold chromium);
    subsequent callers reuse the same browser.  Pages are cheap to
    create per-fetch; the heavy bit (the chromium process tree)
    lives for the lifetime of the Python process and is closed at
    interpreter shutdown.

    IMPORTANT: sync_playwright creates a greenlet bound to the
    thread that called `.start()`.  We must initialize it on
    the dedicated executor's worker thread so that all subsequent
    `executor.submit(...)` calls run in the SAME thread.  If we
    initialized on the calling thread, the worker thread would be
    different and every call would fail with
    "Cannot switch to a different thread".
    """
    global _playwright, _browser, _browser_lock_pid, _browser_executor
    if _browser is not None and _playwright is not None and _browser_executor is not None and _is_browser_alive():
        return _playwright, _browser
    with _browser_lock:
        if _browser is not None and _playwright is not None and _browser_executor is not None and _is_browser_alive():
            return _playwright, _browser
        # 1. Create the executor FIRST (so its worker thread exists).
        with _browser_executor_lock:
            if _browser_executor is None:
                _browser_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="dltv_browser-fetch",
                )
        # 2. Run the actual init on the executor's worker thread —
        #    this is what binds the greenlet to the right thread.
        try:
            init_fut = _browser_executor.submit(_init_playwright_on_worker)
            _playwright, _browser = init_fut.result(timeout=30.0)
        except Exception as exc:
            log.warning("dltv_browser: chromium launch failed: %s", exc)
            return None
        log.info("dltv_browser: shared chromium launched (single-process, on executor)")
        _ensure_zombie_reaper()
        return _playwright, _browser


def _init_playwright_on_worker():
    """Initialize playwright + chromium on the executor's worker thread.

    Runs INSIDE the single-worker ThreadPoolExecutor, so the
    greenlet that sync_playwright.start() creates is bound to
    that worker thread.  Every later `executor.submit(...)` then
    runs on the same thread, which is what makes the greenlet
    "thread affinity" work.

    `--single-process` was tempting (no subprocess leak) but
    proved unstable: a single render error kills the whole
    chromium.  We use the normal multi-process model + the
    per-fetch `browser.new_context()` boundary for the leak fix
    (see context.close() notes) and a defensive relaunch in
    `_get_playwright()` if the browser died.
    """
    global _playwright, _browser
    from playwright.sync_api import sync_playwright
    _playwright = sync_playwright().start()
    _browser = _playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    return _playwright, _browser


def _is_browser_alive() -> bool:
    """Return True if the cached browser is still usable.

    Chromium in single-process mode dies on the first tab error;
    even multi-process chromium can crash if a renderer segfaults.
    Cheap probe: ask for `contexts` (raises on closed browser).
    """
    if _browser is None:
        return False
    try:
        _ = _browser.contexts
        return True
    except Exception:
        return False


def _shutdown_playwright() -> None:
    """Best-effort cleanup at interpreter shutdown.

    Called from `atexit` so we don't leave a chromium process tree
    behind on a clean Python exit.  Each helper subprocess takes
    ~30MB of WSL virtual memory even when idle, so we want every
    exit path to release them.
    """
    global _playwright, _browser
    if _browser is not None:
        try:
            _browser.close()
        except Exception:
            pass
        _browser = None
    if _playwright is not None:
        try:
            _playwright.stop()
        except Exception:
            pass
        _playwright = None


atexit.register(_shutdown_playwright)


def _zombie_reaper_loop(interval_sec: float = 5.0) -> None:
    """Best-effort zombie reaper.

    v0.3.22: chromium's helper subprocesses (renderer, GPU, V8
    workers) sometimes outlive the browser process — when that
    happens they get reparented to PID 1 (us, in the container
    PID namespace) and become zombies that nobody reaps.  Without
    this we'd hit `pid_max` in a few hours and the container
    stops forking.

    The fix: as PID 1, we can call `waitpid(-1, WNOHANG)` and
    reap any zombie in our namespace.  This is harmless to our
    own children — those are waited on normally by their
    respective parents (uvicorn for workers, etc.).

    Runs in a daemon thread; started lazily on the first browser
    launch (no point running if playwright is never used).
    """
    while True:
        try:
            # Drain ALL zombies in our namespace, then sleep.
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    # No more zombies to reap right now.
                    break
                except Exception:
                    # If waitpid fails for any reason, stop and
                    # try again next interval.
                    break
                if pid == 0:
                    break
        finally:
            time.sleep(interval_sec)


_reaper_thread_started = False
_reaper_lock = threading.Lock()


def _ensure_zombie_reaper() -> None:
    """Start the zombie reaper thread if not already running."""
    global _reaper_thread_started
    with _reaper_lock:
        if _reaper_thread_started:
            return
        t = threading.Thread(
            target=_zombie_reaper_loop,
            name="dltv_browser-zombie-reaper",
            daemon=True,
        )
        t.start()
        _reaper_thread_started = True


# v0.3.22: start the reaper eagerly at module import time.  The
# daemon thread costs nothing when there are no zombies (waitpid
# returns 0 immediately) and is needed BEFORE the first chromium
# launch — otherwise we have a window where chromium subprocesses
# can become zombies and accumulate because the reaper hasn't
# started yet.  In particular, the publisher loop only fires
# fetch_match_state when the board has live matches, so on a
# quiet evening with no live games the reaper would never start
# on its own.
_ensure_zombie_reaper()


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

    v0.3.22: uses a shared, long-lived chromium browser (see
    `_get_playwright()`) so we don't leak subprocesses on every
    publisher tick.  Each fetch creates a *new browser context*
    and tears it down on exit — that's the boundary that owns
    the page-level helper processes (renderer, GPU, V8 workers)
    and closing it kills them.  Without that we'd see ~1-2 leaked
    `chrome-headless` PIDs per fetch.

    Raises DiscoveryError on hard failures (Playwright missing,
    chromium binary missing, network unreachable).  The caller
    should treat that as "skip this update" and retry next tick.
    """
    pw = _get_playwright()
    if pw is None:
        raise DiscoveryError("playwright not available")
    _playwright_ctx, browser = pw
    try:
        from playwright.sync_api import TimeoutError as PWTimeout
    except ImportError as exc:
        raise DiscoveryError(f"playwright not installed: {exc}")

    # Run the whole fetch in the dedicated single-worker executor
    # so playwright's internal callbacks all run on the same
    # thread that started the browser.  Without this, asyncio's
    # default ThreadPoolExecutor can pick a different worker per
    # call and we hit "Cannot switch to a different thread".
    if _browser_executor is None:
        raise DiscoveryError("browser executor not initialized")
    fut = _browser_executor.submit(
        _fetch_player_winrates_inner, browser, url
    )
    return fut.result(timeout=PLAYER_WR_FETCH_TIMEOUT_MS / 1000.0 + 30.0)


def _fetch_player_winrates_inner(browser, url: str) -> Dict[str, float]:
    """Inner fetch — runs on the dedicated playwright thread."""
    from playwright.sync_api import TimeoutError as PWTimeout

    rates: Dict[str, float] = {}
    context = None
    page = None
    try:
        # `new_context()` (not `new_page()` directly) so we have a
        # proper boundary to close.  Closing the context kills all
        # helper subprocesses that the browser spawned for the page.
        context = browser.new_context()
        page = context.new_page()
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
        # Close in reverse order: page first (cheap), then context
        # (this is what actually kills the subprocesses).
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
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

def _read_heroes_from_window(page) -> Dict[str, Dict[str, Any]]:
    """Read `window.__heroes` and return a hash -> hero dict.

    DLTV's live page used to embed all 127 heroes as a single JS
    object `window.__heroes = { "<dltv_id>": {id, steam_id, ...} }`.
    As of v0.3.22 they sometimes drop that block (page is lighter
    and only the live match's hero IDs are needed) — we then
    fall back to our own `client.get_heroes()` index, which is
    the same data and is always available.

    Returns a dict keyed by the image-URL hash (the part between
    `/uploads/heroes/` and `.png`/`.jpg`).
    """
    # Try the embedded block first (faster, no extra network)
    try:
        raw = page.evaluate("() => window.__heroes ? JSON.stringify(window.__heroes) : null")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed:
                    out: Dict[str, Dict[str, Any]] = {}
                    for h in parsed.values():
                        if not isinstance(h, dict):
                            continue
                        image = h.get("image") or ""
                        if "/" in image:
                            hash_part = image.rsplit("/", 1)[-1].split(".", 1)[0]
                        else:
                            hash_part = image
                        if hash_part:
                            out[hash_part] = h
                    if out:
                        return out
            except Exception:
                pass
    except Exception:
        pass

    # Fall back to the project-local hero index.  `client.get_heroes()`
    # is populated at startup from the v1 API and cached for the
    # life of the process; it has the same `image` field as DLTV's
    # page so the hash -> hero lookup works identically.
    try:
        from .dltv_client import client as _dltv_client
        heroes = _dltv_client.get_heroes() or []
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for h in heroes:
        if not isinstance(h, dict):
            continue
        image = h.get("image") or ""
        if "/" in image:
            hash_part = image.rsplit("/", 1)[-1].split(".", 1)[0]
        else:
            hash_part = image
        if hash_part:
            out[hash_part] = h
    return out


def _install_socket_hook(context, expected_steam_id: Optional[int] = None) -> None:
    """Install a JS init-script that intercepts DLTV's socket.io events.

    DLTV ships live match data via socket.io (`__nd2_match_{steam_id}`
    channel).  The payload is the same `result` object the page itself
    uses to update `.head__duration strong`, `.team__score strong`,
    `.team__networth strong` etc.  We wrap socket.io's Manager so
    every match-related callback ALSO copies the payload to
    `window.__dltv_lastResult` (and bumps `__dltv_resultCount`).

    Why this exists (v0.3.25l): the previous extractor relied on
    CSS classes like `.info__duration[data-game-time]`,
    `.team__networth .networth > span` and `.team__scores-kills`.
    DLTV keeps changing the markup — those classes have all been
    renamed/removed (current: `.head__duration strong`,
    `.team__score strong`, `.team__networth` with `Math.abs(lead)`).
    Reading from the socket payload is independent of any CSS
    choice on DLTV's part and is therefore stable.

    v0.3.25l-bugfix: the unfiltered version of this hook captured
    payloads from EVERY `__nd2_match_*` event the page fires,
    including the live-ticker / sidebar / chat widgets DLTV ships
    alongside the main match.  The result was a `__dltv_lastResult`
    that ping-ponged between matches — the cache ended up with
    `radiant_score=50, dire_score=1` for a match that's actually
    44:33.  We now extract `steam_id` from the event name and
    filter: only payloads whose `steam_id` matches
    `expected_steam_id` land in `__dltv_lastResult`.  Per-match
    payloads are still kept in `__dltv_results[steam_id]` for
    debug visibility.

    Must be called on the `BrowserContext` BEFORE
    `context.new_page()` so the init-script runs in every page
    the context produces.
    """
    # The expected_steam_id is interpolated into the JS as a literal
    # so the hook can compare it in the browser.  Default -1 = "any".
    expected_js = int(expected_steam_id) if expected_steam_id is not None else -1
    try:
        context.add_init_script(
            r"""
            (() => {
                if (window.__dltv_hookInstalled) return;
                window.__dltv_hookInstalled = true;
                window.__dltv_lastResult = null;
                window.__dltv_resultCount = 0;
                window.__dltv_results = {};   // steam_id -> payload
                const expected = __DLTV_EXPECTED_STEAM_ID__;
                const store = (steamId, data) => {
                    try {
                        if (expected > 0 && steamId !== null && steamId !== expected) {
                            return;  // skip payloads from other matches
                        }
                        window.__dltv_lastResult = data;
                        window.__dltv_resultCount++;
                        if (steamId) {
                            window.__dltv_results[steamId] = data;
                        }
                    } catch (e) { /* ignore */ }
                };
                const tryWrap = () => {
                    if (window.io && window.io.Manager) {
                        const proto = window.io.Manager.prototype;
                        if (proto.__dltv_onPatched) return;
                        const orig = proto.on;
                        proto.on = function(...args) {
                            try {
                                const ev = String(args[0] || '');
                                // Extract steam_id from event name like
                                // `__nd2_match_8917853656`.  Falls back
                                // to null when the event doesn't carry one
                                // (e.g. global `live` / `game` events).
                                let sid = null;
                                const m = ev.match(/__nd2_match_(\d+)/);
                                if (m) sid = parseInt(m[1], 10);
                                if (/match|game|live|nd2/i.test(ev)) {
                                    const cb = args[args.length - 1];
                                    if (typeof cb === 'function') {
                                        const wrapped = function(data) {
                                            store(sid, data);
                                            return cb.apply(this, arguments);
                                        };
                                        args[args.length - 1] = wrapped;
                                    }
                                }
                            } catch (e) { /* fall through */ }
                            return orig.apply(this, args);
                        };
                        proto.__dltv_onPatched = true;
                    } else {
                        setTimeout(tryWrap, 100);
                    }
                };
                tryWrap();
                // Some DLTV builds expose a global handler instead.
                // Wrap any of the common names so we still capture
                // the payload if socket.io wrapping missed it.
                ['handleGame', 'onResult', 'onGameUpdate', 'handleUpdate'].forEach((n) => {
                    try {
                        const orig = window[n];
                        if (typeof orig === 'function' && !orig.__dltv_patched) {
                            window[n] = function(r) {
                                store(null, r);  // can't filter, no steam_id
                                return orig.apply(this, arguments);
                            };
                            window[n].__dltv_patched = true;
                        }
                    } catch (e) { /* ignore */ }
                });
            })();
            """.replace("__DLTV_EXPECTED_STEAM_ID__", str(expected_js))
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("dltv_browser: install_socket_hook failed: %s", exc)


def _read_hooked_socket_result(page) -> Dict[str, Any]:
    """Read the last socket.io payload captured by `_install_socket_hook`.

    Returns a flat dict with whatever numeric fields we care about;
    missing fields are `None` so the caller can fall through to a
    DOM extractor for them.
    """
    out: Dict[str, Any] = {}
    try:
        data = page.evaluate(
            """() => {
                const r = window.__dltv_lastResult;
                if (!r || typeof r !== 'object') return null;
                const get = (k) => (typeof r[k] === 'number' && Number.isFinite(r[k])) ? r[k] : null;
                return {
                    radiant_score: get('radiant_score'),
                    dire_score:    get('dire_score'),
                    game_time:     get('game_time'),
                    radiant_lead:  get('radiant_lead'),
                    radiant_networth: get('radiant_networth'),
                    dire_networth:    get('dire_networth'),
                    is_picks_ended:    r.is_picks_ended === true,
                    count: window.__dltv_resultCount || 0,
                };
            }"""
        )
    except Exception as exc:
        log.debug("dltv_browser: read_hooked_socket_result failed: %s", exc)
        return out
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        out[k] = v
    return out


def _read_live_state_from_scoreboard(page) -> Dict[str, Any]:
    """Read picks, bans, scores, time, teams from the live scoreboard
    and the page's own `radiant_picks` / `dire_picks` JS state.

    v0.3.23: DLTV redesigned the match page for the third time.  The
    `.map__finished-v2` block is gone — DLTV now renders the in-progress
    game inside `#live_scoreboard` and updates the picks/score via a
    socket.io connection (`__nd2_match_{steam_id}` channel) which
    populates the page's `radiant_picks` / `dire_picks` global arrays
    on every tick.  This extractor reads BOTH sources:

      1. **`radiant_picks` / `dire_picks` page globals** (real-time):
         - populated by DLTV's own `handleGame(result)` callback
         - updated on every socket.io event (real-time, no API lag)
         - each pick is `{id, steam_id, title, slug, image, ...}`

      2. **`#live_scoreboard` DOM** (locale-independent CSS classes):
         - team names / sides from `.team__title-name` (the `.side` element
           has CSS class `radiant` or `dire` — text content is translated
           but the classes are not)
         - kills from `.team__scores-kills` (one per team)
         - game time from `.info__duration[data-game-time]` (in seconds)

    The function is called via `page.evaluate()` and returns a plain
    JSON-serialisable dict, so we get the data at the same speed the
    page is rendering it.

    v0.3.25l: DLTV renamed the CSS classes used in v0.3.24 (the
    `.info__duration[data-game-time]`, `.team__networth .networth > span`,
    `.team__scores-kills` selectors are all gone).  The new layout
    uses `.head__duration strong` for the clock (rendered MM:SS text),
    `.team__score strong` for kills, and `.team__networth` with
    `Math.abs(result.radiant_lead)` for the gold lead.  Rather than
    chase the CSS every release, we read straight from the socket.io
    payload via `_read_hooked_socket_result` (installed by
    `_install_socket_hook`).  DOM is only the fallback for fields the
    hook didn't deliver.
    """
    out: Dict[str, Any] = {
        "picks": {"radiant": [], "dire": []},
        "bans": {"radiant": [], "dire": []},
        "team_order": [],
        "radiant_score": None,
        "dire_score": None,
        "game_time": None,
        "radiant_networth": None,
        "dire_networth": None,
        "gold_lead_radiant": None,
    }
    # v0.3.25l: prefer the socket.io payload.  It is independent of
    # any CSS DLTV might use next release, and carries the data in
    # raw numeric form (seconds for game_time, signed int for lead,
    # raw ints for networth) — no parsing of rendered MM:SS or
    # Math.abs strings.
    hooked = _read_hooked_socket_result(page)
    if hooked.get("count", 0) > 0:
        out["radiant_score"] = hooked.get("radiant_score")
        out["dire_score"] = hooked.get("dire_score")
        out["game_time"] = hooked.get("game_time")
        out["radiant_networth"] = hooked.get("radiant_networth")
        out["dire_networth"] = hooked.get("dire_networth")
        out["gold_lead_radiant"] = hooked.get("radiant_lead")
    try:
        data = page.evaluate(
            """() => {
                const out = {
                    picks: { radiant: [], dire: [] },
                    bans: { radiant: [], dire: [] },
                    team_order: [],
                    radiant_score: null,
                    dire_score: null,
                    game_time: null,
                    teams: [],
                };
                // v0.3.25l: DLTV no longer puts the live state under
                // #live_scoreboard — the scoreboard lives wherever
                // the React tree mounts it.  We walk the whole
                // document and look for the team blocks by their
                // current class names.
                const teamEls = Array.from(document.querySelectorAll('.team'))
                    .filter((t) => t.querySelector('.team__title-name, .team__title, .head__duration'))
                    .slice(0, 2);
                const teams = teamEls.map((t) => {
                    // v0.3.23 selector: `.team__title-name .name` (split with side sibling).
                    // v0.3.25l selector: `.team__title .name` (name only, no side sibling).
                    const nameEl = t.querySelector('.team__title-name .name, .team__title .name, .name');
                    const sideEl = t.querySelector('.team__title-name .side, .team__title .side, .side');
                    // v0.3.23 selector: `.team__scores-kills` (int text).
                    // v0.3.25l selector: `.team__score strong` (int text).
                    const killsEl = t.querySelector('.team__scores-kills, .team__score strong, .team__score');
                    // v0.3.25l: `.team__networth strong` carries
                    // `Math.abs(result.radiant_lead)`.  We can't
                    // recover the sign from the DOM (the icon class
                    // encodes direction via CSS), so the value here
                    // is treated as the absolute lead only when the
                    // socket hook has NOT already supplied a signed
                    // value.
                    const nwEl = t.querySelector('.team__networth strong, .team__networth');
                    let networth = null;
                    if (nwEl) {
                        const raw = nwEl.textContent || '';
                        const n = parseInt(raw.replace(/[^\\d]/g, ''), 10);
                        if (!Number.isNaN(n) && n > 0) networth = n;
                    }
                    let sideKind = 'unknown';
                    if (sideEl) {
                        if (sideEl.classList.contains('radiant')) sideKind = 'radiant';
                        else if (sideEl.classList.contains('dire')) sideKind = 'dire';
                    } else if (t.classList.contains('radiant')) {
                        sideKind = 'radiant';
                    } else if (t.classList.contains('dire')) {
                        sideKind = 'dire';
                    }
                    return {
                        name: nameEl ? (nameEl.textContent || '').trim() : '',
                        side_kind: sideKind,
                        kills: killsEl ? parseInt((killsEl.textContent || '0').trim(), 10) || 0 : null,
                        networth_abs: networth,
                    };
                });
                out.teams = teams;
                out.team_order = teams.map((t) => t.side_kind);
                if (teams.length >= 1) out.radiant_score = teams[0].kills;
                if (teams.length >= 2) out.dire_score = teams[1].kills;
                if (teams.length >= 1) out.radiant_networth = teams[0].networth_abs;
                if (teams.length >= 2) out.dire_networth = teams[1].networth_abs;
                // v0.3.25l: game time lives in `.head__duration strong`
                // (rendered MM:SS) or `.info__duration strong` (older
                // v0.3.24 selector).  Parse the MM:SS back to seconds.
                const parseClock = (txt) => {
                    if (!txt) return null;
                    const m = String(txt).trim().match(/^(\\d{1,2}):(\\d{2})$/);
                    if (!m) return null;
                    const mm = parseInt(m[1], 10);
                    const ss = parseInt(m[2], 10);
                    if (Number.isNaN(mm) || Number.isNaN(ss)) return null;
                    return mm * 60 + ss;
                };
                const durEl = document.querySelector('.head__duration strong, .info__duration strong');
                if (durEl) {
                    const parsed = parseClock(durEl.textContent);
                    if (parsed != null) out.game_time = parsed;
                }
                // Picks: read from the page's own globals
                // (`radiant_picks` / `dire_picks` are populated by
                // the socket.io `__nd2_match_*` handler).  We also
                // fall back to `get_picks_from_live_map` for older
                // page versions.
                const sideMap = (s) => {
                    if (!s) return 'unknown';
                    const x = String(s).toLowerCase();
                    if (x === 'radiant' || x === '1') return 'radiant';
                    if (x === 'dire' || x === '0' || x === '2') return 'dire';
                    return 'unknown';
                };
                if (typeof radiant_picks !== 'undefined' && Array.isArray(radiant_picks)) {
                    const rSide = teamEls.length > 0 ? sideMap(teamEls[0].querySelector('.team__title-name .side, .team__title .side, .side')?.classList.contains('radiant') ? 'radiant' : 'dire') : 'radiant';
                    out.picks[rSide] = radiant_picks.map((p) => ({
                        hero_id: p.id, steam_id: p.steam_id,
                        name: p.title, slug: p.slug, image: p.image,
                    }));
                }
                if (typeof dire_picks !== 'undefined' && Array.isArray(dire_picks)) {
                    const dSide = teamEls.length > 1 ? sideMap(teamEls[1].querySelector('.team__title-name .side, .team__title .side, .side')?.classList.contains('radiant') ? 'radiant' : 'dire') : 'dire';
                    out.picks[dSide] = dire_picks.map((p) => ({
                        hero_id: p.id, steam_id: p.steam_id,
                        name: p.title, slug: p.slug, image: p.image,
                    }));
                }

                // v0.4.0.1: enrich picks with the player nickname from
                // the live scoreboard DOM.  The socket.io fast_picks
                // payload only carries `player.title` DURING the draft
                // (is_picks_ended = false); once the game starts, the
                // live socket payload drops the player field.  DLTV's
                // own UI, however, keeps the nickname rendered under
                // each hero icon throughout the game, so we can pull
                // it from the DOM.  We try a handful of common class
                // names (DLTV has redesigned the scoreboard 3+ times
                // since 0.3.23) and fall back gracefully — if no
                // selector matches, `player_name` stays null and the
                // UI renders the hero name as before.
                const findPlayerNameFor = (pickIdx, side) => {
                    // Pick card containers we know DLTV has used.
                    // Order: newest selectors first.
                    const cardSelectors = [
                        // v0.3.25l: per-side pick cards
                        `.team.${side} .pick`,
                        `.team.${side} .pick-card`,
                        `.team.${side} .picks__item`,
                        `.team.${side} .hero-card`,
                        `.team.${side === 'radiant' ? 'first' : 'second'} .pick`,
                        // global: any pick
                        `.pick:nth-of-type(${pickIdx + 1})`,
                        `.pick-card:nth-of-type(${pickIdx + 1})`,
                    ];
                    for (const sel of cardSelectors) {
                        const card = document.querySelector(sel);
                        if (!card) continue;
                        // Player name lives in a child element with one
                        // of these classes.  Try a few shapes.
                        const nameSelectors = [
                            '.pick__player', '.player-name', '.player__name',
                            '.nickname', '.name-player', '.player',
                            'span.player', 'div.player',
                        ];
                        for (const ns of nameSelectors) {
                            const el = card.querySelector(ns);
                            if (el) {
                                const t = (el.textContent || '').trim();
                                if (t && t.length > 0 && t.length < 64) return t;
                            }
                        }
                        // Fallback: `data-player` attribute on the card.
                        const data = card.getAttribute('data-player')
                                  || card.getAttribute('data-player-name');
                        if (data) return data;
                    }
                    return null;
                };
                for (const side of ['radiant', 'dire']) {
                    if (out.picks[side]) {
                        out.picks[side] = out.picks[side].map((p, i) => ({
                            ...p,
                            player_name: findPlayerNameFor(i, side),
                        }));
                    }
                }

                // v0.4.0.1: destroyed-tower counts from the mini-map.
                // DLTV renders the live map with tower icons; standing
                // towers are full-color, destroyed ones are greyscale
                // (or have a `.destroyed` / `.dead` / `.fallen` class).
                // We count both, knowing each side has 11 towers in a
                // standard map (3 tier-1 + 2 mid + 2 tier-2 + 2 tier-3
                // + 4 barracks = 11) — `destroyed = 11 - standing`.
                // We try several selector shapes because the mini-map
                // CSS has changed across DLTV releases.
                const minimapSelectors = [
                    '.minimap', '.mini-map', '.map-container',
                    '[class*="minimap"]', '[class*="mini_map"]',
                    '.map', '.live-map',
                ];
                let minimap = null;
                for (const sel of minimapSelectors) {
                    const el = document.querySelector(sel);
                    if (el) { minimap = el; break; }
                }
                if (minimap) {
                    // Find all tower-like elements.  DLTV uses different
                    // classes; try a few and union the results.
                    const towerSel = [
                        '[class*="tower"]', 'img[alt*="tower" i]',
                        'svg.tower', '.tower', '.icon-tower',
                    ].join(',');
                    const towers = Array.from(minimap.querySelectorAll(towerSel));
                    if (towers.length > 0) {
                        // Standing tower heuristics:
                        //   - opacity > 0.5 (full color vs greyscale)
                        //   - or no `destroyed`/`dead`/`fallen` class
                        //   - or `data-state` = standing
                        const isStanding = (el) => {
                            const cs = el.classList;
                            if (cs.contains('destroyed') || cs.contains('dead')
                                || cs.contains('fallen') || cs.contains('broken')) {
                                return false;
                            }
                            const state = el.getAttribute('data-state');
                            if (state && /destroy|dead|fallen|broken/i.test(state)) return false;
                            const op = parseFloat(window.getComputedStyle(el).opacity || '1');
                            if (!Number.isNaN(op) && op < 0.5) return false;
                            return true;
                        };
                        // Split by side.  Without a clear "radiant"/"dire"
                        // marker, we just sum standing vs total and let
                        // the caller decide.  If side markers exist
                        // (`.radiant .tower`, `.dire .tower`), use them.
                        let radiantStanding = 0, direStanding = 0;
                        let radiantTotal = 0, direTotal = 0;
                        const rTowers = minimap.querySelectorAll('.radiant [class*="tower"], .first [class*="tower"]');
                        const dTowers = minimap.querySelectorAll('.dire [class*="tower"], .second [class*="tower"]');
                        if (rTowers.length > 0 || dTowers.length > 0) {
                            for (const t of rTowers) {
                                radiantTotal++;
                                if (isStanding(t)) radiantStanding++;
                            }
                            for (const t of dTowers) {
                                direTotal++;
                                if (isStanding(t)) direStanding++;
                            }
                            out.destroyed_towers = {
                                radiant_destroyed: radiantTotal - radiantStanding,
                                dire_destroyed:    direTotal - direStanding,
                                radiant_standing:  radiantStanding,
                                dire_standing:     direStanding,
                            };
                        } else {
                            // No side split available.  Sum across both
                            // teams (each side has 11 towers → 22 total).
                            let standing = 0, total = towers.length;
                            for (const t of towers) {
                                if (isStanding(t)) standing++;
                            }
                            out.destroyed_towers = {
                                standing: standing,
                                total: total,
                                note: 'no side split; count is combined',
                            };
                        }
                    }
                }
                return out;
            }"""
        )
    except Exception as exc:
        log.debug("dltv_browser: _read_live_state_from_scoreboard failed: %s", exc)
        return out
    if not isinstance(data, dict):
        return out
    out["picks"] = data.get("picks") or {"radiant": [], "dire": []}
    out["bans"] = data.get("bans") or {"radiant": [], "dire": []}
    out["team_order"] = data.get("team_order") or []
    # v0.3.25l: only fall through to DOM values for fields the
    # socket hook didn't supply.  Hooked values are the source of
    # truth; DOM is the fallback.  This keeps existing behaviour
    # (DOM wins when the hook is silent) intact for older DLTV
    # builds whose socket payload doesn't carry, e.g., networth.
    for k in ("radiant_score", "dire_score", "game_time",
              "radiant_networth", "dire_networth"):
        if out.get(k) is None:
            v = data.get(k)
            if v is not None:
                out[k] = v
    # v0.3.25l: if the socket hook didn't give us a signed lead
    # but the DOM has an absolute one, surface that too so the UI
    # can at least show the magnitude.
    if out.get("gold_lead_radiant") is None:
        n = data.get("radiant_networth")
        if isinstance(n, int) and n > 0:
            # Without a sign we don't know which side leads; the
            # frontend treats gold_lead_radiant=None as "abs only"
            # and just renders the magnitude.  See board._build_live_gold.
            out["gold_lead_radiant"] = n
    teams = data.get("teams") or []
    out["team_names"] = [(t.get("name") or "") for t in teams]
    out["team_sides"] = [t.get("side_kind") or "unknown" for t in teams]
    return out


def _read_map_block_from_dom(page) -> Dict[str, Any]:
    """LEGACY: read picks, bans, scores, time, teams from `.map__finished-v2`.

    v0.3.22 wrote this for an older DLTV layout where the in-progress
    game was rendered inside a `.map__finished-v2` block.  v0.3.23
    replaced this with `_read_live_state_from_scoreboard` (which reads
    from `#live_scoreboard` and the page's own `live_map` JS state).
    This function is kept as a fallback for older DLTV versions or
    non-hydrated pages where the new selectors haven't been populated
    yet.

    Hero identification uses the image URL hash; the lookup table
    comes from `window.__heroes` (see _read_heroes_from_window).

    Picks come out in DOM order: first the team rendered first in
    the header (left side in DLTV's UI), then the other team.
    The team order is captured in `team_order` (e.g. ["dire",
    "radiant"]) so the caller can split picks correctly.
    """
    out: Dict[str, Any] = {
        "picks": {"radiant": [], "dire": []},
        "bans": {"radiant": [], "dire": []},
        "team_order": [],
    }
    try:
        data = page.evaluate(
            """() => {
                const map = document.querySelector('.map__finished-v2');
                if (!map) return null;
                const heroUrls = (els) => {
                    const out = [];
                    els.forEach(el => {
                        const img = el.querySelector('.pick__image, .ban__image');
                        if (!img) return;
                        const style = img.getAttribute('style') || '';
                        const m = style.match(/url\\(['"]?([^'")]+)['"]?\\)/);
                        if (m) out.push(m[1]);
                    });
                    return out;
                };
                const pickEls = map.querySelectorAll('.pick:not(.pick-sm)');
                const banEls = map.querySelectorAll('.ban');
                const scores = Array.from(map.querySelectorAll('.team__scores-kills')).map(el => (el.textContent || '').trim());
                const dur = map.querySelector('.duration b');
                const time = dur ? (dur.textContent || '').trim() : '';
                const teamEls = Array.from(map.querySelectorAll('.team__title'));
                const teams = teamEls.slice(0, 2).map(t => {
                    const name = t.querySelector('.name');
                    const side = t.querySelector('.side');
                    let side_kind = 'unknown';
                    if (side) {
                        if (side.classList.contains('dire')) side_kind = 'dire';
                        else if (side.classList.contains('radiant')) side_kind = 'radiant';
                    }
                    return {
                        name: name ? (name.textContent || '').trim() : '',
                        side_kind: side_kind,
                    };
                });
                return {
                    picks: heroUrls(pickEls),
                    bans: heroUrls(banEls),
                    scores: scores,
                    time: time,
                    teams: teams,
                };
            }"""
        )
    except Exception as exc:
        log.debug("dltv_browser: _read_map_block_from_dom failed: %s", exc)
        return out
    if not isinstance(data, dict):
        return out

    # Use the locale-independent `side_kind` ("dire" | "radiant" |
    # "unknown") that the JS already pulled from the .side classList.
    team_order: list = []
    for t in data.get("teams") or []:
        kind = (t.get("side_kind") or "unknown").lower().strip()
        if kind in ("dire", "radiant"):
            team_order.append(kind)
        else:
            team_order.append("unknown")
    out["team_order"] = team_order[:2]

    # Read the hero hash → hero lookup once
    heroes = _read_heroes_from_window(page)

    def _hero_for_url(u: str) -> Dict[str, Any]:
        if not u:
            return {}
        hash_part = u.rsplit("/", 1)[-1].split(".", 1)[0]
        return heroes.get(hash_part) or {}

    def _entry_for_url(u: str) -> Dict[str, Any]:
        h = _hero_for_url(u)
        if not h:
            # Fall back to just the URL so the caller can debug
            return {"image": u}
        return {
            "hero_id": h.get("id"),         # DLTV internal
            "steam_id": h.get("steam_id"),  # Valve id
            "name": h.get("title"),
            "slug": h.get("slug"),
        }

    pick_urls = data.get("picks") or []
    n_picks = len(pick_urls)
    if n_picks >= 2:
        split = n_picks // 2
        first_side = out["team_order"][0] if out["team_order"] and out["team_order"][0] != "unknown" else "radiant"
        second_side = out["team_order"][1] if len(out["team_order"]) > 1 and out["team_order"][1] != "unknown" else "dire"
        # Convention: first team in DOM = the team the layout starts with
        # (DLTV puts the dire side first, but we don't want to bake
        # that in — use the team_order we just computed).
        for i, u in enumerate(pick_urls):
            side = first_side if i < split else second_side
            out["picks"][side].append(_entry_for_url(u))

    ban_urls = data.get("bans") or []
    n_bans = len(ban_urls)
    if n_bans >= 2:
        split_b = n_bans // 2
        first_side = out["team_order"][0] if out["team_order"] and out["team_order"][0] != "unknown" else "radiant"
        second_side = out["team_order"][1] if len(out["team_order"]) > 1 and out["team_order"][1] != "unknown" else "dire"
        for i, u in enumerate(ban_urls):
            side = first_side if i < split_b else second_side
            out["bans"][side].append(_entry_for_url(u))

    scores = data.get("scores") or []
    if len(scores) >= 2:
        try:
            a, b = int(scores[0]), int(scores[1])
            # The scores come from the team order in the DOM, which
            # we captured as team_order.  Map them to radiant/dire
            # correctly.
            sides = (out["team_order"] + ["radiant", "dire"])[:2]
            if sides[0] == "dire":
                # DOM order is [dire, radiant] → scores[0]=dire, scores[1]=radiant
                out["dire_score"] = a
                out["radiant_score"] = b
            else:
                out["radiant_score"] = a
                out["dire_score"] = b
        except (ValueError, TypeError):
            pass

    if data.get("time"):
        out["game_time"] = data["time"]

    return out


def _extract_picks_from_dom(page) -> Dict[str, Any]:
    """Return {picks: {radiant: [...], dire: [...]}, bans: ...} from the DOM.

    v0.3.22: DLTV's live page no longer tags picks/bans with
    `data-hero-id` — heroes are referenced by image URL hash
    instead.  The hash → hero mapping is sourced from the
    embedded `window.__heroes` JS object.  See `_read_map_block_from_dom`
    for the full structure of the new markup.
    """
    return _read_map_block_from_dom(page)


def _extract_score_from_text(page) -> Dict[str, Any]:
    """Legacy body-text scan for "<left> <right>" + game time.

    v0.3.22: prefer `_read_map_block_from_dom` for the score
    (it uses the real `.team__scores-kills` numbers and
    `.duration b` time).  This function is kept as a safety
    net for the rare case where the `.map__finished-v2` block
    is missing but the body still has the header numbers.
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
        # v0.3.25m: store as int seconds.  The old code returned
        # the "MM:SS" string here, which the frontend then
        # rejected (`fmtClock` only handles numbers) — every
        # legacy cache entry made the live clock render as "—"
        # even when the text scan found the time.  Returning
        # int seconds lets the board layer pass the value
        # through unchanged.
        try:
            mm = int(m.group(1))
            ss = int(m.group(2))
            if 0 <= mm <= 180 and 0 <= ss < 60:
                out["game_time"] = mm * 60 + ss
        except (TypeError, ValueError):
            pass
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


def fetch_match_state(url: str, expected_steam_id: Optional[int] = None) -> Dict[str, Any]:
    """Open `url` with Playwright and return the live match state.

    The returned dict has these keys (any may be missing if the
    page didn't render the corresponding block):
      - picks.radiant:   list of {hero_id, steam_id, name}
      - picks.dire:      same
      - bans.radiant:    list of {hero_id, steam_id, name}
      - bans.dire:       same
      - radiant_score:   int
      - dire_score:      int
      - game_time:       "MM:SS" string
      - team_order:      ["dire", "radiant"] (or whatever the DOM shows)

    v0.3.25l-bugfix: `expected_steam_id` is forwarded to the
    socket.io hook so it can filter out payloads from neighbouring
    matches on the page (live ticker / sidebar / chat).  When the
    page receives a `__nd2_match_{steam_id}` event whose `steam_id`
    doesn't match, the payload is dropped instead of being
    captured into `__dltv_lastResult`.  See
    `_install_socket_hook` for the full rationale.

    v0.3.22: uses the shared chromium browser (see
    `_get_playwright()`) AND isolates each fetch in its own
    `browser.new_context()` that's explicitly closed in
    `finally`.  Without the context boundary, `page.close()`
    alone leaks ~1-2 helper subprocesses per fetch.  The
    browser itself lives until process exit (atexit-registered
    cleanup).

    Raises DiscoveryError on hard failures (Playwright missing,
    chromium binary missing, network unreachable).
    """
    pw = _get_playwright()
    if pw is None:
        raise DiscoveryError("playwright not available")
    _playwright_ctx, browser = pw
    if _browser_executor is None:
        raise DiscoveryError("browser executor not initialized")
    fut = _browser_executor.submit(
        _fetch_match_state_inner, browser, url, expected_steam_id
    )
    return fut.result(timeout=PLAYER_WR_FETCH_TIMEOUT_MS / 1000.0 + 30.0)


def _wait_for_live_state(page, max_ms: int) -> None:
    """Wait for the page's live state to be populated.

    v0.3.24f: the live card's picks/score lag was dominated by a
    fixed `page.wait_for_timeout(PLAYER_WR_PAGE_LOAD_WAIT_MS)`
    (3.5s) on every fetch.  With a 5s publisher tick and 1-2s
    page-load time, the per-fetch cost was 5-6s; the user saw
    5-15s of staleness in the card.

    DLTV's live state lives in three places, populated by socket.io
    events AFTER React hydrates:
      1. `#live_scoreboard .team__scores-kills` (DOM, scores)
      2. `radiant_picks` / `dire_picks` page globals (picks/bans)
      3. `.info__duration[data-game-time]` (game time)

    The predicate returns true as soon as ANY of these is present.
    In steady state this fires at 0.5-2s; on a cold chromium
    (just-launched V8) it can take the full `max_ms`.  On timeout
    we fall through to the legacy extractors (which read
    `.map__finished-v2` or scan the body text) — those paths
    don't depend on socket.io so they still produce something
    useful for older DLTV page versions.
    """
    from playwright.sync_api import TimeoutError as PWTimeout
    try:
        page.wait_for_function(
            """() => {
                const sb = document.getElementById('live_scoreboard');
                if (!sb) return false;
                const scoreEls = sb.querySelectorAll('.team__scores-kills');
                if (scoreEls.length >= 2) return true;
                if (typeof radiant_picks !== 'undefined' && Array.isArray(radiant_picks) && radiant_picks.length > 0) return true;
                if (typeof dire_picks !== 'undefined' && Array.isArray(dire_picks) && dire_picks.length > 0) return true;
                return false;
            }""",
            timeout=max_ms,
        )
    except PWTimeout:
        # Cold chromium or older DLTV layout — proceed with whatever
        # the legacy extractors can find.  The condition that would
        # have triggered a return is most likely 'page not yet
        # hydrated', which is the same condition the legacy
        # extractors handle gracefully.
        log.debug("dltv_browser: live state wait timed out after %dms", max_ms)


def _fetch_match_state_inner(browser, url: str, expected_steam_id: Optional[int] = None) -> Dict[str, Any]:
    """Inner fetch — runs on the dedicated playwright thread."""
    from playwright.sync_api import TimeoutError as PWTimeout

    state: Dict[str, Any] = {}
    context = None
    page = None
    try:
        context = browser.new_context()
        # v0.3.25l: install the socket.io payload interceptor on the
        # context BEFORE the page is created so the init-script
        # runs before DLTV's own scripts attach handlers.  Without
        # this ordering the Manager.prototype.on patch lands after
        # DLTV has already wired up, and we miss the first event.
        # v0.3.25l-bugfix: pass `expected_steam_id` so the hook can
        # filter out payloads from neighbouring matches (live ticker
        # / sidebar / chat).  Without it the cache accumulates
        # `radiant_score=50, dire_score=1`-style garbage for any
        # match whose sidebar ticker fires a different event.
        _install_socket_hook(context, expected_steam_id=expected_steam_id)
        page = context.new_page()
        try:
            page.goto(url, timeout=PLAYER_WR_FETCH_TIMEOUT_MS,
                      wait_until="domcontentloaded")
        except PWTimeout:
            raise DiscoveryError(f"goto timeout: {url}")
        _wait_for_live_state(page, max_ms=PLAYER_WR_PAGE_LOAD_WAIT_MS)
        # v0.3.23: DLTV redesigned the live page for the third time
        # in 24h.  The new layout puts the in-progress game inside
        # `#live_scoreboard` and computes picks/bans via JS, exposing
        # the result through the page's own `get_picks_from_live_map`
        # function.  This is the primary extractor — it gives us
        # real-time data at the same speed the page renders it,
        # without the API delay.
        state = _read_live_state_from_scoreboard(page)
        # Fallback: v0.3.22's `.map__finished-v2` extractor still
        # works on older page versions or pages that haven't fully
        # hydrated.  Use it when the new extractor returns empty.
        if (not state.get("picks", {}).get("radiant")
                and not state.get("picks", {}).get("dire")
                and state.get("radiant_score") is None):
            legacy = _read_map_block_from_dom(page)
            if legacy:
                state.update({k: v for k, v in legacy.items() if v})
        # Last-resort fallback: body-text scan if scores/time
        # are still missing (older page versions, pre-hydration).
        if state.get("radiant_score") is None or state.get("game_time") is None:
            state.update(_extract_score_from_text(page))
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
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


def get_cached_match_state_by_steam(steam_id: int) -> Optional[Dict[str, Any]]:
    """Same as `get_cached_match_state`, but looks up the cache by the
    Steam match id (alias key written by `update_match_state_cache`).

    v0.3.24h: DLTV's `/live/{id}.json` no longer returns picks/bans
    (it ships scores + team names only).  The only source of picks
    for an in-progress series is the dltv_browser scrape, which
    writes the cache under the DLTV series id (`s{dltv_id}`).  A
    watchlist row that only knows the steam id can't find that
    cache after the tracker is pruned (which happens immediately
    when the match ends).  Writing the cache under both keys keeps
    the live card alive for the post-match "still in live cards"
    window — a few minutes where the user wants the data even
    though the match has technically ended.
    """
    return get_cached_match_state(steam_id)


def update_match_state_cache(series_id: int, url: str, steam_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Fetch + cache the live match state.  Same write semantics as
    `update_player_wr_cache`: returns None on failure and writes
    a failure marker so the next poll skips the URL.

    v0.3.24h: when `steam_id` is provided, also write the entry under
    the alias key `s{steam_id}`.  The watchlist path (which only
    knows the steam id) then finds the same data without going
    through the discovery tracker — a win for two reasons:
      1. The tracker is pruned as soon as the match ends, so
         after-match "live" cards would otherwise be empty.
      2. The dltv_id lookup currently requires walking the
         tracker under its lock, which is wasted work when the
         alias key is right there.

    The two cache entries share the same `match_state` dict and
    `ts`, so they always expire together.
    """
    cache = _read_cache()
    now = time.time()
    try:
        state = fetch_match_state(url, expected_steam_id=steam_id)
    except DiscoveryError as exc:
        log.warning("dltv_browser: match_state fetch failed for %s: %s", series_id, exc)
        # Preserve any prior rates we may have cached.
        prev = cache.get(_cache_key(series_id), {})
        prev["ts"] = now
        prev["url"] = url
        prev["match_state"] = {}
        prev["error"] = str(exc)
        cache[_cache_key(series_id)] = prev
        if steam_id is not None and int(steam_id) != int(series_id):
            cache[_cache_key(int(steam_id))] = prev
        _write_cache(cache)
        return None
    prev = cache.get(_cache_key(series_id), {})
    prev["ts"] = now
    prev["url"] = url
    prev["match_state"] = state
    cache[_cache_key(series_id)] = prev
    if steam_id is not None and int(steam_id) != int(series_id):
        cache[_cache_key(int(steam_id))] = prev
    _write_cache(cache)
    return state
