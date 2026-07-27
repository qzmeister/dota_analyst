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
MATCH_STATE_TTL_SEC = 5.0      # 5 seconds — live changes every tick
# v0.3.21: dltv.org cold load is slow when the chromium
# binary has to JIT its V8 on a small container.  8s was
# too tight — the first goto of a brand-new live match
# timed out before React could render the picks.  Bump to
# 20s (the publisher will skip the round entirely if the
# network is fully down) and let the page settle for an
# extra second after DOMContentLoaded.
PLAYER_WR_FETCH_TIMEOUT_MS = 20000
PLAYER_WR_PAGE_LOAD_WAIT_MS = 3500


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
_playwright = None
_browser = None
_browser_lock = threading.Lock()
_browser_lock_pid = None  # last pid that touched the lock; for diagnostics


def _get_playwright():
    """Return a process-wide Playwright + Chromium, or None if unavailable.

    The first caller pays the launch cost (~3s for cold chromium);
    subsequent callers reuse the same browser.  Pages are cheap to
    create per-fetch; the heavy bit (the chromium process tree)
    lives for the lifetime of the Python process and is closed at
    interpreter shutdown.
    """
    global _playwright, _browser, _browser_lock_pid
    if _browser is not None and _playwright is not None:
        return _playwright, _browser
    with _browser_lock:
        if _browser is not None and _playwright is not None:
            return _playwright, _browser
        _browser_lock_pid = os.getpid()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            log.warning("dltv_browser: playwright not installed — live player.win_rate disabled")
            return None
        try:
            _playwright = sync_playwright().start()
            _browser = _playwright.chromium.launch(headless=True)
            log.info("dltv_browser: shared chromium launched (pid=%s)", os.getpid())
            # Kick off the zombie reaper — the chromium helper
            # subprocesses can outlive the browser and become
            # zombies of PID 1 (us in the container namespace).
            _ensure_zombie_reaper()
            return _playwright, _browser
        except Exception as exc:
            log.warning("dltv_browser: chromium launch failed: %s", exc)
            # If we got the playwright manager but launch failed,
            # stop it so we don't leak the driver process.
            if _playwright is not None:
                try:
                    _playwright.stop()
                except Exception:
                    pass
                _playwright = None
            _browser = None
            return None


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


def _read_map_block_from_dom(page) -> Dict[str, Any]:
    """Read picks, bans, scores, time, teams from `.map__finished-v2`.

    DLTV's current live page renders the in-progress game inside
    a `.map__finished-v2` block (the class name is a legacy from
    when DLTV only showed finished maps; they kept the class when
    they added the live version).  Inside it:

      .team (2 of them) → .team__title-name (name + side)
                        → .team__scores-kills (per-game kills)
      .duration b       → game time "MM:SS"
      .pick (5+5=10)    → real picks, .pick__image is the hero
      .pick-sm          → secondary (smaller) picks, NOT used
      .ban (5+5=10, sometimes 12-14) → bans, .ban__image is the hero

    The hero is identified by the image URL hash; the lookup
    table comes from `window.__heroes` (see _read_heroes_from_window).

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
                    // Pull the side from the .side element's classList
                    // — class names are locale-independent ('side dire'
                    // / 'side radiant'), unlike the text content which
                    // gets translated into the user's locale (Russian,
                    // German, Chinese, ...).
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
      - picks.radiant:   list of {hero_id, steam_id, name}
      - picks.dire:      same
      - bans.radiant:    list of {hero_id, steam_id, name}
      - bans.dire:       same
      - radiant_score:   int
      - dire_score:      int
      - game_time:       "MM:SS" string
      - team_order:      ["dire", "radiant"] (or whatever the DOM shows)

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
    try:
        from playwright.sync_api import TimeoutError as PWTimeout
    except ImportError as exc:
        raise DiscoveryError(f"playwright not installed: {exc}")

    state: Dict[str, Any] = {}
    context = None
    page = None
    try:
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(url, timeout=PLAYER_WR_FETCH_TIMEOUT_MS,
                      wait_until="domcontentloaded")
        except PWTimeout:
            raise DiscoveryError(f"goto timeout: {url}")
        page.wait_for_timeout(PLAYER_WR_PAGE_LOAD_WAIT_MS)
        # v0.3.22: the new DLTV layout puts the entire live
        # state inside `.map__finished-v2` with image-hash
        # hero references resolved via `window.__heroes`.
        # Skip the React-payload walk (no longer works) and
        # go straight to the DOM read.
        state = _read_map_block_from_dom(page)
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
