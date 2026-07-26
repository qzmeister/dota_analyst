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

def _read_heroes_from_window(page) -> Dict[str, Dict[str, Any]]:
    """Read `window.__heroes` and return a hash -> hero dict.

    DLTV's live page embeds all 127 heroes as a single JS object
    `window.__heroes = { "<dltv_id>": {id, steam_id, title, slug,
    image, ...} }`.  The hero cards reference the hero image by
    URL hash like `/uploads/heroes/pethqKDQjJLPolvncSyuAOsDwWZUuUe8.png`
    — the only way to recover the dltv_id/steam_id/title from a
    rendered pick is to look it up in this table.

    Returns {} if the JS object isn't present (e.g. very early
    before hydration, or the page changed shape).
    """
    try:
        raw = page.evaluate("() => window.__heroes ? JSON.stringify(window.__heroes) : null")
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for h in parsed.values():
        if not isinstance(h, dict):
            continue
        image = h.get("image") or ""
        # image is like "/uploads/heroes/XXXX.png" — split to hash
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
                    const side = t.querySelector('.side');
                    return {
                        name: name ? (name.textContent || '').trim() : '',
                        side: side ? (side.textContent || '').trim() : '',
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

    # Resolve the team order: 'dire' if side text is the dark/dire word,
    # 'radiant' for the light/radiant word.  Russian and English both
    # handled (DLTV's server picks the locale; we don't).
    DIRE_WORDS = ("dire", "тьмы", "dark", "тeмные", "тёмные")
    RADIANT_WORDS = ("radiant", "света", "light", "сияющие")
    team_order: list = []
    for t in data.get("teams") or []:
        side_text = (t.get("side") or "").lower().strip()
        if any(w in side_text for w in DIRE_WORDS):
            team_order.append("dire")
        elif any(w in side_text for w in RADIANT_WORDS):
            team_order.append("radiant")
        else:
            # Unknown side — leave it as a placeholder; the caller
            # will fall back to the default radiant-first order.
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
