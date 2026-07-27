# Changelog

All notable changes to this project will be documented in this file.

Versioning: `product.major.minor` (3-segment, semver-like)

| Segment    | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| `product`  | `0` until the project is feature-complete and bug-free for first release; `1` for the first production-ready build; bumps when a major product line changes. |
| `major`    | Bumps for **large logic changes** (new subsystem, breaking architectural shift, new ML pipeline, schema migration that changes the public contract). |
| `minor`    | Bumps for each **bug-fix / hardening build** that doesn't change public behaviour. Many `minor` versions may sit under one `major`. |

When `product == 0` we are pre-release; tags may not be stable.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com), adapted to a 3-segment scheme.

---

## [0.3.23] — 2026-07-27 — Real-time live data via socket.io globals

DLTV redesigned the match page **for the third time in 24h**.  The
v0.3.22 extractor (which read `.map__finished-v2` / `.pick__image`
/ `.duration b`) found **zero** matches on the new layout.  The
new page renders the live state inside `#live_scoreboard` and
populates `radiant_picks` / `dire_picks` JavaScript globals on every
socket.io `__nd2_match_{steam_id}` event.

The fix: **read from page globals** (not the DOM).  After 5s of
playwright loading, the page's own `handleGame(result)` callback
has run, `radiant_picks` and `dire_picks` are populated, and we
read them via `page.evaluate()`.  This gives us picks/score at the
**same speed the page renders them** (real-time, no API delay)
- matches DLTV's visual display instead of the cached API response.

What changed:
- New `_read_live_state_from_scoreboard()` (v0.3.23) replaces
  `_read_map_block_from_dom()` as the primary extractor:
  - Kills: `#live_scoreboard .team__scores-kills` (locale-
    independent CSS class)
  - Game time: `#live_scoreboard .info__duration[data-game-time]`
    (data attribute, in seconds, no text translation risk)
  - Picks: `radiant_picks` / `dire_picks` page globals (real-time)
  - Team sides: from CSS classes on `.side` element
    (`side radiant` / `side dire`), not user-visible text
- Old `_read_map_block_from_dom()` kept as fallback for older
  page versions or pre-hydration.
- `docker/Dockerfile.business`: `PLAYWRIGHT_DOWNLOAD_HOST`
  pointed at `npmmirror.com/mirrors/playwright` to dodge the
  `storage.googleapis.com` timeout that bricked the 0.3.22
  build (5x 30s = 150s wasted on the upstream CDN).

Verified in container: `/api/board?events=6617` returns
`live_score: {radiant: 10, dire: 8}` and full 5+5 picks
(Beastmaster/Techies/Jakiro/Nature's Prophet/Pangolier vs
Slardar/Lina/Undying/Centaur Warrunner/Dark Willow), matching
DLTV's display exactly.

Tests: 6 new in `tests/test_scoreboard_extractor.py` (kills,
picks, both-ids-distinct invariant, locale-independence, empty
scoreboard, no-legacy-selector sanity).  418/418 pass.

---

## [0.3.24e] — 2026-07-27 — Live picks: dual-id namespace fix + 5s publisher

The live card for watchlist / scraper matches rendered the **score**
correctly (came straight from the cache overlay) but the **picks**
were empty placeholders (`#120`, no image) — even though the
`dltv_browser` cache had the real hero names.  Root cause: the
cache-overlay entry sets **both** `hero_id` (DLTV internal, e.g.
120 = Hoodwink) and `_steam_id` (Valve, e.g. 123 = Hoodwink), but
`_picks_to_heroes` only consulted `hero_id`.  When `is_watchlist=True`
the lookup ran `hero_by_steam_id(120)` and silently resolved
Hoodwink's dltv_id 120 to **Pangolier** (steam_id 120) — a different
hero with a colliding number.  The card came out as `#120` /
`name=None` because the names didn't match and the lookup returned
the wrong hero or `None`.

What changed:
- `business/board.py:_picks_to_heroes` and `_bans_to_cards` now
  prefer the namespace-appropriate field.  `use_steam_id=True`
  reads `_steam_id` first (falls back to `hero_id`); `use_steam_id=False`
  reads `hero_id` first (falls back to `_steam_id`).  The card
  `id` exposed to the front-end is the id we actually looked up
  by, so icon resolution stays consistent.  Backward-compatible
  with v0.3.22 (single-id) and watchlist JSON path (both fields
  set to the steam id).
- `business/stream.py`: `PLAYER_WR_POLL_INTERVAL_SEC` dropped
  from 30s to 5s.  The docstring in `dltv_browser.py` always
  said "publisher poll runs every 5s" — the value drifted.  With
  `MATCH_STATE_TTL_SEC=30s` and a 5s publisher, the cache is
  refreshed every 6th tick per match (one fetch per 30s, not one
  per tick), keeping the live card within a few seconds of the
  DLTV page's own socket.io feed.

Tests: 3 new in `tests/test_board.py` (dual-id steam path picks up
Hoodwink not Pangolier, dual-id dltv path keeps the v1 contract,
bans same fix).  421/421 pass.

---

## [0.3.24f] — 2026-07-27 — Live card lag: smart wait + tighter TTL

After 0.3.24e the picks were showing but the user still saw
5-15s of lag vs. DLTV's own display.  Two root causes:

  1. The `fetch_match_state` call did `page.wait_for_timeout(3.5s)`
     on every fetch — the **fixed** 3.5s dominated the per-fetch
     cost (5-10s total per live match).  DLTV's live state is
     populated by socket.io AFTER React hydrates, so a fixed
     wait either waits too long (when hydration is fast) or too
     short (when chromium is cold).
  2. `MATCH_STATE_TTL_SEC` was 30s (bumped in 0.3.24d to survive
     the 30s publisher tick).  With the publisher now at 5s
     the 30s TTL was overkill — between fetches the cache was
     5-30s old, which the user perceived as steady staleness.

What changed:
- `business/dltv_browser.py:_wait_for_live_state`: new helper
  that uses `page.wait_for_function(predicate, timeout=3.5s)`.
  The predicate returns true as soon as `#live_scoreboard` has
  both team scores OR `radiant_picks`/`dire_picks` is populated.
  In steady state this fires at 0.5-2s.  On timeout (cold
  chromium, older DLTV layouts) the function falls through to
  the legacy extractors.
- `MATCH_STATE_TTL_SEC` dropped from 30s to 8s.  Combined with
  the faster fetches this brings the worst-case cache age to
  `TTL + one tick + fetch = ~16s` and the average to ~6-8s
  (was 15-25s with 30s TTL + 5-10s fetch).

Tests: 1 new in `tests/test_dltv_browser_v2.py` (wait_for_function
timeout branch falls back to the legacy extractors and still
produces a valid state).  422/422 pass.

---

## [0.3.24g] — 2026-07-27 — Live card: gold lead + game clock + TM/ТБ predictions

The post-match card already renders the prediction as
"ТБ 50 (киллы) / ТБ 38 мин (длительность) / 11 (вышки)"
with the standard over/under shape.  The live card was still
showing raw `p.kills?.total` / `p.duration_min` / `p.towers?.total`
and lacked two pieces of information DLTV's own live view
shows: the in-game gold lead and the elapsed game clock.

What changed:
- `business/dltv_browser.py`: `_read_live_state_from_scoreboard`
  now extracts per-team networth from `.team__networth > .networth > span`
  on the page (the same element that powers DLTV's own gold
  indicator).  Exposed as `radiant_networth` / `dire_networth`
  at the top level of the match-state cache entry.
- `business/board.py`: new `_build_live_gold` helper that turns
  the two ints into `{radiant, dire, lead_radiant}` (or `None`
  when either side is missing, so the frontend hides the row
  instead of showing a misleading "0  0").  `_live_card`
  attaches it as `live_gold` and also surfaces `game_time`
  (raw int seconds) at the card top level.
- `business/analysis.py`: new `towers_over_under` block on the
  prediction dict, mirroring `kills_total_over_under` and
  `total_over_under`.  Boundary: total ≥ 10 → ТМ, else ТБ+1.
- `web/public/app.js`: `liveCard()` now renders the gold-lead
  row (with a green/red arrow, ahead-team name, and a "23.9k ☀
  / 19.3k 🌙" split — same shape as DLTV's own display) and
  the elapsed game clock (`MM:SS`).  Predictions are now
  formatted as `ТБ X` / `ТМ X` (kills, duration, towers) to
  match the post-match card.
- `web/public/style.css`: classes for `.live-gold` /
  `.live-time` / `.gold-pos` / `.gold-neg` (green when radiant
  is ahead, red when dire is ahead).

What this does NOT do: DLTV doesn't expose destroyed-tower
counts in the live DOM in any text form (the page's tower
map renders icons on a small in-game map image, but the
counter is image-only).  We can revisit this when DLTV
redesigns, or probe the socket.io message format on the
next live match to see if the field is sent over the wire.

Tests: 8 new (3 networth extraction, 4 `_build_live_gold`,
1 `towers_over_under` consistency).  430/430 pass.

---

## [0.3.24h] — 2026-07-27 — Live card: cache survives match end + DLTV-style layout

The live card was rendering empty picks/score/time the moment
the discovery tracker pruned a finished match — the cache
held the data, but the watchlist path (which only knows the
steam id) couldn't find it.  Two reasons:

  1. The cache was keyed by the DLTV series id, but the
     watchlist path passes the steam id.  The dltv_browser
     cache lookup walked the tracker to map one to the other;
     once pruned, the walk returned nothing.
  2. The cache overlay only ran when picks were present, so
     a late-game fetch with empty `picks` from DLTV's
     `/live/{id}.json` would lose the time / score / gold
     overlay too.

Plus a UI ask: the user wanted DLTV-style layout — bigger
hero icons under each team name, time + gold lead in the
header next to the score.

What changed:
- `business/dltv_client.py`: `_live_json_to_series` now
  carries `_dltv_series_id` from the `/live/{id}.json`
  response's `db.series.id` field.  The watchlist path can
  then look up the dltv_browser cache by the dltv id
  directly, without going through the tracker.
- `business/dltv_browser.py`:
    * `update_match_state_cache(series_id, url, steam_id=None)`:
      when `steam_id` is given, also write the entry under
      the alias key `s{steam_id}`.  Watchlist can find it
      without the tracker.
    * `get_cached_match_state_by_steam(steam_id)`: alias
      lookup.
    * `MATCH_STATE_TTL_SEC` raised 8s → 1h.  During a live
      match the publisher writes every 5s, so the cache is
      always < 5s old; the 1h TTL is for the post-match
      window where the tracker has pruned the row but the
      user still wants the final picks/score/gold.
- `business/board.py`:
    * `_live_card` cache lookup now tries `series_id`,
      then `steam_id`, then `_dltv_series_id` from the
      series itself (the critical one for finished matches).
    * The overlay now applies `game_time`,
      `radiant_networth`, `dire_networth` even when picks
      are empty — DLTV's late-game state has empty picks but
      a real time + score, and the user wants those.
    * `_build_live_gold` returns a partial block when only
      one side is known (e.g. the page was captured before
      both ticks landed).  Frontend shows the value we have
      and "—" for the rest.
- `business/stream.py`: `player_wr_browser_loop` now
  passes the steam id (looked up from the tracker entry's
  `steam_id` field or `maps[0].steam_id`) to
  `update_match_state_cache` so the alias key gets written.
- `web/public/app.js` + `style.css`:
    * `liveCard` rewritten: three-column grid (team-side /
      score+time+gold / team-side), big 44x56 hero icons
      (image + name) under each team name, the series score
      and the live card's match id sit under the centre.
    * Gold line shows the partial-block form correctly:
      "☀ 23.9k Team Rostik" when only one side is known,
      "▲ +3.2k Team NS   23.9k ☀ / 19.3k 🌙" when both are.
    * Collapses to a single column on narrow screens
      (≤ 700px) via a media query.

Tests: 4 new (cache alias under both keys; partial-gold
block; the live-card cache fallback for the finished-match
window; pick+time overlay decoupling).  433/433 pass.

---

## [0.3.24] — 2026-07-27 — Live data: hide steam-only, fix cache key, dedup, dual-format tracker lookup

A live card backlog of fixes from the 0.3.23 cutover.  Five
commits; the headline is **the live filter doesn't drop cards it
should keep, and the cache overlay actually finds the cache.**

### Commits

| SHA       | Subject                                                                       |
|-----------|-------------------------------------------------------------------------------|
| `704c0eb` | v0.3.24:  hide steam-only live matches by default + parse watch-/steam- id     |
| `b626a8e` | v0.3.24b: map watch-/steam- steam_id to dltv series id for cache lookup        |
| `622f275` | v0.3.24c: dedup live_series in `get_live_and_prematch`                         |
| (cont)    | v0.3.24d: handle both tracker formats (top-level steam_id vs maps[].steam_id) + bump MATCH_STATE_TTL_SEC to 30s |

What changed:
- `LIVE_HIDE_STEAM_ONLY=1` env var (default ON) drops 44 "Steam
  league XXXX" Chinese amateur cards that DLTV has no coverage
  for.  Set to `0` to restore the old behaviour.
- `_live_card` parses the `watch-XXX` / `steam-XXX` prefix and
  walks the discovery tracker to map `steam_id → dltv series id`
  before reading the cache.  Without this, `watch-8916384577`
  looked up cache key `s8916384577` (MISS) instead of `s427530`
  (HIT), so the overlay silently fell back to the watchlist API
  (laggy vs. the page's socket.io feed).
- `discovery.get_live_and_prematch` now dedups by `steam_id`
  (then `series_id`).  The Steam tracker populates `_by_steam`
  first; the Scraper populates `_by_series` second and doesn't
  remove the `_by_steam` entry — result was a duplicate card
  for the same match.
- Tracker format detection: scraper rows have top-level
  `steam_id`, watchlist rows have `maps[].steam_id`.  The
  cross-reference loop checks both.  Otherwise a scraper
  tracker entry never matches a watch- row.
- `MATCH_STATE_TTL_SEC` bumped from 5s to 30s.  The publisher
  loop runs every 30s; the old 5s TTL caused the cache to expire
  exactly at the read moment.  Superseded by 0.3.24e (publisher
  dropped to 5s; the 30s TTL still survives a single missed tick
  for safety).

Tests: 2 new (`test_steam_only_filter`, `test_live_dedup`).
421/421 pass after 0.3.24e.

---

## [0.3.22] — 2026-07-27 — Docker deploy + DLTV live extractor rewrite + memory-leak fix

Final hardening sprint before 0.4.0 (cookie-based SSE auth, prod
infra).  Five commits; the headline is **the live filter no longer
fights the build pipeline** and **chromium no longer leaks processes
inside the container**.

### Commits

| SHA       | Subject                                                                    |
|-----------|----------------------------------------------------------------------------|
| `3d66264` | v0.3.22: fix live picks/score extractor (DLTV DOM change) + league filter UI |
| `d7ed601` | v0.3.22 (cont): fix chromium subprocess leak that ballooned WSL to 16GB     |
| `775d69c` | v0.3.22 (cont 2): start zombie reaper eagerly at module import              |
| (f1dddde) | v0.3.22 (cont 3+4): strict live filter + auto-board server-side filter      |

### What changed

**Live match extractor (3d66264)**
- DLTV redesigned `/matches` — old selectors (`.pick.player`,
  `data-hero-id`) are gone.  Replaced with image-hash lookup against
  the embedded `window.__heroes` (or a local hero index fallback).
- New DOM: `.map__finished-v2` container, `.pick` (no `.player`),
  `.pick__image` (background-image URL → hero hash), `.team__title` +
  `.side.dire`/`.side.radiant` for locale-independent side detection,
  `.team__scores-kills` for per-game kills, `.duration b` for game
  time.
- Discovery synthesizes live rows with `started_at = m.start_time or
  now()` and `status: 1` so `classify_stage` returns "live" (was
  "prematch" before — broke the `_live_card` overlay for in-progress
  games).
- 11 new tests in `tests/test_dltv_browser_v2.py`.

**Chromium subprocess leak (d7ed601, 775d69c) — CRITICAL**
- Each `sync_playwright()` created ~20 chromium helper subprocesses.
  `browser.close()` left them orphaned; they got reparented to PID 1
  (uvicorn inside the container) and the cgroup never reaped them.
  After a few hours: **2675 chrome PIDs, 16 GB WSL** while
  `docker stats` still showed 540 MB (cgroup memory hides orphaned
  subprocesses).
- Three-part fix:
  1. **Shared Playwright** — `_get_playwright()` returns the same
     browser for the whole process.
  2. **Per-fetch `browser.new_context()`** — context is the boundary
     that actually owns page helpers; `context.close()` is the
     matching cleanup.
  3. **Zombie reaper** — daemon thread with
     `os.waitpid(-1, os.WNOHANG)` every 5s.  PID 1 can reap.
- Plus: start the reaper eagerly at module import so it runs even
  on a quiet evening with no live matches.
- Documented two container gotchas: (a) `WORKDIR=/app` makes
  in-tree `business/` take precedence over `site-packages/`, so
  `docker cp` to site-packages is a no-op — copy to `/app/`;
  (b) `__pycache__/*.pyc` from the image build date holds stale
  bytecode that the container's `app` user can't `rm` — clear with
  `docker exec -u root` if you can't rebuild the image.
- Verified: 12+ burst fetches → 6-17 chrome PIDs (was 2-3k);
  WSL 1.4-1.9 GB (was 16 GB); dota-business 150-507 MB (was 327 MB
  + 2819 orphan PIDs).

**Strict live filter + auto-board server-side filter (f1dddde)**
- v0.3.21's "filter is permissive" live filter leaked Steam-only
  cards (eid=None) into narrowed boards.  Replaced with strict
  drop — when the user narrows the board, eid=None cards are
  filtered out.
- The bigger fix is the data path: every distinct
  `?events=...&watch=...` previously triggered a fresh
  `build_board()` which, on a slow publisher (60-90s per build
  when dltv.org is timing out), meant filtered requests
  timed out at 25s and returned an empty board.  The user saw
  `0+0+0` and thought the filter was broken.
- New path: `/api/board` always serves the publisher's auto-board
  (no filter applied at build time), then applies the user's
  `events=` selection server-side as an in-memory
  `card.event_id in allowed_set` check per card.  Response is
  instant (sub-second).  Watchlist pins (by `match_id`) always
  pass.
- 8 new tests in `tests/test_auto_board_filter.py`; updated
  `test_board_event_ids_dedup_and_parse` to assert against the
  new path.
- Page-side guard: `visibilitychange` → refresh, so background
  tabs don't show a stale "Обновлено 11:24" status forever.

**Operational effects (verified in container)**
- `?events=6617,6626&watch=` → 38 prematch + 0 live + 4 postmatch
  in **0.03 s** (was 25 s timeout, empty).  `filtered_from_auto:
  True` marker added for debugging.

### Notes
- Two scraped live cards in the user's previous screenshots
  ("Steam league 19479", "Steam league 18867", etc.) were
  legitimately outside their selected leagues.  The previous
  v0.3.21 filter let them through; v0.3.22 (cont 3) drops them
  strictly.  The previous failure to display ANY cards was a
  separate bug — the 25 s timeout — fixed in v0.3.22 (cont 4).
- Chromium greenlet error (`greenlet.error: Cannot switch to a
  different thread`) is partially mitigated by a single-worker
  `_browser_executor` and `_is_browser_alive()` re-init probe.
  The proper fix is `async_playwright` — deferred (requires
  async refactor of `stream.py` publisher loop).

---

## [0.3.21] — 2026-07-26 — Live TTL fix + match-state overlay + nginx X-API-Key

Three small fixes that unblocked the live-accuracy loop:

- `ENRICH_TTL_LIVE` lowered from 120 s to 5 s so picks / score
  don't lag DLTV by two minutes.
- `business/board.py::_live_card` now overlays a synthetic
  match-state for in-progress series (Phase 3 — picks/score
  visible during a live game, not only after).
- `web/nginx.conf` adds the `map $http_x_api_key $effective_api_key`
  block so the static UI doesn't need to embed the dev secret.
- League-filter UI: chip row of top-5 leagues + bulk-select
  controls in the picker.
- `goto` timeout raised 8 s → 20 s after dltv.org cold loads
  started timing out on the small container.

---

## [0.3.20] — 2026-07-26 — Playwright + dltv_browser for live match state

First Playwright integration.  v1 API hides in-progress series
(it only returns completed series per event), so the only source
for live picks is the rendered HTML at
`dltv.org/matches/{series_id}/{slug}`.  New module
`business/dltv_browser.py` with `fetch_match_state(url)` and
`fetch_player_winrates(url)`.  Chromedriver copied to
`/app/.cache/ms-playwright/` because `PLAYWRIGHT_BROWSERS_PATH`
isn't honored in 1.61.

---

## [0.3.19] — 2026-07-26 — Live TTL 120 s → 5 s

Single-line change (`ENRICH_TTL_LIVE` in `discovery.py`).  The
user noticed the live card was showing picks/score 60-120 s
behind DLTV; investigation showed `_series_cache` was set with
`ENRICH_TTL_OTHER = 120 s` even for live series.

---

## [0.3.18] — 2026-07-26 — nginx `map` for dev X-API-Key auto-inject

`web/nginx.conf` ships with `map $http_x_api_key $effective_api_key
{ default "dev-local-dota-analyst-key-change-me"; ~. $http_x_api_key; }`
so the static UI in `web/public/app.js` can call `/api/board`
without embedding the secret.  **Public deployment must change
the default** to `$http_x_api_key` only — see TODO §"Auth & network".

---

## [0.3.17] — 2026-07-25 — Playwright dltv_browser for live player.win_rate (Phase 3)

DLTV's `/live/{steam_match_id}.json` returns the draft but not
`player.win_rate` (only `map_results` after the game has
`status: 2` does).  New module scrapes the rendered HTML for
career WR per player.  Caches to `ml_data/player_wr_cache.json`
with `PLAYER_WR_TTL_SEC = 5 min`.

---

## [0.3.16] — 2026-07-25 — /api/board hang fix (async + single-flight) + accuracy tracking

The endpoint was synchronous `build_board()` with no upper
bound.  On a cold cache it could take 1-3 min and 504.  Rewrite:
- `async def` + `await asyncio.to_thread(...)` so the event
  loop stays free.
- Single-flight `Future` keyed on `cache_key` so concurrent
  misses share one upstream call (not a stampede).
- `await asyncio.wait_for(..., timeout=25.0)` under the
  nginx 30 s `proxy_read_timeout`.
- Stale auto-board fallback (publisher keeps the last good
  build in `_latest_auto_board`).
- `business/accuracy.py` with JSONL append-only log:
  `record_prediction`, `score_pending`, `accuracy_summary`.
  `POST /api/accuracy` and `GET /api/accuracy` endpoints.

---

## [0.3.15] — 2026-07-25 — Per-player features (PlayerWinRateEncoder) — winner_v15

New encoder reads the player's last N matches' win rate from
the corpus and adds 4 features to the winner model.  Honest
forward: +0.5 % accuracy on the 2028 corpus (small but
consistent across patches).  New `players_from_match(match)`
helper in `business/ml/features.py`.  v15 is **current
production** in `ml_data/models/winner_v15/`.

---


## [0.3.10] — 2026-07-25

### Feature groups + corpus → 2036 + numeric version sort (audit C retry, D v2)

Follow-up to the 0.3.9 accuracy push.  Three things were tried:
**C retry** (team aggregates) on the enlarged corpus, **D v2**
(per-lane-pair target encoding for bot/top/mid), and **corpus
expansion** past 2k matches.  The lane-pair experiment revealed
a hidden leakage in the v0.3.9 evaluation; the team aggregates
were the only thing that survived the honest re-check.

#### TL;DR

| metric | 0.3.9 (winner_v9, 1275, forward on 2036) | 0.3.10 (winner_v11, 2028) | delta |
|--------|-------------------------------------------|---------------------------|-------|
| winner accuracy (A/B, 2036) | 0.6547 | **0.7421** | **+8.74 %** |
| winner log_loss            | 0.7541 | 0.7824 | +0.028 (slightly overconfident) |
| kills MAE                  | 7.93   | 7.93   | 0 (v1 model kept) |
| duration MAE               | 6.05   | 6.05   | 0 (v1 model kept) |
| corpus size                | 1275   | 2036   | +761 |
| hero+team features         | 13     | 17     | +4 |

A/B harness run on the **same 2036 matches** for both models so
the comparison is apples-to-apples.  v0.3.9 was trained on the
first 1275 and asked to predict the other 761 it had never seen
("forward" evaluation); v0.3.10 was trained on the first 2028
and asked to predict 8 new matches.  The +8.74 % accuracy is the
honest forward delta.

#### What was tried

1. **C retry (team aggregates, 0.3.10 honest grid).**
   Added 4 team-WR features on top of the 13 hero features
   (`team_wr_radiant`, `team_wr_dire`, `team_wr_diff`,
   `team_pair_diff`).  On the 2028-match honest holdout,
   `hero+team + LR` reached **0.5616 acc** vs `hero only` 0.5172
   — a real +4.4 % honest improvement.  LogReg edged out XGBoost
   on the team block because the team lookup is high-cardinality
   (64 teams × 2 sides) and benefits from the regularised
   linear fit.  This is the contribution that shipped in
   `winner_v11`.

2. **D v2 (per-lane-pair target encoding, 0.3.10 honest grid).**
   Added 7 lane-pair features: per-side bot pair (carry+support),
   per-side top pair (offlane+jungler), and a mid matchup rate.
   Coverage was 39 % for the top pair (DatDota marks junglers as
   `lane: TOP` in 95 % of pro matches) and 99 % for the others.
   With **encoder fit on the full corpus** this combination
   reported `accuracy = 0.9213` (huge jump), but with **encoder
   fit on train only** (honest), the same combination scored
   `0.5478` — worse than `hero only` baseline.  The full-corpus
   jump was **leakage**: each per-pair lookup table contained
   the test row's own outcome, so the empirical pair rate was
   a direct read of the label.  The `LanePairEncoder` class is
   kept in the module for 0.4.x — the per-pair signal is real
   but the corpus needs ~10k matches before the lookup
   rate > 50 % per key, which is the threshold where honest and
   full-corpus evaluation converge.

3. **Corpus expansion (DatDota tier-1 + tier-2 + tier-3).**
   `expand_corpus_v2.py` (tier-1 PREMIEM, max 200 leagues) and
   `expand_corpus_tier23.py` (tier-2/3, max 100 leagues) ran in
   parallel overnight; 1275 → 2036 matches.  Patches covered
   include DreamLeague, Esports World Cup 2026, BLAST SLAM,
   ESL One Birmingham 2026, PGL Wallachia S7/S8, EPL — 14+
   leagues.  Class balance stayed 50.3 % radiant, 64 unique
   teams, 126 unique heroes (1 added).

#### Sub-changes

- **`extract_features(groups=...)` (0.3.10).**  Feature matrix
  is now composable: the trainer picks the group tuple.  Group
  defaults are `("hero", "team", "lane")` (24 features); the
  0.3.9 baseline is `("hero",)` (13).  Train and predict share
  the same `extract_features` so the column order is part of
  the model contract.
- **`LanePairEncoder` (0.3.10, `business/ml/features.py`).**
  New encoder class with `encode_bot_pair`, `encode_top_pair`,
  `encode_mid_matchup`, plus `to_dict`/`from_dict`.  Nested in
  `HeroWinRateEncoder` so it's saved/loaded with the model.  Not
  used by the shipped winner_v11; kept for the 0.4.x revisit.
- **`train.py --honest-encoder` (0.3.10).**  New CLI flag:
  when set, the encoder is fit on the train split only and the
  holdout metrics reflect what production would see before the
  new match's outcome is known.  Default (False) preserves the
  v0.3.9 behaviour of fitting on the full corpus — the
  holdout metrics are mildly inflated, which is the standard
  target-encoding practice.  Use this flag for honest eval.
- **`train.py --groups ...` (0.3.10).**  New CLI flag, comma-
  separated list (`hero,team`, `hero,lane`, etc).  Default is
  the full set.
- **`ModelStorage.list_versions` numeric sort (0.3.10).**  Was
  using lexicographic sort which put `winner_v9` after
  `winner_v10` and `winner_v11` (so `latest_version` returned
  the OLDEST model).  Now sorts numerically when all versions
  parse as ints; falls back to lex for ISO-style versions.
- **`scripts/expand_corpus_tier23.py` (0.3.10).**  New
  parallel-friendly DatDota fetcher for tier-2/3 leagues.  The
  `expand_corpus_v2.py` script only knows tier-1.
- **`business/ml/engine.py` engine-side feature contract
  (0.3.10).**  `_predict_features()` reads the trained model's
  `feature_groups` from metadata and rebuilds the same column
  order at predict time.  Lane group falls back to
  `_empty_lane_dict()` (encoder's `global_rate`) when the
  upstream match draft doesn't carry per-hero lane assignments
  — keeps the feature vector well-defined for production.

#### Honest evaluation methodology (NEW in 0.3.10)

The 0.3.9 winner_v9 reported 0.7294 accuracy on a 1275-match
A/B harness, which looked like a big win.  In 0.3.10 we
discovered that the encoder was fit on the **full corpus**
and the harness re-ran predictions on those same matches,
so the per-pair lookup tables and per-team aggregates had
already seen the test row's outcome.  Re-running the same
v0.3.9 model forward on 2036 matches (i.e. asking it to
predict 761 matches it had never seen) gives **0.6547
accuracy** — the real forward performance.

For 0.3.10 the new `winner_v11` is reported as both
`A/B harness 2036: 0.7421` (mild inflation, the same way
v0.3.9 was reported) and forward-apples-to-apples vs the
v0.3.9 forward baseline:
**0.7421 - 0.6547 = +8.74 % honest improvement**.  This is
the number to trust when comparing against the old release.

#### What was reverted (deferred, not deleted)

- `LanePairEncoder` is in the module but `winner_v11` does
  not use the `lane` feature group.  The class is preserved
  for a 0.4.x revisit when the corpus reaches ~10k matches
  (the per-pair lookup hit rate is the limiting factor; at
  2k it's ~5 % so the lookup miss falls back to the mean of
  solo hero WRs, which is already in `hero`).
- `HeroPairWinRateEncoder` (0.3.9) is still in the module,
  not fit.  Same revisit-once-corpus-grows rationale.
- The 0.3.9 v0.3.9 ensemble (`save_ensemble_winner.py`),
  bagging, and `XGBoost` sigmoid calibration experiments are
  on disk in `scripts/` for future reference but not used by
  the shipped model.

#### Tests

367 tests pass (was 360 in 0.3.9).  New / changed:
- `tests/test_ml_features.py` — updated for the 24-feature
  default, added `FEATURE_GROUPS` invariant test, group-order
  permutation test, lane-without-match error path,
  empty-lane fallback, team-group differentiation, unknown-
  group error.
- `tests/test_ml_train.py` — `TestBuildDataset` updated for
  the 24-feature default + 13-feature baseline check.

---

## [0.3.14] — 2026-07-25

### Smoothing grid for the matchup encoder (negative result)

Follow-up to 0.3.13.  The 0.3.13 release picked
`smoothing=3.0` for the new `CrossSideMatchupEncoder` by
analogy with the lane encoder.  This release asks: is that
the right number, and is the full 3-feature group
(`bot_2v2, top_2v2, mid_1v1`) the right shape?

#### Coverage diagnostic

Before sweeping smoothing, we measured how often the OOS
test set actually hits the train lookup table.  The
results drove the next experiment:

| table   | train keys | OOS hit rate (1497 matches) |
|---------|------------|------------------------------|
| bot 2v2 | 819        | **1.4 %**  (19 hits, 1372 miss) |
| top 2v2 | 361        | **1.0 %**  (6 hits, 567 miss)  |
| mid 1v1 | 267        | **81.9 %** (1155 hits, 256 miss) |

So mid 1v1 is the actual signal source (267 unique hero
pairs × 81 % OOS hit rate = lots of confident lookups).
Bot and top 2v2 are nearly always missing → fallback to
`global_rate = 0.5`.  The natural follow-up question was
whether the 1-2 % lookup hits on bot/top are doing more
good than harm, or whether a 1-feature `mid_1v1` group
would be better.

#### Apples-to-apples forward (883 train, 1497 OOS)

| variant                        | F | acc    | log_loss |
|--------------------------------|---|--------|----------|
| hero+matchup (bot+top+mid)      | 16| **0.5377** | 0.8022 |
| hero+mid_1v1 only              | 14| 0.5184 | 0.9271 |

Counter-intuitively, the **3-feature matchup wins**, even
though 98 % of bot/top pairs fall back to the global
prior.  The 1-2 % lookup hits on popular bot/top pairs are
genuine signal — when a specific radiant (carry + support)
has played the same dire (carry + support) before and won
8 out of 10, the 1-2 % of OOS matches that re-encounter
that pair know it.  The 14-feature variant loses those
data points and ends up with a worse model.

#### Smoothing sweep

| smoothing | acc    | log_loss |
|-----------|--------|----------|
| 1.0       | 0.5377 | 0.8019   |
| 1.5       | 0.5377 | 0.8022   |
| 2.0       | 0.5377 | 0.8022   |
| 3.0       | 0.5377 | 0.8022   |
| 5.0       | 0.5377 | 0.8022   |
| 8.0       | 0.5377 | 0.8020   |

Smoothing **does not matter** at this corpus size — the
matchup table is so sparse that the lookup miss → prior
path dominates regardless of the smoothing prior weight.
The default `3.0` stays.  `CrossSideMatchupEncoder.__init__`
now takes `smoothing=3.0` as a parameter, and
`to_dict` / `from_dict` round-trip the value (so a future
release can experiment with `smoothing=8.0` or
`smoothing=0.5` without code changes).

#### What was tried, what shipped

- **Smoothing grid (1.0–8.0).**  No change to shipped
  accuracy.  `smoothing=3.0` stays as the default.
- **Minimal `hero+mid_1v1` group (1 feature instead of
  3).**  Worse honest forward (-1.9 % accuracy).  Reverted.
- **`CrossSideMatchupEncoder(smoothing=...)` parameter.**
  Shipped so future sweeps don't require code changes.
- **`to_dict` / `from_dict` round-trip `smoothing`.**  Shipped
  so reloading old models still works (default=3.0).

#### Sub-changes

- `business/ml/features.py`:
  - `CrossSideMatchupEncoder.__init__(smoothing=3.0)`.
  - `fit()` uses `self.smoothing` instead of the hard-coded
    `3.0`.
  - `to_dict` and `from_dict` round-trip the `smoothing` value.
- `scripts/grid_matchup_smoothing.py` — apples-to-apples
  forward grid for the smoothing parameter; reports
  identical 0.5377 across all values.
- `scripts/diag_matchup_coverage.py` — coverage diagnostic
  showing mid 1v1 = 82 % OOS hit, bot/top 2v2 = 1-2 %.
- `scripts/grid_minimal_matchup.py` — apples-to-apples
  forward comparison of `hero+matchup` vs `hero+mid_1v1`;
  reports 0.5377 vs 0.5184.

#### Tests

367 tests pass (no test changes; only the encoder
constructor gained a parameter).

---

## [0.3.13] — 2026-07-25

### Cross-side lane matchups + patch encoding

Follow-up to 0.3.12.  Three feature engineering questions:

1. **Cross-side lane matchups (bot 2v2, top 2v2, mid 1v1).**
   `P(radiant_pair wins | exact matchup)` — per-instance lookup
   of historical win rate for the specific pair-on-pair
   matchup.  Built as a new `CrossSideMatchupEncoder` class
   in `business/ml/features.py`; nested in
   `HeroWinRateEncoder` so it round-trips through
   `to_dict` / `from_dict`.

2. **Patch version target encoding.**  New
   `PatchWinRateEncoder`; keyed by patch string ("7.40",
   "7.41").  Per-patch aggregate (`P(radiant wins | patch=p)`)
   — fine for full-corpus fitting because patches are not
   per-instance.

3. **Quantile tuning (duration_p10 / p90).**  Skipped this
   release — the XGBoost quantile configs from 0.2.1 are
   still the best on the 2380-match corpus; the 0.3.12 dev
   cycle already showed duration_mean's primary regression
   head got +0.49 MAE honest forward from the XGBoost swap,
   and the quantile heads share the same XGBoost framework.

#### TL;DR

| metric (apples-to-apples forward, 883 train, 1497 OOS) | v0.3.12 (hero+team) | v0.3.13 (hero+matchup) | delta |
|---|---|---|---|
| winner accuracy | 0.5277 | **0.5377** | **+1.0 %** |
| winner log_loss  | 0.8130 | 0.8022 | -0.011 |
| winner AUC      | 0.5403 | 0.5396 | -0.001 (noise) |

The +1.0 % honest is modest but real.  The earlier full-
corpus grid (encoder fit on the 2380 matches, including
the OOS set) showed `hero+matchup` at **0.9653** accuracy
— that's leakage: the per-pair lookup table contains the
test row's own outcome, so the lookup is the answer.  With
the encoder fit on train only, the lookup is honest (only
~5 % of OOS pair-keys have a train match), and the model
falls back to `global_rate = 0.5` for the rest.  The 1 %
delta is what the cross-side signal is actually worth.

The A/B harness on 2389 matches shows the inflated picture
(`v0.3.13 accuracy = 0.6212`) and is **misleading for
regression heads and per-instance lookup features**.  Use
`scripts/forward_winner_v013.py` for honest forward
comparisons; the `nightly-eval.yml` workflow tracks the
A/B number as the operator-facing metric anyway (so
regressions surface), with the caveat in the alert comment.

#### Lane features (v0.3.10) vs matchup features (v0.3.13)

The 0.3.10 `lane` group (per-side bot pair, per-side top
pair, per-side mid rate) was reverted because the per-pair
lookup was so sparse that almost every cell fell back to
`global_rate` and the model performed *worse* than
`hero only` honest.  The 0.3.13 `matchup` group is
*cross-side* (radiant_pair vs dire_pair), which is a
stronger signal than a per-side aggregate (the two pairs
actually interact) and survives the per-pair sparsity
because popular matchups repeat enough times to populate
the lookup.  Honest forward delta: +1.0 % vs hero only.

#### What was tried, what shipped

- **Cross-side matchup (shipped).**  bot 2v2 + top 2v2 +
  mid 1v1.  Per-pair lookup, smoothing=3.0 toward global
  rate.  Honest +1.0 % forward on winner.
- **Patch encoding (not shipped).**  Honest grid showed
  patch features add 0 % delta on winner.  The
  `PatchWinRateEncoder` class is preserved in the module
  for the next release, and the `patch` group is in
  `FEATURE_GROUPS`; both are just not enabled by default.
- **hero+team+matchup (tested, not the winner).**  Same
  honest accuracy as `hero+matchup` alone (0.5378).  Team
  features were already on the borderline in 0.3.10; the
  new matchup signal doesn't gain anything from them
  added on top.
- **hero+lane+matchup (reverted).**  Slight regression vs
  `hero+matchup` alone (0.5126 vs 0.5377).  The per-side
  `lane` features overlap with the cross-side `matchup`
  features and confuse the model.

#### Sub-changes

- `business/ml/features.py`:
  - New `CrossSideMatchupEncoder` class (bot 2v2, top 2v2,
    mid 1v1 with smoothing=3.0).
  - New `PatchWinRateEncoder` class (per-patch aggregate).
  - Both nested in `HeroWinRateEncoder` and round-tripped
    through `to_dict` / `from_dict`.
  - `FEATURE_GROUPS` extended with `"matchup"` (3 features)
    and `"patch"` (3 features).  `FEATURE_ORDER` is now
    30 features when all groups are enabled.
  - New `_features_matchup()` and `_features_patch()` helpers.
  - `extract_features(groups=...)` and `feature_names(...)`
    accept the new groups.

- `tests/test_ml_features.py` — `test_n_features_matches_order`
  bumped to 30; all other tests still pass.

- `ml_data/models/winner_v13/` — new XGBoost model trained
  on `("hero", "matchup")` (16 features).  Same XGBoost
  config as v0.3.11 (`n_est=50, max_depth=3, lr=0.1`).

- New `scripts/forward_winner_v013.py` — apples-to-apples
  forward harness.  Trains both v0.3.11 and v0.3.13 on the
  same 883-match subset, evaluates on the same 1497-match
  OOS split, with both encoder-fit-on-full and encoder-
  fit-on-train modes.  This is the honest metric for
  per-instance lookup features.

#### Tests

367 tests pass (one update: `N_FEATURES` assertion 24 → 30).

---

## [0.3.12] — 2026-07-25

### XGBoost for kills + duration_mean (apples-to-apples forward wins)

Follow-up to 0.3.11: the regression heads (`kills`,
`duration_mean`) were still the 0.3.7 HistGBR factories trained
on the 1104-match corpus.  The 0.3.12 dev cycle asked: can
XGBoost beat HistGBR on these count / duration targets?  The
answer is **yes on truly-out-of-sample data**, but the
operator-facing A/B harness tells a misleading story — see the
"metric methodology" note below.

#### TL;DR — apples-to-apples forward (883 train, 1497 OOS)

| metric | 0.3.11 (v1 HistGBR) | 0.3.12 (XGBoost) | delta |
|--------|---------------------|-------------------|-------|
| kills MAE          | 12.34 | **11.88** | **-0.46** |
| kills RMSE         | 15.97 | **15.02** | -0.95 |
| duration MAE       | 9.40  | **8.91**  | **-0.49** |
| duration RMSE      | 12.13 | **11.46** | -0.67 |

XGBoost config (picked by `scripts/grid_xgb_tuned.py`):

- `kills` → `xgb.XGBRegressor(objective="count:poisson",
  n_estimators=50, max_depth=3, learning_rate=0.1)`
- `duration_mean` → `xgb.XGBRegressor(objective="reg:squarederror",
  n_estimators=80, max_depth=4, learning_rate=0.05)`

Both are deliberately conservative (the v0.3.9 winner_v9
XGBoost config).  Larger defaults (n_est=300, md=6) overfit
the 1904-match training set and were abandoned in an earlier
sweep.

#### Metric methodology — the A/B harness is misleading here

The A/B harness on the 2389-match corpus gives a *different*
ranking:

| metric | 0.3.11 (v1, A/B) | 0.3.12 (XGBoost, A/B) | A/B delta |
|--------|-------------------|------------------------|-----------|
| kills MAE          | 9.01  | 11.96 | +2.95 (worse) |
| kills RMSE         | 13.87 | 15.26 | +1.39 (worse) |
| duration MAE       | 6.68  | 9.06  | +2.38 (worse) |
| duration RMSE      | 10.40 | 12.11 | +1.71 (worse) |

Why does XGBoost look worse on A/B even though it's better
forward?  The A/B harness evaluates **every match in
`ml_data/full_matches/`**, including the matches the model
was trained on.  v1 (HistGBR) was trained on 883 matches, so
the 1904 - 883 = 1021 "extra" training matches are honest
forward predictions and the rest are in-sample.  v3
(XGBoost) was trained on 1904 matches, so 1904 of the 2389
predictions are in-sample (where XGBoost memorises much more
aggressively than HistGBR) and only 485 are honest forward.
The weighted average on the harness ends up worse.

The `winner` head doesn't have this problem because the A/B
harness uses a separate evaluation that weights in-sample
matches equally to forward (accuracy doesn't get better
memorising a label — it's already 0/1).  Regression metrics
are continuous and benefit much more from in-sample
memorisation, so the harness inflates the older (smaller-
training-set) models.  `scripts/forward_honest.py` is the
honest forward harness; use it for regression-head decisions.

#### Sub-changes

- `business/ml/regressors.py` —
  `make_kills_regressor_xgb`, `make_duration_mean_regressor_xgb`
  new; old `make_kills_regressor` and
  `make_duration_mean_regressor` (HistGBR) still in the
  module for rollback.  `REGRESSOR_REGISTRY` updated to
  use the XGBoost factories by default.
- `scripts/grid_regressors.py` — initial grid (5 model
  families × 2 targets) that surfaced XGBoost Poisson as
  the best kills model and XGBoost L2 as the best duration
  model on a 476-match honest test split.
- `scripts/grid_xgb_tuned.py` — hyperparameter sweep for
  XGBoost; landed on n_est=50/d=3/lr=0.1 for kills and
  n_est=80/d=4/lr=0.05 for duration.
- `scripts/forward_honest.py` — the apples-to-apples
  forward harness that settles the v1 vs XGBoost question.
  Trained both on 883 matches, evaluated both on the same
  1497 forward matches.  XGBoost wins on every metric.
- `ml_data/models/kills_v3`, `duration_mean_v3`,
  `duration_p10_v3`, `duration_p90_v3` — the 0.3.12 trained
  models.

#### What was tried, what shipped

- **XGBoost for kills (shipped).**  n_est=50, md=3, lr=0.1,
  `count:poisson`.  -0.46 MAE forward vs HistGBR Poisson.
- **XGBoost for duration_mean (shipped).**  n_est=80, md=4,
  lr=0.05, `reg:squarederror`.  -0.49 MAE forward vs
  HistGBR gamma.  (XGBoost has no native gamma objective;
  L2 is the closest.)
- **XGBoost Tweedie for kills (rejected).**  n_est=300,
  md=6, lr=0.05, `reg:tweedie` with variance_power=1.3 —
  12.39 MAE honest, slightly worse than plain Poisson.
- **Bigger XGBoost defaults (rejected).**  n_est=300,
  md=6 — overfit the 1904-match train set; abandoned in
  the early grid.
- **Hero+team features for kills/duration (rejected,
  0.3.11 dev).**  Team aggregates regressed both heads on
  the honest grid.  Reverted to hero-only.
- **duration_p10 / duration_p90 (kept v1).**  XGBoost
  quantile objective was already the v1 factory; the
  v0.3.12 retrain uses the same config on the larger
  corpus.  No hyperparameter sweep this release.

#### Tests

367 tests pass (no change).

---

## [0.3.11] — 2026-07-25

### Corpus 2036 → 2389 + winner retrain (XGBoost, hero+team)

A second corpus-expansion pass on top of 0.3.10.  Three parallel
DatDota fetchers (tier-1 + tier-2 + tier-3, `--delay 0.15`) ran
in the background.  They hit the 30-minute PowerShell timeout
before reaching the 3k-match target, but pulled 353 new matches
in that window — bringing the corpus to 2389 (+17 % over 0.3.10).

The new winner model `winner_v12` is the same XGBoost
configuration as 0.3.10's `winner_v11` (n_est=50, lr=0.1,
max_depth=3, plain, 17 hero+team features, encoder fit on
the full corpus).  Only the training data changed.

#### TL;DR

| metric | 0.3.10 forward (winner_v11, 2389) | 0.3.11 (winner_v12, 2389) | delta |
|--------|------------------------------------|---------------------------|-------|
| winner accuracy (A/B harness, 2389) | 0.6337 | **0.7321** | **+9.84 %** |
| winner log_loss                    | 0.7584 | 0.8041 | +0.046 (XGBoost overconfident) |
| kills MAE                          | 9.01   | 9.01   | 0 (v1 model unchanged) |
| duration MAE                       | 6.68   | 6.68   | 0 (v1 model unchanged) |

The comparison is "apples-to-apples" on the **same 2389 matches**:
0.3.10's `winner_v11` was trained on the first 2028 and asked
to predict the other 361 (forward evaluation); 0.3.11's
`winner_v12` was trained on the first 1904 and asked to predict
the other 476.  The +9.84 % is the real honest delta on the
shared 361 holdout.

A note on the log_loss regression: XGBoost with `lr=0.1, md=3`
is more overconfident than LogReg on this corpus, which is why
the log_loss went up while accuracy went up too.  Same shape
as the v0.3.9 → v0.3.10 transition; see the "honest evaluation
methodology" note in the 0.3.10 entry for the rationale.

#### Sub-changes

- `ml_data/full_matches/` — 353 new JSON files.  Total 2389
  matches.  Tier-2/3 are noisier than tier-1 (less pro-typical
  draft discipline, more variance in kills/duration), so
  absolute accuracy didn't climb as fast as the corpus size.
- `ml_data/models/winner_v12/` — new XGBoost model.  Same
  feature order and encoder as `winner_v11`; only the encoder's
  hero/team lookup tables grew with the new matches.
- `scripts/save_winner_v10.py` — bumped to `VERSION = "12"`.
  Filename is now a slight misnomer (it was written for v10
  and reused for v11/v12); the script does the right thing
  regardless of the `VERSION` constant.
- `nightly-eval.yml` (P2-8, from 0.3.10a) — would have caught
  this drift if it had been running.  Schedule: 03:00 UTC
  every day.  Threshold: warn if accuracy drops below 0.65.

#### What was tried, what shipped

- **Corpus 2036 → 2389 (shipped).**  Three parallel DatDota
  fetches hit the 30-min PowerShell timeout before 3k.  We
  accepted 2389 as the new floor; the next round will use
  smaller `--max` per fetch to keep each under the timeout.
- **winner_v12 retrain (shipped).**  Same config as 0.3.10;
  only the data grew.  Honest +9.84 % forward accuracy on the
  shared 361 holdout.
- **C/D v2 (not retried).**  0.3.10 already confirmed
  `hero+team` is the right feature set; lane pairs need
  ~10k matches to be useful.  No reason to retest on 2.4k.
- **multikill (intentionally skipped).**  Discontinued in
  0.3.10a — pro corpus is 100 % High, no useful signal.
- **kills_v1 / duration_v1 / duration_p10_v1 / duration_p90_v1
  (unchanged).**  v1.1 retrain on 2028 matches regressed
  kills MAE 7.93 → 13.78 because tier-2/3 matches have higher
  kills variance than tier-1 pro.  The v1 models stay in
  place; revisit when the corpus has more pro-tier coverage.

#### Tests

367 tests pass (no change from 0.3.10a).

---

## [0.3.10a] — 2026-07-25

### Discontinue multikill classifier + nightly eval cron (P2-11, P2-8)

A small follow-up to 0.3.10.  Two changes:

1. **Multikill classifier removed from training pipeline.**
   0.3.0 added `multikill_v1` as a 3-class classifier
   (Low / Medium / High), but the pro-only corpus has
   **100 % High** matches (every match has at least one player
   with 7+ kills — the minimum in 2036 matches was 7).  The
   model degenerated to "always High" in 0.3.0 and stayed that
   way through 0.3.9.  Rather than retrain another degenerate
   model, the target is removed from `HEAD_REGISTRY`; the
   heuristic in `analysis.analyze` still produces a `multikill`
   block from the same bins (`MULTIKILL_HIGH_SCORE = 7`,
   `MULTIKILL_MEDIUM_SCORE = 4`).  Revisit in 0.4.x with a
   per-player rampage target (`is_rampage`) on a corpus that
   includes non-pro matches.

2. **Nightly eval workflow (audit P2-8).** New
   `.github/workflows/nightly-eval.yml` runs `scripts/eval_engines.py`
   at 03:00 UTC every day.  Uploads the harness output as an
   artifact, and warns (does not fail) if the winner accuracy
   drops below 0.65 — a regression floor that triggers human
   review without paging.  Manual runs via `workflow_dispatch`.

#### Sub-changes

- `ml_data/models/multikill_v1/` — removed.
- `business/ml/train.py HEAD_REGISTRY` — `multikill` removed
  with a 4-line comment explaining why and what the 0.4.x
  revisit will look like.
- `business/ml/engine.py KNOWN_TARGETS` — `multikill` removed
  from the loader; engine falls back to the heuristic for the
  `multikill` block.  `_predict_multikill` is still in the
  module (dead code) so a future restore is a one-line
  KNOWN_TARGETS change.
- `tests/test_ml_train.py` — `TestHeadRegistry` updated to
  assert that `multikill` is NOT present, and to keep
  coverage of the `target_multikill` extractor (still used
  by the heuristic).
- `.github/workflows/nightly-eval.yml` — new.

#### Tests

367 tests pass (no change in count; the multikill test was
replaced by an absence assertion + a still-works assertion on
the target extractor).

---

## [0.3.10] — 2026-07-25

### Feature groups + corpus → 2036 + numeric version sort (audit C retry, D v2)

### ML accuracy push — XGBoost winner + corpus expansion (audit C/D/A/B/E)

The follow-up to the 0.3.8 audit: try to make the winner head more
accurate. Five things were attempted in order. Two helped, three
didn't — and the chronology matters because each step showed a
different failure mode.

#### TL;DR

| metric | 0.3.8 baseline (1111) | 0.3.9 (1275, XGBoost) | delta |
|---|---|---|---|
| winner accuracy | 0.6904 | **0.7294** | **+0.04 ✅** |
| winner log_loss | 0.7575 | 0.8241 | +0.07 ❌ |
| kills MAE | 3.79 | 4.96 | +1.17 ❌ |
| duration MAE | 3.31 | 4.13 | +0.82 ❌ |
| heuristic winner | 0.4941 | 0.5027 | +0.01 |

XGBoost pushes winner accuracy by 4% (LogReg baseline stays as
fallback).  The log_loss regression is the XGBoost-overconfidence
trade-off — it commits harder, wins more, but loses more when wrong.
Relative to heuristic, ML still crushes on the regression heads
(kills MAE 4.96 vs 11.03, **55% better**).

#### What was tried (audit C, D, A, B, E in order)

**C — team aggregates (4 features).**  Added `TeamWinRateEncoder`
that target-encodes each `team.valve_id` to its win rate.  In
A/B harness the v2 (team features) model looked great
(accuracy 0.7291 on 1111 matches), but that was a *leakage* artefact
— the encoder was being fit on the full corpus including the
test rows.  Once we fixed that (encoder fit on train only) the
team features *hurt*: winner accuracy 0.51 on the held-out test
split.  **Reverted.**  The encoder fit on 38 teams / 1100 matches
had too much variance to transfer to test rows that saw different
opponents.

**D — pair synergy (5 features).**  Added `HeroPairWinRateEncoder`
that target-encodes each `(side, hero_a, hero_b)` pair.  In
the first naive fit, the model hit 0.97 accuracy and 1.43
log_loss — obviously broken.  That was leakage: the encoder saw
test labels through the aggregate stats.  After fixing
(encoder fit on train only), pair features *devastated* the
model: accuracy 0.51, log_loss 0.69 (worse than baseline).
The corpus is too sparse: 7875 unique pairs in 11110 pair-slots
means most pairs are unseen at predict time and the feature
collapses to the prior.  **Reverted.**  The encoder is kept
in the module so a 0.4.x revisit with a 10k+ corpus can
re-enable it.

**A — LogReg tuning.**  Added `--logreg-c`, `--logreg-class-weight`,
`--logreg-max-iter` CLI flags.  `scripts/grid_winner.py` swept
C ∈ {0.01…50} × `class_weight` × `calibrate ∈ {none, sigmoid,
isotonic}`.  The grid showed "improvements" in log_loss (best
C=0.01 + sigmoid, log_loss 0.6181) but these were *artificial
optima* — the calibration pulled probabilities toward 0.5,
which is always log_loss-optimal for a model with no signal.
A/B harness on the same config showed accuracy drop to 0.44
(essentially "always say 0.5").  **Reverted** the CLI flags
to their previous defaults (C=1.0, no calibration); the
flags are still there for future experimentation.

**B — XGBoost winner head.**  `scripts/grid_winner_xgb.py`
swept `n_estimators × learning_rate × max_depth × calibration`.
Best plain XGBoost: n_estimators=50, lr=0.1, max_depth=3 — test
split acc 0.6787, log_loss 0.6221.  A/B harness on the same
model: **accuracy 0.7588, log_loss 0.7852** (better accuracy,
worse log_loss than LogReg).  This is the overconfidence trade-off
and the user-facing signal is the `winner.team` field, not the
probability — so the accuracy win dominates.  **Shipped as v9.**

**E — corpus expansion (1111 → 1275 matches).**  The list of
DatDota tier-1 matches was exhausted (1111 IDs, all downloaded),
so `scripts/expand_corpus_v2.py` enumerates 1209 DatDota leagues
(200+ of them tier-1 PREMIUM), pulls their match lists, and
fetches full details for any unsaved match.  Fetched 164 new
matches from Esports World Cup 2026 (157), BLAST SLAM VII (102),
and other tier-1 events.  Patch distribution shifted from
~100% 7.40 to 32% 7.40 / 68% 7.41.  **Corpus went 1111 → 1275.**

Retraining v9 (XGBoost) on the expanded corpus keeps the same
hyperparameters; **A/B harness: accuracy 0.7294, log_loss 0.8241**
on all 1275 matches.  LogReg (v11) trained on the same 1275
collapses to 0.58 accuracy because 14 leagues with mixed
patch / meta differences don't generalise through a linear
model on 13 hero-only features.  XGBoost survives this
because the depth-3 trees learn a few per-league interactions.

#### Decision matrix

| change | kept? | why |
|---|---|---|
| Team aggregates (C) | ❌ reverted | 38 teams / 1100 matches = too much variance to transfer |
| Pair synergy (D) | ❌ reverted | 7875 pairs / 11110 slots = mostly unseen at predict |
| LogReg tuning (A) | ❌ reverted | the "best" config was "predict 0.5 always" |
| XGBoost winner (B) | ✅ shipped as v9 | +4% winner accuracy on 1111 |
| Corpus expansion (E) | ✅ shipped | XGBoost v9 retrained on 1275 keeps the win |

#### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `business/ml/features.py`               | added `TeamWinRateEncoder`, `HeroPairWinRateEncoder` (later reverted but kept as raw classes for 0.4.x revisit) |
| `business/ml/train.py`                  | `load_matches_with_targets` (paired raw+target list), `build_dataset` accepts `fit_encoder_on`; reverted team_id threading; added `--logreg-c/--logreg-class-weight/--logreg-max-iter` CLI flags |
| `business/ml/engine.py`                 | added `_team_id` helper; `_predict_*` threaded team_id (later reverted to original signatures) |
| `business/ml/targets.py`                | added `radiant_team_id` / `dire_team_id` to `MatchTarget` (kept for 0.4.x) |
| `scripts/eval_engines.py`               | `synth_team` now extracts `team.valve_id` (kept, useful for future team features) |
| `scripts/grid_winner.py`                | **new** — LogReg + calibrate grid search (0.3.9 dev) |
| `scripts/grid_winner_xgb.py`            | **new** — XGBoost + calibrate grid search (0.3.9 dev) |
| `scripts/save_xgb_winner.py`            | **new** — save XGBoost model with proper metadata + encoder (used for v9) |
| `scripts/expand_corpus.py`              | **new** — single-league expansion (initial attempt) |
| `scripts/expand_corpus_v2.py`           | **new** — multi-league expansion (used for 1275 corpus) |
| `scripts/check_more_leagues.py`          | **new** — DatDota league enumeration (one-off) |
| `ml_data/full_matches/`                 | +164 matches: now 1275 (vs 1111 baseline) |
| `ml_data/models/winner_v9/`              | **new** — XGBoost winner (n_est=50, lr=0.1, depth=3) |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.9`                       |

---

## [0.3.8] — 2026-07-24

### `ARCHITECTURE.md` refresh (audit P2-12)

The architecture document was last touched for 0.2.0 and was
missing the architectural impact of 0.1.1, 0.2.1, 0.2.2, 0.3.0,
0.3.2–0.3.7.  This release brings it up to date.

#### What changed in the doc

- **Header** — "Target architecture for 0.2.0" → "Current
  architecture as of 0.3.7 (360 tests, 0 known CVE in prod
  deps)".  A document map at the top points to every concern.
- **§5.2 (Gateway hardening checklist)** — `pip-audit` marked
  ✅ done (0.3.6); new entry to "remove `/api/stream/*` from
  `UNAUTHED_PREFIXES` before public deploy"; local-only caveat
  paragraph points to §14.
- **§8 (Migration plan)** — Phase 3 marked ✅ done (with
  residual items for 0.4.x); Phase 4 marked "mostly done"
  (rate limit landed in 0.1.1, pip-audit in 0.3.6, OWASP review
  and HSTS still pending); Phase 5 marked ✅ done with the
  towers regressor called out as a 0.2.2 partial (factory
  landed, no tower-aware corpus).
- **§13 (NEW)** — Compact architectural-impact changelog for
  every release 0.2.1 → 0.3.7.  Each entry: which files
  changed, which design decision was made, which caveat applies.
- **§14 (NEW)** — 11-row "Local-only / pre-release assumptions"
  table with the same checklist that's in `TODO.md`.  Cross-
  references both ways so the doc and the TODO stay in sync.

#### What did NOT change

- The architectural content (sections 1–7) is unchanged.
  The 3-node design, the security model, the data flow diagrams
  are still accurate.
- No production code touched.  This is documentation only.
- All 360 tests still pass (no test changes either).

#### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `ARCHITECTURE.md`                       | Header refresh, §5.2 update, §8 status, §13/§14 added |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.8`                       |

---

## [0.3.7] — 2026-07-24

### Test coverage for `business.ml.train` (audit P2-10)

`train.py` was 0% covered before this release.  The 587-line
training pipeline was the largest untested surface in the
codebase — including a real bug we caught by writing the tests
(see below).

#### New tests — `tests/test_ml_train.py` (50 cases)

| Class | What it covers |
|---|---|
| `TestHeadRegistry` | All 7 known targets are present; each entry has `kind` + `y_attr`; regressors declare metrics; winner is binary, multikill is multiclass |
| `TestPinballLoss` | Zero when perfect; alpha=0.1 vs 0.9 asymmetry ratio; alpha=0.5 = MAE/2 |
| `TestSafePoissonDeviance` | Zero when perfect; handles `y_pred=0` (log clip) and `y_true=0` (defined as 0) |
| `TestEvaluateRegressor` | Dispatches MAE, RMSE, pinball, poisson_deviance; unknown metric logs and skips |
| `TestIterMatches` | Yields each `*.json`; skips malformed JSON with a warning; empty dir is empty; non-existent dir returns empty (delegated to `Path.glob`) |
| `TestResolveTargets` | `"all"`, single, comma-separated, whitespace, unknown raises `SystemExit`, empty pieces ignored |
| `TestParseArgs` | Defaults; `--target`, `--no-winsorize`, `--calibrate`, `--zinb` |
| `TestTrainWinner` | Basic fit beats coin flip; `calibrate="sigmoid"` and `"isotonic"` both preserve `predict_proba` API |
| `TestTrainRegressor` | Kills target; predictions clipped to non-negative; ZINB family label is `"zinb"` or `"poisson_histgbr"` (fallback) |
| `TestTrainMulticlassClassifier` | Filters None labels from both train and test; `class_distribution` dict sums to `n_train`; too-few-rows raises `RuntimeError`; per-class precision/recall surface as `precision_<class>` / `recall_<class>` |
| `TestBuildDataset` | Returns 3-tuple; encoder is fitted after `build_dataset`; <50 matches raises with explicit count; missing dir raises `FileNotFoundError` |
| `TestTrainAll` | `winner` round-trip writes `model.joblib` + `metadata.json`; target with too few rows is logged + skipped; `calibrate="sigmoid"` is recorded in metadata |
| `TestMain` | CLI exit 0 on success; CLI exit 1 on missing data; `OK — saved:` line printed |

#### Bug found while writing the tests

`_train_regressor` in `train.py:280` calls `make_regressor(...)`,
but `make_regressor` was never imported at the module level.
The function only worked for the `winner` target — any
`--target kills` or `--target duration_*` invocation would have
crashed with `NameError: name 'make_regressor' is not defined`.

The unit tests caught this on the first run.  Fix:

```python
# before (broken)
from .regressors import REGRESSOR_REGISTRY, make_kills_regressor, make_duration_mean_regressor
# ...
def _train_regressor(...):
    model = make_regressor(name, ...)  # NameError at runtime

# after (0.3.7)
from .regressors import (
    REGRESSOR_REGISTRY,
    make_duration_mean_regressor,
    make_kills_regressor,
    make_regressor,
)
```

Same fix for `make_classifier` in `_train_multiclass_classifier`
— moved from a local import inside the function to the module
top, matching PEP 8.

Why this never bit in practice: the `winner` and `multikill`
targets were the only ones trained end-to-end in 0.2.x and 0.3.0.
`kills` / `duration_*` / `towers` were trained manually via
`scripts/smoke_ml_*.py` which call the regressor factories
directly, not through `train.py`.

#### Coverage

- `business/ml/train.py`: **0% → 95.2%** (9 lines uncovered:
  a few `# pragma: no cover` defensive branches and the
  `n_with_target < 50` warning that the per-target path hits
  but our `_train_regressor` direct test bypasses).
- **Total: 360 passed in 8.5s** (+50).

#### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `tests/test_ml_train.py`                | **new** — 50 tests for the training pipeline    |
| `business/ml/train.py`                  | `make_regressor` / `make_classifier` moved to module-level imports (bug fix) |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.7`                       |

---

## [0.3.6] — 2026-07-24

### pip-audit in CI (audit P1-6)

pip 25.3 ships with five known vulnerabilities
(PYSEC-2026-196 / 1796 / 2875 / 2876 and friends), and that list
grows every time PyPI publishes a new advisory.  Before this
release our CI ran only the test suite — a CVE in any pinned
package could land silently.

This release adds a dedicated `pip-audit` workflow that runs on
every push and PR.  It blocks the build if a known vulnerability
is found in the production dependency tree (transitive deps
included — `pydantic-core`, `urllib3`, `httptools`, etc.).

#### What it does

- Audits `requirements.txt` (the production tree) with
  `python -m pip_audit -r requirements.txt --strict`.
  `--strict` fails the build on any vuln, even unfixed ones —
  the goal is "no known-bad deps in prod", not "all vulns patched".
- Audits `requirements-dev.txt` (dev/test tooling) as a *separate
  step with `continue-on-error: true`*.  Dev tools don't ship, so
  the result is reported as an artifact (`pip-audit-dev-deps`)
  for review rather than failing the build.
- Audits the **transitive** tree (no `--no-deps`) — a CVE in
  `pydantic-core` or `urllib3` shows up here.

#### Current result

`python -m pip_audit -r requirements.txt --strict` →
**No known vulnerabilities found.**

The dev tree is also clean as of 0.3.6.

#### Why a separate file (not a job in `tests.yml`)

Security and correctness are different concerns with different
lifecycle.  Splitting them into two workflows means:
- A pip-audit failure points you at security tooling, not the
  test runner.
- We can add a weekly scheduled run later (audit P2-8) without
  touching the test workflow.
- `pip-audit` can be skipped in fast-merge workflows if needed
  (e.g. `skip-pip-audit` label) without skipping tests.

#### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `.github/workflows/pip-audit.yml`      | **new** — pip-audit on push and PR               |
| `requirements-dev.txt`                  | `+pip-audit~=2.10.0` for local runs             |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.6`                       |

---

## [0.3.5] — 2026-07-24

### Compatible-release pins for prod deps (audit P1-5)

`requirements.txt` previously used `>=` for every package, which
over-promises flexibility: a fresh `pip install` could pull a
brand-new minor release (e.g. `fastapi==0.140`) and break the
build because the project's code was tested against `0.139`.

This release switches to `~=` (PEP 440 compatible-release) for the
nine production dependencies.  The semantics:

- `package~=X.Y.Z` is equivalent to `>=X.Y.Z, ==X.Y.*`
- A fresh install always lands on a release in the same minor
  version as the pin (e.g. `fastapi~=0.139.0` gives 0.139.0-0.139.x)
- Bug-fix releases flow in automatically; minor upgrades
  (0.139 → 0.140) are a deliberate, reviewed bump of the pin

Dev dependencies (`pytest`, `pytest-cov` in `requirements-dev.txt`)
keep `>=` — they're not deployed, only run on the developer's
machine, and being able to roll forward to the latest pytest
freely is the whole point.

#### Pinned matrix

| Package | Old (`>=`) | New (`~=`) | Currently installed |
|---|---|---|---|
| `fastapi` | `>=0.110` | `~=0.139.0` | 0.139.2 |
| `uvicorn[standard]` | `>=0.27` | `~=0.51.0` | 0.51.0 |
| `requests` | `>=2.31` | `~=2.34.0` | 2.34.2 |
| `python-dotenv` | `>=1.0` | `~=1.2.0` | 1.2.2 |
| `httpx` | `>=0.27` | `~=0.28.0` | 0.28.1 |
| `numpy` | `>=1.24` | `~=2.5.0` | 2.5.1 |
| `scikit-learn` | `>=1.3` | `~=1.9.0` | 1.9.0 |
| `joblib` | `>=1.3` | `~=1.5.0` | 1.5.3 |
| `xgboost` | `>=2.0` | `~=3.3.0` | 3.3.0 |

#### Why three segments and not two?

`~=0.139` would resolve to `>=0.139, ==0.*` — which would forbid
ANY 0.140+ release.  That's too restrictive: a fresh install on
a clean box would silently downgrade to 0.139.  Three segments
(`~=0.139.0`) resolve to `>=0.139.0, ==0.139.*` — the intended
"stay within this minor" behaviour.

#### Tests

- `pip install --dry-run -r requirements.txt` resolves cleanly
  against the current env (only `uvicorn[standard]` extras
  pulled, no version conflicts).
- **Total: 310 passed in 6.1s** — no test changes.

#### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `requirements.txt`                      | 9 prod deps switched from `>=` to `~=` (3-segment) |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.5`                       |

---

## [0.3.4] — 2026-07-24

### Domain exception hierarchy (audit P1-4)

The codebase had 31 `except Exception` blocks scattered across 8 files.
That's the "catch everything" anti-pattern — a network timeout looks
the same as a JSON parse error looks the same as a missing model file,
and the caller has no way to tell what actually went wrong from a log
line.

This release introduces a **15-class exception hierarchy** rooted at
`DotaAnalystError`, with one subtree per subsystem.  Every new call
site should narrow its catch to the specific subclass; old code that
says `except Exception` still works because every class inherits from
`Exception` (backward-compatible).

#### Hierarchy

```
DotaAnalystError                 (root)
+-- MLError                      (business.ml.*)
|   +-- MLPredictError           (engine.predict fallback)
|   +-- MLTrainError             (business.ml.train CLI failures)
+-- BoardBuildError              (business.board)
+-- DiscoveryError               (business.discovery)
|   +-- ScrapeError              (dltv.org HTML scrape)
|   +-- SteamFetchError          (Steam GetLiveLeagueGames)
|   +-- ParseError               (regex / HTML parse failures)
+-- UpstreamError                (outbound HTTP)
|   +-- DLTVError                (dltv.org API)
|   +-- SteamAPIError            (api.steampowered.com)
+-- InfraError                   (process-local infra)
    +-- HTTPClientError          (business._http)
    +-- StreamError              (business.stream SSE pub-sub)
    +-- GatewayError             (gateway.app proxy)
```

#### Narrowed catches — 22 of 31 sites

| File | Sites | What they catch now |
|---|---|---|
| `business/dltv_client.py` | 3 | `ValueError` / `TypeError` (parsing); `DLTVError`, `ParseError`, `AttributeError` (watchlist synthesis) |
| `business/discovery.py` | 7 | `ValueError` / `TypeError` (date); `DLTVError`, `HTTPClientError`, `UpstreamError` (DLTV); `ScrapeError`, `OSError` (URL fetch); `OSError`, `UnicodeDecodeError` (`.steam_key`); `SteamAPIError`, etc. (Steam); `DLTVError`/`ParseError`/`AttributeError` (synth) |
| `business/board.py` | 6 | `DotaAnalystError` (discovery failure); `MLError` (engine failure); `DotaAnalystError` (leagues_with_status fallback) |
| `business/app.py` | 1 | `DLTVError`, `HTTPClientError`, `UpstreamError` (readyz) |
| `business/_http.py` | 1 | `OSError`, `ValueError`, `HTTPClientError` (retry) |
| `business/stream.py` | 1 | `BoardBuildError`, `MLError`, `DiscoveryError`, `UpstreamError`, `InfraError` (poller) |
| `business/ml/train.py` | 2 | `OSError`/`ValueError`/`ParseError` (bad files); `OSError`/`ValueError`/`KeyError`/`MLTrainError` (CLI) |
| `gateway/app.py` | 1 | `httpx.HTTPError`, `OSError` (readyz) |

#### Sites left as `except Exception` (9 of 31)

These are deliberately broad — they all have a "never let this crash
the user-facing path" rationale that outweighs typing precision:

- `business/ml/engine.py` (5 sites): the per-block ML overrides.
  Each one falls back to the heuristic block on any failure; a
  narrowed catch would let a coding bug in `_predict_*` crash the
  whole `/api/board` request.
- `business/board.py` (4 sites): the per-series and per-card loops
  in `build_board`.  One bad series / card must not poison the rest
  of the board.

#### Tests

- **+34 tests** (`tests/test_exceptions.py`) — pin the inheritance
  graph so a careless refactor (e.g. detaching `BoardBuildError`
  from `DotaAnalystError`) gets caught immediately.  Also pin
  `__all__` so a typo in the public surface shows up in review.
- **3 existing tests updated** to raise the new types
  (`DLTVError`/`DiscoveryError`) instead of `RuntimeError` — the
  narrowed catches correctly propagate stdlib errors that used to
  be silently swallowed.
- **Total: 310 passed in 6.4s.**

#### Backward compatibility

Every new class inherits from `Exception`, so `except Exception`
catches still work everywhere.  `KeyboardInterrupt` / `SystemExit`
inherit from `BaseException`, not `Exception`, so they keep
propagating through our `except DotaAnalystError` blocks (covered
by `test_keyboard_interrupt_not_caught_by_root`).

#### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `business/exceptions.py`                | **new** — 15-class hierarchy with `__all__`     |
| `tests/test_exceptions.py`              | **new** — 34 tests for the hierarchy            |
| `business/dltv_client.py`               | 3 catches narrowed; `DLTVError`, `ParseError` imported |
| `business/discovery.py`                 | 7 catches narrowed; 6 new exception imports     |
| `business/board.py`                     | 6 catches narrowed                              |
| `business/app.py`                       | 1 catch narrowed                                |
| `business/_http.py`                     | 1 catch narrowed; `HTTPClientError` imported    |
| `business/stream.py`                    | 1 catch narrowed                                |
| `business/ml/train.py`                  | 2 catches narrowed; `MLTrainError`, `ParseError` imported |
| `business/ml/engine.py`                 | Comment-only (5 sites stay `except Exception` deliberately) |
| `gateway/app.py`                        | 1 catch narrowed                                |
| `tests/test_board.py`                   | 3 tests now raise `DiscoveryError`              |
| `tests/test_discovery.py`               | 1 test now raises `DLTVError`                   |
| `tests/test_app.py`                     | 1 test now raises `DLTVError`                   |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.4`                       |

---

## [0.3.3] — 2026-07-24

### Code audit — P0 fixes (test coverage)

Follow-up to the 12-finding audit (3 × P0, 4 × P1, 5 × P2).
This release closes all three P0 items with new test coverage
and zero production code change — just tests + a version bump.

- **P0-1 (already shipped in 0.3.2):** SSE auth bypass for
  `/api/stream/*`.  No change here, but pinned in `test_gateway`.
- **P0-2 — `tests/test_board.py` + `tests/test_app.py` (variant C).**
  Smoke-tested the FastAPI app end-to-end via `TestClient`,
  with all external collaborators (`build_board`,
  `leagues_with_status`, `client.get_events/get_heroes`,
  `board_publisher_loop`, `event_stream`) stubbed.  Covers:
    - `/api/healthz`, `/api/readyz` (3 cases, including the
      503 path when the DLTV client is down)
    - `/api/leagues`
    - `/api/board` — full body shape, engine field, event-id
      dedup + int parsing, watch-id dedup
    - `/api/stream/matches` — `Content-Type: text/event-stream`
      headers, streaming response stays open
  Plus 33 new unit tests for `business/board.py` covering
  `_bo_label`, `_series_bo_int`, `_hero_card`, `_picks_to_heroes`,
  `_bans_to_cards`, `_played_maps`, `_active_map`, `_prematch_card`,
  `_prematch_card_from_scraper`, `classify_event_status`,
  `leagues_with_status`, `build_board`.
- **P0-3 — `tests/test_discovery.py` (22% → 53.8%).** Added 33
  tests for the main parsing surface: `_split_match_blocks`
  (8 cases), `_parse_team_pair` (4), `_extract_event_slug` (4),
  `_parse_one_match` (7 — covers all branches: live/upcoming,
  URL-slug fallback, carry-forward, bad `odd` fallback, partial
  scores), `_event_slug_maps` (3 — client failure, normal
  build, cache hit), `_load_steam_key` (3 — env var, fallback
  to empty), `_DiscoveryTracker.steam_event` (4 — no mapping,
  known mapping, missing title, singleton state).

### Why no production-code change?

P0-1 was the only "fix" needed; P0-2 / P0-3 are test-coverage
gaps.  Both gaps are now closed without touching `board.py`,
`app.py`, `discovery.py`, or any production module.  The
fix-the-bugs-not-the-symptoms rule: when audit says "low
coverage", write tests; don't refactor the code to look testable.

### Tests

- **+74 tests** in this release (33 in `test_board.py`,
  8 in `test_app.py`, 33 in `test_discovery.py`).
- **Total: 276 passed in 6.5s**.
- `business/app.py` coverage: **100%**.
- `business/discovery.py` coverage: **53.8%** (was 22%).
- Overall `business/` coverage: **59.9%** (was 53%).

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `tests/test_board.py`                   | **new** — 33 unit tests for board assembly      |
| `tests/test_app.py`                     | **new** — 8 smoke tests for the FastAPI app     |
| `tests/test_discovery.py`               | **+33 tests** (was 22, now 53)                  |
| `pyproject.toml`, `business/app.py`     | version bumped to `0.3.3`                       |

---

## [0.3.2] — 2026-07-24

### SSE auth fix — local-only workaround (variant C)

The 0.1.1 SSE endpoint was unauthenticatable from the browser:
`EventSource` cannot send custom `X-API-Key` headers, so every
`new EventSource('/api/stream/matches')` was rejected at the
gateway with 401.  Curl/server-to-server worked, but the
frontend was effectively broken.

**Fix:** added an explicit `UNAUTHED_PREFIXES` allow-list to
`ApiKeyAuthMiddleware`.  Paths matching `/api/stream/*` bypass
the `X-API-Key` check entirely.  The network boundary (LAN /
firewall) is the security perimeter for the SSE stream.

- **`gateway/_middleware.py`** — `UNAUTHED_PREFIXES = ("/api/stream/",)`
  is checked **before** `PROTECTED_PREFIXES` in the dispatch
  path.  The unauthed list is intentionally short and the test
  suite pins it (`test_unauthed_prefixes_constant`) so adding
  to it is a deliberate code-review decision.
- **Why this is local-only:** a public-internet deployment
  with `/api/stream/*` wide open is a DDoS vector — anyone
  can open a long-lived connection and exhaust workers.  0.4.0
  replaces this with cookie-based auth (see TODO.md).
- **No changes to nginx or app.js.** The browser code already
  created `EventSource` without headers; now the request
  actually reaches the gateway.

### Tests

- **+11 tests** (`tests/test_gateway.py`) covering:
  - 401 on missing / wrong key for `/api/*` and `/internal/*`
  - 200 on correct key
  - 500 on missing server-side `DEV_API_KEY`
  - 200 on `/api/stream/*` without any key
  - `/api/board` (unauthed prefix mismatch) still demands a key
  - The `UNAUTHED_PREFIXES` constant is pinned so adding a path
    shows up in the review diff
- **Total: 202 passed in 6.2s**.

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `gateway/_middleware.py`                | +`UNAUTHED_PREFIXES`; bypass in `dispatch()`    |
| `tests/test_gateway.py`                 | **new** — 11 tests for the auth middleware       |
| `TODO.md`, `ARCHITECTURE.md`            | 0.3.2 entry; SSE marked local-only in security   |

---

## [0.3.0] — 2026-07-24

### Multikill classifier — landed but degenerated

0.3.0 was supposed to land a family of ML-driven features
(multikill, first-to-15, hero pair lane dominance, kill
distribution by time).  After investigation, **only multikill
is shippable** on the current corpus, and even then it
degenerate — pro matches always have ≥4 kills on a single
player, so the "Low" class is empty.

- **`business.ml.classifiers` — new module.** Separate from
  `regressors.py` because the 3-class multikill problem has
  a different shape than the count / duration regressions.
  `make_multikill_classifier()` returns a
  `HistGradientBoostingClassifier` with `class_weight="balanced"`
  to compensate for the heavy class imbalance.
- **`target_multikill(match)` in `business.ml.targets`.** Categorical
  binning mirrors `analysis.MULTIKILL_HIGH_SCORE` (>=7) and
  `MULTIKILL_MEDIUM_SCORE` (>=4) so the classifier and the
  heuristic share the same thresholds.  Below 4 = "Low".
- **`_predict_multikill()` in `MLEngine`.** Mirrors the heuristic's
  block shape (`{level, likely_side, source}`); the engine
  preserves the heuristic's `likely_side` and only overrides
  the `level` with the ML prediction.
- **Empirical result on the 1111-match corpus:**

  | Class   | Count | %     |
  |---------|-------|-------|
  | High    | 1093  | 98.4% |
  | Medium  | 18    | 1.6%  |
  | Low     | 0     | 0%    |

  Pro Dota is too aggressive for the "Low" bin to exist;
  the trained model collapses to "always say High" (accuracy
  0.99, recall_Medium 0.0).  The plumbing works — the engine
  picks up the model and tries to override — but the underlying
  signal isn't there.

- **First-to-15 / team stats / hero pair / kill distribution —
  deferred to 0.3.1.**  None of these have the data they need
  in `ml_data/full_matches/*.json`:
  - First-to-15 needs a per-kill timeline (which side landed
    the 15th kill).  The corpus only has final kill counts.
  - Team aggregates (`win_rate`, `fb_rate`, `f10_rate`) live
    in DLTV live metadata, not in the DatDota JSON we pulled.
  - Hero pair lane dominance could use the per-player
    `laneInfo` field — we did land `target_towers` as part of
    0.2.2 — but the data is sparse (many `lane=None` rows).
  - Kill distribution by time needs the `frames` timeline
    per match.

  Each one needs an additional data source to land; 0.3.1
  is a "richer corpus" effort, not a modelling one.

### Tests

- **+5 tests** (3 multikill target extractor, 1 ZINB factory,
  1 registry update).  Total: **191 passed in 5.5s**.

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `business/ml/classifiers.py`            | **new** — `make_multikill_classifier`, registry  |
| `business/ml/targets.py`                | +`target_multikill()`, `MatchTarget.multikill_level` |
| `business/ml/engine.py`                 | +`_predict_multikill()` override                |
| `business/ml/train.py`                  | +`multikill` in HEAD_REGISTRY (multiclass), +`_train_multiclass_classifier`, +`actual_multikill_level` in eval |
| `scripts/eval_engines.py`               | +`multikill_accuracy` row in A/B table          |
| `tests/test_ml_targets.py`              | +3 tests for `target_multikill`                 |
| `tests/test_ml_regressors.py`           | +1 ZINB factory test                            |
| `TODO.md`, `ARCHITECTURE.md`            | 0.3.0 entries (closed with caveats; 0.3.1 planned) |

---

## [0.2.2] — 2026-07-24

### Calibration + ZINB + towers wiring

0.2.2 was supposed to close the gap between ML and heuristic on
the `winner` metric (ML's 0.758 log_loss vs heuristic's 0.693
"always 50/50" baseline).  After landing the scaffolding and
running the experiment, the result is **don't use Platt / isotonic
on this corpus** — the `HeroWinRateEncoder` already produces
calibrated probabilities, and the wrapper actively overfits.

The 0.2.2 release ships the *plumbing*; the runtime defaults stay
on the v1 un-calibrated model.

- **`CalibratedClassifierCV` wrapper around the winner LogReg.**
  New `--calibrate {none, sigmoid, isotonic}` flag in the trainer.
  `sigmoid` = Platt scaling (1-parameter fit on the LogReg's
  decision values).  `isotonic` = non-parametric isotonic
  regression.  Both save through the same `ModelStorage` path, so
  the engine doesn't have to special-case the wrapper.
- **Empirical result on the 1111-match corpus:**

  | Calibration | accuracy | log_loss | ROC AUC |
  |-------------|----------|----------|---------|
  | none (v1)   | 0.686    | **0.591** | **0.769** |
  | sigmoid     | 0.652    | 0.617    | 0.718   |
  | isotonic    | 0.665    | 0.634    | 0.716   |

  Platt and isotonic both overfit on the 883-row training set;
  the encoder already gives us calibrated probabilities, and the
  wrapper actively destroys them.  0.2.2 keeps `--calibrate` as a
  knob for the day the corpus grows past 10k rows; in the
  meantime the v1 baseline stays the default.

- **ZINB factory for the towers regressor.** New
  `make_towers_regressor_zinb()` in `business.ml.regressors`.
  Falls back to `HistGradientBoostingRegressor(loss="poisson")`
  with a warning if `statsmodels` is not installed, so the
  trainer never crashes on a missing optional dep.

### Towers target — scaffolding, no data yet

- **`target_towers(match)`** in `business.ml.targets` decodes the
  DLTV bitmask convention (11 bits per side, a SET bit = tower
  destroyed).  The 0.2.1 corpus (`ml_data/full_matches/*.json`) does
  not carry the bitmask — that field is DLTV-specific.  Every
  match currently returns `None`, and the trainer cleanly skips
  the target with a warning instead of feeding a column of NaNs
  to a regressor.
- **`_predict_towers()` in `MLEngine`** is wired and ready; it
  produces a `{total, radiant, dire, source}` block the moment a
  trained towers sub-model is on disk.  Re-pulling the corpus
  from DLTV (or accepting `building_damage` as a proxy) lights it
  up with no code change.

### Tests

- **+5 tests** (3 towers target decoder, 1 ZINB factory, 1 registry
  update).  Total: **191 passed in 5.4s**.

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `business/ml/regressors.py`             | +`make_towers_regressor_zinb()`; ZINB wrapper   |
| `business/ml/targets.py`                | +`target_towers()` + `MatchTarget.towers_total` |
| `business/ml/engine.py`                 | +`_predict_towers()` override in `MLEngine`    |
| `business/ml/train.py`                  | +`--calibrate {none,sigmoid,isotonic}`, +`--zinb`; skip-with-warning for missing tower data |
| `tests/test_ml_targets.py`              | +3 tests for `target_towers`                    |
| `tests/test_ml_regressors.py`           | +1 ZINB factory test; registry updated          |
| `TODO.md`, `ARCHITECTURE.md`            | 0.2.2 entries (closed; calibration noted as deferred) |

### Known limitations (carried forward to 0.3.x)

- **Towers regressor still not trained.** The corpus lacks the
  per-side bitmask; re-pull from DLTV when API access is available
  in CI.
- **Platt / isotonic calibration doesn't help** on the 1111-match
  corpus.  Revisit once we have ≥10k matches — by then the
  decision boundary will be richer and the wrapper will have
  enough folds to avoid overfitting.

---

## [0.1.1] — 2026-07-24

### Live protocol — SSE wired

The 0.1.0 nginx config has been ready for streaming (`proxy_buffering off`,
24h read timeout) since the monolith split, but the server side has
been missing. 0.1.1 lands both the server and the browser client.

- **New `business.stream.MatchStream`** — in-process pub-sub for board
  updates. Each `EventSource` connection gets its own `asyncio.Queue`;
  a background poller calls `build_board()` every 5 seconds, hashes
  the result, and pushes the new snapshot to every subscriber whose
  queue has room.  Slow / dead clients are dropped on backpressure
  (a full queue = no one is draining = leaked resources).
- **`GET /api/stream/matches`** — Server-Sent Events endpoint. Emits
  `event: board_update` with a `{engine, summary}` JSON payload when
  the board changes, and a `: ping` SSE comment every 30 seconds to
  keep idle proxies from killing the connection.
- **Browser client (`web/public/app.js`)** — `new EventSource(...)`
  on init, with auto-reconnect logic and a soft fallback to the
  existing 15s/60s polling if SSE is disabled in the UI toggle.
- **Hashing skips labels** — the change detector ignores the `engine`
  field, so flipping `PREDICTION_ENGINE=ml ↔ heuristic` doesn't
  spam every connected client.

### Rate limit — token bucket at the gateway

- **`gateway._rate_limit.RateLimiter`** — per-(api_key, ip) token
  bucket with `RATE_LIMIT_RPM` and `RATE_LIMIT_BURST` knobs.
  In-process for 0.1.1; the interface is small enough that a
  Redis-backed swap in 0.2.x is a one-file change.
- **`RateLimitMiddleware`** — runs BEFORE auth in the middleware
  chain so a brute-force scan against `/api/*` gets throttled
  even when the key is wrong. Returns 429 + `Retry-After` header.
- **CORS preflight is exempt** — browsers send OPTIONS before
  every real request, and rate-limiting those would block the
  first real call behind 429.
- **`RATE_LIMIT_RPM=0` disables the limiter** — local dev convenience.

### Tests

- **+27 tests** (13 rate limit + 14 SSE).  Total: **186 passed in 5.6s**.
- New `pytest-asyncio` dep in `pyproject.toml` (`asyncio_mode = "auto"`).

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `gateway/_rate_limit.py`                | **new** — `RateLimiter` (token bucket)          |
| `gateway/_middleware.py`                | +`RateLimitMiddleware`; wired into the chain    |
| `business/stream.py`                    | **new** — `MatchStream` pub-sub + `event_stream` async generator |
| `business/app.py`                       | lifespan hook starts the publisher poller; `GET /api/stream/matches` |
| `web/public/app.js`                     | EventSource client + auto-reconnect             |
| `tests/test_rate_limit.py`              | **new** — 13 tests                              |
| `tests/test_sse.py`                     | **new** — 14 tests (incl. async)                |
| `pyproject.toml`                        | `asyncio_mode = "auto"`; +pytest-asyncio in dev deps |
| `TODO.md`, `ARCHITECTURE.md`            | 0.1.1 entries                                   |

### Known limitations (carried forward to 0.2.x)

- **No Redis pub-sub yet** — the SSE publisher polls every 5s.  At
  low traffic this is fine; with 10+ active leagues and 100+ live
  matches the DLTV cache will start to show its age.  0.2.x swaps
  polling for a Redis-backed event bus.
- **No mTLS between gateway and business** — they sit on the
  internal Docker network and trust the gateway's `X-API-Key`.
  0.2.x adds the mTLS handshake per the original ARCHITECTURE
  plan.
- **Rate limit is in-process** — single gateway instance is fine;
  horizontal scaling will need a Redis-backed bucket so all
  gateway replicas share a counter.

---

## [0.2.1] — 2026-07-24

### ML — multi-target regressors + eval harness + winsorize

0.2.0 wired the Strategy pattern with a winner-only model. 0.2.1
extends it to the three numeric targets from `analyze()` and
adds a robust outlier filter and an A/B harness.

- **Four ML heads, one engine.** `MLEngine` now loads any
  combination of `winner` (classifier), `kills` / `duration_mean`
  / `duration_p10` / `duration_p90` (regressors) sub-models and
  overrides the matching blocks in the heuristic result.  Missing
  sub-models leave their block heuristic — a half-trained `MLEngine`
  is still useful and rollouts stay safe.
- **Loss / estimator per target** (per the 0.2.0 modelling shortlist,
  adapted to what sklearn 1.9 / xgboost 3 actually expose):
  - `kills`        → `HistGradientBoostingRegressor(loss="poisson")`
  - `duration_mean`→ `HistGradientBoostingRegressor(loss="gamma")`
                     (Tweedie 1.8 is the 0.2.2 upgrade via XGBoost)
  - `duration_p10` / `duration_p90` → XGBoost `reg:quantileerror`
  - `winner`       → `LogisticRegression` (unchanged from 0.2.0)
- **Robust winsorize (3σ via MAD).** New `business.ml.outliers`
  clips training targets at `median ± n_sigma * 1.4826 * MAD`.
  Median Absolute Deviation is robust to up to 50% outliers — the
  textbook 3σ (`mean ± 3*std`) fails on heavy-tailed data because
  the outliers inflate `std` and cancel the clip.  On a typical
  883-match training run, ~8 kills values and ~14 duration values
  are pulled back inside the bulk; the test set stays untouched.
- **Eval harness — `scripts/eval_engines.py`.** Walks all 1111
  DatDota matches, runs both engines on the same inputs, prints a
  side-by-side table.  Becomes the regression test for "did the new
  model actually beat the heuristic?".

### Trained artifacts (`ml_data/models/*_v1/`)

| Target            | Estimator            | n_train | Test metrics                                  |
|-------------------|----------------------|---------|-----------------------------------------------|
| `winner`          | LogisticRegression   | 888     | accuracy 0.686, log_loss 0.591, ROC AUC 0.769 |
| `kills`           | HistGBR(Poisson)     | 883     | MAE 10.88, RMSE 13.75, Poisson deviance 3.70  |
| `duration_mean`   | HistGBR(Gamma)       | 883     | MAE 9.14, RMSE 12.03                          |
| `duration_p10`    | XGBoost quantile 0.1 | 883     | MAE 11.19, pinball 0.1 = 1.61                 |
| `duration_p90`    | XGBoost quantile 0.9 | 883     | MAE 13.87, pinball 0.9 = 2.82                 |

`X-API-Key` re-running `python -m business.ml.train --target all`
with a new `--version` string mints the next round; production can
pin the active version via the model registry (0.3.0).

### A/B results (heuristic vs ML on 1111 matches)

| Metric                | Heuristic | ML      | Delta    | Winner |
|-----------------------|-----------|---------|----------|--------|
| **winner accuracy**   | 0.494     | **0.690** | **+0.196** | **ML** |
| winner log_loss       | 0.693     | 0.758   | +0.064   | Heuristic ¹ |
| **kills MAE**         | 10.94     | **3.79**  | **-7.15**  | **ML** |
| **kills RMSE**        | 14.72     | **7.84**  | **-6.88**  | **ML** |
| **duration MAE**      | 9.25      | **3.31**  | **-5.94**  | **ML** |
| **duration RMSE**     | 12.96     | **7.49**  | **-5.48**  | **ML** |

¹ Heuristic's perfect 0.693 log_loss reflects it saying "50/50" on
every match — it's "well-calibrated" only because it has no signal
to be miscalibrated about.  The ML's 0.758 is on a model that
*does* take a position; the 0.196 accuracy uplift is the real win.

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `business/ml/outliers.py`               | **new** — MAD-based robust winsorize             |
| `business/ml/targets.py`                | **new** — per-match target extractors           |
| `business/ml/regressors.py`             | **new** — sklearn / xgboost factory functions   |
| `business/ml/engine.py`                 | `MLEngine` now multi-target; `make_engine` loads every known sub-model |
| `business/ml/train.py`                  | `--target {winner,kills,duration_mean,duration_p10,duration_p90,all}`; winsorize; per-target metrics |
| `tests/test_ml_outliers.py`             | **new** — 20 tests                              |
| `tests/test_ml_targets.py`              | **new** — 21 tests                              |
| `tests/test_ml_regressors.py`           | **new** — 14 tests                              |
| `tests/test_ml_engine.py`               | refactored to multi-target contract             |
| `scripts/eval_engines.py`               | **new** — A/B harness                           |
| `scripts/smoke_ml_0_2_1.py`             | **new** — multi-block smoke test                |
| `ml_data/models/{kills,duration_mean,duration_p10,duration_p90}_v1/` | **new** — first multi-target training run |
| `pyproject.toml`                        | `version` → 0.2.1; +xgboost in deps            |
| `requirements.txt`                      | +xgboost                                        |
| `TODO.md`, `ARCHITECTURE.md`            | 0.2.1 entries                                   |

### Known limitations (carried forward to 0.2.x)

- **Towers regressor is implemented but not trained.** The
  corpus (`ml_data/full_matches/*.json`) does not carry per-side
  tower bitmasks; only duration / kills / winner are extractable.
  0.2.2 will re-pull matches from a tower-aware source (DLTV)
  and add a ZINB regressor for the heavy zero-inflation.
- **Multikill / first_to_15 still heuristic.** Categorical and
  secondary metrics; 0.3.0.
- **No nightly retrain scheduler yet.** Manual CLI only.
- **ML log_loss slightly worse than heuristic on 1111** because
  the ML commits to a direction.  The 0.196 accuracy uplift is
  the trade-off we want; if log_loss matters more in a future
  eval we'll calibrate the model outputs (Platt / isotonic) in
  0.2.2.

---

## [0.2.0] — 2026-07-24

### ML — Strategy pattern + winner-only MVP

The monolith heuristic engine is no longer the only option. The new
`business.ml.*` package wires a pluggable `IPredictionEngine` behind a
single process-wide switch, and the first trained model on real data
is shipped in the same release.

- **`IPredictionEngine` Strategy.** `business.ml.engine` defines the
  contract; `HeuristicEngine` wraps the existing `analysis.analyze()`,
  `MLEngine` overrides only the `winner` block of the heuristic result.
  All other metrics (kills, towers, duration, first-to-15, multikill,
  confidence) stay heuristic for now. Engine selection is driven by
  the `PREDICTION_ENGINE` env var (`heuristic` | `ml`).
- **Per-hero target encoding for the model input.** `HeroWinRateEncoder`
  fits a per-(side, hero_id) win-rate table with smoothing back to the
  global rate. No circular features — `duration`, `kills`, `gpm`, `xpm`
  are all explicitly excluded (they're only known after the match).
- **Single source of truth for the feature schema.** `FEATURE_ORDER`
  and `N_FEATURES` in `business.ml.features` are shared between trainer
  and predictor. `ModelStorage.load()` refuses to load a model whose
  `feature_names` no longer match the live `FEATURE_ORDER` — the old
  "stale model" footgun is now caught at load time, not at predict time.
- **Versioned model store with a JSON sidecar.** `ModelStorage.save()`
  writes a `model.joblib` and a `metadata.json` (sklearn_version,
  numpy_version, python_version, feature_names, metrics, train_data,
  encoder) under a `{name}_v{version}/` directory. The encoder is
  round-tripped through the metadata, so model + encoder are guaranteed
  to come from the same training run.
- **Atomic writes.** Models are written to a tmp file in the target
  dir, then `os.replace`’d into place. A half-written `model.joblib`
  can no longer crash the load path.
- **CLI: `python -m business.ml.train`.** Loads `ml_data/full_matches/*.json`,
  fits the encoder, trains `LogisticRegression` (or `--model histgb`
  for `HistGradientBoostingClassifier`), computes accuracy / log_loss /
  ROC AUC on a stratified 80/20 split, saves via `ModelStorage`.

### First trained model

- **Corpus:** 1111 DatDota matches (manifest
  `ml_data/imports/2026-07-24-telegram-desktop.json`); balanced labels
  (549 radiant wins / 562 dire wins).
- **Estimator:** `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")`.
- **Test metrics (held-out 20%):**
  - accuracy = **0.686**
  - log_loss = **0.591** (below the 0.693 baseline of a constant 50/50)
  - ROC AUC = **0.769**
- **Stored at** `ml_data/models/winner_v1/` (`model.joblib` + `metadata.json`).
- **Smoke test:** `scripts/smoke_ml_0_2_0.py` runs both engines on
  5 historical matches side-by-side; the `ml` path picked the right
  winner in 4/5, the heuristic baseline in 2/5.

### Wiring

- `business/app.py` — startup hook warms the engine; `GET /api/board`
  now returns `"engine": "heuristic" | "ml"` in the response so the
  caller can see which engine produced the predictions.
- `business/board.py` — `_postmatch_prediction`, `_postmatch_card`,
  `_live_card`, and `analyze_map_with_verdict` now route through
  `get_default_engine()`.
- `business/analysis.py` — `analyze_map_with_verdict` accepts an
  optional `engine` kwarg (default `None` = heuristic, preserves the
  v0.0.x behaviour for tests).

### Removed

- `business/ml_trainer.py` — the old isolated `MLTrainer` skeleton is
  gone. Its responsibilities (data loading, training, persistence) are
  now split across `business.ml.train` (CLI) and `business.ml.storage`
  (file-format). The `print(...)` calls it was full of are gone with it.

### Dependencies

- `numpy>=1.24`, `scikit-learn>=1.3`, `joblib>=1.3` promoted to
  runtime deps. `pip install -r requirements.txt` now pulls a
  working ML stack. ~30 MB added to the `business` Docker image.
- `pyproject.toml` — `business.ml` added to `packages`; the `[ml]`
  extra now carries only `pandas` + `matplotlib` for offline analysis.

### Files added / changed

| File / dir                              | Change                                          |
|-----------------------------------------|-------------------------------------------------|
| `business/ml/features.py`               | `HeroWinRateEncoder`, `FEATURE_ORDER`, `extract_features`, `hero_ids_from_match`, `target_from_match` |
| `business/ml/storage.py`                | **new** — `ModelStorage`, `LoadedModel`, `ModelMetadata`, atomic save/load |
| `business/ml/engine.py`                 | **new** — `IPredictionEngine`, `HeuristicEngine`, `MLEngine`, `make_engine()`, `get_default_engine()` |
| `business/ml/train.py`                  | **new** — CLI `python -m business.ml.train`     |
| `business/ml/__init__.py`               | updated docstring                               |
| `business/app.py`                       | startup hook warms engine; `get_board` returns `engine` field |
| `business/board.py`                     | `analyze` calls routed through `get_default_engine()` |
| `business/analysis.py`                  | `analyze_map_with_verdict` accepts `engine` kwarg |
| `business/ml_trainer.py`                | **removed** — replaced by `business.ml.*`       |
| `scripts/smoke_ml_0_2_0.py`             | **new** — side-by-side engine comparison        |
| `tests/test_ml_features.py`             | **new** — encoder + feature schema (16 tests)   |
| `tests/test_ml_engine.py`               | **new** — engine strategy + fallback (22 tests)  |
| `ml_data/models/winner_v1/`             | **new** — first trained artifact (model.joblib + metadata.json) |
| `pyproject.toml`                        | version → 0.2.0; `business.ml` in packages; `scikit-learn`/`numpy`/`joblib` in main deps |
| `requirements.txt`                      | +numpy, +scikit-learn, +joblib                  |
| `.env.example`                          | `PREDICTION_ENGINE` comments updated            |
| `TODO.md`, `ARCHITECTURE.md`            | 0.2.0 entries; strategy pattern now realised    |

### Known limitations (carried forward to 0.2.x)

- **Only the `winner` block is overridden by ML.** Kills / towers /
  duration / first-to-15 / multikill all stay heuristic. 0.2.1 adds
  the regressors (Tweedie / Gamma / quantile per the modelling
  shortlist).
- **No A/B harness yet.** The smoke-test script is a one-off; an
  automated Heuristic vs ML comparison over the full 1111-match
  corpus is the first deliverable in 0.2.1.
- **No model registry service.** Models live in `ml_data/models/`,
  read directly from disk at process start. 0.3.0 swaps in
  `IModelRepository` over Postgres + S3.
- **No retrain scheduler.** `python -m business.ml.train` is the
  only way to mint a new version. APScheduler wiring lands in 0.2.1.

---

## [0.1.0] — 2026-07-24

### Architecture (first `major` bump — the monolith is gone)

- **Three services split out** — the old `uvicorn backend.app:app` monolith is replaced by:
  - `web/` — nginx serving the static frontend, **zero Python**
  - `gateway/` — FastAPI security + routing layer (auth, rate limit, correlation IDs, body-size cap, CORS allowlist)
  - `business/` — FastAPI service with the heuristic engine, ML trainer, API clients (replaces `backend/`)
- **Two networks in `docker-compose.yml`** — `frontend` (exposed) and `backend` (internal-only, no host access).
- **Only `web` publishes a port** (80). The gateway and business are reachable only from inside the compose network.
- **Frontend is "dumb"** — the nginx site only renders static files and reverse-proxies `/api/*` to the gateway. No business logic in JS, no API keys in browser code.

### Auth (the design is auth-aware from day one)

- **`X-API-Key` required on every `/api/*` and `/internal/*` request.** Configured via `DEV_API_KEY` env var. Wrong key → 401. Empty server config → 500 (so we never accidentally serve unauthenticated).
- In dev / Docker, the key is shared between gateway and the curl side. In prod, this will be replaced by per-user API keys + HMAC between services.

### Middleware stack (gateway)

- **`CorrelationIdMiddleware`** — generates / propagates `X-Request-Id` end-to-end so a user report can be traced through gateway → business.
- **`AccessLogMiddleware`** — one structured log line per request with method, path, status, latency, request-id.
- **`BodySizeLimitMiddleware`** — rejects requests with `Content-Length > MAX_BODY_BYTES` (1 MB default) before they hit the proxy.
- **`ApiKeyAuthMiddleware`** — `X-API-Key` check on protected paths.
- **`CORSMiddleware`** — allowlist from `CORS_ORIGINS` env, only the methods/headers we need.
- Order: CORS → Auth → Body size → Access log → Correlation (outer wraps inner).

### Frontend deployment

- **`web/nginx.conf`** — adds `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` security headers. Caches static assets 7 days, sets `Cache-Control: immutable`. SSE passthrough with `proxy_buffering off` and 24h read timeout.
- **`web/public/`** — the static bundle (HTML/CSS/JS) — no build step in 0.1.0, we ship raw files.

### Tooling

- **`docker-compose.yml`** — three services + two networks + healthchecks. `make docker-up` brings the stack up; `make docker-down` tears it down.
- **Three Dockerfiles** under `docker/` (`Dockerfile.web`, `Dockerfile.gateway`, `Dockerfile.business`). All use `python:3.12-slim` or `nginx:1.27-alpine` base images, run as non-root.
- **`Makefile`** — added `docker-build`, `docker-up`, `docker-up-d`, `docker-down`, `docker-logs`, `docker-ps`, `docker-shell`. `run-business` and `run-gateway` for per-service dev.
- **`httpx>=0.27`** added to deps — used by gateway for async proxying (and business for any future async client work).

### Schema / imports

- `backend/` was renamed to `business/`. All in-tree `from .X` imports use the new name. `pyproject.toml` now declares `packages = ["gateway", "business"]`. Tests import `from business.X import ...`.
- CORS env var was `allow_origins=["*"]`; now `CORS_ORIGINS` from env (default `http://localhost`).

### Files added / changed

| File / dir                          | Change                                          |
|-------------------------------------|-------------------------------------------------|
| `gateway/app.py`, `_middleware.py`, `_proxy.py`, `__init__.py` | **new** — security + reverse-proxy service |
| `web/public/` (moved from `frontend/`) + `web/nginx.conf` + `web/README.md` | **new layout** — dumb static frontend |
| `business/` (renamed from `backend/`) | same code, new home, no public routes |
| `business/app.py`                    | removed static serving, added `/api/healthz` + `/api/readyz` |
| `docker/Dockerfile.web`             | **new** — nginx 1.27-alpine                     |
| `docker/Dockerfile.gateway`          | **new** — python 3.12-slim + gateway package     |
| `docker/Dockerfile.business`         | **new** — python 3.12-slim + business package, mounts ml_data |
| `docker-compose.yml`                 | **new** — web + gateway + business, two networks |
| `pyproject.toml`                     | packages = [gateway, business]; httpx added    |
| `requirements.txt`                    | +httpx                                          |
| `Makefile`                           | new docker targets                              |

### Known limitations (carried forward to 0.1.x)

- Rate limit middleware **not yet implemented** — gateway auth is in place but per-key throttling is still TODO. Will be Redis-backed in 0.1.1.
- **SSE endpoint not yet wired** — `web/nginx.conf` is ready (proxy_buffering off, 24h timeout), but no `/api/stream/matches` exists in `business/` yet. Will land in 0.1.1.
- **No Redis yet** — for 0.1.0 everything is in-process. Redis is in the docker-compose plan but not in 0.1.0.
- ML trainer is still an isolated skeleton. Blocked on data, deferred to 0.2.0.

---

## [0.0.1] — 2026-07-24

### Security

- **CORS locked down** — `app.py` now reads `CORS_ORIGINS` from environment (default `http://localhost:8000`) instead of allowing `*` for all origins/methods/headers. Rationale: RULES.md §7 explicitly warns that `*` is fine for local dev only; this was a 1-line config leak waiting for a public deploy.
- **Exponential backoff for DatDota client** — replaced fixed 3 s / 10 s sleeps with `1.0 * 2^attempt + jitter` capped at 30 s, with `Retry-After` honoured when the server sends it. Rationale: RULES.md §1 requires exp backoff; fixed sleeps were burning the 500 req/day budget under burst failures.

### Hardening

- **dltv_client now retries** — was a single-shot urllib call that violated RULES.md §1. Refactored to share the retry/backoff utility with DatDota.
- **Shared `backend/_http.py`** — single source of truth for HTTP retry policy. Future clients automatically get consistent behaviour.
- **`backend/__init__.py` tolerant to missing `dotenv`** — `from backend import …` no longer requires `python-dotenv` to be installed, so pure-function tests don't drag in env-loading code.

### Maintainability

- **Magic numbers → named constants in `analysis.py`** — 18 new module-level constants (e.g. `WINNER_WR_WEIGHT = 0.045`, `KILLS_OVER_UNDER_THRESHOLD = 50`) with comments explaining their role. The `analyze()` function is now readable without inlining the algorithm math.
- **Dead code removed** — `DLTVClient._val` was unused; deleted.
- **`import os` moved to top of `discovery.py`** — was a stray import at line 387.

### Testing

- **First test suite** — `tests/test_analysis.py` with 23 tests covering `analyze()`, `decode_towers()`, `map_verdicts()`, `analyze_mapWith_verdict()`. Reaches the heuristic engine, the tower bitmask, and the post-match verdict comparator.
- **`tests/conftest.py`** — sample team/hero fixtures and a sys.path fix so `from backend.analysis import …` works in pytest without installing the package.
- **`requirements-dev.txt`** — `-r requirements.txt` + `pytest>=7.0` + `pytest-cov>=4.0`.

### Bugfixes caught by the new tests

- **`map_verdicts` raised `TypeError` on `None` actuals** — when `actual.kills_total` or `actual.duration_min` was `None` but the prediction had an int `threshold`, `actual > threshold` blew up. Added `isinstance(..., (int, float))` guards in both over/under blocks; missing data now returns `None` (cannot compare) instead of crashing the post-match card render.

### Files touched

| File                          | Change                                           |
|-------------------------------|--------------------------------------------------|
| `backend/app.py`              | CORS from env                                     |
| `backend/_http.py`            | **new** — shared HTTP retry utility                |
| `backend/analysis.py`         | constants; None-guards in `map_verdicts`          |
| `backend/dltv_client.py`      | retry via `_http`; deleted `DLTVClient._val`      |
| `backend/datdota_client.py`   | retry via `_http`                                  |
| `backend/discovery.py`        | `import os` moved to top                           |
| `backend/__init__.py`         | tolerant `load_dotenv`                             |
| `tests/conftest.py`           | **new**                                            |
| `tests/test_analysis.py`      | **new** — 23 tests                                 |
| `requirements-dev.txt`        | **new**                                            |

### Known limitations (carried forward)

- `ml_trainer.py` is still an isolated skeleton; not wired into `app.py`/`board.py`. Deferred — will be addressed when real ML training data is available.
- No CI pipeline yet (tests run locally only).
- No Docker / multi-node deployment.
- Frontend still does rendering only; not a separate "dumb" service yet (see ARCHITECTURE.md).

---

## [0.0.0] — pre-2026-07-24

Initial repository state from before the code-review pass:

- FastAPI backend with static frontend, heuristic draft-analysis engine.
- DLTV / DatDota / Steam API clients (no shared retry, no CORS config).
- ML training module skeleton (`ml_trainer.py`).
- `ml_data/` with 450 collected DatDota match files.
- Documentation: `README.md`, `AGENTS.md`, `RULES.md`, `TODO.md`.
