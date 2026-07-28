## v0.3.24 → v0.3.25j: live data reliability + DLTV-style live card + ML v16 + UI honesty

Спринт из **16 коммитов**, ~1 200 строк, 433/433 теста зелёные.  Три смысловых блока:

1. **v0.3.24 (a–h)** — надёжность live-данных и приведение live-карточки к виду DLTV
2. **v0.3.25** — overnight grid research → 5 production-моделей v16 (kills / duration × 3 / winner)
3. **v0.3.25e–j** — UI-честность (towers / Ultra Kill) + браузерный кэш + bans-строка + тема + radio-кнопки

---

## 1. Live data reliability (v0.3.24 a–d, e, f, h)

Карточка лайв-матча раньше показывала пустые пики / score / time сразу после того, как discovery-tracker вычищал завершённый матч. Кэш с данными был на диске, но лукап по нему ломался — вотчлист знает только `steam_id`, а ключ кэша был `dltv_id`.

| Commit | Что |
|--------|-----|
| `704c0eb` | `LIVE_HIDE_STEAM_ONLY=1` — отрезает 44 китайских любительских матча |
| `b626a8e` | `watch-/steam-` → dltv-id mapping для cache lookup |
| `622f275` | dedup Steam+Scraper double-adds в `get_live_and_prematch` |
| `60ef38a` | dual-id namespace fix в `_picks_to_heroes` (Hoodwink↔Pangolier коллизия) + publisher 30s→5s |
| `d184aac` | `wait_for_function` вместо фиксированного `wait_for_timeout(3.5s)` — фетч 2-3s вместо 5-10s + `MATCH_STATE_TTL_SEC` 30s→8s |
| `8f70d7d` | TTL 8s→1h, alias-ключ `s{steam_id}` + `_dltv_series_id` из `/live/{id}.json` + оверлей time/networth даже когда пики пустые |

## 2. DLTV-style лайв-карточка (v0.3.24g, h)

- **networth + game time** в реальном времени, извлекаются из `.team__networth > .networth > span` + `.info__duration[data-game-time]`
- **TM/ТБ формат** для kills / duration / towers — такой же, как в постматч-карточке
- **3-колоночная раскладка**: team-side / score+time+gold / team-side, большие иконки героев 44×56 (картинка + имя) **под названием команды**, серия-скор и `Игра N` под центром, partial-gold block для случая "одна сторона известна"
- коллапс в одну колонку на узких экранах (≤700px)
- destroyed-tower counts в лайве **не реализованы** (DLTV отдаёт их только как иконки на мини-карте, без текста/JSON) — отложено до reverse-engineering socket.io payload

## 3. ML v16 (v0.3.25) — overnight grid research

Overnight grid research с **1 200+ комбинациями** (model × hyperparams × feature-groups) через 5-fold CV с честным refit энкодера на каждом фолде. Выбрали реально-лучшую honest-конфигурацию для каждой target (а не leaky in-sample winner) и зашили v16 production-модели.

Что гоняли:
- 3 семейства моделей: XGBoost (Poisson / squared / Tweedie / quantile), HistGradientBoosting (Poisson / gamma / squared), sklearn LogReg (с/без calibration) + Ridge
- 5 таргетов: `kills`, `duration_mean`, `duration_p10`, `duration_p90`, `winner`
- 10 feature-group подмножеств на таргет (hero, team, lane, matchup, patch, player; отдельно + лучшие комбинации)
- 5-fold CV с энкодером, **обученным только на train-фолде** — иначе target-encoding сливает test-строку через lookup

**Honest 5-fold CV (encoder refit per fold, 2 380 matches):**

| Target | Old "honest" | New v16 honest | Δ | Notes |
|--------|--------------|----------------|---|-------|
| kills MAE | 11.95 (v3) | **11.56** | −0.39 | target не в фичах → minor leak; honest ≈ same |
| duration MAE | 9.07 (v3) | **8.67** | −0.40 | target не в фичах → minor leak |
| winner acc | 67.6% (v15)* | **60.04%** | (leak) | *v15 leaked — real honest ≈ 60%* |
| winner logloss | n/a | **0.7109** |   | best logloss = 0.6992 (logreg_c0.5) |

"−3% kills / −4% duration" — это **реальные** улучшения (target не в фичах, leak маленький). "winner 67.6 → 60.0%" — **исправление измерения**, а не регрессия: v15-овский "honest" был раздут encoder-утечкой.

**v16 configs:**
- `kills`         → XGBoost Poisson,   n=100 / d=4 / lr=0.1, on `[hero, player]` (17 feat)
- `duration_mean` → XGBoost squared,    n=50  / d=2 / lr=0.1, on `[hero, team, player, matchup]` (24 feat)
- `duration_p10`  → XGBoost quantile α=0.1, n=50 / d=3 / lr=0.1, on all 6 groups (34 feat)
- `duration_p90`  → XGBoost quantile α=0.9, n=50 / d=3 / lr=0.1, on all 6 groups (34 feat)
- `winner`        → sklearn LogReg,     C=1.0,  on `[hero, team, player]` (21 feat)

**Что не сработало:**
- Большие feature-наборы (все 6 групп = 34 feat) выигрывают на leaky in-sample grid, но проигрывают на honest CV для KILLS — лишние фичи добавляют шум регрессионным головам. Для quantile-голов помогают (больший spread → лучше откалиброваны).
- RandomForest / ExtraTrees: дропнуты (медленно + без улучшения на honest CV).
- Tweedie variance power должен быть `< 2.0` (constraint XGBoost); v0.3.12-овский gamma proxy через `reg:tweedie` с `tweedie_variance_power=2.0` крашится.
- Calibrated LogReg (`CalibratedClassifierCV(sigmoid, cv=3)`) не улучшает winner-голову материально относительно plain LogReg.

**Новые скрипты** в `scripts/`:
- `grid_night.py`             — полный grid (~1 100 конфигов, single-thread, часы CPU)
- `grid_honest_winner.py`     — 75 winner-конфигов, honest 5-fold CV
- `grid_honest_regressors.py` — KILLS + DURATION, honest 5-fold CV
- `compare_v15_honest.py`     — 80/20 mirror v15-протокола, доказывает утечку
- `grid_summarize.py`         — живой top-N дамп во время работы grid-а
- `train_v16.py`              — обучить все 5 v16 production-моделей на полном корпусе

## 4. UI-честность и deploy-pipeline фиксы (v0.3.25e–j)

UI-only патчи поверх v0.3.25. Никаких изменений backend-а или моделей.

| Patch | Что |
|-------|-----|
| `3f4e7a9` (v0.3.25e) | **Скрыть весь towers UI** — postmatch stat, postmatch prediction, live card prediction. Число бралось из эвристики (нет per-side tower bitmask в `full_matches`), оно вводило в заблуждение. Закомментировано в трёх местах с явной пометкой "Uncomment when we have a real per-side tower source". |
| `db4db38` (v0.3.25f) | **Ultra Kill / Rampage скрыты** — multikill-классификатор вырождался в "always High" на про-корпусе (см. `HEAD_REGISTRY['multikill']` в `train.py`). Discontinued, закомментирован. <br> **Auto-refresh radio** — `Вкл / Выкл` pill group вместо старого checkbox (touch-friendly, единый источник истины для `isAutoRefreshOn()`). <br> **Theme switcher** — `Тёмная / Светлая`, в `localStorage.dota_analyst_theme`, применяется к `<html data-theme="…">` чтобы избежать FOUC. Светлая палитра — GitHub-light. <br> **Hide empty leagues** — `LEAGUES = (d.leagues \|\| []).filter((l) => (l.match_count \|\| 0) > 0)` отрезает пустые лиги. |
| `8e018b5` (v0.3.25g) | **Stale-league-ID auto-reset** — если `localStorage.dota_analyst_leagues` содержит только ID, которых больше нет в текущем `/api/leagues` ответе (после server-side ротации ID), `loadLeagues()` сбрасывает выбор на все текущие ID вместо показа пустой доски. |
| `1171cd3` (v0.3.25h) | **JS-синтаксис фикс** — лишние backticks внутри HTML-комментария, который сам лежал внутри backtick-template-literal (`<!-- that fills \`p.multikill\` is a guess -->`) рано закрывали literal и ломали весь `init()`. Заменено на без-backtick комментарий. **Больше никаких backticks в HTML-комментариях внутри template-literals** — добавлено в личный gotcha-list. |
| `a04bfd1` (v0.3.25i) | **Static bans row под пиками** в live-карточке — 28×28 десатурированные (opacity .55, grayscale .7) иконки героев с красной диагональной `::after` зачёркивалкой + `BANS` лейбл, всегда видно (без collapse). Зеркалит DLTV-подачу без лишнего клика. |
| `3d1217b` (v0.3.25j) | **nginx cache-control** — убрать `immutable` со static-ассетов, чтобы `?v=X.Y.Z` query-string реально сбивал браузерный кэш. Сам `index.html` получает `no-cache, must-revalidate` чтобы свежие bundle-указатели подхватывались всегда. |

## Тесты

- 418 → 433 (+15): networth extraction, dual-id picks, partial-gold block, cache alias (steam+dltv ключи), wait_for_function timeout fallback, `towers_over_under` consistency
- v0.3.25 + e/f/g/h/i/j: 433/433 pass (всё либо ML-внутри, либо UI-only, без изменения public contract)

---

## v0.3.25 → 0.4.0 (roll-forward + perf + real-time)

Спринт из **8 коммитов** поверх v0.3.25. Два смысловых блока: (1) **roll-forward после частичного rollback'а v0.3.25l-t** (live cards исчезли из UI), (2) **v0.4.0 — direct socket.io + 40-200× build speedup**.

### 1. Roll-forward (v0.3.25k → 0.4.0 base)

Пользователь откатил v0.3.25l-t (живые карточки пропали) к `ec61adb` (v0.3.25k). Затем ре-применили только безопасные куски + починили регрессии:

| Commit | Что |
|--------|-----|
| `dbf0c8e` (v0.3.25k-patch) | Re-applies **только** v0.3.25t publisher daemon-thread фикс (без него v0.3.25k падает в `asyncio.to_thread` deadlock). |
| `2e31936` (v0.3.25l+m) | Re-applies socket.io hook + `_parse_clock_seconds()` game_time MM:SS→int + `m["duration"]` fallback для watchlist матчей. |
| `7b51c2e` (v0.3.25r re-applied + cache trust) | Re-applies TBD-vs-TBD фильтр в **обоих** местах (`_filter_auto_board` в app.py + `build_board` в board.py) + bypass broken dltv_browser cache когда `/live/{id}.json` имеет реальные данные. |
| `dbef2a4` (v0.3.25l-bugfix) | **Socket.io hook filter by `expected_steam_id`** — без этого фильтра hook перехватывал payload'ы от соседних матчей (live ticker / sidebar / chat) и кэш забивался мусором (50:1, 12:35) для чужих матчей. |
| `6f4ad5b` (backup branch `backup-v0.3.25t-broken`) | Полный v0.3.25l-t + 30+ diag-скриптов сохранены на side-branch. `git checkout backup-v0.3.25t-broken -- business/` восстанавливает. |

### 2. v0.4.0 — real-time live data + 40-200× build speedup

| Commit | Что |
|--------|-----|
| `c857fc7` | **`business/dltv_socket.py`** — direct WebSocket клиент к `wss://dltv.org/socket.io/?EIO=4&transport=websocket`. EIO=4 + SIO EVENT протокол реализован вручную (без `python-socketio` зависимости). Public API: `get_live_state(steam_id)`, `subscribe()`, `unsubscribe()`, `start_socket_client()`, `stop_socket_client()`. State в module-level `_state` + `_state_ts` (60s TTL) под `RLock`. Lifespan wired в `app.py`; publisher подписывает каждый live-матч после каждого build. **Real-time данные подтверждены**: `live_score 15-24, game_time 1690` совпало с DLTV UI. |
| `a59170a` | **Drop WS-level PINGs** (`ping_interval=None`). Standalone test: default `ping_interval=20` доживает 120s+ в изоляции, но в app context (uvicorn + publisher builds + dltv_browser scrapes делят bandwidth контейнера) с WS-pings падает в 30-60s, без — 2-3 мин. Сервер, видимо, трактует WS-level PING как активность которая ротирует сессию; полагаемся на EIO PING/PONG (server-initiated) + `__nd2_series` channel для keepalive. |
| `b45e46d` | **`_last_good_board` fallback в publisher thread** — когда `build_board` возвращает пустой payload (DLTV scrape упал, Steam лежит, enrichment таймауты), publisher отдаёт последний непустой кэшированный board если ему < 5 мин, и бампит `_latest_auto_board_ts` чтобы UI-овские "обновлено HH:MM" продолжали тикать. Чинит 0/0/0 dead-board. |
| `7c0b178` | **Parallel `/live/{id}.json` enrichment (40-200× speedup)**. `get_live_json` урезан с `retries=3, timeout=3s, backoff=1+2+4s` до `retries=1, timeout=1.5s, no backoff` (одна быстрая попытка; следующий 5s TTL всё равно повторит). `tracker.get_live_and_prematch` фан-аутит enrichment через `ThreadPoolExecutor(max_workers=6)`. 22 матча × 1.5s / 6 workers ≈ 6s; наблюдаемо cold = 3.5s, warm = 0.76-0.9s. Плюс: если enrichment упал (timeout/404), синтезируем v1-shaped series с `_live_enrich_failed=True` — карточка остаётся в board вместо тихого исчезновения. |

### Результаты

**Build time** (логи контейнера):
```
17:03:22  build_board done in 3.53s (live=10)   ← cold
17:03:29  build_board done in 2.30s (live=10)
17:03:35  build_board done in 0.76s (live=10)   ← warm
17:03:41  build_board done in 0.91s (live=10)
17:03:47  build_board done in 0.82s (live=10)
```
Было: 200-500 секунд. **40-200× ускорение.**

**Live data flow** (v0.4.0 — direct socket.io, no chromium):
```
                         ┌─────────────────────────────────────────┐
                         │           dltv.org /socket.io            │
                         │     wss://dltv.org/?EIO=4&transport=ws   │
                         │  channels: __nd2_match_{steam_id},       │
                         │           __nd2_series, __nd2_odds_*     │
                         └───────────────────┬─────────────────────┘
                                             │  EIO=4 frames
                                             │  0{json} → 40{sid}
                                             │  42["ch", args]
                                             │  2/3 (PING/PONG)
                                             ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  business/dltv_socket.py (daemon thread, private asyncio loop)   │
   │                                                                  │
   │  on connect:  subscribe to __nd2_series + all known match_ids   │
   │  on event:    extract steam_id from /__nd2_match_(\d+)/         │
   │               check payload["match_id"] == sid                   │
   │               if match → _state[sid] = payload, _state_ts[sid]=t │
   │  on drop:     backoff 1→2→4→8→16→30s, reconnect forever         │
   │  no WS-PINGs (a59170a) — relies on EIO PING/PONG + series ch    │
   └──────────────────────────┬───────────────────────────────────────┘
                              │  module-level state under RLock
                              │  (publisher thread reads from any thread)
                              ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  business/board.py — _live_card()  (publisher thread, each build)│
   │                                                                  │
   │  overlay = dltv_socket.get_live_state(steam_id)                  │
   │            └─ if fresh (< 60s TTL):  use it (REAL-TIME)          │
   │            └─ else:                  fall back to dltv_browser   │
   │                                       cache (5min TTL) or        │
   │                                       m["duration"] / m["score"] │
   │                                       from /live/{id}.json       │
   └──────────────────────────┬───────────────────────────────────────┘
                              │  JSON board
                              ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  business/stream.py — SSE publisher (asyncio half)               │
   │  publish_if_changed(board) → SSE → EventSource("/api/stream/")   │
   └──────────────────────────┬───────────────────────────────────────┘
                              │  data: {...live: [...]} every 5s
                              ▼
                          Browser UI
```

**Live data fallback chain** (если socket.io connection упал):
```
  ┌─ request live state for steam_id X ─────────────────────────────┐
  │                                                                 │
  │  1. dltv_socket.get_live_state(X)                               │
  │     └─ fresh? (< 60s) ─── yes ──► use it  (REAL-TIME)           │
  │                                                                 │
  │  2. dltv_browser.get_cached_match_state(series_id)              │
  │     └─ has real data? (radiant_score > 0 OR game_time > 240)    │
  │         └─ yes ──► use cache  (5min TTL)                        │
  │                                                                 │
  │  3. /live/{id}.json (5s TTL in tracker._series_cache)           │
  │     └─ has picks/score/time?  ── yes ──► use it                 │
  │                                                                 │
  │  4. m["radiant_score"] / m["dire_score"] from /live adapter     │
  │     └─ non-zero?  ── yes ──► trust over cache                   │
  │                                                                 │
  │  5. _live_enrich_failed synthesis (v0.4.0-perf)                 │
  │     └─► render card with empty picks + "Live state unavailable" │
  │                                                                 │
  │  ★ _last_good_board (b45e46d) — if entire build is empty,        │
  │    fall back to last non-empty cached board if < 5min old.       │
  │    Bumps _latest_auto_board_ts so "обновлено HH:MM" keeps moving│
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

**Build time before/after** (v0.4.0-perf — parallel enrichment):
```
  per-cycle build_board (22 live matches)

  before (v0.3.25l-t):                after (v0.4.0):
  ─────────────────────               ─────────────────────
  Steam API        0.7s               Steam API        0.7s
  DLTV scrape      0.5s               DLTV scrape      0.5s
  ┌────────────── sequential ──┐      ┌── parallel (6 workers) ──┐
  │ /live/8918...json  3.7s   │      │ /live/8918...json  1.5s  │
  │ /live/8918...json  3.7s   │      │ /live/8918...json  1.5s  │
  │ /live/8918...json  3.7s   │      │ /live/8918...json  1.5s  │
  │  ... (×22) ...             │      │ /live/8918...json  1.5s  │
  │ /live/8918...json  3.7s   │      │ /live/8918...json  1.5s  │
  │                            │      │ /live/8918...json  1.5s  │
  │ 22 × 3.7s = 81s           │      │ 22 / 6 = ~6s             │
  └────────────────────────────┘      └──────────────────────────┘
  ML predict        0.5s               ML predict        0.5s
  ─────────────────────               ─────────────────────
  TOTAL:  ~85s cold                   TOTAL:  ~8s cold
          200-500s observed                   0.76-3.5s observed
                                                         (40-200×)
```

**Live card** (скриншоты):
- Real-time score/time/lead для матчей в live (KW vs PuckChamp: 15-24, 28:10)
- `__nd2_match_{steam_id}` events приходят без chromium (raw WebSocket)
- При drop'е соединения — fallback на `/live/{id}.json` (5s TTL) + `_last_good_board` (5min TTL)

**Версия** в `business/app.py`:
- `0.3.19` → `0.4.0` (никогда не бампался через 0.3.x)

## Что **не** вошло (запланировано в 0.4.1+, зафиксировано в TODO.md)

- **Dual-instance socket** — DLTV-овский server-side limit 30-150s на сессию. Run 2 sockets параллельно + round-robin: continuous real-time. ~50 LoC, более сложная реконсиляция. 0.4.1.
- **Async playwright refactor** — настоящая починка chromium greenlet error (одно-воркер + probe только частично). 0.4.2.
- **Cookie-based SSE auth** — browser `EventSource` не поддерживает custom headers. Public deploy blocker. 0.4.2.
- **Postgres** миграция + auto-retrain + observability (Prometheus + Grafana + OTel). 0.5.0.
- **Multikill classifier** — discontinued (нужен amateur-корпус с редкими multikill-кейсами).
- **Towers regressor** — discontinued до появления tower bitmask в `full_matches` (или mini-map icon scraping).
- **Audit P1-15**: `except Exception` sites в business-модулях, которые swallow + log без re-raise.
