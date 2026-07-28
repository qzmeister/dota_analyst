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

## Что **не** вошло (запланировано в 0.4.0, зафиксировано в TODO.md)

- **Direct socket.io client** из Python (без chromium вообще, ~1s fetch, снимает greenlet error, parallelizable)
- **Live data fallback chain**: socket.io → `/live/{id}.json` → Steam `GetLiveLeagueGames` → "unavailable"
- **Async playwright refactor** (настоящая починка chromium greenlet error)
- **Cookie-based SSE auth** (разблокирует public deploy)
- **Postgres** миграция
- **Multikill classifier** — discontinued (нужен amateur-корпус с редкими multikill-кейсами)
- **Towers regressor** — discontinued до появления tower bitmask в `full_matches` (или mini-map icon scraping)
- **Audit P1-15**: `except Exception` sites в business-модулях, которые swallow + log без re-raise
