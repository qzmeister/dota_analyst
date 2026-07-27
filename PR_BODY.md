## v0.3.24 (a-d, e, f, g, h): live data reliability + DLTV-style live card

Спринт по надёжности live-данных и приведению лайв-карточки к виду DLTV. 8 коммитов, ~750 строк, 433/433 теста зелёные.

### Live data reliability (v0.3.24 a-d, e, f, h)

Карточка лайв-матча раньше показывала пустые пики / score / time сразу после того, как discovery-tracker вычищал завершённый матч. Кэш с данными был на диске, но лукап по нему ломался — вотчлист знает только steam_id, а ключ кэша был dltv_id.

| Commit    | Что |
|-----------|-----|
| `704c0eb` | `LIVE_HIDE_STEAM_ONLY=1` — отрезает 44 китайских любительских матча |
| `b626a8e` | `watch-/steam-` → dltv-id mapping для cache lookup |
| `622f275` | dedup Steam+Scraper double-adds в `get_live_and_prematch` |
| (cont)     | два формата tracker-строк (top-level vs maps[].steam_id) + TTL 5s→30s |
| `60ef38a` | dual-id namespace fix в `_picks_to_heroes` (Hoodwink↔Pangolier коллизия) + паблишер 30s→5s |
| `d184aac` | `wait_for_function` вместо фиксированного `wait_for_timeout(3.5s)` — фетч 2-3s вместо 5-10s + `MATCH_STATE_TTL_SEC` 30s→8s |
| `8f70d7d` | TTL 8s→1h, alias-ключ `s{steam_id}` + `_dltv_series_id` из `/live/{id}.json` + оверлей time/networth даже когда пики пустые |

### DLTV-style лайв-карточка (v0.3.24g, h)

- **networth + game time** в реальном времени, извлекаются из `.team__networth > .networth > span` + `.info__duration[data-game-time]`
- **TM/ТБ формат** для kills / duration / towers — такой же, как в постматч-карточке
- **новая раскладка**: 3 колонки (team-side / score+time+gold / team-side), большие иконки героев 44×56 (картинка + имя) **под названием команды**, серия-скор и `Игра N` под центром, partial-gold block для случая "одна сторона известна"
- коллапс в одну колонку на узких экранах (≤700px)
- destroyed-tower counts в лайве **не реализованы** (DLTV отдаёт их только как иконки на мини-карте, без текста/JSON) — отложено до reverse-engineering socket.io payload

### Тесты

- 418 → 433 (+15): networth extraction, dual-id picks, partial-gold block, cache alias (steam+dltv ключи), wait_for_function timeout fallback, `towers_over_under` consistency
- Все существующие тесты проходят без изменений

### Что **не** вошло

- **Live destroyed-tower counts** — нет data-hook на DLTV-странице
- **SSE cookie auth** + **Postgres** + **direct socket.io** — запланировано в 0.4.0 (зафиксировано в TODO.md)
- **async_playwright** рефактор для починки chromium greenlet error — тоже 0.4.0
