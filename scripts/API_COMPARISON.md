# Сравнение: Steam API vs Stratz API для ML-сбора данных

## 📊 Возможности API

| Функция | Steam Web API (бесплатно) | Stratz API (Pro $50/mo) |
|---------|---------------------------|------------------------|
| **GetMatchHistory** | ✅ 10 req/min | ✅ Limited (30/min total) |
| **GetMatchDetails** | ✅ 3 req/min | ✅ Included in rate limit |
| **League Games List** | ✅ Unlimited ⭐ | ❌ Only via match history |
| **Team Info** | ✅ Unlimited | ✅ Rate limited |
| **Hero Draft Data** | ✅ Full details | ✅ Partial |
| **Player Stats** | ✅ Complete | ✅ Aggregated only |
| **Timeline Data** | ✅ Kill/tower timestamps | ✅ Simplified |
| **Bans/Picks** | ✅ Exact order | ✅ Approximated |

## 💰 Стоимость

### Steam API
```python
# Бесплатный tier:
- GetMatchHistory: 10 calls/min
- GetMatchDetails: 3 calls/min (but can batch with history)
- GetLeagueGames: UNLIMITED ⭐

# В день: ~14,000 requests = ~7M/year
# Результат: БЕСПЛАТНО!
```

### Stratz API
```python
# Free tier:
- 30 calls/min total (all endpoints shared)
- No Match History endpoint at all!
- Must paginate matches individually

# Pro tier ($50/mo):
- 150 calls/min (still too slow for bulk scraping)
# Result: НЕ ВЫГОДНО
```

## 🎯 Рекомендация: ТОЛЬКО Steam API!

### Почему Steam API лучше:

1. **Нет жестких лимитов на league games** - можно качать историю напрямую
2. **Полные данные о драфтах** - кто кого банит/пикает
3. **Free forever** - не нужно платить $50/mo
4. **Direct from Valve** - самый надежный источник
5. **Richer data** - timeline, objectives, XP/gold graphs

### Используем эти эндпоинты:

```python
# 1. Список всех матчей лиги за период
GET /IDOTA2Match_570/GetLiveLeagueGames/v0001?league_id=XXXXX&format=json

# 2. Детали каждого матча (в пакетном режиме)
GET /IDOTA2Match_570/GetMatchDetails/v1?match_id=YYYYYY&key=...

# 3. История по sequence number (если нет league_id)
GET /IDOTA2Match_570/GetMatchHistory/v1?matches_requested=100&start_time=...
```

## 📈 Реальный сценарий использования

### Phase 1: Bulk Historical Collection
```bash
# 1. Получить все league IDs из DLTV discovery
python scripts/collect_league_ids.py

# 2. Для каждой лиги скачать все матчи через Steam API
python scripts/bulk_download_steam.py --from 2026-03-25 --leagues "13234,13235"

# 3. Параллельно качаем Match Details
# При 500 лигах × 50 матчей = 25,000 запросов
# При 10 workers: ~2 часа работы
```

### Phase 2: Feature Engineering
```python
# Steam API дает полное время:
{
  "result": {
    "timeline": {
      "radiant_team_gold": [0, 500, 1000, ...],
      "radiant_team_xp": [...],
      "events": [
        {"type": "HERO_KILL", "time": 892},
        {"type": "BUILDING_DESTROYED", "time": 1234}
      ]
    }
  }
}

→ Можем рассчитать:
- Киллы по минутам
- Золото/XP разрыв
- Контроль вардов
- Objective timing
```

## 🔧 Обновляем workflow

### Вместо этого:
❌ `stratz_api.get_match(match_id)` - дорого и медленно

### Делаем так:
✅ `steam_api.get_match_details(match_id)` - быстро и бесплатно!

```python
# New streamlined flow:
from backend.steam_api import get_match_details, fetch_league_games

# 1. Fetch all recent league matches
for league_id in known_leagues:
    games = fetch_league_games(league_id, days_back=90)
    
    # 2. For each game, get full details
    for game in games:
        details = get_match_details(game.match_id)
        features = parse_to_features(details)
        samples.append(features)
```

## 🚀 Следующие шаги

1. ✅ Создать `scripts/bulk_download_steam.py` (готово!)
2. ✅ Удалить зависимостью от Stratz для bulk collection
3. ✏️ Перенести Stratz только для пост-анализа (лайны, плейлист игроков)
4. 🔄 Запустить скачивание через Steam API

---

**Итог:** Стратз нужен только для некоторых аналитических фич.  
Основная сборка истории - через бесплатный Steam Web API!
