# Dota Analyst - Draft Prediction System

Профессиональная система анализа драфтов и предсказания исходов матчей Dota 2. Фокусируется исключительно на профессиональных турнирных матчах (не pub games).

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

**Зависимости:**
- `fastapi>=0.110` - Web framework
- `uvicorn[standard]>=0.27` - ASGI server
- `requests>=2.31` - HTTP client
- `python-dotenv>=1.0` - Environment variables

### 2. Настройка API ключей

Создайте файл `.env` в корне проекта:

```env
# Steam Web API Key (for GetLiveLeagueGames endpoint)
STEAM_API_KEY=your_steam_api_key_here

# Stratz API Token (for detailed draft/lanes analysis)
STRAZT_API_KEY=your_stratz_token_here
```

**Где получить:**
- **Steam API Key**: https://steamcommunity.com/dev/apikey (бесплатно)
- **Stratz API Token**: https://stratz.com/oauth (бесплатный tier: 30 req/min)

### 3. Запуск сервера

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Или через Python:

```bash
python -m uvicorn backend.app:app --reload
```

### 4. Открыть в браузере

```
http://localhost:8000
```

## 📊 Сбор данных для ML обучения

### DatDota API (рекомендуется)

**Преимущества:**
- ✅ Бесплатно: 500 запросов/день
- ✅ Быстро: 3 секунды между запросами
- ✅ Полные данные: все матчи турнира сразу
- ✅ Профессиональные: только турнирные матчи

**Сбор Tier 1 турниров:**

```bash
python scripts/collect_datdota_targeted.py
```

**Результат:** `ml_data/datdota_tier1_matches.json` (~1,111 матчей)

**Обогащение полными деталями (draft, items, stats):**

```python
from backend.datdota_client import get_match_details

# Получить полную статистику матча
details = get_match_details(match_id=8885183102)
# Включает: K/D/A, GPM/XPM, hero picks, items, timeline, map control
```

### Альтернативные источники

**Steam API** (для live match discovery):
```bash
python scripts/bulk_download_steam.py
```

**DLTV Scraper** (для текущих матчей):
```bash
python scripts/collect_manual_tier1.py
```

## 🏗️ Архитектура проекта

```
Dota_analyst/
├── backend/                    # FastAPI backend
│   ├── app.py                 # FastAPI app, endpoints
│   ├── board.py               # Board assembly (Kanban cards)
│   ├── analysis.py            # Draft analysis engine (heuristic)
│   ├── dltv_client.py         # DLTV v1 API client
│   ├── discovery.py           # Match discovery (scraper + Steam)
│   ├── datdota_client.py      # DatDota API client (ML data)
│   ├── stratz_api.py          # Stratz API client (lane data)
│   └── ml_trainer.py          # ML training pipeline
├── frontend/                   # Static frontend
│   ├── index.html
│   ├── app.js
│   └── style.css
├── ml_data/                    # Collected match data
│   ├── datdota_tier1_matches.json
│   └── all_matches_steam.json
├── scripts/                    # Data collection scripts
│   ├── collect_datdota_targeted.py
│   ├── find_league_ids.py
│   └── bulk_download_steam.py
├── .env                        # API keys (не коммитить!)
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

### GET `/api/leagues`

Вернуть список лиг с статусами (live/upcoming/finished).

**Response:**
```json
{
  "leagues": [
    {
      "id": 123,
      "title": "Esports World Cup 2026",
      "status": "live",
      "is_active": true
    }
  ]
}
```

### GET `/api/board`

Вернуть Kanban board с prematch/live/postmatch карточками.

**Parameters:**
- `events` (List[str]) - event IDs (можно несколько через запятую)
- `watch` (List[str]) - steam match IDs для watchlist

**Response:**
```json
{
  "prematch": [...],
  "live": [...],
  "postmatch": [...],
  "selected": [123, 456],
  "watch": [789]
}
```

### GET `/`

Вернуть frontend (Kanban board UI).

## 📈 ML Training Pipeline

### 1. Сбор данных

```python
from backend.datdota_client import collect_all_tier1_matches

# Собрать все матчи Tier 1 турниров
matches = collect_all_tier1_matches()
# -> 1,111 матчей с базовой информацией
```

### 2. Обогащение деталями

```python
from backend.datdota_client import get_match_details

for match in matches:
    details = get_match_details(match['matchId'])
    # Добавить: player stats, hero picks, items, timeline
```

### 3. Обучение моделей

```python
from backend.ml_trainer import MLTrainer

trainer = MLTrainer(data_dir="ml_data")

# Загрузить данные
trainer.load_datdota_data("ml_data/datdota_tier1_matches.json")

# Обучить модели
trainer.train_winner_model()      # Random Forest Classifier
trainer.train_duration_model()    # Random Forest Regressor
trainer.train_kills_model()       # Random Forest Regressor

# Сохранить модели
trainer.save_models()
```

### 4. Предсказание

```python
# Использовать обученные модели для предсказания
prediction = trainer.predict(match_features)
```

## 🎯 Система предсказаний

Текущая система использует **эвристический анализ** (analysis.py):

### 6 предсказаний:

1. **Winner probability** - вероятность победителя (Radiant/Dire)
2. **Total kills** - общее количество убийств
3. **Duration** - длительность матча (в минутах)
4. **Towers destroyed** - количество разрушенных башен
5. **First to 15 kills** - кто первым наберет 15 убийств
6. **Multikill potential** - потенциал к мультикиллам (High/Medium/Low)

### Over/Under ставки

Для duration и kills система генерирует **Over/Under** ставки:

```python
# Duration prediction
{
  "total_over_under": {
    "side": "over",        # "over" или "under"
    "threshold": 42,       # пороговое значение (минуты)
    "formatted": "42:00"   # человекочитаемый формат
  }
}

# Kills prediction
{
  "kills_total_over_under": {
    "side": "under",
    "threshold": 48
  }
}
```

## 📊 Источники данных

### DatDota API (основной для ML)

- **Base URL**: `https://api.datdota.com`
- **Rate limit**: 3 секунды между запросами
- **Daily limit**: 500 запросов/день (без ключа)
- **Данные**: турниры, матчи, player stats, drafts, timeline

**Endpoints:**
- `/api/leagues` - список лиг
- `/api/leagues/{id}` - лига с матчами
- `/api/matches/{id}` - полная статистика матча

### Steam API (для live discovery)

- **Base URL**: `https://api.steampowered.com`
- **Rate limit**: 100,000 calls/day
- **Данные**: live matches, match history

**Endpoints:**
- `GetLiveLeagueGames` - текущие live матчи
- `GetMatchDetails` - детали матча

### DLTV v1 API (для scraper)

- **Base URL**: `https://dltv.org/api/v1`
- **Данные**: events, series, heroes, teams
- **Используется**: для discovery и текущих матчей

### Stratz API (для lane analysis)

- **Base URL**: `https://api.stratz.com`
- **Rate limit**: 30 req/min (бесплатный tier)
- **Данные**: lane assignments, FB/F10 rates

## 🔧 Разработка

### Добавление нового endpoint

```python
# backend/app.py

@app.get("/api/new-endpoint")
def new_endpoint():
    return {"data": "..."}
```

### Изменение analysis logic

```python
# backend/analysis.py

def analyze(team_a, team_b, heroes_a, heroes_b):
    # Модифицировать логику предсказания
    ...
```

### Добавление нового API клиента

```python
# backend/new_api_client.py

import requests

BASE_URL = "https://api.example.com"

def get_data(param):
    resp = requests.get(f"{BASE_URL}/endpoint", params={"param": param})
    return resp.json()
```

## 📝 Лицензия

MIT

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

Для вопросов и поддержки создайте Issue в GitHub repository.
