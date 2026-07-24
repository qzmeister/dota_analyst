# RULES.md - Архитектура и правила разработки

## 🏗️ Архитектура проекта

### Высокоуровневая схема

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (Static)                       │
│  index.html + app.js + style.css                             │
│  - Kanban board UI                                           │
│  - Fetches /api/board                                        │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP
┌────────────────▼────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                          │
│  app.py → board.py → analysis.py                            │
│       ↓          ↓           ↓                              │
│  dltv_client  discovery  datdota_client                     │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
┌───▼───┐  ┌────▼────┐  ┌───▼────┐         │
│ DLTV  │  │  Steam  │  │DatDota │         │
│  v1   │  │   API   │  │  API   │         │
│  API  │  │         │  │        │         │
└───────┘  └─────────┘  └────────┘         │
```

### Слои архитектуры

#### 1. **Presentation Layer** (frontend/)
- **index.html** - структура страницы
- **app.js** - логика UI, fetch API, rendering
- **style.css** - стили

**Правила:**
- Только статические файлы (HTML/JS/CSS)
- Никакой бизнес-логики
- Все данные через `/api/*` endpoints

#### 2. **API Layer** (backend/app.py)
- FastAPI app
- Endpoints
- Request/response schemas
- CORS middleware

**Правила:**
- Минимальная логика (только routing)
- Делегировать в business layer
- Использовать type hints
- Документировать endpoints

#### 3. **Business Layer** (backend/board.py, analysis.py)
- Board assembly (Kanban cards)
- Draft analysis (heuristic engine)
- Prediction logic

**Правила:**
- Чистые функции (без side effects)
- Не зависеть от HTTP/DB
- Легко тестировать
- Документировать алгоритмы

#### 4. **Data Access Layer** (backend/*_client.py)
- API clients (DLTV, Steam, DatDota)
- HTTP requests
- Rate limiting
- Error handling

**Правила:**
- Инкапсулировать HTTP детали
- Rate limiting ОБЯЗАТЕЛЕН
- Retry logic для transient errors
- Кэширование где возможно

#### 5. **Data Layer** (ml_data/)
- JSON files с собранными данными
- Модели (pickle files)

**Правила:**
- Не коммитить в git (.gitignore)
- Документировать структуру данных
- Использовать версионирование

## 📊 Data Flow

### Live Match Discovery

```
1. User opens http://localhost:8000
2. Frontend fetches GET /api/board
3. app.py → board.build_board()
4. board.py → discovery.discover()
   ├─ Scraper: dltv.org/matches (HTML parsing)
   └─ Steam API: GetLiveLeagueGames
5. discovery → dltv_client.get_events()
6. board.py → analysis.analyze() для каждого матча
7. Return JSON → Frontend renders cards
```

### ML Data Collection

```
1. scripts/collect_datdota_targeted.py
2. datdota_client.get_leagues() → список лиг
3. datdota_client.get_league_matches(id) → матчи турнира
4. Save to ml_data/datdota_tier1_matches.json
5. (Optional) datdota_client.get_match_details(id) → полная статистика
6. ml_trainer.load_datdota_data() → загрузка
7. ml_trainer.train_*_model() → обучение
8. ml_trainer.save_models() → сохранение
```

### Post-Match Analysis

```
1. Postmatch card в UI
2. board._postmatch_prediction() → анализ
3. analysis.analyze() → предсказание
4. analysis.map_verdicts() → сравнение с фактом
5. Frontend показывает ✓/✗ для каждого предсказания
```

## 🎯 Ключевые модули

### backend/analysis.py

**Назначение:** Эвристический анализ драфта

**Вход:**
- `team_a`, `team_b` - нормализованные команды
- `heroes_a`, `heroes_b` - список героев

**Выход:**
```python
{
  "winner": {"team": "...", "probability": 65},
  "kills": {"total": 48, "radiant": 25, "dire": 23},
  "duration_min": 42.3,
  "total_over_under": {"side": "over", "threshold": 42},
  "towers": {"total": 18, "radiant": 11, "dire": 7},
  ...
}
```

**Алгоритм:**
1. Winner probability - logistic regression на win_rate, draft quality, rank
2. Total kills - базовое значение + корректировка на KDA героев
3. Duration - среднее avg_duration героев
4. Towers - функция от dominance (разница в win probability)
5. First to 15 - комбинация fb_rate, f10_rate, winner probability
6. Multikill - count героев с fight roles

**Правила модификации:**
- Не менять сигнатуру `analyze()`
- Возвращать тот же формат
- Документировать изменения в алгоритме
- Тестировать на исторических данных

### backend/board.py

**Назначение:** Сборка Kanban board из DLTV series

**Функции:**
- `leagues_with_status()` - лиги с статусами
- `build_board()` - собрать board
- `_prematch_card()` - карточка prematch
- `_live_card()` - карточка live
- `_postmatch_card()` - карточка postmatch

**Правила:**
- Использовать только из `app.py`
- Не делать HTTP requests напрямую
- Делегировать в `dltv_client`

### backend/dltv_client.py

**Назначение:** Клиент для DLTV v1 API

**Методы:**
- `get_events()` - список событий
- `get_series(event_id)` - series для события
- `hero_by_dltv_id(id)` - герой по DLTV ID
- `normalize_team(team)` - нормализовать команду

**Правила:**
- Singleton pattern (`client` instance)
- Rate limiting (если нужно)
- Кэширование hero metadata
- Error handling

### backend/datdota_client.py

**Назначение:** Клиент для DatDota API (ML данные)

**Методы:**
- `get_leagues()` - список лиг
- `get_league_matches(id)` - матчи лиги
- `get_match_details(id)` - полная статистика
- `collect_all_tier1_matches()` - сбор всех Tier 1

**Правила:**
- Rate limit: 3 секунды между запросами
- Daily limit: 500 запросов
- Retry logic для 429/5xx
- Кэширование где возможно

### backend/discovery.py

**Назначение:** Discovery live/prematch матчей

**Источники:**
1. DLTV scraper (dltv.org/matches HTML)
2. Steam API (GetLiveLeagueGames)

**Функции:**
- `discover()` - main entry point
- `_split_match_blocks()` - парсинг HTML
- `_http_json()` - HTTP requests

**Правила:**
- Приоритет: scraper > Steam API
- Кэшировать результаты (60 сек)
- Не блокировать main thread

## 🔧 Правила разработки

### 1. API Clients

**ОБЯЗАТЕЛЬНО:**
- ✅ Rate limiting (3s для DatDota)
- ✅ Retry logic (3 attempts, exponential backoff)
- ✅ Error handling (не падать на network errors)
- ✅ Timeout (10 сек default)
- ✅ User-Agent header

**Запрещено:**
- ❌ Бесконечные retries
- ❌ Игнорировать 429/5xx
- ❌ Делать requests без timeout
- ❌ Hardcode API keys

### 2. Environment Variables

**Использование:**
```python
# backend/__init__.py
from dotenv import load_dotenv
load_dotenv()

# В коде
import os
api_key = os.environ.get("STEAM_API_KEY")
```

**Правила:**
- ✅ Все секреты в `.env`
- ✅ `.env` в `.gitignore`
- ✅ Использовать `os.environ.get()` (не `os.environ[]`)
- ✅ Документировать переменные в README

### 3. Error Handling

**Паттерн:**
```python
try:
    data = api_client.get_data()
    if not data:
        logger.warning("No data returned")
        return fallback
    return data
except Exception as e:
    logger.error(f"API error: {e}")
    return fallback
```

**Правила:**
- ✅ Логировать ошибки
- ✅ Возвращать fallback (не None)
- ✅ Не падать на network errors
- ✅ Показывать user-friendly messages

### 4. Data Structures

**Match card:**
```python
{
  "stage": "prematch|live|postmatch",
  "series_id": int,
  "event_id": int,
  "event": str,
  "bo": "BO1|BO2|BO3|BO5",
  "team_a": {...},
  "team_b": {...},
  "prediction": {...}  # optional
}
```

**Team:**
```python
{
  "name": str,
  "logo": str,
  "tag": str,
  "rank": int,
  "win_rate": float,
  "fb_rate": float,
  "f10_rate": float
}
```

**Правила:**
- ✅ Использовать type hints
- ✅ Документировать поля
- ✅ Не менять формат без причины
- ✅ Backward compatibility

### 5. Testing

**Unit tests:**
```python
# tests/test_analysis.py
def test_analyze():
    team_a = {...}
    team_b = {...}
    heroes_a = [...]
    heroes_b = [...]
    
    result = analyze(team_a, team_b, heroes_a, heroes_b)
    
    assert "winner" in result
    assert "kills" in result
    assert 0 <= result["winner"]["probability"] <= 100
```

**Правила:**
- ✅ Писать tests для business logic
- ✅ Mock API clients
- ✅ Test edge cases (empty data, errors)
- ✅ Coverage > 70%

### 6. Performance

**Кэширование:**
```python
# Кэшировать hero metadata
_hero_cache = {}

def hero_by_dltv_id(hero_id):
    if hero_id in _hero_cache:
        return _hero_cache[hero_id]
    # ... fetch from API
    _hero_cache[hero_id] = hero
    return hero
```

**Правила:**
- ✅ Кэшировать API responses (где возможно)
- ✅ TTL для кэша (60 сек для live data)
- ✅ Не кэшировать user-specific data
- ✅ Очищать кэш при необходимости

### 7. Security

**API Keys:**
- ✅ Только в `.env`
- ✅ Не логировать
- ✅ Не показывать в UI
- ✅ Ротировать периодически

**Rate Limiting:**
- ✅ Соблюдать limits всех API
- ✅ Exponential backoff на 429
- ✅ Alert при приближении к limit

### 8. Code Style

**Форматирование:**
- ✅ 4 spaces indentation
- ✅ Max line length: 100 chars
- ✅ Docstrings для public functions
- ✅ Type hints

**Именование:**
- ✅ `snake_case` для variables/functions
- ✅ `PascalCase` для classes
- ✅ `UPPER_CASE` для constants
- ✅ Префикс `_` для private methods

## 📝 Git Workflow

### Branches

- `main` - production-ready
- `develop` - integration branch
- `feature/*` - новые фичи
- `fix/*` - багфиксы
- `hotfix/*` - срочные фиксы

### Commits

```
feat: добавить DatDota API client
docs: обновить README с инструкциями
refactor: упростить analysis.analyze()
test: добавить tests для board.py
chore: обновить requirements.txt
```

### Pull Requests

1. Create branch from `develop`
2. Make changes
3. Write tests
4. Update documentation
5. Submit PR
6. Code review
7. Merge to `develop`

## 🚀 Deployment

### Local Development

```bash
# 1. Clone repo
git clone https://github.com/yourusername/dota_analyst.git
cd dota_analyst

# 2. Create .env
cp .env.example .env
# Edit .env with your API keys

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run server
uvicorn backend.app:app --reload
```

### Production

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export STEAM_API_KEY=...

# 3. Run with gunicorn
gunicorn backend.app:app -w 4 -k uvicorn.workers.UvicornWorker

# 4. Nginx reverse proxy
server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

## 📚 Дополнительные ресурсы

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DatDota API Docs](https://api.datdota.com/swagger-ui/index.html)
- [Steam Web API](https://steamcommunity.com/dev/apidocs)

---

**Версия:** 1.0  
**Последнее обновление:** 2026-07-24
