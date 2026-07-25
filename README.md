# Dota Analyst - Draft Prediction System

Профессиональная система анализа драфтов и предсказания исходов матчей Dota 2. Фокусируется исключительно на профессиональных турнирных матчах (не pub games).

**Версия: 0.1.0** — три сервиса, Docker Compose, авторизация с первого дня.

## 🏗️ Архитектура

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│   web (nginx)  │───▶│ gateway (FAPI) │───▶│business (FAPI) │
│   :80 static   │    │  :8000 auth    │    │ :8000 heuristic│
└────────────────┘    └────────────────┘    └────────────────┘
       ▲                                          ▲
       └──── :80 ─── host                         └── private net
```

- **web** — nginx, отдаёт статику. **Ноль Python, ноль логики.**
- **gateway** — единственная публичная точка. Проверяет `X-API-Key`, проксирует в business.
- **business** — внутренний. Вся эвристика, ML, API-клиенты.

См. [ARCHITECTURE.md](ARCHITECTURE.md) для подробной диаграммы и обоснования.

## 🚀 Быстрый старт (Docker — рекомендуемый)

### 1. Подготовь `.env`

```bash
cp .env.example .env
# Открой .env и подставь свой DEV_API_KEY (сгенерируй: openssl rand -hex 32)
# Опционально: STEAM_API_KEY, STRAZT_API_KEY
```

### 2. Запусти стек

```bash
docker compose up --build
```

Через несколько секунд:
- **Фронт:** http://localhost
- **Gateway health:** http://localhost/healthz
- **Business health (через gateway):** `curl -H "X-API-Key: $DEV_API_KEY" http://localhost/api/healthz`

### 3. Только API (без Docker)

```bash
pip install -e ".[dev]"

# Только business
make run-business
# -> http://localhost:8000/api/healthz

# Бизнес + gateway (в двух терминалах)
make run-business   # :8000
make run-gateway    # :8000 (на другом порту в compose, или :8001 локально)
```

### 4. Тесты

```bash
make test           # pytest -v
make test-cov       # с coverage
make ci             # compile + test
```

## 📦 Что внутри

```
.
├── web/                  # nginx + static frontend
│   ├── public/           # index.html, app.js, style.css
│   ├── nginx.conf        # reverse-proxy to gateway
│   └── README.md
├── gateway/             # security + routing (FastAPI)
│   ├── app.py
│   ├── _middleware.py    # CORS, auth, body-size, access log, correlation id
│   └── _proxy.py         # httpx reverse-proxy to business
├── business/            # heuristic + ML (FastAPI)
│   ├── app.py            # /api/leagues, /api/board, /api/healthz
│   ├── analysis.py       # the 6-prediction heuristic
│   ├── dltv_client.py
│   ├── datdota_client.py
│   ├── discovery.py      # scraper + Steam
│   ├── board.py          # Kanban assembly
│   ├── _http.py          # shared retry + backoff
│   └── _logging.py       # shared JSON logger
├── docker/              # Dockerfile.web, Dockerfile.gateway, Dockerfile.business
├── docker-compose.yml    # 3 services, 2 networks
├── tests/                # 23 tests for analysis
├── ml_data/              # collected matches (gitignored)
├── pyproject.toml
├── Makefile              # make help
├── ARCHITECTURE.md
├── CHANGELOG.md
└── TODO.md
```

## 🔌 API (с версии 0.1.0 — требует `X-API-Key`)

### GET `/api/leagues`
```bash
curl -H "X-API-Key: $DEV_API_KEY" http://localhost/api/leagues
```

### GET `/api/board?events=1,2&watch=789`
```bash
curl -H "X-API-Key: $DEV_API_KEY" "http://localhost/api/board?events=1"
```

### GET `/api/healthz`
Не требует auth — для liveness/readiness проб.

## 📊 Сбор данных для ML обучения

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

## 🏗️ Архитектура проекта (0.2.0 — 3-сервисная + ML)

```
Dota_analyst/
├── business/                  # FastAPI service — internal only
│   ├── app.py                 # /api/board, /api/leagues, /api/healthz
│   ├── board.py               # Kanban assembly
│   ├── analysis.py            # Heuristic engine (winner, kills, towers, duration, f15, multikill)
│   ├── dltv_client.py         # DLTV v1 API client (thread-safe)
│   ├── discovery.py           # Match discovery (scraper + Steam)
│   ├── datdota_client.py      # DatDota API client (ML data)
│   ├── _http.py               # Shared HTTP retry + exp backoff
│   ├── _logging.py            # JSON / text logger
│   └── ml/                    # 0.2.0 — ML engine (Strategy pattern)
│       ├── features.py        # HeroWinRateEncoder, FEATURE_ORDER, extract_features
│       ├── storage.py         # ModelStorage (versioned joblib + metadata.json)
│       ├── engine.py          # IPredictionEngine, HeuristicEngine, MLEngine
│       └── train.py           # CLI: python -m business.ml.train
├── gateway/                   # FastAPI gateway — auth, CORS, body-size, proxy
│   ├── app.py
│   ├── _middleware.py
│   └── _proxy.py
├── web/                       # Static frontend (nginx)
│   ├── public/                # index.html, app.js, style.css
│   └── nginx.conf
├── ml_data/
│   ├── full_matches/          # 1111 DatDota match JSONs (training corpus)
│   ├── models/                # Trained artifacts (winner_v1/, ...)
│   └── imports/               # Ingest manifests
├── scripts/                   # One-off utilities
│   ├── collect_full_matches.py
│   ├── smoke_ml_0_2_0.py      # 0.2.0 — side-by-side engine comparison
│   └── ...
├── tests/                     # pytest suite (97 tests, 4 files)
├── docker/                    # 3 Dockerfiles (web, gateway, business)
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .env                       # API keys (never committed)
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

## 📈 ML Training Pipeline (0.2.0)

### 1. Сбор данных

```python
from business.datdota_client import collect_all_tier1_matches

# Собрать все матчи Tier 1 турниров
matches = collect_all_tier1_matches()
# -> 1,111 матчей в ml_data/full_matches/*.json
```

### 2. Обогащение деталями (уже сделано в `ml_data/full_matches/`)

`ml_data/imports/2026-07-24-telegram-desktop.json` — манифест импорта
1111 DatDota full match JSONs (по одному на матч, schema v1).

### 3. Обучение моделей (CLI)

```bash
# Defaults: data-dir=ml_data/full_matches, model-dir=ml_data/models, version=1
python -m business.ml.train

# Явные пути и версия
python -m business.ml.train \
    --data-dir ml_data/full_matches \
    --model-dir ml_data/models \
    --version 1

# Альтернативный estimator (вместо LogisticRegression)
python -m business.ml.train --model histgb
```

Что делает команда:
1. Загружает каждый `ml_data/full_matches/*.json`
2. Отбрасывает битые/errored и матчи без 5+5 hero picks
3. Обучает `HeroWinRateEncoder` (per-hero, per-side target encoding со сглаживанием)
4. Собирает матрицу признаков (N, 13) по `FEATURE_ORDER`
5. Делит 80/20 (stratified) и обучает `LogisticRegression`
6. Считает accuracy / log_loss / ROC AUC на hold-out
7. Сохраняет `ml_data/models/winner_v1/model.joblib` + `metadata.json`

### 4. Предсказание (внутри `business.app`)

```python
from business.ml.engine import make_engine, get_default_engine

# Прямое использование (без FastAPI)
engine = make_engine("ml", model_dir=Path("ml_data/models"))
result = engine.analyze(team_a, team_b, heroes_a, heroes_b)
# result["winner"] — перезаписан ML-моделью
# result["winner"]["source"] == "ml:v1"
# result["kills"], ["towers"], ["duration_min"] и т.д. — от эвристики
```

Внутри приложения engine выбирается через env:

```bash
PREDICTION_ENGINE=ml   uvicorn business.app:app
PREDICTION_ENGINE=heuristic   uvicorn business.app:app
```

`/api/board` теперь возвращает поле `engine` в ответе, чтобы клиент
видел, какая именно реализация сделала предсказание.

### 5. Замена предсказательной модели (CLI)

```bash
# 1. Обучить новую версию (v2)
python -m business.ml.train --version 2

# 2. В .env поменять активную версию (когда добавим pin, в 0.2.x)
#    Сейчас используется последняя версия по лексикографическому порядку.
PREDICTION_ENGINE=ml   uvicorn business.app:app
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
