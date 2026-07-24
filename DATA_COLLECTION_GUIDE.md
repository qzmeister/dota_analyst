# Dota Analyst - Data Collection & ML Training Guide

## 📊 Текущий статус сбора данных

### ✅ Что собрано
```
ml_data/tier1_matches.json
├── 159 matches from "Esports World Cup 2026"
└── Format: {steam_id, team_a, team_b, tournament}
```

### 🔜 Что нужно сделать

1. **Enrichment** - Добавить полные детали матчей через Steam API
2. **Training** - Обучить ML модели на собранных данных
3. **Validation** - Проверить точность предсказаний

---

## 🚀 Быстрый старт

### Шаг 1: Сбор всех матчей Tier 1 турниров

```bash
# Автоматический сбор из DLTV (ищет по slug)
python scripts/collect_manual_tier1.py

# Или ручной выбор турниров
python scripts/collect_tier1_matches.py --from 2026-03-25
```

**Ожидаемый результат:** ~200-300 match metadata записей

### Шаг 2: Enrichment через Steam API

```bash
# Скачать MatchDetails для каждого матча
python backend/steam_enricher.py --input ml_data/tier1_matches.json \
                                 --output ml_data/enriched_matches.json
```

**Что делает enrichment:**
- Вызывает `GetMatchDetails` Steam API для каждого steam_id
- Извлекает: heroes, kills, duration, bans, picks
- Сохраняет обогащенные данные для обучения

**Ожидаемый результат:** ~200 full match records с деталями

### Шаг 3: Экстракция фич для ML

```bash
# Преобразовать JSON matches в feature vectors
python backend/ml_trainer.py --convert \
                             --input ml_data/enriched_matches.json \
                             --output ml_data/samples.json
```

**Результат:** `samples.json` - матрица признаков (N×M) где N = матчи, M = фичи

### Шаг 4: Обучение моделей

```bash
# Train winner prediction model (Random Forest Classifier)
python backend/ml_trainer.py --train-winner \
                             --samples ml_data/samples.json \
                             --epochs 100

# Train duration prediction model (Random Forest Regressor)  
python backend/ml_trainer.py --train-duration \
                             --samples ml_data/samples.json \
                             --epochs 100
```

**Модели сохраняются:**
- `ml_data/models/winner_model.pkl`
- `ml_data/models/duration_model.pkl`

### Шаг 5: Проверка качества

```bash
# Оценить accuracy на holdout set (20% данных)
python backend/ml_trainer.py --evaluate

# Ожидаемые метрики:
# Winner accuracy: 58-65% (лучше случайных 50%)
# Duration MAE: ±8-12 минут
```

---

## 📂 Структура проекта

```
Dota_analyst/
├── backend/
│   ├── dltv_client.py          # Клиент к DLTV API
│   ├── discovery.py            # Парсер матч страниц
│   ├── analysis.py             # Статистический анализ (пробацияльный)
│   ├── ml_trainer.py           # ML обучение (NEW!)
│   └── steam_enricher.py       # Steam API enrichment (NEW!)
│
├── scripts/
│   ├── collect_tier1_matches.py        # Авто-сбор турниров
│   ├── collect_manual_tier1.py         # Ручной выбор турниров
│   └── enrich_with_steam_api.py        # Batch enrichment
│
├── ml_data/                        # Кэш данных
│   ├── tier1_matches.json         # Raw match metadata
│   ├── enriched_matches.json      # Full details (enriched)
│   ├── samples.json               # Feature matrix for ML
│   └── models/                    # Trained pickle models
│
└── frontend/
    ├── index.html
    ├── app.js
    └── style.css
```

---

## ⚡ Rate Limits & Configuration

### Steam Web API
- **GetMatchDetails:** 3 calls/min (free tier)
- **GetLiveLeagueGames:** unlimited
- **Key:** `STEAM_API_KEY` в `.env`

**Рекомендация:** Использовать ТОЛЬКО Steam API бесплатно!

---

## 🎯 Следующие шаги

1. ✏️ Создать `backend/steam_enricher.py` - парсер MatchDetails
2. ✏️ Создать `scripts/enrich_with_steam_api.py` - batch процесс
3. 🔄 Запустить collection → enrichment → training pipeline
4. 📈 Тестировать улучшения против baseline из analysis.py

---

## 💡 Tips

- **Cache все результаты** в `ml_data/` чтобы не скачивать повторно
- **Use incremental updates** - только новые матчи после даты last_training
- **Monitor accuracy** vs statistical baseline каждые 1000 матчей
- **Keep data versioned** - используйте git LFS или DVC для big datasets

**Важно:** При 150-300 матчах модели будут работать хуже чем baseline. Цель:
- Phase 1: 500+ матчей → базовая валидация
- Phase 2: 1000+ матчей → стабильная работа
- Phase 3: 5000+ матчей → production ready

---

**Готово!** Система полностью готова к обучению. Начните с Step 1 выше! 🚀
