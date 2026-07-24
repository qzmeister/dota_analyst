# Обновленный план сбора данных через Steam API

## 🔄 Новая стратегия

### Phase 1: Сбор League IDs из DLTV
```python
# Вместо сложного парсинга времени через Steam API:
# 1. Берем все лиги из DLTV discovery.py
# 2. Парсим match page HTML чтобы найти Steam IDs игр
# 3. Для каждого Steam ID fetch details
```

### Phase 2: Fetch Match Details через Steam API
```bash
# Каждый матч из discovery уже имеет:
- steam_id (match ID)
- event_slug (для определения лиги)
- start_time
- team_a, team_b

# Далее:
for match in discovery.all_matches():
    if not match.steam_id: continue
    
    # Скачиваем детали через Steam API (бесплатно!)
    details = get_match_details(match.steam_id)
    
    # Извлекаем features
    feat = parse_match_to_ml_features(details)
    samples.append(feat)
```

## ✅ Преимущества нового подхода

1. **Discovery уже знает все матчи** от DLTV scraper
2. **Steam API только enriches данными** - 1 call per match max
3. **Можно параллелить** без rate limit проблем
4. **Полные данные**: timeline, bans, picks, objectives

## 🚀 Реализация

### Новый скрипт: `scripts/enrich_with_steam_api.py`

```python
"""
Enrich discovered matches with full details from Steam API.

For each match that has a steam_id:
1. Call GetMatchDetails
2. Extract ML features
3. Save to ml_data/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.discovery import tracker
from backend.steam_enricher import (
    fetch_match_details,
    parse_to_ml_features,
    save_samples
)


def main():
    # 1. Discover all current matches from DLTV
    print("📡 Discovering matches from DLTV...")
    live, prematch = tracker.get_live_and_prematch()
    all_matches = live + prematch
    
    matches_with_ids = [m for m in all_matches if m.get("steam_id")]
    print(f"Found {len(matches_with_ids)} matches with Steam IDs\n")
    
    # 2. Enrich each match via Steam API
    samples = []
    seen_ids = set()
    
    for match in matches_with_ids:
        mid = match["steam_id"]
        if mid in seen_ids:
            continue
        
        print(f"[{len(seen_ids)+1}/{len(matches_with_ids)}] Fetching {mid}...")
        
        # Get full details from Steam
        details = fetch_match_details(mid)
        if not details:
            continue
        
        # Parse to features
        feat = parse_to_ml_features(details, match)
        if feat:
            samples.append(feat)
            seen_ids.add(mid)
        
        # Small delay to be polite
        time.sleep(0.2)
    
    # 3. Save dataset
    save_samples(samples)
    print(f"\n✅ Collected {len(samples)} complete match records!")


if __name__ == "__main__":
    main()
```

## 📊 Результат

| Source | Data Available | Rate Limit | Cost |
|--------|---------------|------------|------|
| **DLTV Discovery** | Match metadata | None | Free |
| **Steam API** | Full details | 3 req/min ⭐ | Free |
| **Stratz** | Lane analysis | 30 req/min | $0 or $50/mo |

### Итоговая архитектура:

```
discovery.py         → List of matches (metadata only)
   ↓
steam_api.py         → Enrichment (full game details)
   ↓
ml_trainer.py        → Feature extraction & training
   ↓
analysis.py          → Pre-match predictions using trained models
```

## 🔧 Следующие шаги

1. ✏️ Создать `backend/steam_enricher.py` с функциями:
   - `fetch_match_details(match_id)`
   - `parse_to_ml_features(details, metadata)`
   
2. ✏️ Обновить `scripts/enrich_with_steam_api.py` (новый файл)

3. 🚀 Запустить:
   ```bash
   python scripts/enrich_with_steam_api.py
   ```

4. 🎯 После сбора начать обучение:
   ```bash
   python backend/ml_trainer.py --train
   ```

---

**Итого:** Steam API используем ТОЛЬКО для enrichment существующих матчей из discovery, а не как primary source!
