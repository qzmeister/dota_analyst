# Dota Analyst - Bulk Download Script

Download historical match data from Stratz API for ML training.

## ⚡ Quick Start

```bash
# Basic usage (default: last 3 months of matches)
python scripts/bulk_download.py

# Custom date range
python scripts/bulk_download.py --from 2026-03-25

# Specific teams only
python scripts/bulk_download.py --teams "123,456,789"

# Increase parallelism (warning: may hit rate limits)
python scripts/bulk_download.py --workers 5
```

## 📊 Rate Limits

**Stratz API Free Tier:**
- 30 requests/minute (0.5 req/sec)
- We use conservative 0.4s delay = safe!

**Estimated times:**
- 50 teams × 200 matches × 0.4s = ~6.7 hours (serial)
- With 3 workers: ~2-3 hours recommended
- With 5 workers: ~1.5 hours (riskier)

## 💾 Output Files

```
ml_data/
├── teams/
│   ├── team_58514931.json     # Per-team cache
│   └── team_123456.json
└── all_matches.json           # Combined dataset (~5MB for 5K matches)
```

## 🔧 Configuration

| Parameter | Default | Max Safe | Description |
|-----------|---------|----------|-------------|
| `--per-team` | 200 | 200 | Max matches per team |
| `--workers` | 3 | 5 | Parallel threads |
| `--from` | 2026-03-25 | any | Start date |

## ⚠️ Important Notes

1. **Don't increase `--workers` above 5** - you'll hit rate limits faster than backoff can recover
2. **Keep running overnight** - 10K+ matches take time but worth it
3. **Check progress** in console output (shows cached vs fresh fetches)
4. **Interrupt safe** - partial data is saved incrementally, resume anytime

## 🔄 Resume Downloads

Script automatically:
- ✅ Loads from cache if exists
- ✅ Only downloads new matches
- ✅ Deduplicates by match ID

Run again anytime to get latest data!

## 📈 Next Steps

After collection complete:

```bash
# Extract features for ML training
python backend/ml_trainer.py --convert

# Train models  
python backend/ml_trainer.py --train

# Check accuracy metrics
cat ml_data/training_log.jsonl
```
