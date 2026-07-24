# TODO - Future Features & Improvements

> Features to implement after ML pipeline is ready. Prioritize by impact on prediction accuracy.

---

## 🎯 Lane & Hero Matchup Analysis

### 1. First Tower Fall Timing
**Goal:** Analyze average time of first tower destruction per team.

**Data needed (from DatDota):**
- Building damage per player (already in match JSON)
- Timeline data (frames) to correlate with game time
- Item timings to estimate push power

**Output:**
```
Team Falcons: avg first tower fall = 18:30 (own) / 22:15 (enemy)
BoomBoys:     avg first tower fall = 16:45 (own) / 24:00 (enemy)
```

**ML feature:** `avg_first_tower_fall_time` per team → predicts early/mid/late game dominance.

---

### 2. Team Statistics Aggregator
**Goal:** Collect per-team stats across all collected matches.

**Metrics:**
- Avg kills per map
- Avg game duration
- Avg towers destroyed per game
- % of matches ending with megacreeps (ancient creeps spawning)
- Avg GPM/XPM per team
- Comeback rate (% of matches won after being behind at 10 min)
- First blood win rate

**Data source:** `ml_data/full_matches/*.json`

**Output:** `ml_data/team_stats.json`
```json
{
  "Team Falcons": {
    "matches": 45,
    "avg_kills": 32.5,
    "avg_duration_min": 38.2,
    "avg_towers_destroyed": 8.3,
    "megacreep_rate": 0.15,
    "comeback_rate": 0.22,
    "fb_win_rate": 0.68
  }
}
```

**Use:** Team profile for predictions, draft strategy analysis.

---

### 3. Mid Lane Matchup Comparison
**Goal:** Compare all mid heroes that appeared in matches. Who wins matchups and by how much.

**Metrics per mid hero:**
- Win rate (when picked mid)
- Avg networth difference at 10 min vs opponent mid
- Avg XP difference at 10 min vs opponent mid
- Avg GPM/XPM
- Avg hero damage
- Common items and timings

**Data extraction:**
```python
# From each match:
mid_player = find_player_by_lane(match, "MID")
opponent_mid = find_opposite_mid(match)

networth_diff = mid_player.gpm - opponent_mid.gpm  # proxy
xp_diff = mid_player.xpm - opponent_mid.xpm
```

**Output:** `ml_data/mid_matchups.json`
```json
{
  "Invoker": {
    "matches": 85,
    "win_rate": 0.58,
    "avg_nw_diff_at_10": 450,
    "avg_xp_diff_at_10": 320,
    "loses_to": ["Storm Spirit", "Ember Spirit"],
    "wins_against": ["Zeus", "Lina"]
  }
}
```

**ML feature:** `mid_matchup_advantage` for draft phase prediction.

---

### 4. Kill Distribution by Time Intervals
**Goal:** Track how many kills each team scores in 10-minute intervals.

**Intervals:**
- 0-10 min (laning phase)
- 10-20 min (mid game)
- 20-30 min (mid-late game)
- 30+ min (late game)

**Data extraction:**
```python
# From timeline data:
# radiant_networth_advantage at each time point
# Can infer kill timing from networth spikes

# Alternative: use player kill timestamps from items/abilities
```

**Output:** `ml_data/kill_intervals.json`
```json
{
  "Team Falcons": {
    "kills_0_10": 3.2,
    "kills_10_20": 8.5,
    "kills_20_30": 12.1,
    "kills_30_plus": 15.3
  }
}
```

**ML feature:** `team_kill_phase_preference` → predicts early/late game dominance.

---

## 🚀 Recommended Additions

### 5. Hero Pair Lane Dominance Score
**Goal:** Calculate lane pair strength using GPM difference vs direct opponent pair.

**Logic:**
```
Radiant safe lane: AM + Oracle (bottom)
vs
Dire offlane: Axe + ES (bottom)

pair_a_score = (AM.gpm + Oracle.gpm) × 10
pair_b_score = (Axe.gpm + ES.gpm) × 10
lane_diff = pair_a_score - pair_b_score  # > 0 = safe lane won
```

**Output:** `ml_data/hero_pair_stats.json`
```json
{
  "1_5_safe": {
    "heroes": [1, 5],
    "lane": "safe",
    "matches": 15,
    "avg_lane_diff": 1200,
    "win_rate": 0.73,
    "strong_vs": ["6_14_offlane"],
    "weak_vs": ["2_11_safe"]
  }
}
```

**ML feature:** `lane_pair_advantage` for draft phase.

---

### 6. Item Timing Analysis
**Goal:** Track when key items are purchased and correlate with win rate.

**Key items to track:**
- Boots (timing)
- Blink Dagger (timing)
- BKB (timing)
- First major item (timing)
- Aegis pickup (timing, if available)

**Output:** `ml_data/item_timings.json`
```json
{
  "hero_id_1": {
    "avg_boots_time": 380,
    "avg_blink_time": 920,
    "avg_bkb_time": 1450,
    "win_rate_with_early_blink": 0.65
  }
}
```

**ML feature:** `item_power_spike_time` → predicts mid-game dominance.

---

### 7. Gold/XP Graph Shape Analysis
**Goal:** Classify match tempo from gold/xp graph shape.

**Graph patterns:**
- **Snowball:** One team dominates from start (monotonic increase)
- **Comeback:** One team behind, then overtakes (V-shape)
- **Close:** Teams stay even throughout (flat line)
- **Late game:** Even until 30 min, then one team spikes

**Data:** `radiant_networth_advantage` time series (46 points)

**ML feature:** `graph_pattern` + `tempo_score` → predicts if match will be close or one-sided.

---

### 8. Patch Meta Analysis
**Goal:** Track how hero win rates and pick rates change across patches.

**Data:** `patch` field in each match JSON

**Output:** `ml_data/patch_meta.json`
```json
{
  "7.41": {
    "heroes": {
      "Invoker": {"win_rate": 0.58, "pick_rate": 0.12},
      "Storm Spirit": {"win_rate": 0.52, "pick_rate": 0.08}
    }
  }
}
```

**Use:** Adjust predictions based on current patch meta.

---

### 9. Tournament Tier Weighting
**Goal:** Weight training data by tournament tier (Tier 1 > Tier 2 > Tier 3).

**Logic:**
- Tier 1 matches: weight 1.0
- Tier 2 matches: weight 0.7
- Tier 3 matches: weight 0.4

**Use:** Model focuses more on high-quality data.

---

### 10. Player Form Tracking
**Goal:** Track individual player performance over recent matches.

**Metrics per player:**
- Recent KDA trend (last 10 matches)
- Recent GPM/XPM trend
- Hero pool diversity (how many different heroes played)
- Consistency score (variance in performance)

**Output:** `ml_data/player_form.json`

**ML feature:** `player_form_delta` → predicts if player is in good/bad form.

---

## 📋 Implementation Priority

| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| 🔴 High | Hero Pair Lane Dominance | High | Medium |
| 🔴 High | Team Statistics Aggregator | High | Low |
| 🟡 Medium | Mid Lane Matchup | Medium | Medium |
| 🟡 Medium | Kill Distribution by Time | Medium | Low |
| 🟡 Medium | First Tower Fall Timing | Medium | Medium |
| 🟢 Low | Item Timing Analysis | Medium | High |
| 🟢 Low | Gold/XP Graph Shape | Low | Medium |
| 🟢 Low | Patch Meta Analysis | Low | Low |
| 🟢 Low | Tournament Tier Weighting | Low | Low |
| 🟢 Low | Player Form Tracking | Medium | High |

---

## 📝 Notes

- All features require full match data from `ml_data/full_matches/`
- Start with **Team Stats** and **Kill Distribution** (easiest to implement)
- **Hero Pair Lane Dominance** is the most impactful for prediction accuracy
- Consider caching all analysis results in `ml_data/` as JSON files
- Use Pandas DataFrame for all analysis (see development practices)

---

**Last updated:** 2026-07-24  
**Status:** Waiting for full match collection to complete (1,111 matches)
