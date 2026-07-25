"""
ML Training Pipeline — Phase 1
Build training DataFrame from 1,111 matches + feature JSONs.
Train XGBoost model for match outcome prediction.

Features (~80):
  - Hero picks (10)
  - Team stats (12)
  - Player form (18)
  - Mid matchup (4)
  - Timeline NW advantage (10)
  - Laning stats (6)
  - Tournament tier weight (1)
  - Graph pattern (3)
  - Meta hero win rates (10)
  - Draft diversity (2)
  - Duration (1)

Target: radiant_victory (binary)
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Paths
MATCHES_DIR = Path("ml_data/full_matches")
ML_DATA = Path("ml_data")
MODEL_DIR = Path("ml_models")
MODEL_DIR.mkdir(exist_ok=True)


# ─── Load pre-computed feature files ────────────────────────────────
def load_json(name: str) -> dict:
    path = ML_DATA / name
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


print("Loading feature files...")
team_stats = load_json("team_stats.json")
mid_matchups = load_json("mid_matchups.json")
hero_pair_stats = load_json("hero_pair_stats.json")
player_form = load_json("player_form.json")
patch_meta = load_json("patch_meta.json")
tournament_tiers = load_json("tournament_tiers.json")
team_laning = load_json("team_laning_profiles.json")
graph_patterns_data = load_json("graph_patterns.json")

# Build graph pattern lookup by match_id
graph_pattern_lookup = {}
if graph_patterns_data and 'matches' in graph_patterns_data:
    for gp in graph_patterns_data['matches']:
        graph_pattern_lookup[gp['match_id']] = gp

# Build tournament tier lookup
tier_lookup = {}
if tournament_tiers and 'leagues' in tournament_tiers:
    for league_name, data in tournament_tiers['leagues'].items():
        tier_lookup[league_name] = data.get('weight', 0.4)


# ─── Helper functions ───────────────────────────────────────────────
def get_team_stat(team_name: str, stat: str, default=0.0) -> float:
    """Look up team stat, fuzzy match."""
    if not team_name or not team_stats:
        return default
    # Exact match
    if team_name in team_stats:
        return team_stats[team_name].get(stat, default)
    # Fuzzy: lowercase, no spaces
    norm = team_name.lower().replace(' ', '')
    for t in team_stats:
        if t.lower().replace(' ', '') == norm:
            return team_stats[t].get(stat, default)
    return default


def get_player_form_stat(player_name: str, stat: str, default=0.0) -> float:
    """Look up player form stat."""
    if not player_name or not player_form:
        return default
    if player_name in player_form:
        return player_form[player_name].get(stat, default)
    return default


def get_mid_hero_wr(hero_name: str) -> float:
    """Look up mid hero win rate."""
    if not hero_name or not mid_matchups:
        return 0.5
    if hero_name in mid_matchups:
        return mid_matchups[hero_name].get('win_rate', 0.5)
    return 0.5


def get_hero_patch_wr(hero_name: str, patch: str) -> float:
    """Look up hero win rate in specific patch."""
    if not hero_name or not patch_meta:
        return 0.5
    patch_data = patch_meta.get(patch, {})
    heroes = patch_data.get('heroes', {})
    if hero_name in heroes:
        return heroes[hero_name].get('win_rate', 50.0) / 100.0
    return 0.5


def get_laning_stat(team_name: str, stat: str, default=0.0) -> float:
    """Look up team laning stat."""
    if not team_name or not team_laning:
        return default
    # team_laning has team profiles
    profiles = team_laning if isinstance(team_laning, dict) else {}
    if team_name in profiles:
        return profiles[team_name].get(stat, default)
    norm = team_name.lower().replace(' ', '')
    for t in profiles:
        if t.lower().replace(' ', '') == norm:
            return profiles[t].get(stat, default)
    return default


def get_nw_at_time(frames: dict, target_sec: int) -> float:
    """Get radiant NW advantage at specific time."""
    if not frames:
        return 0.0
    times = frames.get('times', [])
    nw = frames.get('radiant_networth_advantage', [])
    if not times or not nw:
        return 0.0
    
    # Find closest time point
    best_idx = 0
    best_diff = abs(times[0] - target_sec)
    for i, t in enumerate(times):
        diff = abs(t - target_sec)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    
    return nw[best_idx] if best_idx < len(nw) else 0.0


# ─── Build DataFrame ────────────────────────────────────────────────
def build_training_data():
    """Build training DataFrame from all matches."""
    matches = []
    for file in sorted(MATCHES_DIR.glob("*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            matches.append(json.load(f))
    
    print(f"Loaded {len(matches)} matches")
    
    rows = []
    for i, match in enumerate(matches, 1):
        if i % 200 == 0:
            print(f"  Building features {i}/{len(matches)}")
        
        try:
            row = build_match_features(match)
            if row:
                rows.append(row)
        except Exception as e:
            pass  # Skip problematic matches silently
    
    df = pd.DataFrame(rows)
    print(f"Built DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_match_features(match: dict) -> dict:
    """Extract all features for a single match."""
    row = {}
    
    # Target
    row['target'] = 1 if match['radiant_victory'] else 0
    row['match_id'] = match['match_id']
    row['duration_min'] = match['duration'] / 60.0
    
    patch = match.get('patch', 'unknown')
    league_name = match.get('league', {}).get('name', '')
    
    radiant_team = match['radiant']['team']['name']
    dire_team = match['dire']['team']['name']
    r_players = match['radiant']['player_performances']
    d_players = match['dire']['player_performances']
    
    # ── Hero picks (10 features) ──
    r_heroes = sorted([p['performance']['hero']['valve_id'] for p in r_players])
    d_heroes = sorted([p['performance']['hero']['valve_id'] for p in d_players])
    for j, h in enumerate(r_heroes):
        row[f'r_hero_{j}'] = h
    for j, h in enumerate(d_heroes):
        row[f'd_hero_{j}'] = h
    
    # ── Hero meta win rates (10 features) ──
    r_hero_wrs = [get_hero_patch_wr(p['performance']['hero']['short_name'], patch) for p in r_players]
    d_hero_wrs = [get_hero_patch_wr(p['performance']['hero']['short_name'], patch) for p in d_players]
    row['r_avg_hero_wr'] = np.mean(r_hero_wrs)
    row['d_avg_hero_wr'] = np.mean(d_hero_wrs)
    row['hero_wr_diff'] = row['r_avg_hero_wr'] - row['d_avg_hero_wr']
    
    # ── Team stats (12 features) ──
    for prefix, team_name in [('r', radiant_team), ('d', dire_team)]:
        row[f'{prefix}_team_wr'] = get_team_stat(team_name, 'win_rate')
        row[f'{prefix}_team_avg_kills'] = get_team_stat(team_name, 'avg_kills')
        row[f'{prefix}_team_avg_gpm'] = get_team_stat(team_name, 'avg_gpm')
        row[f'{prefix}_team_avg_xpm'] = get_team_stat(team_name, 'avg_xpm')
        row[f'{prefix}_team_avg_duration'] = get_team_stat(team_name, 'avg_duration_min')
        row[f'{prefix}_team_nw_adv'] = get_team_stat(team_name, 'avg_nw_advantage')
    
    row['team_wr_diff'] = row['r_team_wr'] - row['d_team_wr']
    row['team_gpm_diff'] = row['r_team_avg_gpm'] - row['d_team_avg_gpm']
    
    # ── Player form (18 features: 3 key players per team × 6 stats) ──
    # Sort players by GPM to identify carry/mid/support
    r_by_gpm = sorted(r_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)
    d_by_gpm = sorted(d_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)
    
    for prefix, players in [('r', r_by_gpm[:3]), ('d', d_by_gpm[:3])]:
        for j, p in enumerate(players):
            pname = p['player']['nickname']
            row[f'{prefix}_p{j}_wr'] = get_player_form_stat(pname, 'recent_win_rate', 0.5)
            row[f'{prefix}_p{j}_gpm'] = get_player_form_stat(pname, 'recent_avg_gpm', 500)
            row[f'{prefix}_p{j}_kda'] = get_player_form_stat(pname, 'recent_avg_kda', 0)
            row[f'{prefix}_p{j}_form_delta'] = get_player_form_stat(pname, 'form_delta.win_rate_delta', 0)
    
    # Player form diffs (carry vs carry, mid vs mid, support vs support)
    for j in range(3):
        row[f'p{j}_wr_diff'] = row.get(f'r_p{j}_wr', 0.5) - row.get(f'd_p{j}_wr', 0.5)
        row[f'p{j}_gpm_diff'] = row.get(f'r_p{j}_gpm', 500) - row.get(f'd_p{j}_gpm', 500)
    
    # ── Mid matchup (4 features) ──
    r_mid = next((p for p in r_players if p.get('laneInfo', {}).get('lane') == 'MIDDLE'), None)
    d_mid = next((p for p in d_players if p.get('laneInfo', {}).get('lane') == 'MIDDLE'), None)
    
    if r_mid and d_mid:
        r_mid_hero = r_mid['performance']['hero']['short_name']
        d_mid_hero = d_mid['performance']['hero']['short_name']
        row['r_mid_wr'] = get_mid_hero_wr(r_mid_hero)
        row['d_mid_wr'] = get_mid_hero_wr(d_mid_hero)
        row['mid_wr_diff'] = row['r_mid_wr'] - row['d_mid_wr']
        row['mid_gpm_diff'] = (r_mid['performance']['gpm'] or 0) - (d_mid['performance']['gpm'] or 0)
    else:
        row['r_mid_wr'] = row['d_mid_wr'] = row['mid_wr_diff'] = row['mid_gpm_diff'] = 0
    
    # ── Timeline NW advantage (10 features) ──
    frames = match.get('frames')
    time_points = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]  # 5,10,15,...50 min
    for t in time_points:
        row[f'nw_at_{t//60}min'] = get_nw_at_time(frames, t)
    
    # NW trend (late - early)
    early_nw = get_nw_at_time(frames, 600)
    late_nw = get_nw_at_time(frames, min(match['duration'], 2700))
    row['nw_trend'] = late_nw - early_nw
    
    # ── Laning stats (6 features) ──
    for prefix, team_name in [('r', radiant_team), ('d', dire_team)]:
        row[f'{prefix}_lanes_won'] = get_laning_stat(team_name, 'lanes_won_pct', 50)
        row[f'{prefix}_nw_adv_laning'] = get_laning_stat(team_name, 'nw_advantage', 0)
        row[f'{prefix}_fb_pct'] = get_laning_stat(team_name, 'fb_pct', 50)
    
    # ── Tournament tier (1 feature) ──
    row['tier_weight'] = tier_lookup.get(league_name, 0.4)
    
    # ── Graph pattern (3 features) ──
    gp = graph_pattern_lookup.get(match['match_id'], {})
    pattern = gp.get('pattern', 'unknown')
    row['pattern_snowball'] = 1 if pattern == 'snowball' else 0
    row['pattern_late_game'] = 1 if pattern == 'late_game' else 0
    row['pattern_close'] = 1 if pattern == 'close' else 0
    
    # ── Draft diversity (2 features) ──
    row['r_hero_diversity'] = len(set(r_heroes)) / 5.0
    row['d_hero_diversity'] = len(set(d_heroes)) / 5.0
    
    # ── Actual match performance (for reference, not used as features) ──
    row['r_total_kills'] = sum(p['performance']['kills'] or 0 for p in r_players)
    row['d_total_kills'] = sum(p['performance']['kills'] or 0 for p in d_players)
    row['r_avg_gpm_actual'] = np.mean([p['performance']['gpm'] or 0 for p in r_players])
    row['d_avg_gpm_actual'] = np.mean([p['performance']['gpm'] or 0 for p in d_players])
    
    return row


# ─── Main training pipeline ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("ML Training Pipeline — Phase 1")
    print("=" * 60)
    
    # Build training data
    df = build_training_data()
    
    # Define feature columns (exclude target, match_id, and reference columns)
    exclude_cols = {'target', 'match_id', 'duration_min',
                    'r_total_kills', 'd_total_kills', 
                    'r_avg_gpm_actual', 'd_avg_gpm_actual'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution: {np.sum(y==1)} Radiant wins, {np.sum(y==0)} Dire wins")
    print(f"Radiant win rate: {np.mean(y):.3f}")
    
    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    if nan_count > 0 or inf_count > 0:
        print(f"\nWarning: {nan_count} NaN, {inf_count} Inf values — filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ─── Train with cross-validation ───
    print("\n" + "-" * 40)
    print("Training XGBoost with 5-fold CV...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print(f"\n  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  CV ROC AUC:  {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")
    print(f"  Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # ─── Train final model on all data ───
    print("\n" + "-" * 40)
    print("Training final model on all data...")
    model.fit(X, y)
    
    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 20 features by importance:")
    for name, imp in feat_imp[:20]:
        bar = "#" * int(imp * 100)
        print(f"    {name:30s} {imp:.4f} {bar}")
    
    # ─── Save model and metadata ───
    model_path = MODEL_DIR / "xgb_phase1.json"
    model.save_model(str(model_path))
    
    metadata = {
        'phase': 1,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'cv_accuracy': float(cv_scores.mean()),
        'cv_roc_auc': float(cv_roc.mean()),
        'radiant_win_rate': float(np.mean(y)),
        'top_features': [(name, float(imp)) for name, imp in feat_imp[:30]]
    }
    
    meta_path = MODEL_DIR / "model_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature columns for inference
    cols_path = MODEL_DIR / "feature_cols.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    print(f"\n  Model saved to {model_path}")
    print(f"  Metadata saved to {meta_path}")
    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
