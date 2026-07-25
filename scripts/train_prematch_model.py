"""
Pre-Match ML Training Pipeline
Predict match outcome BEFORE the game starts (draft phase).

Features (~50):
  - Hero picks (10)
  - Hero meta win rates (3)
  - Team stats (12)
  - Player form (18)
  - Mid matchup hero win rates (3)
  - Laning stats (6)
  - Tournament tier (1)
  - Draft diversity (2)

Target: radiant_victory (binary)

NO in-game features (no NW advantage, no actual GPM, etc.)
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
player_form = load_json("player_form.json")
patch_meta = load_json("patch_meta.json")
tournament_tiers = load_json("tournament_tiers.json")
team_laning = load_json("team_laning_profiles.json")

# Build tournament tier lookup
tier_lookup = {}
if tournament_tiers and 'leagues' in tournament_tiers:
    for league_name, data in tournament_tiers['leagues'].items():
        tier_lookup[league_name] = data.get('weight', 0.4)


# ─── Helper functions ───────────────────────────────────────────────
def get_team_stat(team_name: str, stat: str, default=0.0) -> float:
    if not team_name or not team_stats:
        return default
    if team_name in team_stats:
        return team_stats[team_name].get(stat, default)
    norm = team_name.lower().replace(' ', '')
    for t in team_stats:
        if t.lower().replace(' ', '') == norm:
            return team_stats[t].get(stat, default)
    return default


def get_player_form_stat(player_name: str, stat: str, default=0.0) -> float:
    if not player_name or not player_form:
        return default
    if player_name in player_form:
        return player_form[player_name].get(stat, default)
    return default


def get_mid_hero_wr(hero_name: str) -> float:
    if not hero_name or not mid_matchups:
        return 0.5
    if hero_name in mid_matchups:
        return mid_matchups[hero_name].get('win_rate', 0.5)
    return 0.5


def get_hero_patch_wr(hero_name: str, patch: str) -> float:
    if not hero_name or not patch_meta:
        return 0.5
    patch_data = patch_meta.get(patch, {})
    heroes = patch_data.get('heroes', {})
    if hero_name in heroes:
        return heroes[hero_name].get('win_rate', 50.0) / 100.0
    return 0.5


def get_laning_stat(team_name: str, stat: str, default=0.0) -> float:
    if not team_name or not team_laning:
        return default
    profiles = team_laning if isinstance(team_laning, dict) else {}
    if team_name in profiles:
        return profiles[team_name].get(stat, default)
    norm = team_name.lower().replace(' ', '')
    for t in profiles:
        if t.lower().replace(' ', '') == norm:
            return profiles[t].get(stat, default)
    return default


# ─── Build DataFrame ────────────────────────────────────────────────
def build_training_data():
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
            pass
    
    df = pd.DataFrame(rows)
    print(f"Built DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_match_features(match: dict) -> dict:
    """Extract PRE-MATCH features only (no in-game data)."""
    row = {}
    
    # Target
    row['target'] = 1 if match['radiant_victory'] else 0
    row['match_id'] = match['match_id']
    
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
    
    # ── Hero meta win rates (3 features) ──
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
    
    # ── Player form (18 features: 3 key players per team) ──
    r_by_gpm = sorted(r_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)
    d_by_gpm = sorted(d_players, key=lambda p: p['performance']['gpm'] or 0, reverse=True)
    
    for prefix, players in [('r', r_by_gpm[:3]), ('d', d_by_gpm[:3])]:
        for j, p in enumerate(players):
            pname = p['player']['nickname']
            row[f'{prefix}_p{j}_wr'] = get_player_form_stat(pname, 'recent_win_rate', 0.5)
            row[f'{prefix}_p{j}_gpm'] = get_player_form_stat(pname, 'recent_avg_gpm', 500)
            row[f'{prefix}_p{j}_kda'] = get_player_form_stat(pname, 'recent_avg_kda', 0)
            row[f'{prefix}_p{j}_form_delta'] = get_player_form_stat(pname, 'form_delta.win_rate_delta', 0)
    
    for j in range(3):
        row[f'p{j}_wr_diff'] = row.get(f'r_p{j}_wr', 0.5) - row.get(f'd_p{j}_wr', 0.5)
        row[f'p{j}_gpm_diff'] = row.get(f'r_p{j}_gpm', 500) - row.get(f'd_p{j}_gpm', 500)
    
    # ── Mid matchup HERO win rates (3 features) — pre-match hero stats ──
    r_mid = next((p for p in r_players if p.get('laneInfo', {}).get('lane') == 'MIDDLE'), None)
    d_mid = next((p for p in d_players if p.get('laneInfo', {}).get('lane') == 'MIDDLE'), None)
    
    if r_mid and d_mid:
        r_mid_hero = r_mid['performance']['hero']['short_name']
        d_mid_hero = d_mid['performance']['hero']['short_name']
        row['r_mid_hero_wr'] = get_mid_hero_wr(r_mid_hero)
        row['d_mid_hero_wr'] = get_mid_hero_wr(d_mid_hero)
        row['mid_hero_wr_diff'] = row['r_mid_hero_wr'] - row['d_mid_hero_wr']
    else:
        row['r_mid_hero_wr'] = row['d_mid_hero_wr'] = row['mid_hero_wr_diff'] = 0.5
    
    # ── Laning stats (6 features) ──
    for prefix, team_name in [('r', radiant_team), ('d', dire_team)]:
        row[f'{prefix}_lanes_won'] = get_laning_stat(team_name, 'lanes_won_pct', 50)
        row[f'{prefix}_nw_adv_laning'] = get_laning_stat(team_name, 'nw_advantage', 0)
        row[f'{prefix}_fb_pct'] = get_laning_stat(team_name, 'fb_pct', 50)
    
    # ── Tournament tier (1 feature) ──
    row['tier_weight'] = tier_lookup.get(league_name, 0.4)
    
    # ── Draft diversity (2 features) ──
    row['r_hero_diversity'] = len(set(r_heroes)) / 5.0
    row['d_hero_diversity'] = len(set(d_heroes)) / 5.0
    
    return row


# ─── Main training pipeline ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("Pre-Match ML Training Pipeline")
    print("Predict match outcome BEFORE game starts")
    print("=" * 60)
    
    df = build_training_data()
    
    # Define feature columns
    exclude_cols = {'target', 'match_id'}
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
        print(f"\nWarning: {nan_count} NaN, {inf_count} Inf values - filling with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ─── Train with cross-validation ───
    print("\n" + "-" * 40)
    print("Training XGBoost with 5-fold CV...")
    
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print(f"\n  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  CV ROC AUC:  {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")
    print(f"  Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # ─── Train final model ───
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
    
    # ─── Save model ───
    model_path = MODEL_DIR / "xgb_prematch.json"
    model.save_model(str(model_path))
    
    metadata = {
        'phase': '1-prematch',
        'type': 'prematch',
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'cv_accuracy': float(cv_scores.mean()),
        'cv_roc_auc': float(cv_roc.mean()),
        'radiant_win_rate': float(np.mean(y)),
        'top_features': [(name, float(imp)) for name, imp in feat_imp[:30]]
    }
    
    meta_path = MODEL_DIR / "model_metadata_prematch.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    cols_path = MODEL_DIR / "feature_cols_prematch.json"
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    print(f"\n  Model saved to {model_path}")
    print(f"  Metadata saved to {meta_path}")
    print(f"\n{'=' * 60}")
    print("PRE-MATCH MODEL DONE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
