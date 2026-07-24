"""
ML Training Module for Dota Analyst.

Collects historical match data, trains models, and provides predictions.

Data sources:
- Stratz API: /api/team/{id}/matches (historical data)
- Stratz API: /api/match/{match_id} (detailed stats)
- DLTV /live/{id}.json: postmatch verification

Training schedule:
- Initial bulk training: past 90 days of matches
- Incremental updates: new matches daily
- Retraining trigger: when >1000 new samples collected
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# External deps (install via pip)
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error
except ImportError:
    print("[ml] scikit-learn not installed, training disabled")
    np = None


import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stratz_api import _http_json, BASE_URL


class MLTrainer:
    """Main class for ML training and prediction."""
    
    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Models
        self.winner_model: Optional[RandomForestClassifier] = None
        self.duration_model: Optional[RandomForestRegressor] = None
        self.kills_model: Optional[RandomForestRegressor] = None
        
        # Encoders
        self.hero_encoder: Optional[LabelEncoder] = None
        self.team_encoder: Optional[LabelEncoder] = None
        
        # Scaler
        self.scaler: Optional[StandardScaler] = None
        
        # Training timestamp
        self.last_training_ts = 0.0
    
    # ========================================================================= #
    # DATA COLLECTION
    # ========================================================================= #
    
    def fetch_team_history(self, team_id: int, limit: int = 100) -> List[Dict]:
        """Fetch recent matches for a team from Stratz API."""
        url = f"{BASE_URL}/api/team/{team_id}/matches"
        params = {
            "limit": min(limit, 200),  # Stratz max
            "order": "desc",
            "type": "all",
        }
        
        headers = {}
        api_key = os.environ.get("STRAZT_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DotaAnalyst/1.0")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return []
                data = json.loads(resp.read().decode("utf-8"))
                
            matches = data.get("matches", [])[:limit]
            
            # Enrich each match with details
            enriched = []
            for m in matches:
                mid = m.get("id") or m.get("match_id")
                if mid:
                    details = self.fetch_match_details(mid)
                    if details:
                        m["_details"] = details
                        enriched.append(m)
                    else:
                        enriched.append(m)
                else:
                    enriched.append(m)
            
            return enriched
            
        except Exception as e:
            print(f"[ml] Failed to fetch team {team_id}: {e}")
            return []
    
    def fetch_match_details(self, match_id: int) -> Optional[Dict]:
        """Fetch full match details from Stratz."""
        url = f"{BASE_URL}/api/match/{match_id}"
        
        api_key = os.environ.get("STRAZT_API_KEY")
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DotaAnalyst/1.0")
            for k, v in headers.items():
                req.add_header(k, v)
            
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                if resp.status != 200:
                    return None
                return json.loads(resp.read().decode("utf-8"))
                
        except Exception as e:
            print(f"[ml] Match details fetch failed for {match_id}: {e}")
            return None
    
    def parse_match_to_features(self, match: Dict) -> Optional[Dict]:
        """Convert a Stratz match dict to ML feature vector."""
        if not match:
            return None
        
        details = match.get("_details", {}) or {}
        
        # Basic stats
        radiant_win = match.get("radiant_win") or details.get("radiant_win")
        if radiant_win is None:
            return None
        
        duration_sec = match.get("duration") or details.get("duration") or 0
        kills_radiant = match.get("radiant_score") or details.get("radiant_score") or 0
        kills_dire = match.get("dire_score") or details.get("dire_score") or 0
        
        # Parse players
        radiant_players = details.get("players", [])[:5] if details.get("players") else []
        dire_players = [p for p in details.get("players", []) if p.get("player_slot", 255) >= 128][:5]
        
        # Feature extraction
        features = {
            # Duration
            "duration_min": duration_sec / 60.0,
            
            # Kills
            "total_kills": kills_radiant + kills_dire,
            "radiant_kills": kills_radiant,
            "dire_kills": kills_dire,
            
            # Win label (0: Dire win, 1: Radiant win)
            "winner_radiant": 1 if radiant_win else 0,
            
            # Draft features (if available)
            "radiant_hero_ids": [
                details["heroes"][i]["hero_id"] 
                for i in range(5)
            ] if details.get("heroes") else [0]*5,
            
            # Time features
            "duration_bucket": self._duration_bucket(duration_sec),
            "early_game_factor": min(1.0, kills_radiant + kills_dire) / 60.0,
        }
        
        return features
    
    @staticmethod
    def _duration_bucket(sec: int) -> str:
        """Categorize duration into buckets."""
        if sec < 1800:
            return "short"      # < 30 min
        elif sec < 2700:
            return "medium"     # 30-45 min
        else:
            return "long"       # > 45 min
    
    # ========================================================================= #
    # MODEL TRAINING
    # ========================================================================= #
    
    def collect_training_samples(self, min_matches: int = 500) -> List[Dict]:
        """Collect historical match samples for training."""
        samples = []
        
        # If we have cached samples, load them
        cache_file = self.data_dir / "samples.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    samples = json.load(f)
                print(f"[ml] Loaded {len(samples)} cached samples")
            except Exception as e:
                print(f"[ml] Cache load failed: {e}")
        
        # Fetch more from Stratz if needed
        if len(samples) < min_matches:
            # TODO: This needs team IDs - we'll fetch from discovery or user input
            print(f"[ml] Need {min_matches - len(samples)} more samples")
        
        return samples
    
    def train_winner_model(self, samples: List[Dict]):
        """Train Random Forest classifier for winner prediction."""
        if not samples:
            print("[ml] No samples to train winner model")
            return
        
        # Extract features
        X = []
        y = []
        
        for s in samples:
            feat = [
                s.get("duration_min", 0),
                s.get("total_kills", 50),
                s.get("early_game_factor", 0.5),
            ]
            label = s.get("winner_radiant", 0)
            X.append(feat)
            y.append(label)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        self.winner_model = model
        self.last_training_ts = time.time()
        
        print(f"[ml] Winner model trained! Accuracy: {acc:.2%} on {len(X_test)} samples")
        
        # Save model
        self._save_model("winner.pkl", {
            "model": model,
            "scaler": self.scaler,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        })
    
    def train_duration_model(self, samples: List[Dict]):
        """Train Random Forest regressor for match duration prediction."""
        if not samples:
            print("[ml] No samples to train duration model")
            return
        
        X = []
        y = []
        
        for s in samples:
            feat = [
                s.get("total_kills", 50),
                s.get("early_game_factor", 0.5),
            ]
            target = s.get("duration_min", 40)
            X.append(feat)
            y.append(target)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        mae = mean_absolute_error(y_test, model.predict(X_test_scaled))
        
        self.duration_model = model
        self.last_training_ts = time.time()
        
        print(f"[ml] Duration model trained! MAE: {mae:.2f} min on {len(X_test)} samples")
        
        self._save_model("duration.pkl", {
            "model": model,
            "scaler": self.scaler,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        })
    
    # ========================================================================= #
    # PREDICTION
    # ========================================================================= #
    
    def predict_winner_prob(self, radiant_stats: Dict, dire_stats: Dict) -> float:
        """Predict probability that radiant will win."""
        if not self.winner_model or not self.scaler:
            return 0.5  # Fallback
        
        # Combine team stats into feature vector
        feat = [
            (radiant_stats.get("win_rate", 50) + dire_stats.get("win_rate", 50)) / 100 * 0.6 + 0.5,
            (radiant_stats.get("fb_rate", 0.5) + dire_stats.get("fb_rate", 0.5)),
            0.5,  # placeholder for early_game_factor
        ]
        
        try:
            feat_scaled = self.scaler.transform([feat])
            prob = self.winner_model.predict_proba(feat_scaled)[0][1]
            return float(prob)
        except Exception as e:
            print(f"[ml] Prediction failed: {e}")
            return 0.5
    
    def predict_duration_min(self, total_kills_prediction: int) -> float:
        """Predict match duration in minutes."""
        if not self.duration_model or not self.scaler:
            return 40.0  # Fallback
        
        feat = [[
            total_kills_prediction,
            total_kills_prediction / 60.0,
        ]]
        
        try:
            feat_scaled = self.scaler.transform(feat)
            dur = self.duration_model.predict(feat_scaled)[0]
            return float(dur)
        except Exception as e:
            print(f"[ml] Duration prediction failed: {e}")
            return 40.0
    
    # ========================================================================= #
    # UTILS
    # ========================================================================= #
    
    def _save_model(self, filename: str, data: Dict):
        """Save model to disk (pickle)."""
        try:
            import pickle
            model_path = self.data_dir / filename
            with open(model_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[ml] Saved model to {model_path}")
        except Exception as e:
            print(f"[ml] Model save failed: {e}")
    
    def _load_model(self, filename: str) -> Optional[Dict]:
        """Load model from disk."""
        try:
            import pickle
            model_path = self.data_dir / filename
            if not model_path.exists():
                return None
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ml] Model load failed: {e}")
            return None


# Singleton instance
trainer = MLTrainer()


def init_ml_trainer():
    """Initialize trainer and load pre-trained models."""
    trainer.winner_model = trainer._load_model("winner.pkl")
    trainer.duration_model = trainer._load_model("duration.pkl")
    print(f"[ml] Trainer initialized (last_training={datetime.fromtimestamp(trainer.last_training_ts)})")


def get_predictions(radiant_stats: Dict, dire_stats: Dict, heroes_a: List, heroes_b: List) -> Dict:
    """Get ML-enhanced predictions."""
    # Use trainer to enrich existing predictions from analysis.py
    # This is called after basic analysis, before returning to frontend
    pass
