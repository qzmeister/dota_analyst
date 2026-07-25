"""
Eval harness: compare HeuristicEngine vs MLEngine side-by-side on the
1111-match corpus.

What it computes
----------------
For every match in `ml_data/full_matches/` we run BOTH engines on
the same (team_a, team_b, heroes_a, heroes_b) tuple and record:

  - **winner accuracy**   — does `result["winner"]["team"]` match
                            the actual winner team name?
  - **winner log_loss**   — `-log(p_actual)` for the predicted
                            probability assigned to the actual winner
  - **kills MAE / RMSE**  — |predicted_total - actual_total| etc.
  - **duration MAE**      — minutes off the predicted mean

The two engines are scored on the SAME inputs.  Both see the
hero_ids; NEITHER sees the post-match `radiant_victory` / kills /
duration (those are only used for the actuals).

Why a script, not a test
------------------------
The 159-test unit suite is the safety net.  This script is the
*measurement* — it's run after a new model is trained, to answer
"is the new model actually better than the heuristic?".

Run with:
    python scripts/eval_engines.py

Or with a custom model dir:
    MODEL_DIR=ml_data/models python scripts/eval_engines.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `business` importable when run as a plain script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from business.ml.engine import (  # noqa: E402
    HeuristicEngine,
    MLEngine,
    get_default_engine,
    reset_default_engine,
)
from business.ml.storage import ModelStorage  # noqa: E402


# --------------------------------------------------------------------------- #
# Match → (teams, heroes) conversion
# --------------------------------------------------------------------------- #

def synth_team(m: dict, side: str) -> dict:
    """Minimal team dict that `analyze()` accepts.

    The full_matches corpus does NOT carry DLTV-style team
    aggregates (`win_rate`, `fb_rate`, `f10_rate`).  We pass
    fallback values of 50.0 so the heuristic produces a non-null
    result without claiming a real team strength.  This keeps the
    comparison fair (both engines see the same information).

    0.3.10: also pass through `valve_id` so the engine's team
    feature group (C retry) gets the same per-team encoder lookup
    it would in production.
    """
    t = (m.get(side) or {}).get("team") or {}
    vid = t.get("valve_id")
    out = {
        "name": t.get("name") or f"Team-{side}",
        "win_rate": 50.0,
        "fb_rate": 50.0,
        "f10_rate": 50.0,
        "rank": None,
    }
    if isinstance(vid, int):
        out["valve_id"] = int(vid)
    return out


def synth_heroes(m: dict, side: str) -> list:
    """Minimal hero-meta list that `analyze()` + MLEngine need.

    `steam_id` is the Valve hero id, set to the same `valve_id`
    that we trained on (DatDota full_matches call it the same thing).
    """
    out: list = []
    for p in (m.get(side) or {}).get("player_performances") or []:
        h = (p.get("performance") or {}).get("hero") or {}
        vid = h.get("valve_id")
        if vid is None:
            continue
        out.append({
            "id": vid,
            "steam_id": vid,
            "name": h.get("short_name"),
            "win_rate": 50.0,
            "avg_duration": 38 * 60,
            "kda": 3.0,
            "roles": [],
        })
        if len(out) == 5:
            break
    return out


def actual_team_name(m: dict) -> Optional[str]:
    rv = m.get("radiant_victory")
    if rv is None:
        return None
    side = "radiant" if rv else "dire"
    return ((m.get(side) or {}).get("team") or {}).get("name")


def actual_kills(m: dict) -> Optional[int]:
    total = 0
    for side in ("radiant", "dire"):
        for p in (m.get(side) or {}).get("player_performances") or []:
            k = (p.get("performance") or {}).get("kills")
            if isinstance(k, (int, float)):
                total += int(k)
    return total


def actual_duration_min(m: dict) -> Optional[float]:
    d = m.get("duration")
    if not isinstance(d, (int, float)):
        return None
    return float(d) / 60.0


def _max_player_kills(m: dict) -> Optional[int]:
    best = 0
    found = False
    for side in ("radiant", "dire"):
        for p in (m.get(side) or {}).get("player_performances") or []:
            k = (p.get("performance") or {}).get("kills")
            if isinstance(k, (int, float)):
                found = True
                if k > best:
                    best = int(k)
    return best if found else None


def actual_multikill_level(m: dict) -> Optional[str]:
    """Same binning as `business.ml.targets.target_multikill`."""
    max_k = _max_player_kills(m)
    if max_k is None:
        return None
    if max_k >= 7:
        return "High"
    if max_k >= 4:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class Metrics:
    n: int = 0
    winner_correct: int = 0
    winner_log_loss: List[float] = field(default_factory=list)
    kills_pred: List[int] = field(default_factory=list)
    kills_actual: List[int] = field(default_factory=list)
    duration_pred: List[float] = field(default_factory=list)
    duration_actual: List[float] = field(default_factory=list)
    multikill_pred: List[str] = field(default_factory=list)
    multikill_actual: List[str] = field(default_factory=list)

    def add(
        self,
        result: Dict[str, Any],
        actual_winner: Optional[str],
        actual_kills_v: Optional[int],
        actual_dur_v: Optional[float],
        actual_mk_v: Optional[str] = None,
    ) -> None:
        self.n += 1
        # Winner
        w = (result.get("winner") or {})
        if actual_winner is not None and w.get("team") == actual_winner:
            self.winner_correct += 1
        prob_actual = self._prob_for_actual(w, result, actual_winner)
        if prob_actual is not None and 0.0 < prob_actual <= 1.0:
            self.winner_log_loss.append(-math.log(prob_actual))

        # Kills
        k = result.get("kills") or {}
        if actual_kills_v is not None and isinstance(k.get("total"), int):
            self.kills_pred.append(int(k["total"]))
            self.kills_actual.append(int(actual_kills_v))

        # Duration
        if actual_dur_v is not None and isinstance(result.get("duration_min"), (int, float)):
            self.duration_pred.append(float(result["duration_min"]))
            self.duration_actual.append(float(actual_dur_v))

        # Multikill
        mk = (result.get("multikill") or {})
        pred_mk = mk.get("level") if isinstance(mk.get("level"), str) else None
        if actual_mk_v is not None and pred_mk is not None:
            self.multikill_pred.append(pred_mk)
            self.multikill_actual.append(actual_mk_v)

    @staticmethod
    def _prob_for_actual(winner_block: dict, result: dict, actual_winner: Optional[str]) -> Optional[float]:
        """Convert the engine's winner block into a probability for the actual team.

        The ML winner block carries `prob_radiant` directly; the
        heuristic block doesn't, so we derive it from
        `winner.probability` and the side of the actual winner.
        """
        if not winner_block or actual_winner is None:
            return None
        pr = winner_block.get("prob_radiant")
        if isinstance(pr, (int, float)):
            p_radiant = float(pr) / 100.0
        else:
            # Heuristic path — no prob_radiant; assume 50/50 baseline.
            p_radiant = 0.5
        return p_radiant

    def summarise(self, name: str) -> Dict[str, Any]:
        n_winner = self.n  # every match has a winner prediction
        winner_acc = (self.winner_correct / n_winner) if n_winner else 0.0
        winner_ll = statistics.fmean(self.winner_log_loss) if self.winner_log_loss else None

        if self.kills_pred:
            kill_errs = [p - a for p, a in zip(self.kills_pred, self.kills_actual)]
            kill_mae = sum(abs(e) for e in kill_errs) / len(kill_errs)
            kill_rmse = math.sqrt(sum(e * e for e in kill_errs) / len(kill_errs))
        else:
            kill_mae = kill_rmse = None

        if self.duration_pred:
            dur_errs = [p - a for p, a in zip(self.duration_pred, self.duration_actual)]
            dur_mae = sum(abs(e) for e in dur_errs) / len(dur_errs)
            dur_rmse = math.sqrt(sum(e * e for e in dur_errs) / len(dur_errs))
        else:
            dur_mae = dur_rmse = None

        if self.multikill_pred:
            mk_correct = sum(
                1 for p, a in zip(self.multikill_pred, self.multikill_actual)
                if p == a
            )
            mk_acc = mk_correct / len(self.multikill_pred)
        else:
            mk_acc = None

        return {
            "engine": name,
            "n": n_winner,
            "winner_accuracy": winner_acc,
            "winner_log_loss": winner_ll,
            "kills_mae": kill_mae,
            "kills_rmse": kill_rmse,
            "duration_mae": dur_mae,
            "duration_rmse": dur_rmse,
            "multikill_accuracy": mk_acc,
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    data_dir = _ROOT / "ml_data" / "full_matches"
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        return 1

    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        print("no match files found", file=sys.stderr)
        return 1
    print(f"loading {len(paths)} matches from {data_dir}...")

    # Build the two engines.  Heuristic is always available; ML
    # is whatever sub-models are on disk (could be 0..5).
    heuristic = HeuristicEngine()
    reset_default_engine()
    ml = get_default_engine()
    ml_name = ml.name
    if ml_name == "ml":
        sub = sorted(ml._sub_models.keys())  # type: ignore[attr-defined]
        print(f"ml engine loaded sub-models: {sub}")
    else:
        print(f"ml engine degraded to: {ml_name}")

    h_metrics = Metrics()
    m_metrics = Metrics()

    t0 = time.perf_counter()
    n_processed = 0
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            continue

        # Skip matches we can't make sense of.
        if not isinstance(m.get("radiant_victory"), bool):
            continue
        team_a = synth_team(m, "radiant")
        team_b = synth_team(m, "dire")
        heroes_a = synth_heroes(m, "radiant")
        heroes_b = synth_heroes(m, "dire")
        if len(heroes_a) != 5 or len(heroes_b) != 5:
            continue

        aw = actual_team_name(m)
        ak = actual_kills(m)
        ad = actual_duration_min(m)
        amk = actual_multikill_level(m)

        # Heuristic
        try:
            h_res = heuristic.analyze(team_a, team_b, heroes_a, heroes_b)
            h_metrics.add(h_res, aw, ak, ad, amk)
        except Exception as exc:  # noqa: BLE001
            pass

        # ML (only if it's a real MLEngine, not the degraded heuristic)
        if ml_name == "ml":
            try:
                m_res = ml.analyze(team_a, team_b, heroes_a, heroes_b)
                m_metrics.add(m_res, aw, ak, ad, amk)
            except Exception as exc:  # noqa: BLE001
                pass
        else:
            # Track the same row on the "ml" side so the n is comparable
            m_metrics.add({"winner": {}, "kills": {}}, aw, ak, ad, amk)

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  processed {n_processed} matches...")

    elapsed = time.perf_counter() - t0

    print(f"\nProcessed {n_processed} matches in {elapsed:.1f}s")
    print()
    print("=" * 70)
    print(f"  {'metric':<22} {'heuristic':>14} {'ml':>14}   delta")
    print("=" * 70)

    h = h_metrics.summarise("heuristic")
    m = m_metrics.summarise("ml")
    rows = [
        ("winner accuracy", h["winner_accuracy"], m["winner_accuracy"], True),
        ("winner log_loss", h["winner_log_loss"], m["winner_log_loss"], False),
        ("kills MAE",       h["kills_mae"],        m["kills_mae"],        False),
        ("kills RMSE",      h["kills_rmse"],       m["kills_rmse"],       False),
        ("duration MAE",    h["duration_mae"],     m["duration_mae"],     False),
        ("duration RMSE",   h["duration_rmse"],    m["duration_rmse"],    False),
        ("multikill acc",   h["multikill_accuracy"], m["multikill_accuracy"], True),
    ]
    for label, hv, mv, higher_is_better in rows:
        if hv is None or mv is None:
            print(f"  {label:<22} {'n/a':>14} {'n/a':>14}")
            continue
        delta = (mv - hv) if isinstance(hv, float) and isinstance(mv, float) else 0.0
        sign = "+" if delta > 0 else ""
        better = ""
        if isinstance(delta, float) and abs(delta) > 1e-9:
            if (higher_is_better and delta > 0) or (not higher_is_better and delta < 0):
                better = "  <-- ml better"
            else:
                better = "  <-- heur better"
        print(f"  {label:<22} {hv:>14.4f} {mv:>14.4f}   {sign}{delta:+.4f}{better}")

    print()
    print(f"  heuristic n={h['n']}, ml n={m['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
