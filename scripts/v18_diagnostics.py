"""v18 winner model diagnostics.

Loads the v18 stage 2 XGBClassifier (or LGB head when present) and
runs it over the v17_match_*.json corpus to produce:

  1. Feature importance (gain) — top 25 features that actually move
     the prediction.  Helps spot "the model is leaning on X" patterns
     (e.g. one specific hero's one-hot dominating gain).
  2. Per-group bias — accuracy / log-loss broken down by:
       * patch (7.40 vs 7.41 vs ...)
       * tier (premium / professional / minor)
       * game duration (short / medium / long)
       * top_team flag (was at least one team a "top" team?)
     Surfaces which slices the model is systematically wrong on.
  3. Error breakdown — top-N most-confident wrong predictions
     (the "head-shakers" the model bet the house on and lost).
     Lists the draft + tier + days_since_patch so we can eyeball
     whether they share a common cause.
  4. Calibration check — bins predicted_prob_radiant into 10
     deciles and reports actual win-rate per bin.  A well-calibrated
     model has actual ≈ predicted in every bin.

Output: a JSON report at `ml_data/diagnostics/v18_<timestamp>.json`
plus a one-screen text summary on stdout.  The UI's diagnostics
tab reads the latest JSON (no recompute on every page load).

Run:  python scripts/v18_diagnostics.py [--limit N] [--model-dir _v18_winner_xgb_stage2]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v18 import (  # noqa: E402
    extract_features, list_match_files, _load_top_teams, _load_patch_info,
    PRO_ROOT, ML_DATA, IMPORTS, MODELS, NUM_HEROES,
)

# Default model dir / corpus dir — overridable via CLI.
DEFAULT_MODEL = "_v18_winner_xgb_stage2"
DIAG_DIR = ML_DATA / "diagnostics"


def _load_model(model_dir: Path):
    """Load model + metadata + return the booster (for feature importance)."""
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_names = list(meta.get("feature_columns") or [])
    n_expected = meta.get("n_features")
    if feature_names and n_expected and len(feature_names) != n_expected:
        print(f"WARNING: metadata.n_features={n_expected} but feature_columns has {len(feature_names)}",
              file=sys.stderr)
    model = joblib.load(model_dir / "model.joblib")
    return model, feature_names, meta


def _feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Top features by gain (XGBoost booster) and by weight (#splits)."""
    out: List[Dict[str, Any]] = []
    try:
        booster = model.get_booster()
        gain_score = booster.get_score(importance_type="gain") or {}
        weight_score = booster.get_score(importance_type="weight") or {}
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: booster.get_score failed: {exc}", file=sys.stderr)
        return out
    # XGBoost returns "fNNN" keys; map back to the real column name.
    rows = []
    for k, g in gain_score.items():
        try:
            idx = int(k.lstrip("f"))
        except ValueError:
            continue
        name = feature_names[idx] if 0 <= idx < len(feature_names) else k
        rows.append({
            "feature":   name,
            "index":     idx,
            "gain":      round(float(g), 4),
            "weight":    int(weight_score.get(k, 0)),
        })
    rows.sort(key=lambda r: -r["gain"])
    return rows


def _group_accuracy(groups: Dict[str, List[Tuple[int, float]]]) -> List[Dict[str, Any]]:
    """groups: {label: [(actual, predicted_prob), ...]}.
    Returns per-label: {label, n, acc, log_loss, brier}.
    """
    out: List[Dict[str, Any]] = []
    for label, items in groups.items():
        n = len(items)
        if n == 0:
            continue
        actuals = np.array([a for a, _ in items], dtype=np.int8)
        probs   = np.array([p for _, p in items], dtype=np.float32)
        preds   = (probs >= 0.5).astype(np.int8)
        acc = float((preds == actuals).mean())
        # log_loss — clip to avoid log(0)
        eps = 1e-7
        probs_c = np.clip(probs, eps, 1 - eps)
        ll = float(-(actuals * np.log(probs_c) + (1 - actuals) * np.log(1 - probs_c)).mean())
        brier = float(((probs - actuals) ** 2).mean())
        out.append({"label": label, "n": n, "acc": round(acc, 4),
                    "log_loss": round(ll, 4), "brier": round(brier, 4)})
    # Sort by n descending so the "main" group is first.
    out.sort(key=lambda r: -r["n"])
    return out


def _calibration_buckets(actual: np.ndarray, prob: np.ndarray, n_buckets: int = 10
                         ) -> List[Dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    out: List[Dict[str, Any]] = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        if i < n_buckets - 1:
            mask = (prob >= lo) & (prob < hi)
        else:
            mask = (prob >= lo) & (prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bucket":      f"{lo:.1f}–{hi:.1f}",
            "n":           n,
            "mean_pred":   round(float(prob[mask].mean()), 4),
            "actual_rate": round(float(actual[mask].mean()), 4),
            "gap_pp":      round(float((prob[mask].mean() - actual[mask].mean()) * 100), 2),
        })
    return out


def _duration_bucket(seconds: int) -> str:
    """Roughly split a Dota 2 game into short/med/long/mega."""
    if seconds < 1500: return "<25m"
    if seconds < 2100: return "25-35m"
    if seconds < 2700: return "35-45m"
    if seconds < 3300: return "45-55m"
    return ">55m"


def _patch_bucket(start_time: int) -> str:
    """Coarse patch bucket from Unix start time.  7.40 went live ~2025-04;
    7.41 ~2025-09; 7.41d ~2025-12; 7.42 ~2026 Q1 (rough)."""
    if start_time < 1_750_000_000:    # before ~ Jun 2025
        return "pre-7.40"
    if start_time < 1_760_000_000:    # ~ Sep 2025
        return "7.40"
    if start_time < 1_770_000_000:    # ~ Dec 2025
        return "7.41"
    if start_time < 1_780_000_000:    # ~ Mar 2026
        return "7.41d"
    return "7.42+"


def _safe_label(s: Any, limit: int = 60) -> str:
    """Compact label for a draft (hero IDs joined)."""
    if not s:
        return ""
    return ",".join(str(int(h)) for h in s)[:limit]


def run_diagnostics(model_dir: Path, limit: Optional[int] = None) -> Dict[str, Any]:
    print(f"[diag] loading model: {model_dir}", file=sys.stderr)
    model, feature_names, meta = _load_model(model_dir)

    # 1) Feature importance — straight from the booster, no corpus needed.
    print("[diag] computing feature importance (gain / weight)", file=sys.stderr)
    fi = _feature_importance(model, feature_names)
    top_features = fi[:30]
    # group aggregates
    aggr = collections.Counter()
    for r in fi:
        name = r["feature"]
        if name.startswith("r_h_") or name.startswith("d_h_"):
            aggr["hero_onehot"] += r["gain"]
        elif name.startswith("league_id_") or name.startswith("r_league") or name.startswith("d_league"):
            aggr["league_onehot"] += r["gain"]
        elif name in ("r_tier", "d_tier", "r_top_team", "d_top_team", "r_premium", "d_premium"):
            aggr["tier_flags"] += r["gain"]
        elif name in ("r_picks", "d_picks", "r_bans", "d_bans"):
            aggr["count_features"] += r["gain"]
        elif name == "days_since_patch":
            aggr["patch_recency"] += r["gain"]
        else:
            aggr["other"] += r["gain"]
    fi_aggr = [{"group": k, "total_gain": round(v, 2)} for k, v in aggr.most_common()]

    # 2) Walk the corpus, predict, collect (actual, prob, label, draft) tuples.
    print("[diag] loading top_teams + patch_info", file=sys.stderr)
    top_teams = _load_top_teams()
    patch_info = _load_patch_info()

    files = list_match_files()
    if limit:
        files = files[:limit]
    print(f"[diag] walking {len(files)} match files", file=sys.stderr)

    n_kept = 0
    n_errored = 0
    actual_arr: List[int] = []
    prob_arr:   List[float] = []
    by_patch:   Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    by_tier:    Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    by_dur:     Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    by_top:     Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    # Per-row info for the "biggest mistakes" section.
    rows: List[Dict[str, Any]] = []

    t0 = time.monotonic()
    for i, fp in enumerate(files):
        if i and i % 500 == 0:
            print(f"[diag]   {i}/{len(files)}  ({n_kept} kept, {n_errored} skip)", file=sys.stderr)
        try:
            raw = json.loads(fp.read_text(encoding="utf-8-sig"))
        except Exception:
            n_errored += 1
            continue
        if not isinstance(raw, dict):
            n_errored += 1
            continue
        try:
            feats = extract_features(raw, top_teams=top_teams, patch_info=patch_info)
        except Exception:
            n_errored += 1
            continue
        if feats is None:
            n_errored += 1
            continue
        # Build the X vector in the model's expected column order.
        try:
            x = np.array([[float(feats.get(c, 0.0)) for c in feature_names]], dtype=np.float32)
        except Exception:
            n_errored += 1
            continue
        try:
            prob = float(model.predict_proba(x)[0, 1])
        except Exception:
            n_errored += 1
            continue
        actual = 1 if feats.get("__winner") is not None else int(bool(raw.get("radiant_win")))
        # __winner is set inside extract_features; if missing fall back to raw.
        if "__winner" in feats:
            actual = int(feats["__winner"])
        else:
            actual = 1 if bool(raw.get("radiant_win")) else 0

        actual_arr.append(actual)
        prob_arr.append(prob)
        n_kept += 1

        # group buckets
        try:
            start_time = int(raw.get("start_time") or 0)
        except (TypeError, ValueError):
            start_time = 0
        try:
            duration = int(raw.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0
        r_top_team = int(feats.get("r_top_team", 0))
        d_top_team = int(feats.get("d_top_team", 0))
        r_premium  = int(feats.get("r_premium", 0))
        d_premium  = int(feats.get("d_premium", 0))

        by_patch[_patch_bucket(start_time)].append((actual, prob))
        # tier: premium if either side is premium, professional if either is top,
        # else minor.  (coarse, but useful.)
        if r_premium or d_premium:
            tier_label = "premium"
        elif r_top_team or d_top_team:
            tier_label = "professional"
        else:
            tier_label = "minor"
        by_tier[tier_label].append((actual, prob))
        by_dur[_duration_bucket(duration)].append((actual, prob))
        by_top["top_team" if (r_top_team or d_top_team) else "no_top_team"].append((actual, prob))

        # for the "biggest mistake" section, capture enough context
        rows.append({
            "match_id":   int(raw.get("match_id") or fp.stem.split("_")[-1] or 0),
            "start_time": start_time,
            "duration":   duration,
            "actual":     actual,
            "prob":       round(prob, 4),
            "r_top_team": r_top_team,
            "d_top_team": d_top_team,
            "r_premium":  r_premium,
            "d_premium":  d_premium,
            "days_since_patch": float(feats.get("days_since_patch", 0.0)),
        })

    if n_kept == 0:
        print("ERROR: no matches extracted — check v17_match corpus path", file=sys.stderr)
        sys.exit(2)

    actual_np = np.array(actual_arr, dtype=np.int8)
    prob_np   = np.array(prob_arr, dtype=np.float32)
    preds_np  = (prob_np >= 0.5).astype(np.int8)
    overall_acc = float((preds_np == actual_np).mean())
    eps = 1e-7
    prob_c = np.clip(prob_np, eps, 1 - eps)
    overall_ll = float(-(actual_np * np.log(prob_c) + (1 - actual_np) * np.log(1 - prob_c)).mean())
    overall_brier = float(((prob_np - actual_np) ** 2).mean())

    # Confidence histogram — how often does the model go very confident?
    conf_high = float(((prob_np >= 0.8) | (prob_np <= 0.2)).mean())
    conf_med  = float((((prob_np >= 0.6) & (prob_np < 0.8)) |
                      ((prob_np > 0.2) & (prob_np <= 0.4))).mean())
    conf_low  = 1.0 - conf_high - conf_med

    # Head-shakers: rows where pred was very confident AND wrong.
    confident_mask = (prob_np >= 0.8) | (prob_np <= 0.2)
    wrong_mask      = preds_np != actual_np
    big_mask        = confident_mask & wrong_mask
    big_idx         = np.where(big_mask)[0]
    # Sort by confidence extremity: |prob - 0.5| desc.
    big_idx_sorted = sorted(
        big_idx.tolist(),
        key=lambda j: -abs(float(prob_np[j]) - 0.5),
    )[:25]
    biggest_mistakes = [rows[j] for j in big_idx_sorted]

    # Report
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at":   _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model_dir":      str(model_dir),
        "model_meta":     {k: meta.get(k) for k in ("target", "model_class", "n_features",
                                                   "framework", "trained_at")},
        "corpus": {
            "files_total": len(files),
            "files_kept":  n_kept,
            "files_errored_or_skipped": n_errored,
            "elapsed_sec": round(time.monotonic() - t0, 1),
        },
        "overall": {
            "n": n_kept,
            "acc":     round(overall_acc, 4),
            "log_loss": round(overall_ll, 4),
            "brier":    round(overall_brier, 4),
            "radiant_share": round(float(actual_np.mean()), 4),
            "confidence_buckets": {
                "high (>0.8 or <0.2)": round(conf_high, 4),
                "med  (0.6-0.8 or 0.2-0.4)": round(conf_med, 4),
                "low  (0.4-0.6)":         round(conf_low, 4),
            },
        },
        "feature_importance": {
            "top": top_features,
            "by_group": fi_aggr,
        },
        "bias": {
            "by_patch":      _group_accuracy(by_patch),
            "by_tier":       _group_accuracy(by_tier),
            "by_duration":   _group_accuracy(by_dur),
            "by_top_team":   _group_accuracy(by_top),
        },
        "calibration": _calibration_buckets(actual_np, prob_np),
        "biggest_mistakes": biggest_mistakes,
    }
    return report


def _print_text_summary(r: Dict[str, Any]) -> None:
    print("=" * 70)
    print(f"v18 diagnostics — {r['model_dir']}")
    print(f"  generated: {r['generated_at']}")
    print(f"  corpus:    {r['corpus']['files_kept']} matches  "
          f"({r['corpus']['files_errored_or_skipped']} skipped, "
          f"{r['corpus']['elapsed_sec']}s)")
    o = r["overall"]
    print(f"  overall:   acc {o['acc']:.4f}  log_loss {o['log_loss']:.4f}  "
          f"brier {o['brier']:.4f}  (radiant share {o['radiant_share']:.2%})")
    cb = o["confidence_buckets"]
    print(f"  conf:      high {cb['high (>0.8 or <0.2)']:.1%}  "
          f"med {cb['med  (0.6-0.8 or 0.2-0.4)']:.1%}  "
          f"low {cb['low  (0.4-0.6)']:.1%}")

    print("\nFeature importance (top 10 by gain):")
    for rk in r["feature_importance"]["top"][:10]:
        print(f"  {rk['feature']:30s}  gain={rk['gain']:8.2f}  weight={rk['weight']:5d}")
    print("\nFeature groups:")
    for rk in r["feature_importance"]["by_group"]:
        print(f"  {rk['group']:20s}  total_gain={rk['total_gain']:8.2f}")

    print("\nBias by patch:")
    for rk in r["bias"]["by_patch"]:
        print(f"  {rk['label']:12s}  n={rk['n']:5d}  acc={rk['acc']:.4f}  "
              f"log_loss={rk['log_loss']:.4f}  brier={rk['brier']:.4f}")
    print("\nBias by tier:")
    for rk in r["bias"]["by_tier"]:
        print(f"  {rk['label']:12s}  n={rk['n']:5d}  acc={rk['acc']:.4f}  "
              f"log_loss={rk['log_loss']:.4f}  brier={rk['brier']:.4f}")
    print("\nBias by duration:")
    for rk in r["bias"]["by_duration"]:
        print(f"  {rk['label']:12s}  n={rk['n']:5d}  acc={rk['acc']:.4f}  "
              f"log_loss={rk['log_loss']:.4f}  brier={rk['brier']:.4f}")
    print("\nCalibration (10 buckets):")
    for rk in r["calibration"]:
        gap_sign = "+" if rk["gap_pp"] >= 0 else ""
        print(f"  pred {rk['bucket']:8s}  n={rk['n']:5d}  actual={rk['actual_rate']:.3f}  "
              f"gap={gap_sign}{rk['gap_pp']:.1f}pp")

    print("\nBiggest mistakes (top 10 by |prob - 0.5|):")
    for rk in r["biggest_mistakes"][:10]:
        actual_str = "Radiant" if rk["actual"] == 1 else "Dire"
        pred_str   = f"Radiant {rk['prob']:.0%}" if rk["prob"] >= 0.5 else f"Dire {1 - rk['prob']:.0%}"
        print(f"  match {rk['match_id']:>10}  {pred_str:>16s}  →  {actual_str} won  "
              f"(days_p={rk['days_since_patch']:.1f}, r_top={rk['r_top_team']}, d_top={rk['d_top_team']})")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL,
                        help="model subdir under ml_data/models/ (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only walk the first N match files (debug)")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: ml_data/diagnostics/v18_<ts>.json)")
    args = parser.parse_args()

    model_dir = MODELS / args.model_dir
    if not (model_dir / "metadata.json").exists():
        print(f"ERROR: no metadata at {model_dir / 'metadata.json'}", file=sys.stderr)
        return 2

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = DIAG_DIR / f"v18_{ts}.json"
    # Also write a stable "latest" copy the UI can pick up without listing.
    latest_path = DIAG_DIR / "v18_latest.json"

    report = run_diagnostics(model_dir, limit=args.limit)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_text_summary(report)
    print(f"\nWrote {out_path}")
    print(f"Wrote {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
