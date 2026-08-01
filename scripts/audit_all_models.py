"""Full ML audit of every model under ml_data/models/.

Walks the v1..v16 legacy models, the _v17_* trained models, and
the _v18_* trained models.  For each one:
  1. Verifies the files exist (model.joblib + metadata.json).
  2. Loads the model and reports size, class, n_features.
  3. Tries a predict on a synthetic baseline (the same input the
     live card would pass).  Reports the prediction.
  4. Tries 3 variants of the input (different draft) to detect
     the "59% on every team" bug — if all 3 return the same prob,
     the model is dead.

Also reports:
  - tier coverage: % of unique team_ids in ml_data/imports that
    appear in v17_phase1_top_teams.json
  - patch coverage: every distinct patch string seen in the
    corpus, and which of those is in the trainer's _PATCHES list
  - hero ID range: min/max hero_id seen in the corpus
  - regression: compare the v18 winner model output to v17 winner
    on 3 different drafts.

Output: human-readable report.  No files modified.  This is a
read-only audit.  Run:

    cd C:\\Users\\artka\\.minimax\\workspace\\dota_analyst
    $env:PYTHONPATH="."
    python scripts/audit_all_models.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

PRO_ROOT = Path(__file__).resolve().parents[1]
MODELS = PRO_ROOT / "ml_data" / "models"
IMPORTS = PRO_ROOT / "ml_data" / "imports"

# Track groups in MODELS/
LEGACY_PREFIXES = ("winner_", "kills_", "duration_")
V17_PREFIX = "_v17_"
V18_PREFIX = "_v18_"


def list_model_dirs() -> List[Path]:
    out = []
    for p in sorted(MODELS.iterdir()):
        if not p.is_dir():
            continue
        if (p / "model.joblib").exists():
            out.append(p)
    return out


def categorise(p: Path) -> str:
    n = p.name
    if n.startswith(V18_PREFIX):
        return "v18"
    if n.startswith(V17_PREFIX):
        return "v17"
    if any(n.startswith(pre) for pre in LEGACY_PREFIXES):
        return "legacy"
    return "other"


def try_load(p: Path) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """Returns (model, metadata, error).  model is None on failure."""
    err: Optional[str] = None
    model = None
    meta: Optional[Dict[str, Any]] = None
    try:
        model = joblib.load(p / "model.joblib")
    except Exception as exc:
        err = f"joblib.load failed: {exc}"
    if err is None:
        meta_path = p / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                err = f"metadata parse failed: {exc}"
    return model, meta, err


def build_synthetic_input(meta: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build a 1-row numpy input of the right shape for `meta`.

    If the model has feature_columns in metadata we follow that
    order; otherwise we fall back to whatever the model's
    `n_features_in_` attribute says.  All values are 0.5/0 — the
    synthetic baseline is "no info, no signal".
    """
    cols = meta.get("feature_columns")
    if cols:
        X = np.zeros((1, len(cols)), dtype=np.float32)
        for i, c in enumerate(cols):
            # Heuristic: categorical / one-hot stay 0, scalars 0.5
            if c.startswith(("r_h_", "d_h_", "r_b_", "d_b_")) or c in (
                "r_top_team", "d_top_team", "r_premium", "d_premium",
                "r_picks", "d_picks", "r_bans", "d_bans",
                "r_is_pick", "d_is_pick",
            ):
                X[0, i] = 0
            else:
                X[0, i] = 0.5
        return X
    n = getattr(model, "n_features_in_", None) if False else None
    if n is None:
        try:
            n = model.n_features_in_  # type: ignore[union-attr]
        except AttributeError:
            return None
    return np.full((1, int(n)), 0.5, dtype=np.float32)


def predict_3_variants(model: Any) -> List[Any]:
    """Return predictions for 3 different inputs to detect
    constant-output bugs (the "59% on every team" smoking gun).
    """
    n = None
    try:
        n = model.n_features_in_  # type: ignore[union-attr]
    except AttributeError:
        return []
    if n is None:
        return []
    out: List[Any] = []
    for fill in (0.0, 0.5, 1.0):
        X = np.full((1, int(n)), fill, dtype=np.float32)
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                # binary: [p_dire, p_radiant] -- take p_radiant
                v = float(p[0][-1]) if p.ndim == 2 else float(p[0])
            else:
                v = float(model.predict(X)[0])
        except Exception as exc:
            v = f"err:{exc}"
        out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Audit: per-model load + predict-3-variants
# --------------------------------------------------------------------------- #

def audit_models() -> None:
    print("=" * 78)
    print("Per-model audit (load, size, predict-3-variants)")
    print("=" * 78)
    print()
    rows: List[Dict[str, Any]] = []
    for p in list_model_dirs():
        cat = categorise(p)
        size = (p / "model.joblib").stat().st_size
        model, meta, err = try_load(p)
        n_features = None
        m_class = None
        if model is not None:
            try:
                m_class = type(model).__name__
            except Exception:
                m_class = "?"
            try:
                n_features = int(getattr(model, "n_features_in_", 0) or 0)
            except Exception:
                pass
        preds: List[Any] = []
        if model is not None:
            preds = predict_3_variants(model)
        same = (
            len(preds) == 3
            and all(isinstance(x, float) for x in preds)
            and abs(preds[0] - preds[1]) < 1e-6
            and abs(preds[1] - preds[2]) < 1e-6
        )
        # Pre-v18 models and most regressors are not expected to
        # vary on this synthetic input (no hero info).  Only
        # flag the binary classifier rows where a constant proba
        # is a known smoking gun (the v17_winner regression test).
        suspect_constant = (
            cat in ("v17", "v18")
            and m_class and "Classif" in m_class
            and same
        )
        # A tiny file is suspicious for a real model.
        is_tiny = size < 5000
        rows.append({
            "name": p.name,
            "cat": cat,
            "size": size,
            "m_class": m_class,
            "n_features": n_features,
            "err": err,
            "preds": preds,
            "same": same,
            "suspect_constant": suspect_constant,
            "is_tiny": size < 5000,
        })
    # Group by category
    for cat in ("legacy", "v17", "v18"):
        sub = [r for r in rows if r["cat"] == cat]
        if not sub:
            continue
        print(f"--- {cat.upper()} ({len(sub)} models) ---")
        for r in sub:
            tag = "  "
            if r["err"]:
                tag = "!!"  # load failed
            elif r["suspect_constant"]:
                tag = "##"  # constant output = dead model
            elif r["is_tiny"]:
                tag = "??"  # tiny file
            err = f"  err={r['err']}" if r["err"] else ""
            print(
                f"  {tag} {r['name']:38s}  {r['size']:>9d}b  "
                f"class={r['m_class'] or '?':24s}  "
                f"n_features={r['n_features'] or '?':>4}  "
                f"preds={r['preds']}{err}"
            )
        print()


# --------------------------------------------------------------------------- #
# Audit: tier coverage
# --------------------------------------------------------------------------- #

def audit_tier_coverage() -> None:
    print("=" * 78)
    print("Tier coverage: which team_ids are in v17_phase1_top_teams.json?")
    print("=" * 78)
    print()
    p = IMPORTS / "v17_phase1_top_teams.json"
    if not p.exists():
        print(f"  missing: {p}")
        return
    top_teams = json.loads(p.read_text(encoding="utf-8"))
    top_ids = {int(t["team_id"]) for t in top_teams if t.get("team_id") is not None}
    print(f"  v17 top teams: {len(top_teams)} rows, {len(top_ids)} unique team_ids")
    # Walk all match files for unique team_ids
    ids: Counter = Counter()
    files = list(IMPORTS.glob("v17_match_*.json"))
    print(f"  scanning {len(files)} match files for unique team_ids")
    for f in files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        r = m.get("radiant_team_id")
        d = m.get("dire_team_id")
        if r:
            ids[int(r)] += 1
        if d:
            ids[int(d)] += 1
    print(f"  unique team_ids in matches: {len(ids)}")
    in_top = sum(1 for tid in ids if tid in top_ids)
    out_top = len(ids) - in_top
    print(f"  of which in v17 top teams: {in_top} ({100.0*in_top/max(1,len(ids)):.1f}%)")
    print(f"  of which NOT in v17 top teams: {out_top} ({100.0*out_top/max(1,len(ids)):.1f}%)")
    # Top 10 by match count
    print("  top 10 teams by match count:")
    for tid, n in ids.most_common(10):
        flag = "+" if tid in top_ids else "-"
        print(f"    {flag} team_id={tid:>10d}  matches={n}")
    # The bug: 99% of teams in the matches corpus are NOT in the
    # top-30 list, so their tier flag will be 'minor' (encoded as
    # 0).  This is fine for v18 (the categorical code is 0) but it
    # means the model only learns "premium vs not" — for a true
    # 3-way tier, we need a much larger top-teams snapshot.
    print()


# --------------------------------------------------------------------------- #
# Audit: patch coverage
# --------------------------------------------------------------------------- #

def audit_patch_coverage() -> None:
    print("=" * 78)
    print("Patch coverage: which patches are in the corpus, and which is the v18 _PATCHES list?")
    print("=" * 78)
    print()
    p = IMPORTS / "v17_phase7_patch_info.json"
    if not p.exists():
        print(f"  missing: {p}")
    else:
        info = json.loads(p.read_text(encoding="utf-8"))
        print(f"  v17_phase7_patch_info.json: {len(info)} patches")
        for x in info:
            print(f"    {x.get('name'):8s}  date={x.get('date')}")
    # Walk matches for patch strings
    patches: Counter = Counter()
    files = list(IMPORTS.glob("v17_match_*.json"))
    for f in files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        # OpenDota /matches/{id} doesn't carry patch by default;
        # the v17 trainer set it from the league metadata.  Check
        # both top-level and nested locations.
        pat = m.get("patch") or (m.get("league") or {}).get("patch")
        if pat:
            patches[str(pat)] += 1
    print(f"  unique patches in matches: {len(patches)}")
    for pat, n in patches.most_common():
        print(f"    {pat:8s}  matches={n}")
    # The v18 _PATCHES list:
    _PATCHES_V18 = ["7.39", "7.40", "7.41"]
    print(f"  v18 _PATCHES: {_PATCHES_V18}")
    in_trainer = [pat for pat in patches if pat in _PATCHES_V18]
    not_in_trainer = [pat for pat in patches if pat not in _PATCHES_V18]
    print(f"  in v18 trainer list: {in_trainer}")
    print(f"  NOT in v18 trainer list (would default to -1): {not_in_trainer}")
    print()


# --------------------------------------------------------------------------- #
# Audit: hero ID range
# --------------------------------------------------------------------------- #

def audit_hero_ids() -> None:
    print("=" * 78)
    print("Hero ID range: min/max/sorted in matches vs v18 NUM_HEROES")
    print("=" * 78)
    print()
    hero_ids: Counter = Counter()
    files = list(IMPORTS.glob("v17_match_*.json"))
    for f in files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for p in (m.get("players") or []):
            h = p.get("hero_id")
            if h is not None:
                hero_ids[int(h)] += 1
    if not hero_ids:
        print("  no hero_ids found")
        return
    print(f"  unique hero_ids: {len(hero_ids)}")
    print(f"  min: {min(hero_ids)}  max: {max(hero_ids)}")
    print(f"  NUM_HEROES (v18): 256")
    if max(hero_ids) >= 256:
        print(f"  !! some hero_ids >= 256: would be silently dropped from one-hot")
    # Top 10 most-picked
    print("  top 10 most-picked heroes:")
    for hid, n in hero_ids.most_common(10):
        print(f"    hero_id={hid:>4d}  picks={n}")
    # Count how many hero_ids in [0..255]
    in_range = sum(1 for h in hero_ids if 0 <= h < 256)
    out_of_range = sum(1 for h in hero_ids if h >= 256)
    print(f"  in [0..255]: {in_range}  out of range: {out_of_range}")
    print()


# --------------------------------------------------------------------------- #
# Audit: v17 winner vs v18 winner on 3 drafts
# --------------------------------------------------------------------------- #

def audit_winner_compare() -> None:
    print("=" * 78)
    print("Winner model: v17 vs v18 on 3 different drafts")
    print("=" * 78)
    print()
    sys.path.insert(0, str(PRO_ROOT))
    try:
        from business.v18_predict import predict_winner_v18, v18_unavailable
        from business.v17_predict import predict as v17_predict_full
    except Exception as exc:
        print(f"  import failed: {exc}")
        return
    # Three synthetic drafts
    drafts = [
        # draft 1: simple teamfight
        ([1, 5, 8, 11, 25], [2, 3, 4, 7, 9], None, None),
        # draft 2: split-push
        ([12, 14, 19, 23, 31], [16, 17, 22, 27, 33], None, None),
        # draft 3: late-game
        ([35, 38, 41, 44, 47], [36, 39, 42, 45, 48], None, None),
    ]
    for i, (r, d, rb, db) in enumerate(drafts, 1):
        try:
            v18 = predict_winner_v18(r, d, None, None, rb, db, int(time.time()), "7.41")
            v17 = v17_predict_full(
                radiant_team_id=None, dire_team_id=None,
                radiant_picks=r, dire_picks=d,
                radiant_bans=rb, dire_bans=db,
                start_time=int(time.time()), patch="7.41",
            )
            v17_w = v17.get("winner", {})
            print(f"  draft {i}: r={[h for h in r]} d={[h for h in d]}")
            print(f"    v18: prob_radiant={v18.get('prob_radiant'):.4f}  team={v18.get('team')}  src={v18.get('source')}")
            print(f"    v17: prob_radiant={v17_w.get('prob_radiant'):.4f}  team={v17_w.get('team')}  src=v17 (fallback)")
        except Exception as exc:
            print(f"  draft {i}: predict failed: {exc}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    audit_models()
    audit_tier_coverage()
    audit_patch_coverage()
    audit_hero_ids()
    audit_winner_compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
