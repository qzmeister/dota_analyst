"""Add legacy ModelMetadata fields to v17 metadata.json files.

The v17 trainer uses a different metadata schema (target / config /
feature_columns).  The legacy `business/ml/storage.py` loads it via
`ModelMetadata.from_dict` which expects `name`, `version`,
`trained_at`, `sklearn_version`, `numpy_version`, `python_version`,
and optional `feature_names`, `n_features`, `metrics`, `train_data`,
`encoder`.

This script backfills the missing fields so `storage.load(target)`
doesn't crash on `KeyError: 'name'` when it picks up `winner_v17`
(the lexicographically latest version under `models/winner_v*/`).
"""
import json
import os
import time
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
MODELS = PRO_ROOT / "ml_data" / "models"

# Map v17 target name -> legacy engine name.  v17 uses
# 'kills_total' / 'duration_sec'; legacy uses 'kills' / 'duration_mean'.
TARGET_NAME = {
    "kills_total":     "kills",
    "duration_sec":    "duration_mean",
    "first_15_kills":  None,    # not in legacy KNOWN_TARGETS
    "winner":          "winner",
}

DEFAULTS = {
    "sklearn_version": "1.9.0",
    "numpy_version":   "2.5.1",
    "python_version":  "3.14.3",
}


def fix_one(path: Path) -> bool:
    if not path.exists():
        print(f"  skip: {path} missing")
        return False
    m = json.loads(path.read_text(encoding="utf-8"))
    target = m.get("target", "")
    legacy_name = TARGET_NAME.get(target)
    if legacy_name is None:
        print(f"  skip: {path} target={target} not in legacy KNOWN_TARGETS")
        return False
    m["name"] = legacy_name
    m["version"] = "17"
    m.setdefault("trained_at", int(time.time()))
    for k, v in DEFAULTS.items():
        m.setdefault(k, v)
    m.setdefault("feature_names", m.get("feature_columns", []))
    m.setdefault("n_features", len(m.get("feature_columns", [])))
    m.setdefault("metrics", m.get("metrics_honest", {}))
    m.setdefault("train_data", {})
    m.setdefault("encoder", {})
    path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  fixed: {path} name={legacy_name}")
    return True


def main() -> int:
    print("[fix_v17_metadata] backfilling v17 metadata with legacy fields...", flush=True)
    fixed = 0
    for d in sorted(MODELS.glob("*_v17")):
        if fix_one(d / "metadata.json"):
            fixed += 1
    print(f"[fix_v17_metadata] fixed {fixed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
