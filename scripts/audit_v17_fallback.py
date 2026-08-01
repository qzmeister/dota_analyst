"""Verify the v18 -> v17 fallback chain.  Simulates the live card
calling v17_predict.predict() when v18 fails (model file missing,
joblib error, numpy error, etc.).  The fallback must produce
sensible output and not crash.
"""
import sys
import time
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))


def main():
    from business.v17_predict import predict, is_available

    # Sanity: v17 should be available
    assert is_available(), "v17 models not on disk"
    print("v17 models present")

    # Case 1: v18 available, normal predict
    print("\nCase 1: v18 available")
    out = predict(
        radiant_team_id=9878494, dire_team_id=9768243,
        radiant_picks=[1, 5, 8, 11, 25],
        dire_picks=[2, 3, 4, 7, 9],
        start_time=int(time.time()), patch="7.41",
    )
    print(f"  source: {out['source']}")
    print(f"  winner: {out['winner']}")
    print(f"  kills={out['kills']:.1f}  duration={out['duration_sec']:.0f}  first_15={out['first_15_kills']:.1f}")

    # Case 2: simulate v18 unavailable
    print("\nCase 2: simulate v18 unavailable (v17 fallback)")
    import business.v18_predict as v18
    v18._MODEL_CACHE = None
    v18._META_CACHE = None
    # Rename the model file to simulate it being deleted
    model_file = v18.MODELS_DIR / "_v18_winner" / "model.joblib"
    backup = v18.MODELS_DIR / "_v18_winner" / "model.joblib.bak.test"
    if model_file.exists():
        model_file.rename(backup)
        try:
            out = predict(
                radiant_team_id=9878494, dire_team_id=9768243,
                radiant_picks=[1, 5, 8, 11, 25],
                dire_picks=[2, 3, 4, 7, 9],
                start_time=int(time.time()), patch="7.41",
            )
            print(f"  source: {out['source']}")
            print(f"  winner: {out['winner']}")
        finally:
            backup.rename(model_file)
    else:
        print(f"  model file not found at {model_file}; skipping")

    # Case 3: simulate v18 throwing on predict (not load)
    print("\nCase 3: simulate v18 predict_proba throwing")
    from unittest.mock import patch
    import business.v18_predict as v18mod
    # Patch predict_winner_v18 to raise
    def boom(*a, **kw):
        raise RuntimeError("simulated v18 crash")
    with patch.object(v18mod, "predict_winner_v18", side_effect=boom):
        out = predict(
            radiant_team_id=9878494, dire_team_id=9768243,
            radiant_picks=[1, 5, 8, 11, 25],
            dire_picks=[2, 3, 4, 7, 9],
            start_time=int(time.time()), patch="7.41",
        )
        print(f"  source: {out['source']}")
        print(f"  winner: {out['winner']}")
    print("\nAll fallback cases work.")


if __name__ == "__main__":
    main()
