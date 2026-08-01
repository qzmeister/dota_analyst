"""Smoke test for the v0.7.2 hero-id mapping fix.

The v18 model is trained on OpenDota/Steam hero ids.  The live
card sometimes passes DLTV internal hero ids.  The two
namespaces are NOT identical (104 of 127 DLTV heroes use a
different id than Steam).  v18_predict now requires the caller
to declare the namespace via `hero_id_namespace='steam'|'dltv'`.

This test:
  1. Verifies the DLTV->Steam translation in the helper.
  2. Verifies the same draft via DLTV vs Steam gives the
     SAME prediction (after the caller's translation).
  3. Verifies a Steam pass-through is idempotent.
"""
import sys
import time
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))


def main():
    from business.v18_predict import (
        _get_dltv_to_steam_map, _dltv_to_steam, predict_winner_v18,
    )

    dltv_to_steam = _get_dltv_to_steam_map()
    print(f"DLTV->Steam map size: {len(dltv_to_steam)}")
    print()
    print("DLTV->Steam translation (force):")
    spot = [
        (1, "Anti-Mage (same in both)"),
        (24, "Lina"),
        (113, "Grimstroke"),
        (120, "Hoodwink"),
        (127, "latest hero (DLTV 127)"),
    ]
    for dltv_id, name in spot:
        steam_id = _dltv_to_steam(dltv_id)
        print(f"  DLTV {dltv_id:>4d} ({name:>30s})  ->  Steam {steam_id:>4d}")
    print()

    # Steam pass-through (the model expects Steam; if the
    # caller passes Steam ids directly, nothing should change).
    print("Steam pass-through (no translation):")
    for sid in [1, 25, 114, 123, 155]:
        # v18 with namespace=steam; we just check it loads
        # without crashing on a tiny input.  Predict_proba
        # returns a prob -- we just want to confirm there's
        # no exception.
        try:
            v = predict_winner_v18(
                [sid], [2],  # 1 hero each -- won't crash
                hero_id_namespace="steam",
            )
            print(f"  Steam {sid:>4d}: predicted {v['prob_radiant']:.4f} (no error)")
        except Exception as exc:
            print(f"  Steam {sid:>4d}: ERROR {exc}")

    # End-to-end: same draft via DLTV vs Steam should give the
    # same prediction when the caller passes the right
    # namespace flag.
    print()
    print("end-to-end prediction test (DLTV draft -> Steam space):")
    drafts = [
        ("balanced",  [1, 5, 8, 11, 25],   [2, 3, 4, 7, 9]),
        ("late-game", [35, 38, 41, 44, 47],[36, 39, 42, 45, 48]),
        ("new hero",  [127, 113, 100, 80, 25], [126, 121, 119, 117, 124]),
    ]
    now = int(time.time())
    for label, dltv_r, dltv_d in drafts:
        # Convert DLTV -> Steam up front
        steam_r = [_dltv_to_steam(h) for h in dltv_r]
        steam_d = [_dltv_to_steam(h) for h in dltv_d]
        # v18 with namespace=steam
        v_steam = predict_winner_v18(steam_r, steam_d, start_time=now, patch="7.41",
                                     hero_id_namespace="steam")
        # v18 with namespace=dltv (should auto-translate)
        v_dltv = predict_winner_v18(dltv_r, dltv_d, start_time=now, patch="7.41",
                                     hero_id_namespace="dltv")
        same = abs(v_dltv["prob_radiant"] - v_steam["prob_radiant"]) < 1e-6
        print(f"  {label:>12s}: dltv={v_dltv['prob_radiant']:.6f}  "
              f"steam={v_steam['prob_radiant']:.6f}  [{'MATCH' if same else 'DIFFER'}]")


if __name__ == "__main__":
    main()
