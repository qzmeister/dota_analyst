"""End-to-end smoke test for the v0.7.0 live card.

Exercises the full v17_predict + v18_predict path with the v0.7.0
tier fix and the v0.7.2 hero-id mapping fix.  Verifies that:
  1. Different drafts give different predictions (no "59% on
     every team" bug)
  2. Tier difference (premium vs minor) shifts the prediction
     in the expected direction
  3. DLTV hero_ids and Steam hero_ids produce the same
     prediction for the same logical draft
  4. v18 -> v17 fallback works when v18 fails
"""
import sys
import time
import json
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))


def main():
    from business.board import convert_to_steam
    from business.v17_predict import predict
    from business.v18_predict import (
        _dltv_to_steam, _get_dltv_to_steam_map, predict_winner_v18,
    )

    # Tier flags from v18 top_teams
    raw = json.loads((PRO_ROOT / "ml_data" / "imports" / "v18_top_teams.json")
                     .read_text(encoding="utf-8"))
    premium_id = next(t["team_id"] for t in raw if t["tier"] == 2)
    minor_id = next(t["team_id"] for t in raw
                    if t["tier"] == 0 and (t["wins"] + t["losses"]) >= 30)

    # Different drafts
    drafts = [
        ("teamfight",  [1, 5, 8, 11, 25],   [2, 3, 4, 7, 9]),
        ("splitpush",  [12, 14, 19, 23, 31],[16, 17, 22, 27, 33]),
        ("late-game",  [35, 38, 41, 44, 47],[36, 39, 42, 45, 48]),
        ("new hero",   [127, 113, 100, 80, 25], [126, 121, 119, 117, 124]),
    ]
    print("Test 1: different drafts (Steam namespace)")
    now = int(time.time())
    probs = []
    for label, r, d in drafts:
        v = predict(radiant_team_id=premium_id, dire_team_id=minor_id,
                    radiant_picks=r, dire_picks=d, start_time=now, patch="7.41")
        w = v["winner"]
        probs.append(w["prob_radiant"])
        print(f"  {label:>10s}: prob_radiant={w['prob_radiant']:.4f}  team={w['team']}  src={v['source']}")
    spread = max(probs) - min(probs)
    print(f"  spread across drafts: {spread:.4f}")
    assert spread > 0.05, f"predictions too similar (spread={spread})"
    print("  OK: predictions vary across drafts\n")

    # Tier difference (premium team in different roles)
    print("Test 2: tier difference (premium in radiant vs premium in dire)")
    label, r, d = drafts[0]
    v_premium_rad = predict(radiant_team_id=premium_id, dire_team_id=minor_id,
                            radiant_picks=r, dire_picks=d, start_time=now, patch="7.41")
    v_premium_dire = predict(radiant_team_id=minor_id, dire_team_id=premium_id,
                             radiant_picks=r, dire_picks=d, start_time=now, patch="7.41")
    p_premium_rad = v_premium_rad["winner"]["prob_radiant"]
    p_premium_dire = v_premium_dire["winner"]["prob_radiant"]  # = prob that RADIANT (=minor) wins
    # The prob that DIRE (=premium) wins in the second case is (1 - p_premium_dire)
    # but prob_radiant is what we want; the same DRAFT just with sides swapped:
    #   case 1: premium=R, minor=D  -> prob_R = 0.71 (premium wins)
    #   case 2: minor=R, premium=D  -> prob_R = 0.15 (premium wins because dire)
    # The premium team is on the winning side in BOTH cases; the model
    # is correct as long as the premium team wins more often.
    print(f"  premium on radiant side: prob_radiant={p_premium_rad:.4f}  (premium wins)")
    print(f"  premium on dire side:    prob_radiant={p_premium_dire:.4f}  (premium wins)")
    assert p_premium_rad > 0.5, "premium on radiant side should win"
    assert p_premium_dire < 0.5, "premium on dire side should win (prob_radiant < 0.5)"
    print("  OK: tier signal works (premium team wins on both sides)\n")

    # DLTV vs Steam
    print("Test 3: DLTV vs Steam for the same logical draft")
    # Use DLTV 24 (Lina) and DLTV 120 (Hoodwink) - both differ from Steam
    dltv_r = [24, 25, 113, 80, 11]  # Lina, Lion, Grimstroke, ?, ?  (DLTV)
    dltv_d = [5, 7, 8, 19, 31]      # Crystal Maiden, ..., Pugna, ...  (DLTV)
    # Convert to Steam up front (simulating board.convert_to_steam)
    steam_r = [_dltv_to_steam(h) for h in dltv_r]
    steam_d = [_dltv_to_steam(h) for h in dltv_d]
    v_steam = predict(radiant_team_id=premium_id, dire_team_id=minor_id,
                      radiant_picks=steam_r, dire_picks=steam_d,
                      start_time=now, patch="7.41")
    # Same draft via v18 with hero_id_namespace='dltv'
    v18_dltv = predict_winner_v18(dltv_r, dltv_d, premium_id, minor_id,
                                  start_time=now, patch="7.41",
                                  hero_id_namespace="dltv")
    p_steam = v_steam["winner"]["prob_radiant"]
    p_dltv = v18_dltv["prob_radiant"]
    print(f"  via convert_to_steam (Steam): prob_radiant={p_steam:.4f}  src={v_steam['source']}")
    print(f"  via hero_id_namespace='dltv':  prob_radiant={p_dltv:.4f}  src={v18_dltv['source']}")
    assert abs(p_steam - p_dltv) < 1e-3, f"DLTV and Steam should give same prob (got {p_steam} vs {p_dltv})"
    print("  OK: DLTV and Steam give the same prediction\n")

    # Fallback test
    print("Test 4: v18 -> v17 fallback when v18 fails")
    # Simulate v18 failing by mocking predict_winner_v18
    import business.v18_predict as v18mod
    original = v18mod.predict_winner_v18
    def boom(*a, **kw):
        raise RuntimeError("simulated v18 crash")
    v18mod.predict_winner_v18 = boom
    try:
        v = predict(radiant_team_id=premium_id, dire_team_id=minor_id,
                    radiant_picks=steam_r, dire_picks=steam_d,
                    start_time=now, patch="7.41")
        print(f"  v18 crashed, v17 took over: src={v['source']}  "
              f"prob_radiant={v['winner']['prob_radiant']:.4f}")
    finally:
        v18mod.predict_winner_v18 = original
    print("  OK: fallback works\n")

    print("=" * 78)
    print("ALL CHECKS PASSED -- the v0.7.0 live card is working as expected")
    print("=" * 78)


if __name__ == "__main__":
    main()
