"""Verify v17_predict.py fallback still works (it should NOT be
affected by the v18 tier fix because v17 has its own _load_top_teams
that reads only v17_phase1_top_teams.json).
"""
import sys
import time
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))

from business.v17_predict import build_features, _load_model, _encode_features  # noqa


def main():
    feats = build_features(
        radiant_team_id=9572001,  # top team from v17 list
        dire_team_id=9823272,     # second top team
        radiant_picks=[1, 5, 8, 11, 25],
        dire_picks=[2, 3, 4, 7, 9],
        start_time=int(time.time()),
        patch="7.41",
    )
    print("v17 features (fallback only, not v18):")
    print(f"  r_tier={feats['r_tier']}  d_tier={feats['d_tier']}")
    print(f"  r_top_team={feats['r_top_team']}  d_top_team={feats['d_top_team']}")
    # These should still come from the v17 30-team snapshot,
    # NOT the v18 540-team one.  Both 9572001 and 9823272 are in
    # the v17 top list, so r_tier and d_tier should be 'premium'.

    # Also verify v18 uses new file:
    from business.v18_predict import _load_top_teams
    top_teams = _load_top_teams()
    v18_r_tier = top_teams.get(9572001, 0)
    v18_d_tier = top_teams.get(9823272, 0)
    print(f"\nv18 tier lookup (new 540-team file):")
    print(f"  v18_r_tier={v18_r_tier}  v18_d_tier={v18_d_tier}")


if __name__ == "__main__":
    main()
