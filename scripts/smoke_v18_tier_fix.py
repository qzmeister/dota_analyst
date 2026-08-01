"""Smoke test: verify v18_predict.py picks up the new v18_top_teams.json
with the pre-computed `tier` field, and that predictions differ when
the radiant vs dire tiers differ.
"""
import json
import time
import sys
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))

from business.v18_predict import _load_top_teams, predict_winner_v18  # noqa


def main():
    top_teams = _load_top_teams()
    print(f"loaded {len(top_teams)} teams (cached)")

    # Show distribution by walking all teams (re-read JSON for the
    # per-row tier field — the cache only stores int tier).
    raw = json.loads((PRO_ROOT / "ml_data" / "imports" / "v18_top_teams.json").read_text(encoding="utf-8"))
    tiers = [0, 0, 0]
    for t in raw:
        tiers[int(t["tier"])] += 1
    print(f"raw tier distribution: minor={tiers[0]} pro={tiers[1]} prem={tiers[2]}")

    # Sanity: find a real premium team and a real minor team.
    premium = next(t for t in raw if t["tier"] == 2)
    minor = next(t for t in raw if t["tier"] == 0 and (t["wins"] + t["losses"]) >= 30)
    print(f"premium: id={premium['team_id']}  tier={premium['tier']}  rating={premium['rating']}  W={premium['wins']}  L={premium['losses']}")
    print(f"minor:   id={minor['team_id']}  tier={minor['tier']}  rating={minor['rating']}  W={minor['wins']}  L={minor['losses']}")

    now = int(time.time())
    drafts = [
        ("premium vs minor", premium["team_id"], minor["team_id"]),
        ("minor vs premium", minor["team_id"], premium["team_id"]),
        ("premium vs premium", premium["team_id"], raw[0]["team_id"]),
        ("minor vs minor", minor["team_id"], raw[-1]["team_id"]),
    ]
    print()
    for label, r_id, d_id in drafts:
        v = predict_winner_v18(
            radiant_picks=[1, 5, 8, 11, 25],
            dire_picks=[2, 3, 4, 7, 9],
            radiant_team_id=r_id,
            dire_team_id=d_id,
            start_time=now,
            patch="7.41",
        )
        print(f"  {label:>22s}  prob_radiant={v['prob_radiant']:.4f}  team={v['team']}")


if __name__ == "__main__":
    main()
