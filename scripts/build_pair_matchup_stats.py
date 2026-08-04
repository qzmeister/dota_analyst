"""Pre-compute two v19 feature lookup tables from the corpus:

  1. hero pair synergy  — P(win | side=s, pair=(h1, h2))
                          key (side, frozenset({h1, h2})) -> (wins, games)

  2. lane matchup        — P(radiant wins | mid 1v1 / bot 2v2 / top 2v2)
                            key frozenset of 1/2/4 hero ids -> (wins, games)

Both target the v18 trainer's new v19 feature group.  Stored together
in `ml_data/imports/pair_matchup_stats.json`.  See:

  - scripts/train_v18.py::extract_features  (consumer)
  - v0.7.63 dev notes

Only OpenDota-format files with lane info are used for the lane
matchup table; the hero pair table uses all files (hero ids are
present everywhere).  All counts are kept raw; the trainer applies
Bayesian smoothing when it consumes them.

Run:  python scripts/build_pair_matchup_stats.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRO_ROOT = Path(r"C:\Users\artka\.minimax\workspace\dota_analyst")
CORPUS_GLOB = str(PRO_ROOT / "ml_data" / "full_matches" / "*.json")
OUT = PRO_ROOT / "ml_data" / "imports" / "pair_matchup_stats.json"

# v19 train/test split must match scripts/train_v18.py::build_dataset
# (sort by start_time, last 20% is test).  Without --train-only the
# cache includes the test set's own data and the model memorises it.
TEST_FRAC = 0.20

# OpenDota lane constants
LANE_TOP = 1
LANE_MID = 2
LANE_BOT = 3
# OpenDota lane_role constants (rough — varies by source)
ROLE_CARRY = 1
ROLE_SUPPORT = 2
ROLE_OFFLANE = 3
ROLE_JUNGLER = 4


def _players_by_lane(match: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """For an OpenDota match, return a dict of lane -> list-of-players
    per side.  Used for mid 1v1 / bot 2v2 / top 2v2 lookups.

    NOTE: OpenDota's `lane_role` is often missing or unreliable, so
    we relax the bot / top matching to "≥2 players with this lane
    value" rather than insisting on carry/support or offlane/jungler
    pairing.  This loses some precision but gets ~3000 matches per
    lane instead of ~0.
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        'r_mid': [], 'd_mid': [],
        'r_bot': [], 'd_bot': [],
        'r_top': [], 'd_top': [],
    }
    for p in (match.get('players') or []):
        hid = p.get('hero_id')
        if not isinstance(hid, int) or hid <= 0:
            continue
        lane = p.get('lane')
        side = 'r' if p.get('isRadiant') else 'd'
        if lane == LANE_MID:
            out[f'{side}_mid'].append(p)
        elif lane == LANE_BOT:
            out[f'{side}_bot'].append(p)
        elif lane == LANE_TOP:
            out[f'{side}_top'].append(p)
    return out


def _iter_pairs(heroes: List[int]) -> List[Tuple[int, int]]:
    """All C(n, 2) pairs (sorted, direction-insensitive)."""
    pairs = []
    h = sorted(set(int(x) for x in heroes if isinstance(x, int) and x > 0))
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            pairs.append((h[i], h[j]))
    return pairs


def _hero_ids_from_match(match: Dict[str, Any], side: str) -> List[int]:
    """Return the 5 hero ids on `side` for the OpenDota layout (players
    0-4 = radiant, 5-9 = dire by position).  Used for hero pair
    synergy so we don't need lane info.
    """
    players = match.get('players') or []
    heroes: List[int] = []
    for p in players:
        is_rad = bool(p.get('isRadiant'))
        if (side == 'radiant' and not is_rad) or (side == 'dire' and is_rad):
            continue
        h = p.get('hero_id')
        if isinstance(h, int) and h > 0:
            heroes.append(int(h))
        if len(heroes) == 5:
            break
    return heroes


def main(argv: list = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-only', action='store_true',
                    help='build cache only from the first 80%% of matches '
                         'by start_time (matches v18 trainer walk-forward '
                         'split, avoids test-set leakage)')
    args = ap.parse_args(argv)
    # Stash on the function so the file-iteration block above can read it
    # (cheaper than threading it through a dozen local variables).
    main._train_only = args.train_only  # type: ignore[attr-defined]

    files = sorted(glob.glob(CORPUS_GLOB))
    print(f"scanning {len(files)} files in {CORPUS_GLOB}")

    # Hero pair synergy (same side, stringified "side_h1_h2" key)
    pair_wins: Dict[str, int] = defaultdict(int)
    pair_totals: Dict[str, int] = defaultdict(int)

    # Lane matchup
    mid_wins: Dict[str, int] = defaultdict(int)
    mid_totals: Dict[str, int] = defaultdict(int)
    bot_wins: Dict[str, int] = defaultdict(int)
    bot_totals: Dict[str, int] = defaultdict(int)
    top_wins: Dict[str, int] = defaultdict(int)
    top_totals: Dict[str, int] = defaultdict(int)

    n = 0
    n_with_lanes = 0
    n_with_lane_matchups = 0

    # Optionally filter to the first (1 - TEST_FRAC) of matches by
    # start_time, matching the v18 trainer's walk-forward split.
    # Without this the test set's own data leaks into the cache and
    # the model memorises it (acc -> 1.0 on test).
    train_only = bool(getattr(main, '_train_only', False))
    if train_only:
        ts: list[tuple[int, str]] = []
        for fp in files:
            try:
                with open(fp, encoding='utf-8-sig') as f:
                    d = json.load(f)
            except Exception:
                continue
            st = d.get('start_time')
            if not (isinstance(st, (int, float)) and st > 0):
                sd = d.get('start_date')
                if isinstance(sd, (int, float)) and sd > 0:
                    st = sd // 1000 if sd > 10_000_000_000 else sd
                else:
                    continue
            ts.append((int(st), fp))
        ts.sort()
        n_keep = int(len(ts) * (1.0 - TEST_FRAC))
        files = [fp for _, fp in ts[:n_keep]]
        print(f'--train-only: kept {len(files)} / {len(ts)} matches '
              f'(first {int((1-TEST_FRAC)*100)}%% by start_time)')

    for fp in files:
        try:
            with open(fp, encoding='utf-8-sig') as f:
                d = json.load(f)
        except Exception:
            continue
        # winner: prefer radiant_win, fall back to radiant_victory
        rw = d.get('radiant_win')
        if rw is None:
            rw = d.get('radiant_victory')
        if rw is None:
            continue
        n += 1
        target = int(bool(rw))

        # Hero pair synergy (uses top-level players[] or normalized)
        for side in ('radiant', 'dire'):
            heroes = _hero_ids_from_match(d, side)
            won = target if side == 'radiant' else 1 - target
            for h1, h2 in _iter_pairs(heroes):
                key = f"{side[0]}_{min(h1,h2)}_{max(h1,h2)}"
                pair_wins[key] += won
                pair_totals[key] += 1

        # Lane matchup (only if OpenDota-format with lane info).
        # We use the relaxed lane-only approach (no role distinction)
        # because OpenDota `lane_role` is often missing.
        lanes = _players_by_lane(d)
        if any(lanes.values()):
            n_with_lanes += 1
            # mid 1v1 — first mid on each side
            if lanes['r_mid'] and lanes['d_mid']:
                key = f"m_{lanes['r_mid'][0]['hero_id']}_{lanes['d_mid'][0]['hero_id']}"
                mid_wins[key] += target
                mid_totals[key] += 1
                n_with_lane_matchups += 1
            # bot 2v2 — first 2 players in lane 3 on each side
            if len(lanes['r_bot']) >= 2 and len(lanes['d_bot']) >= 2:
                key = (f"b_{lanes['r_bot'][0]['hero_id']}_"
                       f"{lanes['r_bot'][1]['hero_id']}_"
                       f"{lanes['d_bot'][0]['hero_id']}_"
                       f"{lanes['d_bot'][1]['hero_id']}")
                bot_wins[key] += target
                bot_totals[key] += 1
            # top 2v2 — first 2 players in lane 1 on each side
            if len(lanes['r_top']) >= 2 and len(lanes['d_top']) >= 2:
                key = (f"t_{lanes['r_top'][0]['hero_id']}_"
                       f"{lanes['r_top'][1]['hero_id']}_"
                       f"{lanes['d_top'][0]['hero_id']}_"
                       f"{lanes['d_top'][1]['hero_id']}")
                top_wins[key] += target
                top_totals[key] += 1

    def _flatten(wins_d, totals_d) -> Dict[str, Dict[str, int]]:
        return {k: {'wins': wins_d[k], 'games': totals_d[k]}
                for k in totals_d}

    out = {
        '_meta': {
            'source_matches': n,
            'with_lane_info': n_with_lanes,
            'with_lane_matchups': n_with_lane_matchups,
            'pair_keys': len(pair_totals),
            'mid_keys': len(mid_totals),
            'bot_keys': len(bot_totals),
            'top_keys': len(top_totals),
            'train_only': bool(getattr(main, '_train_only', False)),
        },
        'pair_synergy': _flatten(pair_wins, pair_totals),
        'mid_matchup':  _flatten(mid_wins, mid_totals),
        'bot_matchup':  _flatten(bot_wins, bot_totals),
        'top_matchup':  _flatten(top_wins, top_totals),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding='utf-8')

    m = out['_meta']
    print(f"scanned: {m['source_matches']} matches "
          f"({m['with_lane_info']} with lane info)")
    print(f"  pair_synergy: {m['pair_keys']} unique pairs (same side)")
    print(f"  mid_matchup:  {m['mid_keys']} unique matchups")
    print(f"  bot_matchup:  {m['bot_keys']} unique matchups")
    print(f"  top_matchup:  {m['top_keys']} unique matchups")
    print(f"  -> {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
