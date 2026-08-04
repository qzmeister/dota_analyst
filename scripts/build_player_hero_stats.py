"""Build (player, hero) -> (games, wins) stats from the full_matches corpus.

Each `ml_data/full_matches/{match_id}.json` carries per-side
`player_performances[].player.nickname` (and steam32) plus
`performance.hero.valve_id` (the steam hero id, e.g. 100 = Tusk)
and a top-level `radiant_victory` boolean.  From those we can
derive, for every (nickname, hero_valve_id) tuple we've seen:

    games  = total matches where this player played this hero
    wins   = games where their team won

Output goes to `ml_data/imports/player_hero_stats.json` and is loaded
by `business/player_hero_stats.py` at module import time.  The live
card's hero badge then shows "N | X%" (games | win-rate) like DLTV.

The cache is gitignored (`ml_data/**`) — it's a denormalised index
that can always be rebuilt from the source corpus, so we don't
version-control it.
"""
import argparse
import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

CORPUS_GLOB = r'C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\full_matches\*.json'
CACHE_OUT   = r'C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\imports\player_hero_stats.json'

# v19 train/test split must match scripts/train_v18.py::build_dataset
# (sort by start_time, last 20% is test).  Without --train-only the
# cache includes the test set's own data and the model memorises it.
TEST_FRAC = 0.20


def _iter_player_perfs(match: dict):
    """Yield (nickname_lower, hero_valve_id, did_win) for every
    player entry in the match.  Supports two corpus formats:

      A) Stratz — `radiant`/`dire` blocks with `player_performances[]`,
         where each pp has `player.nickname` and `performance.hero.valve_id`.
         Detected by the presence of `radiant_victory` (Stratz-specific).

      B) OpenDota — top-level `players[]` with `name`, `hero_id`,
         `isRadiant` flag, and the winner label is `radiant_win`.

    Normalised Stratz files (post `normalize_full_matches.py`) have
    BOTH schemas populated, so we explicitly skip (B) when (A) is
    present to avoid double-counting the same 10 players.
    """
    rv_stratz = bool(match.get('radiant_victory'))
    rv_opendota = bool(match.get('radiant_win'))

    if rv_stratz or (match.get('radiant') or {}).get('player_performances'):
        # Stratz / Stratz-normalised path.
        for side, won in (('radiant', rv_stratz), ('dire', not rv_stratz)):
            block = match.get(side) or {}
            for pp in (block.get('player_performances') or []):
                player = pp.get('player') or {}
                nick = (player.get('nickname') or '').strip().lower()
                if not nick:
                    continue
                perf = pp.get('performance') or {}
                hero = perf.get('hero') or {}
                hid = hero.get('valve_id')
                if not isinstance(hid, int) or hid <= 0:
                    # Some entries put the hero in a different shape
                    # (legacy corpus); skip rather than crash.
                    continue
                yield nick, int(hid), bool(won)
    else:
        # OpenDota path.  `players[]` is at the top level and uses
        # `isRadiant` to flag the side.  Winner label is `radiant_win`.
        for p in (match.get('players') or []):
            nick = (p.get('name') or p.get('personaname') or '').strip().lower()
            hid = p.get('hero_id')
            if not nick or not isinstance(hid, int) or hid <= 0:
                continue
            is_rad = bool(p.get('isRadiant'))
            won = rv_opendota if is_rad else (not rv_opendota)
            yield nick, int(hid), bool(won)


def _match_start_time(d: dict) -> int:
    """Return a sortable start_time (unix seconds) regardless of schema.
    Stratz-normalised files have start_time already in seconds; raw
    Stratz dumps have start_date in milliseconds; OpenDota has
    start_time in seconds.
    """
    st = d.get('start_time')
    if isinstance(st, (int, float)) and st > 0:
        return int(st)
    sd = d.get('start_date')
    if isinstance(sd, (int, float)) and sd > 0:
        # millis (Stratz) -> seconds
        return int(sd // 1000) if sd > 10_000_000_000 else int(sd)
    return 0


def main(argv: list = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-only', action='store_true',
                    help='build cache only from the first 80%% of matches '
                         'by start_time (matches v18 trainer walk-forward '
                         'split, avoids test-set leakage)')
    args = ap.parse_args(argv)

    files = glob.glob(CORPUS_GLOB)
    if args.train_only:
        # Sort files by start_time, drop the last 20% to match the
        # trainer's walk-forward split.  We re-parse each file once
        # for the timestamp, then again to count.
        ts: list[tuple[int, str]] = []
        for fp in files:
            try:
                with open(fp, encoding='utf-8-sig') as f:
                    d = json.load(f)
            except Exception:
                continue
            t = _match_start_time(d)
            if t > 0:
                ts.append((t, fp))
        ts.sort()
        n_keep = int(len(ts) * (1.0 - TEST_FRAC))
        files = [fp for _, fp in ts[:n_keep]]
        print(f'--train-only: kept {len(files)} / {len(ts)} matches '
              f'(first {int((1-TEST_FRAC)*100)}%% by start_time)')

    games: dict[tuple[str, int], int] = defaultdict(int)
    wins:  dict[tuple[str, int], int] = defaultdict(int)
    seen_matches = 0
    seen_players = 0
    for fp in files:
        try:
            with open(fp, encoding='utf-8-sig') as f:
                d = json.load(f)
        except Exception:
            continue
        seen_matches += 1
        for nick, hid, won in _iter_player_perfs(d):
            key = (nick, hid)
            games[key] += 1
            if won:
                wins[key] += 1

    # Flatten to a JSON-friendly structure.  Group by player first so
    # the file is human-readable when a human opens it.
    by_player: dict[str, dict] = defaultdict(dict)
    for (nick, hid), g in games.items():
        w = wins.get((nick, hid), 0)
        by_player[nick][str(hid)] = {'games': g, 'wins': w}

    # Stable output order: by total games desc, then alpha.
    out: dict = {
        '_meta': {
            'source_matches': seen_matches,
            'players':       len(by_player),
            'pairs':         sum(len(v) for v in by_player.values()),
            'train_only':    bool(args.train_only),
        },
        'players': dict(sorted(
            by_player.items(),
            key=lambda kv: (-sum(p['games'] for p in kv[1].values()), kv[0]),
        )),
    }

    Path(CACHE_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE_OUT).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    meta = out['_meta']
    print(f'wrote {meta["pairs"]} (player,hero) pairs across {meta["players"]} players '
          f'from {meta["source_matches"]} matches -> {CACHE_OUT}')

    # Sample top 8 players
    print('  Top players (by total games):')
    for nick, heroes in list(out['players'].items())[:8]:
        total = sum(p['games'] for p in heroes.values())
        w = sum(p['wins'] for p in heroes.values())
        wr = (w / total * 100) if total else 0
        print(f'  {nick:<24} {total:>4} games  {wr:>5.1f}% WR  ({len(heroes)} heroes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
