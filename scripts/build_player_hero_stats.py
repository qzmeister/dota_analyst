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
import json
import glob
import os
from collections import defaultdict
from pathlib import Path

CORPUS_GLOB = r'C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\full_matches\*.json'
CACHE_OUT   = r'C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\imports\player_hero_stats.json'


def _iter_player_perfs(match: dict):
    """Yield (nickname_lower, hero_valve_id, did_win) for every
    player_performances entry in the match (both sides).
    """
    rv = bool(match.get('radiant_victory'))
    for side, won in (('radiant', rv), ('dire', not rv)):
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


def main() -> int:
    files = glob.glob(CORPUS_GLOB)
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
    raise SystemExit(main())
