"""Build leagues cache from v17_match corpus.

When DLTV /api/v1/events is blocked (Cloudflare, network), /api/leagues
returns empty.  As a fallback, scan the local v17_match_*.json corpus
which carries the full `league` block (id, name, tier) per match and
emit a `leagues_cache.json` next to the v17 corpus.

The cache is read by `business/dltv_client.py` (or directly by
`/api/leagues` in `business/app.py`) when the live DLTV response is
empty.  The list is augmented with `match_count` from the live
auto-board so the picker can still rank by activity.
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

CORPUS_GLOB = r'C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\imports\v17_match_*.json'
CACHE_OUT   = r'C:\Users\artka\.minimax\workspace\dota_analyst\ml_data\imports\leagues_cache.json'


def _tier_is_active(tier: str | None) -> bool:
    """DLTV's `is_active` field.  We use a positive match on tier
    instead — the corpus doesn't carry `is_active` directly.
    """
    if not tier:
        return False
    tier = tier.lower()
    # 'excluded' is the bucket DLTV uses for amateur/third-party
    # cups.  We DO want to keep them in the picker so the user can
    # still filter by them (the board just doesn't show steam-only
    # cards that map to them).  Match the DLTV rule: any tier except
    # 'excluded' is active.
    return tier != 'excluded'


def main() -> int:
    files = glob.glob(CORPUS_GLOB)
    by_lid: dict[int, dict] = {}
    match_counts: dict[int, int] = defaultdict(int)
    for fp in files:
        try:
            d = json.load(open(fp, encoding='utf-8-sig'))
        except Exception:
            continue
        lg = d.get('league') or {}
        lid = lg.get('leagueid')
        name = lg.get('name')
        tier = lg.get('tier')
        if not lid or not name:
            continue
        match_counts[int(lid)] += 1
        if int(lid) not in by_lid:
            by_lid[int(lid)] = {
                'id': int(lid),
                'name': name,
                'tier': tier,
            }

    out = []
    for lid, info in by_lid.items():
        out.append({
            'id': info['id'],
            'title': info['name'],
            'tier': info['tier'],
            'is_active': _tier_is_active(info['tier']),
            'match_count': match_counts.get(info['id'], 0),
        })
    out.sort(key=lambda L: (-L['match_count'], L['title'].lower()))

    Path(CACHE_OUT).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'wrote {len(out)} leagues -> {CACHE_OUT}')
    for L in out[:8]:
        print(f"  {L['id']:>6} {L['tier'] or '?':<12} {L['match_count']:>4}  {L['title'][:50]}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
