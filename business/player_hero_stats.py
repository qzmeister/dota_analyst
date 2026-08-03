"""In-memory lookup of (player, hero) -> (games, wins) over the
full_matches corpus.

The data lives in `ml_data/imports/player_hero_stats.json` (built
by `scripts/build_player_hero_stats.py`).  We load it once at
module import; subsequent `get()` calls hit a plain dict and are
O(1).

Used by `business/board.py:_hero_card()` to enrich the live
hero card with a per-player-on-this-hero "N | X%" badge (DLTV
parity).  Coverage is best for pro players who show up a lot
in our 5k+ match corpus; pub/semi-pro players will see
"нет данных" until they accumulate corpus.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional, Tuple

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml_data", "imports", "player_hero_stats.json",
)

_lock = threading.Lock()
_loaded = False
# player_nickname (lowercased) -> { hero_id_str: { games, wins } }
_by_player: dict = {}


def _load_once() -> None:
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                blob = json.load(f)
        except FileNotFoundError:
            # Cache hasn't been built yet — degrade gracefully,
            # all lookups return None and the badge is hidden.
            _by_player.clear()
            _loaded = True
            return
        except Exception:
            _by_player.clear()
            _loaded = True
            return
        _by_player.clear()
        _by_player.update(blob.get("players") or {})
        _loaded = True


def get(player_nickname: Optional[str], hero_id: Optional[int]) -> Optional[Tuple[int, int, float]]:
    """Return (games, wins, win_rate) for this (player, hero) pair, or
    `None` if the player or hero is not in our corpus.

    `win_rate` is `wins / games` as a 0..1 float.  Callers may want
    to gate on `games >= MIN_GAMES` for a meaningful display value
    (the cache has single-game outliers).
    """
    if not player_nickname or hero_id is None or hero_id <= 0:
        return None
    _load_once()
    if not _by_player:
        return None
    nick = player_nickname.strip().lower()
    if not nick:
        return None
    heroes = _by_player.get(nick)
    if not heroes:
        return None
    entry = heroes.get(str(int(hero_id)))
    if not entry:
        return None
    games = int(entry.get("games", 0))
    wins = int(entry.get("wins", 0))
    if games <= 0:
        return None
    return games, wins, wins / games


def is_ready() -> bool:
    """True if the cache has been loaded (or attempted and missed).
    Useful for the API layer to surface a /api/player_hero_stats
    health probe.
    """
    _load_once()
    return bool(_by_player)


def stats_summary() -> dict:
    """Diagnostic summary for /api/healthz."""
    _load_once()
    total_pairs = sum(len(v) for v in _by_player.values())
    return {
        "loaded": bool(_by_player),
        "players": len(_by_player),
        "pairs": total_pairs,
    }
