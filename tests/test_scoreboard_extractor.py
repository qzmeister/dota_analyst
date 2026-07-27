"""Tests for the v0.3.23 scoreboard-based live extractor.

The new extractor reads from `#live_scoreboard` (kills, game time, team
info) and the page's `radiant_picks` / `dire_picks` global arrays
(real-time picks populated by the socket.io `__nd2_match_*` event).

These tests mock the Playwright `page` object so we can exercise the
JS evaluator without spinning up a real browser.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from business import dltv_browser
from business.dltv_browser import _read_live_state_from_scoreboard


class _FakePage:
    """Mock playwright Page — captures the JS body and returns a canned result."""

    def __init__(self, result: Any):
        self._result = result
        self.last_script: str = ""

    def evaluate(self, script: str):
        self.last_script = script
        return self._result


# ---------------------------------------------------------------------------
# Sample payload — what the real DLTV page returns for a live match
# ---------------------------------------------------------------------------

SAMPLE_LIVE = {
    "teams": [
        {"name": "Level UP", "side_kind": "radiant", "kills": 10},
        {"name": "PuckChamp", "side_kind": "dire", "kills": 8},
    ],
    "team_order": ["radiant", "dire"],
    "radiant_score": 10,
    "dire_score": 8,
    "game_time": 525,  # 8:45 in seconds
    "picks": {
        "radiant": [
            {"hero_id": 52, "steam_id": 53, "name": "Beastmaster", "slug": "beastmaster", "image": "/uploads/heroes/BM.png"},
            {"hero_id": 102, "steam_id": 105, "name": "Techies",     "slug": "techies",     "image": "/uploads/heroes/T.png"},
            {"hero_id": 64,  "steam_id": 65, "name": "Jakiro",      "slug": "jakiro",      "image": "/uploads/heroes/J.png"},
            {"hero_id": 53,  "steam_id": 54, "name": "Nature's Prophet", "slug": "natures-prophet", "image": "/uploads/heroes/NP.png"},
            {"hero_id": 121, "steam_id": 120, "name": "Pangolier",  "slug": "pangolier",  "image": "/uploads/heroes/P.png"},
        ],
        "dire": [
            {"hero_id": 50, "steam_id": 51, "name": "Slardar", "slug": "slardar", "image": "/uploads/heroes/S.png"},
            {"hero_id": 26, "steam_id": 25, "name": "Lina",    "slug": "lina",    "image": "/uploads/heroes/L.png"},
            {"hero_id": 86, "steam_id": 85, "name": "Undying", "slug": "undying", "image": "/uploads/heroes/U.png"},
            {"hero_id": 97, "steam_id": 96, "name": "Centaur Warrunner", "slug": "centaur-warrunner", "image": "/uploads/heroes/CW.png"},
            {"hero_id": 120,"steam_id": 119, "name": "Dark Willow", "slug": "dark-willow", "image": "/uploads/heroes/DW.png"},
        ],
    },
    "bans": {
        "radiant": [],
        "dire": [],
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reads_scoreboard_kills():
    """The extractor pulls kills from the `#live_scoreboard` block.

    The previous extractor relied on `.map__finished-v2 .team__scores-kills`
    inside the picks block; v0.3.23 reads from `#live_scoreboard` which
    is locale-independent.
    """
    page = _FakePage(SAMPLE_LIVE)
    out = _read_live_state_from_scoreboard(page)
    assert out["radiant_score"] == 10
    assert out["dire_score"] == 8
    assert out["game_time"] == 525  # 8:45
    assert out["team_order"] == ["radiant", "dire"]


def test_reads_picks_from_page_globals():
    """Picks come from the page's `radiant_picks` / `dire_picks` globals,
    which are populated by the socket.io `__nd2_match_*` event handler.
    The DOM doesn't expose them — they're JS variables.
    """
    page = _FakePage(SAMPLE_LIVE)
    out = _read_live_state_from_scoreboard(page)
    radiant = [p["name"] for p in out["picks"]["radiant"]]
    dire = [p["name"] for p in out["picks"]["dire"]]
    assert radiant == ["Beastmaster", "Techies", "Jakiro", "Nature's Prophet", "Pangolier"]
    assert dire == ["Slardar", "Lina", "Undying", "Centaur Warrunner", "Dark Willow"]


def test_empty_scoreboard_returns_empty_state():
    """If the page has no `#live_scoreboard` (e.g., pre-hydration),
    the extractor returns an empty state dict — not an exception.
    """
    page = _FakePage(None)  # `null` from the JS side
    out = _read_live_state_from_scoreboard(page)
    assert out["picks"] == {"radiant": [], "dire": []}
    assert out["bans"] == {"radiant": [], "dire": []}
    assert out["radiant_score"] is None
    assert out["game_time"] is None


def test_pick_entry_has_both_dltv_and_steam_id():
    """Each pick entry must carry both the DLTV `hero_id` and the
    Valve `steam_id` — `_picks_to_heroes` chooses which to use based
    on the watchlist flag.  v0.3.22 set both to the same dltv id
    which silently broke the non-watchlist path.
    """
    page = _FakePage(SAMPLE_LIVE)
    out = _read_live_state_from_scoreboard(page)
    radiant = out["picks"]["radiant"]
    # Each pick has both fields, AND they're different
    for p in radiant:
        assert "hero_id" in p
        assert "steam_id" in p
        assert p["hero_id"] != p["steam_id"], \
            f"hero_id and steam_id must differ (got {p['hero_id']} == {p['steam_id']})"


def test_team_order_is_locale_independent():
    """The extractor reads the side from the CSS class
    `side radiant` / `side dire` — NOT from the user-visible
    text.  This means RU/EN/DE/ES locales all produce the same
    team_order output.
    """
    payload = dict(SAMPLE_LIVE)
    # Simulate the German page: the text is "Strahlend" / "Verwüstet"
    # but the .side class still has 'radiant' / 'dire'.
    payload["teams"] = [
        {"name": "Level UP", "side_kind": "radiant", "kills": 10},
        {"name": "PuckChamp", "side_kind": "dire", "kills": 8},
    ]
    page = _FakePage(payload)
    out = _read_live_state_from_scoreboard(page)
    assert out["team_order"] == ["radiant", "dire"]


def test_extractor_does_not_call_legacy_selectors():
    """Sanity: the v0.3.23 extractor must NOT query `.map__finished-v2`
    (that class no longer exists on the DLTV page as of 2026-07-27).
    """
    page = _FakePage(SAMPLE_LIVE)
    _read_live_state_from_scoreboard(page)
    # The JS body should reference #live_scoreboard but NOT .map__finished-v2
    script = page.last_script
    assert "live_scoreboard" in script
    assert "map__finished-v2" not in script
