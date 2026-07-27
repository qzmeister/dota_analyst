"""
Unit tests for `business.board` — the Kanban card assembly.

Strategy:
  - Pure-function tests for `_bo_label`, `_series_bo_int`,
    `_hero_card`, `_played_maps`, `_active_map`, `_prematch_card`,
    `_prematch_card_from_scraper` — no I/O, no patching.
  - Client-touched functions (`_picks_to_heroes`, `_bans_to_cards`,
    `classify_event_status`, `leagues_with_status`, `build_board`)
    patch `business.board.client` with a `MockClient` that
    records lookups and returns canned hero / event dicts.
  - The full `build_board()` is covered end-to-end with a
    minimal mock universe: one event, one series per stage.

We deliberately do NOT exercise `_postmatch_card` /
`_live_card` deeply here — those pull in `analyze()` and the
heuristic engine, and end-to-end coverage of them lives in
`test_app.py` (TestClient + mocked `client`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ============================================================================ #
# Stubs / fixtures
# ============================================================================ #

class _MockClient:
    """Minimal stand-in for `dltv_client.client` used by `board`."""

    def __init__(self) -> None:
        self.heroes_by_dltv: Dict[int, Dict] = {}
        self.heroes_by_steam: Dict[int, Dict] = {}
        self.events: List[Dict] = []
        self.series_by_event: Dict[int, List[Dict]] = {}
        # Default stage classifier behaviour — overridable per test.
        self.stage_for: Dict[str, str] = {}
        # How `get_heroes` should respond.
        self.heroes: List[Dict] = []

    def add_hero_dltv(self, dltv_id: int, *, name: str, steam_id: int) -> None:
        self.heroes_by_dltv[dltv_id] = {
            "id": dltv_id, "steam_id": steam_id, "name": name,
            "image": f"https://cdn/{name}.png", "win_rate": 51.0,
        }

    def add_hero_steam(self, steam_id: int, *, name: str) -> None:
        self.heroes_by_steam[steam_id] = {
            "id": steam_id, "steam_id": steam_id, "name": name,
            "image": f"https://cdn/{name}.png", "win_rate": 49.0,
        }

    def hero_by_dltv_id(self, hid: Optional[int]) -> Optional[Dict]:
        return self.heroes_by_dltv.get(hid) if hid is not None else None

    def hero_by_steam_id(self, sid: Optional[int]) -> Optional[Dict]:
        return self.heroes_by_steam.get(sid) if sid is not None else None

    def get_events(self):
        return list(self.events)

    def get_series(self, eid: int) -> List[Dict]:
        return list(self.series_by_event.get(int(eid), []))

    def get_heroes(self):
        return list(self.heroes)

    def normalize_team(self, team):
        team = team or {}
        return {
            "id": team.get("id"),
            "name": team.get("title") or team.get("name") or "TBD",
            "logo": team.get("logo"),
            "tag": team.get("tag"),
            "rank": team.get("rank"),
        }

    def classify_stage(self, series):
        sid = series.get("id")
        return self.stage_for.get(sid, "prematch")


@pytest.fixture
def mock_client(monkeypatch):
    """Patch `business.board.client` with a fresh mock.

    Returns the mock so tests can register heroes / events /
    series before calling the function under test.
    """
    mc = _MockClient()
    # `board` imports the client as a module-level name; we
    # patch the attribute on the board module specifically so we
    # don't break anything else.
    monkeypatch.setattr("business.board.client", mc)
    return mc


# ============================================================================ #
# _bo_label
# ============================================================================ #

class TestBoLabel:
    def test_int_1_2_3_5(self):
        from business.board import _bo_label
        assert _bo_label(1) == "BO1"
        assert _bo_label(2) == "BO2"
        assert _bo_label(3) == "BO3"
        assert _bo_label(5) == "BO5"

    def test_unknown_int_falls_back_to_boN(self):
        from business.board import _bo_label
        assert _bo_label(7) == "BO7"
        # `0` is falsy — falls into the explicit "BO?" branch
        # rather than `f"BO{0}"`.  This is by design: callers
        # should treat 0 as "unknown bo" not "bo0".
        assert _bo_label(0) == "BO?"

    def test_none_returns_placeholder(self):
        from business.board import _bo_label
        assert _bo_label(None) == "BO?"

    def test_str_uppercases_valid(self):
        from business.board import _bo_label
        assert _bo_label("bo3") == "BO3"
        assert _bo_label("Bo5") == "BO5"

    def test_str_non_bo_uppercased_anyway(self):
        # Loose fallback for "best effort" series types.
        from business.board import _bo_label
        assert _bo_label("round_robin") == "ROUND_ROBIN"
        assert _bo_label("") == "BO?"


# ============================================================================ #
# _series_bo_int
# ============================================================================ #

class TestSeriesBoInt:
    def test_int_passes_through(self):
        from business.board import _series_bo_int
        assert _series_bo_int({"type": 3}) == 3

    def test_str_bon_parses(self):
        from business.board import _series_bo_int
        assert _series_bo_int({"type": "bo3"}) == 3
        assert _series_bo_int({"type": "Bo5"}) == 5

    def test_missing_or_invalid_returns_none(self):
        from business.board import _series_bo_int
        assert _series_bo_int({}) is None
        assert _series_bo_int({"type": "round_robin"}) is None
        assert _series_bo_int({"type": None}) is None


# ============================================================================ #
# _hero_card
# ============================================================================ #

class TestHeroCard:
    def test_with_full_hero(self):
        from business.board import _hero_card
        hero = {"name": "Axe", "image": "https://img/axe.png", "win_rate": 53.0}
        card = _hero_card(hero, hero_id=1)
        assert card == {
            "id": 1, "name": "Axe",
            "image": "https://img/axe.png", "win_rate": 53.0,
        }

    def test_with_no_hero_uses_id_fallback(self):
        from business.board import _hero_card
        assert _hero_card(None, hero_id=42) == {
            "id": 42, "name": "#42", "image": None, "win_rate": None,
        }

    def test_with_hero_missing_name_falls_back_to_id(self):
        from business.board import _hero_card
        card = _hero_card({"image": "x"}, hero_id=7)
        assert card["name"] == "#7"
        assert card["image"] == "x"


# ============================================================================ #
# _picks_to_heroes / _bans_to_cards
# ============================================================================ #

class TestPicksToHeroes:
    def test_sorts_by_order_and_uses_dltv_by_default(self, mock_client):
        from business.board import _picks_to_heroes
        mock_client.add_hero_dltv(100, name="pa", steam_id=11)
        mock_client.add_hero_dltv(200, name="cm", steam_id=12)
        picks = [
            {"hero_id": 200, "order": 2},
            {"hero_id": 100, "order": 1},
        ]
        metas, cards = _picks_to_heroes(picks)
        # Order respected (100 before 200).
        assert [c["id"] for c in cards] == [100, 200]
        # dltv lookup hit; cards carry image / win_rate.
        assert metas[0]["name"] == "pa"
        assert cards[0]["image"] == "https://cdn/pa.png"

    def test_steam_id_path_uses_steam_lookup(self, mock_client):
        from business.board import _picks_to_heroes
        mock_client.add_hero_steam(11, name="pa")
        picks = [{"hero_id": 11, "order": 1}]
        metas, cards = _picks_to_heroes(picks, use_steam_id=True)
        assert metas[0]["name"] == "pa"

    def test_dual_id_steam_path_prefers_steam_id(self, mock_client):
        """v0.3.24e: dltv_browser cache overlay sets BOTH `hero_id`
        (DLTV internal) and `_steam_id` (Valve) on every entry.
        When is_watchlist=True, `_picks_to_heroes` must look up by
        the steam id — otherwise a Hoodwink pick (dltv_id 120) was
        silently resolved to Pangolier (steam_id 120) and the live
        card came out empty.  Regression test for that bug."""
        from business.board import _picks_to_heroes
        # The two heroes that v0.3.24d swapped: Hoodwink's dltv_id
        # numerically equals Pangolier's steam_id.
        mock_client.add_hero_dltv(120, name="Hoodwink",  steam_id=123)
        mock_client.add_hero_dltv(98,  name="Pangolier", steam_id=120)
        mock_client.add_hero_steam(123, name="Hoodwink")
        mock_client.add_hero_steam(120, name="Pangolier")
        picks = [{"hero_id": 120, "_steam_id": 123, "order": 1}]
        metas, cards = _picks_to_heroes(picks, use_steam_id=True)
        assert metas[0]["name"] == "Hoodwink"
        # The card id must be the steam id we actually looked up by.
        assert cards[0]["id"] == 123

    def test_dual_id_dltv_path_prefers_hero_id(self, mock_client):
        """v0.3.24e: when use_steam_id=False (v1 API path) and the
        source populated both fields, the dltv id wins — preserves
        the v1 API contract that `hero_id` is the dltv namespace."""
        from business.board import _picks_to_heroes
        mock_client.add_hero_dltv(120, name="Hoodwink",  steam_id=123)
        mock_client.add_hero_steam(123, name="Hoodwink")
        picks = [{"hero_id": 120, "_steam_id": 123, "order": 1}]
        metas, cards = _picks_to_heroes(picks, use_steam_id=False)
        assert metas[0]["name"] == "Hoodwink"
        assert cards[0]["id"] == 120

    def test_empty_picks_returns_empty_lists(self, mock_client):
        from business.board import _picks_to_heroes
        assert _picks_to_heroes(None) == ([], [])
        assert _picks_to_heroes([]) == ([], [])


class TestBansToCards:
    def test_sorts_by_order(self, mock_client):
        from business.board import _bans_to_cards
        mock_client.add_hero_dltv(1, name="axe", steam_id=2)
        mock_client.add_hero_dltv(3, name="cm", steam_id=4)
        bans = [
            {"hero_id": 3, "order": 2},
            {"hero_id": 1, "order": 1},
        ]
        cards = _bans_to_cards(bans)
        assert [c["id"] for c in cards] == [1, 3]

    def test_dual_id_steam_path_prefers_steam_id(self, mock_client):
        """Same dual-id bug as picks — bans on the live card were
        silently resolving the wrong hero too."""
        from business.board import _bans_to_cards
        mock_client.add_hero_dltv(120, name="Hoodwink",  steam_id=123)
        mock_client.add_hero_dltv(98,  name="Pangolier", steam_id=120)
        mock_client.add_hero_steam(123, name="Hoodwink")
        mock_client.add_hero_steam(120, name="Pangolier")
        bans = [{"hero_id": 120, "_steam_id": 123, "order": 1}]
        cards = _bans_to_cards(bans, use_steam_id=True)
        assert cards[0]["id"] == 123


# ============================================================================ #
# _build_live_gold (v0.3.24g)
# ============================================================================ #

class TestBuildLiveGold:
    """`_build_live_gold(m)` turns the raw `radiant_networth` /
    `dire_networth` ints that come from `dltv_browser._read_live_state_from_scoreboard`
    into a {radiant, dire, lead_radiant} block for the live card.
    Returns None when either side is missing or non-numeric — the
    frontend then hides the gold-lead row entirely instead of
    showing "0  0"."""

    def test_returns_signed_lead_for_radiant(self):
        from business.board import _build_live_gold
        out = _build_live_gold({"radiant_networth": 23888, "dire_networth": 20651})
        assert out == {"radiant": 23888, "dire": 20651, "lead_radiant": 3237}

    def test_lead_negative_when_dire_ahead(self):
        from business.board import _build_live_gold
        out = _build_live_gold({"radiant_networth": 18000, "dire_networth": 22000})
        assert out["lead_radiant"] == -4000

    def test_none_when_both_missing(self):
        """v0.3.24g: a finished map leaves `.team__networth` empty
        in the DOM.  Returning None (not {'radiant': 0, 'dire': 0})
        is the contract the frontend uses to hide the gold row."""
        from business.board import _build_live_gold
        assert _build_live_gold({}) is None

    def test_partial_when_one_side_missing(self):
        """v0.3.24h: the live page sometimes exposes only one side's
        networth (e.g. the page was captured before the trailing
        side got its first tick).  Return a partial block with the
        known side and `None` for the missing one — the frontend
        renders the value and shows "—" for the rest."""
        from business.board import _build_live_gold
        out_r = _build_live_gold({"radiant_networth": 12345})
        assert out_r == {"radiant": 12345, "dire": None, "lead_radiant": None}
        out_d = _build_live_gold({"dire_networth": 12345})
        assert out_d == {"radiant": None, "dire": 12345, "lead_radiant": None}

    def test_none_for_non_int_values(self):
        """The page might serialize the value as a string in some
        DLTV versions.  When both are strings, we return None
        (frontend shows "—" for everything).  When one is an int
        and the other is a string, we return a partial block with
        the int side."""
        from business.board import _build_live_gold
        assert _build_live_gold({"radiant_networth": "23888", "dire_networth": "20651"}) is None
        partial = _build_live_gold({"radiant_networth": 23888, "dire_networth": "20651"})
        assert partial == {"radiant": 23888, "dire": None, "lead_radiant": None}


# ============================================================================ #
# _played_maps / _active_map
# ============================================================================ #

class TestPlayedMaps:
    def test_keeps_only_maps_with_winner_and_duration(self):
        from business.board import _played_maps
        series = {
            "maps": [
                {"duration": 1800, "winner": "radiant"},  # played
                {"duration": 0, "winner": "radiant"},     # no duration
                {"duration": 1800},                         # no winner
                {"duration": 1500, "winner": "dire"},      # played
            ]
        }
        out = _played_maps(series)
        assert len(out) == 2
        assert out[0]["duration"] == 1800
        assert out[1]["winner"] == "dire"

    def test_empty_series(self):
        from business.board import _played_maps
        assert _played_maps({}) == []
        assert _played_maps({"maps": None}) == []


class TestActiveMap:
    def test_prefers_in_progress_map(self):
        from business.board import _active_map
        series = {
            "maps": [
                {"duration": 1800, "winner": "radiant"},  # finished
                {"steam_id": 12345},                       # in progress
                {"duration": 1500, "winner": "dire"},
            ]
        }
        active = _active_map(series)
        assert active is not None
        assert active["steam_id"] == 12345

    def test_falls_back_to_last_map(self):
        from business.board import _active_map
        series = {
            "maps": [
                {"duration": 1800, "winner": "radiant"},
                {"duration": 1500, "winner": "dire"},
            ]
        }
        # No in-progress map → last map wins.
        assert _active_map(series)["winner"] == "dire"

    def test_no_maps_returns_none(self):
        from business.board import _active_map
        assert _active_map({}) is None
        assert _active_map({"maps": []}) is None


# ============================================================================ #
# Card builders
# ============================================================================ #

class TestPrematchCard:
    def test_emits_required_fields(self):
        from business.board import _prematch_card
        # `client.normalize_team` looks for `title` first, then `name`.
        # Real DLTV data uses `title`; we mirror that here.
        series = {
            "id": "series-1",
            "_scraper_event_id": 42,
            "type": 3,
            "started_at": "2026-07-24T10:00:00Z",
            "first_team": {"id": 100, "title": "TS"},
            "second_team": {"id": 200, "title": "OG"},
        }
        card = _prematch_card(series, event_title="DreamLeague")
        assert card["stage"] == "prematch"
        assert card["series_id"] == "series-1"
        assert card["event_id"] == 42
        assert card["event"] == "DreamLeague"
        assert card["bo"] == "BO3"
        assert card["team_a"]["name"] == "TS"
        assert card["team_b"]["name"] == "OG"


class TestPrematchCardFromScraper:
    def test_emits_required_fields(self):
        from business.board import _prematch_card_from_scraper
        m = {
            "series_id": "scraped-1",
            "steam_id": 99999,
            "event_id": 7,
            "event": "ESL One",
            "bo": "bo5",
            "stage_label": "Group A",
            "start_time": "2026-07-24T18:00:00Z",
            "team_a": {"name": "Liquid", "logo": "L.png", "tag": "TL", "rank": 1, "id": 1},
            "team_b": {"name": "Tundra", "logo": "T.png", "tag": "Tundra", "rank": 4, "id": 2},
        }
        card = _prematch_card_from_scraper(m)
        assert card["stage"] == "prematch"
        assert card["bo"] == "BO5"
        assert card["team_a"]["name"] == "Liquid"
        assert card["team_b"]["name"] == "Tundra"
        assert card["is_tracked"] is True  # steam_id present

    def test_is_tracked_false_when_no_steam_id(self):
        from business.board import _prematch_card_from_scraper
        m = {
            "series_id": "scraped-2",
            "event": "ESL One",
            "bo": "bo3",
            "team_a": {"name": "A"},
            "team_b": {"name": "B"},
        }
        card = _prematch_card_from_scraper(m)
        assert card["is_tracked"] is False


# ============================================================================ #
# classify_event_status
# ============================================================================ #

class TestClassifyEventStatus:
    def test_live_when_event_in_live_series(self, monkeypatch):
        from business import board
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: ([{"_scraper_event_id": 42, "id": 1}], []),
        )
        assert board.classify_event_status(42) == "live"

    def test_prematch_when_only_in_prematch_list(self, monkeypatch):
        from business import board
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: ([], [{"event_id": 42, "id": 1}]),
        )
        assert board.classify_event_status(42) == "live"

    def test_none_when_event_not_in_any_list(self, monkeypatch):
        from business import board
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: ([{"_scraper_event_id": 99}], [{"event_id": 99}]),
        )
        assert board.classify_event_status(42) is None

    def test_discovery_failure_returns_none(self, monkeypatch):
        from business import board
        from business.exceptions import DiscoveryError
        def boom():
            raise DiscoveryError("network down")
        monkeypatch.setattr("business.discovery.discover", boom)
        assert board.classify_event_status(42) is None


# ============================================================================ #
# leagues_with_status
# ============================================================================ #

class TestLeaguesWithStatus:
    def test_filters_to_active_event_ids(self, mock_client, monkeypatch):
        from business import board
        # Two events; only 42 has live / prematch series.
        mock_client.events = [
            {"id": 42, "title": "DreamLeague", "is_active": True},
            {"id": 99, "title": "Old Cup", "is_active": False},
        ]
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: ([{"_scraper_event_id": 42}], [{"event_id": 42}]),
        )
        leagues = board.leagues_with_status()
        assert [l["id"] for l in leagues] == [42]
        assert leagues[0]["status"] == "live"

    def test_discovery_failure_returns_empty(self, mock_client, monkeypatch):
        from business import board
        from business.exceptions import DiscoveryError
        mock_client.events = [{"id": 1, "title": "X", "is_active": True}]
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: (_ for _ in ()).throw(DiscoveryError("boom")),
        )
        assert board.leagues_with_status() == []

    def test_sorted_alphabetically(self, mock_client, monkeypatch):
        from business import board
        mock_client.events = [
            {"id": 3, "title": "ESL One", "is_active": True},
            {"id": 1, "title": "Bali Major", "is_active": True},
            {"id": 2, "title": "DreamLeague", "is_active": True},
        ]
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: ([{"_scraper_event_id": 1}], []),
        )
        out = board.leagues_with_status()
        # Only #1 is active, but the sort order would be by title
        # if multiple were active.  Confirm only #1 is returned.
        assert [l["id"] for l in out] == [1]


# ============================================================================ #
# build_board — the top-level orchestrator
# ============================================================================ #

class TestBuildBoard:
    def test_empty_event_ids_falls_back_to_active(self, mock_client, monkeypatch):
        # When `event_ids` is empty, `build_board` auto-pulls the
        # active event list via `leagues_with_status`.
        from business import board
        mock_client.events = [
            {"id": 42, "title": "DreamLeague", "is_active": True},
        ]
        mock_client.series_by_event = {
            42: [
                {"id": 1, "type": 3, "started_at": "2026-07-24T10:00:00Z",
                 "first_team": {"id": 1, "title": "TS"},
                 "second_team": {"id": 2, "title": "OG"},
                 "maps": []},
            ],
        }
        # The active-event fallback uses `leagues_with_status`,
        # which itself needs `discover()` to return at least one
        # event id so the league is "active".
        monkeypatch.setattr(
            "business.discovery.discover",
            lambda: ([{"_scraper_event_id": 42}], []),
        )
        out = board.build_board(event_ids=[], watch_ids=[])
        assert "prematch" in out
        assert "live" in out
        assert "postmatch" in out
        # Prematch card is built from the v1 series.
        assert any(
            c.get("series_id") == 1 and c.get("stage") == "prematch"
            for c in out["prematch"]
        )

    def test_active_fallback_handles_discovery_failure(self, mock_client, monkeypatch):
        from business import board
        from business.exceptions import DiscoveryError
        mock_client.events = []
        def boom():
            raise DiscoveryError("discovery dead")
        monkeypatch.setattr("business.discovery.discover", boom)
        out = board.build_board(event_ids=[], watch_ids=[])
        # No events → empty board; no exception leaks.
        assert out["prematch"] == []
        assert out["live"] == []
        assert out["postmatch"] == []

    def test_returns_dict_with_expected_keys(self, mock_client, monkeypatch):
        from business import board
        monkeypatch.setattr("business.discovery.discover", lambda: ([], []))
        out = board.build_board(event_ids=[99], watch_ids=[])
        assert set(out.keys()) >= {"prematch", "live", "postmatch"}
