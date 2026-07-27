"""v0.3.22 cont 4: server-side filter on the auto-board.

The publisher rebuilds the unfiltered board every 5s.  `/api/board`
now applies the user's `events=` selection in-memory instead of
triggering a full rebuild, which was the cause of the 25s timeouts
the user kept seeing.

These tests cover the helper directly — they don't go through HTTP.
"""
from __future__ import annotations

import pytest

from business.app import _filter_auto_board


def _card(event_id, match_id=None, league="L", team_a="A", team_b="B"):
    return {
        "event_id": event_id,
        "event": league,
        "match_id": match_id,
        "team_a": {"name": team_a},
        "team_b": {"name": team_b},
    }


AUTO = {
    "prematch": [
        _card(6617, match_id=None, league="EPL Masters 1"),
        _card(6620, match_id=None, league="Lunar Horse Trophy 8"),
        _card(6626, match_id=None, league="BetBoom Streamers Battle 14"),
        _card(None,  match_id=None, league="Steam league 19479"),
    ],
    "live": [
        _card(6620, match_id=None, league="Lunar Horse Trophy 8"),
        _card(None,  match_id=None, league="Steam league 19479"),
        _card(None,  match_id=None, league="Steam league 18867"),
    ],
    "postmatch": [
        _card(6617, match_id=None, league="EPL Masters 1"),
        _card(6626, match_id=None, league="BetBoom Streamers Battle 14"),
        _card(None,  match_id=None, league="Steam league 17599"),
    ],
    "engine": "ml",
}


def test_unfiltered_returns_auto_unchanged():
    """No `events=` and no `watch=` — return the auto-board as-is."""
    out = _filter_auto_board(AUTO, [], [])
    # When no filter, we return a copy of the auto-board (without the
    # `filtered_from_auto` flag — caller treats it as the live auto-board).
    assert out["prematch"] == AUTO["prematch"]
    assert out["live"] == AUTO["live"]
    assert out["postmatch"] == AUTO["postmatch"]


def test_filter_drops_other_leagues():
    """Selecting only {6617, 6620} drops cards with other event_ids."""
    out = _filter_auto_board(AUTO, [6617, 6620], [])
    # 6617 and 6620 cards stay; 6626 and eid=None are dropped.
    assert [c["event_id"] for c in out["prematch"]] == [6617, 6620]
    assert [c["event_id"] for c in out["live"]] == [6620]
    assert [c["event_id"] for c in out["postmatch"]] == [6617]


def test_filter_strictly_drops_unmapped_live_cards():
    """v0.3.22 cont 4: when a user narrows the board, steam-only
    live cards (event_id=None) MUST be dropped, otherwise "Russian
    Esports live" would leak into the user's EPL view.
    """
    out = _filter_auto_board(AUTO, [6617, 6620], [])
    assert all(c.get("event_id") is not None for c in out["live"])
    # And the actual count: only the Lunar Horse live card survives.
    assert len(out["live"]) == 1


def test_watchlist_pins_pass_through():
    """Watchlist matches (by match_id) are kept even if the event is
    not in the user's selected set — the user explicitly pinned them.
    """
    watch_card = _card(6626, match_id=8910670427, league="BetBoom")
    auto = dict(AUTO, live=[watch_card] + AUTO["live"])
    # User selected only 6617, but the watch_id 8910670427 should pass.
    out = _filter_auto_board(auto, [6617], [8910670427])
    assert any(c.get("match_id") == 8910670427 for c in out["live"])


def test_watchlist_only_no_league_filter():
    """A user can use just the watchlist with no league filter."""
    out = _filter_auto_board(AUTO, [], [8910670427])
    # No league filter, no watch match in AUTO → empty live.
    assert out["live"] == []
    # But prematch and postmatch are still filtered strictly
    # (has_filter=True because watch_ids is set).
    # Wait — actually for prematch/postmatch we still drop eid=None
    # when there's any filter.  Hmm — let me re-check the contract.
    # The strict filter applies when has_filter=True.  So with only
    # watch_ids and no events, prematch/postmatch of unmapped leagues
    # are dropped.  This matches the live behaviour.
    assert all(c.get("event_id") is not None for c in out["prematch"])


def test_no_matches_for_unselected_event():
    """A user with no live matches in their selected leagues gets
    an empty live column — not the auto-board's full live set.
    """
    out = _filter_auto_board(AUTO, [6626], [])  # only BetBoom
    assert out["live"] == []  # BetBoom has no live cards in AUTO
    assert all(c["event_id"] == 6626 for c in out["prematch"])


def test_engine_field_preserved():
    out = _filter_auto_board(AUTO, [6617], [])
    assert out["engine"] == "ml"


def test_filtered_from_auto_marker():
    out = _filter_auto_board(AUTO, [6617], [])
    # The marker helps debugging — clients can see they got the
    # auto-board filter path rather than a fresh build.
    assert out.get("filtered_from_auto") is True
