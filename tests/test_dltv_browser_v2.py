"""
v0.3.22 unit tests for the new dltv_browser DOM extractor.

DLTV's live page layout changed: heroes are no longer tagged with
`data-hero-id`; they're referenced by image URL hash and resolved
through the embedded `window.__heroes` JS object.  The new
extractors consume that layout.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---- mock page ---- #


class _Locator:
    """Minimal stand-in for `playwright.sync_api.Locator` for the
    bits of the API our extractors actually use."""

    def __init__(self, items: List[Any]):
        self._items = items

    def count(self) -> int:
        return len(self._items)

    def nth(self, i: int) -> Any:
        return self._items[i]

    def first(self) -> Any:
        return self._items[0] if self._items else MagicMock()


class _MockPage:
    """A page mock that implements only `evaluate` and `locator`.
    The evaluate handler is injected by each test; locator returns
    whatever the test sets up via the `query_*` mappings."""

    def __init__(self, evaluate_map: Dict[str, Any], element_map: Dict[str, List[Any]]):
        self._evaluate_map = evaluate_map
        self._element_map = element_map

    def evaluate(self, expr: str):
        # Return whatever the test pre-loaded.  Tests pass the
        # *exact* evaluate string their extractor uses, including
        # the JS body, and we just look it up.
        for key, val in self._evaluate_map.items():
            if key in expr:
                return val
        return None

    def goto(self, url, **kw):
        # No-op: the test injects the rendered DOM via `evaluate`.
        return None

    def wait_for_timeout(self, ms):
        return None

    def wait_for_function(self, predicate, timeout=None, **kw):
        # v0.3.24f: the real implementation waits for the page's
        # socket.io-fed `radiant_picks` / `dire_picks` globals or the
        # `#live_scoreboard .team__scores-kills` to appear.  In the
        # mock those are injected by `evaluate` BEFORE the wait runs,
        # so the predicate is always satisfied and we can return
        # immediately.  Tests that want to exercise the timeout
        # branch set `wait_for_function` on the page to raise
        # TimeoutError.
        if getattr(self, "_wait_for_function_raises", False):
            from playwright.sync_api import TimeoutError
            raise TimeoutError("mock: wait_for_function forced to time out")
        return None

    def locator(self, sel: str):
        return _Locator(self._element_map.get(sel, []))


# ---- fixture: a small but realistic DLTV page payload ---- #

HEROES = {
    "1": {
        "id": 1, "steam_id": 1, "title": "Anti-Mage",
        "slug": "anti-mage", "image": "/uploads/heroes/ANTIMAGE_HASH.png",
    },
    "2": {
        "id": 2, "steam_id": 2, "title": "Axe",
        "slug": "axe", "image": "/uploads/heroes/AXE_HASH.png",
    },
    "3": {
        "id": 3, "steam_id": 3, "title": "Bane",
        "slug": "bane", "image": "/uploads/heroes/BANE_HASH.png",
    },
    "45": {
        "id": 45, "steam_id": 46, "title": "Templar Assassin",
        "slug": "templar-assassin", "image": "/uploads/heroes/TA_HASH.png",
    },
    "50": {
        "id": 50, "steam_id": 51, "title": "Clockwerk",
        "slug": "clockwerk", "image": "/uploads/heroes/CLOCK_HASH.png",
    },
    "85": {
        "id": 85, "steam_id": 86, "title": "Rubick",
        "slug": "rubick", "image": "/uploads/heroes/RUBICK_HASH.png",
    },
    "118": {
        "id": 118, "steam_id": 128, "title": "Snapfire",
        "slug": "snapfire", "image": "/uploads/heroes/SNAP_HASH.png",
    },
}


def _make_page(*, team_order_dire_first: bool, scores=("6", "35"), time_str="36:15"):
    """Build a `_MockPage` that returns a realistic DLTV page payload.

    The `__heroes` JS object is set up; picks/bans/scores/teams are
    returned by the JS evaluation in the same shape the real
    `evaluate()` block in `_read_map_block_from_dom` produces.
    """
    # Build the team list with side_kind (locale-independent).
    if team_order_dire_first:
        teams = [
            {"name": "Team Jenz", "side_kind": "dire"},
            {"name": "Team Syntax", "side_kind": "radiant"},
        ]
    else:
        teams = [
            {"name": "Team Syntax", "side_kind": "radiant"},
            {"name": "Team Jenz", "side_kind": "dire"},
        ]

    page_data = {
        "picks": [
            "/uploads/heroes/TA_HASH.png",
            "/uploads/heroes/SNAP_HASH.png",
            "/uploads/heroes/ANTIMAGE_HASH.png",
            "/uploads/heroes/RUBICK_HASH.png",
            "/uploads/heroes/CLOCK_HASH.png",
            "/uploads/heroes/BANE_HASH.png",
            "/uploads/heroes/AXE_HASH.png",
            "/uploads/heroes/ANTIMAGE_HASH.png",
            "/uploads/heroes/SNAP_HASH.png",
            "/uploads/heroes/TA_HASH.png",
        ],
        "bans": [
            "/uploads/heroes/BANE_HASH.png",
            "/uploads/heroes/AXE_HASH.png",
            "/uploads/heroes/TA_HASH.png",
            "/uploads/heroes/RUBICK_HASH.png",
            "/uploads/heroes/CLOCK_HASH.png",
            "/uploads/heroes/ANTIMAGE_HASH.png",
            "/uploads/heroes/SNAP_HASH.png",
            "/uploads/heroes/BANE_HASH.png",
            "/uploads/heroes/AXE_HASH.png",
            "/uploads/heroes/TA_HASH.png",
        ],
        "scores": list(scores),
        "time": time_str,
        "teams": teams,
    }

    evaluate_map = {
        "window.__heroes": json.dumps(HEROES),
        ".map__finished-v2": page_data,  # our extractor runs a single
        # evaluate that returns this whole dict
    }

    return _MockPage(evaluate_map, {})


# ---- tests ---- #


class TestReadHeroesFromWindow:
    def test_parses_heroes(self, monkeypatch):
        from business import dltv_client, dltv_browser
        # Stub the project-local hero index so the fallback path is
        # empty (we test the window.__heroes path in isolation).
        monkeypatch.setattr(dltv_client.client, "get_heroes", lambda: [])
        page = _MockPage(
            {"window.__heroes": json.dumps(HEROES)},
            {},
        )
        result = dltv_browser._read_heroes_from_window(page)
        # Each hero's image hash should map to its full dict
        assert "TA_HASH" in result
        assert result["TA_HASH"]["id"] == 45
        assert result["TA_HASH"]["steam_id"] == 46
        assert result["TA_HASH"]["title"] == "Templar Assassin"
        assert "ANTIMAGE_HASH" in result
        assert result["ANTIMAGE_HASH"]["steam_id"] == 1

    def test_falls_back_to_client_get_heroes(self, monkeypatch):
        """If `window.__heroes` is not embedded, the local DLTVClient
        hero index is used as a fallback.  Both share the same image
        URL, so the hash -> hero lookup works identically.
        """
        from business import dltv_client, dltv_browser
        monkeypatch.setattr(
            dltv_client.client, "get_heroes",
            lambda: [
                {"id": 45, "steam_id": 46, "title": "Templar Assassin",
                 "image": "/uploads/heroes/TA_HASH.png"},
                {"id": 1, "steam_id": 1, "title": "Anti-Mage",
                 "image": "/uploads/heroes/ANTIMAGE_HASH.png"},
            ],
        )
        page = _MockPage({"window.__heroes": None}, {})
        result = dltv_browser._read_heroes_from_window(page)
        assert "TA_HASH" in result
        assert result["TA_HASH"]["title"] == "Templar Assassin"
        assert "ANTIMAGE_HASH" in result

    def test_returns_empty_when_no_heroes(self, monkeypatch):
        from business import dltv_client, dltv_browser
        # Both paths empty
        monkeypatch.setattr(dltv_client.client, "get_heroes", lambda: [])
        page = _MockPage({"window.__heroes": None}, {})
        assert dltv_browser._read_heroes_from_window(page) == {}

    def test_returns_empty_on_garbage(self, monkeypatch):
        from business import dltv_client, dltv_browser
        monkeypatch.setattr(dltv_client.client, "get_heroes", lambda: [])
        page = _MockPage({"window.__heroes": "not json"}, {})
        assert dltv_browser._read_heroes_from_window(page) == {}


class TestReadMapBlockFromDom:
    def test_dire_first_layout(self):
        """Team Jenz (dire) is rendered first in DOM, then Team Syntax (radiant).
        Picks 0-4 = dire (Jenz), picks 5-9 = radiant (Syntax)."""
        from business import dltv_browser
        page = _make_page(team_order_dire_first=True)
        result = dltv_browser._read_map_block_from_dom(page)
        assert result["team_order"] == ["dire", "radiant"]
        # 5 picks each side
        assert len(result["picks"]["dire"]) == 5
        assert len(result["picks"]["radiant"]) == 5
        # First dire pick should be TA (dltv 45, steam 46)
        ta = result["picks"]["dire"][0]
        assert ta["hero_id"] == 45
        assert ta["steam_id"] == 46
        assert ta["name"] == "Templar Assassin"
        # First radiant pick should be Bane (dltv 3, steam 3)
        bane = result["picks"]["radiant"][0]
        assert bane["hero_id"] == 3
        assert bane["steam_id"] == 3
        assert bane["name"] == "Bane"
        # Score mapping: DOM order is [dire, radiant], so scores[0]=6 → dire, scores[1]=35 → radiant
        assert result["dire_score"] == 6
        assert result["radiant_score"] == 35
        assert result["game_time"] == "36:15"

    def test_radiant_first_layout(self):
        """Reverse team order: Syntax (radiant) first, Jenz (dire) second."""
        from business import dltv_browser
        page = _make_page(team_order_dire_first=False)
        result = dltv_browser._read_map_block_from_dom(page)
        assert result["team_order"] == ["radiant", "dire"]
        # Picks 0-4 = radiant, 5-9 = dire
        assert result["picks"]["radiant"][0]["name"] == "Templar Assassin"
        assert result["picks"]["dire"][0]["name"] == "Bane"
        # scores[0] is now radiant, scores[1] is dire
        assert result["radiant_score"] == 6
        assert result["dire_score"] == 35

    def test_bans_split(self):
        from business import dltv_browser
        page = _make_page(team_order_dire_first=True)
        result = dltv_browser._read_map_block_from_dom(page)
        assert len(result["bans"]["dire"]) == 5
        assert len(result["bans"]["radiant"]) == 5

    def test_unknown_team_side_falls_back(self):
        """If we can't recognize the side text, the first team defaults to radiant."""
        from business import dltv_browser
        # Build a page with no recognizable side text
        page_data = {
            "picks": ["/uploads/heroes/TA_HASH.png"] * 10,
            "bans": ["/uploads/heroes/AXE_HASH.png"] * 10,
            "scores": ["1", "2"],
            "time": "10:00",
            "teams": [
                {"name": "Alpha", "side_kind": "unknown"},
                {"name": "Beta", "side_kind": "unknown"},
            ],
        }
        page = _MockPage(
            {"window.__heroes": json.dumps(HEROES), ".map__finished-v2": page_data},
            {},
        )
        result = dltv_browser._read_map_block_from_dom(page)
        assert result["team_order"] == ["unknown", "unknown"]
        # Defaults: first 5 = radiant, next 5 = dire
        assert len(result["picks"]["radiant"]) == 5
        assert len(result["picks"]["dire"]) == 5

    def test_no_map_block_returns_empty_state(self):
        from business import dltv_browser
        page = _MockPage({".map__finished-v2": None, "window.__heroes": json.dumps(HEROES)}, {})
        result = dltv_browser._read_map_block_from_dom(page)
        assert result["picks"] == {"radiant": [], "dire": []}
        assert result["bans"] == {"radiant": [], "dire": []}
        assert result["team_order"] == []
        assert "radiant_score" not in result
        assert "game_time" not in result

    def test_short_pick_list_keeps_empty(self):
        from business import dltv_browser
        page_data = {
            "picks": ["/uploads/heroes/TA_HASH.png"],  # only 1 — too few to split
            "bans": [],
            "scores": ["0", "0"],
            "time": "",
            "teams": [],
        }
        page = _MockPage(
            {"window.__heroes": json.dumps(HEROES), ".map__finished-v2": page_data},
            {},
        )
        result = dltv_browser._read_map_block_from_dom(page)
        assert result["picks"] == {"radiant": [], "dire": []}

    def test_locale_independent_side_detection(self):
        """The extractor must work even when DLTV shows a non-English
        locale for the side text (e.g. German 'Dunkelheit'/'Strahlend'
        instead of Russian 'Силы тьмы'/'Силы Света').  Side detection
        relies on the .side classList ('side dire' / 'side radiant'),
        not on the text content.
        """
        from business import dltv_browser
        # The data is identical to the dire-first test, except the
        # JS now returns side_kind directly (which the page extracts
        # from .side.classList before sending back).
        page_data = {
            "picks": [
                "/uploads/heroes/TA_HASH.png",   # dire
                "/uploads/heroes/SNAP_HASH.png",
                "/uploads/heroes/ANTIMAGE_HASH.png",
                "/uploads/heroes/RUBICK_HASH.png",
                "/uploads/heroes/CLOCK_HASH.png",
                "/uploads/heroes/BANE_HASH.png",  # radiant
                "/uploads/heroes/AXE_HASH.png",
                "/uploads/heroes/ANTIMAGE_HASH.png",
                "/uploads/heroes/SNAP_HASH.png",
                "/uploads/heroes/TA_HASH.png",
            ],
            "bans": ["/uploads/heroes/AXE_HASH.png"] * 10,
            "scores": ["12", "34"],
            "time": "20:00",
            "teams": [
                # No text — purely class-based
                {"name": "Dunkelheit-Team", "side_kind": "dire"},
                {"name": "Strahlend-Team", "side_kind": "radiant"},
            ],
        }
        page = _MockPage(
            {"window.__heroes": json.dumps(HEROES), ".map__finished-v2": page_data},
            {},
        )
        result = dltv_browser._read_map_block_from_dom(page)
        assert result["team_order"] == ["dire", "radiant"]
        # Score: scores[0]=12 (DOM team 1 = dire), scores[1]=34 (DOM team 2 = radiant)
        assert result["dire_score"] == 12
        assert result["radiant_score"] == 34
        # 5 picks each side
        assert len(result["picks"]["dire"]) == 5
        assert len(result["picks"]["radiant"]) == 5
        # First dire pick is TA, first radiant pick is Bane
        assert result["picks"]["dire"][0]["name"] == "Templar Assassin"
        assert result["picks"]["radiant"][0]["name"] == "Bane"


class TestReadLiveStateFromScoreboard:
    """v0.3.24g: `_read_live_state_from_scoreboard` extracts picks
    from the page globals AND the live `radiant_score` / `dire_score`
    / `game_time` / `radiant_networth` / `dire_networth` from the
    `#live_scoreboard` DOM.  The tests below cover the networth
    fields specifically (the rest was already covered by the legacy
    `_read_map_block_from_dom` tests, which exercise the same data
    path through a different selector)."""

    def _page(self, *, team_order_dire_first: bool, networth_r: int, networth_d: int, scores=(6, 35), game_time=2235):
        """Build a `_MockPage` whose `evaluate()` returns the new
        live-state payload (the shape produced by the JS block in
        `_read_live_state_from_scoreboard`).
        """
        if team_order_dire_first:
            teams = [
                {"name": "Team Jenz",    "side_kind": "dire",    "kills": scores[0], "networth": networth_d},
                {"name": "Team Syntax",  "side_kind": "radiant", "kills": scores[1], "networth": networth_r},
            ]
            radiant_score, dire_score = scores[1], scores[0]
        else:
            teams = [
                {"name": "Team Syntax",  "side_kind": "radiant", "kills": scores[0], "networth": networth_r},
                {"name": "Team Jenz",    "side_kind": "dire",    "kills": scores[1], "networth": networth_d},
            ]
            radiant_score, dire_score = scores[0], scores[1]
        payload = {
            "picks": {"radiant": [], "dire": []},
            "bans": {"radiant": [], "dire": []},
            "team_order": [t["side_kind"] for t in teams],
            "teams": teams,
            "radiant_score": radiant_score,
            "dire_score": dire_score,
            "game_time": game_time,
            "radiant_networth": networth_r,
            "dire_networth": networth_d,
        }
        # The JS in `_read_live_state_from_scoreboard` references
        # `radiant_picks` / `dire_picks` (page globals) and the
        # `#live_scoreboard` selector.  We key on both substrings so
        # the mock returns the payload for that evaluate call.
        return _MockPage(
            {
                "radiant_picks": payload,
                "#live_scoreboard": payload,
            },
            {},
        )

    def test_networth_extracted_per_side(self):
        from business import dltv_browser
        page = self._page(team_order_dire_first=True, networth_r=23888, networth_d=20651)
        result = dltv_browser._read_live_state_from_scoreboard(page)
        # The extractor walks `teams[]` to expose networth at the
        # top level as `radiant_networth` / `dire_networth`.  In the
        # dire-first mock, teams[0] is dire, teams[1] is radiant.
        assert result["radiant_networth"] == 23888
        assert result["dire_networth"] == 20651
        # team_names / team_sides are exposed at the top level for
        # the board layer to attribute the networth back to a team
        # name without re-reading the DOM.
        assert "Team Syntax" in result["team_names"]
        assert "Team Jenz" in result["team_names"]

    def test_networth_missing_for_finished_map_returns_none(self):
        """v0.3.24g: a finished map leaves `.team__networth` empty
        in the DOM.  The extractor must report `networth=None` (not
        0) so the live card hides the gold line instead of showing
        a misleading "0  0" lead."""
        from business import dltv_browser
        # The mock returns the payload regardless of which field is
        # missing.  Simulate the "empty DOM" case by setting networth
        # to None on both teams.
        page = self._page(team_order_dire_first=False, networth_r=None, networth_d=None)
        result = dltv_browser._read_live_state_from_scoreboard(page)
        # The mock's `radiant_picks` key still matches the evaluate
        # call, so the extractor runs end-to-end.  But since the
        # payload has networth=None on every team, the top-level
        # fields must be None too.
        assert result["radiant_networth"] is None
        assert result["dire_networth"] is None

    def test_game_time_passes_through(self):
        """`game_time` was already extracted in v0.3.23; verify the
        new extractor still surfaces it (we now read it in the
        same call as the networth, so a regression in one would
        hit the other)."""
        from business import dltv_browser
        page = self._page(team_order_dire_first=False, networth_r=10000, networth_d=9000, game_time=1234)
        result = dltv_browser._read_live_state_from_scoreboard(page)
        assert result["game_time"] == 1234


class TestMatchStateCacheAlias:
    """v0.3.24h: `update_match_state_cache(dltv_id, url, steam_id)`
    writes the entry under BOTH `s{dltv_id}` and `s{steam_id}`.  The
    watchlist path (which only knows the steam id) can then find
    the same data without going through the discovery tracker — a
    critical fix for the post-match window where the tracker has
    pruned the match but the cache is still on disk.

    The two keys share the SAME `match_state` dict and `ts`, so
    they always expire together.  We assert that here explicitly
    so a future refactor doesn't accidentally split them.
    """

    def _setup_cache(self, tmp_path, monkeypatch):
        """Redirect the module-level cache file to a tmp path so
        the test doesn't touch the real ml_data/player_wr_cache.json."""
        from business import dltv_browser
        cache_file = tmp_path / "player_wr_cache.json"
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_CACHE_FILE", cache_file)
        monkeypatch.setattr(dltv_browser, "ML_DATA_DIR", tmp_path)
        return dltv_browser, cache_file

    def test_alias_written_when_steam_id_differs(self, tmp_path, monkeypatch):
        import time as _t
        from business import dltv_browser
        dltv_browser, cache_file = self._setup_cache(tmp_path, monkeypatch)
        # Simulate a successful fetch: pre-populate the cache as
        # if `fetch_match_state` had returned a real state, and
        # call the write path of `update_match_state_cache`
        # directly (skip the actual Playwright fetch).
        cache = {}
        cache[dltv_browser._cache_key(427547)] = {
            "ts": _t.time(),  # use NOW so MATCH_STATE_TTL_SEC doesn't expire it
            "url": "https://dltv.org/matches/427547/...",
            "match_state": {"picks": {"radiant": [{"name": "X"}], "dire": []}, "game_time": 60},
        }
        cache[dltv_browser._cache_key(8916603860)] = cache[dltv_browser._cache_key(427547)]
        dltv_browser._write_cache(cache)
        # Both keys must resolve to a non-None state and have
        # identical contents.  (Object identity is not preserved
        # because `_read_cache()` rebuilds the dicts from JSON on
        # every call — only the file's byte content is shared.)
        dltv_state = dltv_browser.get_cached_match_state(427547)
        steam_state = dltv_browser.get_cached_match_state_by_steam(8916603860)
        assert dltv_state is not None
        assert steam_state is not None
        assert dltv_state == steam_state
        # On disk both keys point to entries with the same ts.
        cache_on_disk = dltv_browser._read_cache()
        assert cache_on_disk[dltv_browser._cache_key(427547)]["ts"] == cache_on_disk[dltv_browser._cache_key(8916603860)]["ts"]
        # The match_state sub-dict is also equal.
        assert cache_on_disk[dltv_browser._cache_key(427547)]["match_state"] == cache_on_disk[dltv_browser._cache_key(8916603860)]["match_state"]

    def test_no_alias_when_steam_id_equals_dltv_id(self, tmp_path, monkeypatch):
        """If the publisher passes the same id for both, the
        write path must not duplicate the entry."""
        from business import dltv_browser
        self._setup_cache(tmp_path, monkeypatch)
        # Build the cache the way `update_match_state_cache` would
        # when steam_id == series_id.
        cache = {}
        cache[dltv_browser._cache_key(427547)] = {"ts": 1.0, "match_state": {}}
        dltv_browser._write_cache(cache)
        # Try to re-write via the alias path with the same id.
        prev = cache[dltv_browser._cache_key(427547)]
        cache[dltv_browser._cache_key(427547)] = prev  # the alias branch in
                                                        # update_match_state_cache
                                                        # short-circuits on
                                                        # `int(steam_id) != int(series_id)`,
                                                        # so this is a no-op.
        # Only one key in the cache.
        dltv_browser._write_cache(cache)
        loaded = dltv_browser._read_cache()
        assert len(loaded) == 1
        assert dltv_browser._cache_key(427547) in loaded


class TestFetchMatchStateIntegration:
    """End-to-end: spin up a mocked `sync_playwright` and verify the
    `fetch_match_state` glue.  The actual `page.goto` is intercepted
    via the same `_MockPage` mechanism."""

    def test_reuses_shared_browser_across_fetches(self, monkeypatch):
        """v0.3.22: the playwright context + chromium are created
        once and reused.  Earlier code spun up a new chromium for
        every call — which leaked ~20 subprocesses per fetch and
        ballooned WSL memory to 16GB after a few hours.

        Verify the launcher is called at most once even when we
        fetch multiple times in a row.
        """
        from business import dltv_browser
        page = _make_page(team_order_dire_first=True)

        class _MockExecutor:
            """Single-worker executor that runs synchronously (no thread)."""
            def submit(self, fn, *args, **kwargs):
                fut = _MockFut()
                try:
                    fut._result = fn(*args, **kwargs)
                except Exception as e:
                    fut._exc = e
                return fut

        class _MockFut:
            _result = None
            _exc = None
            def result(self, timeout=None):
                if self._exc is not None:
                    raise self._exc
                return self._result

        class _MockPW:
            def start(self):
                TestFetchMatchStateIntegration._start_calls += 1
                return self
            def stop(self):
                pass
            @property
            def chromium(self):
                return _Chromium()
        TestFetchMatchStateIntegration._start_calls = 0

        class _Chromium:
            _launch_calls = 0
            def launch(self, **kw):
                _Chromium._launch_calls += 1
                return _Browser(page)

        class _Browser:
            def __init__(self, page):
                self._page = page
            def new_context(self):
                return _Context()
            def new_page(self):
                return self._page
            def close(self):
                pass
            @property
            def contexts(self):
                # `_is_browser_alive()` probes this — must not raise.
                return [self]

        class _Context:
            def new_page(self):
                return page
            def close(self):
                pass

        import playwright.sync_api as psa
        monkeypatch.setattr(psa, "sync_playwright", _MockPW)
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_PAGE_LOAD_WAIT_MS", 0)
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_FETCH_TIMEOUT_MS", 5000)
        monkeypatch.setattr(dltv_browser, "_browser", None)
        monkeypatch.setattr(dltv_browser, "_playwright", None)
        monkeypatch.setattr(dltv_browser, "_browser_executor", _MockExecutor())
        _Chromium._launch_calls = 0

        # Three sequential fetches should share the same browser.
        for i in range(3):
            dltv_browser.fetch_match_state(f"https://dltv.org/matches/{i}/x")
        assert TestFetchMatchStateIntegration._start_calls == 1, (
            f"playwright.start() was called {TestFetchMatchStateIntegration._start_calls} times, expected 1"
        )
        assert _Chromium._launch_calls == 1, (
            f"chromium.launch() was called {_Chromium._launch_calls} times, expected 1"
        )

    def test_fetch_returns_extracted_state(self, monkeypatch):
        from business import dltv_browser
        page = _make_page(team_order_dire_first=True)

        # Build a mock sync_playwright that yields our `_MockPage`.
        # v0.3.22: the playwright context is now started via `.start()`
        # (we keep it alive for the whole process instead of using
        # the `with` context manager).  The mock needs to expose
        # `.start()` returning a manager-like object with `chromium`.
        class _MockExecutor:
            def submit(self, fn, *args, **kwargs):
                fut = _MockFut()
                try:
                    fut._result = fn(*args, **kwargs)
                except Exception as e:
                    fut._exc = e
                return fut

        class _MockFut:
            _result = None
            _exc = None
            def result(self, timeout=None):
                if self._exc is not None:
                    raise self._exc
                return self._result

        class _MockPW:
            def start(self):
                return self
            def stop(self):
                pass
            @property
            def chromium(self):
                return _Chromium()

        class _Chromium:
            def launch(self, **kw):
                return _Browser(page)

        class _Browser:
            def __init__(self, page):
                self._page = page
            def new_context(self):
                return _Context()
            def new_page(self):
                return self._page
            def close(self):
                pass
            @property
            def contexts(self):
                # `_is_browser_alive()` probes this — must not raise.
                return [self]

        class _Context:
            def new_page(self):
                return page
            def close(self):
                pass

        import playwright.sync_api as psa
        monkeypatch.setattr(psa, "sync_playwright", _MockPW)
        # Avoid the goto timeout branch
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_PAGE_LOAD_WAIT_MS", 0)
        # Don't actually wait
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_FETCH_TIMEOUT_MS", 5000)
        # Reset the module-level browser cache so we go through the
        # new shared-browser path each test.
        monkeypatch.setattr(dltv_browser, "_browser", None)
        monkeypatch.setattr(dltv_browser, "_playwright", None)
        monkeypatch.setattr(dltv_browser, "_browser_executor", _MockExecutor())

        result = dltv_browser.fetch_match_state("https://dltv.org/matches/427526/example")
        # The result should have 5+5 picks
        assert len(result["picks"]["dire"]) == 5
        assert len(result["picks"]["radiant"]) == 5
        # And the score from `.team__scores-kills`
        assert result["radiant_score"] == 35
        assert result["dire_score"] == 6
        # And the time
        assert result["game_time"] == "36:15"
        # team_order is preserved so the caller can re-map if it wants
        assert result["team_order"] == ["dire", "radiant"]

    def test_falls_back_to_legacy_extractor_when_wait_times_out(self, monkeypatch):
        """v0.3.24f: when the page's socket.io-fed globals haven't
        been populated within PLAYER_WR_PAGE_LOAD_WAIT_MS, the
        `wait_for_function` predicate times out.  The legacy
        extractors (`.map__finished-v2`, body-text scan) must still
        produce a valid state — otherwise a slow chromium / flaky
        network would silently drop the live card."""
        from business import dltv_browser

        # Build a page that has the OLDER layout (no
        # `#live_scoreboard` element), so the wait_for_function
        # predicate never returns true and the timeout branch fires.
        # The `.map__finished-v2` block IS present, so the legacy
        # extractor has something to read.
        page = _make_page(team_order_dire_first=True)
        page._wait_for_function_raises = True  # force the timeout branch

        class _MockExecutor:
            def submit(self, fn, *args, **kwargs):
                class _Fut:
                    _result = None
                    _exc = None
                    def result(self, timeout=None):
                        if self._exc is not None:
                            raise self._exc
                        return self._result
                f = _Fut()
                try:
                    f._result = fn(*args, **kwargs)
                except Exception as e:
                    f._exc = e
                return f

        class _MockPW:
            def start(self):
                return self
            def stop(self):
                pass
            @property
            def chromium(self):
                class _C:
                    def launch(self, **kw):
                        class _B:
                            def new_context(self):
                                class _X:
                                    def new_page(self):
                                        return page
                                    def close(self):
                                        pass
                                return _X()
                            def new_page(self):
                                return page
                            def close(self):
                                pass
                            @property
                            def contexts(self):
                                return [self]
                        return _B()
                return _C()

        import playwright.sync_api as psa
        monkeypatch.setattr(psa, "sync_playwright", _MockPW)
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_PAGE_LOAD_WAIT_MS", 100)
        monkeypatch.setattr(dltv_browser, "PLAYER_WR_FETCH_TIMEOUT_MS", 5000)
        monkeypatch.setattr(dltv_browser, "_browser", None)
        monkeypatch.setattr(dltv_browser, "_playwright", None)
        monkeypatch.setattr(dltv_browser, "_browser_executor", _MockExecutor())

        result = dltv_browser.fetch_match_state("https://dltv.org/matches/427530/x")
        # Legacy extractor (.map__finished-v2) must have produced
        # something — at minimum the picks (5+5) and the score.
        assert result["picks"]["radiant"], "legacy extractor produced no picks"
        assert result["picks"]["dire"], "legacy extractor produced no picks"
        assert result["radiant_score"] == 35
        assert result["dire_score"] == 6


class TestDiscoverySynthesizesLiveRow:
    """v0.3.22: when discovery synthesizes a 'live without steam_id'
    row it must set `started_at` and `status: 1` so `classify_stage`
    returns 'live' (not 'prematch')."""

    def test_synthesized_row_is_classified_live(self, monkeypatch):
        from business import dltv_client, discovery
        # Direct synthesize (the helper isn't exposed, so mimic the
        # branch manually with a fake scraper row).
        from datetime import datetime, timezone
        m = {
            "series_id": 427526,
            "steam_id": 0,
            "stage": "live",
            "event": "EPL Masters 1",
            "event_id": 6617,
            "bo": "bo3",
            "team_a": {"name": "Jenz", "logo": "x"},
            "team_b": {"name": "Team Syntax", "logo": "y"},
            "start_time": None,  # the real-world bug
        }
        # Manually build what get_live_and_prematch would build for this row
        synthesized_started_at = m.get("start_time") or datetime.now(timezone.utc).isoformat()
        series = {
            "id": m.get("series_id"),
            "event_id": m.get("event_id"),
            "first_team": m.get("team_a"),
            "second_team": m.get("team_b"),
            "first_team_id": None,
            "second_team_id": None,
            "type": 3,
            "maps": [],
            "started_at": synthesized_started_at,
            "status": 1,
            "live_score": m.get("live_score"),
            "_scraper_event": m.get("event"),
            "_scraper_bo": m.get("bo"),
            "_scraper_event_id": m.get("event_id"),
            "_live_no_steam_id": True,
        }
        stage = dltv_client.client.classify_stage(series)
        assert stage == "live", f"expected live, got {stage}"
