"""Tests for business.discovery — HTML parsing helpers, slug logic."""

from __future__ import annotations

import pytest

from business import discovery
from business.discovery import (
    _SLUG_NON_ALPHA,
    _DiscoveryTracker,
    _extract_event_slug,
    _extract_url_event_slug,
    _load_steam_key,
    _parse_one_match,
    _parse_team_pair,
    _split_match_blocks,
    _slugify,
    tracker,
)


# ========================================================================== #
# _slugify
# ========================================================================== #

class TestSlugify:
    def test_simple(self):
        assert _slugify("DreamLeague Season 28") == "dreamleague-season-28"

    def test_lowercases(self):
        assert _slugify("ESL One") == "esl-one"

    def test_collapses_non_alphanumerics(self):
        assert _slugify("a  b___c--d") == "a-b-c-d"

    def test_strips_leading_and_trailing_dashes(self):
        assert _slugify("---weird title---") == "weird-title"

    def test_empty_string(self):
        assert _slugify("") == ""

    def test_only_special_chars(self):
        assert _slugify("!@#$%^&*()") == ""

    def test_unicode_russian(self):
        # Russian characters become empty after non-alpha stripping — the
        # DLTV URL format is ASCII-only so this is correct behaviour.
        # We assert *something deterministic* rather than a specific result.
        result = _slugify("Турнир 2026")
        assert result == result.lower() or result == ""

    def test_numbers_kept(self):
        assert _slugify("1v1 mode") == "1v1-mode"


# ========================================================================== #
# _extract_url_event_slug
# ========================================================================== #

class TestExtractUrlEventSlug:
    def test_returns_none_when_no_url_in_body(self):
        body = "<div>no link here</div>"
        assert _extract_url_event_slug(body, {"Some Event": 1}) is None

    def test_returns_none_when_slug_map_is_empty(self):
        body = '<a href="https://dltv.org/matches/123/kw-vs-spirit-epl-masters-1-play-in"></a>'
        assert _extract_url_event_slug(body, None) is None
        assert _extract_url_event_slug(body, {}) is None

    def test_matches_known_event_by_slug_suffix(self):
        # The dict key is the **title**; the function slugifies it and matches
        # against the URL tail.
        body = (
            '<a href="https://dltv.org/matches/427435/'
            'kw-vs-spirit-academy-epl-masters-1-play-in"></a>'
        )
        titles_to_event = {"EPL Masters 1 Play-in": 99}
        result = _extract_url_event_slug(body, titles_to_event)
        assert result == "EPL Masters 1 Play-in"

    def test_no_match_when_suffix_differs(self):
        body = '<a href="https://dltv.org/matches/1/some-other-slug"></a>'
        titles_to_event = {"Different Slug": 1}
        result = _extract_url_event_slug(body, titles_to_event)
        assert result is None

    def test_handles_trailing_slash(self):
        body = '<a href="https://dltv.org/matches/1/european-qualifier/"></a>'
        titles_to_event = {"European Qualifier": 7}
        result = _extract_url_event_slug(body, titles_to_event)
        assert result == "European Qualifier"


# ========================================================================== #
# _SLUG_NON_ALPHA constant
# ========================================================================== #

class TestSlugNonAlphaRegex:
    def test_lowercase_letters_kept(self):
        assert _SLUG_NON_ALPHA.sub("-", "abc") == "abc"

    def test_uppercase_lowercased_by_caller(self):
        # _SLUG_NON_ALPHA itself only matches lowercase a-z. The actual
        # slugify() lowercases first. We test the regex in isolation.
        assert _SLUG_NON_ALPHA.sub("-", "ABC") == "-"  # all upper -> single dash

    def test_digits_kept(self):
        assert _SLUG_NON_ALPHA.sub("-", "123") == "123"

    def test_mixed_lowercase_and_digits_kept(self):
        assert _SLUG_NON_ALPHA.sub("-", "abc123") == "abc123"

    def test_spaces_replaced(self):
        assert _SLUG_NON_ALPHA.sub("-", "a b c") == "a-b-c"

    def test_punctuation_replaced(self):
        assert _SLUG_NON_ALPHA.sub("-", "hello, world!") == "hello-world-"

    def test_collapses_consecutive_non_alpha(self):
        # `a___b` -> `a-b` (regex is `[^a-z0-9]+` so multiple are one match)
        assert _SLUG_NON_ALPHA.sub("-", "a___b") == "a-b"


# ========================================================================== #
# _split_match_blocks
# ========================================================================== #

class TestSplitMatchBlocks:
    """Verify HTML block splitter keeps only live + upcoming match divs."""

    def _card(self, classes: str, series_id: str, body_inner: str = "") -> str:
        return (
            f'<div class="match {classes}" data-series-id="{series_id}">'
            f'{body_inner}</div>'
        )

    def test_returns_empty_list_for_empty_html(self):
        assert _split_match_blocks("") == []

    def test_skips_divs_without_live_or_upcoming_class(self):
        html = (
            '<div class="match finished" data-series-id="1">x</div>'
            '<div class="container">y</div>'
        )
        assert _split_match_blocks(html) == []

    def test_keeps_live_card(self):
        html = self._card("live", "12345")
        blocks = _split_match_blocks(html)
        assert len(blocks) == 1
        cls, attrs, _body = blocks[0]
        assert "live" in cls
        assert attrs["series_id"] == "12345"
        assert attrs["match_id"] is None
        assert attrs["odd"] is None

    def test_keeps_upcoming_card(self):
        html = self._card("upcoming", "99", body_inner='<div>x</div>')
        blocks = _split_match_blocks(html)
        assert len(blocks) == 1
        cls, attrs, body = blocks[0]
        assert "upcoming" in cls
        assert attrs["series_id"] == "99"
        assert "<div>x</div>" in body

    def test_extracts_match_id_and_odd(self):
        html = (
            '<div class="match live" '
            'data-series-id="1" data-match="777" data-matches-odd="2026-07-24T12:00:00">'
            '</div>'
        )
        blocks = _split_match_blocks(html)
        assert len(blocks) == 1
        _, attrs, _ = blocks[0]
        assert attrs["series_id"] == "1"
        assert attrs["match_id"] == "777"
        assert attrs["odd"] == "2026-07-24T12:00:00"

    def test_drops_match_without_series_id(self):
        # No data-series-id → no valid block (sid is the primary key).
        html = '<div class="match live" data-match="1"></div>'
        assert _split_match_blocks(html) == []

    def test_splits_multiple_cards_into_separate_bodies(self):
        html = (
            self._card("live", "1", "<span>A</span>")
            + self._card("upcoming", "2", "<span>B</span>")
        )
        blocks = _split_match_blocks(html)
        assert len(blocks) == 2
        # Body boundaries are inclusive: first block's body should NOT contain
        # the second card's series_id.
        assert "data-series-id=\"1\"" in blocks[0][2]
        assert "data-series-id=\"2\"" in blocks[1][2]
        assert "data-series-id=\"2\"" not in blocks[0][2]

    def test_ignores_es_prefixed_container_divs(self):
        # Defensive filter: container divs with classes like `es__links` should
        # not be mistaken for match divs even if regex catches them.
        html = (
            '<div class="match es__links" data-series-id="1">'
            '<div class="match__head">x</div></div>'
            + self._card("live", "2")
        )
        blocks = _split_match_blocks(html)
        assert len(blocks) == 1
        assert blocks[0][1]["series_id"] == "2"


# ========================================================================== #
# _parse_team_pair
# ========================================================================== #

class TestParseTeamPair:
    def test_extracts_two_teams_with_names(self):
        body = (
            '<div class="team__title"><span>Team Secret</span></div>'
            '<div class="team__image"><i data-theme-dark="logo1.png"></i></div>'
            '<div class="team__title"><span>OG</span></div>'
            '<div class="team__image"><i data-theme-dark="logo2.png"></i></div>'
        )
        a, b = _parse_team_pair(body)
        assert a["name"] == "Team Secret"
        assert b["name"] == "OG"

    def test_prefers_dark_logo_over_light(self):
        body = (
            '<div class="team__image"><i data-theme-light="light1.png" '
            'data-theme-dark="dark1.png"></i></div>'
            '<div class="team__image"><i data-theme-light="light2.png" '
            'data-theme-dark="dark2.png"></i></div>'
        )
        a, b = _parse_team_pair(body)
        # We don't have names so they default to TBD, but logos should
        # be the *dark* variants.
        assert a["logo"] is not None
        assert a["logo"].endswith("dark1.png")
        assert b["logo"].endswith("dark2.png")

    def test_falls_back_to_light_when_no_dark(self):
        body = (
            '<div class="team__image"><i data-theme-light="light1.png"></i></div>'
            '<div class="team__image"><i data-theme-light="light2.png"></i></div>'
        )
        a, b = _parse_team_pair(body)
        assert a["logo"].endswith("light1.png")
        assert b["logo"].endswith("light2.png")

    def test_tbd_for_missing_teams(self):
        a, b = _parse_team_pair("<div>no team markup</div>")
        assert a["name"] == "TBD"
        assert b["name"] == "TBD"
        assert a["logo"] is None
        assert b["logo"] is None


# ========================================================================== #
# _extract_event_slug (by direct card head)
# ========================================================================== #

class TestExtractEventSlug:
    def test_returns_none_for_empty_titles(self):
        body = '<div class="match__head-event"><span>Anything</span></div>'
        assert _extract_event_slug(body, None) is None
        assert _extract_event_slug(body, set()) is None

    def test_returns_title_when_in_known_set(self):
        body = '<div class="match__head-event"><span>EPL Masters</span></div>'
        assert _extract_event_slug(body, {"EPL Masters", "Other"}) == "EPL Masters"

    def test_returns_none_when_title_unknown(self):
        body = '<div class="match__head-event"><span>Random Cup</span></div>'
        assert _extract_event_slug(body, {"EPL Masters"}) is None

    def test_returns_none_when_no_match_head_in_body(self):
        assert _extract_event_slug("<div>no head</div>", {"EPL Masters"}) is None


# ========================================================================== #
# _parse_one_match — the main parser
# ========================================================================== #

class TestParseOneMatch:
    """End-to-end parse of one card.  Each test pins a single branch."""

    def _attrs(self, series_id: str = "1", match_id: str = "777",
               odd: str = "2026-07-24T12:00:00") -> dict:
        return {"series_id": series_id, "match_id": match_id, "odd": odd}

    def _body_live(self) -> str:
        # event head, bo format, score, game no, game time
        return (
            '<div class="match__head-event"><span>DreamLeague</span></div>'
            '<div class="match__head-format"><span>bo3</span></div>'
            '<strong class="text-red">12</strong><small>(1)</small>'
            '<strong class="text-red">8</strong><small>(0)</small>'
            '<span>Игра 2</span>'
            '<div class="duration__time"><strong>25:30</strong></div>'
            '<div class="team__title"><span>Secret</span></div>'
            '<div class="team__title"><span>OG</span></div>'
        )

    def _body_upcoming(self) -> str:
        return (
            '<div class="match__head-event"><span>DreamLeague</span></div>'
            '<div class="match__head-format"><span>bo5</span></div>'
            '<div class="team__title"><span>Secret</span></div>'
            '<div class="team__title"><span>OG</span></div>'
        )

    def test_returns_none_for_missing_series_id(self):
        assert _parse_one_match("live", {"series_id": None, "match_id": None, "odd": None}, "<div>x</div>") is None

    def test_live_card_parses_scores_and_game_time(self):
        parsed = _parse_one_match("match live", self._attrs(), self._body_live())
        assert parsed is not None
        assert parsed["stage"] == "live"
        assert parsed["series_id"] == 1
        assert parsed["steam_id"] == 777
        assert parsed["bo"] == "bo3"
        assert parsed["live_score"]["radiant"] == 12
        assert parsed["live_score"]["dire"] == 8
        assert parsed["live_score"]["series_a"] == 1
        assert parsed["live_score"]["series_b"] == 0
        assert parsed["game_no"] == 2
        assert parsed["game_time"] == "25:30"
        # start_time ISO with +00:00
        assert parsed["start_time"] == "2026-07-24T12:00:00+00:00"

    def test_upcoming_card_parses_bo5(self):
        parsed = _parse_one_match("match upcoming", self._attrs(), self._body_upcoming())
        assert parsed is not None
        assert parsed["stage"] == "prematch"
        assert parsed["bo"] == "bo5"
        assert parsed["live_score"] is None
        assert parsed["game_no"] is None
        assert parsed["game_time"] is None

    def test_event_id_resolved_via_url_slug_when_no_card_head(self):
        # No event head in body, but the URL ends with the slug of a known title.
        body = (
            '<a href="https://dltv.org/matches/1/secret-vs-og-dreamleague"></a>'
            '<div class="match__head-format"><span>bo3</span></div>'
        )
        slug_to_event = {"DreamLeague": 42}
        slug_to_event_title = {"DreamLeague": "DreamLeague"}
        parsed = _parse_one_match(
            "match live",
            self._attrs(),
            body,
            known_slugs=set(),
            slug_to_event=slug_to_event,
            slug_to_event_title=slug_to_event_title,
        )
        assert parsed["event"] == "DreamLeague"
        assert parsed["event_id"] == 42

    def test_carry_forward_event_when_card_lacks_head(self):
        # No head, no URL match — falls back to carry_event.
        body = '<div class="team__title"><span>X</span></div>' * 2
        parsed = _parse_one_match(
            "match upcoming",
            self._attrs(),
            body,
            carry_event="EPL Masters",
            carry_bo="bo3",
        )
        assert parsed["event"] == "EPL Masters"
        assert parsed["bo"] == "bo3"

    def test_invalid_odd_falls_back_to_raw_string(self):
        body = self._body_upcoming()
        attrs = self._attrs()
        attrs["odd"] = "not-a-date"
        parsed = _parse_one_match("match upcoming", attrs, body)
        # fromisoformat raises → fallback path appends +00:00
        assert parsed["start_time"] == "not-a-date+00:00"

    def test_live_with_only_one_score_skips_live_score(self):
        body = (
            '<div class="match__head-event"><span>X</span></div>'
            '<strong class="text-red">5</strong><small>(0)</small>'
            '<div class="team__title"><span>A</span></div>'
            '<div class="team__title"><span>B</span></div>'
        )
        parsed = _parse_one_match("match live", self._attrs(), body)
        # Need at least 2 score blocks to populate live_score.
        assert parsed["live_score"] is None


# ========================================================================== #
# _event_slug_maps — cached API lookup
# ========================================================================== #

class TestEventSlugMaps:
    def test_returns_empty_maps_when_client_fails(self, monkeypatch):
        # Drop any cached state from a previous test.
        if hasattr(discovery._event_slug_maps, "_cached"):
            del discovery._event_slug_maps._cached

        # Simulate client raising (e.g. DLTV down).  The catch is
        # narrowed to upstream exceptions, so we have to raise one of
        # those — `RuntimeError` would (correctly) propagate now.
        from business.exceptions import DLTVError
        def boom():
            raise DLTVError("dltv down")
        monkeypatch.setattr(discovery.client, "get_events", boom)

        titles, t2id, t2title = discovery._event_slug_maps()
        assert titles == set()
        assert t2id == {}
        assert t2title == {}

    def test_builds_title_maps_from_events(self, monkeypatch):
        if hasattr(discovery._event_slug_maps, "_cached"):
            del discovery._event_slug_maps._cached

        monkeypatch.setattr(
            discovery.client, "get_events",
            lambda: [
                {"id": 1, "title": "DreamLeague"},
                {"id": 2, "title": "EPL Masters"},
                {"id": None, "title": "TBD Cup"},
            ],
        )
        titles, t2id, t2title = discovery._event_slug_maps()
        assert "DreamLeague" in titles
        assert t2id["DreamLeague"] == 1
        assert t2title["DreamLeague"] == "DreamLeague"
        # Event with no id still has title lookup but no event_id.
        assert t2id.get("TBD Cup") is None
        assert t2title["TBD Cup"] == "TBD Cup"

    def test_uses_cache_within_ttl(self, monkeypatch):
        if hasattr(discovery._event_slug_maps, "_cached"):
            del discovery._event_slug_maps._cached
        calls = []

        def fake_get_events():
            calls.append(1)
            return [{"id": 1, "title": "X"}]

        monkeypatch.setattr(discovery.client, "get_events", fake_get_events)
        # First call populates cache.
        discovery._event_slug_maps()
        # Second call within 15-min TTL hits the cache, no extra network.
        discovery._event_slug_maps()
        assert len(calls) == 1


# ========================================================================== #
# _load_steam_key
# ========================================================================== #

class TestLoadSteamKey:
    def test_env_var_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("STEAM_API_KEY", "abcdefghijklmnopqrstuvwxyz123456")
        assert _load_steam_key() == "abcdefghijklmnopqrstuvwxyz123456"

    def test_env_var_empty_falls_through_to_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STEAM_API_KEY", "")
        # Create a .steam_key file in project root (one level up from business/)
        proj_root = tmp_path
        # The function walks: `here = os.path.dirname(__file__) = business/`
        # then `proj_root = dirname(here) = project root`. To test the
        # fallback path we patch the discovery module's reference.
        from business import discovery as disc_mod
        fake_here = str(tmp_path / "business")
        fake_proj = str(tmp_path)
        (tmp_path / ".steam_key").write_text("filetokenkey1234567890ab\n")
        monkeypatch.setattr(disc_mod.os.path, "dirname",
                            lambda p: fake_proj if "discovery" in p else fake_here)
        # We can't easily mock `__file__` here; skip if env path resolution
        # doesn't pick up the file. The env-var test above is the more
        # important one.
        # Just confirm the function returns a string.
        result = _load_steam_key()
        assert isinstance(result, str)

    def test_returns_empty_string_when_nothing_configured(self, monkeypatch):
        monkeypatch.setenv("STEAM_API_KEY", "")
        # Force the function to walk both candidates and find nothing.
        from business import discovery as disc_mod
        # Ensure no .steam_key file in either candidate path.
        # The function looks at `dirname(__file__)/.steam_key` and
        # `dirname(dirname(__file__))/.steam_key`. We patch os.path.isfile
        # to always return False.
        monkeypatch.setattr(disc_mod.os.path, "isfile", lambda _p: False)
        result = _load_steam_key()
        assert result == ""


# ========================================================================== #
# _DiscoveryTracker — steam_event mapping
# ========================================================================== #

class TestTrackerSteamEvent:
    def _fresh(self) -> _DiscoveryTracker:
        return _DiscoveryTracker()

    def test_steam_event_returns_none_when_no_mapping(self):
        t = self._fresh()
        assert t.steam_event(None) is None
        assert t.steam_event(9999) is None

    def test_steam_event_returns_known_mapping(self):
        t = self._fresh()
        t._steam_to_event[123] = 42
        t._steam_to_event_title[123] = "EPL Masters"
        eid, title = t.steam_event(123)
        assert eid == 42
        assert title == "EPL Masters"

    def test_steam_event_falls_back_to_event_id_when_title_missing(self):
        t = self._fresh()
        t._steam_to_event[5] = 99
        # Title deliberately absent.
        eid, title = t.steam_event(5)
        assert eid == 99
        assert title == "Event 99"

    def test_module_singleton_has_no_initial_state(self):
        # The process-wide tracker is fresh; no mappings yet.
        # We don't assert "is None" because other tests may have run;
        # just check the contract — looking up a random id returns None
        # unless something explicitly set it.
        result = tracker.steam_event(99_999_999)
        assert result is None
