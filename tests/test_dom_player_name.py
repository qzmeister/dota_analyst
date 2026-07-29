"""v0.4.0.1: tests for the DOM-based player_name extractor on the
live DLTV match page.

Background
----------
v0.4.0.0 added a DOM extractor (`findPlayerNameFor` inside
`_read_live_state_from_scoreboard`) that used guessed selectors
(`.team.radiant .pick .player-name`, etc).  In practice the page
globals `radiant_picks`/`dire_picks` were empty AND the guessed
selectors didn't match — the live card never showed player names.

v0.4.0.1 switched to selectors that were *actually observed* on a
live DLTV page on 2026-07-29 (match 427520 — Amaru vs Nemiga,
EPL Masters 1).  The structure is:

    .map__finished-v2__pick > .heroes > .heroes__player   (× 10, 5 per side)
      .pick[data-tippy-content="<HERO NAME>"]
      a.heroes__player-player[href*="/players/<slug>"]
        .heroes__player-player__flag      (style="background-image: .../flag-icon/flags/4x3/<cc>.svg")
        .heroes__player-player__name      (text: player nickname)
      .heroes__player-rank

DOM order is pick order; the first 5 cards are radiant, the next 5
are dire.  These tests run the actual JS against the saved HTML
fixture (downloaded with curl, not React-hydrated) to verify the
end-to-end extraction.

Why a real Chromium instance?
-----------------------------
The DOM extractor runs as a `page.evaluate` JS block in the
browser; mocking `page.evaluate` would only verify the JS *runs*
without errors, not that the right elements are selected.  Using
`page.set_content()` with a saved HTML snapshot is the cheapest
real-DOM test that doesn't need the network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from business import dltv_browser  # noqa: E402
from business.dltv_browser import _read_live_state_from_scoreboard  # noqa: E402


_FIXTURE_HTML = Path(__file__).resolve().parent / "fixtures" / "dltv_live_match_427520.html"


def _is_playwright_browser_available() -> bool:
    """Return True if a chromium binary is installed and a browser
    instance can be launched in this test env.  Skip otherwise.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            browser.close()
        return True
    except Exception:
        return False


# 10 unique player nicknames seen on 2026-07-29 on match 427520
# (5 radiant + 5 dire).  Order matches the DOM order in the picks chart.
EXPECTED_RADIANT_PLAYERS = ["PiPi", "K1", "Oscar", "Genek", "Panda"]
EXPECTED_DIRE_PLAYERS    = ["byun", "young G", "Covisnine", "hotoke", "ariel"]

# Hero names from data-tippy-content on the same match
EXPECTED_RADIANT_HEROES = ["Earthshaker", "Alchemist", "Centaur Warrunner",
                            "Keeper of the Light", "Winter Wyvern"]
EXPECTED_DIRE_HEROES    = ["Tiny", "Ember Spirit", "Mirana", "Night Stalker", "Bane"]


@pytest.fixture(scope="module")
def live_dom_state():
    """Run the extractor against the saved live-match HTML and return
    the result.  Skips the test if chromium is not available.
    """
    if not _FIXTURE_HTML.exists():
        pytest.skip(f"fixture HTML missing: {_FIXTURE_HTML}")
    if not _is_playwright_browser_available():
        pytest.skip("chromium not available")
    from playwright.sync_api import sync_playwright

    html = _FIXTURE_HTML.read_text(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.set_content(html, wait_until="domcontentloaded")
        # give React a chance to hydrate (the picks chart renders server-side
        # already, but hydration sets data-hero-id which we don't rely on)
        page.wait_for_timeout(500)
        out = _read_live_state_from_scoreboard(page)
        browser.close()
    return out


def test_picks_source_is_dom(live_dom_state):
    """The DOM extractor finds 10 player cards, so picks_source='dom'."""
    assert live_dom_state.get("picks_source") == "dom", (
        f"expected picks_source=dom, got {live_dom_state.get('picks_source')!r}"
    )


def test_radiant_picks_have_player_name(live_dom_state):
    """Each radiant pick carries a non-empty `player_name` from the DOM."""
    radiant = live_dom_state["picks"]["radiant"]
    assert len(radiant) == 5, f"expected 5 radiant picks, got {len(radiant)}: {radiant!r}"
    names = [p.get("player_name") for p in radiant]
    assert names == EXPECTED_RADIANT_PLAYERS, (
        f"radiant player names mismatch:\n  expected: {EXPECTED_RADIANT_PLAYERS}\n  got:      {names}"
    )


def test_dire_picks_have_player_name(live_dom_state):
    """Each dire pick carries a non-empty `player_name` from the DOM."""
    dire = live_dom_state["picks"]["dire"]
    assert len(dire) == 5, f"expected 5 dire picks, got {len(dire)}: {dire!r}"
    names = [p.get("player_name") for p in dire]
    assert names == EXPECTED_DIRE_PLAYERS, (
        f"dire player names mismatch:\n  expected: {EXPECTED_DIRE_PLAYERS}\n  got:      {names}"
    )


def test_picks_have_hero_name(live_dom_state):
    """Each pick has `hero_name` from `data-tippy-content`."""
    radiant_heroes = [p.get("hero_name") for p in live_dom_state["picks"]["radiant"]]
    dire_heroes    = [p.get("hero_name") for p in live_dom_state["picks"]["dire"]]
    assert radiant_heroes == EXPECTED_RADIANT_HEROES, (
        f"radiant heroes: expected {EXPECTED_RADIANT_HEROES}, got {radiant_heroes}"
    )
    assert dire_heroes == EXPECTED_DIRE_HEROES, (
        f"dire heroes: expected {EXPECTED_DIRE_HEROES}, got {dire_heroes}"
    )


def test_picks_have_player_slug(live_dom_state):
    """Each pick has a `player_slug` parsed from the `a[href*="/players/"]` href."""
    for side in ("radiant", "dire"):
        for p in live_dom_state["picks"][side]:
            slug = p.get("player_slug")
            assert slug, f"{side} pick missing player_slug: {p!r}"
            # must be a real slug, not a placeholder
            assert slug != "null" and "/" not in slug, f"bad slug: {slug!r}"


def test_picks_have_player_country(live_dom_state):
    """Each pick has `player_country` (2-letter country code) extracted from
    the flag-icon background-image URL.  Some flags use 3-letter codes (e.g.
    'eng' for England) — accept both, but require uppercase.
    """
    for side in ("radiant", "dire"):
        for p in live_dom_state["picks"][side]:
            cc = p.get("player_country")
            assert cc, f"{side} pick missing player_country: {p!r}"
            assert cc == cc.upper(), f"country code must be uppercase, got {cc!r}"
            assert 2 <= len(cc) <= 3, f"country code must be 2-3 chars, got {cc!r}"


def test_picks_have_position(live_dom_state):
    """Each pick has `position` from `.pick__position` text.  DLTV's
    `.pick__position` element holds a hero-related numeric id that does
    NOT match Dota 2's standard hero_id (e.g. Earthshaker shows 25, not
    7; Tiny shows 30, not 19).  Whatever it represents, the values are
    stable per match snapshot and the extractor must surface them so
    callers can correlate with DLTV's UI.
    """
    # Match 427520: actual values observed in the static HTML
    expected_radiant = [25, 26, 24, 19, 20]  # Earthshaker, Alchemist, Centaur, KotL, WW
    expected_dire    = [30, 28, 25, 27, 21]  # Tiny, Ember, Mirana, NS, Bane
    rad = [p.get("position") for p in live_dom_state["picks"]["radiant"]]
    dire = [p.get("position") for p in live_dom_state["picks"]["dire"]]
    assert rad == expected_radiant, f"radiant positions: expected {expected_radiant}, got {rad}"
    assert dire == expected_dire, f"dire positions: expected {expected_dire}, got {dire}"


def test_pick_array_order_is_pick_order(live_dom_state):
    """The DOM extractor returns picks in document order which matches
    the actual pick order (radiant 1-5, then dire 1-5).  Each side's
    array length is exactly 5.
    """
    assert len(live_dom_state["picks"]["radiant"]) == 5
    assert len(live_dom_state["picks"]["dire"]) == 5
    # radiant heroes come in the same order as their player__name pairs
    radiant = [(p["hero_name"], p["player_name"]) for p in live_dom_state["picks"]["radiant"]]
    assert radiant[0] == ("Earthshaker", "PiPi")
    assert radiant[-1] == ("Winter Wyvern", "Panda")
    dire = [(p["hero_name"], p["player_name"]) for p in live_dom_state["picks"]["dire"]]
    assert dire[0] == ("Tiny", "byun")
    assert dire[-1] == ("Bane", "ariel")


# ---------------------------------------------------------------------------
# Mock-based regression tests (don't need Chromium; verify the JS body shape
# so a future refactor of `_read_live_state_from_scoreboard` can't quietly
# remove the new selectors)
# ---------------------------------------------------------------------------

class _FakePage:
    def __init__(self, result):
        self._result = result
        self.last_script = ""

    def evaluate(self, script):
        self.last_script = script
        return self._result


def test_js_body_targets_heroes_player_cards():
    """The new primary path must query `.heroes__player` (the DLTV
    per-player card class) and the `data-tippy-content` attribute on
    `.pick` — both verified on a live page.
    """
    page = _FakePage(None)  # value ignored; we only inspect the JS
    _read_live_state_from_scoreboard(page)
    script = page.last_script
    assert ".heroes__player" in script, "must query .heroes__player for per-side pick cards"
    assert "data-tippy-content" in script, "must read hero name from .pick[data-tippy-content]"
    assert "heroes__player-player__name" in script, "must read player name from .heroes__player-player__name"
    assert "map__finished-v2__pick" in script, "must use .map__finished-v2__pick (DLTV v0.4.0 layout)"


def test_js_body_extracts_player_slug_and_country():
    """The new extractor must parse the player slug (from /players/ href)
    and country (from the flag background-image URL — the URL contains
    `/flags/4x3/<cc>.svg` even though the directory is named `flag-icon`).
    """
    page = _FakePage(None)
    _read_live_state_from_scoreboard(page)
    script = page.last_script
    assert "/players/" in script, "must parse /players/<slug> from anchor href"
    # After the v0.4.0.1 refactor we use `[/]` instead of `\/` in the
    # JS regex to silence Python's SyntaxWarning about invalid escape
    # sequences.  The regex still matches the flag URL substring.
    assert "flags[/]4x3[/]" in script, "must parse country code from flag background-image URL"
    assert "[.]svg" in script, "country-code regex must match the .svg extension in flag URLs"
