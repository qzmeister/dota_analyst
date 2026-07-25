"""Read the Liquipedia Tier 1 and Tier 2 tournament indexes.

Liquipedia is the source of truth for the league selector.  Only the rendered
Tier 1 and Tier 2 index pages are queried, cached in memory for six hours, and
used to match DLTV event titles.
"""

from __future__ import annotations

import html
import json
import re
import time
from html.parser import HTMLParser
from threading import Lock
from typing import List, Set

import requests


API_URL = "https://liquipedia.net/dota2/api.php"
CACHE_TTL_SECONDS = 6 * 60 * 60
HEADERS = {"User-Agent": "DotaAnalyst/1.0 (local draft-analysis tool)"}
_lock = Lock()
_cached_names: Set[str] = set()
_cache_until = 0.0


def _normalise(value: str) -> str:
    value = html.unescape(value or "").lower()
    value = re.sub(r"\b(play[- ]?in|closed qualifier|open qualifier)\b", "", value)
    # DLTV commonly uses Arabic ordinal labels while Liquipedia uses Roman ones
    # in tournament page titles (for example, "EPL Masters 1" vs "EPL Masters I").
    roman_numbers = {"iv": "4", "iii": "3", "ii": "2", "vi": "6", "v": "5", "i": "1"}
    for roman, arabic in roman_numbers.items():
        value = re.sub(rf"\b{roman}\b", arabic, value)
    return re.sub(r"[^a-z0-9]+", "", value)


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.names: List[str] = []
        self._title = ""
        self._text: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            attributes = dict(attrs)
            href = attributes.get("href", "")
            self._title = attributes.get("title", "") if href.startswith("/dota2/") else ""
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._title:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._title:
            name = "".join(self._text).strip() or self._title
            if name:
                self.names.append(name)
            self._title = ""
            self._text = []


def _fetch_tier_page(page: str) -> Set[str]:
    response = requests.get(
        API_URL,
        params={"action": "parse", "page": page, "prop": "text", "format": "json"},
        headers=HEADERS,
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    rendered = payload.get("parse", {}).get("text", {}).get("*", "")
    parser = _AnchorCollector()
    parser.feed(rendered)
    return {_normalise(name) for name in parser.names if len(_normalise(name)) >= 5}


def tier_1_2_tournaments() -> Set[str]:
    """Return normalised names appearing on Liquipedia's Tier 1/2 pages."""
    global _cached_names, _cache_until
    with _lock:
        if _cached_names and time.monotonic() < _cache_until:
            return _cached_names
        try:
            names = _fetch_tier_page("Tier_1_Tournaments") | _fetch_tier_page("Tier_2_Tournaments")
        except Exception:
            return _cached_names
        _cached_names = names
        _cache_until = time.monotonic() + CACHE_TTL_SECONDS
        return _cached_names


def is_tier_1_or_2(title: str) -> bool:
    """Match a DLTV event title against Liquipedia's Tier 1/2 tournament names."""
    candidate = _normalise(title)
    if not candidate:
        return False
    for tournament in tier_1_2_tournaments():
        if candidate == tournament or (len(tournament) >= 12 and tournament in candidate) or (
            len(candidate) >= 12 and candidate in tournament
        ):
            return True
    return False
