"""Real odds backend example: the-odds-api.com (v4).

Setup:
  1. Sign up at https://the-odds-api.com/  (paid, ~$79/mo for
     100k requests).
  2. Get an API key.
  3. Set `ODDS_API_KEY=<key>` in the env.
  4. Set `ODDS_BACKEND=business.odds_backends.TheOddsApiBackend`.

The v4 API at `https://api.the-odds-api.com/v4/sports/esports_dota2/odds/`
returns a list of upcoming + live Dota 2 matches with one row
per bookmaker, each with `markets[].outcomes[]` for h2h / totals /
spreads.  We map:
  * h2h    → OddsQuote(market="winner", selection="P1" or "P2")
  * totals → OddsQuote(market="total_kills", selection="over_<T>")
  * Over/Under on a separate "spreads"-like market is what
    some books use for "match duration" (e.g. over 35.5 min
    on a sub-market).  We don't have a clean field for
    "duration minutes" out of the box, so duration is
    best-effort.

Live odds are exposed via `oddsFormat=decimal&dateFormat=iso`
and `regions=eu,us,uk`.  Free tier is 500 req/mo; paid is
100k+ req/mo.

This backend is here as a reference implementation.  Switching
to api-sport.ru, Stratz, or a self-hosted scrape is a matter
of replacing `get_quotes()` with the same return shape.
"""
from __future__ import annotations

import os
import time
import urllib.request
import urllib.error
import json
from typing import Any, Dict, List, Optional

from .odds import OddsBackend, OddsQuote

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "esports_dota2"


class TheOddsApiBackend:
    """Reference backend for the-odds-api.com v4.  Enable by setting
    `ODDS_API_KEY=<key>` and `ODDS_BACKEND=business.odds_backends.TheOddsApiBackend`.
    """
    name = "the_odds_api"

    def __init__(self) -> None:
        self.api_key = os.environ.get("ODDS_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY env var is required for TheOddsApiBackend")
        # 60-second in-memory cache — the front-end asks for the
        # same match_id on every board rebuild (5s), so we'd
        # otherwise burn ~720 req/hr per live match.
        self._cache: Dict[int, List[OddsQuote]] = {}
        self._cache_ts: Dict[int, float] = {}
        self._cache_ttl_sec = 60.0

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        now = time.monotonic()
        ts = self._cache_ts.get(int(match_id))
        if ts is not None and (now - ts) < self._cache_ttl_sec:
            return list(self._cache.get(int(match_id)) or [])

        # The v4 API doesn't accept Steam match_id directly — it
        # lists events by `id` (their internal id) or you can list
        # all upcoming + live.  We list live and match by team
        # names if the caller passed a `match_id` arg that we
        # recognise; otherwise we return [].
        # In practice, `match_id` here is a Steam match_id; the
        # only reliable way to bridge is by team names, which
        # we'd need the caller to pass.  The simplest approach
        # is to expose a separate `get_all_live_quotes()` for the
        # front-end to call and match on the client side.
        # For now: return [] for unknown match_ids.  See
        # `get_all_live_quotes()` for the full snapshot.
        return []

    def get_all_live_quotes(self) -> Dict[str, List[OddsQuote]]:
        """Return all live Dota 2 quotes keyed by bookmaker-team-pair.

        Bridge: the front-end can call this and match our
        Steam `team_id` to the bookmaker's `P1`/`P2` selection
        by team name.  The matching is fuzzy (e.g. "Team Spirit"
        vs "Spirit") — see `business/odds_match.py` for the
        planned matcher.
        """
        url = (
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds/"
            f"?apiKey={self.api_key}&regions=eu&markets=h2h,totals"
            f"&oddsFormat=decimal&dateFormat=iso"
        )
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "dota-analyst/0.4 (research; +https://github.com/qzmeister/dota_analyst)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            print(f"[odds] the-odds-api fetch failed: {exc}")
            return {}
        out: Dict[str, List[OddsQuote]] = {}
        for event in data or []:
            key = f"{event.get('home_team','')}|{event.get('away_team','')}"
            quotes: List[OddsQuote] = []
            for bk in (event.get("bookmakers") or []):
                bk_name = bk.get("title") or bk.get("key") or "unknown"
                for market in (bk.get("markets") or []):
                    mkey = market.get("key")
                    if mkey == "h2h":
                        # h2h: list of outcomes with `name` and `price`
                        # Outcomes come in team-name order matching
                        # home_team / away_team.  We expose them as
                        # "P1" / "P2" relative to the bookmaker's
                        # order.
                        outcomes = market.get("outcomes") or []
                        if len(outcomes) >= 2:
                            quotes.append(OddsQuote(
                                market="winner", selection="P1",
                                decimal_odds=float(outcomes[0].get("price") or 0),
                                bookmaker=bk_name,
                                raw={"name": outcomes[0].get("name")},
                            ))
                            quotes.append(OddsQuote(
                                market="winner", selection="P2",
                                decimal_odds=float(outcomes[1].get("price") or 0),
                                bookmaker=bk_name,
                                raw={"name": outcomes[1].get("name")},
                            ))
                    elif mkey == "totals":
                        # totals: list of outcomes each with
                        # `name` ("Over"/"Under") + `point` (the
                        # threshold).  We expose as "over_<T>" /
                        # "under_<T>".
                        for oc in (market.get("outcomes") or []):
                            side = (oc.get("name") or "").lower()
                            point = oc.get("point")
                            if not point:
                                continue
                            if side not in ("over", "under"):
                                continue
                            try:
                                point_f = float(point)
                            except (TypeError, ValueError):
                                continue
                            quotes.append(OddsQuote(
                                market="total_kills",
                                selection=f"{side}_{point_f:g}",
                                decimal_odds=float(oc.get("price") or 0),
                                bookmaker=bk_name,
                            ))
            out[key] = quotes
        return out


# Reference: api.api-sport.ru
# --------------
# If you have an api-sport.ru key (paid), drop a sibling class
# `ApiSportRuBackend` that:
#   * GET https://api.api-sport.ru/v1/esports/dota2/odds?bookmaker=betboom
#   * Map their `odds[].h2h` / `odds[].totals` to OddsQuote the
#     same way as the-odds-api above.
# The interface (OddsBackend / get_quotes) stays the same.
