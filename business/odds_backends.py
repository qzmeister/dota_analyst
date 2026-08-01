"""Bookmaker odds backends (v0.4.1+).

This file ships one backend per aggregator:

  * `BetBoomBackend`    — replays user cookies + JWT against
                          siteapi.betboom.ru.  Live odds path
                          is undocumented / WS-only as of
                          2026-07-31 — currently returns [].
  * `OddsApiIOBackend`  — REST poller against
                          https://api.odds-api.io/v3 (15 esports
                          titles, 265+ bookmakers).  Free tier
                          100 req/hour + 2 bookmakers.  Recommended
                          default for `ODDS_BACKEND`.

The `OddsBackend` Protocol + `OddsQuote` dataclass live in
`business.odds`.  Both backends conform to that protocol — the
live card doesn't know which one is wired up.

Why a separate backend per aggregator: each one has its own auth
shape, its own rate limit, and its own quirks (BetBoom is
session-cookie based, odds-api.io is a simple API key in a query
param, Stratz is GraphQL).  Hiding those behind a common Protocol
keeps the live card simple.
"""


from __future__ import annotations

import importlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .odds import OddsBackend, OddsQuote
from .odds_match import normalize_team_name, team_pair_key
from ._logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# BetBoom.ru backend (legacy stub)
# --------------------------------------------------------------------------- #
#
# The class below used to be the only thing in this file.  Its live
# odds path is undocumented / WS-only as of 2026-07-31; the WS
# endpoints are reachable from a Russian IP (ru-ws2.sporthub.bet)
# but the data plane (fw-wc2.sportbook.bet) is parked on Namecheap
# and not resolvable from the public Internet.  We keep the class
# around so users with a known-good Russian ISP can still wire it
# up; the default backend is now `OddsApiIOBackend` further down.
#
# Approach: we don't re-implement login (Cloudflare + Fingerprint.js +
# reCAPTCHA + Group-IB block that).  Instead, the user logs in once
# in a real browser, exports the cookies + the auth-token from the
# login response, and we replay them.
#
# The cookies / token are short-lived (csrf_token has a 1y TTL but
# can rotate; the JWT expires 6h).  The user has to refresh them
# every so often.  When BETBOOM_AUTH_TOKEN is missing, the backend
# falls back to "no quotes" rather than failing the live card.
#
# Real auth flow (HAR 2026-07-31):
#   URL:     https://betboom.ru/api/auth/login
#   Method:  POST
#   Headers: x-platform: web
#   Body:    {phone, password, fingerprint, fingerprint_request_id,
#             is_wrong_data_auth_support, ym_id, captcha_key,
#             features.is_support_conditional_captcha}
#   Server:  QRATOR (not Cloudflare)
#   Response: {code: 200, status: "OK", token: <JWT, 6h>,
#              refresh_token: <JWT, 1y), trace_id}
#
# Cookie distribution:
#   * betboom.ru          — SPA cookies (mindbox, Yandex, theme,
#                            authorizedId, directCrm, GIB fingerprint,
#                            Intercom; csrf_token NOT here)
#   * siteapi.betboom.ru  — only `csrf_token` (JWT, 1y TTL) is
#                            bound to this API host
#   * The /auth/login response itself uses envelope
#     {code, status, ...} where code=200 means OK and code=404
#     means "path not found" (but the HTTP status is 200).
#
# Env vars (loaded from .env.betboom — gitignored):
#   BETBOOM_PHONE          — phone (logged only)
#   BETBOOM_AUTH_TOKEN     — JWT from /auth/login response body
#                            (used as Authorization: Bearer)
#   BETBOOM_REFRESH_TOKEN  — JWT refresh token, 1y TTL (unused yet)
#   BETBOOM_COOKIES        — full "name=value; name=value; ..."
#                            cookie string from the authed browser.
#                            Includes csrf_token, GIB fingerprint,
#                            Yandex Metrika, etc.
#   BETBOOM_BASE_URL       — override the API base (default
#                            https://siteapi.betboom.ru).
#
# Why this design:
#   * No CAPTCHA solving in our code (2captha is the right call if
#     we want automation; we deliberately don't for now).
#   * No persistent Playwright browser (heavy + fragile).
#   * User does the login once in their own browser, exports state,
#     paste into .env.betboom.  Restarts pick it up.
#   * If/when the session expires, the user re-logs-in and we get
#     a fresh batch of creds.  The backend logs a clear "session
#     expired" so the user knows.
# --------------------------------------------------------------------------- #


import os
import time
import urllib.request
import urllib.error
import json
import re
from typing import Any, Dict, List, Optional

from .odds import OddsBackend, OddsQuote

# v0.4.1: live API base is `siteapi.betboom.ru` (the cookie
# distribution in the user's browser session shows that
# `csrf_token` is the ONLY cookie bound to that domain — every
# other cookie (GIB fingerprint, Yandex, theme, etc.) is bound
# to `betboom.ru`.  This is the standard pattern: SPA on the
# main domain, data API on a separate host, csrf_token is the
# cross-origin auth.  Login itself stays on `betboom.ru`.
# Override with BETBOOM_BASE_URL env var.
BASE = os.environ.get("BETBOOM_BASE_URL", "https://siteapi.betboom.ru").rstrip("/")
ORIGIN = "https://betboom.ru"  # CORS origin; SPA shell is on betboom.ru

# v0.4.1: live data is on the WebSocket, not the HTTP API.  Reverse
# engineering (2026-07-31) found:
#   * BetBoom HTML config exposes:
#       WS_URL             = wss://ws.{current_domain}:444
#       SPORTBOOK_API_URL  = https://siteapi.betboom.ru/api/site_api/v1
#       SITE_V3_API_URL    = https://{current_domain}/api
#   * The HTTP API at siteapi.betboom.ru/api/site_api/v1/* returns
#     `{code: 404, status: "NOT_FOUND", ...}` for every path we
#     tried (200+ candidates including /live, /feed/live/dota2,
#     /esports, /events, /matches, /sport/dota2, etc.) — auth is
#     accepted (no 401/403) but no path is exposed.
#   * The SPA bundle's accounting_ws reducer wires up a WebSocket
#     to `${WS_URL}/api/accounting_ws/v1`.  Handshake succeeds;
#     server then sends PING every ~5s.
#   * Messages are Centrifugo v6 protobuf (binary frames, not JSON).
#     PING is a WebSocket protocol-level ping (op=0x9) with payload
#     "PING"; we MUST reply with a protocol-level pong (op=0xA)
#     carrying the same payload — not a binary command.
#   * ConnectRequest: `{ name, version }` (anonymous) works and the
#     server replies with `Reply{code:200, reason:"OK"}`.  JWT in
#     `token` field currently rejects with "Unsubscribed" — the
#     site likely signs Centrifugo tokens server-side and our
#     `BETBOOM_AUTH_TOKEN` isn't the right shape.
#   * Subscribe / RPC names: 200+ candidates tried, every one
#     returned BAD_REQUEST.  Without a real HAR of the WS frames
#     (DevTools → Network → WS, copy frames while on the live
#     betting page) we can't reverse-engineer the channel naming.
#
# So the HTTP backend in this file is a stub until the user
# supplies either a live-page WS HAR or the channel naming.  The
# shape below (OddsBackend protocol, OddsQuote dataclass) is
# still correct and ready to wire up once a real parser exists.
#


class BetBoomBackend:
    """Replay-cookies backend.  Reads creds from env on every call so
    a quick `docker compose restart business` after editing
    .env.betboom picks up a fresh session.

    Public methods:
      get_quotes(match_id) — same as the rest of the backends
      probe_session()       — verify creds are still valid (200 vs 401)
      live_dota2()         — raw call to the live dota2 endpoint,
                              returns whatever shape the API gives us
                              (used by `get_all_live_quotes()` and for
                              debugging the parser)
    """
    name = "betboom"

    def __init__(self) -> None:
        # We read creds lazily on every call.  This way updating
        # .env.betboom and bouncing the container is enough; no
        # need to also reload the backend instance.
        self._last_warn_no_creds: float = 0.0
        self._last_session_check: float = 0.0
        self._session_ok: Optional[bool] = None
        self._session_check_ttl = 30.0  # seconds

    def _creds(self) -> Dict[str, str]:
        return {
            "phone":       os.environ.get("BETBOOM_PHONE", "").strip(),
            # The JWT "token" field from the /auth/login response.
            # Used as `Authorization: Bearer <token>`.  Note:
            # there is no separate `cf_clearance` — BetBoom is
            # behind QRATOR, not Cloudflare, so the auth flow
            # is a different anti-bot stack.
            "auth_token":  os.environ.get("BETBOOM_AUTH_TOKEN", "").strip(),
            "refresh_token": os.environ.get("BETBOOM_REFRESH_TOKEN", "").strip(),
            # All cookies as a single semicolon-separated
            # "name=value" string.  Captured from the
            # authed browser session; replay verbatim.
            "cookies":     os.environ.get("BETBOOM_COOKIES", "").strip(),
        }

    def _has_creds(self, c: Dict[str, str]) -> bool:
        # We need at least the auth token.  Cookies are
        # strongly recommended for non-GET endpoints (CSRF
        # is enforced via the `csrf_token` cookie + the
        # matching header that mirrors it) but not strictly
        # required for /api/live reads.
        return bool(c["auth_token"])

    def _cookie_header(self, c: Dict[str, str]) -> str:
        return c["cookies"]

    def _request(
        self, method: str, path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        # v0.4.1: paths are relative to `https://betboom.ru` now
        # (was siteapi.betboom.ru).  If the caller passes an
        # absolute URL, use it as-is.
        if path.startswith("http"):
            url = path
        else:
            url = BASE + path
        if params:
            from urllib.parse import urlencode
            url = url + "?" + urlencode(params)
        c = self._creds()
        if not self._has_creds(c):
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                print(f"[odds] BetBoomBackend: missing BETBOOM_AUTH_TOKEN — "
                      f"see .env.betboom template")
                self._last_warn_no_creds = now
            return None
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x64) dota-analyst/0.4",
            "Accept": "application/json",
            "x-platform": "web",
            "Origin": ORIGIN,
            "Referer": ORIGIN + "/",
            "Cookie": self._cookie_header(c),
            # The HAR's /auth/login response sends the token
            # back as `Authorization: Bearer <token>` for all
            # subsequent API calls.  The CORS allow-list also
            # mentions X-Access-Token; we send both so the
            # server's auth middleware picks the one it
            # understands (some endpoints check both).
            "Authorization": f"Bearer {c['auth_token']}",
            "X-Access-Token": c["auth_token"],
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json;charset=UTF-8"
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
        else:
            req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (401, 403):
                # Session expired or cf_clearance was rejected.
                # Tell the user loudly — they need to refresh
                # .env.betboom.
                self._session_ok = False
                now = time.monotonic()
                if now - self._last_warn_no_creds > 60:
                    try:
                        body = e.read()[:200].decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    print(f"[odds] BetBoom session expired ({code}): {body[:120]!r}")
                    print(f"[odds] Re-login at https://betboom.ru and update .env.betboom")
                    self._last_warn_no_creds = now
            else:
                # Other 4xx/5xx — log once per minute, return None
                now = time.monotonic()
                if now - self._last_warn_no_creds > 60:
                    try:
                        body = e.read()[:200].decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    print(f"[odds] BetBoom {method} {path} -> HTTP {code}: {body[:120]!r}")
                    self._last_warn_no_creds = now
            return None
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"[odds] BetBoom {method} {path} network/parse error: {exc}")
            return None

    def probe_session(self) -> bool:
        """Hit a lightweight authed endpoint to verify the creds
        are still valid.  Caches the result for `_session_check_ttl`."""
        now = time.monotonic()
        if self._session_ok is not None and (now - self._last_session_check) < self._session_check_ttl:
            return self._session_ok
        # Use the /api/live endpoint as the probe — public-ish
        # but auth-required for real data.  A 200 means the
        # auth token works; 401/403 means the session is dead.
        resp = self._request("GET", "/api/live")
        self._session_ok = resp is not None
        self._last_session_check = now
        return self._session_ok

    def live_dota2(self) -> List[Dict[str, Any]]:
        """Fetch the live dota2 events list.  Returns whatever shape
        the BetBoom API gives us — the caller (or a future parser)
        is responsible for mapping to OddsQuote.

        Path is heuristic — we don't actually know BetBoom's
        exact endpoint yet (waiting on a live-page HAR from
        the user).  We try the most common shapes:
          * /api/site_api/v1/live*  (BetBoom's "site_api" gateway)
          * /api/live*              (the SPA's possibly-shorter path)
          * /api/feed/live*         (some betting sites use a feed)
          * /api/esports/dota2/live (esports-scoped path)
          * /api/events*            (generic events list)

        The first path that returns HTTP 200 with code=200 (NOT
        code=404) AND a non-empty list wins.  Application-level
        404s on siteapi.betboom.ru return HTTP 200 with
        {code: 404, ...} so we filter on the envelope code.
        """
        candidate_paths = [
            # Most likely: the site_api v1 gateway (used by the
            # SPA's own code; cookie domain confirms).
            "/api/site_api/v1/live",
            "/api/site_api/v1/live/dota2",
            "/api/site_api/v1/live/esports",
            "/api/site_api/v1/live/events",
            "/api/site_api/v1/live?sport=dota2",
            "/api/site_api/v1/live?status=1",
            "/api/site_api/v1/live?status=live",
            "/api/site_api/v1/live?category=esports",
            "/api/site_api/v1/feed/live",
            "/api/site_api/v1/feed/live/dota2",
            "/api/site_api/v1/feed/live/esports",
            "/api/site_api/v1/esports/dota2/live",
            "/api/site_api/v1/esports/live",
            "/api/site_api/v1/events?status=live",
            "/api/site_api/v1/events?status=1",
            "/api/site_api/v1/matches?status=live",
            "/api/site_api/v1/matches/live",
            "/api/site_api/v1/matches/live?game=dota2",
            # Shorter paths on the same host (some apps omit the
            # /site_api/v1 prefix).
            "/api/live",
            "/api/live/dota2",
            "/api/live/esports",
            "/api/live?status=1",
            "/api/live?status=live",
            "/api/live?sport=dota2",
            "/api/live?category=esports",
            "/api/feed/live",
            "/api/feed/live/dota2",
            "/api/esports/dota2/live",
            "/api/esports/live",
            "/api/events?status=live",
            "/api/events?status=1",
            "/api/matches?status=live",
        ]
        for path in candidate_paths:
            resp = self._request("GET", path)
            if not resp:
                continue
            # siteapi.betboom.ru returns HTTP 200 with code:404 in
            # the body for unknown paths.  Treat those as "not
            # found" and keep trying.
            code = resp.get("code")
            if code is not None and code != 200:
                continue
            for key in ("data", "events", "matches", "items", "result",
                        "sports", "leagues", "data_list", "list",
                        "games", "fixtures"):
                items = resp.get(key)
                if isinstance(items, list) and items:
                    return items
            if isinstance(resp, list) and resp:
                return resp
        return []

    def get_all_live_quotes(self) -> Dict[str, List[OddsQuote]]:
        """Return all live Dota 2 quotes keyed by team-pair string.

        The frontend (or a future `odds_match.py` fuzzy matcher)
        maps our `(first_team, second_team)` to these keys.
        We don't yet know the exact BetBoom payload shape; this
        method will get fleshed out once the user provides a
        sample of the live list response.
        """
        events = self.live_dota2()
        if not events:
            return {}
        out: Dict[str, List[OddsQuote]] = {}
        for ev in events:
            # Best-effort team name extraction — adjust to real
            # shape once we see one.  Common keys: team_a/team_b,
            # home/away, first/second, opponents.
            team_a = (
                ev.get("team_a_name") or
                ev.get("home_name") or
                ev.get("first_team_name") or
                ev.get("team_a") or
                ev.get("home") or
                ev.get("title") or
                (ev.get("opponents") or [{}])[0].get("name") if ev.get("opponents") else ""
            )
            team_b = (
                ev.get("team_b_name") or
                ev.get("away_name") or
                ev.get("second_team_name") or
                ev.get("team_b") or
                ev.get("away") or
                (ev.get("opponents") or [{}, {}])[1].get("name") if ev.get("opponents") else ""
            )
            if not (team_a and team_b):
                continue
            key = f"{team_a}|{team_b}"
            quotes: List[OddsQuote] = []
            for market in (ev.get("markets") or []):
                mkey = (market.get("key") or market.get("type") or "").lower()
                for oc in (market.get("outcomes") or []):
                    price = oc.get("price") or oc.get("odds")
                    if not price:
                        continue
                    try:
                        price_f = float(price)
                    except (TypeError, ValueError):
                        continue
                    if mkey in ("h2h", "winner", "match_winner", "1x2"):
                        sel_name = (oc.get("name") or "").lower()
                        # Map name -> P1/P2
                        if "1" in sel_name or "p1" in sel_name or sel_name == team_a.lower():
                            quotes.append(OddsQuote(
                                market="winner", selection="P1",
                                decimal_odds=price_f, bookmaker="betboom",
                            ))
                        elif "2" in sel_name or "p2" in sel_name or sel_name == team_b.lower():
                            quotes.append(OddsQuote(
                                market="winner", selection="P2",
                                decimal_odds=price_f, bookmaker="betboom",
                            ))
                    elif mkey in ("totals", "total_kills", "total"):
                        side = (oc.get("name") or "").lower()
                        point = oc.get("point")
                        if side in ("over", "under") and point is not None:
                            try:
                                pt = float(point)
                            except (TypeError, ValueError):
                                continue
                            quotes.append(OddsQuote(
                                market="total_kills",
                                selection=f"{side}_{pt:g}",
                                decimal_odds=price_f, bookmaker="betboom",
                            ))
                    # duration markets — BetBoom uses different
                    # names depending on sport; we'll add the
                    # parser once we see the real shape.
            if quotes:
                out[key] = quotes
        return out

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        # v0.4.1: the live card asks for a single match_id
        # (Steam), but BetBoom quotes are keyed by team-pair.
        # We don't have a Steam <-> BetBoom mapper yet (that
        # would be a fuzzy match on team names + maybe league).
        # For now, the simplest path is: have the front-end call
        # /api/odds/all, cache it, and lookup by team-pair.  See
        # the wiring in business/app.py for the cache hook.
        # Direct per-match_id lookup returns []; the full snapshot
        # is fetched by get_all_live_quotes().
        return []


# To enable, set:
#   ODDS_BACKEND=business.odds_backends.BetBoomBackend
#   BETBOOM_AUTH_TOKEN=<from /auth/login response body>
#   BETBOOM_CF_CLEARANCE=<from Set-Cookie header of /auth/login>
#   BETBOOM_SESSION_COOKIE=session=eyJ...
#   BETBOOM_OTHER_COOKIES=_csrf=...; device_id=...; _ym_uid=...


# =========================================================================== #
# OddsApiIOBackend  (v0.4.2 — recommended default for ODDS_BACKEND)
# =========================================================================== #
#
# REST poller against https://api.odds-api.io/v3 — a free / paid
# aggregator with 15 esports titles and 265+ bookmakers.
#
# Free tier: 100 req/hour, 2 bookmakers (recreational only).  Paid
# tier: 5,000 req/hour, sharp books, exchanges, prediction markets.
# Dota 2 is exposed as sport=esports.  Live events come with a
# `clock` object (minute / period) and `scores.periods.mapN` for
# per-map scores — directly useful for the live card.
#
# Auth: a single `?apiKey=...` query param on every request (except
# the public `/sports` and `/bookmakers` endpoints).  No headers,
# no cookies, no signing.  Trivially cachable.
#
# Response shape (relevant subset of /v3/odds):
#   {
#     "id": 123456,                    # eventId, int
#     "home": "Team Liquid",
#     "away": "Gaimin Gladiators",
#     "date": "2026-07-31T18:00:00Z",
#     "status": "live",                # pending | live | settled
#     "sport": {"name": "Esports", "slug": "esports"},
#     "league": {"name": "BLAST Slam", "slug": "..."},
#     "scores": {"home": 1, "away": 0, "periods": {"map1": {"home": 1, "away": 0}}},
#     "bookmakers": {
#       "Bet365": [
#         {"name": "ML", "updatedAt": "...", "odds": [{"home": "1.85", "away": "2.40"}]},
#         {"name": "Asian Handicap", "odds": [{"hdp": -0.5, "home": "1.95", "away": "1.85"}]},
#         {"name": "Over/Under", "odds": [{"max": 2.5, "over": "1.90", "under": "1.90"}]}
#       ]
#     }
#   }
#
# We pick the markets we care about:
#   * "ML"        → market="winner"   (P1/P2 — Dota 2 = no draw)
#   * "Over/Under" → market="total_kills" (over_N.N / under_N.N)
#   * "Asian Handicap" → SKIP (Dota 2 has no map handicap in this shape)
#
# We only surface each market ONCE per bookmaker per event — the
# live card takes the BEST price per side (lowest implied prob).
# Other bookmakers in the response are kept on the quote for
# debugging; the live card's `compute_edge_for_card` already picks
# the best across the list.
#
# Env vars (loaded from .env / .env.betboom):
#   ODDS_API_KEY              — required, from https://odds-api.io
#   ODDS_API_BASE_URL         — optional, override for testing
#   ODDS_API_BOOKMAKERS       — optional, comma-separated list of
#                               bookmaker names to filter.  Free
#                               tier only has 2 slots, so pick the
#                               sharpest available (e.g.
#                               "Pinnacle,Bet365").
#   ODDS_LIVE_POLL_SEC        — poller interval, see business/odds_live.py
#
# To enable:
#   ODDS_BACKEND=business.odds_backends.OddsApiIOBackend
#   ODDS_API_KEY=<your key from odds-api.io>
# ---------------------------------------------------------------------------


class OddsApiIOBackend:
    """REST-poller backend for https://api.odds-api.io/v3.

    See module docstring above for the full contract.  The class
    conforms to the `OddsBackend` Protocol in `business.odds` and
    is loaded by `get_backend()` when `ODDS_BACKEND` is set to
    `"business.odds_backends.OddsApiIOBackend"`.

    Public methods:
      get_quotes(match_id)        — returns [] (we don't have
                                    steam_id ↔ event_id mapping
                                    in this method; the poller
                                    does the snapshot-based
                                    lookup, see `get_all_live_quotes`)
      get_all_live_quotes()      — full snapshot keyed by
                                    `"{normalized_a}|{normalized_b}"`
                                    (so the live card's exact-
                                    match lookup works)
      probe_session()             — verify the key works
    """
    name = "odds-api.io"

    # Sport slug for Dota 2 (per /v3/sports response, 2026-07-31)
    SPORT_SLUG = "esports"

    # We also accept Dota-2-specific league slugs as a filter, in
    # case `/events/live` ever returns the wrong sport.  League
    # slugs we know about (from the Dota 2 page; expand as we
    # discover more):
    DOTA2_LEAGUE_HINTS = (
        "dota-2-blast", "dota-2-esl-one", "dota-2-dpc", "dota-2-ti",
        "dota-2-rlcs",  # not actually a thing; keep for future
        "dota-2",
        "esports-dota-2",
    )

    # Substrings of participant names that strongly suggest a Dota 2
    # match.  Used as a tiebreaker when the league slug is missing
    # or doesn't contain "dota" — e.g. a third-party organiser
    # that doesn't tag their league with "dota" in the slug.
    # Keep this list small and conservative: a false positive
    # here means we pay for /v3/odds/multi on a non-Dota2 match
    # and the live card shows "no odds" anyway.  Better to miss
    # and let the live card show empty than to spam the API.
    DOTA2_TEAM_TOKENS = (
        "team liquid", "team spirit", "team falcons", "team tidebound",
        "team secret", "team heroic", "og", "tsm", "eg ",
        "evil geniuses", "betboom", "gaimin gladiators", "parivision",
        "nouns", "aurora", "xtreme gaming", "talon", "tundra",
        "shopify rebellion", "mouz", "lgd", "psg.lgd", "rng",
        "invictus gaming", "virtus.pro", "vp ", "spirit",
    )

    def __init__(self) -> None:
        self._last_warn_no_creds: float = 0.0
        self._last_session_check: float = 0.0
        self._session_ok: Optional[bool] = None
        self._session_check_ttl = 60.0  # seconds
        self._base_url = os.environ.get(
            "ODDS_API_BASE_URL", "https://api.odds-api.io/v3"
        ).rstrip("/")
        # Comma-separated bookmaker filter; default = use whatever
        # the user's plan has.  When set, we send this on every
        # /odds call so we only pay for the books we care about.
        self._bookmakers_env = (
            os.environ.get("ODDS_API_BOOKMAKERS", "").strip()
        )
        # Bookmakers we discovered from /v3/bookmakers/selected at
        # the last probe.  Cached for 1 hour so we don't re-probe
        # on every poll cycle.  When env is not set, we use these.
        self._selected_bookmakers: List[str] = []
        self._selected_bookmakers_ts: float = 0.0
        self._selected_bookmakers_ttl = 3600.0  # 1 hour

    # -- env helpers -----------------------------------------------------

    def _key(self) -> str:
        return os.environ.get("ODDS_API_KEY", "").strip()

    def _has_key(self) -> bool:
        return bool(self._key())

    # -- low-level HTTP --------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> Optional[Any]:
        """Thin wrapper around urllib that injects the apiKey.

        Returns parsed JSON (list or dict) on 2xx, None on any
        error.  Errors are logged once per minute to avoid log
        floods when the upstream is down.
        """
        if not self._has_key():
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                log.warning(
                    "OddsApiIOBackend: missing ODDS_API_KEY — see .env template"
                )
                self._last_warn_no_creds = now
            return None
        # Build URL.  apiKey goes in the query string.
        qp: Dict[str, str] = {"apiKey": self._key()}
        if params:
            qp.update({k: v for k, v in params.items() if v is not None})
        url = self._base_url + path + "?" + urllib.parse.urlencode(qp)
        req = urllib.request.Request(
            url,
            method=method,
            headers={
                "User-Agent": "dota-analyst/0.4.2 (+https://odds-api.io)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if not body:
                    return None
                try:
                    return json.loads(body.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    log.warning(
                        "OddsApiIOBackend: %s %s returned non-JSON (%d bytes): %s",
                        method, path, len(body), exc,
                    )
                    return None
        except urllib.error.HTTPError as e:
            self._handle_http_error(method, path, e)
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                log.warning(
                    "OddsApiIOBackend: %s %s network error: %s",
                    method, path, e,
                )
                self._last_warn_no_creds = now
            return None

    def _handle_http_error(
        self, method: str, path: str, e: "urllib.error.HTTPError",
    ) -> None:
        """Map HTTPError to a one-line log; mark session dead on 401/403."""
        code = e.code
        try:
            body = e.read()[:200].decode("utf-8", errors="replace")
        except Exception:
            body = ""
        now = time.monotonic()
        if now - self._last_warn_no_creds > 60:
            if code in (401, 403):
                self._session_ok = False
                log.warning(
                    "OddsApiIOBackend: %s %s -> HTTP %d (auth?): %s",
                    method, path, code, body[:120],
                )
            elif code == 429:
                log.warning(
                    "OddsApiIOBackend: %s %s -> HTTP 429 (rate limit): %s",
                    method, path, body[:120],
                )
            else:
                log.warning(
                    "OddsApiIOBackend: %s %s -> HTTP %d: %s",
                    method, path, code, body[:120],
                )
            self._last_warn_no_creds = now

    # -- session probe ---------------------------------------------------

    def probe_session(self) -> bool:
        """Verify the API key works.  Caches the result for
        `_session_check_ttl` seconds.

        The cheapest authenticated endpoint is `/bookmakers/selected`
        which returns the user's enabled bookmakers.  We use that.

        Returns True if the last probe returned 2xx, False on
        401/403.  Returns the *previous* value on transient
        network errors so the caller doesn't flip to "dead" on
        a one-off timeout.

        Side effects:
          * Sets `self._session_ok` to True / False / leaves alone
            depending on outcome (see above).
          * Refreshes `self._selected_bookmakers` from the
            response (used by `_effective_bookmakers`).
        """
        now = time.monotonic()
        if (
            self._session_ok is not None
            and (now - self._last_session_check) < self._session_check_ttl
        ):
            return self._session_ok
        # We can't tell apart 401/403 from network errors with
        # `_request()` alone (it returns None for both).  Make
        # the call manually here so we can classify.
        if not self._has_key():
            self._session_ok = False
            return False
        qp: Dict[str, str] = {"apiKey": self._key()}
        url = self._base_url + "/bookmakers/selected?" + urllib.parse.urlencode(qp)
        req = urllib.request.Request(
            url, method="GET",
            headers={"User-Agent": "dota-analyst/0.4.2", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as r:
                body = r.read().decode("utf-8", errors="replace")
                try:
                    resp = json.loads(body) if body else None
                except json.JSONDecodeError:
                    resp = None
                # 200 with a list (possibly empty) = OK
                self._session_ok = True
                self._last_session_check = now
                # Refresh selected-bookmakers cache
                if isinstance(resp, dict) and isinstance(resp.get("bookmakers"), list):
                    self._selected_bookmakers = [
                        str(b) for b in resp["bookmakers"] if b
                    ]
                    self._selected_bookmakers_ts = now
                elif isinstance(resp, list):
                    self._selected_bookmakers = [str(b) for b in resp if b]
                    self._selected_bookmakers_ts = now
                return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                # Auth really failed.  Mark dead.
                self._session_ok = False
                self._last_session_check = now
                self._handle_http_error("GET", "/bookmakers/selected", e)
                return False
            # Other HTTP errors (5xx, 429) — log but don't mark dead
            self._handle_http_error("GET", "/bookmakers/selected", e)
            # Leave _session_ok as-is (could be True from prior
            # successful probe, or None if first try).
            return bool(self._session_ok)
        except (urllib.error.URLError, TimeoutError) as e:
            # Network blip — log once per minute but DON'T change
            # _session_ok.  Returning the previous value keeps
            # the data-fetch path open for the actual call.
            if time.monotonic() - self._last_warn_no_creds > 60:
                log.warning(
                    "OddsApiIOBackend: probe network error: %s", e,
                )
                self._last_warn_no_creds = time.monotonic()
            return bool(self._session_ok)

    def _effective_bookmakers(self) -> Optional[str]:
        """Pick the bookmaker list to send on /v3/odds/multi.

        Priority:
          1. ODDS_API_BOOKMAKERS env var (explicit override)
          2. The list we cached from /v3/bookmakers/selected at
             the last probe (the user's enabled books on their
             plan; required for free tier which rejects calls
             without a bookmakers filter)
          3. None — let the API use its default (paid plans only)

        Returns the comma-separated string to send, or None.
        """
        if self._bookmakers_env:
            return self._bookmakers_env
        if self._selected_bookmakers:
            return ",".join(self._selected_bookmakers)
        return None

    # -- live events fetch ----------------------------------------------

    def _fetch_live_dota_events(self) -> List[Dict[str, Any]]:
        """GET /v3/events/live and filter for Dota 2.

        The free-tier `/events/live` does NOT accept a sport filter,
        so we fetch all live events and drop the non-Dota-2 ones
        client-side.  A future request to add `?sport=esports` to
        `/events/live` would be a nice optimisation; until then,
        this is one request per poll cycle.

        Heuristics for "is this Dota 2?":
          1. `sport.slug == "esports"` AND
          2. league slug contains one of our DOTA2_LEAGUE_HINTS, OR
             the team names include known Dota 2 team keywords
             ("dota", "team liquid", "og", "tsm", etc.)
        """
        data = self._request("GET", "/events/live")
        if not isinstance(data, list):
            return []
        out: List[Dict[str, Any]] = []
        for ev in data:
            if not isinstance(ev, dict):
                continue
            sport_slug = (
                (ev.get("sport") or {}).get("slug")
                or (ev.get("sport") or {}).get("name")
                or ""
            ).lower()
            if sport_slug != self.SPORT_SLUG:
                # The /events/live endpoint is "all live events";
                # drop non-esports ones first to keep the working
                # set small.
                continue
            league_slug = (
                (ev.get("league") or {}).get("slug")
                or (ev.get("league") or {}).get("name")
                or ""
            ).lower()
            if not self._looks_like_dota2(league_slug, ev):
                continue
            out.append(ev)
        return out

    def _looks_like_dota2(
        self, league_slug: str, event: Dict[str, Any],
    ) -> bool:
        """Heuristic: is this an esports event actually for Dota 2?

        `league_slug` has already been lowercased.  We allow it if
        any of our known hints is a substring; otherwise we look
        for "dota" in the slug itself (covers e.g. `dota-2-blast`
        and `dota-2-esl-one`).

        v0.4.2 (post-key test): the earlier "any non-empty team
        names -> True" fallback was too permissive — it caught
        Counter-Strike / LoL / Valorant events that the umbrella
        sport="Esports" rolled up under Dota 2.  We now require
        a positive Dota-2 indicator from EITHER the league slug
        OR a small set of well-known Dota 2 team tokens.
        """
        if not league_slug:
            # No league info — drop.  The /events/live response
            # always populates `league`, so this is defensive.
            return False
        if "dota" in league_slug:
            return True
        for hint in self.DOTA2_LEAGUE_HINTS:
            if hint in league_slug:
                return True
        # Last-resort: team-name sniff.  Only return True if we
        # see a *known Dota 2 team token* in the participant
        # names.  We deliberately do NOT match on "any non-empty
        # names" because that lets through CS/LoL/Valorant
        # matches that share the umbrella sport="Esports".
        names = " ".join(
            str(event.get(k) or "")
            for k in ("home", "away", "homeName", "awayName")
        ).lower()
        return any(token in names for token in self.DOTA2_TEAM_TOKENS)

    def _fetch_odds_multi(
        self, event_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """GET /v3/odds/multi?eventIds=1,2,3 — up to 10 per call.

        Splits the list into chunks of 10 and makes one request per
        chunk.  The first 10-event chunk is what hits the wire
        for a typical live Dota 2 window (≤ 10 concurrent matches).

        v0.4.2 (post-key test): the API returns 400 "Missing
        bookmakers" when no bookmakers filter is passed on the
        free tier.  We now always include one (env override,
        cached selected list, or — if neither is set — fall back
        to an empty string which the API interprets as "all",
        which works on paid plans but errors on free).
        """
        if not event_ids:
            return []
        book_filter = self._effective_bookmakers() or ""
        results: List[Dict[str, Any]] = []
        for i in range(0, len(event_ids), 10):
            chunk = event_ids[i : i + 10]
            params: Dict[str, str] = {
                "eventIds": ",".join(str(e) for e in chunk),
            }
            if book_filter:
                params["bookmakers"] = book_filter
            data = self._request("GET", "/odds/multi", params=params)
            if isinstance(data, list):
                results.extend([e for e in data if isinstance(e, dict)])
        return results

    # -- parser ---------------------------------------------------------

    # Market names we recognise on the wire.  Anything not in this
    # set is ignored (we don't surface exotic markets for Dota 2).
    _WINNER_MARKET_NAMES = ("ML", "Match Winner", "Moneyline", "1X2")
    _TOTAL_MARKET_NAMES = ("Over/Under", "Total", "Totals", "Total Goals")

    def _event_to_quotes(
        self, event: Dict[str, Any],
    ) -> List[OddsQuote]:
        """Convert one event's full /odds payload to OddsQuote list.

        We pick:
          * market="winner"   (P1/P2)   from "ML"
          * market="total_kills" (over/under_N.N) from "Over/Under"

        The selection names use the bookmaker's first/second team
        labels: the live card's `_compute_odds` flips P1/P2 based
        on `radiant_is_first` so this works regardless of whether
        the bookmaker's "home" is our radiant or dire.
        """
        out: List[OddsQuote] = []
        ev_id = event.get("id")
        home = event.get("home") or event.get("homeName") or ""
        away = event.get("away") or event.get("awayName") or ""
        if not (home and away):
            return out
        for book_name, markets in (event.get("bookmakers") or {}).items():
            for m in markets or []:
                mname = (m.get("name") or "").strip()
                mname_low = mname.lower()
                # ---- Winner (ML) ----
                if any(
                    mname == w or mname_low == w.lower()
                    for w in self._WINNER_MARKET_NAMES
                ):
                    for cell in (m.get("odds") or []):
                        try:
                            h = float(cell.get("home"))
                            a = float(cell.get("away"))
                        except (TypeError, ValueError):
                            continue
                        out.append(OddsQuote(
                            market="winner", selection="P1",
                            decimal_odds=h, bookmaker=str(book_name),
                            raw={"event_id": ev_id, "market": mname,
                                 "side": "home", "updated": m.get("updatedAt")},
                        ))
                        out.append(OddsQuote(
                            market="winner", selection="P2",
                            decimal_odds=a, bookmaker=str(book_name),
                            raw={"event_id": ev_id, "market": mname,
                                 "side": "away", "updated": m.get("updatedAt")},
                        ))
                # ---- Total (Over/Under) ----
                elif any(
                    mname == w or mname_low == w.lower()
                    for w in self._TOTAL_MARKET_NAMES
                ):
                    for cell in (m.get("odds") or []):
                        try:
                            o = float(cell.get("over"))
                            u = float(cell.get("under"))
                        except (TypeError, ValueError):
                            continue
                        # The threshold comes back as either `max`
                        # (legacy) or `total` / `line` (newer).
                        # `max` is the more common one; we fall
                        # back gracefully.
                        threshold = (
                            cell.get("max")
                            or cell.get("total")
                            or cell.get("line")
                            or cell.get("points")
                        )
                        if threshold is None:
                            # Without a threshold we can't surface
                            # it as `over_2.5`; skip rather than
                            # emit a degenerate `over_0`.
                            continue
                        try:
                            thr_f = float(threshold)
                        except (TypeError, ValueError):
                            continue
                        # Dota 2 totals come in as map total
                        # (e.g. 2.5) — not kill totals.  We
                        # still surface them under "total_kills"
                        # because the live card only has one
                        # `total_kills` market slot and most Dota
                        # 2 bettors interpret "Over 2.5 maps" as
                        # the closest equivalent.  A future
                        # refinement could split into a separate
                        # `total_maps` market.
                        out.append(OddsQuote(
                            market="total_kills",
                            selection=f"over_{thr_f:g}",
                            decimal_odds=o, bookmaker=str(book_name),
                            raw={"event_id": ev_id, "market": mname,
                                 "side": "over", "threshold": thr_f,
                                 "updated": m.get("updatedAt")},
                        ))
                        out.append(OddsQuote(
                            market="total_kills",
                            selection=f"under_{thr_f:g}",
                            decimal_odds=u, bookmaker=str(book_name),
                            raw={"event_id": ev_id, "market": mname,
                                 "side": "under", "threshold": thr_f,
                                 "updated": m.get("updatedAt")},
                        ))
                # Anything else: log at debug and skip.
                # Future markets we may want to add:
                #   * "Map Handicap"  → market="map_handicap"
                #   * "Map 1 Winner"  → market="map_winner"
                #   * "First Blood"   → market="first_blood"
        return out

    # -- public API ------------------------------------------------------

    def get_all_live_quotes(self) -> Dict[str, List[OddsQuote]]:
        """Full snapshot for the live-quotes cache.

        Returns `Dict[team_pair_key, List[OddsQuote]]` where
        `team_pair_key = "{normalized_a}|{normalized_b}"` (lower-
        cased, suffixed-stripped — see `business.odds_match`).  The
        live card's `get_quotes_for_teams` uses the same key
        shape so the cache lookup is direct.

        Behaviour:
          1. Refuse if no key is set.
          2. Best-effort session probe — if it 401/403s we
             return {} (the poller logs and tries again next
             cycle).  If the probe TIMES OUT (network blip),
             we still attempt the data fetch — `/events/live`
             and `/odds/multi` will surface the real error if
             auth is actually broken.
          3. Fetch live Dota 2 events from `/events/live`.
          4. Batch-fetch odds via `/odds/multi` (up to 10/call).
          5. Parse each event to OddsQuote list.
          6. Bucket by `team_pair_key(home, away)`.  We DON'T
             add a second bucket under the swapped key — the
             live card already tries both orderings itself.
        """
        if not self._has_key():
            return {}
        # v0.4.2: probe is best-effort.  We distinguish the
        # "session is dead" case (probe returned an HTTPError
        # 401/403) from the "network blip" case (probe timed
        # out).  Only the first blocks the data fetch.
        if self._session_ok is False and self._auth_actually_failed():
            return {}
        events = self._fetch_live_dota_events()
        if not events:
            return {}
        event_ids = [
            int(ev["id"]) for ev in events
            if isinstance(ev.get("id"), (int, float, str))
            and str(ev.get("id")).strip()
        ]
        odds_data = self._fetch_odds_multi(event_ids)
        # Build a quick index from id -> event-with-odds for the
        # parser.  Some events might not have odds (bookmakers
        # don't cover every match); those return [].
        by_id: Dict[Any, Dict[str, Any]] = {
            od.get("id"): od for od in odds_data
        }
        out: Dict[str, List[OddsQuote]] = {}
        for ev in events:
            ev_id = ev.get("id")
            od = by_id.get(ev_id)
            if not isinstance(od, dict):
                continue
            quotes = self._event_to_quotes(od)
            if not quotes:
                continue
            home = od.get("home") or ev.get("home") or ""
            away = od.get("away") or ev.get("away") or ""
            k = team_pair_key(home, away)
            if not k or k == "|":
                # Defensive: normalisation returned empty sides.
                # Skip rather than pollute the cache.
                continue
            out[k] = quotes
        return out

    def _auth_actually_failed(self) -> bool:
        """Did the most recent probe return a 401/403?

        Used to gate `get_all_live_quotes` — we only block on
        *real* auth failure, not on transient network blips.
        `self._session_ok` is only set to False on a 401/403,
        so this is the right thing to check.
        """
        return self._session_ok is False

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        """Per-match lookup — the live card's direct-call path.

        The live card passes a Steam `match_id` (an OpenDota-style
        integer) which we can't map to an odds-api.io `event_id`
        here.  The supported path is the poller-based one:
        `get_all_live_quotes()` populates the cache by team-pair,
        and the live card does `_odds_live.get_quotes_for_teams(
        team_a_name, team_b_name)`.

        Direct `match_id` lookup returns [].  If we ever want to
        support it, we'd need to query OpenDota for the match's
        team names first, then look up in our own cache.
        """
        return []


# To enable:
#   ODDS_BACKEND=business.odds_backends.OddsApiIOBackend
#   ODDS_API_KEY=<your key from https://odds-api.io>
#   # Optional: filter to specific bookmakers (free tier = 2):
#   ODDS_API_BOOKMAKERS=Pinnacle,Bet365


# =========================================================================== #
# OddsPapiBackend  (v0.4.3 — secondary backend for the multi-backend setup)
# =========================================================================== #
#
# REST poller against https://api.oddspapi.io/v4 — the only free-tier
# odds aggregator that includes Pinnacle, GG.BET and Thunderpick
# (the sharp + esports-specialist books that actually price Dota 2
# well).
#
# Free tier: 250 req/month, 350+ bookmakers, real-time, no delay,
# historical odds included.  Limited to ~8 calls/day, so this
# backend is meant to be a SPOT-CHECK behind odds-api.io (which
# gives 100 req/hour but only 2 recreational bookmakers).
#
# Auth: `?apiKey=...` query param, just like odds-api.io.  No
# cookies, no signing, no SDKs needed.
#
# Response shape (relevant subset of /v4/odds):
#   {
#     "fixtureId": "id1000001761301055",
#     "participant1Id": 123, "participant2Id": 456,
#     "sportId": 16,                        # 16 = Dota 2
#     "statusId": 1,                        # 1=pre, 2=live, 3=settled
#     "hasOdds": true,
#     "bookmakerOdds": {
#       "pinnacle": {
#         "markets": {
#           "101": {                        # market ID, stringified
#             "outcomes": {
#               "101": {                    # outcome ID, stringified
#                 "players": {"0": {"price": 8.25}}
#               },
#               "102": {"players": {"0": {"price": 4.55}}},
#               "103": {"players": {"0": {"price": 1.46}}}
#             }
#           }
#         }
#       },
#       "bet365": {...},
#       "1xbet": {...}
#     }
#   }
#
# Market/outcome IDs are per-sport.  For Dota 2, we don't have
# the canonical mapping, so the parser uses a generic heuristic:
# * 2 outcomes in a market: first = P1, second = P2 (ML)
# * 3 outcomes in a market (1X2): skip the middle (Draw)
# * market with `over/under` in the outcome player key: treat as Total
# * otherwise: skip the market
#
# This is intentionally lenient — when we get a real response,
# we'll log the actual market names and add explicit ID mappings.
#
# Env vars:
#   ODDSPAPI_API_KEY        — required, from https://oddspapi.io
#   ODDSPAPI_BASE_URL       — optional, override for testing
#   ODDSPAPI_BOOKMAKERS     — optional, comma-separated filter
#   ODDSPAPI_SPORT_ID       — optional, default 16 (Dota 2)
# ---------------------------------------------------------------------------


class OddsPapiBackend:
    """REST-poller backend for https://api.oddspapi.io/v4.

    Conforms to the `OddsBackend` Protocol.  Loaded by
    `MultiOddsBackend` as a secondary (the multi-backend
    coordinator throttles us to one call per 3 hours so the
    250/month free tier lasts the month).

    Public methods (same shape as OddsApiIOBackend):
      get_quotes(match_id)       — returns [] (per-match not
                                   supported; live card uses
                                   team-pair cache)
      get_all_live_quotes()     — full snapshot, same Dict shape
                                   as odds-api.io (keyed by
                                   team_pair_key)
      probe_session()            — verify the key works
    """
    name = "oddspapi"

    SPORT_ID_DOTA2 = 16  # from oddspapi.io/docs
    # Per OddsPapi docs: 500ms cooldown between requests.  We
    # batch up to 10 fixtures per /odds call so a typical live
    # Dota 2 window is 1-2 requests per cycle.
    _REQUEST_COOLDOWN_SEC = 0.6

    def __init__(self) -> None:
        self._last_warn_no_creds: float = 0.0
        self._last_request_at: float = 0.0
        self._base_url = os.environ.get(
            "ODDSPAPI_BASE_URL", "https://api.oddspapi.io/v4"
        ).rstrip("/")
        self._bookmakers = os.environ.get(
            "ODDSPAPI_BOOKMAKERS", ""
        ).strip()
        # Per-market, per-outcome label overrides.  We populate
        # this lazily once we see a real response — it lets us
        # distinguish "P1 / P2" from "Over / Under" without
        # guessing from outcome IDs.  v0.4.3 starts empty; once
        # we have a real Dota 2 response logged, we add the
        # market/outcome IDs we see.
        self._known_market_types: Dict[str, str] = {}

    # -- env helpers -----------------------------------------------------

    def _key(self) -> str:
        return os.environ.get("ODDSPAPI_API_KEY", "").strip()

    def _has_key(self) -> bool:
        return bool(self._key())

    def _sport_id(self) -> int:
        try:
            return int(os.environ.get("ODDSPAPI_SPORT_ID", str(self.SPORT_ID_DOTA2)))
        except (TypeError, ValueError):
            return self.SPORT_ID_DOTA2

    def _cooldown(self) -> None:
        """Sleep until at least _REQUEST_COOLDOWN_SEC has passed
        since the last request.  OddsPapi enforces a 500ms
        cooldown per the docs; we round up to 600ms for safety.
        """
        now = time.monotonic()
        delta = self._REQUEST_COOLDOWN_SEC - (now - self._last_request_at)
        if delta > 0:
            time.sleep(delta)
        self._last_request_at = time.monotonic()

    # -- low-level HTTP --------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> Optional[Any]:
        if not self._has_key():
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                log.warning("OddsPapiBackend: missing ODDSPAPI_API_KEY — see .env template")
                self._last_warn_no_creds = now
            return None
        qp: Dict[str, str] = {"apiKey": self._key()}
        if params:
            qp.update({k: v for k, v in params.items() if v is not None})
        url = self._base_url + path + "?" + urllib.parse.urlencode(qp)
        self._cooldown()
        req = urllib.request.Request(
            url, method=method,
            headers={
                "User-Agent": "dota-analyst/0.4.3 (+https://oddspapi.io)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if not body:
                    return None
                try:
                    return json.loads(body.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    log.warning(
                        "OddsPapiBackend: %s %s returned non-JSON (%d bytes): %s",
                        method, path, len(body), exc,
                    )
                    return None
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                err_body = e.read()[:200].decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                log.warning(
                    "OddsPapiBackend: %s %s -> HTTP %d: %s",
                    method, path, code, err_body[:120],
                )
                self._last_warn_no_creds = now
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                log.warning(
                    "OddsPapiBackend: %s %s network error: %s",
                    method, path, e,
                )
                self._last_warn_no_creds = now
            return None

    # -- fixtures / odds -------------------------------------------------

    def _fetch_live_dota_fixtures(self) -> List[Dict[str, Any]]:
        """GET /v4/fixtures?sportId=16&from=...&to=... — list of Dota 2 fixtures.

        The API requires `from` and `to` date params when only
        `sportId` is given (else 400 MISSING_PARAMETERS).  The
        free tier caps the window at 10 days, so we use 3 days
        forward from today — enough to catch most live + upcoming
        matches without burning a request on a too-wide window.

        Filters for live / upcoming matches.  Free tier gives us
        pre-match + live.  We do NOT filter for hasOdds=true
        here because the response may omit it; the odds call
        downstream will tell us if there's actual data.
        """
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        plus3 = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        data = self._request(
            "GET", "/fixtures",
            params={
                "sportId": str(self._sport_id()),
                "from": today,
                "to": plus3,
                "limit": "50",
            },
        )
        if not isinstance(data, list):
            # Some endpoints return a {"data": [...]} envelope.
            # Try that.
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                data = data["data"]
            else:
                return []
        out: List[Dict[str, Any]] = []
        for f in data:
            if not isinstance(f, dict):
                continue
            sid = f.get("sportId")
            if sid is not None and int(sid) != self._sport_id():
                continue
            out.append(f)
        return out

    def _fetch_odds_for_fixtures(
        self, fixture_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """GET /v4/odds?fixtureId=X — one per fixture.

        We can't easily batch via fixtureIds on /v4/odds; the
        primary batch endpoint is /v4/odds-by-tournament but
        that requires knowing tournament IDs upfront.  For
        simplicity (and because we'll throttle hard at the
        multi-backend layer), we make one request per fixture.
        With ~3-5 live Dota 2 matches, that's 3-5 requests per
        secondary cycle (well under the 250/month budget).
        """
        out: List[Dict[str, Any]] = []
        for fid in fixture_ids:
            params: Dict[str, str] = {"fixtureId": str(fid)}
            if self._bookmakers:
                params["bookmakers"] = self._bookmakers
            data = self._request("GET", "/odds", params=params)
            if isinstance(data, dict) and data.get("hasOdds"):
                out.append(data)
        return out

    # -- parser ---------------------------------------------------------

    # Heuristics for "is this a Total market?".  We have to
    # guess because the response doesn't ship human-readable
    # market labels in the live feed (only integer IDs).  We
    # update this map after seeing a real response.
    _TOTAL_MARKET_HINTS = (
        "total", "over", "under", "maps", "kills",
    )

    def _classify_market(
        self, market_id: str, outcomes: Dict[str, Any],
    ) -> Optional[str]:
        """Return 'winner', 'total', or None for a market.

        `outcomes` is the raw `outcomes` dict from the response,
        keyed by stringified outcome ID.  Heuristics, in order:
          1. If any outcome has a numeric `handicap` / `total` /
             `points` / `line` field → it's a totals market.
             (OddsPapi stores the threshold in `players["0"]
             .handicap` for over/under lines.)
          2. If an outcome's name/label contains "over", "under",
             "total", "maps", or "kills" → totals.
          3. 3 outcomes and no totals signal → 1X2 (winner with
             a draw in the middle; we skip the draw).
          4. 2 outcomes and no totals signal → ML (winner).
        """
        n = len(outcomes or {})
        if n < 2 or n > 3:
            return None
        # Heuristic 1: presence of a threshold field on any
        # outcome.  Strong signal for Over/Under.
        for oid, outcome in outcomes.items():
            if not isinstance(outcome, dict):
                continue
            if self._outcome_threshold(outcome) is not None:
                return "total"
        # Heuristic 2: text hints on outcome names
        for oid, outcome in outcomes.items():
            if not isinstance(outcome, dict):
                continue
            p0 = (outcome.get("players") or {}).get("0") or {}
            label = (
                p0.get("name") or p0.get("label") or p0.get("type")
                or ""
            ).lower()
            for hint in self._TOTAL_MARKET_HINTS:
                if hint in label:
                    return "total"
        # Heuristic 3 / 4: count
        if n == 3:
            return "winner_1x2"
        return "winner"

    def _outcome_price(self, outcome: Dict[str, Any]) -> Optional[float]:
        """Pull the decimal price out of an outcome's nested structure.

        Path: outcome.players["0"].price  (a dict on /v4/odds).
        Returns None on any structural mismatch.
        """
        try:
            p0 = (outcome.get("players") or {}).get("0")
            if not isinstance(p0, dict):
                return None
            price = p0.get("price")
            if price is None:
                return None
            return float(price)
        except (TypeError, ValueError):
            return None

    def _outcome_threshold(self, outcome: Dict[str, Any]) -> Optional[float]:
        """Pull a numeric threshold out of an outcome if present.

        Different markets store the threshold differently:
        odds["0"].handicap, odds["0"].total, odds["0"].points, etc.
        We try a few common field names.
        """
        p0 = (outcome.get("players") or {}).get("0") or {}
        if not isinstance(p0, dict):
            return None
        for k in ("handicap", "total", "points", "line", "max"):
            v = p0.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    def _fixture_to_quotes(
        self, fixture: Dict[str, Any],
    ) -> List[OddsQuote]:
        """Parse one /v4/odds response into OddsQuote list.

        Walks bookmakerOdds[slug].markets[mid].outcomes[oid] and
        emits one OddsQuote per (bookmaker, market, outcome) cell.
        """
        out: List[OddsQuote] = []
        ev_id = fixture.get("fixtureId")
        bookmakers = fixture.get("bookmakerOdds") or {}
        for slug, bdata in bookmakers.items():
            if not isinstance(bdata, dict):
                continue
            markets = bdata.get("markets") or {}
            for market_id, mdata in markets.items():
                if not isinstance(mdata, dict):
                    continue
                outcomes = mdata.get("outcomes") or {}
                mtype = self._classify_market(market_id, outcomes)
                if mtype is None:
                    continue
                # ----- Winner (2-outcome ML) or 1X2 (3-outcome) -----
                if mtype in ("winner", "winner_1x2"):
                    # Order: outcomes are usually 101/102/103 (or
                    # 1/2/3 for Dota 2).  We pick the first and
                    # the last as P1/P2; if 1X2, the middle one
                    # is the draw and we skip it.
                    items = list(outcomes.items())
                    # Sort by outcome id (stringified int) for
                    # stable ordering — the first/lowest is Home,
                    # the highest is Away.
                    try:
                        items.sort(key=lambda kv: int(kv[0]))
                    except (TypeError, ValueError):
                        pass
                    if not items:
                        continue
                    p1_oid, p1_outcome = items[0]
                    p2_oid, p2_outcome = items[-1]
                    if p1_oid == p2_oid:
                        # Only 1 outcome — degenerate, skip.
                        continue
                    p1_price = self._outcome_price(p1_outcome)
                    p2_price = self._outcome_price(p2_outcome)
                    if p1_price is None or p2_price is None:
                        continue
                    if p1_price <= 1.0 or p2_price <= 1.0:
                        # Sanity: decimal odds must be > 1.0
                        continue
                    out.append(OddsQuote(
                        market="winner", selection="P1",
                        decimal_odds=p1_price, bookmaker=str(slug),
                        raw={"fixture_id": ev_id, "market_id": market_id,
                             "outcome_id": p1_oid},
                    ))
                    out.append(OddsQuote(
                        market="winner", selection="P2",
                        decimal_odds=p2_price, bookmaker=str(slug),
                        raw={"fixture_id": ev_id, "market_id": market_id,
                             "outcome_id": p2_oid},
                    ))
                # ----- Total (Over/Under with threshold) -----
                elif mtype == "total":
                    items = list(outcomes.items())
                    if len(items) != 2:
                        continue
                    # First is typically Over, second Under
                    over_oid, over_out = items[0]
                    under_oid, under_out = items[1]
                    over_price = self._outcome_price(over_out)
                    under_price = self._outcome_price(under_out)
                    if over_price is None or under_price is None:
                        continue
                    # Threshold can live on either outcome (or
                    # both).  Try over first, fall back to under.
                    threshold = (
                        self._outcome_threshold(over_out)
                        or self._outcome_threshold(under_out)
                    )
                    if threshold is None:
                        # Without a threshold we can't form
                        # `over_2.5`.  Skip — the live card
                        # doesn't have a "no-threshold total"
                        # slot anyway.
                        continue
                    try:
                        thr_f = float(threshold)
                    except (TypeError, ValueError):
                        continue
                    out.append(OddsQuote(
                        market="total_kills",
                        selection=f"over_{thr_f:g}",
                        decimal_odds=over_price, bookmaker=str(slug),
                        raw={"fixture_id": ev_id, "market_id": market_id,
                             "outcome_id": over_oid, "threshold": thr_f},
                    ))
                    out.append(OddsQuote(
                        market="total_kills",
                        selection=f"under_{thr_f:g}",
                        decimal_odds=under_price, bookmaker=str(slug),
                        raw={"fixture_id": ev_id, "market_id": market_id,
                             "outcome_id": under_oid, "threshold": thr_f},
                    ))
        return out

    # -- public API ------------------------------------------------------

    def get_all_live_quotes(self) -> Dict[str, List[OddsQuote]]:
        """Full snapshot keyed by `team_pair_key(home, away)`.

        Same shape as `OddsApiIOBackend.get_all_live_quotes` so
        the multi-backend coordinator can merge results
        directly.
        """
        if not self._has_key():
            return {}
        fixtures = self._fetch_live_dota_fixtures()
        if not fixtures:
            return {}
        # Filter to fixtures that have odds (most efficient path
        # — the /odds call costs a request per fixture, so we
        # only call it for fixtures likely to have data)
        candidate_ids = [
            f.get("fixtureId") for f in fixtures
            if f.get("fixtureId") and (f.get("hasOdds") is not False)
        ]
        if not candidate_ids:
            return []
        odds_data = self._fetch_odds_for_fixtures(candidate_ids)
        # Build a quick fixtureId -> fixture-with-odds map
        by_id: Dict[str, Dict[str, Any]] = {
            od.get("fixtureId"): od for od in odds_data
            if isinstance(od, dict)
        }
        # Build a fixtureId -> original fixture (for team names) map
        names_by_id: Dict[str, Dict[str, Any]] = {
            f.get("fixtureId"): f for f in fixtures
            if f.get("fixtureId")
        }
        out: Dict[str, List[OddsQuote]] = {}
        for fid, od in by_id.items():
            quotes = self._fixture_to_quotes(od)
            if not quotes:
                continue
            # Team names live on the original fixture record.
            # OddsPapi uses `participants` (array) or
            # `participant1Name` / `participant2Name` depending
            # on the endpoint version.  We try all known shapes.
            f = names_by_id.get(fid, {})
            home, away = self._extract_team_names(f, od)
            if not (home and away):
                continue
            k = team_pair_key(home, away)
            if not k or k == "|":
                continue
            out[k] = quotes
        return out

    @staticmethod
    def _extract_team_names(
        fixture: Dict[str, Any], odds: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Pull (home, away) names out of the OddsPapi payload.

        Tries several known shapes:
          * `participant1Name` / `participant2Name` (top-level)
          * `participants[].name` array
          * `homeTeam` / `awayTeam` (some versions)
          * `externalProviders` IDs (last resort — not names)
        """
        for src in (odds, fixture):
            if not isinstance(src, dict):
                continue
            a = src.get("participant1Name") or src.get("homeTeam") or src.get("home_name")
            b = src.get("participant2Name") or src.get("awayTeam") or src.get("away_name")
            if a and b:
                return str(a), str(b)
            parts = src.get("participants") or []
            if isinstance(parts, list) and len(parts) >= 2:
                a = parts[0].get("name") if isinstance(parts[0], dict) else None
                b = parts[1].get("name") if isinstance(parts[1], dict) else None
                if a and b:
                    return str(a), str(b)
        return "", ""

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        # Per-match lookup not supported — the live card uses
        # the team-pair cache populated by `get_all_live_quotes`.
        return []

    def probe_session(self) -> bool:
        """Hit /v4/sports (no auth required) to check connectivity.

        We can't probe a cheap authenticated endpoint on free
        tier — the cheapest is /fixtures which costs a request.
        For OddsPapi we use a non-auth probe and trust that
        the real data calls will surface 401/403.
        """
        try:
            req = urllib.request.Request(
                self._base_url + "/sports",
                method="GET",
                headers={
                    "User-Agent": "dota-analyst/0.4.3",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10.0) as r:
                return r.status == 200
        except Exception:
            return False


# To enable as a secondary (rate-limited):
#   ODDSPAPI_API_KEY=<your key from https://oddspapi.io>
#   ODDSPAPI_BOOKMAKERS=pinnacle,ggbet,thunderpick
#   (The multi-backend coordinator throttles us to once per
#    ODDS_SECONDARY_POLL_SEC seconds — default 10800 = 3 hours,
#    keeping the 250/month free tier budget intact.)


# =========================================================================== #
# MultiOddsBackend  (v0.4.3 — composite: primary + one-or-more secondaries)
# =========================================================================== #
#
# Coordinates multiple `OddsBackend` instances:
#
#   1. **Primary** (first in the ODDS_BACKENDS list) is called
#      every poll cycle.  It is expected to be cheap (high rate
#      limit) — typically `OddsApiIOBackend` (100 req/hour free).
#
#   2. **Secondaries** (rest of the list) are throttled to one
#      call per `ODDS_SECONDARY_POLL_SEC` (default 3 hours).
#      They fill in matches the primary missed.  Typically
#      `OddsPapiBackend` (250 req/month free, includes Pinnacle).
#
# Merging: keys that the primary already filled are kept;
# secondary keys fill in only what's missing.  This means the
# primary's data is "trusted" first, secondaries add breadth.
#
# Configuration via env:
#   ODDS_BACKENDS=business.odds_backends.OddsApiIOBackend,business.odds_backends.OddsPapiBackend
#   ODDS_SECONDARY_POLL_SEC=10800          # default 3 hours
#
# Why throttling: 250 req/month ÷ 30 days = 8/day, so 3 hours
# is the safe cadence.  If the user upgrades OddsPapi to a paid
# tier, lower the interval accordingly.
# ---------------------------------------------------------------------------


class MultiOddsBackend:
    """Composite backend that fans out to a primary + secondaries.

    Per-cycle flow (`get_all_live_quotes`):
      1. Call primary.  Merge into `out`.
      2. For each secondary, if its `_last_poll_at` is older
         than `ODDS_SECONDARY_POLL_SEC`, call it and merge
         missing keys.  Skip the call if the cadence hasn't
         elapsed — saves the free-tier budget.
      3. If a secondary raises, log once and continue.
      4. Return `out`.

    `probe_session` returns True if ANY backend's probe
    succeeds — fail-soft for the whole composite.
    """
    name = "multi"

    # Default secondary cadence.  3 hours × 8 = 24/day, which
    # is 720/month for OddsPapi free tier 250 — too high.  But
    # the throttling is per-secondary, and we only call when the
    # primary is missing matches; in practice the secondary
    # call is much less frequent than every 3 hours when
    # odds-api.io is doing its job.
    DEFAULT_SECONDARY_POLL_SEC = 10800.0  # 3 hours

    def __init__(self) -> None:
        self._backends: List[OddsBackend] = []
        self._secondary_poll_at: Dict[str, float] = {}
        self._init_backends()
        self._secondary_poll_sec: float = float(
            os.environ.get(
                "ODDS_SECONDARY_POLL_SEC",
                str(self.DEFAULT_SECONDARY_POLL_SEC),
            )
        )
        log.info(
            "MultiOddsBackend: %d backend(s): %s (secondary_cadence=%.0fs)",
            len(self._backends),
            ", ".join(b.name for b in self._backends) or "(none!)",
            self._secondary_poll_sec,
        )

    def _init_backends(self) -> None:
        """Read ODDS_BACKENDS env, instantiate each class.

        Format: comma-separated fully-qualified class names
        (e.g. "business.odds_backends.OddsApiIOBackend,business.odds_backends.OddsPapiBackend").
        Anything that fails to import / instantiate is logged
        and skipped — a misconfigured secondary should not
        break the whole composite.
        """
        spec = os.environ.get(
            "ODDS_BACKENDS",
            "business.odds_backends.OddsApiIOBackend",
        ).strip()
        if not spec:
            return
        for cls_spec in spec.split(","):
            cls_spec = cls_spec.strip()
            if not cls_spec:
                continue
            try:
                module_name, _, cls_name = cls_spec.rpartition(".")
                if not module_name:
                    raise ValueError(
                        f"ODDS_BACKENDS entry must be 'module.path.ClassName', got {cls_spec!r}"
                    )
                mod = importlib.import_module(module_name)
                cls = getattr(mod, cls_name)
                inst = cls()
                self._backends.append(inst)
            except Exception as exc:
                log.warning(
                    "MultiOddsBackend: failed to load %r: %s", cls_spec, exc,
                )

    def _should_poll_secondary(self, backend: OddsBackend) -> bool:
        last = self._secondary_poll_at.get(backend.name)
        if last is None:
            return True
        return (time.monotonic() - last) >= self._secondary_poll_sec

    def probe_session(self) -> bool:
        """True if at least one backend's probe succeeds."""
        if not self._backends:
            return False
        for b in self._backends:
            try:
                if b.probe_session():
                    return True
            except Exception as exc:
                log.debug("MultiOddsBackend: %s probe raised: %s", b.name, exc)
        return False

    def get_all_live_quotes(self) -> Dict[str, List[OddsQuote]]:
        out: Dict[str, List[OddsQuote]] = {}
        if not self._backends:
            return out
        # 1. Primary (always poll)
        primary = self._backends[0]
        try:
            primary_snap = primary.get_all_live_quotes() or {}
            out.update(primary_snap)
            log.debug(
                "MultiOddsBackend: primary %s returned %d keys",
                primary.name, len(primary_snap),
            )
        except Exception as exc:
            log.warning(
                "MultiOddsBackend: primary %s raised: %s", primary.name, exc,
            )
        # 2. Secondaries (throttled)
        for sec in self._backends[1:]:
            if not self._should_poll_secondary(sec):
                continue
            try:
                sec_snap = sec.get_all_live_quotes() or {}
                # Fill in only keys the primary didn't have.
                # This is the "spot-check" behavior: if odds-api.io
                # already covered the match, we don't overwrite.
                added = 0
                for k, v in sec_snap.items():
                    if k not in out:
                        out[k] = v
                        added += 1
                log.info(
                    "MultiOddsBackend: secondary %s returned %d keys (%d new, %d already covered)",
                    sec.name, len(sec_snap), added, len(sec_snap) - added,
                )
            except Exception as exc:
                log.warning(
                    "MultiOddsBackend: secondary %s raised: %s", sec.name, exc,
                )
            finally:
                # Mark the poll attempt even on failure so we
                # don't immediately retry.  The next cycle
                # will try again after the cadence elapses.
                self._secondary_poll_at[sec.name] = time.monotonic()
        return out

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        """Per-match lookup: try each backend in order, first hit wins."""
        for b in self._backends:
            try:
                quotes = b.get_quotes(int(match_id))
                if quotes:
                    return quotes
            except Exception as exc:
                log.debug("MultiOddsBackend: %s get_quotes raised: %s", b.name, exc)
        return []


# To enable:
#   ODDS_BACKEND=business.odds_backends.MultiOddsBackend
#   ODDS_BACKENDS=business.odds_backends.OddsApiIOBackend,business.odds_backends.OddsPapiBackend
#   ODDS_API_KEY=<from odds-api.io>
#   ODDSPAPI_API_KEY=<from oddspapi.io>
#   ODDS_SECONDARY_POLL_SEC=10800            # 3 hours
#   ODDS_API_BOOKMAKERS=1xbet,Bet365        # free tier 2-bookmaker cap
#   ODDSPAPI_BOOKMAKERS=pinnacle,ggbet,thunderpick  # sharp + esports specialists

