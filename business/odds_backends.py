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
        self._bookmakers = (
            os.environ.get("ODDS_API_BOOKMAKERS", "").strip()
        )

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
        timeout: float = 6.0,
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
        """
        now = time.monotonic()
        if (
            self._session_ok is not None
            and (now - self._last_session_check) < self._session_check_ttl
        ):
            return self._session_ok
        resp = self._request("GET", "/bookmakers/selected")
        # 200 with a list (possibly empty) = OK
        # 401/403 = dead
        # None (network) = not "OK" but not necessarily "dead"
        self._session_ok = resp is not None
        self._last_session_check = now
        return self._session_ok

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
        # Last resort: peek at the team names.  Dota 2 team
        # names rarely include digits; CS2 / Valorant / LoL team
        # names are usually different.  This is a very rough
        # fallback — better to over-include than under-include
        # so the live card just shows empty odds for non-Dota2.
        names = " ".join(
            str(event.get(k) or "")
            for k in ("home", "away", "homeName", "awayName")
        ).lower()
        # Heuristic-only; we err on the side of inclusion.
        return "dota" in names or bool(names)

    def _fetch_odds_multi(
        self, event_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """GET /v3/odds/multi?eventIds=1,2,3 — up to 10 per call.

        Splits the list into chunks of 10 and makes one request per
        chunk.  The first 10-event chunk is what hits the wire
        for a typical live Dota 2 window (≤ 10 concurrent matches).
        """
        if not event_ids:
            return []
        results: List[Dict[str, Any]] = []
        for i in range(0, len(event_ids), 10):
            chunk = event_ids[i : i + 10]
            params: Dict[str, str] = {"eventIds": ",".join(str(e) for e in chunk)}
            if self._bookmakers:
                params["bookmakers"] = self._bookmakers
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
          1. Probe the session.  If it fails, return {} (the
             poller will log and try again next cycle).
          2. Fetch live Dota 2 events from `/events/live`.
          3. Batch-fetch odds via `/odds/multi` (up to 10/call).
          4. Parse each event to OddsQuote list.
          5. Bucket by `team_pair_key(home, away)`.  We DON'T
             add a second bucket under the swapped key — the
             live card already tries both orderings itself.
        """
        if not self._has_key():
            return {}
        if not self.probe_session():
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

