"""BetBoom.ru backend using user-provided session cookies + auth token.

Approach: we don't re-implement login (Cloudflare + Fingerprint.js +
reCAPTCHA + Group-IB block that).  Instead, the user logs in once
in a real browser, exports the cookies + the auth-token from the
login response, and we replay them.

The cookies / token are short-lived (csrf_token has a 1y TTL but
can rotate; the JWT expires 6h).  The user has to refresh them
every so often.  When BETBOOM_AUTH_TOKEN is missing, the backend
falls back to "no quotes" rather than failing the live card.

Real auth flow (HAR 2026-07-31):
  URL:     https://betboom.ru/api/auth/login
  Method:  POST
  Headers: x-platform: web
  Body:    {phone, password, fingerprint, fingerprint_request_id,
            is_wrong_data_auth_support, ym_id, captcha_key,
            features.is_support_conditional_captcha}
  Server:  QRATOR (not Cloudflare)
  Response: {code: 200, status: "OK", token: <JWT, 6h>,
             refresh_token: <JWT, 1y>, trace_id}

The live API is on `siteapi.betboom.ru` (NOT betboom.ru):
  * User HAR of /auth/login (on betboom.ru) returns 200, sets NO
    cookies in the response (the JWT is in the body).
  * The cookies the browser holds are split by domain:
      betboom.ru     — SPA cookies (mindbox, Yandex, theme,
                       authorizedId, directCrm, GIB fingerprint,
                       Intercom, csrf_token NOT here)
      siteapi.betboom.ru — only `csrf_token` (JWT, 1y TTL) is
                           bound to this API host
  * So the data API lives on `siteapi.betboom.ru` and needs the
    csrf_token cookie (which the SPA set during initialisation
    before /auth/login).  Login is on `betboom.ru` and is the
    only thing that needs the SPA fingerprint cookies.
  * The /auth/login response itself uses envelope
    {code, status, ...} where code=200 means OK and code=404
    means "path not found" (but the HTTP status is 200).  This
    is application-level routing on the API host.

Env vars (loaded from .env.betboom — gitignored):
  BETBOOM_PHONE          — phone (logged only)
  BETBOOM_AUTH_TOKEN      — JWT from /auth/login response body
                            (used as Authorization: Bearer)
  BETBOOM_REFRESH_TOKEN   — JWT refresh token, 1y TTL (unused yet)
  BETBOOM_COOKIES         — full "name=value; name=value; ..."
                            cookie string from the authed browser.
                            Includes csrf_token, GIB fingerprint,
                            Yandex Metrika, etc.
  BETBOOM_BASE_URL        — override the API base (default
                            https://siteapi.betboom.ru).  Useful
                            for testing against a different env.

Why this design:
  * No CAPTCHA solving in our code (2captha is the right call if
    we want automation; we deliberately don't for now).
  * No persistent Playwright browser (heavy + fragile).
  * User does the login once in their own browser, exports state,
    paste into .env.betboom.  Restarts pick it up.
  * If/when the session expires, the user re-logs-in and we get
    a fresh batch of creds.  The backend logs a clear "session
    expired" so the user knows.
"""
from __future__ import annotations

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
