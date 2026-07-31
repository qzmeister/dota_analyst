"""BetBoom.ru backend using user-provided session cookies + auth token.

Approach: we don't re-implement login (Cloudflare + Fingerprint.js +
reCAPTCHA + Group-IB block that).  Instead, the user logs in once
in a real browser, exports the cookies + the auth-token from the
login response, and we replay them.

The cookies / token are short-lived (cf_clearance ~30 min, auth-token
varies).  The user has to refresh them every so often.  When
BETBOOM_AUTH_TOKEN is missing, the backend falls back to
"no quotes" rather than failing the live card.

Env vars (loaded from .env.betboom — gitignored):
  BETBOOM_PHONE          — used for logs only
  BETBOOM_AUTH_TOKEN      — the `access_token` from /auth/login response
                            (or whatever the response carries — could be
                            `session_token`, `X-Access-Token`, etc.)
  BETBOOM_CF_CLEARANCE    — Cloudflare clearance cookie value
  BETBOOM_SESSION_COOKIE  — "name=value" pair of the auth session cookie
                            (e.g. "_session=eyJhbGciOi...")
  BETBOOM_OTHER_COOKIES   — additional "name=value" pairs to send, joined
                            by "; " (e.g. "_csrf=...; device_id=...")

Why this design:
  * No CAPTCHA solving in our code (2captcha is the right call if
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

BASE = "https://siteapi.betboom.ru"
ORIGIN = "https://betboom.ru"


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
            "auth_token":  os.environ.get("BETBOOM_AUTH_TOKEN", "").strip(),
            "cf_clearance":os.environ.get("BETBOOM_CF_CLEARANCE", "").strip(),
            "session":     os.environ.get("BETBOOM_SESSION_COOKIE", "").strip(),
            "other":       os.environ.get("BETBOOM_OTHER_COOKIES", "").strip(),
        }

    def _has_creds(self, c: Dict[str, str]) -> bool:
        # We need at least the auth token + cf_clearance.  Without
        # cf_clearance, Cloudflare will 403; without the auth token,
        # the response is a 401.
        return bool(c["auth_token"]) and bool(c["cf_clearance"])

    def _cookie_header(self, c: Dict[str, str]) -> str:
        parts = []
        if c["cf_clearance"]:
            parts.append(f"cf_clearance={c['cf_clearance']}")
        if c["session"]:
            parts.append(c["session"])
        if c["other"]:
            parts.append(c["other"])
        return "; ".join(parts)

    def _request(
        self, method: str, path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        url = BASE + path
        if params:
            from urllib.parse import urlencode
            url = url + "?" + urlencode(params)
        c = self._creds()
        if not self._has_creds(c):
            now = time.monotonic()
            if now - self._last_warn_no_creds > 60:
                print(f"[odds] BetBoomBackend: missing BETBOOM_AUTH_TOKEN or "
                      f"BETBOOM_CF_CLEARANCE — see .env.betboom template")
                self._last_warn_no_creds = now
            return None
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) dota-analyst/0.4",
            "Accept": "application/json",
            "Origin": ORIGIN,
            "Referer": ORIGIN + "/",
            "Cookie": self._cookie_header(c),
            # Try a few common auth header shapes — BetBoom
            # hasn't documented which one.  The browser sends
            # whichever the SPA's `runtimeConfig` lookup resolves
            # to.  We send all three and let the server pick.
            "Authorization": f"Bearer {c['auth_token']}",
            "X-Access-Token": c["auth_token"],
            "platform": "web",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
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
        # The /auth/session endpoint exists (we saw it earlier)
        # and the right answer for a valid session is a 200 with
        # the user payload.  We don't have the exact path —
        # fall back to /ping (which is public) and consider
        # session valid if we get a non-401.  This is best-effort.
        resp = self._request("GET", "/api/site_api/v1/ping")
        self._session_ok = resp is not None
        self._last_session_check = now
        return self._session_ok

    def live_dota2(self) -> List[Dict[str, Any]]:
        """Fetch the live dota2 events list.  Returns whatever shape
        the BetBoom API gives us — the caller (or a future parser)
        is responsible for mapping to OddsQuote.

        Path is heuristic — we don't actually know BetBoom's
        endpoint.  Common shapes are `/live`, `/events/live`,
        `/feed/live/dota2`.  The first one that returns a
        non-empty list wins.
        """
        for path in (
            "/api/site_api/v1/live",
            "/api/site_api/v1/live/dota2",
            "/api/site_api/v1/live/events",
            "/api/site_api/v1/events?status=live",
            "/api/site_api/v1/events?status=1",
            "/api/site_api/v1/feed/live",
            "/api/site_api/v1/feed/live/dota2",
            "/api/site_api/v1/prematch?status=live",
        ):
            resp = self._request("GET", path)
            if not resp:
                continue
            # The response shape varies.  Try common envelopes.
            for key in ("data", "events", "matches", "items", "result"):
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
                (ev.get("opponents") or [{}])[0].get("name") or
                ev.get("title") or
                ""
            )
            team_b = (
                ev.get("team_b_name") or
                ev.get("away_name") or
                ev.get("second_team_name") or
                (ev.get("opponents") or [{}, {}])[1].get("name") or
                ""
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
#   BETBOOM_AUTH_TOKEN=<from /auth/login response>
#   BETBOOM_CF_CLEARANCE=<from cookies>
#   BETBOOM_SESSION_COOKIE=session=eyJ...
#   BETBOOM_OTHER_COOKIES=_csrf=...; device_id=...
