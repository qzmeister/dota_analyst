"""
Stateless session tokens for the SSE path.

The browser's `EventSource` cannot send custom headers (so X-API-Key
won't work for `/api/stream/*`).  Cookies ARE sent automatically
when the EventSource is opened with `withCredentials: true`, so
the cleanest fix is a session cookie that the gateway can verify
without hitting a database.

`sign_session_token` produces an HMAC-SHA256 over the user's
api_key + an expiry timestamp.  `verify_session_token` checks
both the HMAC and the expiry.  No state on the server — the
token is the session.

v0.4.0.x: this is the public-deploy-friendly replacement for
the "trust the network boundary" approach documented at
`gateway/_middleware.py:ApiKeyAuthMiddleware`.  Login flow is
in `gateway/app.py` (POST /api/auth/login).  Frontend sets
`withCredentials=true` on the EventSource so the cookie is
sent on the long-lived stream.

Security notes:
  - HMAC key is `DEV_API_KEY` (the same secret the X-API-Key
    check uses).  The session token never reveals the api_key
    — an attacker who reads the cookie can replay it for
    `SESSION_COOKIE_MAX_AGE` seconds, but they can't recover
    the underlying api_key from it.
  - Token format: `<expiry_unix>.<hex_hmac>`.  We split on
    `.` and validate both halves.  This is the same shape
    used by JWT's `alg=HS256` "compact JWS" — deliberately
    simple so an auditor can verify it in 5 lines of
    Python.
  - Cookie attributes: `HttpOnly` (JS can't read it),
    `SameSite=Lax` (cross-site GETs work, CSRF resistance),
    `Secure` (HTTPS-only — set when the prod edge enables
    TLS).  Path `/` so the gateway sees it on every request.
  - We do NOT sign the user's api_key directly into the
    cookie.  A user could revoke their api_key and a stolen
    cookie would still work until expiry.  This is the
    standard tradeoff for stateless HMAC sessions — the
    alternative is a server-side store, which kills the
    "no state" property.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

SESSION_COOKIE_NAME = "dota_analyst_session"
# 7 days.  Long enough that the user doesn't get logged out
# every time they close the tab; short enough that a stolen
# cookie has a finite blast radius.
SESSION_COOKIE_MAX_AGE = 7 * 24 * 60 * 60

# Clock skew tolerance.  If the client clock is 60s ahead,
# the token's "expires_at - now" is briefly negative and a
# naive verify would reject it.  60s absorbs typical NTP drift
# without making the replay window materially larger.
_CLOCK_SKEW_SEC = 60


def _sign(secret: str, payload: str) -> str:
    """HMAC-SHA256 hex digest of `payload` keyed by `secret`."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def sign_session_token(secret: str, *, ttl_sec: int = SESSION_COOKIE_MAX_AGE) -> str:
    """Return a `expiry_unix.hmac` token signed with `secret`.

    The HMAC is over `f"{expiry_unix}:{nonce}"` so two tokens
    minted at the same expiry are still distinguishable (no
    preimage collision).  The nonce is 16 hex chars from
    os.urandom — enough to make accidental collision vanishing.
    """
    expires_at = int(time.time()) + ttl_sec
    nonce = os.urandom(8).hex()
    payload = f"{expires_at}:{nonce}"
    sig = _sign(secret, payload)
    return f"{expires_at}:{nonce}.{sig}"


def verify_session_token(secret: str, token: str) -> bool:
    """Return True iff `token` is a valid session for `secret`.

    Checks (in order):
      1. shape: must have one '.' separator; the LHS is
         "expiry:nonce", the RHS is hex.
      2. expiry: must be in the future (with `_CLOCK_SKEW_SEC`
         tolerance for client clock drift).
      3. HMAC: `hmac_sha256(secret, expiry:nonce) == rhs`.
    """
    if not token or "." not in token:
        return False
    head, sig = token.rsplit(".", 1)
    if ":" not in head:
        return False
    expires_at_s, nonce = head.split(":", 1)
    try:
        expires_at = int(expires_at_s)
    except (TypeError, ValueError):
        return False
    now = int(time.time())
    if expires_at < (now - _CLOCK_SKEW_SEC):
        return False
    expected = _sign(secret, f"{expires_at_s}:{nonce}")
    # constant-time compare to avoid timing leaks on the HMAC
    return hmac.compare_digest(expected, sig)


def set_session_cookie(response, token: str, *, max_age: int = SESSION_COOKIE_MAX_AGE,
                      secure: bool = False) -> None:
    """Attach a Set-Cookie header to `response`.

    `secure=False` is the dev default (the user is on
    http://localhost).  When the prod edge has TLS, the
    gateway should call this with `secure=True` (or set
    via env: `SESSION_COOKIE_SECURE=1`).
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def clear_session_cookie(response, *, secure: bool = False) -> None:
    """Attach a Set-Cookie header that expires the session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=secure,
    )


def read_session_cookie(request) -> Optional[str]:
    """Read the session cookie from `request`, or None if absent."""
    return request.cookies.get(SESSION_COOKIE_NAME)
