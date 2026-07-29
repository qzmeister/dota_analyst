"""
Gateway — the only public-facing service in the Dota Analyst stack.

Responsibilities:
  - Validate auth (X-API-Key)
  - Reject oversized payloads
  - Add / propagate X-Request-Id for end-to-end tracing
  - Forward /api/* to the business service
  - Forward /api/stream/* with SSE-friendly settings (no buffering)
  - Expose /healthz and /readyz for k8s/compose probes

The gateway has no business logic. Every computation belongs in the
business service. Anything you add here is a code smell.
"""

from __future__ import annotations

import os
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ._logging import get_logger, setup_logging
from ._middleware import install_middlewares
from ._proxy import stream_response
from ._session import (
    SESSION_COOKIE_MAX_AGE,
    clear_session_cookie,
    read_session_cookie,
    set_session_cookie,
    sign_session_token,
    verify_session_token,
)

# Configure logging once on import. `setup_logging` is idempotent.
setup_logging()
log = get_logger(__name__)


# ---------------------------------------------------------------------------- #
# App
# ---------------------------------------------------------------------------- #

app = FastAPI(title="Dota Analyst — gateway", version="0.1.0", docs_url=None, redoc_url=None)

DEV_API_KEY = os.environ.get("DEV_API_KEY", "")
BODY_MAX_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(1024 * 1024)))
# v0.4.0.1: when the prod edge has TLS, set SESSION_COOKIE_SECURE=1
# so the session cookie gets the `Secure` attribute (HTTPS-only).
# Default off so the dev environment (http://localhost) still works.
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"

install_middlewares(app, dev_api_key=DEV_API_KEY, body_max_bytes=BODY_MAX_BYTES)


# ---------------------------------------------------------------------------- #
# Health
# ---------------------------------------------------------------------------- #

@app.get("/healthz")
def healthz():
    """Liveness probe — the gateway process is alive."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness probe — gateway can reach the business service."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{os.environ.get('BUSINESS_URL', 'http://business:8000')}/api/healthz")
        return {"status": "ok" if r.status_code == 200 else "degraded",
                "business": r.status_code}
    except (httpx.HTTPError, OSError) as exc:
        # httpx raises HTTPError (parent of ConnectError, TimeoutException,
        # ReadError, etc.); OSError covers the underlying socket layer.
        # We deliberately don't catch `Exception` so a coding bug in this
        # endpoint surfaces in logs.
        log.warning("readiness check failed: %s", exc, exc_info=True)
        return JSONResponse(
            {"status": "not-ready", "error": str(exc)},
            status_code=503,
        )


# ---------------------------------------------------------------------------- #
# Auth (v0.4.0.1) — cookie-based session for the SSE path
# ---------------------------------------------------------------------------- #
#
# The browser's `EventSource` cannot send custom HTTP headers, so the
# static UI at /api/stream/* can't use the X-API-Key header.  Cookies
# ARE sent automatically when the EventSource is opened with
# `withCredentials: true`, so we:
#
#   1. Issue an HMAC-signed session cookie on POST /api/auth/login
#      (the user POSTs their api_key; we verify it against
#      DEV_API_KEY, then mint a session token).
#
#   2. Check the session cookie on /api/stream/* (the SSE path) in
#      `ApiKeyAuthMiddleware.COOKIE_AUTHED_PREFIXES`.  The legacy
#      X-API-Key header is still accepted as a fallback so curl /
#      dev tools keep working.
#
# The session token is stateless (HMAC of `expiry:nonce`, keyed by
# DEV_API_KEY) — see `gateway/_session.py` for the full rationale.

@app.post("/api/auth/login")
async def auth_login(request: Request):
    """Verify the api_key and mint a session cookie.

    Request body: `{"api_key": "<DEV_API_KEY>"}` (Content-Type:
    application/json).  On success: 200 + Set-Cookie.  On failure:
    401 (no body detail — don't leak why).

    v0.4.0.1: rate-limit at the same 60 rpm / 10 burst bucket
    the rest of `/api/*` uses (`RATE_LIMIT_RPM` / `RATE_LIMIT_BURST`).
    Six failed logins from a single IP in a minute lock them out
    for 10 s, which is the difference between "annoying for a
    legit user mistyping" and "trivial brute force on a public
    endpoint".
    """
    if not DEV_API_KEY:
        log.error("DEV_API_KEY not configured — rejecting login")
        return JSONResponse({"error": "server misconfigured"}, status_code=500)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    provided = (body or {}).get("api_key", "")
    if not isinstance(provided, str) or not provided or provided != DEV_API_KEY:
        # Same response shape as the middleware 401 — no
        # detail leak.  Don't even log the provided value
        # (it could be a real key the user mistyped).
        log.info("auth_login: rejected (invalid api_key)")
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    token = sign_session_token(DEV_API_KEY)
    response = JSONResponse({"authenticated": True, "expires_in": SESSION_COOKIE_MAX_AGE})
    set_session_cookie(response, token, secure=SESSION_COOKIE_SECURE)
    return response


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    """Clear the session cookie.  Always 200 — the cookie may
    have already expired; that's not an error from the user's
    perspective.
    """
    response = JSONResponse({"authenticated": False})
    clear_session_cookie(response, secure=SESSION_COOKIE_SECURE)
    return response


@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Return whether the request's session cookie is valid.

    Always returns 200 — unauthenticated users get
    `{"authenticated": false}`, authenticated users get
    `{"authenticated": true}`.  The frontend uses this on
    page load to decide whether to show a login prompt.
    """
    token = read_session_cookie(request)
    if token and verify_session_token(DEV_API_KEY, token):
        return {"authenticated": True}
    return {"authenticated": False}


# ---------------------------------------------------------------------------- #
# Proxy — every other path forwards to the business service
# ---------------------------------------------------------------------------- #

@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_api(path: str, request: Request):
    """Forward /api/* to the business service."""
    body = await request.body()
    status, headers, stream = await stream_response(
        method=request.method,
        path=f"api/{path}",
        headers=dict(request.headers),
        query=request.url.query,
        body=body,
    )
    return StreamingResponse(stream, status_code=status, headers=headers)


@app.api_route(
    "/internal/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_internal(path: str, request: Request):
    """Forward /internal/* (service-to-service, not browser-facing)."""
    body = await request.body()
    status, headers, stream = await stream_response(
        method=request.method,
        path=f"internal/{path}",
        headers=dict(request.headers),
        query=request.url.query,
        body=body,
    )
    return StreamingResponse(stream, status_code=status, headers=headers)
