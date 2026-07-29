"""
Gateway middleware stack.

Composition (outermost first):
  1. correlation_id   — generate / propagate X-Request-Id
  2. access_log       — structured one-line-per-request
  3. body_size_limit  — reject oversized payloads early
  4. cors             — allowlist from env
  5. auth             — X-API-Key for /api/* and /internal/*

Each middleware is small and explicit. No "framework magic" we can't
trace in five lines of reading.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ._rate_limit import RateLimiter

log = logging.getLogger("gateway")


# ---------------------------------------------------------------------------- #
# 1. Correlation ID
# ---------------------------------------------------------------------------- #

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Add or propagate an `X-Request-Id` for end-to-end tracing."""

    HEADER = "x-request-id"

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get(self.HEADER) or uuid.uuid4().hex
        request.state.request_id = cid
        response = await call_next(request)
        response.headers[self.HEADER] = cid
        return response


# ---------------------------------------------------------------------------- #
# 2. Access log
# ---------------------------------------------------------------------------- #

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "%s %s -> %s in %.1fms",
            request.method, request.url.path,
            response.status_code, elapsed_ms,
            extra={"request_id": getattr(request.state, "request_id", "")},
        )
        return response


# ---------------------------------------------------------------------------- #
# 3. Body size limit
# ---------------------------------------------------------------------------- #

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured cap."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            return JSONResponse(
                {"error": "payload too large", "max_bytes": self.max_bytes},
                status_code=413,
            )
        return await call_next(request)


# ---------------------------------------------------------------------------- #
# 4. Auth — X-API-Key for protected paths
# ---------------------------------------------------------------------------- #

class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate credentials on protected paths.

    v0.4.0.x: the SSE stream (`/api/stream/*`) now accepts EITHER
    the legacy `X-API-Key` header OR the `dota_analyst_session`
    cookie (HMAC-signed, set by `POST /api/auth/login`).  This
    unblocks public deploys where the browser's `EventSource`
    can't send custom headers — the cookie is sent automatically
    when the EventSource is opened with `withCredentials: true`.

    `UNAUTHED_PREFIXES` is the explicit bypass list (paths that
    pass through with no auth at all — health checks, etc.).
    Order matters in `dispatch()`: we test unauthed first, then
    the SSE cookie-or-header shortcut, then the full X-API-Key
    check for everything else.
    """

    PROTECTED_PREFIXES = ("/api/", "/internal/")
    # Paths that explicitly bypass ALL auth (even the cookie).
    # Keep tight — anything here is reachable without credentials.
    # `/api/auth/login` and friends are on this list because the
    # user has to be able to LOGIN before they can have a session
    # cookie.  `/api/auth/status` is also unauthed — it just
    # reports whether the request's cookie is valid; the answer
    # is `"authenticated": false` for unauthenticated users, which
    # is a fine public response.
    UNAUTHED_PREFIXES: tuple = (
        "/api/auth/login",
        "/api/auth/status",
    )
    # Paths that accept the session cookie as a credential.
    # Currently the SSE stream; could grow to cover any other
    # `EventSource`/long-poll endpoint the frontend needs.
    COOKIE_AUTHED_PREFIXES = ("/api/stream/",)

    def __init__(self, app: ASGIApp, expected_key: str):
        super().__init__(app)
        self.expected_key = expected_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # 1. Hard bypass.
        if any(path.startswith(p) for p in self.UNAUTHED_PREFIXES):
            return await call_next(request)
        # 2. SSE cookie auth.
        if any(path.startswith(p) for p in self.COOKIE_AUTHED_PREFIXES):
            # Header path: the X-API-Key check is the legacy / curl
            # path.  If the browser is using the static-UI nginx
            # edge that injects the dev key, this still works.
            provided = request.headers.get("x-api-key", "")
            if provided and provided == self.expected_key:
                return await call_next(request)
            # Cookie path: HMAC-verified, stateless.
            from ._session import read_session_cookie, verify_session_token
            token = read_session_cookie(request)
            if token and verify_session_token(self.expected_key, token):
                return await call_next(request)
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        # 3. Standard X-API-Key check.
        if any(path.startswith(p) for p in self.PROTECTED_PREFIXES):
            provided = request.headers.get("x-api-key", "")
            if not self.expected_key:
                log.error("DEV_API_KEY not configured — rejecting all auth-required requests")
                return JSONResponse({"error": "server misconfigured"}, status_code=500)
            if not provided or provided != self.expected_key:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


# ---------------------------------------------------------------------------- #
# 5. Rate limit — token bucket per (api_key, ip)
# ---------------------------------------------------------------------------- #

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose (api_key, ip) bucket is empty.

    Reads `RATE_LIMIT_RPM` and `RATE_LIMIT_BURST` from the env at
    construction time.  Set `RATE_LIMIT_RPM=0` to disable (useful
    in dev / smoke tests).

    The bucket key is the *provided* `X-API-Key` (or `"<none>"` for
    unauthenticated traffic) plus the client IP.  Pairing them keeps
    a single user on a shared key from starving the rest, and keeps
    an unauthenticated attacker from being limited only by their
    ability to rotate IPs.
    """

    def __init__(self, app: ASGIApp, rpm: int, burst: int):
        super().__init__(app)
        self.limiter = RateLimiter(rpm=rpm, burst=burst)

    async def dispatch(self, request: Request, call_next):
        # Skip the limiter for CORS preflight — those are cheap and
        # rate-limiting them would block the browser's first real
        # request behind 429.
        if request.method == "OPTIONS":
            return await call_next(request)
        key = request.headers.get("x-api-key") or "<none>"
        # `request.client` can be None in some ASGI test harnesses.
        ip = (request.client.host if request.client else "unknown")
        allowed, retry_after = self.limiter.try_consume(key, ip)
        if not allowed:
            log.info(
                "rate-limited %s %s for key=%s ip=%s retry_after=%ds",
                request.method, request.url.path, key, ip, retry_after,
                extra={"request_id": getattr(request.state, "request_id", "")},
            )
            resp = JSONResponse(
                {"error": "rate limit exceeded", "retry_after": retry_after},
                status_code=429,
            )
            resp.headers["Retry-After"] = str(retry_after)
            return resp
        return await call_next(request)


# ---------------------------------------------------------------------------- #
# Wiring helper
# ---------------------------------------------------------------------------- #

def install_middlewares(app, *, dev_api_key: str, body_max_bytes: int) -> None:
    """Add the gateway's middleware chain to a FastAPI app.

    Order matters — the last `add_middleware` call runs first.
    Starlette runs middlewares in reverse order of registration.
    """
    # 1. CORS (closest to the wire — runs first)
    origins = [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://localhost"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-Id"],
        allow_credentials=False,
    )
    # 2. Rate limit (before auth so brute-force on /api/* doesn't
    #    bypass the limiter just because the key is wrong).
    rate_rpm = int(os.environ.get("RATE_LIMIT_RPM", "60"))
    rate_burst = int(os.environ.get("RATE_LIMIT_BURST", "10"))
    app.add_middleware(RateLimitMiddleware, rpm=rate_rpm, burst=rate_burst)
    # 3. Auth
    app.add_middleware(ApiKeyAuthMiddleware, expected_key=dev_api_key)
    # 4. Body size
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=body_max_bytes)
    # 5. Access log
    app.add_middleware(AccessLogMiddleware)
    # 6. Correlation ID (outermost — wraps everything)
    app.add_middleware(CorrelationIdMiddleware)
