"""
Tests for `gateway._middleware.ApiKeyAuthMiddleware` — the auth
gate that protects `/api/*` and `/internal/*`.

The middleware has two opt-out categories:
  - `PROTECTED_PREFIXES` — requires `X-API-Key` matching `DEV_API_KEY`
  - `UNAUTHED_PREFIXES` — bypasses the check entirely (SSE today)

We don't boot a full FastAPI app here; we hit the middleware
directly with a fake `Request` + a passthrough `call_next` that
records the URL it saw.  That keeps the test surface tight and
free of the rest of the gateway.
"""

from __future__ import annotations

import asyncio
from typing import Callable, List, Optional

import pytest
from starlette.requests import Request
from starlette.responses import Response

from gateway._middleware import ApiKeyAuthMiddleware


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

class _FakeClient:
    """Minimal stand-in for `starlette.requests.Client`."""
    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host


def _make_request(
    method: str,
    path: str,
    *,
    api_key: Optional[str] = None,
    host: str = "127.0.0.1",
) -> Request:
    """Build a Request the middleware can introspect."""
    headers: List[tuple] = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode("utf-8")))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": headers,
        "client": (host, 50000),
        "server": ("testserver", 80),
        "scheme": "http",
        "asgi": {"version": "3.0"},
    }
    return Request(scope)


def _recording_handler(record: List[str]) -> Callable:
    """A `call_next` that records the URL it was called with."""
    async def call_next(request: Request) -> Response:
        record.append(request.url.path)
        return Response("ok", status_code=200)
    return call_next


# --------------------------------------------------------------------------- #
# Authed paths
# --------------------------------------------------------------------------- #

class TestProtectedPaths:
    def test_no_key_returns_401(self):
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        # Replace `app` with None — `call_next` is the only path
        # we ever exercise here.
        mw.dispatch = lambda req, _next: ApiKeyAuthMiddleware.dispatch(mw, req, _next)
        response = asyncio.run(
            mw.dispatch(_make_request("GET", "/api/board"), _recording_handler(record))
        )
        assert response.status_code == 401
        assert record == []  # call_next was NOT invoked

    def test_wrong_key_returns_401(self):
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/board", api_key="wrong"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 401
        assert record == []

    def test_correct_key_passes_through(self):
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/board", api_key="secret"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/api/board"]

    def test_internal_path_also_requires_key(self):
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("POST", "/internal/retrain"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 401

    def test_unprotected_path_passes_through(self):
        # `/healthz` is not in PROTECTED_PREFIXES — no key check.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/healthz"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/healthz"]

    def test_missing_server_key_returns_500(self):
        # If DEV_API_KEY is empty on the server but the client
        # sends a key, we should reject (not silently accept) —
        # the "server misconfigured" branch.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/board", api_key="anything"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 500
        assert record == []


# --------------------------------------------------------------------------- #
# Cookie-authed paths (SSE, v0.4.0.1)
# --------------------------------------------------------------------------- #

class TestCookieAuthedSse:
    """The /api/stream/* prefix accepts EITHER the legacy
    X-API-Key header OR the HMAC-signed `dota_analyst_session`
    cookie.  This replaces the pre-0.4.0.1 "trust the network
    boundary" approach so the SSE path can be public-deploy-safe.
    """

    def _make_request_with_cookie(
        self, method: str, path: str, *, cookie: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Request:
        headers = []
        if api_key is not None:
            headers.append((b"x-api-key", api_key.encode("utf-8")))
        if cookie is not None:
            # ASGI cookies are a single header value; we encode
            # the cookie name + value with a single space.
            headers.append((
                b"cookie",
                f"dota_analyst_session={cookie}".encode("utf-8"),
            ))
        scope = {
            "type": "http", "method": method, "path": path,
            "raw_path": path.encode("utf-8"), "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
        }
        return Request(scope)

    def test_sse_path_with_valid_x_api_key_passes(self):
        # Legacy path: curl with the dev key still works.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/stream/matches", api_key="secret"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/api/stream/matches"]

    def test_sse_path_with_valid_session_cookie_passes(self):
        # Public-deploy path: browser logged in, has a cookie.
        from gateway._session import sign_session_token
        token = sign_session_token("secret", ttl_sec=60)
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                self._make_request_with_cookie(
                    "GET", "/api/stream/matches", cookie=token,
                ),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/api/stream/matches"]

    def test_sse_path_with_no_credentials_returns_401(self):
        # No header, no cookie — the unauthenticated browser
        # can't reach the SSE path.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/stream/matches"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 401
        assert record == []

    def test_sse_path_with_tampered_cookie_returns_401(self):
        # An attacker who guessed a valid expiry:nonce but
        # not the HMAC must be rejected.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                self._make_request_with_cookie(
                    "GET", "/api/stream/matches", cookie="1234567890:nonce.deadbeef",
                ),
                _recording_handler(record),
            )
        )
        assert response.status_code == 401
        assert record == []

    def test_sse_path_with_wrong_x_api_key_returns_401(self):
        # Legacy path with a wrong key: 401.  (In pre-0.4.0.1
        # this was a 200 because the SSE prefix was unauthed.
        # The test rename is intentional — see class docstring.)
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/stream/matches", api_key="wrong"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 401
        assert record == []

    def test_other_api_paths_still_require_key(self):
        # The cookie auth is prefix-specific.  `/api/board` is
        # NOT in COOKIE_AUTHED_PREFIXES — it must still demand
        # an X-API-Key (or get 401).
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/board"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 401
        assert record == []

    def test_unauthed_prefixes_constant(self):
        # v0.4.0.1: /api/stream/ moved out of UNAUTHED_PREFIXES
        # and into COOKIE_AUTHED_PREFIXES (cookie OR X-API-Key).
        # Only the login/status endpoints are fully unauthed.
        from gateway._middleware import ApiKeyAuthMiddleware as M
        assert M.UNAUTHED_PREFIXES == ("/api/auth/login", "/api/auth/status")
        assert M.COOKIE_AUTHED_PREFIXES == ("/api/stream/",)
