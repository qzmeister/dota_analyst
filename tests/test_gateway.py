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
# Unauthed paths (SSE)
# --------------------------------------------------------------------------- #

class TestUnauthedSse:
    """The /api/stream/* prefix bypasses the X-API-Key check.

    This is the 0.3.2 fix for "EventSource cannot send custom
    headers".  It is a LOCAL-ONLY workaround — for any public
    deployment, see TODO.md 0.4.0 (cookie-based auth) and remove
    the entry from `UNAUTHED_PREFIXES`.
    """

    def test_sse_path_passes_without_key(self):
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/stream/matches"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/api/stream/matches"]

    def test_sse_path_passes_with_wrong_key_too(self):
        # We never even CHECK the key on unauthed paths.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/stream/matches", api_key="wrong"),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/api/stream/matches"]

    def test_sse_prefix_takes_priority_over_protected(self):
        # `/api/stream/*` matches both `/api/` (protected) and
        # `/api/stream/` (unauthed).  The unauthed check runs
        # first and wins — that's the whole point of the bypass.
        record: List[str] = []
        mw = ApiKeyAuthMiddleware(app=None, expected_key="secret")
        response = asyncio.run(
            mw.dispatch(
                _make_request("GET", "/api/stream/something/else", api_key=None),
                _recording_handler(record),
            )
        )
        assert response.status_code == 200
        assert record == ["/api/stream/something/else"]

    def test_other_api_paths_still_require_key(self):
        # The bypass is prefix-specific.  `/api/board` is NOT
        # under the bypass and must still demand a key.
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
        # Pin the public contract.  Adding a path here is a
        # security decision; the test fails so the change shows
        # up in the review diff.
        from gateway._middleware import ApiKeyAuthMiddleware as M
        assert M.UNAUTHED_PREFIXES == ("/api/stream/",)
