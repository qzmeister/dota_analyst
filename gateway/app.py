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

# Configure logging once on import. `setup_logging` is idempotent.
setup_logging()
log = get_logger(__name__)


# ---------------------------------------------------------------------------- #
# App
# ---------------------------------------------------------------------------- #

app = FastAPI(title="Dota Analyst — gateway", version="0.1.0", docs_url=None, redoc_url=None)

DEV_API_KEY = os.environ.get("DEV_API_KEY", "")
BODY_MAX_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(1024 * 1024)))

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
