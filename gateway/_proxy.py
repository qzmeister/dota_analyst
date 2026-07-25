"""
Reverse-proxy helper for the gateway service.

Forward incoming requests to the business service, streaming the
response back. Used for both REST (`/api/*`) and SSE (`/api/stream/*`).

Uses `httpx` for async proxying. `httpx` is already a transitive dep of
FastAPI's test client, but we list it explicitly in pyproject.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Dict, Iterable, Tuple

import httpx


BUSINESS_URL = os.environ.get("BUSINESS_URL", "http://business:8000").rstrip("/")
PROXY_TIMEOUT = float(os.environ.get("GATEWAY_PROXY_TIMEOUT", "30"))


# Headers we never forward to the business service. The gateway owns these.
_HOP_BY_HOP = frozenset({
    "host", "content-length", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade",
})


def _filter_request_headers(headers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers:
        lk = k.lower()
        if lk in _HOP_BY_HOP:
            continue
        out[k] = v
    return out


async def stream_response(
    method: str,
    path: str,
    *,
    headers: Dict[str, str],
    query: str = "",
    body: bytes = b"",
) -> Tuple[int, Dict[str, str], AsyncIterator[bytes]]:
    """Forward `method /path?query` to the business service.

    Returns (status, response_headers, response_body_stream).
    The stream is async-iterable and must be consumed by the caller.
    """
    url = f"{BUSINESS_URL}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"

    client = httpx.AsyncClient(timeout=PROXY_TIMEOUT)
    req = client.build_request(
        method=method,
        url=url,
        headers=_filter_request_headers(headers.items()),
        content=body if body else None,
    )

    resp = await client.send(req, stream=True)

    async def _iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    # Filter hop-by-hop from response too
    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    return resp.status_code, response_headers, _iter()
