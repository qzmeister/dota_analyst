"""Stratz query pattern probe -- try several known
query shapes and report which one returns data.

If the introspection query gets blocked (Cloudflare or
Stratz rate-limit on introspection specifically), we
fall back to trying many known query patterns and
picking the one that works.

Run:
    set STRATZ_API_KEY=<key>
    python stratz_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://api.stratz.com/graphql"


def _key() -> str:
    k = os.environ.get("STRATZ_API_KEY")
    if not k:
        print("ERROR: STRATZ_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)
    return k


def _gql(query: str) -> tuple[bool, dict, str]:
    """Returns (ok, parsed_or_empty_dict, raw_body_text)."""
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_key()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return (True, json.loads(raw), raw[:200])
            except Exception:
                return (False, {}, raw[:400])
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return (True, json.loads(raw), raw[:200])
        except Exception:
            return (False, {}, f"HTTP {e.code}: {raw[:300]}")
    except Exception as e:
        return (False, {}, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 78)
    print("Stratz query pattern probe -- trying many shapes")
    print("=" * 78)
    print()

    # The user's previous run got "Unknown argument 'take'" and
    # then "Unknown argument 'orderBy'", which means the field
    # accepts DIFFERENT arguments than my first guess.  The
    # Stratz docs (from memory) use `request: { ... }` input
    # object patterns.  Let's probe exhaustively.

    candidates = [
        # teams patterns
        ('teams(request: { take: 5, isPro: true })',
         '{ teams(request: { take: 5, isPro: true }) { id name rating } }'),
        ('teams(request: { take: 5 })',
         '{ teams(request: { take: 5 }) { id name rating } }'),
        ('teams(request: { isPro: true })',
         '{ teams(request: { isPro: true }) { id name rating } }'),
        ('teams(request: { isPro: true, take: 5 }, take: 5)',
         '{ teams(request: { isPro: true, take: 5 }, take: 5) { id name rating } }'),
        ('topTeams(take: 5)',
         '{ topTeams(take: 5) { id name rating } }'),
        ('proTeams(take: 5)',
         '{ proTeams(take: 5) { id name rating } }'),
        # matches patterns
        ('matches(request: { take: 5 })',
         '{ matches(request: { take: 5 }) { id } }'),
        ('matches(request: { take: 5, orderBy: START_DATE_TIME_DESC })',
         '{ matches(request: { take: 5, orderBy: START_DATE_TIME_DESC }) { id } }'),
        # league / hero -- should always work
        ('leagues(take: 1) (sanity check)',
         '{ leagues(take: 1) { id name } }'),
        ('heroes(take: 1) (sanity check)',
         '{ heroes(take: 1) { id displayName } }'),
    ]
    for label, q in candidates:
        ok, payload, raw = _gql(q)
        if "errors" in payload:
            errs = [e.get("message", "?")[:120] for e in payload["errors"]]
            print(f"  {label}")
            print(f"    ERR: {errs[0] if errs else '?'}")
        else:
            data = payload.get("data", {})
            key = list(data.keys())[0] if data else "?"
            val = data.get(key)
            if isinstance(val, list):
                first = val[0] if val else None
                print(f"  {label}")
                print(f"    OK  list[{len(val)}]  first={str(first)[:150]}")
            else:
                print(f"  {label}")
                print(f"    OK  {val}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
