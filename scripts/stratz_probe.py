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


def _gql(query: str) -> dict:
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
                return json.loads(raw)
            except Exception:
                print(f"  (non-JSON body, {len(raw)} bytes):",
                      file=sys.stderr)
                print("    " + raw[:400], file=sys.stderr)
                return {"errors": [{"message": "non-JSON body"}]}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            print(f"  HTTP {e.code} (non-JSON body): {raw[:300]}",
                  file=sys.stderr)
            return {"errors": [{"message": f"non-JSON HTTP {e.code}"}]}
    except Exception as e:
        return {"errors": [{"message": f"{type(e).__name__}: {e}"}]}


def main() -> int:
    print("=" * 78)
    print("Stratz raw-response probe -- dumps the full JSON shape")
    print("=" * 78)
    print()

    # 1. The "leagues(take:1) { id name }" query -- should always
    # return a list.  Dump the FULL JSON so we can see if it
    # returns a list, a connection object, or something else.
    print("--- Test 1: full JSON dump of leagues(take:1) ---")
    q = '{ leagues(take: 1) { id name } }'
    payload = _gql(q)
    print(json.dumps(payload, indent=2)[:1500])
    print()

    # 2. Same for teams
    print("--- Test 2: full JSON dump of teams(request: { take: 5, isPro: true }) ---")
    q = '{ teams(request: { take: 5, isPro: true }) { id name rating } }'
    payload = _gql(q)
    print(json.dumps(payload, indent=2)[:1500])
    print()

    # 3. Try the connection.nodes pattern (Stratz / Relay convention)
    print("--- Test 3: teams with .nodes wrapper ---")
    q = '{ teams(request: { take: 5, isPro: true }) { nodes { id name rating } } }'
    payload = _gql(q)
    print(json.dumps(payload, indent=2)[:1500])
    print()

    # 4. Try matches with .nodes
    print("--- Test 4: matches with .nodes wrapper ---")
    q = '{ matches(request: { take: 3 }) { nodes { id startDateTime } } }'
    payload = _gql(q)
    print(json.dumps(payload, indent=2)[:1500])
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
