"""Stratz GraphQL introspection: dump the valid arguments
on the `teams` and `matches` root fields so we can write a
working query.

Usage:
    set STRATZ_API_KEY=<key>
    python stratz_introspect.py
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
            raw = r.read()
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            # body isn't JSON -- print the first 400 chars so
            # the user can see what Stratz / Cloudflare returned
            print(f"  HTTP {e.code} (non-JSON body, {len(raw)} bytes):",
                  file=sys.stderr)
            print("    " + raw[:400].decode("utf-8", errors="replace"),
                  file=sys.stderr)
            return {"errors": [{"message": f"non-JSON body (HTTP {e.code})"}]}


def main() -> int:
    print("=" * 78)
    print("Args on `teams` field:")
    print("=" * 78)
    q = """
    {
      __type(name: "DotaQuery") {
        fields {
          name
          args { name type { name kind ofType { name } } }
        }
      }
    }
    """
    payload = _gql(q)
    if "errors" in payload:
        print(json.dumps(payload["errors"], indent=2))
    else:
        for f in payload["data"]["__type"]["fields"]:
            if f["name"] in ("teams", "matches", "leagues", "heroes"):
                print(f"\n  Field: {f['name']}")
                for a in f["args"]:
                    t = a["type"]
                    tname = t.get("name") or t.get("ofType", {}).get("name", "?")
                    kind = t.get("kind", "?")
                    print(f"    {a['name']:30s}  {kind:10s}  {tname}")

    print()
    print("=" * 78)
    print("Trying candidate queries:")
    print("=" * 78)
    candidates = [
        ("request object: isPro",
         '{ teams(request: { isPro: true, take: 5 }) { id name rating } }'),
        ("topTeams field",
         '{ topTeams(take: 5) { id name rating } }'),
        ("heroes field test",
         '{ heroes(take: 1) { id displayName } }'),
        ("leagues field test",
         '{ leagues(take: 1) { id name } }'),
    ]
    for label, q in candidates:
        payload = _gql(q)
        if "errors" in payload:
            errs = [e.get("message", "?")[:100] for e in payload["errors"]]
            print(f"  {label:>40s}: ERR {errs}")
        else:
            data = payload.get("data", {})
            key = list(data.keys())[0] if data else "?"
            val = data.get(key)
            if isinstance(val, list):
                print(f"  {label:>40s}: OK  list[{len(val)}] first={val[0] if val else None}")
            else:
                print(f"  {label:>40s}: OK  {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
