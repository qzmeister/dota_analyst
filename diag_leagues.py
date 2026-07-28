"""Check /api/leagues — see if EPL has match_count > 0."""
import json
import urllib.request

req = urllib.request.Request(
    "http://localhost/api/leagues",
    headers={"X-API-Key": "dev-local-dota-analyst-key-change-me"},
)
with urllib.request.urlopen(req, timeout=10) as r:
    d = json.loads(r.read().decode("utf-8"))

# Find EPL and a few others
for l in (d.get("leagues") or []):
    eid = l.get("id")
    if eid in (6617, 6620, 6624, 6490, 6626):
        print(
            f"  eid={eid:5} name={l.get('name', '?')!r:30} match_count={l.get('match_count')}"
        )
