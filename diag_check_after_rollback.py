"""Check API live count for different event combinations."""
import json
import urllib.request

combos = [
    ("none", ""),
    ("6617", "6617"),
    ("6620", "6620"),
    ("6624", "6624"),
    ("6490", "6490"),
    ("6626", "6626"),
    ("all5", "6620,6617,6624,6490,6626"),
    ("6620+6617", "6620,6617"),
    ("6620+6626", "6620,6626"),
]

for label, ids in combos:
    url = f"http://localhost/api/board?events={ids}"
    req = urllib.request.Request(url, headers={"X-API-Key": "dev-local-dota-analyst-key-change-me"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    n_live = len(d.get("live") or [])
    n_pre = len(d.get("prematch") or [])
    n_post = len(d.get("postmatch") or [])
    print(
        f"  events={label!r:>15} -> live={n_live:>3} prematch={n_pre:>3} postmatch={n_post:>3}  "
        f"filtered={d.get('filtered_from_auto')}"
    )
