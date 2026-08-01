"""Hero ID mapping audit: DLTV internal vs Steam vs OpenDota.

Dota 2 has two parallel hero ID namespaces:
  - DLTV internal (1..127 historically; some heroes added with new IDs)
  - Valve/Steam/OpenDota (1..155 in the current hero pool)

The v18 model was trained on OpenDota hero IDs (Steam namespace).
The live card pulls hero IDs from DLTV's API.  If those IDs are
in the DLTV namespace and we don't convert, the model sees
garbage hero IDs (e.g. DLTV id=120 -> OpenDota 120 = Pangolier,
but DLTV 120 might be Hoodwink).

This script dumps the DLTV hero index and flags any heroes whose
DLTV id != Steam id, so we can build a proper mapping.
"""
import sys
import json
from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRO_ROOT))


def main():
    from business.dltv_client import client
    heroes = client.get_heroes() or []
    print(f"total heroes in DLTV: {len(heroes)}")
    if not heroes:
        print("  (DLTV hero index empty)")
        return

    # Sample
    print("\nSample 5 heroes (first 5 by DLTV id):")
    for h in sorted(heroes, key=lambda x: (x.get("id") or 0))[:5]:
        sid = h.get("steam_id")
        print(f"  DLTV id={h.get('id'):>4}  steam_id={str(sid):>4}  title={h.get('title')}")

    # Range check
    dltv_ids = [h.get("id") for h in heroes if h.get("id") is not None]
    steam_ids = [h.get("steam_id") for h in heroes if h.get("steam_id") is not None]
    print(f"\nDLTV id range: min={min(dltv_ids)}  max={max(dltv_ids)}  count={len(dltv_ids)}")
    print(f"Steam id range: min={min(steam_ids) if steam_ids else None}  max={max(steam_ids) if steam_ids else None}  count={len(steam_ids)}")

    # Mismatches: DLTV id != steam_id
    mismatches = [h for h in heroes
                  if h.get("id") is not None and h.get("steam_id") is not None
                  and h.get("id") != h.get("steam_id")]
    print(f"\nheroes with DLTV id != steam_id: {len(mismatches)}")
    print("(most of these are real remappings; we want to know if the")
    print(" live card passes DLTV id where v18 expects steam id)")
    for h in sorted(mismatches, key=lambda x: x.get("id"))[:10]:
        sid = h.get("steam_id")
        print(f"  DLTV id={h.get('id'):>4}  steam_id={str(sid):>4}  title={h.get('title')}")


if __name__ == "__main__":
    main()
