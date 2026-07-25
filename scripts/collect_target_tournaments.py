"""Collect full DatDota maps for the explicitly selected historical tournaments.

The collector is resumable, skips maps already present in ``ml_data/full_matches``
and respects DatDota's three-second request interval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.datdota_client import get_league_matches, get_match_details


TARGETS = {
    17875: "Hero Esports Asian Champions League 2025 (ACL×ESL Challenger China)",
    19255: "NarodCast Premier Series 1",
    19239: "FISSURE Universe 8",
    18633: "FISSURE Universe 7",
    18107: "FISSURE Universe 5",
    18046: "FISSURE Special 1",
}
OUTPUT_DIR = Path("ml_data/full_matches")
MANIFEST_PATH = Path("ml_data/target_tournaments_matches.json")
PROGRESS_PATH = Path("ml_data/target_tournaments_progress.json")
REQUEST_DELAY_SECONDS = 3.0


def log_text(value: object) -> str:
    """Keep a collector running even when Windows console lacks Unicode glyphs."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {"fetched_ids": [], "failed_ids": [], "last_run": None}


def save_progress(progress: dict) -> None:
    progress["last_run"] = datetime.now().isoformat()
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def collect_manifest() -> list[dict]:
    """Fetch only the match indexes for the requested tournaments."""
    rows: list[dict] = []
    for league_id, requested_name in TARGETS.items():
        data = get_league_matches(league_id) or {}
        matches = (data.get("data") or {}).get("matches", {}).get("data", [])
        league = (data.get("data") or {}).get("league", {})
        actual_name = league.get("name") or requested_name
        for match in matches:
            match_id = match.get("matchId")
            if match_id:
                rows.append({"match_id": int(match_id), "league_id": league_id, "league": actual_name})
        print(f"{log_text(actual_name)}: {len(matches)} maps")
        time.sleep(REQUEST_DELAY_SECONDS)
    unique = {row["match_id"]: row for row in rows}
    manifest = list(unique.values())
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Optional cap for a single run; omitted means no cap")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = collect_manifest()
    if args.manifest_only:
        return
    progress = load_progress()
    fetched = set(progress["fetched_ids"])
    failed = set(progress["failed_ids"])
    remaining = [row for row in manifest if row["match_id"] not in fetched and not (OUTPUT_DIR / f"{row['match_id']}.json").exists()]
    batch = remaining[:args.limit] if args.limit else remaining
    for index, row in enumerate(batch, start=1):
        match_id = row["match_id"]
        print(f"[{index}/{len(batch)}] {log_text(row['league'])} - {match_id}")
        payload = get_match_details(match_id)
        details = (payload or {}).get("data")
        if details:
            (OUTPUT_DIR / f"{match_id}.json").write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
            fetched.add(match_id)
        else:
            failed.add(match_id)
        progress["fetched_ids"] = sorted(fetched)
        progress["failed_ids"] = sorted(failed)
        save_progress(progress)
        if not details:
            print("[STOP] DatDota returned no match details; progress was saved.")
            break
        if index < len(batch):
            time.sleep(REQUEST_DELAY_SECONDS)
    print(f"Saved: {len(fetched)}; failed: {len(failed)}; remaining: {len(remaining) - len(batch)}")


if __name__ == "__main__":
    main()
