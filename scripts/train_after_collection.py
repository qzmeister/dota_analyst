"""Wait for the target-tournament collector, then retrain the prematch model.

Designed to run as a detached helper alongside collect_target_tournaments.py.
It trains only when every map in the current manifest was downloaded
successfully; a stopped collector or failed map never silently produces a
partially refreshed model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "ml_data" / "target_tournaments_matches.json"
PROGRESS_PATH = ROOT / "ml_data" / "target_tournaments_progress.json"
STATUS_PATH = ROOT / "ml_data" / "post_collection_training_status.json"
TRAINER = ROOT / "scripts" / "train_temporal_prematch.py"
POLL_SECONDS = 20


def write_status(state: str, **extra: object) -> None:
    STATUS_PATH.write_text(
        json.dumps({"state": state, "updated_at": time.time(), **extra}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    while True:
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
            total = len(manifest)
            fetched = len(progress.get("fetched_ids", []))
            failed = len(progress.get("failed_ids", []))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            write_status("waiting_for_manifest", error=str(exc))
            time.sleep(POLL_SECONDS)
            continue

        if failed:
            write_status("blocked_by_failed_maps", total=total, fetched=fetched, failed=failed)
            return
        if fetched < total:
            write_status("waiting_for_collection", total=total, fetched=fetched, remaining=total - fetched)
            time.sleep(POLL_SECONDS)
            continue

        write_status("training", total=total, fetched=fetched)
        try:
            subprocess.run([sys.executable, str(TRAINER)], cwd=ROOT, check=True, timeout=300)
        except Exception as exc:
            write_status("training_failed", total=total, fetched=fetched, error=str(exc))
            raise
        write_status("complete", total=total, fetched=fetched)
        return


if __name__ == "__main__":
    main()
