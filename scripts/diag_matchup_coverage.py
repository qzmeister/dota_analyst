"""Diagnostic: how many OOS test matchups hit the lookup vs fallback."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from business.ml.features import (
    HeroWinRateEncoder, CrossSideMatchupEncoder,
    lane_heroes_from_match,
)
from business.ml.targets import extract_target

DATA_DIR = ROOT / "ml_data" / "full_matches"


def load():
    raw, targets = [], []
    for p in sorted(DATA_DIR.glob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except: continue
        t = extract_target(d)
        if t is None: continue
        raw.append(d); targets.append(t)
    return raw, targets


def main():
    raw, targets = load()
    print(f"loaded {len(raw)} matches")
    y = np.asarray([t.winner for t in targets], dtype=int)
    idx = np.arange(len(targets))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y,
    )
    v1_train = idx_train[:883]
    v1_test = np.concatenate([idx_train[883:], idx_test])
    raw_train = [raw[i] for i in v1_train]

    enc = CrossSideMatchupEncoder(smoothing=3.0).fit(raw_train)

    n_bot_hit = n_bot_miss = 0
    n_top_hit = n_top_miss = 0
    n_mid_hit = n_mid_miss = 0
    for i in v1_test:
        lanes = lane_heroes_from_match(raw[i])
        r = lanes["radiant"]; d = lanes["dire"]
        if r["BOT_CARRY"] is not None and r["BOT_SUPPORT"] is not None and d["BOT_CARRY"] is not None and d["BOT_SUPPORT"] is not None:
            k = frozenset({r["BOT_CARRY"], r["BOT_SUPPORT"], d["BOT_CARRY"], d["BOT_SUPPORT"]})
            if k in enc._bot_2v2: n_bot_hit += 1
            else: n_bot_miss += 1
        if r["TOP_OFFLANE"] is not None and r["TOP_JUNGLER"] is not None and d["TOP_OFFLANE"] is not None and d["TOP_JUNGLER"] is not None:
            k = frozenset({r["TOP_OFFLANE"], r["TOP_JUNGLER"], d["TOP_OFFLANE"], d["TOP_JUNGLER"]})
            if k in enc._top_2v2: n_top_hit += 1
            else: n_top_miss += 1
        if r["MID"] is not None and d["MID"] is not None:
            k = frozenset({r["MID"], d["MID"]})
            if k in enc._mid_1v1: n_mid_hit += 1
            else: n_mid_miss += 1

    print(f"\n=== OOS coverage (1497 matches) ===")
    print(f"bot  2v2:  hit={n_bot_hit}  miss={n_bot_miss}  hit_rate={n_bot_hit/(n_bot_hit+n_bot_miss):.1%}")
    print(f"top  2v2:  hit={n_top_hit}  miss={n_top_miss}  hit_rate={n_top_hit/(n_top_hit+n_top_miss):.1%}")
    print(f"mid  1v1:  hit={n_mid_hit}  miss={n_mid_miss}  hit_rate={n_mid_hit/(n_mid_hit+n_mid_miss):.1%}")
    print(f"\n=== Train lookup table size ===")
    print(f"bot  2v2 keys: {len(enc._bot_2v2)}")
    print(f"top  2v2 keys: {len(enc._top_2v2)}")
    print(f"mid  1v1 keys: {len(enc._mid_1v1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
