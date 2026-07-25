# -*- coding: utf-8 -*-
"""
Target extraction for the regression / quantile heads.

The classifier (`winner`) and the encoders (`features.py`) live in
their own files; this one collects the per-match target extractors
for the count / duration regressors that 0.2.1 introduces.

Why per file?  Three reasons:

  1. Each target has its own "skip me if I'm unreliable" condition
     (e.g. duration < 5 min = remake = drop).  Centralising the
     filter in `iter_clean_targets` keeps the trainer's train loop
     trivial.
  2. Several targets share the same hero_id extraction (already in
     `features.py`); re-using `hero_ids_from_match` here avoids
     duplicating that list comprehension.
  3. The pre-match features are the same `extract_features(...)`
     across all targets; only the `y` changes.  The trainer can
     therefore iterate `iter_clean_targets` once and build the full
     `(X, y_kills, y_duration, y_winner, ...)` matrix in one pass.

Targets available in 0.2.1
--------------------------
  - `winner`            -- binary (radiant win = 1, dire win = 0)
  - `kills_total`       -- sum of `performance.kills` for all 10 players
  - `duration_minutes`  -- match duration in minutes (seconds / 60)
  - `duration_p10`      -- placeholder for the 10th-percentile quantile
  - `duration_p90`      -- placeholder for the 90th-percentile quantile

For 0.2.1 we train point predictors for kills + duration; the P10/P90
quantile heads are filled in by the same models (the point estimate
becomes the median) and XGBoost's `reg:quantileerror` takes over
in a later pass.

Targets NOT in 0.2.1 (deferred)
-------------------------------
  - `towers_total`      -- DatDota full_matches do NOT carry per-side
                          tower bitmasks.  Need to either (a) re-pull
                          matches from DLTV, or (b) use `building_damage`
                          per player as a proxy.  Both are 0.2.2 work.
  - `multikill`         -- categorical (Low/Medium/High), not a regression
                          target.  Will need its own classifier and its
                          own evaluation.  Deferred to 0.3.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .features import hero_ids_from_match


# Match-quality filter -- drop remakes / abandons.  In a pro-scene
# corpus the practical floor is ~10 minutes (anything shorter is a
# remake); the ceiling catches corrupted rows.
MIN_DURATION_SEC = 600          # 10 min
MAX_DURATION_SEC = 90 * 60      # 90 min (some long late-game stalls)


@dataclass
class MatchTarget:
    """All the per-match labels we might want to fit.

    `kills_total` and `duration_minutes` are 0/None when the match
    was filtered out.  The trainer uses these to decide whether to
    include a row in a given regressor's (X, y) matrix.
    """
    match_id: int
    winner: Optional[int]            # 1 = radiant, 0 = dire, None = skip
    kills_total: Optional[int]
    duration_minutes: Optional[float]
    towers_total: Optional[int]      # 0.2.2 -- None until tower data lands
    multikill_level: Optional[str]   # 0.3.0 -- "Low" | "Medium" | "High" | None
    radiant_hero_ids: List[int]
    dire_hero_ids: List[int]
    radiant_team_id: Optional[int] = None  # 0.3.9 — team.valve_id for team features
    dire_team_id: Optional[int] = None     # 0.3.9 — team.valve_id for team features


# --------------------------------------------------------------------------- #
# target_multikill  (0.3.0)
# --------------------------------------------------------------------------- #

#: Thresholds for the categorical multikill target.  "Ultra-kill /
#: rampage" in Dota terminology is the player with the most kills
#: in a single match.  The heuristic in `analysis.py` uses
#: `MULTIKILL_HIGH_SCORE = 7` and `MULTIKILL_MEDIUM_SCORE = 4` --
#: we mirror that here so train and heuristic agree on the bins.
MULTIKILL_HIGH_THRESHOLD = 7
MULTIKILL_MEDIUM_THRESHOLD = 4


def _max_player_kills(match: Dict) -> Optional[int]:
    """The single highest individual kill count in the match."""
    best = 0
    found = False
    for side in ("radiant", "dire"):
        for p in (match.get(side) or {}).get("player_performances") or []:
            k = (p.get("performance") or {}).get("kills")
            if isinstance(k, (int, float)):
                found = True
                if k > best:
                    best = int(k)
    return best if found else None


def target_multikill(match: Dict) -> Optional[str]:
    """Categorical multikill level: "High" / "Medium" / "Low".

    Returns None if the match lacks per-player performance data.
    The threshold matches `analysis.MULTIKILL_HIGH_SCORE` and
    `analysis.MULTIKILL_MEDIUM_SCORE` so the classifier and the
    heuristic share the same bins (otherwise the A/B comparison
    would be apples-to-oranges).
    """
    max_kills = _max_player_kills(match)
    if max_kills is None:
        return None
    if max_kills >= MULTIKILL_HIGH_THRESHOLD:
        return "High"
    if max_kills >= MULTIKILL_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def _player_kill_sum(side: Dict) -> int:
    """Sum `performance.kills` for every player on a side."""
    total = 0
    for p in side.get("player_performances") or []:
        perf = p.get("performance") or {}
        k = perf.get("kills")
        if isinstance(k, (int, float)):
            total += int(k)
    return total


def target_kills(match: Dict) -> Optional[int]:
    """Total kills across both teams.

    Returns None when the match lacks per-player performance data
    (e.g. a dropped parse) -- the trainer should filter those rows.
    """
    if not isinstance(match, dict):
        return None
    radiant = match.get("radiant") or {}
    dire = match.get("dire") or {}
    if not radiant.get("player_performances") or not dire.get("player_performances"):
        return None
    return _player_kill_sum(radiant) + _player_kill_sum(dire)


# --------------------------------------------------------------------------- #
# target_towers  (0.2.2 -- corpus has no per-side tower bitmask yet)
# --------------------------------------------------------------------------- #

#: DLTV bitmask convention: 11 bits per side, one tower per bit.
#: A SET bit means "tower destroyed" -- see `analysis.decode_towers`
#: for the historical background.  We only need the *total* here.
_DLTV_TOWER_TOTAL = 11


def target_towers(match: Dict) -> Optional[int]:
    """Total towers destroyed across both sides.

    The 0.2.1 corpus (`ml_data/full_matches/*.json`) does NOT carry
    per-side tower bitmasks -- that field is DLTV-specific.  0.2.2
    re-pulls from a tower-aware source (or accepts `building_damage`
    as a proxy); until then, this returns `None` for every match
    and the trainer simply skips the towers target.

    When tower data is present, the match dict is expected to have
    `tower_radiant` and `tower_dire` (int bitmasks) at the top
    level -- same convention DLTV uses.
    """
    if not isinstance(match, dict):
        return None
    tower_r = match.get("tower_radiant")
    tower_d = match.get("tower_dire")
    if not isinstance(tower_r, int) or not isinstance(tower_d, int):
        return None
    return min(_DLTV_TOWER_TOTAL, bin(tower_r).count("1")) + min(
        _DLTV_TOWER_TOTAL, bin(tower_d).count("1")
    )


def target_duration_minutes(match: Dict) -> Optional[float]:
    """Match duration in minutes (float), or None on missing / out-of-range."""
    d = match.get("duration") if isinstance(match, dict) else None
    if not isinstance(d, (int, float)):
        return None
    if d < MIN_DURATION_SEC or d > MAX_DURATION_SEC:
        return None
    return float(d) / 60.0


def extract_target(match: Dict) -> Optional[MatchTarget]:
    """Build a `MatchTarget` for one match dict, or None if it can't be used.

    A match is usable iff:
      - it's a dict
      - not errored
      - has a clear winner
      - has 5+5 hero picks (so we can build the feature vector)
      - duration is in the sane range
    """
    if not isinstance(match, dict) or match.get("has_error"):
        return None
    if "radiant_victory" not in match:
        return None

    r_ids, d_ids = hero_ids_from_match(match)
    if len(r_ids) != 5 or len(d_ids) != 5 or any(x is None for x in r_ids + d_ids):
        return None

    duration = target_duration_minutes(match)
    if duration is None:
        return None

    kills = target_kills(match)
    if kills is None:
        return None

    winner = 1 if match.get("radiant_victory") else 0
    # Team id for the team aggregates feature (0.3.9).  We mirror
    # the lookup in `TeamWinRateEncoder._team_key` so train and
    # predict agree on the keying.
    r_team = (match.get("radiant") or {}).get("team") or {}
    d_team = (match.get("dire") or {}).get("team") or {}
    r_team_id = r_team.get("valve_id") if isinstance(r_team.get("valve_id"), int) else None
    d_team_id = d_team.get("valve_id") if isinstance(d_team.get("valve_id"), int) else None

    return MatchTarget(
        match_id=int(match.get("match_id") or 0),
        winner=winner,
        kills_total=int(kills),
        duration_minutes=float(duration),
        towers_total=target_towers(match),  # None for 0.2.1 corpus
        multikill_level=target_multikill(match),  # "Low"/"Medium"/"High"
        radiant_hero_ids=r_ids,  # type: ignore[arg-type]
        dire_hero_ids=d_ids,     # type: ignore[arg-type]
        radiant_team_id=r_team_id,
        dire_team_id=d_team_id,
    )


def iter_clean_targets(matches) -> List[MatchTarget]:
    """Walk a sequence of raw match dicts and return the usable ones.

    Returning a `list` (not a generator) is deliberate -- the trainer
    iterates twice: once to build the per-target matrices and once
    to apply winsorize.  A list gives the second pass stable indices.
    """
    out: List[MatchTarget] = []
    for m in matches:
        t = extract_target(m)
        if t is not None:
            out.append(t)
    return out
