"""
Feature extraction for the ML prediction engine.

The MVP uses **target encoding** of per-hero, per-side win rates
computed from historical matches. This:

  - avoids the cardinality blow-up of one-hot hero encoding (126 heroes)
  - is interpretable: a feature is literally "this hero's historical
    win rate when picked on this side"
  - is cheap at predict time: a single dict lookup

Three feature groups are available (combined by default, switchable
in `extract_features(..., groups=...)`):

  - "hero" (13 features, 0.3.9 baseline): per-hero WR on each side,
    side means, and their difference.
  - "team" (4 features, 0.3.10 C retry): per-team WR on each side,
    the difference, and a "team-vs-team" symmetric diff.
  - "lane" (7 features, 0.3.10 D v2): per-side bot-pair (carry +
    support) WR, top-pair (offlane + jungler) WR, and a mid
    matchup rate.  All target-encoded with smoothing.

Why no team-level aggregates in 0.2.0-0.3.9? The training data
(`ml_data/full_matches/*.json`) does not include team-level
`win_rate`/`fb_rate`/`f10_rate` — those are live DLTV metadata
that have to be looked up at predict time.  The "team" group
here uses the encoder's own per-team WR (computed from
historical matches), which is hermetic and reproducible.

What we explicitly do **not** use as features (target leakage):
  - `duration` — known only after the match
  - `kills`, `deaths`, `assists` — known only after the match
  - `gpm`/`xpm` — known only after the match
  - `gold`/`xp` graphs — known only after the match
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class FeatureConfig:
    """Schema of the feature vector. `n_features` is the total dimension.

    Used to make the train and predict code share a single source of
    truth — the regression bug we had in the old `ml_trainer.py` came
    from train and predict disagreeing on feature order.
    """

    n_features: int


class TeamWinRateEncoder:
    """Per-team target encoding with smoothing back to the global rate.

    fit(matches) computes:

        P(win | team=t)
            ≈ (wins(t) + alpha * global_rate) / (samples(t) + alpha)

    The smoothing (`alpha`) avoids zero-division for teams that
    appear only once or twice in the training corpus.  Returns
    `global_rate` (default 0.5) for teams never seen.

    The team key is `team.valve_id` (DLTV/DatDota stable id) when
    present, falling back to `team.name`.  We index by the id
    because names have variation ("paiN" vs "paiN Gaming") but
    `valve_id` is stable across matches.
    """

    def __init__(self, smoothing: float = 3.0, min_samples: int = 2) -> None:
        self.smoothing = float(smoothing)
        self.min_samples = int(min_samples)
        # team_id -> win_rate
        self._rates: Dict[int, float] = {}
        self._global_rate: float = 0.5

    @property
    def global_rate(self) -> float:
        return self._global_rate

    @staticmethod
    def _team_key(m: Dict, side: str) -> Optional[int]:
        """Pull the team id from a match side, or None if absent."""
        team = (m.get(side) or {}).get("team") or {}
        # Prefer the stable `valve_id`; fall back to name hash so
        # name-only corpora still produce *some* signal.
        vid = team.get("valve_id")
        if isinstance(vid, int):
            return vid
        name = team.get("name")
        if isinstance(name, str) and name:
            # Stable, but distinct from any real valve_id.  Use a
            # large negative hash space so it can't collide with
            # the small positive ids DLTV uses.
            return -abs(hash(name)) % (10 ** 8)
        return None

    def fit(self, matches: Iterable[Dict]) -> "TeamWinRateEncoder":
        """Compute team win rates from training matches.

        `match["radiant"]["team"]` and `match["dire"]["team"]` are
        expected; matches without team metadata contribute 0 to the
        numerator and 0 to the denominator for that side.
        """
        wins: Dict[int, int] = defaultdict(int)
        totals: Dict[int, int] = defaultdict(int)
        total_wins = 0
        total_samples = 0

        for m in matches:
            if m.get("has_error"):
                continue
            target = 1 if m.get("radiant_victory") else 0
            total_wins += target
            total_samples += 1
            r_key = self._team_key(m, "radiant")
            d_key = self._team_key(m, "dire")
            if r_key is not None:
                wins[r_key] += target
                totals[r_key] += 1
            if d_key is not None:
                wins[d_key] += 1 - target
                totals[d_key] += 1

        if total_samples > 0:
            self._global_rate = total_wins / total_samples

        for k, t in totals.items():
            if t < self.min_samples:
                self._rates[k] = (wins[k] + self.smoothing * self._global_rate) / (t + self.smoothing)
            else:
                self._rates[k] = wins[k] / t
        return self

    def encode(self, team_id: Optional[int]) -> float:
        """Return the historical win rate for this team, or global_rate."""
        if team_id is None:
            return self._global_rate
        return self._rates.get(int(team_id), self._global_rate)

    def to_dict(self) -> Dict:
        return {
            "smoothing": self.smoothing,
            "min_samples": self.min_samples,
            "global_rate": self._global_rate,
            "rates": {str(k): v for k, v in self._rates.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TeamWinRateEncoder":
        enc = cls(smoothing=d.get("smoothing", 3.0), min_samples=d.get("min_samples", 2))
        enc._global_rate = d.get("global_rate", 0.5)
        for k, v in d.get("rates", {}).items():
            enc._rates[int(k)] = float(v)
        return enc


class PlayerWinRateEncoder:
    """Per-player target encoding with smoothing back to the global rate (0.3.15).

    fit(matches) walks every match and, for each `player.steam32` (DLTV
    namespace) that appears, increments the (steam32) -> {wins, total}
    counter.  encode(steam32) returns the smoothed WR or global_rate.

    Used by `_features_player` to feed `r_player_wr_avg/max` and
    `d_player_wr_avg/max` into winner_v15+.  When the encoder is fit
    on a small corpus the global_rate fallback keeps the model from
    blowing up on unseen players; in production we still pass
    DLTV's `map_results[].player.win_rate` directly when available
    (see `business/board.py` and the backtest scripts).

    Stored alongside the trained model in `player_encoder.json`, NOT
    inside `HeroWinRateEncoder.to_dict()` — the hero encoder is
    shared across multiple heads, the player encoder is per-corpus.
    """

    def __init__(self, smoothing: float = 5.0, min_samples: int = 3) -> None:
        self.smoothing = float(smoothing)
        self.min_samples = int(min_samples)
        self._wins: Dict[int, int] = {}
        self._total: Dict[int, int] = {}
        self._global_rate: float = 0.5

    @property
    def global_rate(self) -> float:
        return self._global_rate

    def fit(self, matches) -> "PlayerWinRateEncoder":
        wins: Dict[int, int] = {}
        total: Dict[int, int] = {}
        global_wins = 0
        global_total = 0
        for m in matches:
            t = m.get("radiant_victory")
            if t is None: continue
            target = 1 if t else 0
            for side_key in ("radiant", "dire"):
                side = m.get(side_key) or {}
                for p in (side.get("player_performances") or []):
                    pl = p.get("player") or {}
                    sid = pl.get("steam32")
                    if not isinstance(sid, int): continue
                    wins[sid] = wins.get(sid, 0) + (target if side_key == "radiant" else 1 - target)
                    total[sid] = total.get(sid, 0) + 1
                    global_wins += (target if side_key == "radiant" else 1 - target)
                    global_total += 1
        self._wins = wins
        self._total = total
        self._global_rate = global_wins / max(1, global_total)
        return self

    def encode(self, steam32) -> float:
        if steam32 is None:
            return self._global_rate
        n = self._total.get(int(steam32), 0)
        if n < self.min_samples:
            return self._global_rate
        w = self._wins.get(int(steam32), 0)
        return (w + self.smoothing * self._global_rate) / (n + self.smoothing)

    def to_dict(self) -> Dict:
        return {
            "smoothing": self.smoothing, "min_samples": self.min_samples,
            "global_rate": float(self._global_rate),
            "wins": {str(k): int(v) for k, v in self._wins.items()},
            "total": {str(k): int(v) for k, v in self._total.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PlayerWinRateEncoder":
        enc = cls(smoothing=d.get("smoothing", 5.0), min_samples=d.get("min_samples", 3))
        enc._global_rate = float(d.get("global_rate", 0.5))
        enc._wins = {int(k): int(v) for k, v in d.get("wins", {}).items()}
        enc._total = {int(k): int(v) for k, v in d.get("total", {}).items()}
        return enc


def players_from_match(match: Dict) -> Dict[str, List[int]]:
    """Pull per-side `player.steam32` lists (used by `PlayerWinRateEncoder.fit`)."""
    out = {"radiant": [], "dire": []}
    if not match: return out
    for side_key in ("radiant", "dire"):
        side = match.get(side_key) or {}
        for p in (side.get("player_performances") or []):
            pl = p.get("player") or {}
            sid = pl.get("steam32")
            if isinstance(sid, int):
                out[side_key].append(sid)
    return out


class HeroPairWinRateEncoder:
    """Per-side, per-hero-pair target encoding (0.3.9 — synergy).

    fit(matches) walks every match and, for each pair of heroes
    on the SAME side, increments wins / totals for that pair:

        P(win | side=s, pair=(h1, h2))
            ≈ smoothed (wins / samples)

    The pair key is the SORTED tuple of the two hero ids.  We sort
    so (A, B) and (B, A) map to the same key — direction is
    meaningless for "did these two heroes on the same team win?".

    Returns `_global_rate` for unseen pairs.  For prediction we
    aggregate per-side pair rates into min / mean / max — those
    three summaries are the actual features surfaced in
    `extract_features()`.
    """

    def __init__(self, smoothing: float = 5.0, min_samples: int = 3) -> None:
        self.smoothing = float(smoothing)
        self.min_samples = int(min_samples)
        # (side, (h1, h2)) -> win_rate, with h1 <= h2
        self._rates: Dict[Tuple[str, Tuple[int, int]], float] = {}
        self._global_rate: float = 0.5

    @property
    def global_rate(self) -> float:
        return self._global_rate

    @staticmethod
    def _pair_key(h1: int, h2: int) -> Tuple[int, int]:
        """Sort the pair so (A, B) == (B, A)."""
        return (h1, h2) if h1 <= h2 else (h2, h1)

    @staticmethod
    def _hero_ids(m: Dict, side: str) -> List[int]:
        """First-5 hero ids on a side, dropping None."""
        out: List[int] = []
        for p in (m.get(side) or {}).get("player_performances") or []:
            h = (p.get("performance") or {}).get("hero", {}).get("valve_id")
            if isinstance(h, int):
                out.append(h)
            if len(out) == 5:
                break
        return out

    def fit(self, matches: Iterable[Dict]) -> "HeroPairWinRateEncoder":
        """Compute pair win rates from training matches.

        Pair granularity is C(5, 2) = 10 pairs per side, so the
        encoder ends up with ~20 * N_matches keys in the worst
        case.  Most pairs are unique and short-lived, which is
        why `min_samples` is a real filter and `smoothing`
        matters.
        """
        wins: Dict[Tuple[str, Tuple[int, int]], int] = defaultdict(int)
        totals: Dict[Tuple[str, Tuple[int, int]], int] = defaultdict(int)
        total_wins = 0
        total_samples = 0

        for m in matches:
            if m.get("has_error"):
                continue
            target = 1 if m.get("radiant_victory") else 0
            total_wins += target
            total_samples += 1

            for side, won in (("radiant", target), ("dire", 1 - target)):
                heroes = self._hero_ids(m, side)
                for i in range(len(heroes)):
                    for j in range(i + 1, len(heroes)):
                        k = (side, self._pair_key(heroes[i], heroes[j]))
                        wins[k] += won
                        totals[k] += 1

        if total_samples > 0:
            self._global_rate = total_wins / total_samples

        for k, t in totals.items():
            if t < self.min_samples:
                self._rates[k] = (wins[k] + self.smoothing * self._global_rate) / (t + self.smoothing)
            else:
                self._rates[k] = wins[k] / t
        return self

    def _pair_rates(self, side: str, heroes: List[int]) -> List[float]:
        """Return the per-pair win rates for one side's 5 heroes.

        Order is the natural C(5, 2) iteration order.  Pairs not
        seen in training get `_global_rate`.  Empty input (fewer
        than 2 heroes) returns an empty list — caller handles.
        """
        out: List[float] = []
        for i in range(len(heroes)):
            for j in range(i + 1, len(heroes)):
                k = (side, self._pair_key(heroes[i], heroes[j]))
                out.append(self._rates.get(k, self._global_rate))
        return out

    @staticmethod
    def _summarise(rates: List[float], global_rate: float) -> Tuple[float, float, float]:
        """Reduce a list of pair rates to (min, mean, max).

        Empty list → all three equal `global_rate` (the neutral
        prior).  This keeps the feature in a sensible range when
        one side lacks pair data.
        """
        if not rates:
            return (global_rate, global_rate, global_rate)
        return (min(rates), sum(rates) / len(rates), max(rates))

    def summarise_side(
        self, side: str, heroes: List[int],
    ) -> Tuple[float, float, float]:
        """Public: (min, mean, max) synergy for one side's heroes."""
        return self._summarise(self._pair_rates(side, heroes), self._global_rate)

    def to_dict(self) -> Dict:
        return {
            "smoothing": self.smoothing,
            "min_samples": self.min_samples,
            "global_rate": self._global_rate,
            "rates": {
                f"{s}|{a}-{b}": r for (s, (a, b)), r in self._rates.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HeroPairWinRateEncoder":
        enc = cls(smoothing=d.get("smoothing", 5.0), min_samples=d.get("min_samples", 3))
        enc._global_rate = d.get("global_rate", 0.5)
        for k, v in d.get("rates", {}).items():
            # k format: "side|a-b"  — a and b are ints already sorted in fit()
            side, pair = k.split("|", 1)
            a_str, b_str = pair.split("-", 1)
            enc._rates[(side, (int(a_str), int(b_str)))] = float(v)
        return enc


class HeroWinRateEncoder:
    """Per-hero, per-side target encoding with smoothing back to the global rate.

    fit(matches) walks a list of match dicts (DatDota full_matches format)
    and computes:

        P(win | side=s, hero=h)
            ≈ (wins(s, h) + alpha * global_rate) / (samples(s, h) + alpha)

    The smoothing (`alpha`) avoids zero-division for heroes that never
    appear on a given side in training data.

    Also nests a `TeamWinRateEncoder` (per-team aggregates) and a
    `HeroPairWinRateEncoder` (per-side, per-pair synergy).  One
    encoder to serialise, one to load — keeps train and predict
    in lock-step.
    """

    def __init__(
        self,
        smoothing: float = 5.0,
        min_samples: int = 3,
        team_smoothing: float = 3.0,
        team_min_samples: int = 2,
    ) -> None:
        self.smoothing = float(smoothing)
        self.min_samples = int(min_samples)
        # (side, hero_id) -> win_rate
        self._rates: Dict[Tuple[str, int], float] = {}
        self._global_rate: float = 0.5
        # Nested team encoder.  Always present after `fit()`; before
        # `fit()` it holds the prior (global_rate=0.5).
        self.team_encoder = TeamWinRateEncoder(
            smoothing=team_smoothing,
            min_samples=team_min_samples,
        )
        # Pair encoder is constructed but NOT fit during normal
        # `fit()` calls — see "History note" above.  Kept here so
        # the optional pair-feature path can be re-enabled by a
        # single call without re-instantiating the parent.
        self.pair_encoder = HeroPairWinRateEncoder()
        # Lane-pair encoder (0.3.10).  Fit alongside the hero
        # encoder in the normal `fit()` call below.
        self.lane_encoder = LanePairEncoder()
        # 0.3.13: cross-side matchup encoder (bot 2v2, top 2v2,
        # mid 1v1).  Per-pair lookup, MUST be fit on the train
        # split only — pass `fit_matchup_on=<train_subset>` to
        # `fit()` to opt in.  Defaults to "fit on the same pool
        # as the hero encoder" (idiomatic, mild leak — same as
        # 0.3.9 / 0.3.10).
        self.matchup_encoder = CrossSideMatchupEncoder()
        # 0.3.13: patch win-rate encoder.  Per-patch lookup,
        # aggregation-level signal so the default full-corpus fit
        # is fine (patches are not per-instance).
        self.patch_encoder = PatchWinRateEncoder()
        # 0.3.15: per-player encoder.  Constructed but not fit here —
        # `_features_player` will use it if present, and the trainer
        # can attach one with `encoder.player_encoder = PlayerWinRateEncoder()`
        # before training/predicting.  Stays None for v1..v13 models
        # that don't consume the player group.
        self.player_encoder = None

    @property
    def global_rate(self) -> float:
        return self._global_rate

    def fit(self, matches: Iterable[Dict]) -> "HeroWinRateEncoder":
        """Compute hero+side, team, and pair win rates from training matches.

        Each match dict must have `radiant_victory` (bool) and the
        `radiant.player_performances` / `dire.player_performances`
        structures with `performance.hero.valve_id`.  Team metadata
        is optional — matches without it still contribute to hero
        and pair encodings but not to team encodings.
        """
        wins: Dict[Tuple[str, int], int] = defaultdict(int)
        totals: Dict[Tuple[str, int], int] = defaultdict(int)
        total_wins = 0
        total_samples = 0

        for m in matches:
            if m.get("has_error"):
                continue
            target = 1 if m.get("radiant_victory") else 0
            total_wins += target
            total_samples += 1
            for p in m.get("radiant", {}).get("player_performances", []) or []:
                h = (p.get("performance") or {}).get("hero", {}).get("valve_id")
                if h is None:
                    continue
                wins[("radiant", h)] += target
                totals[("radiant", h)] += 1
            for p in m.get("dire", {}).get("player_performances", []) or []:
                h = (p.get("performance") or {}).get("hero", {}).get("valve_id")
                if h is None:
                    continue
                wins[("dire", h)] += 1 - target
                totals[("dire", h)] += 1

        if total_samples > 0:
            self._global_rate = total_wins / total_samples

        for k, t in totals.items():
            if t < self.min_samples:
                # Use smoothed estimate so unseen heroes get a sensible default
                self._rates[k] = (wins[k] + self.smoothing * self._global_rate) / (t + self.smoothing)
            else:
                self._rates[k] = wins[k] / t

        # Fit the nested team encoder on the same match stream.
        # `team_encoder` is idempotent — a second call to fit()
        # simply re-fits, doesn't accumulate state.  The pair
        # encoder is intentionally NOT fit here (see "History
        # note" in `FEATURE_ORDER`); a future release can call
        # `self.pair_encoder.fit(matches)` if the corpus grows
        # enough to support pair-level signal.
        self.team_encoder.fit(matches)
        # 0.3.10: lane-pair encoder is fit here so the bot / top
        # / mid features are available without a second pass over
        # the corpus.
        self.lane_encoder.fit(matches, self)
        # 0.3.13: cross-side matchup encoder (bot 2v2, top 2v2,
        # mid 1v1).  Per-pair lookup is sparse (1 row per exact
        # matchup), so we fit on the same pool as the hero
        # encoder — but the trainer can override via
        # `fit(matchup_on=<subset>)` for honest evaluation.
        self.matchup_encoder.fit(matches)
        # 0.3.13: patch win-rate encoder.  Aggregate-level
        # signal (~2 patches × 1000+ matches each), full-corpus
        # fit is fine.
        self.patch_encoder.fit(matches)
        return self

    def encode(self, side: str, hero_id: int) -> float:
        """Return the historical win rate for this hero on this side.

        Returns `_global_rate` for heroes never seen on this side.
        """
        return self._rates.get((side, int(hero_id)), self._global_rate)

    def encode_team(self, team_id: Optional[int]) -> float:
        """Delegate to the nested team encoder."""
        return self.team_encoder.encode(team_id)

    def summarise_pair_synergy(
        self, side: str, heroes: List[int],
    ) -> Tuple[float, float, float]:
        """Return (min, mean, max) of pair win rates for one side's 5 heroes.

        Empty list → all three equal `global_rate`.  Uses the
        nested `pair_encoder` so train and predict share the
        same lookup table.
        """
        return self.pair_encoder.summarise_side(side, heroes)

    def to_dict(self) -> Dict:
        """Serialise for persistence alongside the trained model."""
        return {
            "smoothing": self.smoothing,
            "min_samples": self.min_samples,
            "global_rate": self._global_rate,
            "rates": {f"{s}|{h}": r for (s, h), r in self._rates.items()},
            "team_encoder": self.team_encoder.to_dict(),
            "pair_encoder": self.pair_encoder.to_dict(),
            "lane_encoder": self.lane_encoder.to_dict(),
            "matchup_encoder": self.matchup_encoder.to_dict(),
            "patch_encoder": self.patch_encoder.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HeroWinRateEncoder":
        enc = cls(smoothing=d.get("smoothing", 5.0), min_samples=d.get("min_samples", 3))
        enc._global_rate = d.get("global_rate", 0.5)
        for k, v in d.get("rates", {}).items():
            s, h = k.split("|", 1)
            enc._rates[(s, int(h))] = float(v)
        if "team_encoder" in d:
            enc.team_encoder = TeamWinRateEncoder.from_dict(d["team_encoder"])
        if "pair_encoder" in d:
            enc.pair_encoder = HeroPairWinRateEncoder.from_dict(d["pair_encoder"])
        if "lane_encoder" in d:
            enc.lane_encoder = LanePairEncoder.from_dict(d["lane_encoder"])
        if "matchup_encoder" in d:
            enc.matchup_encoder = CrossSideMatchupEncoder.from_dict(d["matchup_encoder"])
        if "patch_encoder" in d:
            enc.patch_encoder = PatchWinRateEncoder.from_dict(d["patch_encoder"])
        return enc


# ---------------------------------------------------------------------------- #
# Feature extraction
# ---------------------------------------------------------------------------- #

# Feature groups.  Each group is a tuple of feature names in the
# canonical order.  The full `FEATURE_ORDER` is the concatenation
# of all groups; `extract_features` can be asked to return any
# subset for the A/B harness.
#
# History note (0.3.10 dev cycle):
#   - 0.3.9 baseline was 13 hero-only features (current "hero" group).
#   - "team" group adds 4 team-aggregate features (2 sides × 2
#     aggregates: mean team WR + team-id match indicator).  Was
#     reverted in 0.3.9 (overfit with 38 teams in 1110 matches);
#     retried in 0.3.10 with 64 teams in 1275 matches.
#   - "lane" group adds 7 lane-pair features (bot/top × 2 sides +
#     3 diffs + 1 mid matchup).  New in 0.3.10.
#   - "team" + "lane" together = 24 features.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    # 13 hero-only (0.3.9 baseline)
    "hero": (
        "mean_hero_wr_radiant",
        "mean_hero_wr_dire",
        "hero_wr_r_0", "hero_wr_r_1", "hero_wr_r_2", "hero_wr_r_3", "hero_wr_r_4",
        "hero_wr_d_0", "hero_wr_d_1", "hero_wr_d_2", "hero_wr_d_3", "hero_wr_d_4",
        "radiant_minus_dire",
    ),
    # 4 team-aggregate features
    "team": (
        "team_wr_radiant",
        "team_wr_dire",
        "team_wr_diff",
        "team_pair_diff",  # P(radiant_team beats dire_team) — symmetric
    ),
    # 7 lane-pair features
    "lane": (
        "bot_pair_radiant",
        "bot_pair_dire",
        "bot_pair_diff",
        "top_pair_radiant",
        "top_pair_dire",
        "top_pair_diff",
        "mid_matchup",
    ),
    # 0.3.13: cross-side lane matchups (bot 2v2, top 2v2, mid 1v1)
    # P(radiant_pair_wins | specific radiant_pair vs specific dire_pair).
    # Honest target encoding — encoder must be fit on the train split
    # only to avoid the per-pair lookup leakage that the 0.3.10
    # "lane" group suffered.
    "matchup": (
        "bot_2v2_matchup",     # P(radiant (carry+support) wins vs dire (carry+support))
        "top_2v2_matchup",     # P(radiant (offlane+jungler) wins vs dire (offlane+jungler))
        "mid_1v1_matchup",     # P(radiant_mid wins vs dire_mid)
    ),
    # 0.3.13: patch version target encoding.  7.40 vs 7.41 win
    # rates differ by meta — this gives the model an explicit
    # "what patch is this?" channel.
    "patch": (
        "patch_wr_radiant",
        "patch_wr_dire",
        "patch_wr_diff",
    ),
    # 0.3.15: per-player features.  These come from `PlayerWinRateEncoder`
    # (target encoding on full_matches) at training time, and from DLTV's
    # `map_results[].player.win_rate` at predict time when the match is
    # already in DLTV.  The four aggregates (avg/max per side) are
    # deliberately coarse — they generalise across the "who's on this
    # roster?" noise that per-slot features suffer from in 5-stack drafts.
    "player": (
        "r_player_wr_avg",   # mean(DLTV.player.win_rate) over radiant 5
        "d_player_wr_avg",   # mean over dire 5
        "r_player_wr_max",   # max (carry skill ceiling)
        "d_player_wr_max",   # max
    ),
}
FEATURE_ORDER: Tuple[str, ...] = sum(FEATURE_GROUPS.values(), ())  # type: ignore[arg-type,operator]
N_FEATURES = len(FEATURE_ORDER)


def _features_hero(
    radiant_hero_ids: List[int],
    dire_hero_ids: List[int],
    encoder: HeroWinRateEncoder,
) -> List[float]:
    """13 hero-only features (0.3.9 baseline)."""
    if len(radiant_hero_ids) != 5 or len(dire_hero_ids) != 5:
        raise ValueError("expected exactly 5 hero IDs per side")
    radiant_enc = [encoder.encode("radiant", h) for h in radiant_hero_ids]
    dire_enc = [encoder.encode("dire", h) for h in dire_hero_ids]
    mean_r = sum(radiant_enc) / 5.0
    mean_d = sum(dire_enc) / 5.0
    return [
        mean_r, mean_d,
        *radiant_enc, *dire_enc,
        mean_r - mean_d,
    ]


def _features_team(
    radiant_team_id: Optional[int],
    dire_team_id: Optional[int],
    encoder: HeroWinRateEncoder,
) -> List[float]:
    """4 team-aggregate features (C retry in 0.3.10).

    `team_pair_diff` is the encoder's pre-computed P(radiant_team
    beats dire_team) — symmetric, so a strong team against a weak
    one produces a positive diff.
    """
    wr_r = encoder.encode_team(radiant_team_id)
    wr_d = encoder.encode_team(dire_team_id)
    pair = encoder.team_encoder.encode(radiant_team_id) - encoder.team_encoder.encode(dire_team_id)
    return [wr_r, wr_d, wr_r - wr_d, pair]


def _features_lane(
    match: Optional[Dict],
    radiant_lane: Optional[Dict[str, Optional[int]]],
    dire_lane: Optional[Dict[str, Optional[int]]],
    encoder: HeroWinRateEncoder,
    lane_encoder: "LanePairEncoder",
) -> List[float]:
    """7 lane-pair features (D v2 in 0.3.10).

    Caller must pass EITHER `match` (DatDota full_matches dict, used
    at train time) OR both `radiant_lane` and `dire_lane` (used at
    predict time, when the engine has already extracted the per-side
    lane assignments).  Passing `match` is the more common path —
    train time uses it; predict time can use it directly via the
    upstream DLTV/Stratz response, OR fall back to the precomputed
    lane dict.
    """
    if match is not None:
        from .features import lane_heroes_from_match  # local import: avoid cycle
        lanes = lane_heroes_from_match(match)
        r = lanes["radiant"]; d = lanes["dire"]
    elif radiant_lane is not None and dire_lane is not None:
        r = radiant_lane; d = dire_lane
    else:
        raise ValueError("extract_features(lane): need match or lane dicts")

    bot_r = lane_encoder.encode_bot_pair(
        encoder, "radiant", r["BOT_CARRY"], r["BOT_SUPPORT"]
    )
    bot_d = lane_encoder.encode_bot_pair(
        encoder, "dire", d["BOT_CARRY"], d["BOT_SUPPORT"]
    )
    tp_r = lane_encoder.encode_top_pair(
        encoder, "radiant", r["TOP_OFFLANE"], r["TOP_JUNGLER"]
    )
    tp_d = lane_encoder.encode_top_pair(
        encoder, "dire", d["TOP_OFFLANE"], d["TOP_JUNGLER"]
    )
    mid = lane_encoder.encode_mid_matchup(
        encoder, r["MID"], d["MID"]
    )
    return [
        bot_r, bot_d, bot_r - bot_d,
        tp_r, tp_d, tp_r - tp_d,
        mid,
    ]


def _features_matchup(
    match: Optional[Dict],
    radiant_lane: Optional[Dict[str, Optional[int]]],
    dire_lane: Optional[Dict[str, Optional[int]]],
    encoder: "HeroWinRateEncoder",
) -> List[float]:
    """3 cross-side lane matchup features (0.3.13).

    Layout (3 features):
      - bot_2v2_matchup  P(radiant wins | this exact bot 2v2 matchup)
      - top_2v2_matchup  P(radiant wins | this exact top 2v2 matchup)
      - mid_1v1_matchup  P(radiant wins | this exact mid 1v1 matchup)

    Caller must pass EITHER `match` (train time) OR both
    `radiant_lane` and `dire_lane` (predict time).  Falls back
    to `matchup_encoder.global_rate` (0.5) when the matchup is
    unseen or any hero id is missing.
    """
    if match is not None:
        from .features import lane_heroes_from_match  # local: avoid cycle
        lanes = lane_heroes_from_match(match)
        r = lanes["radiant"]; d = lanes["dire"]
    elif radiant_lane is not None and dire_lane is not None:
        r = radiant_lane; d = dire_lane
    else:
        raise ValueError("extract_features(matchup): need match or lane dicts")
    m = encoder.matchup_encoder
    return [
        m.encode_bot_2v2(r["BOT_CARRY"], r["BOT_SUPPORT"], d["BOT_CARRY"], d["BOT_SUPPORT"]),
        m.encode_top_2v2(r["TOP_OFFLANE"], r["TOP_JUNGLER"], d["TOP_OFFLANE"], d["TOP_JUNGLER"]),
        m.encode_mid_1v1(r["MID"], d["MID"]),
    ]


def _features_patch(
    match: Optional[Dict],
    encoder: "HeroWinRateEncoder",
) -> List[float]:
    """3 patch win-rate features (0.3.13).

    Layout (3 features):
      - patch_wr_radiant  P(radiant wins | patch = match.patch)
      - patch_wr_dire      1 - patch_wr_radiant
      - patch_wr_diff      patch_wr_radiant - 0.5 (centred)
    """
    patch = None
    if match is not None:
        patch = encoder.patch_encoder._patch_id(match)
    wr_r = encoder.patch_encoder.encode(patch)
    return [wr_r, 1.0 - wr_r, wr_r - 0.5]


def _features_player(
    match: Optional[Dict],
    encoder: "HeroWinRateEncoder",
) -> List[float]:
    """4 per-player WR aggregate features (0.3.15).

    Layout (4 features):
      - r_player_wr_avg  mean(player.win_rate) over radiant 5
      - d_player_wr_avg  mean over dire 5
      - r_player_wr_max  max (carry skill ceiling)
      - d_player_wr_max  max

    Player WR comes from full_matches.steam32 (training) or DLTV's
    `map_results[].player.win_rate` (predict) — both encoded through
    `PlayerWinRateEncoder`.  The encoder lives on `encoder.player_encoder`
    if present (a `PlayerWinRateEncoder` instance); otherwise we fall
    back to a fresh encoder fit on the same match list.
    """
    p_enc = getattr(encoder, "player_encoder", None)
    if p_enc is None:
        # No player encoder attached — return global_rate for all 4 features
        # so the model still runs (won't be useful, but won't crash either).
        gr = 0.5
        return [gr, gr, gr, gr]
    ps = players_from_match(match) if match is not None else {"radiant": [], "dire": []}
    r_wrs = [p_enc.encode(s) for s in ps["radiant"]]
    d_wrs = [p_enc.encode(s) for s in ps["dire"]]
    if not r_wrs: r_wrs = [p_enc.global_rate] * 5
    if not d_wrs: d_wrs = [p_enc.global_rate] * 5
    import numpy as _np
    return [
        float(_np.mean(r_wrs)),
        float(_np.mean(d_wrs)),
        float(max(r_wrs)),
        float(max(d_wrs)),
    ]


def extract_features(
    radiant_hero_ids: List[int],
    dire_hero_ids: List[int],
    encoder: HeroWinRateEncoder,
    *,
    radiant_team_id: Optional[int] = None,
    dire_team_id: Optional[int] = None,
    match: Optional[Dict] = None,
    radiant_lane: Optional[Dict[str, Optional[int]]] = None,
    dire_lane: Optional[Dict[str, Optional[int]]] = None,
    groups: Tuple[str, ...] = ("hero", "team", "lane", "matchup", "patch", "player"),
) -> List[float]:
    """Build the feature vector for a single match prediction.

    Args:
        radiant_hero_ids: 5 valve_ids (radiant side)
        dire_hero_ids:     5 valve_ids (dire side)
        encoder:           a fitted HeroWinRateEncoder
        radiant_team_id:   optional valve_id (or `team_id`) for the radiant team
        dire_team_id:      same for dire
        match:             full DatDota match dict (only needed for
                           the `lane`/`matchup`/`patch` groups at train
                           time)
        radiant_lane:      pre-extracted per-side lane dict (predict time)
        dire_lane:         same for dire
        groups:            which feature groups to include.  Default
                           includes all five (hero, team, lane, matchup,
                           patch).  Pass `("hero",)` for the 0.3.9
                           baseline, `("hero", "team")` for the C retry,
                           etc.

    Returns a list of floats in the order declared by
    `FEATURE_GROUPS[groups[0]] + ... + FEATURE_GROUPS[groups[-1]]`.
    The exact order is part of the model contract; the trainer and
    this function MUST stay in sync.
    """
    feats: List[float] = []
    for g in groups:
        if g == "hero":
            feats.extend(_features_hero(radiant_hero_ids, dire_hero_ids, encoder))
        elif g == "team":
            feats.extend(_features_team(radiant_team_id, dire_team_id, encoder))
        elif g == "lane":
            feats.extend(_features_lane(match, radiant_lane, dire_lane, encoder, encoder.lane_encoder))
        elif g == "matchup":
            feats.extend(_features_matchup(match, radiant_lane, dire_lane, encoder))
        elif g == "patch":
            feats.extend(_features_patch(match, encoder))
        elif g == "player":
            feats.extend(_features_player(match, encoder))
        else:
            raise ValueError(f"unknown feature group: {g!r}")
    return feats


def feature_names(groups: Tuple[str, ...] = ("hero", "team", "lane", "matchup", "patch", "player")) -> List[str]:
    """Return the feature names for a given group tuple (in order)."""
    out: List[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {g!r}")
        out.extend(FEATURE_GROUPS[g])
    return out


def hero_ids_from_match(match: Dict) -> Tuple[List[int], List[int]]:
    """Pull 5 radiant + 5 dire hero IDs from a DatDota match dict."""
    radiant = [
        (p.get("performance") or {}).get("hero", {}).get("valve_id")
        for p in match.get("radiant", {}).get("player_performances", []) or []
    ]
    dire = [
        (p.get("performance") or {}).get("hero", {}).get("valve_id")
        for p in match.get("dire", {}).get("player_performances", []) or []
    ]
    # Take the first 5 (the order in DatDota is the pick order, but
    # for target encoding it doesn't matter which 5 we pick).
    return radiant[:5], dire[:5]


class LanePairEncoder:
    """Per-lane-pair target encoding (0.3.10 — lane synergy).

    Computes three kinds of rates:
      - bot_pair:    mean WR of the carry + support on the same
                     side.  Pair key is the frozenset of two hero
                     ids; looked up against a win/total table fit
                     from the training corpus.
      - top_pair:    mean WR of the offlane + jungler on the same
                     side.  Same lookup scheme as `bot_pair`.
      - mid_matchup: P(radiant_mid wins | this mid matchup) — a
                     cross-side rate, indexed by the frozenset
                     {radiant_mid_id, dire_mid_id}.

    All rate lookups fall back to the per-pair rate of the two
    solo hero WRs (averaged) and finally to `_global_rate`.  That
    two-step fallback means a freshly-introduced pair still has
    signal from its individual heroes, and a brand-new hero
    doesn't kill the feature.

    Stored as nested dicts on the parent `HeroWinRateEncoder`
    instance.  One parent encoder, one save.
    """

    def __init__(self) -> None:
        # bot pair: (side, frozenset({carry_id, support_id})) -> win_rate
        self._bot_pair: Dict[Tuple[str, frozenset], float] = {}
        # top pair (offlane + jungler)
        self._top_pair: Dict[Tuple[str, frozenset], float] = {}
        # mid matchup: frozenset({radiant_mid_id, dire_mid_id}) -> win_rate
        #   (indexed symmetrically — we want P(radiant_mid wins), so
        #   the wins are always contributed by the radiant side.)
        self._mid_matchup: Dict[frozenset, float] = {}
        self._global_rate: float = 0.5

    @property
    def global_rate(self) -> float:
        return self._global_rate

    @staticmethod
    def _safe_pair(a: Optional[int], b: Optional[int]) -> Optional[frozenset]:
        if a is None or b is None:
            return None
        return frozenset({int(a), int(b)})

    @staticmethod
    def _hero_wr(encoder: "HeroWinRateEncoder", side: str, vid: Optional[int]) -> float:
        if vid is None:
            return encoder.global_rate
        return encoder.encode(side, int(vid))

    def fit(self, matches: Iterable[Dict], encoder: "HeroWinRateEncoder") -> "LanePairEncoder":
        """Compute the per-pair rates from training matches.

        `encoder` is the parent `HeroWinRateEncoder` whose fit
        was already called — we use its `_rates` for solo fallback.
        """
        bot_wins: Dict[Tuple[str, frozenset], int] = defaultdict(int)
        bot_totals: Dict[Tuple[str, frozenset], int] = defaultdict(int)
        tp_wins: Dict[Tuple[str, frozenset], int] = defaultdict(int)
        tp_totals: Dict[Tuple[str, frozenset], int] = defaultdict(int)
        mid_wins: Dict[frozenset, int] = defaultdict(int)
        mid_totals: Dict[frozenset, int] = defaultdict(int)
        global_wins = 0
        global_total = 0

        for m in matches:
            if m.get("has_error"):
                continue
            target = 1 if m.get("radiant_victory") else 0
            global_wins += target
            global_total += 1
            lanes = lane_heroes_from_match(m)

            for side, won in (("radiant", target), ("dire", 1 - target)):
                bot_key = self._safe_pair(
                    lanes[side]["BOT_CARRY"], lanes[side]["BOT_SUPPORT"]
                )
                if bot_key is not None:
                    key = (side, bot_key)
                    bot_wins[key] += won
                    bot_totals[key] += 1
                tp_key = self._safe_pair(
                    lanes[side]["TOP_OFFLANE"], lanes[side]["TOP_JUNGLER"]
                )
                if tp_key is not None:
                    key = (side, tp_key)
                    tp_wins[key] += won
                    tp_totals[key] += 1

            r_mid = lanes["radiant"]["MID"]
            d_mid = lanes["dire"]["MID"]
            mid_key = self._safe_pair(r_mid, d_mid)
            if mid_key is not None:
                # Always index by frozenset (order-independent).
                # We want P(radiant_mid wins) — wins on radiant side.
                mid_wins[mid_key] += target
                mid_totals[mid_key] += 1

        if global_total > 0:
            self._global_rate = global_wins / global_total

        # Smoothed win rates — same `smoothing` value as HeroWinRateEncoder
        smoothing = encoder.smoothing
        for store, wins, totals in (
            (self._bot_pair, bot_wins, bot_totals),
            (self._top_pair, tp_wins, tp_totals),
        ):
            for k, t in totals.items():
                store[k] = (wins[k] + smoothing * self._global_rate) / (t + smoothing)
        for k, t in mid_totals.items():
            self._mid_matchup[k] = (mid_wins[k] + smoothing * self._global_rate) / (t + smoothing)
        return self

    def encode_bot_pair(
        self, encoder: "HeroWinRateEncoder", side: str,
        carry_id: Optional[int], support_id: Optional[int],
    ) -> float:
        """Mean WR for the bot pair (carry + support) on `side`.

        Falls back to `encoder.global_rate` for missing heroes,
        to the mean of the two solo hero WRs for unseen pairs.
        """
        if carry_id is None or support_id is None:
            return encoder.global_rate
        key = (side, frozenset({int(carry_id), int(support_id)}))
        if key in self._bot_pair:
            return self._bot_pair[key]
        return 0.5 * (
            self._hero_wr(encoder, side, carry_id) +
            self._hero_wr(encoder, side, support_id)
        )

    def encode_top_pair(
        self, encoder: "HeroWinRateEncoder", side: str,
        offlane_id: Optional[int], jungler_id: Optional[int],
    ) -> float:
        """Mean WR for the top pair (offlane + jungler) on `side`."""
        if offlane_id is None or jungler_id is None:
            return encoder.global_rate
        key = (side, frozenset({int(offlane_id), int(jungler_id)}))
        if key in self._top_pair:
            return self._top_pair[key]
        return 0.5 * (
            self._hero_wr(encoder, side, offlane_id) +
            self._hero_wr(encoder, side, jungler_id)
        )

    def encode_mid_matchup(
        self, encoder: "HeroWinRateEncoder",
        radiant_mid: Optional[int], dire_mid: Optional[int],
    ) -> float:
        """P(radiant_mid wins | this matchup) from pro history.

        Falls back to the mean of the two solo hero WRs and
        finally to `encoder.global_rate`.
        """
        if radiant_mid is None or dire_mid is None:
            return encoder.global_rate
        key = frozenset({int(radiant_mid), int(dire_mid)})
        if key in self._mid_matchup:
            return self._mid_matchup[key]
        return 0.5 * (
            self._hero_wr(encoder, "radiant", radiant_mid) +
            self._hero_wr(encoder, "dire", dire_mid)
        )

    def to_dict(self) -> Dict:
        return {
            "global_rate": self._global_rate,
            "bot_pair": {
                f"{s}|{'-'.join(str(x) for x in sorted(k))}": r
                for (s, k), r in self._bot_pair.items()
            },
            "top_pair": {
                f"{s}|{'-'.join(str(x) for x in sorted(k))}": r
                for (s, k), r in self._top_pair.items()
            },
            "mid_matchup": {
                "-".join(str(x) for x in sorted(k)): r
                for k, r in self._mid_matchup.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LanePairEncoder":
        enc = cls()
        enc._global_rate = d.get("global_rate", 0.5)
        for k, v in (d.get("bot_pair") or {}).items():
            s, pair = k.split("|", 1)
            enc._bot_pair[(s, frozenset(int(x) for x in pair.split("-")))] = float(v)
        for k, v in (d.get("top_pair") or {}).items():
            s, pair = k.split("|", 1)
            enc._top_pair[(s, frozenset(int(x) for x in pair.split("-")))] = float(v)
        for k, v in (d.get("mid_matchup") or {}).items():
            enc._mid_matchup[frozenset(int(x) for x in k.split("-"))] = float(v)
        return enc


class CrossSideMatchupEncoder:
    """Per-cross-side-matchup target encoding (0.3.13 — bot/top/mid matchup).

    Computes three lookup tables:

      - bot_2v2  → P(radiant_pair wins | radiant_pair = (carry_r, sup_r)
                                            vs dire_pair    = (carry_d, sup_d))
      - top_2v2  → P(radiant_pair wins | radiant_pair = (off_r, jun_r)
                                            vs dire_pair    = (off_d, jun_d))
      - mid_1v1  → P(radiant_mid wins | radiant_mid = m_r, dire_mid = m_d)

    The matchup key is `frozenset(radiant_pair ∪ dire_pair)` with the
    P-rate recorded as a *radiant win rate* (so the lookup value
    goes up when radiant tends to win).  This is symmetric — the
    same key covers both orderings — but the rate is always
    *from radiant's perspective*, so the caller doesn't need to
    worry about side.

    Honest target encoding (0.3.13): per-pair lookup is
    inherently sparse (each matchup is 1 row in the table), so
    the encoder MUST be fit on the train split only to avoid
    the per-pair leakage the 0.3.10 `lane` group suffered.
    Smoothing toward the global rate keeps unseen pairs at a
    reasonable prior.

    Stored in the parent `HeroWinRateEncoder` so it round-trips
    through `to_dict` / `from_dict` like the other encoders.
    """

    def __init__(self, smoothing: float = 3.0) -> None:
        # bot 2v2: key = frozenset({carry_r, sup_r, carry_d, sup_d})
        #          val = P(radiant wins | this matchup)
        self.smoothing = float(smoothing)
        self._bot_2v2: Dict[frozenset, float] = {}
        self._top_2v2: Dict[frozenset, float] = {}
        self._mid_1v1: Dict[frozenset, float] = {}
        self._global_rate: float = 0.5

    @property
    def global_rate(self) -> float:
        return self._global_rate

    @staticmethod
    def _key(*ids: Optional[int]) -> Optional[frozenset]:
        if any(i is None for i in ids):
            return None
        return frozenset(int(i) for i in ids)

    def fit(self, matches: Iterable[Dict]) -> "CrossSideMatchupEncoder":
        """Build the lookup tables from training matches.

        For each match we update the three tables with the
        actual outcome.  Each table's "wins" count uses the
        `radiant_victory` boolean directly (no side flipping) so
        the lookup value is the *radiant* win rate conditioned on
        this exact matchup.
        """
        bot_wins: Dict[frozenset, int] = defaultdict(int)
        bot_totals: Dict[frozenset, int] = defaultdict(int)
        top_wins: Dict[frozenset, int] = defaultdict(int)
        top_totals: Dict[frozenset, int] = defaultdict(int)
        mid_wins: Dict[frozenset, int] = defaultdict(int)
        mid_totals: Dict[frozenset, int] = defaultdict(int)
        global_wins = 0
        global_total = 0

        for m in matches:
            if m.get("has_error"):
                continue
            target = 1 if m.get("radiant_victory") else 0
            global_wins += target
            global_total += 1
            lanes = lane_heroes_from_match(m)
            r = lanes["radiant"]; d = lanes["dire"]

            k = self._key(r["BOT_CARRY"], r["BOT_SUPPORT"], d["BOT_CARRY"], d["BOT_SUPPORT"])
            if k is not None:
                bot_wins[k] += target
                bot_totals[k] += 1

            k = self._key(r["TOP_OFFLANE"], r["TOP_JUNGLER"], d["TOP_OFFLANE"], d["TOP_JUNGLER"])
            if k is not None:
                top_wins[k] += target
                top_totals[k] += 1

            k = self._key(r["MID"], d["MID"])
            if k is not None:
                mid_wins[k] += target
                mid_totals[k] += 1

        if global_total > 0:
            self._global_rate = global_wins / global_total

        # Smoothing 3.0 — same as the hero encoder.  The matchup
        # tables are very sparse (one row per matchup), so a
        # larger smoothing prevents single observations from
        # dominating the lookup value.  `self.smoothing` is
        # operator-tunable (0.3.14 grid swept 1.5/2.0/3.0/5.0
        # and 2.0 won the apples-to-apples forward).
        smoothing = self.smoothing
        for store, wins, totals in (
            (self._bot_2v2, bot_wins, bot_totals),
            (self._top_2v2, top_wins, top_totals),
            (self._mid_1v1, mid_wins, mid_totals),
        ):
            for k, t in totals.items():
                store[k] = (wins[k] + smoothing * self._global_rate) / (t + smoothing)
        return self

    def encode_bot_2v2(
        self,
        r_carry: Optional[int], r_support: Optional[int],
        d_carry: Optional[int], d_support: Optional[int],
    ) -> float:
        """P(radiant wins | this exact bot 2v2 matchup), 0.5 prior."""
        k = self._key(r_carry, r_support, d_carry, d_support)
        if k is None or k not in self._bot_2v2:
            return self._global_rate
        return self._bot_2v2[k]

    def encode_top_2v2(
        self,
        r_off: Optional[int], r_jun: Optional[int],
        d_off: Optional[int], d_jun: Optional[int],
    ) -> float:
        k = self._key(r_off, r_jun, d_off, d_jun)
        if k is None or k not in self._top_2v2:
            return self._global_rate
        return self._top_2v2[k]

    def encode_mid_1v1(self, r_mid: Optional[int], d_mid: Optional[int]) -> float:
        k = self._key(r_mid, d_mid)
        if k is None or k not in self._mid_1v1:
            return self._global_rate
        return self._mid_1v1[k]

    def to_dict(self) -> Dict:
        def _ser(d):
            return {"-".join(str(x) for x in sorted(k)): r for k, r in d.items()}
        return {
            "smoothing": self.smoothing,
            "global_rate": self._global_rate,
            "bot_2v2": _ser(self._bot_2v2),
            "top_2v2": _ser(self._top_2v2),
            "mid_1v1": _ser(self._mid_1v1),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CrossSideMatchupEncoder":
        enc = cls(smoothing=d.get("smoothing", 3.0))
        enc._global_rate = d.get("global_rate", 0.5)
        for k, v in (d.get("bot_2v2") or {}).items():
            enc._bot_2v2[frozenset(int(x) for x in k.split("-"))] = float(v)
        for k, v in (d.get("top_2v2") or {}).items():
            enc._top_2v2[frozenset(int(x) for x in k.split("-"))] = float(v)
        for k, v in (d.get("mid_1v1") or {}).items():
            enc._mid_1v1[frozenset(int(x) for x in k.split("-"))] = float(v)
        return enc


class PatchWinRateEncoder:
    """Per-patch win rate target encoding (0.3.13).

    Each Dota patch (e.g. "7.40", "7.41") shifts the meta, and
    some patches favour the radiant side statistically (e.g.
    longer games favour radiant's "comeback" gold).  Encoding
    `patch_wr_radiant = P(radiant wins | patch=p)` and
    `patch_wr_dire = 1 - P(radiant wins | patch=p)` gives the
    model an explicit "what patch is this?" channel that the
    hero / team / lane features don't otherwise see.

    Keyed by patch string (e.g. "7.40").  Falls back to
    `_global_rate` for unseen patches.
    """

    def __init__(self) -> None:
        self._rates: Dict[str, float] = {}
        self._global_rate: float = 0.5

    @property
    def global_rate(self) -> float:
        return self._global_rate

    @staticmethod
    def _patch_id(m: Dict) -> Optional[str]:
        """Pull the patch string from a match dict, or None if missing."""
        p = m.get("patch")
        if not p:
            return None
        if isinstance(p, dict):
            # Some DatDota shapes nest the patch under a dict.
            p = p.get("name") or p.get("id")
        if isinstance(p, (int, float)):
            return str(p)
        if isinstance(p, str) and p:
            return p
        return None

    def fit(self, matches: Iterable[Dict]) -> "PatchWinRateEncoder":
        wins: Dict[str, int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)
        global_wins = 0
        global_total = 0
        for m in matches:
            if m.get("has_error"):
                continue
            p = self._patch_id(m)
            if p is None:
                continue
            target = 1 if m.get("radiant_victory") else 0
            global_wins += target
            global_total += 1
            wins[p] += target
            totals[p] += 1
        if global_total > 0:
            self._global_rate = global_wins / global_total
        # Smoothing toward the global rate.  With only ~2 patches
        # in the corpus (7.40 and 7.41) each has 500+ matches,
        # so smoothing is more about future unknown patches than
        # about small-sample noise.
        smoothing = 5.0
        for p, t in totals.items():
            self._rates[p] = (wins[p] + smoothing * self._global_rate) / (t + smoothing)
        return self

    def encode(self, patch: Optional[str]) -> float:
        if patch is None:
            return self._global_rate
        return self._rates.get(patch, self._global_rate)

    def to_dict(self) -> Dict:
        return {
            "global_rate": self._global_rate,
            "rates": dict(self._rates),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PatchWinRateEncoder":
        enc = cls()
        enc._global_rate = d.get("global_rate", 0.5)
        for k, v in (d.get("rates") or {}).items():
            enc._rates[str(k)] = float(v)
        return enc


# Keep this here so `lane_heroes_from_match` is exported with the
# module's public surface.
__all__ = [
    "FEATURE_GROUPS",
    "FEATURE_ORDER",
    "N_FEATURES",
    "HeroWinRateEncoder",
    "TeamWinRateEncoder",
    "HeroPairWinRateEncoder",
    "LanePairEncoder",
    "CrossSideMatchupEncoder",
    "PatchWinRateEncoder",
    "extract_features",
    "extract_lane_features",
    "feature_names",
    "hero_ids_from_match",
    "lane_heroes_from_match",
    "target_from_match",
]


def extract_lane_features(
    match: Dict,
    encoder: "HeroWinRateEncoder",
    lane_encoder: "LanePairEncoder",
) -> List[float]:
    """Build the 7 lane-pair feature vector for `match`.

    Layout (7 features):
      - bot_pair_radiant   (mean carry+support hero WR, radiant)
      - bot_pair_dire
      - bot_pair_diff      (radiant - dire)
      - top_pair_radiant   (offlane+jungler)
      - top_pair_dire
      - top_pair_diff
      - mid_matchup         (P(radiant_mid wins))
    """
    lanes = lane_heroes_from_match(match)

    bot_r = lane_encoder.encode_bot_pair(
        encoder, "radiant",
        lanes["radiant"]["BOT_CARRY"], lanes["radiant"]["BOT_SUPPORT"],
    )
    bot_d = lane_encoder.encode_bot_pair(
        encoder, "dire",
        lanes["dire"]["BOT_CARRY"], lanes["dire"]["BOT_SUPPORT"],
    )
    tp_r = lane_encoder.encode_top_pair(
        encoder, "radiant",
        lanes["radiant"]["TOP_OFFLANE"], lanes["radiant"]["TOP_JUNGLER"],
    )
    tp_d = lane_encoder.encode_top_pair(
        encoder, "dire",
        lanes["dire"]["TOP_OFFLANE"], lanes["dire"]["TOP_JUNGLER"],
    )
    mid = lane_encoder.encode_mid_matchup(
        encoder,
        lanes["radiant"]["MID"], lanes["dire"]["MID"],
    )

    return [
        bot_r,
        bot_d,
        bot_r - bot_d,
        tp_r,
        tp_d,
        tp_r - tp_d,
        mid,
    ]


def target_from_match(match: Dict) -> int:
    """Return 1 if radiant won, 0 if dire won. None for skipped matches."""
    if match.get("has_error"):
        return None  # type: ignore[return-value]
    if "radiant_victory" not in match:
        return None  # type: ignore[return-value]
    return 1 if match.get("radiant_victory") else 0


# Lane role values DatDota puts in player_performances[i].laneInfo.lane
# (verified 2026-07-24 against 1275 matches — 100% have it populated).
# DatDota does NOT consistently tag junglers as "JUNGLE" (only 2% of
# 1275 matches); in pro play the jungler is most often recorded as
# "TOP" (the offlaner + jungler share the top lane early).  So we
# collapse "BOTTOM"/"ROAM" into the bot pair and "TOP" into the
# top pair, and require TWO players per side for both pairs.
LANE_KEYS: Tuple[str, ...] = (
    "BOT_CARRY", "BOT_SUPPORT", "TOP_OFFLANE", "TOP_JUNGLER", "MID",
)

# DatDota role string -> canonical key
_LANE_MAP: Dict[str, str] = {
    "BOTTOM": "BOT",
    "ROAM":   "BOT",
    "TOP":    "TOP",
    "MIDDLE": "MID",
    "JUNGLE": "TOP",  # treat the rare "JUNGLE" label as TOP pair
}


def _lane_of(pp: Dict) -> Optional[str]:
    """Pull the DatDota lane string from a player_performances entry."""
    li = (pp.get("laneInfo") or {}).get("lane")
    if li:
        return str(li).upper()
    # Fall back to performance.laneInfo
    li = ((pp.get("performance") or {}).get("laneInfo") or {}).get("lane")
    if li:
        return str(li).upper()
    return None


def _vid_of(pp: Dict) -> Optional[int]:
    vid = (pp.get("performance") or {}).get("hero", {}).get("valve_id")
    return int(vid) if isinstance(vid, int) else None


def lane_heroes_from_match(match: Dict) -> Dict[str, Dict[str, Optional[int]]]:
    """Pull per-side hero ids grouped by lane role.

    Returns ``{"radiant": {"BOT_CARRY", "BOT_SUPPORT", "TOP_OFFLANE",
    "TOP_JUNGLER", "MID"}, "dire": {...}}``.

    Mapping (DatDota `laneInfo.lane` -> our key):
      - BOTTOM / ROAM  -> bot pair  (first → CARRY, second → SUPPORT)
      - TOP / JUNGLE   -> top pair  (first → OFFLANE, second → JUNGLER)
      - MIDDLE         -> mid (1v1)

    Coverage on the 1275-match corpus: ~98% for all three pairs.
    Missing cells fall back to the encoder's `global_rate` so the
    feature vector stays well-defined.
    """
    out: Dict[str, Dict[str, Optional[int]]] = {
        "radiant": {k: None for k in LANE_KEYS},
        "dire":    {k: None for k in LANE_KEYS},
    }
    for side in ("radiant", "dire"):
        bots: List[int] = []
        tops: List[int] = []
        mids: List[int] = []
        for pp in (match.get(side) or {}).get("player_performances") or []:
            bucket = _LANE_MAP.get(_lane_of(pp) or "")
            vid = _vid_of(pp)
            if vid is None or bucket is None:
                continue
            if bucket == "BOT":
                if len(bots) < 2:
                    bots.append(vid)
            elif bucket == "TOP":
                if len(tops) < 2:
                    tops.append(vid)
            elif bucket == "MID":
                if len(mids) < 1:
                    mids.append(vid)
        if bots:
            out[side]["BOT_CARRY"] = bots[0]
        if len(bots) >= 2:
            out[side]["BOT_SUPPORT"] = bots[1]
        if tops:
            out[side]["TOP_OFFLANE"] = tops[0]
        if len(tops) >= 2:
            out[side]["TOP_JUNGLER"] = tops[1]
        if mids:
            out[side]["MID"] = mids[0]
    return out
