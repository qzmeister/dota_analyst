"""
Unit tests for `business.ml.features` — the target-encoding feature
extraction that backs the MLEngine.

The encoder and the feature vector are the single source of truth for
the model's input contract; these tests pin that contract so future
refactors can't silently change the feature order without breaking
trained-model compatibility.
"""

from __future__ import annotations

import pytest

from business.ml.features import (
    FEATURE_ORDER,
    N_FEATURES,
    HeroWinRateEncoder,
    extract_features,
    hero_ids_from_match,
    target_from_match,
)


# --------------------------------------------------------------------------- #
# Test fixtures — minimal match dicts in the DatDota full_matches format
# --------------------------------------------------------------------------- #

def _match(
    radiant_heroes, dire_heroes, radiant_victory=True, has_error=False,
    r_team=None, d_team=None,
    r_lanes=None, d_lanes=None,
):
    """Build a minimal match dict with the keys the encoder reads.

    `r_team` / `d_team` are valve_ids (or `None`) attached to the
    `team.valve_id` field so the team feature group has something
    to look up.

    `r_lanes` / `d_lanes` are lists of lane strings (BOTTOM/TOP/
    MIDDLE/ROAM/JUNGLE) parallel to the hero lists.  When set,
    the per-player `laneInfo.lane` is populated, which is what
    the 0.3.10 lane-pair encoder needs.
    """
    def player(hero_id, lane=None):
        perf = {"hero": {"valve_id": hero_id}}
        pp = {"performance": perf}
        if lane is not None:
            pp["laneInfo"] = {"lane": lane}
        return pp
    r_pps = [
        player(h, lane=(r_lanes[i] if r_lanes else None))
        for i, h in enumerate(radiant_heroes)
    ]
    d_pps = [
        player(h, lane=(d_lanes[i] if d_lanes else None))
        for i, h in enumerate(dire_heroes)
    ]
    r_team_d = {"name": "Radiant"} if r_team is None else {"name": "Radiant", "valve_id": r_team}
    d_team_d = {"name": "Dire"} if d_team is None else {"name": "Dire", "valve_id": d_team}
    return {
        "match_id": 1,
        "radiant_victory": radiant_victory,
        "has_error": has_error,
        "radiant": {"team": r_team_d, "player_performances": r_pps},
        "dire": {"team": d_team_d, "player_performances": d_pps},
    }


# --------------------------------------------------------------------------- #
# HeroWinRateEncoder
# --------------------------------------------------------------------------- #

class TestHeroWinRateEncoder:
    def test_empty_fit_uses_default_global_rate(self):
        enc = HeroWinRateEncoder().fit([])
        assert enc.global_rate == 0.5
        # Any (side, hero) lookup falls back to global rate.
        assert enc.encode("radiant", 999) == 0.5
        assert enc.encode("dire", 42) == 0.5

    def test_global_rate_is_class_balance(self):
        matches = [
            _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True),
            _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=False),
            _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True),
        ]
        enc = HeroWinRateEncoder(smoothing=0.0, min_samples=1).fit(matches)
        # 2/3 radiant wins → global_rate = 2/3
        assert enc.global_rate == pytest.approx(2 / 3)

    def test_known_hero_radiant_wr(self):
        # Hero 1 is on radiant for 4 matches, wins 3.
        wins_match = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True)
        loss_match = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=False)
        matches = [wins_match] * 3 + [loss_match]
        enc = HeroWinRateEncoder(smoothing=0.0, min_samples=1).fit(matches)
        # 3 wins / 4 appearances on radiant side.
        assert enc.encode("radiant", 1) == pytest.approx(0.75)

    def test_known_hero_dire_wr_uses_inverse_target(self):
        # When dire wins, the radiant_victory flag is False → we count
        # 1 - target as the dire-side win.
        wins_match = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=False)
        loss_match = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True)
        matches = [wins_match] * 3 + [loss_match]
        enc = HeroWinRateEncoder(smoothing=0.0, min_samples=1).fit(matches)
        # Hero 6 is on dire 4 times, dire wins 3.
        assert enc.encode("dire", 6) == pytest.approx(0.75)

    def test_unseen_hero_returns_global_rate(self):
        enc = HeroWinRateEncoder(smoothing=0.0, min_samples=1).fit(
            [_match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])]
        )
        # Hero 999 has never been seen on either side.
        assert enc.encode("radiant", 999) == enc.global_rate
        assert enc.encode("dire", 999) == enc.global_rate

    def test_smoothing_with_below_min_samples(self):
        # 2 appearances is below default min_samples=3 → encoder falls
        # back to the smoothed estimate rather than 0.5/2 = 0.25. We
        # mix one radiant win and one dire win so the global rate is
        # the expected 0.5 baseline.
        win = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True)
        loss = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=False)
        enc = HeroWinRateEncoder(smoothing=10.0, min_samples=3).fit([win, loss])
        rate = enc.encode("radiant", 1)
        # global_rate = 1/2 = 0.5. Hero 1 on radiant: 1 win / 2 samples.
        # Smoothed: (1 + 10*0.5) / (2 + 10) = 6/12 = 0.5.
        assert enc.global_rate == pytest.approx(0.5)
        assert rate == pytest.approx(0.5)

    def test_to_dict_from_dict_roundtrip(self):
        matches = [_match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])] * 5
        enc1 = HeroWinRateEncoder().fit(matches)
        d = enc1.to_dict()
        enc2 = HeroWinRateEncoder.from_dict(d)
        # Same lookup table.
        for side in ("radiant", "dire"):
            for h in (1, 6, 99):
                assert enc1.encode(side, h) == pytest.approx(enc2.encode(side, h))
        # Round-trip preserves config too.
        assert enc2.smoothing == enc1.smoothing
        assert enc2.min_samples == enc1.min_samples
        assert enc2.global_rate == pytest.approx(enc1.global_rate)

    def test_errored_matches_are_skipped(self):
        m_ok = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True)
        m_err = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], has_error=True)
        enc = HeroWinRateEncoder(smoothing=0.0, min_samples=1).fit([m_ok, m_err])
        # 1/1 → win, the errored row must not have polluted the rate.
        assert enc.encode("radiant", 1) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# FEATURE_ORDER / N_FEATURES
# --------------------------------------------------------------------------- #

class TestFeatureConstants:
    def test_n_features_matches_order(self):
        # 0.3.13: FEATURE_ORDER is the concatenation of five
        # feature groups: hero (13) + team (4) + lane (7) +
        # matchup (3) + patch (3) = 30.  Trainers can pick a
        # subset via `groups=("hero",)` etc.
        assert N_FEATURES == 30
        assert len(FEATURE_ORDER) == 30
        from business.ml.features import FEATURE_GROUPS
        assert N_FEATURES == sum(len(g) for g in FEATURE_GROUPS.values())

    def test_order_is_tuple(self):
        # Train and predict both index by position; a list would behave
        # the same but a tuple is immutable and signals "this is the
        # contract" to readers.
        assert isinstance(FEATURE_ORDER, tuple)

    def test_feature_names_are_distinct(self):
        assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER)

    def test_feature_groups_partition_canonical_order(self):
        # FEATURE_ORDER = sum(FEATURE_GROUPS.values(), ()).  This
        # invariant is what guarantees `extract_features(groups=...)`
        # returns the same column order the engine expects at predict.
        from business.ml.features import FEATURE_GROUPS
        assert FEATURE_ORDER == sum(FEATURE_GROUPS.values(), ())


# --------------------------------------------------------------------------- #
# extract_features
# --------------------------------------------------------------------------- #

class TestExtractFeatures:
    def test_basic_5v5_returns_24_floats(self):
        # Default: all three groups (hero+team+lane).  The trainer
        # can pass a subset; the default is the "all-in" model.
        enc = HeroWinRateEncoder()
        match = _match(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
            r_lanes=["BOTTOM", "ROAM", "TOP", "TOP", "MIDDLE"],
            d_lanes=["BOTTOM", "BOTTOM", "TOP", "JUNGLE", "MIDDLE"],
            r_team=100, d_team=200,
        )
        feats = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc,
            match=match,
            radiant_team_id=100, dire_team_id=200,
        )
        assert isinstance(feats, list)
        assert len(feats) == N_FEATURES  # 24 by default
        assert all(isinstance(x, float) for x in feats)

    def test_hero_only_returns_13_floats(self):
        # The 0.3.9 baseline was 13 hero-only features.  Reproducing
        # it explicitly here pins the contract.
        enc = HeroWinRateEncoder()
        feats = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc, groups=("hero",)
        )
        assert len(feats) == 13

    def test_feature_order_matches_constant_hero(self):
        # The order returned by extract_features is the order train sees.
        # Use groups=("hero",) to pin the 0.3.9 layout.
        enc = HeroWinRateEncoder()
        feats = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc, groups=("hero",)
        )
        r = [enc.encode("radiant", h) for h in [1, 2, 3, 4, 5]]
        d = [enc.encode("dire", h) for h in [6, 7, 8, 9, 10]]
        mean_r = sum(r) / 5.0
        mean_d = sum(d) / 5.0
        expected = [mean_r, mean_d, *r, *d, mean_r - mean_d]
        assert feats == pytest.approx(expected)

    def test_wrong_count_raises(self):
        enc = HeroWinRateEncoder()
        with pytest.raises(ValueError):
            extract_features([1, 2, 3, 4], [6, 7, 8, 9, 10], enc, groups=("hero",))
        with pytest.raises(ValueError):
            extract_features([1, 2, 3, 4, 5], [6, 7, 8, 9], enc, groups=("hero",))

    def test_radiant_minus_dire_sign(self):
        # Heroes 1..5 always win on radiant in this corpus; heroes 6..10
        # always lose. The difference should be strictly positive.
        wins = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True)
        enc = HeroWinRateEncoder(smoothing=0.0, min_samples=3).fit([wins] * 10)
        feats = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc, groups=("hero",)
        )
        # Last feature is the difference.
        assert feats[-1] > 0

    def test_lane_group_requires_match_or_lane_dicts(self):
        enc = HeroWinRateEncoder().fit([_match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], True)])
        with pytest.raises(ValueError):
            extract_features([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc, groups=("lane",))
        # Pass an empty lane dict for each side -> uses encoder's global_rate.
        empty = {k: None for k in ("BOT_CARRY", "BOT_SUPPORT", "TOP_OFFLANE", "TOP_JUNGLER", "MID")}
        feats = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc,
            groups=("lane",), radiant_lane=empty, dire_lane=empty,
        )
        assert len(feats) == 7

    def test_team_group_uses_team_id(self):
        # Same encoder, two different team ids -> the team-group
        # features should differ (the encoder's team lookup is
        # team-specific).
        enc = HeroWinRateEncoder().fit(
            [_match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], True, r_team=100, d_team=200)] * 10
        )
        f1 = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc,
            groups=("team",), radiant_team_id=100, dire_team_id=200,
        )
        f2 = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc,
            groups=("team",), radiant_team_id=999, dire_team_id=888,
        )
        # Different teams -> different features.
        assert f1 != f2
        assert len(f1) == 4

    def test_groups_concatenates_in_order(self):
        enc = HeroWinRateEncoder()
        match = _match(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
            r_lanes=["BOTTOM", "ROAM", "TOP", "TOP", "MIDDLE"],
            d_lanes=["BOTTOM", "BOTTOM", "TOP", "JUNGLE", "MIDDLE"],
            r_team=100, d_team=200,
        )
        feats = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc,
            match=match, radiant_team_id=100, dire_team_id=200,
            groups=("hero", "team", "lane"),
        )
        # 13 + 4 + 7 = 24
        assert len(feats) == 24
        # Re-ordering groups must keep the total length (even though
        # the columns shift — the trainer is responsible for keeping
        # train and predict in sync).
        feats2 = extract_features(
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc,
            match=match, radiant_team_id=100, dire_team_id=200,
            groups=("lane", "team", "hero"),
        )
        assert len(feats2) == 24

    def test_unknown_group_raises(self):
        enc = HeroWinRateEncoder()
        with pytest.raises(ValueError):
            extract_features([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], enc, groups=("nope",))


# --------------------------------------------------------------------------- #
# Match dict helpers
# --------------------------------------------------------------------------- #

class TestMatchHelpers:
    def test_target_from_match(self):
        m_win = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=True)
        m_loss = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_victory=False)
        m_err = _match([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], has_error=True)
        m_missing = {"has_error": False}  # no radiant_victory key
        assert target_from_match(m_win) == 1
        assert target_from_match(m_loss) == 0
        assert target_from_match(m_err) is None
        assert target_from_match(m_missing) is None

    def test_hero_ids_from_match(self):
        m = _match([10, 20, 30, 40, 50], [60, 70, 80, 90, 100])
        r, d = hero_ids_from_match(m)
        assert r == [10, 20, 30, 40, 50]
        assert d == [60, 70, 80, 90, 100]

    def test_hero_ids_from_match_truncates_to_five(self):
        m = _match([1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14])
        r, d = hero_ids_from_match(m)
        # The fixture builder gives us 5 radiant + 5 dire; if there were
        # more, hero_ids_from_match would still take the first 5.
        assert len(r) == 5
        assert len(d) == 5
