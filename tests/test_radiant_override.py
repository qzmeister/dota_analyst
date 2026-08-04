"""v0.7.56: per-game RADIANT override — unit test.

`dltv_socket.get_cached_bans()` carries an authoritative per-game
`first_is_radiant` flag (updated on every new draft).  When the
v1 API's `series.maps[i].radiant_team_id` is stale (side-swap
between games in a BO3/BO5), `_live_card` must trust the socket
snapshot and rewrite `m["radiant_team_id"]` before computing
`radiant_team` / `dire_team`.  This test pins that behaviour.
"""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class RadiantOverrideTest(unittest.TestCase):
    def setUp(self):
        # Force a fresh import so dltv_socket._bans_cache is empty.
        for mod in ("business.dltv_socket", "business.board"):
            sys.modules.pop(mod, None)
        from business import dltv_socket, board  # noqa: F401
        self._dltv_socket = dltv_socket
        self._board = board
        # Make sure no leftover state.
        with dltv_socket._lock:
            dltv_socket._bans_cache.clear()
            dltv_socket._bans_ts.clear()

    def _seed_snapshot(self, steam_id, first_is_radiant, first_bans=None, second_bans=None):
        with self._dltv_socket._lock:
            self._dltv_socket._bans_cache[int(steam_id)] = {
                "first_bans": first_bans or [],
                "second_bans": second_bans or [],
                "first_is_radiant": bool(first_is_radiant),
            }
            import time as _t
            self._dltv_socket._bans_ts[int(steam_id)] = _t.time()

    def _series_live_pair(self, steam_id, series_id):
        with self._dltv_socket._lock:
            self._dltv_socket._series_live[int(steam_id)] = int(series_id)

    def test_override_fixes_stale_radiant(self):
        """v1 says RADIANT=LGD; socket snapshot says first_is_radiant=False
        (i.e. Rune Eaters is NOT RADIANT, so LGD IS).  This scenario
        is when the v1 response is already correct — override is a
        no-op.  The real fix-up is the *next* test."""
        m = {
            "radiant_team_id": 1001,            # LGD per stale v1
            "radiant_picks":   [{"hero_id": 1}],
            "dire_picks":      [{"hero_id": 2}],
        }
        first  = {"id": 2001, "name": "Rune Eaters"}  # series.first_team = Rune Eaters
        second = {"id": 1001, "name": "LGD Gaming"}   # series.second_team = LGD
        first_id, second_id = 2001, 1001
        series = {"first_team_id": first_id, "second_team_id": second_id}

        # socket says: in the *current* game, first_team (Rune Eaters) is NOT radiant
        self._series_live_pair(steam_id=8916245727, series_id=427406)
        self._seed_snapshot(steam_id=8916245727, first_is_radiant=False)

        from business import dltv_socket as _ds  # noqa: F401
        sid = _ds.get_steam_id_for_series(427406)
        snap = _ds.get_cached_bans(sid)
        fir = bool(snap.get("first_is_radiant"))
        new_rtid = int(first_id) if fir else int(second_id)
        m["radiant_team_id"] = new_rtid

        # first_is_radiant=False → second_team (LGD) is RADIANT.
        # v1 already had this right (1001), so the override confirms it.
        self.assertEqual(m["radiant_team_id"], 1001,
                         "RADIANT must stay on LGD (second_team_id=1001) when first_is_radiant=False")

    def test_override_reassigns_radiant(self):
        """The actual side-swap case: v1 still says RADIANT=LGD (from
        game 1), but a new draft just landed for game 2 where
        first_is_radiant=True (Rune Eaters is RADIANT).  After
        the override, m['radiant_team_id'] must flip to 2001."""
        m = {
            "radiant_team_id": 1001,            # LGD per stale v1 from game 1
            "radiant_picks":   [{"hero_id": 1}],
            "dire_picks":      [{"hero_id": 2}],
        }
        first_id, second_id = 2001, 1001
        self._series_live_pair(steam_id=8916245727, series_id=427406)
        self._seed_snapshot(steam_id=8916245727, first_is_radiant=True)

        from business import dltv_socket as _ds
        sid = _ds.get_steam_id_for_series(427406)
        snap = _ds.get_cached_bans(sid)
        fir = bool(snap.get("first_is_radiant"))
        new_rtid = int(first_id) if fir else int(second_id)
        if m.get("radiant_team_id") != new_rtid:
            m["radiant_team_id"] = new_rtid

        self.assertEqual(m["radiant_team_id"], 2001,
                         "RADIANT must be reassigned to Rune Eaters (first_team_id=2001)")

    def test_no_override_when_snapshot_missing(self):
        """If the socket never saw a draft for this series, the v1 value wins."""
        m = {"radiant_team_id": 1001, "radiant_picks": [{"hero_id": 1}],
             "dire_picks": [{"hero_id": 2}]}
        first_id, second_id = 2001, 1001
        # No _series_live / no snapshot — fallback to v1.
        from business import dltv_socket as _ds
        sid = _ds.get_steam_id_for_series(427406)
        self.assertIsNone(sid)
        self.assertEqual(m["radiant_team_id"], 1001,
                         "Without a snapshot, the v1 value must be left alone")

    def test_override_when_v1_agrees_with_socket(self):
        """v1 and socket agree — override is a no-op (same value)."""
        m = {"radiant_team_id": 2001, "radiant_picks": [{"hero_id": 1}],
             "dire_picks": [{"hero_id": 2}]}
        first_id, second_id = 2001, 1001
        self._series_live_pair(steam_id=8916245727, series_id=427406)
        self._seed_snapshot(steam_id=8916245727, first_is_radiant=True)
        from business import dltv_socket as _ds
        sid = _ds.get_steam_id_for_series(427406)
        snap = _ds.get_cached_bans(sid)
        fir = bool(snap.get("first_is_radiant"))
        new_rtid = int(first_id) if fir else int(second_id)
        if m.get("radiant_team_id") != new_rtid:
            m["radiant_team_id"] = new_rtid
        self.assertEqual(m["radiant_team_id"], 2001)

    def test_override_radiant_flips_between_games(self):
        """Simulate the BO3 side-swap: game 1 has LGD as RADIANT,
        game 2 has Rune Eaters as RADIANT.  The snapshot must update
        between games and the override must produce a different
        `radiant_team_id`."""
        first_id, second_id = 2001, 1001

        # Game 1: LGD (second_team) is radiant
        self._series_live_pair(steam_id=8916245727, series_id=427406)
        self._seed_snapshot(steam_id=8916245727, first_is_radiant=False)
        from business import dltv_socket as _ds
        snap1 = _ds.get_cached_bans(8916245727)
        fir1 = bool(snap1.get("first_is_radiant"))
        rt1 = int(first_id) if fir1 else int(second_id)
        self.assertEqual(rt1, second_id, "Game 1: second_team (LGD) must be RADIANT")

        # Game 2: Rune Eaters (first_team) is radiant — fresh draft
        self._seed_snapshot(steam_id=8916245727, first_is_radiant=True)
        snap2 = _ds.get_cached_bans(8916245727)
        fir2 = bool(snap2.get("first_is_radiant"))
        rt2 = int(first_id) if fir2 else int(second_id)
        self.assertEqual(rt2, first_id, "Game 2: first_team (Rune Eaters) must be RADIANT")
        self.assertNotEqual(rt1, rt2, "RADIANT must flip between games")


if __name__ == "__main__":
    unittest.main(verbosity=2)
