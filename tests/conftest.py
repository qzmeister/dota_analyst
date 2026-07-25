"""Shared pytest fixtures for the Dota Analyst test suite.

The business/ package uses relative imports (`from .analysis import ...`),
so the simplest way to make it importable from `tests/` is to add the
project root to sys.path here. This keeps `import business.analysis`
working both in pytest and in normal runs.
"""

import os
import sys
from pathlib import Path

# Make the project root (the directory that contains `business/`) importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ----- reusable test data ------------------------------------------------- #

import pytest


@pytest.fixture
def sample_team_a():
    """A team with strong baseline stats."""
    return {
        "id": 1001,
        "name": "Team Spirit",
        "tag": "TS",
        "logo": None,
        "rank": 3,
        "win_rate": 62.0,
        "fb_rate": 60.0,
        "f10_rate": 58.0,
    }


@pytest.fixture
def sample_team_b():
    """A weaker opponent."""
    return {
        "id": 1002,
        "name": "TBD",
        "tag": "TBD",
        "logo": None,
        "rank": 12,
        "win_rate": 48.0,
        "fb_rate": 45.0,
        "f10_rate": 44.0,
    }


@pytest.fixture
def sample_heroes_balanced():
    """5 heroes per side with average meta stats."""
    def hero(name, win_rate=50.0, avg_duration_sec=38 * 60, kda=3.0, roles=None):
        return {
            "id": 1,
            "steam_id": 1,
            "name": name,
            "win_rate": win_rate,
            "avg_duration": avg_duration_sec,
            "kda": kda,
            "roles": roles or [],
        }
    team_a = [hero("PA", win_rate=55, roles=["Carry"]),
              hero("CM", win_rate=49, roles=["Support"]),
              hero("Lion", win_rate=51, roles=["Support", "Nuker"]),
              hero("Axe", win_rate=52, roles=["Initiator"]),
              hero("Invoker", win_rate=53, roles=["Nuker", "Disabler"])]
    team_b = [hero("Jugg", win_rate=50, roles=["Carry"]),
              hero("WD", win_rate=48, roles=["Support"]),
              hero("Lina", win_rate=50, roles=["Nuker"]),
              hero("Pudge", win_rate=49, roles=["Initiator", "Disabler"]),
              hero("Mirana", win_rate=50, roles=["Escape"])]
    return team_a, team_b
