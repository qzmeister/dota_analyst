"""Team-name normalization + fuzzy matching for odds backends.

Why this exists
---------------
Odds backends (BetBoom, odds-api.io, Pinnacle, ...) all spell team
names slightly differently.  Our local live card uses DLTV /
OpenDota names ("Team Liquid", "Gaimin Gladiators", "BetBoom
Team") which the poller normalises via `odds_live._key()` —
currently just `lower() + strip()`.

That breaks when the bookmaker uses "Team.Liquid" or "BetBoom
Esports" — the cache stores "team.liquid|betboom esports" but the
live card looks up "team liquid|betboom team", and the cache miss
silently kills the live card's odds block.

The fix: a single normalisation function used by BOTH the backend
(store) and the lookup (read).  Plus an explicit alias map for
the worst offenders (Dota 2 pro teams have many spellings).

What "normalised" means here
----------------------------
* lowercase
* strip whitespace
* replace any of `.,_-` with a space
* collapse multiple spaces
* drop a small set of well-known suffixes ("esports", "dota 2",
  "gaming", "team", "gg", "go") when they appear at the END —
  i.e. "BetBoom Team" → "betboom", "OG Dota 2" → "og"
* also drop a small set of well-known prefixes ("team ")

That collapses the most common divergent spellings to a single
canonical form, while keeping the operation idempotent and cheap
(no fuzzy-string library required).

The alias map is the safety net for the rest — e.g. "BB" ↔
"BetBoom", "TSM" ↔ "Team SoloMid", "EG" ↔ "Evil Geniuses".
"""
from __future__ import annotations

import re
from typing import Optional


# Canonical form for the most common Dota 2 pro team divergences.
# Map: any of these → canonical.
_TEAM_ALIASES: dict[str, str] = {
    # BetBoom
    "bb": "betboom",
    "bet boom": "betboom",
    "betboom": "betboom",
    "betboomteam": "betboom",
    "betboom team": "betboom",
    # Team Liquid
    "team liquid": "team liquid",
    "team.liquid": "team liquid",
    "team_liquid": "team liquid",
    "teamliquid": "team liquid",
    "liquid": "team liquid",
    # TSM / Team SoloMid
    "tsm": "tsm",
    "team solomid": "tsm",
    "team solomiddle": "tsm",
    "team.somid": "tsm",
    # Evil Geniuses
    "eg": "eg",
    "evil geniuses": "eg",
    "evil genius": "eg",
    # Gaimin Gladiators
    "gaimin gladiators": "gaimin gladiators",
    "gaimingladiators": "gaimin gladiators",
    "gg": "gaimin gladiators",  # NB: conflicts with "go" suffix removal
    # Team Falcons
    "team falcons": "team falcons",
    "teamfalcons": "team falcons",
    "falcons": "team falcons",
    # PSG.LGD / LGD
    "psg.lgd": "lgd",
    "psg lgd": "lgd",
    "lgd gaming": "lgd",
    "lgd": "lgd",
    # OG
    "og": "og",
    # Tundra
    "tundra esports": "tundra",
    "tundraesports": "tundra",
    "tundra": "tundra",
    # Shopify Rebellion
    "shopify rebellion": "shopify rebellion",
    "shopifyrebellion": "shopify rebellion",
    # nouns / nouns esports
    "nouns": "nouns",
    "nouns esports": "nouns",
    # Xtreme Gaming
    "xtreme gaming": "xtreme gaming",
    "xtremegaming": "xtreme gaming",
    # Talon Esports
    "talon esports": "talon",
    "talone sports": "talon",
    "talon": "talon",
    # Aurora
    "aurora": "aurora",
    "aurora gaming": "aurora",
    # PARIVISION
    "parivision": "parivision",
    # Team Spirit
    "team spirit": "team spirit",
    "teamspirit": "team spirit",
    "spirit": "team spirit",
    # MOUZ (was mousesports)
    "mouz": "mouz",
    "mousesports": "mouz",
}


# Suffixes to strip AFTER the alias lookup.  These are common
# franchise suffixes that don't add information.  Stored without
# the leading space — we add it back when testing.
_DROP_SUFFIXES = (
    "esports",
    "dota 2",
    "dota2",
    "gaming",
    "team",
    "club",
    "brothers",   # Yakult Brothers -> yakult
    "academy",    # Inner Circle Academy -> inner circle
    "gg",         # may conflict — checked AFTER alias map
    "go",         # ditto
)

# Prefixes to strip.
_DROP_PREFIXES = (
    "team ",
)


# Regex: any of `.,_-` is replaced with a space.
_PUNCT_RE = re.compile(r"[._\-]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_team_name(name: str) -> str:
    """Return the canonical short form of a Dota 2 team name.

    Idempotent: `normalize(normalize(x)) == normalize(x)`.

    Examples:
        normalize_team_name("Team Liquid") == "team liquid"
        normalize_team_name("Team.Liquid") == "team liquid"
        normalize_team_name("TEAM_LIQUID Dota 2") == "team liquid"
        normalize_team_name("BetBoom Team") == "betboom"
        normalize_team_name("BB Esports") == "betboom"
        normalize_team_name("OG") == "og"
    """
    if not name:
        return ""
    s = name.strip().lower()
    if not s:
        return ""
    # Replace punctuation with spaces
    s = _PUNCT_RE.sub(" ", s)
    # Collapse whitespace
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    # Alias map (full string match)
    if s in _TEAM_ALIASES:
        return _TEAM_ALIASES[s]
    # Strip well-known prefixes
    changed = True
    while changed:
        changed = False
        for p in _DROP_PREFIXES:
            if s.startswith(p) and len(s) > len(p):
                s = s[len(p):]
                changed = True
                break
    # Strip well-known suffixes (loop so we catch e.g. "Team Liquid Dota 2")
    changed = True
    while changed:
        changed = False
        for suf in _DROP_SUFFIXES:
            with_space = " " + suf
            if s.endswith(with_space) and len(s) > len(with_space):
                s = s[: -len(with_space)]
                changed = True
                break
    # Re-alias after stripping — the prefix/suffix removal may
    # have unmasked a known alias.
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    if s in _TEAM_ALIASES:
        return _TEAM_ALIASES[s]
    return s


def team_pair_key(team_a: str, team_b: str) -> str:
    """Build the cache key for a (a, b) pair, normalised.

    Both inputs are passed through `normalize_team_name` first.
    """
    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)
    return f"{a}|{b}"


def fuzzy_pair_score(
    cand_a: str, cand_b: str,
    query_a: str, query_b: str,
) -> float:
    """Score a candidate (cand_a, cand_b) vs the query (query_a, query_b).

    Returns a float in [0, 1] where 1.0 = exact match on both,
    0.5 = one side exact, the other not, etc.  Used by the live
    card's lookup fallback when an exact key miss happens.

    Simple symmetric: we try the candidate in both (a, b) and
    (b, a) orderings and take the best.
    """
    a = normalize_team_name(cand_a)
    b = normalize_team_name(cand_b)
    qa = normalize_team_name(query_a)
    qb = normalize_team_name(query_b)
    # Best of two orderings
    direct = _score_pair(a, b, qa, qb)
    swapped = _score_pair(b, a, qa, qb)
    return max(direct, swapped)


def _score_pair(a: str, b: str, qa: str, qb: str) -> float:
    """Score 0.0–1.0 for (a, b) matching (qa, qb) in the SAME order."""
    if not a or not b or not qa or not qb:
        return 0.0
    sa = _one_side(a, qa)
    sb = _one_side(b, qb)
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return (sa + sb) / 2.0


def _one_side(cand: str, query: str) -> float:
    """Score 0.0–1.0 for one side.  Exact = 1.0, substring = 0.5,
    no overlap = 0.0.  Could be a real fuzzy metric but the
    simple version works for 95% of Dota 2 team names."""
    if not cand or not query:
        return 0.0
    if cand == query:
        return 1.0
    if cand in query or query in cand:
        return 0.5
    return 0.0


# --------------------------------------------------------------------------- #
# Convenience: a debug "explainer" for manual testing.  Prints the
# normalised form for a list of test strings.  Invoked from a
# one-off Python REPL — not used at runtime.
# --------------------------------------------------------------------------- #

def _demo() -> None:
    samples = [
        "Team Liquid",
        "team.liquid",
        "TEAM_LIQUID Dota 2",
        "BetBoom Team",
        "BB Esports",
        "BetBoom",
        "OG",
        "OG Dota 2",
        "Team Falcons",
        "Falcons",
        "Gaimin Gladiators",
        "GAIMIN_GLADIATORS Esports",
        "TSM",
        "Team SoloMid",
        "PSG.LGD",
        "LGD Gaming",
        "Aurora",
        "Aurora Gaming",
        "Team Spirit",
        "Spirit",
        "MOUZ",
        "mousesports",
        "PARIVISION",
        "Xtreme Gaming",
        "Talon Esports",
        "Tundra Esports",
        "Shopify Rebellion",
        "Nouns",
        "Evil Geniuses",
        "EG",
    ]
    for s in samples:
        print(f"  {s!r:40s} -> {normalize_team_name(s)!r}")


if __name__ == "__main__":
    _demo()
