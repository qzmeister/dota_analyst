"""Bookmaker odds integration (v0.4.1 work).

Goal: given one of our live predictions, fetch the corresponding
market odds from a bookmaker (or aggregator) and surface the
"edge" — our model's probability vs. the implied probability
from the bookmaker's price.

Markets we care about (matches `live_predictions.jsonl` targets):
  * winner      — Moneyline on the match (P1 / P2)
  * total_kills — Over/Under on total kills (e.g. over 49.5)
  * duration    — Over/Under on match duration in minutes
                  (e.g. over 35.5 / over 42.5)

Why a pluggable backend?  The Dota 2 odds space is fragmented —
no single public API covers all of them, and what does exist
moves every 6-12 months.  Concretely:
  * BetBoom.ru         — auth-gated, no anonymous read; needs
                         Cloudflare + Fingerprint.js + Group-IB
                         bypass + a registered account.
  * Stratz.com         — needs a bearer token (free tier exists
                         for some endpoints, paid for others).
  * the-odds-api.com   — paid, multi-bookmaker aggregator,
                         supports Dota 2 as `esports_dota2`.
  * api.api-sport.ru   — paid, supports BetBoom as bookmaker
                         in their parameterised calls.
  * Pinnacle           — public guest API, but bot-protected;
                         only some endpoints open.
  * Self-hosted scrape — headless browser, deep links to
                         bookmaker pages, fragile.

This module ships with the interface and a `StubBackend` for
local dev.  The user wires up the real backend by setting
`ODDS_BACKEND` env var to a fully-qualified module path and
implementing the `OddsBackend` protocol.  The wiring code lives
in `business/app.py` lifespan.

Edge calculation:

    implied_prob = 1 / decimal_odds   (or 1/(odds/100+1) for American)
    edge = our_prob - implied_prob

If edge > +X% the model thinks the book is underpricing us;
if edge < -X% we're overpaying.  Threshold for "value bet"
should be a user-tuned constant.
"""
from __future__ import annotations

import os
import time
import urllib.request
import urllib.error
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class OddsQuote:
    """One bookmaker quote for a single (team, market) cell.

    `decimal_odds` is the European decimal (e.g. 1.85 for ~54%
    implied probability).  `implied_prob` is computed from
    decimal_odds at construction time.
    """
    market: str                # "winner" | "total_kills" | "duration"
    selection: str             # "radiant" | "dire" | "over" | "under" | threshold-as-str
    decimal_odds: float
    bookmaker: str = "unknown"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def implied_prob(self) -> float:
        if self.decimal_odds <= 1.0:
            return 1.0
        return 1.0 / float(self.decimal_odds)

    def edge(self, our_prob: float) -> float:
        """our_prob - implied_prob, in [-1, 1].

        Positive = model thinks the book is underpricing us
        (i.e. the bookmaker's implied probability is lower than
        ours).  Negative = the book is overpricing us.
        """
        return float(our_prob) - self.implied_prob


class OddsBackend(Protocol):
    """Pluggable odds source.  Implementations live in their own
    module so we can A/B between paid/free/scrape without touching
    the live card.

    Concrete impls would subclass this and store API keys in env
    vars; the protocol is what the rest of the codebase depends on.
    """
    name: str

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        """Return all currently-available quotes for `match_id`.

        May be empty if the match is not on the bookmaker's feed.
        Implementations should:
          * raise on transport error (so the caller can log + skip)
          * return [] on "no data" (e.g. match not in feed)
          * NOT block the live card — fail-soft is more important
            than fresh odds.
        """
        ...


class StubBackend:
    """No-op backend that always returns [].  Use for local dev /
    when no real backend is configured.  Logs a debug message
    once per minute to make it obvious odds are disabled.
    """
    name = "stub"

    def __init__(self) -> None:
        self._last_log: float = 0.0

    def get_quotes(self, match_id: int) -> List[OddsQuote]:
        now = time.time()
        if now - self._last_log > 60:
            print(f"[odds] StubBackend active — no real quotes for match {match_id}")
            self._last_log = now
        return []


# Sentinel used by the live card; the real backends are loaded
# lazily by `get_backend()` when the first quote is requested.
_backend: Optional[OddsBackend] = None
_backend_loaded = False


def get_backend() -> OddsBackend:
    """Return the configured backend (or StubBackend if not set).

    Set `ODDS_BACKEND=module.path.ClassName` to wire a real
    backend.  The class must have a no-arg constructor.
    """
    global _backend, _backend_loaded
    if _backend_loaded:
        return _backend or StubBackend()
    _backend_loaded = True
    path = os.environ.get("ODDS_BACKEND", "").strip()
    if not path:
        _backend = StubBackend()
        return _backend
    try:
        module_name, _, cls_name = path.rpartition(".")
        if not module_name:
            raise ValueError(f"ODDS_BACKEND must be 'module.path.ClassName', got {path!r}")
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        _backend = cls()
        print(f"[odds] loaded backend: {path} (name={getattr(_backend, 'name', '?')})")
        return _backend
    except Exception as exc:
        print(f"[odds] ODDS_BACKEND={path!r} failed to load: {exc}; falling back to StubBackend")
        _backend = StubBackend()
        return _backend


# --------------------------------------------------------------------------- #
# Front-end shape: live card surfaces odds as `predictions.odds`.
# --------------------------------------------------------------------------- #


def _winner_quotes(quotes: List[OddsQuote], radiant_is_first: bool) -> Dict[str, Any]:
    """Pick the two side quotes and compute edge for the model's pick.

    Convention: bookmaker labels "P1" / "P2" refer to the first /
    second team in the original series order (not radiant/dire).
    We map via the radiant_is_first flag stored at record time.
    """
    out: Dict[str, Any] = {"market": "winner", "quotes": []}
    p1 = [q for q in quotes if q.market == "winner" and q.selection == "P1"]
    p2 = [q for q in quotes if q.market == "winner" and q.selection == "P2"]
    if not (p1 and p2):
        return out
    out["quotes"] = [
        {"selection": "P1", "decimal_odds": q.decimal_odds,
         "implied_prob": q.implied_prob, "bookmaker": q.bookmaker}
        for q in p1
    ] + [
        {"selection": "P2", "decimal_odds": q.decimal_odds,
         "implied_prob": q.implied_prob, "bookmaker": q.bookmaker}
        for q in p2
    ]
    return out


def _total_kills_quotes(quotes: List[OddsQuote]) -> Dict[str, Any]:
    """Surface over/under pairs by threshold."""
    out: Dict[str, Any] = {"market": "total_kills", "quotes": []}
    for q in quotes:
        if q.market != "total_kills":
            continue
        out["quotes"].append({
            "selection": q.selection,  # "over" | "under" or "over_49.5" / "under_49.5"
            "decimal_odds": q.decimal_odds,
            "implied_prob": q.implied_prob,
            "bookmaker": q.bookmaker,
        })
    return out


def _duration_quotes(quotes: List[OddsQuote]) -> Dict[str, Any]:
    """Surface over/under pairs by threshold (e.g. over 35.5 / over 42.5)."""
    out: Dict[str, Any] = {"market": "duration", "quotes": []}
    for q in quotes:
        if q.market != "duration":
            continue
        out["quotes"].append({
            "selection": q.selection,
            "decimal_odds": q.decimal_odds,
            "implied_prob": q.implied_prob,
            "bookmaker": q.bookmaker,
        })
    return out


def compute_edge_for_card(
    match_id: int,
    prob_radiant: Optional[float],
    predicted_kills: Optional[float],
    predicted_duration: Optional[float],
    radiant_is_first: bool = True,
) -> Dict[str, Any]:
    """Return a dict of {winner, total_kills, duration} with quotes + edge.

    Used by the live card to surface `predictions.odds`.  Each
    market entry is shaped like:
        {
            "market": "winner",
            "quotes": [{"selection": "P1", "decimal_odds": 1.85, ...}, ...],
            "edge_radiant": 0.07,   # our_prob_radiant - bookmaker_implied
            "edge_dire": -0.07,
        }
    """
    backend = get_backend()
    try:
        quotes = backend.get_quotes(int(match_id)) or []
    except Exception as exc:
        # Fail-soft: log + return empty odds block.  The live card
        # should still render the prediction without odds if the
        # bookmaker is unreachable.
        print(f"[odds] backend {backend.name!r} raised on {match_id}: {exc}")
        quotes = []

    out: Dict[str, Any] = {"winner": {}, "total_kills": {}, "duration": {}}

    # ---- Winner ----
    wq = _winner_quotes(quotes, radiant_is_first)
    p1_quotes = [q for q in wq.get("quotes", []) if q["selection"] == "P1"]
    p2_quotes = [q for q in wq.get("quotes", []) if q["selection"] == "P2"]
    if p1_quotes and p2_quotes and prob_radiant is not None:
        # Use the best (lowest-implied) price per side — i.e. the
        # tightest book — for the edge calc.  Lower implied_prob =
        # better for the bettor.
        best_p1 = min(p1_quotes, key=lambda q: q["implied_prob"])
        best_p2 = min(p2_quotes, key=lambda q: q["implied_prob"])
        # prob_radiant is the model's probability for the radiant side.
        # If radiant_is_first, P1 corresponds to radiant, otherwise
        # P1 corresponds to dire.
        if radiant_is_first:
            prob_p1, prob_p2 = prob_radiant, 1.0 - prob_radiant
        else:
            prob_p1, prob_p2 = 1.0 - prob_radiant, prob_radiant
        wq["best_p1"] = best_p1
        wq["best_p2"] = best_p2
        wq["edge_p1"] = prob_p1 - best_p1["implied_prob"]
        wq["edge_p2"] = prob_p2 - best_p2["implied_prob"]
        wq["edge_radiant"] = wq["edge_p1"] if radiant_is_first else wq["edge_p2"]
        wq["edge_dire"] = wq["edge_p2"] if radiant_is_first else wq["edge_p1"]
    out["winner"] = wq

    # ---- Total kills ----
    tk = _total_kills_quotes(quotes)
    if tk.get("quotes") and predicted_kills is not None:
        # Find the threshold nearest to our predicted value.
        # Each quote's `selection` is e.g. "over_49.5"; parse the
        # threshold.  We surface the closest over/under pair so
        # the user sees "edge on over 49.5 at 1.85" if that
        # matches the model.
        # NB: very simple — doesn't pair over+under for the same
        # threshold.  Backend should give us paired quotes.
        out["total_kills"] = tk
    else:
        out["total_kills"] = tk

    # ---- Duration ----
    du = _duration_quotes(quotes)
    out["duration"] = du

    return out
