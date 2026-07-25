"""
Domain-specific exception hierarchy for the Dota Analyst business service.

Why a hierarchy?
----------------
Before this module the codebase had 31 `except Exception` blocks.  That's
the "catch everything" anti-pattern: a network timeout looks the same as
a JSON parse error looks the same as a missing model file, and the
caller has no way to tell what actually went wrong from a log line.

The hierarchy below groups exceptions by **what subsystem** they
originate from.  Callers can `except MLError` to mean "an ML subsystem
problem — fall back to heuristic" without accidentally swallowing
`KeyError` from a real bug.

Hierarchy
---------
::

    DotaAnalystError                 (root; all our errors subclass this)
    +-- MLError                      (business.ml.*)
    |   +-- MLPredictError           (engine.predict / HeuristicEngine fallback)
    |   +-- MLTrainError             (business.ml.train CLI failures)
    +-- BoardBuildError              (business.board — board assembly)
    +-- DiscoveryError               (business.discovery — scrapers/trackers)
    |   +-- ScrapeError              (dltv.org HTML scrape)
    |   +-- SteamFetchError          (Steam GetLiveLeagueGames)
    |   +-- ParseError               (regex / HTML parse failures)
    +-- UpstreamError                (outbound HTTP to third parties)
    |   +-- DLTVError                (dltv.org API)
    |   +-- SteamAPIError            (api.steampowered.com)
    +-- InfraError                   (process-local infra)
        +-- HTTPClientError          (business._http — shared HTTP client)
        +-- StreamError              (business.stream — SSE pub-sub)
        +-- GatewayError             (gateway.app — proxy dispatch)

Backward compatibility
----------------------
Every class here inherits from `Exception`, so the existing
`except Exception` blocks still catch them.  New code should narrow
the catch to the specific subclass, but old code that says
`except Exception` keeps working — the SSE poller's "never let this
die" stance still covers everything, for instance.
"""

from __future__ import annotations


class DotaAnalystError(Exception):
    """Root of the project's exception hierarchy.

    All custom errors inherit from this so callers can write
    `except DotaAnalystError` to mean "any error from our own code,
    re-raise the rest" — useful at process boundaries where we
    want to log + translate but not swallow stdlib / 3rd-party bugs.
    """


# --------------------------------------------------------------------------- #
# ML subsystem
# --------------------------------------------------------------------------- #

class MLError(DotaAnalystError):
    """Any error from the `business.ml.*` package.

    Caught by callers that want to "fall back to heuristic" — the
    engine's `predict` and the board's per-series loop use this.
    """


class MLPredictError(MLError):
    """A model failed to predict (missing artifact, feature mismatch, etc.).

    Distinct from `MLTrainError` because training failures are operator-
    visible (CLI exits non-zero) while predict failures are runtime
    fallbacks (log + use heuristic).
    """


class MLTrainError(MLError):
    """Training-time failure (bad data, NaN losses, IO error saving model).

    The training CLI surfaces this; the engine never sees it.
    """


# --------------------------------------------------------------------------- #
# Board assembly
# --------------------------------------------------------------------------- #

class BoardBuildError(DotaAnalystError):
    """The Kanban builder failed to assemble a single board snapshot.

    Usually per-series: a single DLTV series that 404'd or returned
    garbage.  The board loop catches these and shows the series as
    "stale" rather than failing the whole /api/board call.
    """


# --------------------------------------------------------------------------- #
# Discovery (scrapers + trackers)
# --------------------------------------------------------------------------- #

class DiscoveryError(DotaAnalystError):
    """Any error from the discovery subsystem (HTML scrape, Steam live, parsing)."""


class ScrapeError(DiscoveryError):
    """dltv.org HTML scrape failed (timeout, non-200, malformed body)."""


class SteamFetchError(DiscoveryError):
    """Steam GetLiveLeagueGames failed (timeout, missing key, non-200)."""


class ParseError(DiscoveryError):
    """A regex / structural parse failed on already-fetched content.

    Distinct from `ScrapeError` so the caller can decide whether to
    retry the fetch (Scrape) or just drop the bad row (Parse).
    """


# --------------------------------------------------------------------------- #
# Upstream HTTP
# --------------------------------------------------------------------------- #

class UpstreamError(DotaAnalystError):
    """Outbound HTTP to a third-party API failed.

    Shared base so generic middleware (e.g. gateway) can do
    `except UpstreamError` and translate to 502 instead of 500.
    """


class DLTVError(UpstreamError):
    """dltv.org API call failed (v1 endpoints, /live/{id}.json)."""


class SteamAPIError(UpstreamError):
    """api.steampowered.com call failed (GetLiveLeagueGames, etc.)."""


# --------------------------------------------------------------------------- #
# Process-local infrastructure
# --------------------------------------------------------------------------- #

class InfraError(DotaAnalystError):
    """Local infra issue (HTTP client pool, SSE pub-sub, gateway dispatch)."""


class HTTPClientError(InfraError):
    """The shared HTTP client (`business._http`) failed.

    E.g. JSON decode of a 200 response, SSL handshake, socket-level error.
    """


class StreamError(InfraError):
    """SSE pub-sub hub failed (queue overflow, broken subscriber)."""


class GatewayError(InfraError):
    """The gateway's reverse-proxy dispatch failed (target unreachable, bad upstream response)."""


__all__ = [
    "DotaAnalystError",
    "MLError",
    "MLPredictError",
    "MLTrainError",
    "BoardBuildError",
    "DiscoveryError",
    "ScrapeError",
    "SteamFetchError",
    "ParseError",
    "UpstreamError",
    "DLTVError",
    "SteamAPIError",
    "InfraError",
    "HTTPClientError",
    "StreamError",
    "GatewayError",
]
