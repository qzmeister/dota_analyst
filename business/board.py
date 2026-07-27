"""
Board assembly: turns DLTV series into Kanban cards
(prematch / live / postmatch) for the selected leagues (events).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from ._logging import get_logger
from .accuracy import record_prediction
from .analysis import analyze, analyze_map_with_verdict, decode_towers
from .dltv_client import client, _parse_dt
from .exceptions import AccuracyError, DotaAnalystError, MLError
from .ml.engine import get_default_engine

log = get_logger(__name__)

BO_LABELS = {1: "BO1", 2: "BO2", 3: "BO3", 5: "BO5"}

# Build tunables (v0.3.15+).
#
# On a cold DLTV cache each `client.get_series(eid)` call walks a
# ~8 MB JSON response with a 12 s upstream timeout.  With 30+ active
# leagues that's a 6-minute serial walk — `/api/board` would
# time out under the nginx 30 s proxy_read_timeout.  Two guards:
#
#   MAX_LEAGUES_PER_BOARD — pick the top-N most recently active leagues
#                            from `leagues_with_status()` to bound the
#                            cold-cache walk.
#   SERIES_FETCH_WORKERS  — fetch the per-league series payloads in
#                            parallel via a thread pool.
MAX_LEAGUES_PER_BOARD = 12
SERIES_FETCH_WORKERS = 5


def classify_event_status(event_id: int) -> str:
    """Return 'live' if the event has any live/prematch match known to discovery,
    else None (the league is filtered out of the selector).

    Legacy helper kept for callers that still distinguish; leagues_with_status()
    no longer surfaces 'finished' or 'upcoming' — the UI only lists active leagues.
    """
    try:
        from .discovery import discover
        live_series, prematch_series = discover()
    except DotaAnalystError:
        # Discovery is best-effort here — failure just means we can't
        # confirm "is the league active right now" and return None.
        # Bugs in our code (KeyError, etc.) deliberately fall through.
        return None

    for ws in live_series:
        eid = ws.get("_scraper_event_id") or ws.get("event_id")
        if eid and int(eid) == int(event_id):
            return "live"
    for pm in prematch_series:
        eid = pm.get("event_id")
        if eid and int(eid) == int(event_id):
            return "live"
    return None


def leagues_with_status() -> List[Dict]:
    """Return leagues that currently have at least one live or upcoming match.

    Leagues with no future matches are hidden from the selector (the user asked
    to drop the 'finished' group entirely). Discovery (scraper + Steam) is the
    source of truth for 'active now'.
    """
    events = client.get_events() or []
    try:
        from .discovery import discover
        live_series, prematch_series = discover()
    except DotaAnalystError:
        # Discovery failure is non-fatal: we still want the league list
        # (from client.get_events) with no live/prematch overlay.
        live_series, prematch_series = [], []

    active_ids: Set[int] = set()
    for ws in live_series:
        eid = ws.get("_scraper_event_id") or ws.get("event_id")
        if eid:
            active_ids.add(int(eid))
    for pm in prematch_series:
        eid = pm.get("event_id")
        if eid:
            active_ids.add(int(eid))

    out: List[Dict] = []
    for e in events:
        eid = e.get("id")
        if not eid or int(eid) not in active_ids:
            continue
        out.append({
            "id": eid,
            "title": e.get("title"),
            "is_active": bool(e.get("is_active")),
            "status": "live",  # all listed leagues have future/live matches
        })
    # stable alphabetical sort
    out.sort(key=lambda x: (x.get("title") or "").lower())
    return out


def _bo_label(t) -> str:
    """Accept either an int (v1 series.type) or a str ('bo3') from scraper."""
    if isinstance(t, str):
        s = t.strip().lower()
        if s.startswith("bo") and s[2:].isdigit():
            return s.upper()
        return s.upper() or "BO?"
    return BO_LABELS.get(t, f"BO{t}" if t else "BO?")


def _series_bo_int(series: Dict) -> Optional[int]:
    """Resolve the series BO number (1/2/3/5) from v1 or scraper data."""
    t = series.get("type")
    if isinstance(t, int):
        return t
    if isinstance(t, str):
        s = t.strip().lower()
        if s.startswith("bo") and s[2:].isdigit():
            return int(s[2:])
    return None


def _hero_card(hero: Optional[Dict], hero_id: Optional[int]) -> Dict:
    if hero:
        return {
            "id": hero_id,
            "name": hero.get("name") or f"#{hero_id}",
            "image": hero.get("image"),
            "win_rate": hero.get("win_rate"),
        }
    return {"id": hero_id, "name": f"#{hero_id}", "image": None, "win_rate": None}


def _names_to_cards(entries: List[Dict]) -> List[Dict]:
    """Resolve a list of {name, hero_id?, steam_id?} into hero cards.

    Used by the v0.3.20+ match-state overlay path.  Resolution
    priority (v0.3.22+, when `dltv_browser` was upgraded to
    return both DLTV and steam ids):
      1. `steam_id` (Valve) — preferred, downstream engine uses
         this namespace.
      2. `hero_id` (DLTV internal) — looked up via
         `client.hero_by_dltv_id`, then by steam as a fallback.
      3. `name` (display title) — looked up in the hero index.
      4. None of the above — placeholder card with just the name.

    Each returned card carries:
      - `id`:       the steam id (Valve namespace) when known
      - `name`:     display name
      - `_dltv_id`: the DLTV internal id (set whenever the
                    caller passed one in), so downstream
                    consumers can re-construct the pick dict
                    in the right format for `_picks_to_heroes`.
    """
    out: List[Dict] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or ""
        steam_id = e.get("steam_id")
        hero_id = e.get("hero_id")
        dltv_id = hero_id  # original DLTV internal id
        if steam_id:
            try:
                meta = client.hero_by_steam_id(int(steam_id))
                if meta:
                    card = _hero_card(meta, int(steam_id))
                    if dltv_id is not None:
                        card["_dltv_id"] = int(dltv_id)
                    out.append(card)
                    continue
            except Exception:
                pass
        if hero_id:
            meta = client.hero_by_dltv_id(int(hero_id)) or client.hero_by_steam_id(int(hero_id))
            card = _hero_card(meta, int(hero_id))
            card["_dltv_id"] = int(hero_id)
            out.append(card)
        elif name:
            # Look up by name (DLTV's `title`).
            try:
                for h in client.get_heroes() or []:
                    if (h.get("title") or "").strip().lower() == name.strip().lower():
                        card = _hero_card(h, h.get("id"))
                        # We didn't get a dltv id from the input; keep
                        # the looked-up hero's id as a best effort.
                        card["_dltv_id"] = h.get("id")
                        out.append(card)
                        break
                else:
                    out.append({"id": None, "name": name, "image": None, "win_rate": None, "_dltv_id": None})
            except Exception:
                out.append({"id": None, "name": name, "image": None, "win_rate": None, "_dltv_id": None})
        else:
            out.append({"id": None, "name": "", "image": None, "win_rate": None, "_dltv_id": None})
    return out



def _picks_to_heroes(picks: Optional[List[Dict]], use_steam_id: bool = False):
    """Return (hero_meta_list, hero_card_list) from a maps[].*_picks list.

    When use_steam_id=True, picks[].hero_id is a Steam hero id (watchlist/live JSON path).
    Otherwise picks[].hero_id is a DLTV internal hero id (v1 API path).
    """
    metas, cards = [], []
    for p in sorted(picks or [], key=lambda x: x.get("order", 0)):
        hid = p.get("hero_id")
        meta = client.hero_by_steam_id(hid) if use_steam_id else client.hero_by_dltv_id(hid)
        metas.append(meta)
        cards.append(_hero_card(meta, hid))
    return metas, cards


def _bans_to_cards(bans: Optional[List[Dict]], use_steam_id: bool = False) -> List[Dict]:
    cards = []
    for b in sorted(bans or [], key=lambda x: x.get("order", 0)):
        hid = b.get("hero_id")
        meta = client.hero_by_steam_id(hid) if use_steam_id else client.hero_by_dltv_id(hid)
        cards.append(_hero_card(meta, hid))
    return cards


def _played_maps(series: Dict) -> List[Dict]:
    return [m for m in (series.get("maps") or []) if m.get("winner") and m.get("duration")]


def _active_map(series: Dict) -> Optional[Dict]:
    maps = series.get("maps") or []
    # prefer an in-progress map (has steam_id, no winner)
    for m in maps:
        if m.get("steam_id") and not m.get("winner"):
            return m
    return maps[-1] if maps else None


def _prematch_card(series: Dict, event_title: str) -> Dict:
    return {
        "stage": "prematch",
        "series_id": series.get("id"),
        "event_id": series.get("_scraper_event_id") or series.get("event_id"),
        "event": event_title,
        "bo": _bo_label(series.get("type")),
        "start_time": series.get("started_at") or series.get("liquipedia_date"),
        "team_a": client.normalize_team(series.get("first_team")),
        "team_b": client.normalize_team(series.get("second_team")),
    }


def _prematch_card_from_scraper(m: Dict) -> Dict:
    """Build a prematch card from the DLTV matches scraper output.

    Scraper provides: team_a/team_b (with name+logo), event, bo ("bo3"), start_time.
    """
    bo_str = (m.get("bo") or "bo?").lower()
    if bo_str.startswith("bo") and bo_str[2:].isdigit():
        bo_label = bo_str.upper()
    else:
        bo_label = "BO?"

    ta = m.get("team_a") or {}
    tb = m.get("team_b") or {}
    team_a = {"name": ta.get("name") or "TBD", "logo": ta.get("logo"), "tag": ta.get("tag"), "rank": ta.get("rank"), "id": ta.get("id")}
    team_b = {"name": tb.get("name") or "TBD", "logo": tb.get("logo"), "tag": tb.get("tag"), "rank": tb.get("rank"), "id": tb.get("id")}

    return {
        "stage": "prematch",
        "series_id": m.get("series_id"),
        "steam_id": m.get("steam_id"),
        "event_id": m.get("event_id"),
        "event": m.get("event") or "Scheduled match",
        "bo": bo_label,
        "stage_label": m.get("stage_label"),
        "start_time": m.get("start_time"),
        "team_a": team_a,
        "team_b": team_b,
        "is_tracked": bool(m.get("steam_id")),  # we already know its steam_id
    }


def _postmatch_prediction(series: Dict, is_watchlist: bool = False) -> Optional[Dict]:
    """Run the analyzer on the last played map and return a compact prediction
    summary, so post-match cards can show predicted vs. actual outcome.
    """
    maps = [m for m in (series.get("maps") or []) if m.get("duration")]
    if not maps:
        return None
    last_map = maps[-1]
    first_id = series.get("first_team_id")
    second_id = series.get("second_team_id")
    first = client.normalize_team(series.get("first_team"))
    second = client.normalize_team(series.get("second_team"))
    radiant_is_first = last_map.get("radiant_team_id") == first_id
    r_picks = last_map.get("radiant_picks") or []
    d_picks = last_map.get("dire_picks") or []
    heroes_a = _picks_to_heroes(r_picks if radiant_is_first else d_picks,
                                 use_steam_id=is_watchlist)[0]
    heroes_b = _picks_to_heroes(d_picks if radiant_is_first else r_picks,
                                 use_steam_id=is_watchlist)[0]
    try:
        engine = get_default_engine()
        pred = engine.analyze(first if radiant_is_first else second,
                              second if radiant_is_first else first,
                              heroes_a, heroes_b)
    except MLError:
        # ML subsystem failed (model missing, features broken, etc.).
        # Fall back to the caller — the per-card builder decides whether
        # to render a card without predictions or skip the series.
        return None
    winner_info = pred.get("winner") or {}
    return {
        "winner_team": winner_info.get("team"),
        "winner_probability": winner_info.get("probability"),
        "prob_radiant": winner_info.get("prob_radiant"),
        "confidence": pred.get("confidence"),
        "kills_total": (pred.get("kills") or {}).get("total"),
        "duration_min": pred.get("duration_min"),
    }


def _postmatch_card(series: Dict, event_title: str, is_watchlist: bool = False) -> Dict:
    first = client.normalize_team(series.get("first_team"))
    second = client.normalize_team(series.get("second_team"))
    first_id = series.get("first_team_id")
    second_id = series.get("second_team_id")

    played = _played_maps(series)
    tally = {first_id: 0, second_id: 0}
    game_details = []
    for i, m in enumerate(played, start=1):
        winner_id = m.get("radiant_team_id") if m.get("winner") == "radiant" else m.get("dire_team_id")
        if winner_id in tally:
            tally[winner_id] += 1
        winner_name = first["name"] if winner_id == first_id else second["name"]
        game_details.append({
            "game": i,
            "duration_min": round((m.get("duration") or 0) / 60.0, 1),
            "radiant_score": m.get("radiant_score"),
            "dire_score": m.get("dire_score"),
            "winner": winner_name,
        })

    score_a = tally.get(first_id, 0)
    score_b = tally.get(second_id, 0)
    winner_name = first["name"] if score_a >= score_b else second["name"]

    # --------------------------------------------------------------------- #
    # Per-game detailed view (actual stats + prediction + verdict)
    # --------------------------------------------------------------------- #
    games_detailed: List[Dict] = []
    for i, m in enumerate(played, start=1):
        radiant_is_first = m.get("radiant_team_id") == first_id
        r_picks = m.get("radiant_picks") or []
        d_picks = m.get("dire_picks") or []
        heroes_a = _picks_to_heroes(
            r_picks if radiant_is_first else d_picks,
            use_steam_id=is_watchlist,
        )[0]
        heroes_b = _picks_to_heroes(
            d_picks if radiant_is_first else r_picks,
            use_steam_id=is_watchlist,
        )[0]

        # Towers: decode bitmask if present (v1 API only). Watchlist path lacks
        # tower data, so we keep None and the UI shows "n/a".
        tower_r = decode_towers(m.get("tower_radiant"))
        tower_d = decode_towers(m.get("tower_dire"))
        towers_total = (tower_r + tower_d) if (tower_r is not None and tower_d is not None) else None

        duration_s = m.get("duration") or 0
        duration_min = round(duration_s / 60.0, 1) if isinstance(duration_s, (int, float)) else None
        radiant_score = m.get("radiant_score") or 0
        dire_score = m.get("dire_score") or 0
        kills_total = radiant_score + dire_score

        # Winner expressed as the team name (matches prediction format)
        raw_winner = m.get("winner")
        if raw_winner == "radiant":
            actual_winner_name = first["name"] if radiant_is_first else second["name"]
        elif raw_winner == "dire":
            actual_winner_name = second["name"] if radiant_is_first else first["name"]
        else:
            actual_winner_name = None

        # Team scores: just sum of kills
        team_a_total_kills = radiant_score if radiant_is_first else dire_score
        team_b_total_kills = dire_score if radiant_is_first else radiant_score

        actual = {
            "winner_team": actual_winner_name,
            "duration_min": duration_min,
            "kills_total": kills_total,
            "towers_total": towers_total,
            "fb_side": m.get("fb"),          # "radiant" | "dire" | None
            "f15_side": m.get("f10"),        # DLTV calls it f10 (= first 15 kills)
            "team_a_score": team_a_total_kills,  # Total kills for team A
            "team_b_score": team_b_total_kills,  # Total kills for team B
        }

        # team_a = first, team_b = second (always, for consistent verdict labels)
        pred_verdict = analyze_map_with_verdict(
            first, second, heroes_a, heroes_b, actual,
            engine=get_default_engine(),
        )

        games_detailed.append({
            "game": i,
            "started_at": m.get("started_at"),
            "duration_sec": duration_s or None,
            "duration_min": duration_min,
            "radiant_score": radiant_score,
            "dire_score": dire_score,
            "winner": actual_winner_name,
            "team_a_score": radiant_score if radiant_is_first else dire_score,
            "team_b_score": dire_score if radiant_is_first else radiant_score,
            "team_a_towers": (tower_r if radiant_is_first else tower_d),
            "team_b_towers": (tower_d if radiant_is_first else tower_r),
            "radiant_towers": tower_r,
            "dire_towers": tower_d,
            "fb_side": m.get("fb"),
            "f15_side": m.get("f10"),
            "prediction": pred_verdict.get("prediction"),
            "verdict": pred_verdict.get("verdict"),
        })

    return {
        "stage": "postmatch",
        "series_id": series.get("id"),
        "event": event_title,
        "event_id": series.get("_scraper_event_id") or series.get("event_id"),
        "bo": _bo_label(_series_bo_int(series) or series.get("_scraper_bo") or series.get("type")),
        "ended_at": series.get("ended_at"),
        "team_a": first,
        "team_b": second,
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner_name,
        "games": game_details,        # kept for backwards-compat (summary view)
        "games_detailed": games_detailed,
        "prediction": _postmatch_prediction(series, is_watchlist=is_watchlist),
    }


def _live_card(series: Dict, event_title: str) -> Dict:
    first = client.normalize_team(series.get("first_team"))
    second = client.normalize_team(series.get("second_team"))
    first_id = series.get("first_team_id")
    is_watchlist = bool(series.get("_watchlist"))

    m = _active_map(series) or {}

    # v0.3.20+: when the v1 API hides the in-progress series
    # (which it does for any live match) we don't have a real
    # `m["radiant_picks"]` to work with.  The Playwright-backed
    # `dltv_browser` writes a `match_state` entry to the cache
    # every 5s for every live row with a URL — overlay it on
    # top of the empty `m` so the card shows real picks and
    # score.  Falls back silently if the cache is empty (e.g.
    # chromium binary missing in the container) — the card then
    # just shows teams + event as before.
    #
    # v0.3.24: the `isinstance(series_id, int)` guard was too narrow.
    # Watchlist live matches carry a string id like "watch-8916245727"
    # (built in dltv_client._live_json_to_series) and the Steam-only
    # fallback path uses "steam-8916245727".  Both forms embed an int
    # series id — that's the dltv_browser cache key.  Parse it; if
    # extraction fails (truly unknown id shape) skip the overlay.
    series_id_raw = series.get("id")
    series_id: Optional[int] = None
    if isinstance(series_id_raw, int):
        series_id = series_id_raw
    elif isinstance(series_id_raw, str):
        # Try the "<prefix>-<int>" form first (watch-…, steam-…).
        for prefix in ("watch-", "steam-"):
            if series_id_raw.startswith(prefix):
                tail = series_id_raw[len(prefix):]
                if tail.isdigit():
                    series_id = int(tail)
                    break
        # Fall back to a bare-numeric string ("1234567").
        if series_id is None and series_id_raw.isdigit():
            series_id = int(series_id_raw)
    if series_id is not None and not (m.get("radiant_picks") or m.get("dire_picks")):
        try:
            from .dltv_browser import get_cached_match_state
            cached_state = get_cached_match_state(series_id) or {}
            ms_picks = cached_state.get("picks") or {}
            ms_bans = cached_state.get("bans") or {}
            # We don't have hero_ids from the DOM extraction, only
            # names.  _picks_to_heroes needs an int hero_id to
            # resolve via hero_by_dltv_id.  Try a name->hero_id
            # lookup; if that fails the card still shows the
            # names with a generic placeholder, which is strictly
            # better than an empty draft.
            r_pick_cards = _names_to_cards(ms_picks.get("radiant", []))
            d_pick_cards = _names_to_cards(ms_picks.get("dire", []))
            r_ban_cards = _names_to_cards(ms_bans.get("radiant", []))
            d_ban_cards = _names_to_cards(ms_bans.get("dire", []))
            if r_pick_cards or d_pick_cards:
                # Build a synthetic map so the rest of the
                # function uses the picked heroes directly.
                # v0.3.22: pass through the original DLTV id
                # (in `hero_id`) AND the steam id (in `_steam_id`).
                # `_picks_to_heroes` chooses which to use based on
                # `is_watchlist`.  Previously both fields were
                # populated from the same dltv id, which silently
                # broke the non-watchlist path (hero_by_dltv_id
                # was called with a steam id).
                def _entry(c: Dict, i: int) -> Dict:
                    return {
                        "hero_id": c.get("_dltv_id") or c.get("id"),
                        "order": i,
                        "_steam_id": c.get("id"),
                    }
                m = {
                    "radiant_picks": [_entry(c, i) for i, c in enumerate(r_pick_cards)],
                    "dire_picks":    [_entry(c, i) for i, c in enumerate(d_pick_cards)],
                    "radiant_bans":  [_entry(c, i) for i, c in enumerate(r_ban_cards)],
                    "dire_bans":     [_entry(c, i) for i, c in enumerate(d_ban_cards)],
                }
                # If we have a real score, overlay it too.
                if "radiant_score" in cached_state:
                    m["radiant_score"] = cached_state["radiant_score"]
                if "dire_score" in cached_state:
                    m["dire_score"] = cached_state["dire_score"]
        except Exception as exc:
            log.debug("match-state overlay failed for %s: %s", series_id, exc)

    # figure out which side each team is on this map
    radiant_is_first = m.get("radiant_team_id") == first_id
    radiant_team = first if radiant_is_first else second
    dire_team = second if radiant_is_first else first

    r_metas, r_cards = _picks_to_heroes(m.get("radiant_picks"), use_steam_id=is_watchlist)
    d_metas, d_cards = _picks_to_heroes(m.get("dire_picks"), use_steam_id=is_watchlist)

    predictions = get_default_engine().analyze(radiant_team, dire_team, r_metas, d_metas)
    engine_name = get_default_engine().name

    # current series score so far — must be computed BEFORE we record
    # the prediction (the dedup key uses game_no, and the verdict
    # logic later reads score_a / score_b from the card).
    played = _played_maps(series)
    if is_watchlist:
        # v1 API doesn't know about this series; use pre-populated scores from db
        score_a = series.get("_series_score_first", 0)
        score_b = series.get("_series_score_second", 0)
        # infer current game number from games already decided
        game_no = (score_a or 0) + (score_b or 0) + 1
    else:
        tally = {series.get("first_team_id"): 0, series.get("second_team_id"): 0}
        for pm in played:
            wid = pm.get("radiant_team_id") if pm.get("winner") == "radiant" else pm.get("dire_team_id")
            if wid in tally:
                tally[wid] += 1
        score_a = tally.get(series.get("first_team_id"), 0)
        score_b = tally.get(series.get("second_team_id"), 0)
        game_no = len(played) + 1

    # ------------------------------------------------------------------ #
    # Live accuracy tracking (v0.3.15+)
    # ------------------------------------------------------------------ #
    # We record exactly one prediction per (match_id, game_no).  The
    # board is rebuilt every 5 seconds; without dedup we'd log 12
    # identical rows per minute per live match.
    try:
        winner_info = predictions.get("winner") or {}
        prob_radiant = winner_info.get("prob_radiant")
        if prob_radiant is not None:
            # ML engine emits `team` as a team NAME; we store a SIDE
            # label so `_compare` in accuracy.py can match it to the
            # actual series winner (a team_id).  prob_radiant is the
            # model's confidence in the RADIANT side, so the probability
            # of our predicted side is just max(p, 1-p).
            predicted_side = (
                "team_a" if (prob_radiant > 0.5) == radiant_is_first
                else "team_b"
            )
            record_prediction(
                match_id=m.get("steam_id"),
                series_id=series.get("id"),
                predicted_winner=predicted_side,
                predicted_probability=max(prob_radiant, 1.0 - prob_radiant),
                engine=engine_name,
                extra={
                    "team_a_id": first_id,
                    "team_a_name": first.get("name"),
                    "team_b_id": series.get("second_team_id"),
                    "team_b_name": second.get("name"),
                    "game_no": game_no,
                    "radiant_is_first": radiant_is_first,
                },
            )
    except AccuracyError as exc:
        # Tracking is best-effort — a missing log dir or write race
        # must not break board rendering.
        log.debug("accuracy: record_prediction failed: %s", exc)

    return {
        "stage": "live",
        "series_id": series.get("id"),
        "match_id": m.get("steam_id"),
        "event_id": series.get("_scraper_event_id") or series.get("event_id"),
        "event": event_title,
        "bo": _bo_label(series.get("type")),
        "game_no": game_no,
        "series_score_a": score_a,
        "series_score_b": score_b,
        "radiant_team": radiant_team,
        "dire_team": dire_team,
        "live_score": {
            "radiant": m.get("radiant_score") or 0,
            "dire": m.get("dire_score") or 0,
        },
        "draft": {
            "radiant_picks": r_cards,
            "dire_picks": d_cards,
            "radiant_bans": _bans_to_cards(m.get("radiant_bans"), use_steam_id=is_watchlist),
            "dire_bans": _bans_to_cards(m.get("dire_bans"), use_steam_id=is_watchlist),
        },
        "predictions": predictions,
        # v0.3.24: surface the "synthesised from Steam raw, no DLTV coverage"
        # marker so the server-side /api/board filter can drop these
        # by default — they were polluting the board with 50+ Chinese
        # amateur league matches.  The flag is set only on the
        # `_steam_game_to_series` fallback path in dltv_client.py
        # (line ~485).
        "_steam_only": bool(series.get("_steam_only")),
        "is_watchlist": is_watchlist,
    }


def build_board(event_ids: List[int], watch_ids: Optional[List[int]] = None) -> Dict:
    # ensure hero index is warm
    client.get_heroes()
    all_events = {e["id"]: e.get("title", f"Event {e['id']}") for e in client.get_events()}

    # Remember the user's explicit filter (may be empty for an
    # unfiltered board).  We use this to decide whether to apply the
    # `allowed_events` filter to discovery cards — the auto-populated
    # league list (below) shouldn't tighten the filter, otherwise
    # the user's unfiltered "show me everything" board hides matches
    # from any league that didn't make our top-N cold-cache cap.
    user_has_filter = bool(event_ids)

    # When no filter is provided, auto-include all currently-active leagues so
    # that the unfiltered board still surfaces post-match cards (v1 API path)
    # and live/prematch discovery data for every league the user can pick.
    if not event_ids:
        try:
            active = leagues_with_status()
            event_ids = [int(l["id"]) for l in active if l.get("id")]
        except DotaAnalystError as exc:
            # Discovery / DLTV down — we just don't auto-populate event_ids
            # and the user gets the empty board.  A bug in leagues_with_status
            # would still surface.
            log.warning("auto-active fallback failed: %s", exc, exc_info=True)
            event_ids = []

    # v0.3.15+: cap the league list so a cold DLTV cache doesn't stall
    # `build_board` for >30s (which would 504 the request).  The
    # publisher rebuilds every 5s, so even a narrow pick surfaces
    # the rest of the world within a minute or two.
    if len(event_ids) > MAX_LEAGUES_PER_BOARD:
        log.info(
            "build_board: capping %d leagues to top %d for cold-cache latency",
            len(event_ids), MAX_LEAGUES_PER_BOARD,
        )
        event_ids = event_ids[:MAX_LEAGUES_PER_BOARD]

    events = all_events
    allowed_events = set(int(x) for x in (event_ids or []))
    # Only treat the user-supplied selection as a filter; the
    # auto-populated cap list must NOT be used to drop discovery
    # cards or the unfiltered board will hide minor-league live
    # matches that didn't make the top-N cut.
    has_filter = user_has_filter

    prematch, live, postmatch = [], [], []

    # v1 series_ids we already cover (for dedup + filter fallback)
    v1_series_ids: set = set()

    # --- 1. v1 API series for user-selected leagues (backward-compat) ---------
    # v0.3.15+: fetch per-league series in parallel so a 12-league board
    # doesn't take 12 × 12s = 144s on cold cache. 5 workers is enough
    # to bound a 12-league walk under ~30s without saturating the
    # DLTV origin (we cap at 5 concurrent HTTPs).
    def _fetch_one(eid: int) -> List[Dict]:
        try:
            return list(client.get_series(eid) or [])
        except DotaAnalystError as exc:
            log.warning("get_series(%s) failed: %s", eid, exc, exc_info=False)
            return []

    from concurrent.futures import ThreadPoolExecutor
    if event_ids:
        with ThreadPoolExecutor(max_workers=SERIES_FETCH_WORKERS,
                                thread_name_prefix="series-fetcher") as pool:
            series_lists = list(pool.map(_fetch_one, event_ids))
    else:
        series_lists = []

    for eid, series_list in zip(event_ids, series_lists):
        title = events.get(eid, f"Event {eid}")
        for series in series_list:
            stage = client.classify_stage(series)
            v1_series_ids.add(series.get("id"))
            # stamp event_id on the synthetic series so _live_card/_prematch_card see it
            series["_scraper_event_id"] = series.get("_scraper_event_id") or eid
            try:
                if stage == "postmatch":
                    postmatch.append(_postmatch_card(series, title, is_watchlist=False))
                elif stage == "live":
                    live.append(_live_card(series, title))
                else:
                    prematch.append(_prematch_card(series, title))
            except Exception as exc:  # never let one bad series break the board
                log.warning("skip series %s (%s): %s", series.get('id'), stage, exc, exc_info=True)

    # --- 2. Discovery: scraper (dltv.org/matches) + Steam live games --------
    try:
        from .discovery import discover, tracker as _discovery_tracker
        live_series, prematch_series = discover()
    except DotaAnalystError as exc:
        log.warning("discovery failed: %s", exc, exc_info=True)
        live_series, prematch_series = [], []
        _discovery_tracker = None

    # Dedup keys for v1 series already added
    v1_prematch_keys = {
        (c.get("start_time"), (c.get("team_a") or {}).get("name"))
        for c in prematch
    }
    covered_steam = {c.get("match_id") for c in live if c.get("match_id")}

    # Add scraper prematch (skip if already in v1 by start_time + team_a name)
    for m in prematch_series:
        key = (m.get("start_time"), (m.get("team_a") or {}).get("name"))
        if key in v1_prematch_keys:
            continue
        # Filter: only drop scraper cards whose league is known to be
        # *outside* the user's selected set.  Cards with an unknown
        # event_id (leagues not covered by /api/v1/events — minor
        # tournaments, qualifiers) pass through; the user can always
        # pin them via the watchlist if needed.
        m_eid = m.get("event_id")
        if has_filter and m_eid is not None and int(m_eid) not in allowed_events:
            continue
        try:
            prematch.append(_prematch_card_from_scraper(m))
        except Exception as exc:
            log.warning("skip prematch scraper %s: %s", m.get('series_id'), exc, exc_info=True)

    # Add discovery live (synthetic series from /live/{id}.json)
    for ws in live_series:
        maps = ws.get("maps") or []
        steam_id = maps[0].get("steam_id") if maps else None
        if steam_id and steam_id in covered_steam:
            continue
        # Resolve event_id for filtering + title
        ws_eid = ws.get("_scraper_event_id") or ws.get("event_id")
        ws_series_id = ws.get("id")
        # If no event_id yet, check whether the synthetic id matches a v1 series we already know
        if not ws_eid and ws_series_id and ws_series_id in v1_series_ids:
            # already covered by v1 (though we also dedup by steam_id — defensive)
            pass
        # Build title: scraper name -> v1 events map -> steam_event mapping -> league id
        title = ws.get("_scraper_event")
        if not title and ws_eid:
            title = events.get(ws_eid)
        if not title and _discovery_tracker and ws.get("_steam_league_id"):
            mapped = _discovery_tracker.steam_event(ws.get("_steam_league_id"))
            if mapped:
                title = mapped[1]
                ws["_scraper_event_id"] = ws.get("_scraper_event_id") or mapped[0]
                ws_eid = ws_eid or mapped[0]
        if not title and ws.get("_steam_league_id"):
            title = f"Steam league {ws['_steam_league_id']}"
        title = title or "Live match"
        # Filter live cards by selected leagues.  When the user has
        # narrowed the board to a specific set of leagues, we apply
        # the filter strictly — including matches with no resolvable
        # event_id (steam-only / minor tournaments).  Earlier we let
        # those pass through "in case they're a minor league the
        # user cares about", but in practice that just leaks the
        # user's selected set: a Russian Esports live match would
        # show up next to the EPL matches the user actually asked
        # for.  Watchlist cards are unaffected — they have their own
        # filter logic that uses `allowed_events` symmetrically.
        if has_filter:
            if ws_eid is None or int(ws_eid) not in allowed_events:
                continue
        stage = client.classify_stage(ws)
        try:
            if stage == "live":
                live.append(_live_card(ws, title))
            elif stage == "postmatch":
                postmatch.append(_postmatch_card(ws, title, is_watchlist=bool(ws.get("_watchlist"))))
            else:
                card = _prematch_card_from_scraper({
                    "series_id": ws.get("id"),
                    "steam_id": steam_id,
                    "event": title,
                    "event_id": ws.get("_scraper_event_id") or ws_eid,
                    "bo": ws.get("_scraper_bo"),
                    "team_a": ws.get("first_team"),
                    "team_b": ws.get("second_team"),
                    "start_time": ws.get("started_at"),
                })
                prematch.append(card)
        except Exception as exc:
            log.warning("skip discovery live %s (%s): %s", ws.get('id'), stage, exc, exc_info=True)

    # --- 3. Manual watchlist (legacy, for explicit user-pinned steam_ids) ----
    if watch_ids:
        from .dltv_client import fetch_watchlist_series
        covered_steam = {c.get("match_id") for c in live if c.get("match_id")}
        try:
            for ws in fetch_watchlist_series(watch_ids):
                maps = ws.get("maps") or []
                steam_id = maps[0].get("steam_id") if maps else None
                if steam_id in covered_steam:
                    continue
                # Apply filter: watchlist is always "user-pinned" so keep it
                # unless user selected leagues that explicitly exclude this match's event
                ws_eid = ws.get("event_id")
                if has_filter and ws_eid is not None and int(ws_eid) not in allowed_events:
                    continue
                stage = client.classify_stage(ws)
                event_id = ws.get("event_id")
                title = events.get(event_id, "Отслеживаемый матч")
                try:
                    if stage == "live":
                        live.append(_live_card(ws, title))
                    elif stage == "postmatch":
                        postmatch.append(_postmatch_card(ws, title, is_watchlist=True))
                    else:
                        prematch.append(_prematch_card(ws, title))
                except Exception as exc:
                    log.warning("skip watch %s (%s): %s", ws.get('id'), stage, exc, exc_info=True)
        except DotaAnalystError as exc:
            log.warning("watchlist fetch failed: %s", exc, exc_info=True)

    # sort: prematch by soonest start, postmatch by most recent end
    prematch.sort(key=lambda c: c.get("start_time") or "")
    postmatch.sort(key=lambda c: c.get("ended_at") or "", reverse=True)

    return {"prematch": prematch, "live": live, "postmatch": postmatch}
