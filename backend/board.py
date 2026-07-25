"""
Board assembly: turns DLTV series into Kanban cards
(prematch / live / postmatch) for the selected leagues (events).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Dict, List, Optional, Set

from .analysis import analyze, analyze_map_with_verdict, analyze_prematch_series, decode_towers
from .dltv_client import client, _parse_dt
from .liquipedia_client import is_tier_1_or_2
from .learning_pipeline import learning_pipeline
from .ml_predictor import draft_context, predict_prematch_winner, predict_winner, team_form_context
from .prediction_audit import prediction_audit
from .series_cache import series_map_cache

BO_LABELS = {1: "BO1", 2: "BO2", 3: "BO3", 5: "BO5"}
LIVE_GRACE_SECONDS = 45 * 60
_LIVE_CARD_CACHE: Dict[tuple, tuple[float, Dict]] = {}


def _series_key(card: Dict) -> tuple:
    """Stable fallback identity for cross-source duplicates."""
    def norm(value: str) -> str:
        return "".join(char for char in (value or "").lower() if char.isalnum())
    first = card.get("team_a") or card.get("radiant_team") or {}
    second = card.get("team_b") or card.get("dire_team") or {}
    return (norm(card.get("event") or ""), tuple(sorted((norm(first.get("name")), norm(second.get("name"))))))


def _team_pair_key(first: Dict, second: Dict) -> tuple:
    def norm(value: str) -> str:
        return "".join(char for char in (value or "").lower() if char.isalnum())
    return tuple(sorted((norm(first.get("name") or first.get("title") or ""), norm(second.get("name") or second.get("title") or ""))))


def _remember_live_card(card: Dict) -> None:
    _LIVE_CARD_CACHE[_series_key(card)] = (time.time(), card)


def _series_outlook(map_probability_a: float, bo: Optional[int], score_a: int, score_b: int) -> Dict:
    """Calculate series win chances from current score and independent map strength."""
    if bo not in {2, 3, 5}:
        return {}
    if bo == 2:
        # BO2 can finish tied: useful signal is the chance to win the remaining map.
        return {"team_a_probability": round(map_probability_a * 100), "mode": "next_map"}
    needed = bo // 2 + 1

    def chance(a: int, b: int) -> float:
        if a >= needed:
            return 1.0
        if b >= needed:
            return 0.0
        return map_probability_a * chance(a + 1, b) + (1 - map_probability_a) * chance(a, b + 1)

    return {"team_a_probability": round(chance(score_a, score_b) * 100), "mode": "series"}


def classify_event_status(event_id: int) -> str:
    """Return 'live' if the event has any live/prematch match known to discovery,
    else None (the league is filtered out of the selector).

    Legacy helper kept for callers that still distinguish; leagues_with_status()
    no longer surfaces 'finished' or 'upcoming' — the UI only lists active leagues.
    """
    try:
        from .discovery import discover
        live_series, prematch_series = discover()
    except Exception:
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
    except Exception:
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
        if not eid or int(eid) not in active_ids or not is_tier_1_or_2(e.get("title") or ""):
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
    team_a, team_b = client.normalize_team(series.get("first_team")), client.normalize_team(series.get("second_team"))
    bo = _bo_label(series.get("type"))
    return {
        "stage": "prematch",
        "series_id": series.get("id"),
        "event_id": series.get("_scraper_event_id") or series.get("event_id"),
        "event": event_title,
        "bo": bo,
        "start_time": series.get("started_at") or series.get("liquipedia_date"),
        "team_a": team_a,
        "team_b": team_b,
        "predictions": _prematch_predictions(team_a, team_b, bo),
        "form_context": team_form_context(team_a, team_b),
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
        "predictions": _prematch_predictions(team_a, team_b, bo_label),
        "form_context": team_form_context(team_a, team_b),
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
        pred = analyze(first if radiant_is_first else second,
                       second if radiant_is_first else first,
                       heroes_a, heroes_b)
    except Exception:
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
    learning_pipeline.queue_completed_maps(series.get("maps") or [])
    first = client.normalize_team(series.get("first_team"))
    second = client.normalize_team(series.get("second_team"))
    series_map_cache.clear(event_title, first["name"], second["name"])
    first_id = series.get("first_team_id")
    second_id = series.get("second_team_id")

    played = series.get("_completed_maps") or _played_maps(series)
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
        )
        # Winner accuracy must be assessed against the prediction actually
        # shown during live, never against a post-match recomputation.
        recorded = prediction_audit.get(m.get("steam_id"))
        if recorded:
            pred_verdict["prediction"]["winner"] = {
                "team": recorded.get("predicted_team"),
                "probability": int(round(float(recorded.get("probability", 0)) * 100)),
            }
            pred_verdict["verdict"]["winner"] = recorded.get("correct")

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
        prediction_audit.settle(m.get("steam_id"), actual_winner_name)

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
        "map_ids": [game_map.get("steam_id") for game_map in played if game_map.get("steam_id")],
        "prediction": _postmatch_prediction(series, is_watchlist=is_watchlist),
    }


def _live_card(series: Dict, event_title: str) -> Dict:
    first = client.normalize_team(series.get("first_team"))
    second = client.normalize_team(series.get("second_team"))
    first_id = series.get("first_team_id")
    is_watchlist = bool(series.get("_watchlist"))

    candidate_map = _active_map(series)
    waiting_for_next_map = not candidate_map or bool(candidate_map.get("winner"))
    m = {} if waiting_for_next_map else candidate_map
    # figure out which side each team is on this map
    radiant_is_first = waiting_for_next_map or m.get("radiant_team_id") == first_id
    radiant_team = first if radiant_is_first else second
    dire_team = second if radiant_is_first else first

    r_metas, r_cards = _picks_to_heroes(m.get("radiant_picks"), use_steam_id=is_watchlist)
    d_metas, d_cards = _picks_to_heroes(m.get("dire_picks"), use_steam_id=is_watchlist)

    predictions = {} if waiting_for_next_map else analyze(radiant_team, dire_team, r_metas, d_metas)
    ml_prediction = None if waiting_for_next_map else predict_winner(
        radiant_team,
        dire_team,
        [hero.get("steam_id") for hero in r_metas if hero and hero.get("steam_id") is not None],
        [hero.get("steam_id") for hero in d_metas if hero and hero.get("steam_id") is not None],
    )
    if ml_prediction:
        predictions["winner"] = {
            "team": ml_prediction["winner"],
            "probability": ml_prediction["probability"],
            "prob_radiant": ml_prediction["prob_radiant"],
        }
        predictions["ml_winner"] = ml_prediction
    radiant_ids = [hero.get("steam_id") for hero in r_metas if hero and hero.get("steam_id") is not None]
    dire_ids = [hero.get("steam_id") for hero in d_metas if hero and hero.get("steam_id") is not None]
    learning_pipeline.observe_live_map(m.get("steam_id"), {
        "event": event_title,
        "radiant_team": radiant_team,
        "dire_team": dire_team,
        "radiant_picks": r_cards,
        "dire_picks": d_cards,
        "prediction": predictions,
    })
    prediction_audit.record_live(m.get("steam_id"), {
        "event": event_title,
        "radiant_team": radiant_team,
        "dire_team": dire_team,
        "prediction": predictions,
    })

    # current series score so far
    played = series.get("_completed_maps") or _played_maps(series)
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

    completed_maps = []
    for index, played_map in enumerate(played, start=1):
        raw_winner = played_map.get("winner")
        winner_id = played_map.get("radiant_team_id") if raw_winner == "radiant" else played_map.get("dire_team_id")
        winner_name = first["name"] if winner_id == first_id else second["name"]
        duration = played_map.get("duration")
        completed_maps.append({
            "game": index,
            "map_id": played_map.get("steam_id"),
            "winner": winner_name,
            "radiant_score": played_map.get("radiant_score") or 0,
            "dire_score": played_map.get("dire_score") or 0,
            "duration_sec": duration if isinstance(duration, (int, float)) else None,
        })
    completed_maps = series_map_cache.merge(event_title, first["name"], second["name"], completed_maps)

    bo = _series_bo_int(series)
    map_probability_first = (predictions.get("winner") or {}).get("prob_radiant", 50) / 100.0
    if not radiant_is_first:
        map_probability_first = 1.0 - map_probability_first
    series_outlook = _series_outlook(map_probability_first, bo, score_a, score_b)
    if series_outlook:
        series_outlook["team_a"] = first["name"]
        series_outlook["team_b"] = second["name"]
        series_outlook["favourite"] = first["name"] if series_outlook["team_a_probability"] >= 50 else second["name"]
        series_outlook["probability"] = max(series_outlook["team_a_probability"], 100 - series_outlook["team_a_probability"])
    winner_probability = (predictions.get("winner") or {}).get("probability", 50)
    signal = "skip" if winner_probability < 55 else ("watch" if winner_probability < 62 else "strong")

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
        "waiting_for_next_map": waiting_for_next_map,
        "completed_maps": completed_maps,
        "draft": {
            "radiant_picks": r_cards,
            "dire_picks": d_cards,
            "radiant_bans": _bans_to_cards(m.get("radiant_bans"), use_steam_id=is_watchlist),
            "dire_bans": _bans_to_cards(m.get("dire_bans"), use_steam_id=is_watchlist),
        },
        "predictions": predictions,
        "draft_context": draft_context(radiant_ids, dire_ids),
        "series_outlook": series_outlook,
        "signal": signal,
        "is_watchlist": is_watchlist,
    }


def build_board(event_ids: List[int], watch_ids: Optional[List[int]] = None) -> Dict:
    # ensure hero index is warm
    client.get_heroes()
    raw_events = client.get_events()
    allowed_tier_events = {
        int(event["id"])
        for event in raw_events
        if event.get("id") and is_tier_1_or_2(event.get("title") or "")
    }
    all_events = {
        event["id"]: event.get("title", f"Event {event['id']}")
        for event in raw_events
        if int(event["id"]) in allowed_tier_events
    }
    event_ids = [event_id for event_id in event_ids if event_id in allowed_tier_events]

    # When no filter is provided, auto-include all currently-active leagues so
    # that the unfiltered board still surfaces post-match cards (v1 API path)
    # and live/prematch discovery data for every league the user can pick.
    if not event_ids:
        try:
            active = leagues_with_status()
            event_ids = [int(l["id"]) for l in active if l.get("id")]
        except Exception as exc:
            print(f"[board] auto-active fallback failed: {exc}")
            event_ids = []
        # Discovery can briefly have no live map between games in a BO series.
        # Keep leagues marked active by DLTV in scope so their unfinished v1
        # series remain visible during that gap.
        if not event_ids:
            event_ids = [
                int(event["id"]) for event in raw_events
                if event.get("id") and event.get("is_active") and int(event["id"]) in allowed_tier_events
            ]

    events = all_events
    allowed_events = set(int(x) for x in (event_ids or []))
    has_filter = bool(allowed_events)

    prematch, live, postmatch = [], [], []

    # v1 series_ids we already cover (for dedup + filter fallback)
    v1_series_ids: set = set()
    v1_completed_by_pair: Dict[tuple, List[Dict]] = {}

    # --- 1. v1 API series for user-selected leagues (backward-compat) ---------
    for eid in event_ids:
        title = events.get(eid, f"Event {eid}")
        for series in client.get_series(eid):
            stage = client.classify_stage(series)
            v1_series_ids.add(series.get("id"))
            # stamp event_id on the synthetic series so _live_card/_prematch_card see it
            series["_scraper_event_id"] = series.get("_scraper_event_id") or eid
            completed = _played_maps(series)
            if completed:
                v1_completed_by_pair[_team_pair_key(series.get("first_team") or {}, series.get("second_team") or {})] = completed
            try:
                if stage == "postmatch":
                    postmatch.append(_postmatch_card(series, title, is_watchlist=False))
                elif stage == "live":
                    live.append(_live_card(series, title))
                else:
                    prematch.append(_prematch_card(series, title))
            except Exception as exc:  # never let one bad series break the board
                print(f"[board] skip series {series.get('id')} ({stage}): {exc}")

    # --- 2. Discovery: scraper (dltv.org/matches) + Steam live games --------
    try:
        from .discovery import discover, tracker as _discovery_tracker
        live_series, prematch_series = discover()
    except Exception as exc:
        print(f"[board] discovery failed: {exc}")
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
        # Filter: if user selected leagues, only keep cards matching one of them.
        # Cards without event_id belong to leagues outside /api/v1/events and
        # cannot be mapped — drop them under filter to avoid noise.
        m_eid = m.get("event_id")
        if has_filter and (m_eid is None or int(m_eid) not in allowed_events):
            continue
        try:
            prematch.append(_prematch_card_from_scraper(m))
        except Exception as exc:
            print(f"[board] skip prematch scraper {m.get('series_id')}: {exc}")

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
        # Filter live cards by selected leagues. For Steam-only matches whose
        # steam_league_id hasn't been cross-referenced with DLTV, event_id stays
        # None — drop them under filter to avoid mixing unrelated leagues.
        if has_filter and (ws_eid is None or int(ws_eid) not in allowed_events):
            continue
        historical_maps = v1_completed_by_pair.get(
            _team_pair_key(ws.get("first_team") or {}, ws.get("second_team") or {})
        )
        if historical_maps:
            ws["_completed_maps"] = historical_maps
        stage = client.classify_stage(ws)
        try:
            if stage == "live":
                live.append(_live_card(ws, title))
                if steam_id:
                    covered_steam.add(steam_id)
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
            print(f"[board] skip discovery live {ws.get('id')} ({stage}): {exc}")

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
                    print(f"[board] skip watch {ws.get('id')} ({stage}): {exc}")
        except Exception as exc:
            print(f"[board] watchlist fetch failed: {exc}")

    # sort: prematch by soonest start, postmatch by most recent end
    prematch.sort(key=lambda c: c.get("start_time") or "")
    postmatch.sort(key=lambda c: c.get("ended_at") or "", reverse=True)

    # The static live endpoint can remain stale after the v1 series endpoint
    # has already provided all completed maps.  A completed series is the
    # source of truth and must never be shown in both columns.
    completed_map_ids = {match_id for card in postmatch for match_id in card.get("map_ids", [])}
    live = [card for card in live if card.get("match_id") not in completed_map_ids]

    # Keep an unfinished BO series visible during short upstream gaps between
    # maps.  A real post-match card always wins and clears the cached copy.
    now = time.time()
    postmatch_keys = {_series_key(card) for card in postmatch}
    live_keys = {_series_key(card) for card in live}
    for key, (seen_at, cached_card) in list(_LIVE_CARD_CACHE.items()):
        if key in postmatch_keys or now - seen_at > LIVE_GRACE_SECONDS:
            _LIVE_CARD_CACHE.pop(key, None)
            continue
        if key not in live_keys:
            live.append({**cached_card, "waiting_for_next_map": True, "match_id": None})
    for card in live:
        _remember_live_card(card)

    return {"prematch": prematch, "live": live, "postmatch": postmatch}


def _prematch_predictions(team_a: Dict, team_b: Dict, bo: str) -> Dict:
    """Use ML historical Elo/form when possible, heuristic only as fallback."""
    if team_a.get("name") == "TBD" or team_b.get("name") == "TBD":
        return {}
    prediction = analyze_prematch_series(team_a, team_b, bo)
    ml_prediction = predict_prematch_winner(team_a, team_b)
    if not ml_prediction:
        return prediction
    prediction["winner"] = {"team": ml_prediction["team"], "probability": ml_prediction["probability"]}
    prediction["source"] = ml_prediction["source"]
    prediction["series_score"]["favourite"] = ml_prediction["team"]
    return prediction
