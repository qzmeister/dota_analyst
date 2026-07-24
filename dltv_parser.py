"""
DLTV Match Parser

Парсит JSON-данные матча с DLTV (https://dltv.org/live/{match_id}.json).
Один и тот же endpoint отдаёт данные и для live, и для завершённых матчей.

Ключевое отличие:
  - live:      is_deactivated == False, нет winner/full_stats
  - completed: is_deactivated == True,  есть winner, full_stats, first_ten

Соглашения DLTV:
  - team: 0 = radiant, 1 = dire (как в Steam API)
  - team_stats: "51 | 65%" -> 51 матч сыгран, 65% winrate; "0 | -" -> нет данных
  - kda:  "kills / deaths / assists"
  - lh:   "last_hits / denies"
  - gpm:  "gpm / xpm"
  - db.first_team / db.second_team имеют флаг is_radiant
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Вспомогательные парсеры составных строк
# --------------------------------------------------------------------------- #

def parse_team_stats(value: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Парсит строку статистики команды по герою.
    "51 | 65%" -> {"matches": 51, "win_rate": 0.65}
    "0 | -"    -> {"matches": 0,  "win_rate": None}
    """
    result: Dict[str, Optional[float]] = {"matches": 0, "win_rate": None}
    if not value or not isinstance(value, str):
        return result

    parts = value.split("|")
    if len(parts) != 2:
        return result

    # matches
    matches_str = parts[0].strip()
    try:
        result["matches"] = int(matches_str)
    except (ValueError, TypeError):
        result["matches"] = 0

    # win rate
    wr_str = parts[1].strip().rstrip("%").strip()
    if wr_str and wr_str != "-":
        try:
            result["win_rate"] = round(float(wr_str) / 100.0, 4)
        except (ValueError, TypeError):
            result["win_rate"] = None

    return result


def parse_slashed_ints(value: Optional[str], count: int) -> List[Optional[int]]:
    """
    Парсит строку вида "3 / 2 / 2" в список int.
    count — ожидаемое число элементов (для padding при неполных данных).
    """
    out: List[Optional[int]] = [None] * count
    if not value or not isinstance(value, str):
        return out

    parts = [p.strip() for p in value.split("/")]
    for i in range(min(count, len(parts))):
        try:
            out[i] = int(parts[i])
        except (ValueError, TypeError):
            out[i] = None
    return out


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class HeroPick:
    """Один пик героя с командной статистикой."""
    hero_id: int
    hero_name: Optional[str] = None
    hero_slug: Optional[str] = None
    team_matches: int = 0          # сколько раз команда играла этого героя
    team_win_rate: Optional[float] = None  # winrate команды на этом герое


@dataclass
class PlayerStats:
    """Полная статистика игрока (только для завершённых матчей)."""
    account_id: Optional[int] = None
    player_name: Optional[str] = None
    player_rank: Optional[int] = None
    country: Optional[str] = None
    hero_id: Optional[int] = None
    hero_name: Optional[str] = None
    level: Optional[int] = None
    kills: Optional[int] = None
    deaths: Optional[int] = None
    assists: Optional[int] = None
    last_hits: Optional[int] = None
    denies: Optional[int] = None
    gpm: Optional[int] = None
    xpm: Optional[int] = None
    net_worth: Optional[int] = None
    gold: Optional[int] = None
    items: List[int] = field(default_factory=list)


@dataclass
class TeamDraft:
    """Драфт одной команды."""
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    team_tag: Optional[str] = None
    team_rank: Optional[int] = None
    is_radiant: bool = False
    picks: List[HeroPick] = field(default_factory=list)
    bans: List[int] = field(default_factory=list)


@dataclass
class TimeSeries:
    """Time-series графики матча (по игровым таймингам)."""
    game_times: List[int] = field(default_factory=list)
    net_worth: List[int] = field(default_factory=list)       # radiant lead (net worth diff)
    radiant_scores: List[int] = field(default_factory=list)
    dire_scores: List[int] = field(default_factory=list)
    radiant_kills: List[int] = field(default_factory=list)
    dire_kills: List[int] = field(default_factory=list)


@dataclass
class MatchMeta:
    """Метаданные матча (турнир, серия, стримы)."""
    league_id: Optional[int] = None
    series_id: Optional[int] = None
    event_id: Optional[int] = None
    series_slug: Optional[str] = None
    series_type: Optional[int] = None       # 1 = BO3, 3 = BO5
    started_at: Optional[str] = None
    radiant_series_wins: int = 0
    dire_series_wins: int = 0
    streams: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedMatch:
    """Нормализованное представление матча DLTV."""
    match_id: int
    is_completed: bool
    is_picks_ended: bool
    game_time: int = 0
    winner: Optional[str] = None            # "radiant" / "dire" / None
    radiant_score: int = 0
    dire_score: int = 0
    radiant_lead: int = 0                   # net worth advantage
    first_blood: Optional[str] = None
    first_ten: Optional[str] = None

    radiant_draft: Optional[TeamDraft] = None
    dire_draft: Optional[TeamDraft] = None

    radiant_players: List[PlayerStats] = field(default_factory=list)
    dire_players: List[PlayerStats] = field(default_factory=list)

    charts: Optional[TimeSeries] = None
    meta: Optional[MatchMeta] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Основной парсер
# --------------------------------------------------------------------------- #

class DLTVParser:
    """Парсит сырой JSON DLTV в структуру ParsedMatch."""

    @staticmethod
    def parse(raw: Dict[str, Any]) -> ParsedMatch:
        match_id = raw.get("match_id")
        is_completed = bool(raw.get("is_deactivated", False))
        is_picks_ended = bool(raw.get("is_picks_ended", False))

        match = ParsedMatch(
            match_id=match_id,
            is_completed=is_completed,
            is_picks_ended=is_picks_ended,
            game_time=raw.get("game_time", 0) or 0,
            winner=raw.get("winner"),
            radiant_score=raw.get("radiant_score", 0) or 0,
            dire_score=raw.get("dire_score", 0) or 0,
            radiant_lead=raw.get("radiant_lead", 0) or 0,
            first_blood=raw.get("first_blood"),
            first_ten=raw.get("first_ten"),
        )

        db = raw.get("db", {}) or {}

        # --- Драфты команд ---
        match.radiant_draft, match.dire_draft = DLTVParser._parse_drafts(db)

        # --- Полная статистика игроков (только completed) ---
        full_stats = raw.get("full_stats", {}) or {}
        match.radiant_players = DLTVParser._parse_players(full_stats.get("radiant", {}))
        match.dire_players = DLTVParser._parse_players(full_stats.get("dire", {}))

        # --- Time-series ---
        match.charts = DLTVParser._parse_charts(raw.get("charts", {}) or {})

        # --- Метаданные ---
        match.meta = DLTVParser._parse_meta(raw, db)

        return match

    # ------------------------------------------------------------------ #
    # Приватные методы парсинга секций
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hero_pick(pick_raw: Dict[str, Any]) -> HeroPick:
        hero = pick_raw.get("hero", {}) or {}
        stats = parse_team_stats(pick_raw.get("team_stats"))
        return HeroPick(
            hero_id=pick_raw.get("hero_id"),
            hero_name=hero.get("title"),
            hero_slug=hero.get("slug"),
            team_matches=stats["matches"],
            team_win_rate=stats["win_rate"],
        )

    @staticmethod
    def _parse_team_draft(team_raw: Dict[str, Any]) -> TeamDraft:
        picks = [
            DLTVParser._parse_hero_pick(p)
            for p in (team_raw.get("picks", []) or [])
        ]
        bans = [
            b.get("hero_id")
            for b in (team_raw.get("bans", []) or [])
            if b.get("hero_id") is not None
        ]
        return TeamDraft(
            team_id=team_raw.get("id"),
            team_name=team_raw.get("title"),
            team_tag=team_raw.get("tag"),
            team_rank=team_raw.get("rank"),
            is_radiant=bool(team_raw.get("is_radiant", False)),
            picks=picks,
            bans=bans,
        )

    @staticmethod
    def _parse_drafts(db: Dict[str, Any]):
        """
        Возвращает (radiant_draft, dire_draft).
        В db есть first_team / second_team с флагом is_radiant.
        """
        first = db.get("first_team", {}) or {}
        second = db.get("second_team", {}) or {}

        radiant_draft = None
        dire_draft = None

        for team_raw in (first, second):
            if not team_raw:
                continue
            draft = DLTVParser._parse_team_draft(team_raw)
            if draft.is_radiant:
                radiant_draft = draft
            else:
                dire_draft = draft

        return radiant_draft, dire_draft

    @staticmethod
    def _parse_players(team_stats_raw: Dict[str, Any]) -> List[PlayerStats]:
        """Парсит full_stats.radiant или full_stats.dire."""
        players_raw = team_stats_raw.get("players", []) or []
        result: List[PlayerStats] = []

        for entry in players_raw:
            player = entry.get("player", {}) or {}
            hero = entry.get("hero", {}) or {}
            country = entry.get("country", {}) or {}

            kda = parse_slashed_ints(entry.get("kda"), 3)
            lh = parse_slashed_ints(entry.get("lh"), 2)
            gpm = parse_slashed_ints(entry.get("gpm"), 2)

            items = [i for i in (entry.get("items", []) or []) if isinstance(i, int) and i > 0]

            result.append(PlayerStats(
                account_id=player.get("steam_id"),
                player_name=player.get("title"),
                player_rank=player.get("rank"),
                country=country.get("title"),
                hero_id=hero.get("steam_id"),   # steam_id = Dota hero_id
                hero_name=hero.get("title"),
                level=entry.get("level"),
                kills=kda[0],
                deaths=kda[1],
                assists=kda[2],
                last_hits=lh[0],
                denies=lh[1],
                gpm=gpm[0],
                xpm=gpm[1],
                net_worth=entry.get("net_worth"),
                gold=entry.get("gold"),
                items=items,
            ))

        return result

    @staticmethod
    def _parse_charts(charts_raw: Dict[str, Any]) -> TimeSeries:
        return TimeSeries(
            game_times=charts_raw.get("game_times", []) or [],
            net_worth=charts_raw.get("net_worth", []) or [],
            radiant_scores=charts_raw.get("radiant_scores", []) or [],
            dire_scores=charts_raw.get("dire_scores", []) or [],
            radiant_kills=charts_raw.get("radiant_kills", []) or [],
            dire_kills=charts_raw.get("dire_kills", []) or [],
        )

    @staticmethod
    def _parse_meta(raw: Dict[str, Any], db: Dict[str, Any]) -> MatchMeta:
        lld = raw.get("live_league_data", {}) or {}
        series = db.get("series", {}) or {}
        streams_raw = db.get("streams", []) or []

        streams = [
            {
                "platform": s.get("platform"),
                "channel": s.get("channel_title"),
                "title": s.get("title"),
                "views": s.get("views"),
            }
            for s in streams_raw
        ]

        return MatchMeta(
            league_id=lld.get("league_id"),
            series_id=series.get("id"),
            event_id=series.get("event_id"),
            series_slug=series.get("slug"),
            series_type=lld.get("series_type"),
            started_at=series.get("started_at"),
            radiant_series_wins=lld.get("radiant_series_wins", 0) or 0,
            dire_series_wins=lld.get("dire_series_wins", 0) or 0,
            streams=streams,
        )


# --------------------------------------------------------------------------- #
# CLI / демонстрация
# --------------------------------------------------------------------------- #

def _demo(source: str):
    """Загружает матч из файла или URL и выводит разобранную структуру."""
    import json

    if source.startswith("http"):
        import requests
        raw = requests.get(source, timeout=10).json()
    else:
        raw = json.load(open(source, encoding="utf-8"))

    match = DLTVParser.parse(raw)

    print("=" * 70)
    print(f"MATCH {match.match_id}")
    print("=" * 70)
    status = "COMPLETED" if match.is_completed else ("DRAFTING" if not match.is_picks_ended else "LIVE")
    print(f"Status: {status}")
    print(f"Game time: {match.game_time}s ({match.game_time/60:.1f} min)")
    if match.winner:
        print(f"Winner: {match.winner.upper()}")
    print(f"Score: {match.radiant_score} - {match.dire_score}")
    print(f"First blood: {match.first_blood} | First ten: {match.first_ten}")

    for side, draft in (("RADIANT", match.radiant_draft), ("DIRE", match.dire_draft)):
        if not draft:
            continue
        print(f"\n{side}: {draft.team_name} (rank {draft.team_rank})")
        for p in draft.picks:
            wr = f"{p.team_win_rate*100:.0f}%" if p.team_win_rate is not None else "-"
            print(f"  {p.hero_name:<16} ({p.team_matches} games | {wr})")
        print(f"  Bans: {draft.bans}")

    if match.is_completed and match.radiant_players:
        print("\n--- PLAYER STATS (Radiant) ---")
        for p in match.radiant_players:
            print(f"  {p.player_name:<12} {p.hero_name:<16} "
                  f"{p.kills}/{p.deaths}/{p.assists}  "
                  f"NW:{p.net_worth} GPM:{p.gpm}")

    if match.meta:
        print(f"\nTournament: league {match.meta.league_id}, series {match.meta.series_id}")
        print(f"Streams: {len(match.meta.streams)}")

    print("=" * 70)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "sample_completed_match.json"
    _demo(src)
