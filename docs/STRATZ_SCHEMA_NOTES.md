# Stratz integration — schema notes & integration log

Captured across v0.7.5 → v0.7.20 (10+ probe iterations, 2026-08-01 → 2026-08-02).
**Updated at v0.7.21 with the actual `__type` introspection results.**

## TL;DR

**v0.7.20 finally reached Stratz with `__type` introspection.** The user changed
networks (new IP) and the Playwright cookie warmup got past Cloudflare with
`cf_clearance + user + _ga + _ga_5PXZMMDTH8`. The full schema is now known.

**The v0.7.6 XGBoost model (62.11% acc / 0.679 AUC) remains production.** Stratz
is now an **augmentation source**, not a critical path:
- v0.7.21: use Stratz to **enrich `v18_top_teams.json`** with `winCount/lossCount`
  → `win_rate` + `lastMatchDateTime` (recency filter) for our 540 seed teams
- v0.7.22+: discover new teamIds via `leagues(...)` → `matches` → `radiantTeamId/direTeamId`
- v0.7.23+: pull **additional match data** (with patch, rank, lane outcomes) to
  expand the v18 training corpus beyond the 3403 OpenDota matches

Network caveat: Stratz blocks most IPs at Cloudflare. The user's new IP works;
the assistant's server IP does not. Future runs of `stratz_dump_playwright.py`
must happen on the user's machine (or any IP that passes CF).

---

## The actual schema (confirmed via `__type` introspection)

### `DotaQuery` — top-level fields

| Field          | Signature                              | Status |
|----------------|----------------------------------------|--------|
| `match`        | `match(id: Long!): MatchType`          | ✅ |
| **`matches`**  | **`matches(ids: [Long!]!): [MatchType]`** | ✅ — **by id list, no pagination** |
| `player`       | `player(steamAccountId: Long!): PlayerType` | ✅ |
| `players`      | `players(steamAccountIds: [Long!]!): [PlayerType]` | ✅ |
| `team`         | `team(teamId: Int!): TeamType`         | ✅ |
| **`teams`**    | **`teams(teamIds: [Int]!): [TeamType]`** | ✅ — **by id list, no pagination** |
| `league`       | `league(id: Int!): LeagueType`         | ✅ |
| `leagues`      | `leagues(request: LeagueRequestType!): [LeagueType]` | ✅ |
| `guild`        | `guild(id: Int!): GuildType`            | ✅ |
| `yogurt`       | `yogurt: YogurtType`                    | 🤷 unknown |
| `plus`         | `plus: PlusType`                        | 🤷 Stratz Plus info |
| `stratz`       | `stratz: StratzType`                    | 🤷 metadata |
| `heroStats`    | `heroStats: HeroStatsType`              | 🤷 global hero stats |
| `constants`    | `constants: ConstantsType`              | 🤷 enums |
| **`leaderboard`** | **`leaderboard: LeaderboardQuery`** | ✅ — **it's a sub-query object, not a list** |
| **`live`**     | **`live: LiveType`**                   | ✅ — live matches |
| `vendor`       | `vendor: VendorType`                    | 🤷 |

❌ **`topTeams`, `proTeams`, `featuredTeams` — DO NOT EXIST** (confirmed via error
responses on v0.7.16 and v0.7.17).

### `TeamType` — actual fields (from `__type`)

| Field | Type | Notes |
|---|---|---|
| `id` | `Int!` | non-null |
| `name` | `String` | display name |
| `tag` | `String` | short tag (often null for newer teams) |
| `dateCreated` | `Long` | Unix timestamp |
| `isPro` | `Boolean` | verified pro team flag (null for non-pro) |
| `isLocked` | `Boolean` | team visibility lock |
| `countryCode` | `String` | ISO code (null for non-pro) |
| `countryName` | `String` | (often empty string) |
| `url` | `String` | |
| `logo` | `String` | |
| `baseLogo` | `String` | |
| `bannerLogo` | `String` | |
| **`winCount`** | **`Int`** | ⭐ total wins all-time (cumulative, includes amateur) |
| **`lossCount`** | **`Int`** | ⭐ total losses all-time |
| **`lastMatchDateTime`** | **`Long`** | ⭐ Unix timestamp, recency signal |
| `coachSteamAccountId` | `Long` | |
| `coachSteamAccount` | `SteamAccountType` | nested |
| `matches` | `[MatchType]` | nested — full match history for this team |
| `series` | `[SeriesType]` | |
| `members` | `[SteamAccountTeamMemberType]` | |
| `matchesGroupBy` | `[MatchGroupByType]` | |
| `heroPickBan` | `[MatchPickBanGroupByType]` | |
| `leagues` | `[LeagueType]` | |

**❌ No `rating`, `elo`, `glicko`, `mmr`, or similar skill field.** Stratz's
"skill proxy" is just `win_rate = winCount / (winCount + lossCount)` plus
recency filter on `lastMatchDateTime`.

### `LeagueType` — fields that matter for team discovery

| Field | Type | Notes |
|---|---|---|
| `id` | `Int` | |
| `name` | `String` | (often null in default ordering) |
| `displayName` | `String` | preferred name field |
| `tier` | `LeagueTier` (ENUM) | `AMATEUR`, `PROFESSIONAL`, `MAJOR`, `PREMIUM`, `INTERNATIONAL` |
| `region` | `LeagueRegion` (ENUM) | |
| `prizePool` | `Int` | |
| `basePrizePool` | `Int` | |
| `startDateTime` | `Long` | |
| `endDateTime` | `Long` | |
| `lastMatchDate` | `Long` | |
| `hasLiveMatches` | `Boolean` | |
| `imageUri` | `String` | |
| `country` | `String` | |
| `venue` | `String` | |
| **`matches`** | **`[MatchType]`** | ⭐ for our pipeline |
| **`standings`** | **`[TeamPrizeType]`** | ⭐ teams in this league |
| `liveMatches` | `[MatchLiveType]` | |
| `series` | `[SeriesType]` | |
| `tables` | `LeagueTableType` | |
| `stats` | `LeagueStatType` | |
| `nodeGroups` | `[LeagueNodeGroupType]` | |

### `LeagueRequestType` — filter shape

```graphql
{
  leagueId: Int
  leagueIds: [Int]
  tiers: [LeagueTier]   # AMATEUR, PROFESSIONAL, MAJOR, PREMIUM, INTERNATIONAL
  requireImage: Boolean
  requirePrizePool: Boolean
  requireStartAndEndDates: Boolean
  hasLiveMatches: Boolean
  leagueEnded: Boolean
  isFutureLeague: Boolean
  startDateTime: Long
  endDateTime: Long
  betweenStartDateTime: Long
  betweenEndDateTime: Long
  orderBy: FilterOrderBy
  take: Int
  skip: Int
}
```

**Best call for our use case (top active pro leagues):**
```graphql
{ leagues(request: {
    tiers: [PROFESSIONAL, MAJOR, PREMIUM, INTERNATIONAL],
    requirePrizePool: true,
    leagueEnded: false,
    take: 100,
    orderBy: DESC
  }) { id displayName tier prizePool startDateTime } }
```

### `MatchType` — fields that matter

| Field | Type | Notes |
|---|---|---|
| `id` | `Long` | match id |
| **`didRadiantWin`** | **`Boolean`** | ⭐ target |
| `durationSeconds` | `Int` | |
| `startDateTime` | `Long` | |
| `endDateTime` | `Long` | |
| `towerStatusRadiant/Dire` | `Int` | bitfield |
| `barracksStatusRadiant/Dire` | `Short` | bitfield |
| `firstBloodTime` | `Int` | |
| `lobbyType` | `LobbyTypeEnum` | |
| `gameMode` | `GameModeEnumType` | |
| `numHumanPlayers` | `Int` | |
| `replaySalt` | `Long` | |
| `isStats` | `Boolean` | |
| `tournamentId` | `Int` | |
| `tournamentRound` | `Short` | |
| `actualRank` | `Short` | ⭐ average rank in match |
| `averageRank` | `Short` | ⭐ |
| `averageImp` | `Short` | ⭐ individual matchmaking points? |
| `parsedDateTime` | `Long` | |
| `statsDateTime` | `Long` | |
| `leagueId` | `Int` | |
| `league` | `LeagueType` | nested |
| **`radiantTeamId`** | **`Int`** | ⭐ for team discovery |
| `radiantTeam` | `TeamType` | |
| **`direTeamId`** | **`Int`** | ⭐ for team discovery |
| `direTeam` | `TeamType` | |
| `seriesId` | `Long` | |
| `series` | `SeriesType` | |
| `gameVersionId` | `Short` | ⭐ patch version |
| `regionId` | `Byte` | |
| `sequenceNum` | `Long` | |
| `rank` | `Int` | |
| `bracket` | `Byte` | |
| `analysisOutcome` | `MatchAnalysisOutcomeType` (ENUM) | ⭐ Stratz's predicted outcome |
| `predictedOutcomeWeight` | `Byte` | ⭐ confidence in Stratz's prediction |
| **`players`** | **`[MatchPlayerType]`** | ⭐ for training |
| `radiantNetworthLeads` | `[Int]` | ⭐ time series, gold lead by minute |
| `radiantExperienceLeads` | `[Int]` | ⭐ time series, XP lead by minute |
| `radiantKills` | `[Int]` | ⭐ time series, kills by minute |
| `direKills` | `[Int]` | ⭐ time series |
| **`pickBans`** | **`[MatchStatsPickBanType]`** | ⭐ for training |
| `towerStatus` | `[MatchStatsTowerReportType]` | |
| `laneReport` | `MatchStatsLaneReportType` | ⭐ lane outcomes |
| `winRates` | `[Decimal]` | per-minute win rate? |
| `predictedWinRates` | `[Decimal]` | ⭐ Stratz's predicted win rate by minute |
| `chatEvents` | `[MatchStatsChatEventType]` | |
| `towerDeaths` | `[MatchStatsTowerDeathType]` | |
| `playbackData` | `MatchPlaybackDataType` | |
| `spectators` | `[MatchPlayerSpectatorType]` | |
| `bottomLaneOutcome` | `LaneOutcomeEnums` | |
| `midLaneOutcome` | `LaneOutcomeEnums` | |
| `topLaneOutcome` | `LaneOutcomeEnums` | |
| `didRequestDownload` | `Boolean` | |

---

## Sanity check on real data (PARIVISION, id=9572001)

```json
{
  "id": 9572001,
  "name": "PARIVISION",
  "tag": null,
  "isPro": null,
  "isLocked": null,
  "countryCode": null,
  "countryName": "",
  "winCount": 330,
  "lossCount": 173,
  "lastMatchDateTime": 1782410022,
  "dateCreated": null
}
```

- `lastMatchDateTime=1782410022` = **2026-06-23 ~20:33 UTC** (recent!)
- `win_rate = 330 / (330+173) = 65.6%` — strong team
- `isPro: null, countryCode: null` — Stratz doesn't verify semi-pro. We don't care.

**This is the exact data we need to enrich `v18_top_teams.json`.**

---

## The pipeline (the plan)

### Phase 1: enrich existing 540 seed teams (v0.7.21)

```graphql
{ teams(teamIds: [9572001, ...539 more IDs in chunks of 50]) {
    id name tag isPro isLocked countryCode countryName
    winCount lossCount lastMatchDateTime dateCreated
} }
```

Then:
- Compute `win_rate = winCount / (winCount + lossCount)` per team
- Filter to `lastMatchDateTime > 6 months ago` (recency)
- Save to `stratz_teams.json` (next to script)
- For each team_id, attach `win_rate_stratz` and `last_match_stratz` to
  `v18_top_teams.json` (or write a new `v18_top_teams_stratz.json` for A/B)

### Phase 2: discover new teamIds via leagues (v0.7.22)

```graphql
# Step 1: top pro leagues
{ leagues(request: {
    tiers: [PROFESSIONAL, MAJOR, PREMIUM, INTERNATIONAL],
    requirePrizePool: true, leagueEnded: false, take: 100
  }) { id displayName tier } }

# Step 2: per league, fetch matches
# (need to find the right query shape -- likely:
#   league(id: X) { matches { id radiantTeamId direTeamId } } )

# Step 3: dedupe teamIds
# Step 4: teams(teamIds: [...new IDs...]) { ... }
```

### Phase 3: expand match corpus (v0.7.23+)

```graphql
{ matches(ids: [...200 match IDs at a time...]) {
    id didRadiantWin durationSeconds startDateTime gameVersionId
    actualRank averageRank averageImp
    radiantTeamId direTeamId leagueId
    players { ... } pickBans { ... }
} }
```

This gets us **patch-versioned, rank-tagged** matches we can use to grow
the v18 training corpus from 3403 → 10K+ matches with better features.

---

## Network failure timeline (preserved for reference)

| Time (MSK) | What we tried                                        | Result                                  |
|------------|------------------------------------------------------|-----------------------------------------|
| 2026-08-01 night | server-side urllib POST → `api.stratz.com/graphql` | HTML 403 ("Just a moment..." Cloudflare) |
| 2026-08-01 late | same, with full Stratz JWT in headers                | same 403, also with `Authorization`     |
| 2026-08-02 12:30 | user runs the script on their machine               | got useful 400 with schema errors ✨    |
| 2026-08-02 13:48 | user retries — IP got TCP-blocked                   | `ConnectTimeoutError` on `api.stratz.com:443` |
| **2026-08-02 14:20** | **user changes network (new IP)**                   | **cf_clearance + user + _ga + _ga_5PXZMMDTH8 issued, GraphQL works** |
| 2026-08-02 14:30 | v0.7.20 probe — `__type` introspection              | full schema dumped, 48272 bytes         |

The mid-session flip from "useful 400 errors" to "TCP timeout" was **IP-block
tightening on the original network**. The user's new IP works.

---

## Artifacts in the repo

- `scripts/stratz_client.py` — server-side GraphQL client (won't work from here)
- `scripts/stratz_dump.py` — portable dump, urllib only (the original 8-query probe)
- `scripts/stratz_dump_playwright.py` — Playwright-based variant (solves Cloudflare
  JS challenge automatically, reuses `cf_clearance` + `user` cookies for
  GraphQL POST). At v0.7.20 it has the extended probe that saved the full
  schema to `stratz_schema_discovery.json`.
- `scripts/stratz_introspect.py` — `__type` introspection (now redundant)
- `scripts/stratz_probe.py` — 10-pattern query probe (now redundant)
- `C:\Users\artka\Downloads\stratz_dump_playwright.py` — the version the user runs
  (synced from repo at v0.7.20)
- `C:\Users\artka\Downloads\stratz_schema_discovery.json` — full untruncated
  `__type` output (48272 bytes, 10 queries, 2026-08-02)
