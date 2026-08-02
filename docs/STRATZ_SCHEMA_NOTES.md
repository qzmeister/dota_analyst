# Stratz integration — schema notes & dead-end log

Captured across v0.7.5 → v0.7.17 (10+ attempts, 2026-08-01 → 2026-08-02).

## TL;DR

**Stratz is unreachable from this network** — both the assistant's server IP and the
user's home IP got TCP-level timeouts on `api.stratz.com` (60s `connect timeout`).
Earlier in the session Stratz responded with HTTP 400 and useful GraphQL schema
errors, but that stopped working partway through. The artifacts in this folder
preserve what we learned so a future attempt (from a different network, or after
the block lifts) doesn't have to rediscover it.

The v0.7.6 XGBoost model (62.11% acc / 0.679 AUC) remains production. The
tier ratings in `ml_data/imports/v18_top_teams.json` are computed from OpenDota
match data and don't depend on Stratz.

---

## What we confirmed about the Stratz GraphQL schema

Source: actual error responses from `POST https://api.stratz.com/graphql`,
captured before the network block hardened.

### Root: `DotaQuery`

| Field          | Signature                       | Status |
|----------------|---------------------------------|--------|
| `teams`        | `teams(teamIds: [Int]!): [TeamType]` | ✅ confirmed |
| `team`         | `team(id: Long!): TeamType`     | ✅ confirmed |
| `topTeams`     | —                               | ❌ "Did you mean 'teams' or 'team'?" |
| `proTeams`     | —                               | ❌ "Did you mean 'teams'?" |
| `leagues`      | `leagues(request: LeagueRequestType): [LeagueType]` | ✅ confirmed in earlier probes |
| `matches`      | `matches(request: MatchRequestType): [MatchType]`   | ✅ confirmed in earlier probes |

### `TeamType` — what we know it does NOT have

The following common guesses were rejected with
`"Cannot query field 'X' on type 'TeamType'"`:

- ❌ `rating`
- ❌ `wins`
- ❌ `losses`
- ❌ `nodes` (so `teams` is not a Relay connection — it's a plain list)

To get the actual field list you would need
`{ __type(name: "TeamType") { fields { name type { name kind ofType { name } } } } }`
**but this introspection call is also blocked from this network now**.

### What we suspect is on `TeamType`

Based on what Stratz's web UI exposes (https://stratz.com/teams) and the
`DotaQuery` schema shape, the team record likely includes some of:
`id, name, tag, captainSteamId, isPro, countryCode, logo, lastMatchDateTime,
memberSteamIds, currentVs, performanceTier (or similar ordinal)`.

None of this was confirmable. **If/when network access is restored, run the
`__type` introspection first** before any `teams(...)` call.

---

## The chicken-and-egg problem

`teams(teamIds: [Int]!)` requires you to *already know* the team IDs. The
documented `topTeams` / `proTeams` shortcuts **don't exist** on this schema.
The only ways to discover team IDs are:

1. `__schema { queryType { fields { name } } }` — to find what *is* on `DotaQuery`
2. `leagues(...)` — to discover leagues, then their matches, then team IDs from match data
3. Web scrape `https://stratz.com/teams` — the web UI lists pro teams publicly
4. Hard-coded seed list from a known good source (Liquipedia Dota 2, OpenDota `/teams`)

**This network can't do any of (1) or (2) right now.**
(3) and (4) are not Stratz integrations; they're alternative data sources.

---

## Network failure timeline

| Time (MSK) | What we tried                                        | Result                                  |
|------------|------------------------------------------------------|-----------------------------------------|
| 2026-08-01 night | server-side urllib POST → `api.stratz.com/graphql` | HTML 403 ("Just a moment..." Cloudflare) |
| 2026-08-01 late | same, with full Stratz JWT in headers                | same 403, also with `Authorization`     |
| 2026-08-02 morning | user runs the same script on their machine          | got useful 400 with schema errors ✨    |
| 2026-08-02 12:30 | user retries `teams 200`                             | `cf_clearance` + `user` cookies collected, then `ConnectTimeoutError` on `api.stratz.com:443` (60s × N retries) |
| 2026-08-02 13:48 | same retry                                            | same timeout — endpoint is hard down   |

The mid-session flip from "useful 400 errors" to "TCP timeout" suggests
**the network block is dynamic and tightened** — the IP that was getting
through the Cloudflare JS challenge once now doesn't.

---

## What does work for team ratings (alternative paths)

If we want authoritative team ratings without Stratz, the options are:

| Source                                | Free? | From this network? | Quality |
|---------------------------------------|-------|--------------------|---------|
| **Computed from OpenDota matches**    | ✅     | ✅                 | OK (we use this in v0.7.0) |
| OpenDota `/teams` endpoint            | ✅     | ❌ 429 rate-limited | good (Glicko-2) |
| OpenDota `/explorer` SQL              | ✅     | ❌ 429 rate-limited | excellent |
| Stratz GraphQL                        | ✅     | ❌ TCP blocked      | excellent |
| Liquipedia Dota 2 scrape              | ✅     | ✅ (HTTP)           | manual only, no ratings |
| Dotabuff                              | ❌ paid | —                 | excellent |
| datdota                               | ❌ paid | —                 | excellent |
| Valve `IEconDOTA2_570/GetHeroes` etc. | —     | —                 | no team data |

The 24-48h OpenDota 429 cooldown is the most likely natural recovery. After it
clears, re-running `scripts/collect_opendota_polite.py` should give us
authoritative Glicko ratings to feed back into `v18_top_teams.json`.

---

## Artifacts left in the repo

- `scripts/stratz_client.py` — server-side GraphQL client (won't work from here)
- `scripts/stratz_dump.py` — portable dump, urllib only (the original 8-query probe)
- `scripts/stratz_dump_playwright.py` — Playwright-based variant (solves Cloudflare
  JS challenge automatically, reuses `cf_clearance` + `user` cookies for
  GraphQL POST). At v0.7.17 it's the most complete variant, but **its target
  endpoint is hard-blocked right now**.
- `scripts/stratz_introspect.py` — `__type` introspection (would be the next
  step, but blocked)
- `scripts/stratz_probe.py` — 10-pattern query probe
- `C:\Users\artka\Downloads\stratz_dump.py` (and `*_playwright.py`) — the
  versions the user runs locally
