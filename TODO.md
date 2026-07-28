# TODO

> Backlog for the Dota Analyst project. Each task is tagged with a target
> version and a status. **Closed** items reference the version they were
> resolved in (see [CHANGELOG.md](CHANGELOG.md)).
>
> Versioning: `product.major.minor`
> - `product` = `0` (pre-release) / `1` (first production-ready) / ...
> - `major`   = bumps for large logic / architectural changes
> - `minor`   = bumps for each bug-fix / hardening build

Status legend: `[ ]` open · `[~]` in progress · `[x]` closed · `[?]` blocked / needs decision

---

## ✅ Closed in 0.0.1 (2026-07-24) — code review pass

These were addressed in the first bug-fix build. Detail in [CHANGELOG.md](CHANGELOG.md#001--2026-07-24).

- [x] Lock down CORS to env-configured origins
- [x] Add exponential backoff to `datdota_client` (`RULES.md §1`)
- [x] Add retry to `dltv_client` (was a single-shot urllib call)
- [x] Extract shared `backend/_http.py` retry utility
- [x] Replace magic numbers in `analysis.py` with named constants
- [x] Remove dead `DLTVClient._val`
- [x] Move stray `import os` in `discovery.py` to module top
- [x] Make `backend/__init__.py` tolerant of missing `python-dotenv`
- [x] First test suite — 23 tests for `analyze()`, `decode_towers()`, `map_verdicts()`
- [x] Add `requirements-dev.txt` and `pytest` configuration
- [x] `None`-guard in `map_verdicts` (bug caught by new tests)

---

## 🚧 Target: 0.0.2 — next hardening build

Small-scope bug fixes and refactors. Anything that doesn't change public contract or architecture.

- [x] `app.py:81-86` — remove the redundant `StaticFiles` mount that overrides route handlers (low risk, low value — review before changing) — ✅ done in 0.1.0 (StaticFiles removed entirely when frontend moved to nginx)
- [x] `dltv_client.py:187-190` — make `hero_by_dltv_id` thread-safe (currently the lazy-init guard is racy) — ✅ done in 0.1.0 (double-checked locking with `threading.Lock` + `_heroes_loaded` flag)
- [x] `_TTLCache` — add `maxsize` and LRU eviction (low risk, prevents unbounded growth in long-running processes) — ✅ done in 0.1.0 (`maxsize=128` LRU eviction)
- [x] Add `/api/health` endpoint to `app.py` (liveness + DB/cache ping) — ✅ done in 0.1.0 (split into `/api/healthz` + `/api/readyz` per K8s convention)
- [x] Replace `print(...)` in `board.py` and `discovery.py` with a proper logger (so log level / format can be configured centrally) — ✅ done in 0.0.2 (`business/_logging.py` + `setup_logging()`); remaining `print(...)` only in legacy `ml_trainer.py` (now removed in 0.2.0)
- [x] Add `pyproject.toml` or `setup.cfg` for `pip install -e .` — ✅ done in 0.0.2
- [x] `.env.example` documenting every env var (CORS, Steam, Stratz, DatDota, future DB) — ✅ done in 0.0.2 (and extended in 0.2.0 with `PREDICTION_ENGINE` / `MODEL_DIR`)
- [x] Tests for `dltv_client._normalize_hero` (string-roles / list-roles branch) — ✅ done in 0.1.0 (`test_dltv_client.py` — 12 tests)
- [x] Tests for `discovery._slugify`, `_extract_url_event_slug` (edge cases on Cyrillic + Unicode) — ✅ done in 0.1.0 (`test_discovery.py` — 10 tests)
- [x] CI: GitHub Actions running `pytest` on PR — ✅ done in 0.1.0 (`.github/workflows/tests.yml`)

> **0.0.2 is fully closed.** All 10 items were completed across 0.0.2 and 0.1.0.

---

## 🏗 Target: 0.1.0 — split into 3 deployable services

**First `major` bump** — this changes the architecture. Old monolithic `uvicorn backend.app:app` becomes three independent processes. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full plan.

### Web node (frontend)

- [x] Move `frontend/` out of the FastAPI app into a standalone static bundle (no FastAPI dependency)
- [x] Pick a real "dumb" stack — static HTML+JS. All data over REST + SSE. **Zero business logic in JS**.
- [ ] **SSE client** (`EventSource`) for live match updates, replaces polling on `/api/board`
- [ ] **REST POST** for client→server commands (add to watchlist, change filter, etc.)
- [x] Containerize: serve via `nginx` with proper cache headers and security headers
- [x] Add `Dockerfile.web` and a `docker-compose` entry

### Gateway node (security + routing)

- [x] New service `gateway/` with FastAPI — only entry point exposed to the network
- [x] Apply: **middleware** chain (request-id, CORS, auth, body-size limit) — rate-limit deferred
- [x] Apply: **Facade** pattern — gateway is a thin wrapper, hides internal topology
- [ ] Apply: **Adapter** pattern — one adapter per backend service for type-safe internal RPC (HTTP or gRPC)
- [ ] Apply: **Circuit Breaker** around every outbound call to `business` node
- [ ] Schema validation on all incoming requests (Pydantic)
- [x] Auth strategy (decided: API keys for now, OAuth/JWT later — see [ARCHITECTURE.md](ARCHITECTURE.md#security))
- [ ] WebSocket gateway with origin checks and per-connection rate limits
- [x] Containerize: `Dockerfile.gateway`

### Business + ML node (core)

- [x] New service `business/` — internal only, never exposed
- [x] Refactor `backend/` → `business/` preserving all current logic
- [ ] Apply: **Repository** pattern for all data access. **Start with `JsonFileRepository`** (no Postgres dependency), add `PostgresRepository` later behind the same interface
- [ ] Apply: **Service Layer** — `PredictionService`, `DataCollectionService`, `MatchIngestionService`
- [ ] Apply: **Strategy** pattern for prediction engines — `IPredictionEngine` with `HeuristicEngine` (current `analysis.py`) and `MLEngine` (current `ml_trainer.py`) as pluggable implementations
- [ ] Apply: **Dependency Injection** — wire services through FastAPI `Depends()` so they're testable in isolation
- [ ] Apply: **DTO / Schema** — Pydantic models for every internal contract (no raw `dict`s between layers)
- [ ] Apply: **Decorator** — `@cached`, `@retried`, `@timed` for hot-path functions
- [ ] Add **Postgres** as primary store (deferred — start with `JsonFileRepository`)
- [ ] Add **Redis** for: pub-sub (live updates), session cache, rate-limit counters
- [ ] Apply: **Observer / Pub-Sub** — Redis pub-sub for match-state changes consumed by gateway and pushed via SSE
- [ ] ML trainer integration — `init_ml_trainer()` called at app startup, `PredictionService` picks engine based on config (`PREDICTION_ENGINE=heuristic|ml|ensemble`)
- [ ] Background scheduler (APScheduler) for nightly retrain + manual CLI trigger
- [x] Containerize: `Dockerfile.business`

### Cross-cutting (all nodes)

- [x] **Docker Compose** for local dev: `web`, `gateway`, `business` services with internal networks. **No Postgres yet** — `JsonFileRepository` is the primary store. **No Redis yet** — pub-sub / rate-limit are deferred.
- [ ] **Security** checklist applied:
  - [x] Secrets in env / secrets manager, never in code or images
  - [x] Network policies: only `web` ports exposed
  - [ ] All inter-service traffic on private network, mTLS for prod
  - [ ] WAF rules: block SQLi patterns, oversized bodies, suspicious User-Agents
  - [ ] OWASP top-10 review of gateway endpoints
  - [ ] Dependency scan in CI (Trivy / `pip-audit`)
  - [ ] SBOM generated per build
- [ ] Observability: structured logging (JSON), Prometheus metrics, OpenTelemetry traces
- [x] Health checks (`/healthz`, `/readyz`) on every service
- [ ] Deployment guide (`docs/deploy.md`): staging → prod, env parity

---

## 🧠 Target: 0.2.0 — ML integration (Strategy + winner-only MVP)

> **In progress — 2026-07-24.** 1111 verified DatDota matches in
> `ml_data/full_matches/` (manifest: `ml_data/imports/2026-07-24-telegram-desktop.json`).
> The old isolated `ml_trainer.py` is gone; replaced by the
> `business.ml.*` package (Strategy pattern).

- [x] Verify `ml_data/full_matches/` is parseable end-to-end — ✅ done (1111 clean, 0 errored; balanced labels 549/562)
- [x] Replace `ml_trainer.py` with `IPredictionEngine` (Strategy) — ✅ done (`business.ml.engine`: `IPredictionEngine` ABC + `HeuristicEngine` + `MLEngine`)
- [x] Drop circular features (`duration_min`, `total_kills`) — ✅ done (`features.py` uses only prematch hero target encodings; no post-match leakage)
- [x] Use identical feature schema in `train()` and `predict()` — ✅ done (`FEATURE_ORDER` + `N_FEATURES` in `features.py`; `ModelStorage.load()` validates metadata against the live `FEATURE_ORDER` and refuses to load a stale model)
- [x] Persist trained model with version metadata: `sklearn_version`, `feature_names`, `trained_at`, `metrics` (JSON sidecar) — ✅ done (`ModelStorage` writes `metadata.json` sidecar to every version dir; encoder is round-tripped through the metadata so model + encoder cannot drift)
- [ ] A/B harness: compare `HeuristicEngine` vs `MLEngine` on historical matches, log per-engine accuracy — 📋 deferred to 0.2.1 (a `scripts/eval_engines.py` will be added alongside the new regressors)
- [x] Hero embeddings: target encoding for high-cardinality hero IDs — ✅ done (one-hot rejected — 126 heroes would blow up the feature space; `HeroWinRateEncoder` is O(1) at predict time, interpretable, smoothable)
- [ ] Champion / Challenger deployment for prediction engines (canary ML alongside heuristic) — 📋 deferred to 0.2.x (needs Redis pub-sub for traffic splitting, see 0.1.1 roadmap)

### 0.2.0 deliverables (this release)

- `business/ml/features.py` — `HeroWinRateEncoder` (smoothing 5.0, min_samples 3), `FEATURE_ORDER` (13 features), `extract_features()`, `hero_ids_from_match()`, `target_from_match()`
- `business/ml/storage.py` — `ModelStorage` (versioned `model.joblib` + `metadata.json` sidecar), `LoadedModel` bundle, atomic writes
- `business/ml/engine.py` — `IPredictionEngine` ABC, `HeuristicEngine` (wraps `analysis.analyze()`), `MLEngine` (overrides only `winner` block; falls back to heuristic on missing hero IDs), `make_engine()` factory driven by `PREDICTION_ENGINE` env
- `business/ml/train.py` — CLI `python -m business.ml.train`; fits encoder, trains `LogisticRegression` (or `HistGradientBoostingClassifier` via `--model histgb`), computes accuracy + log-loss + ROC AUC, saves via `ModelStorage`
- `business/app.py` — startup hook warms the engine; `get_board` now returns `engine` in the response so the UI / curl can see which engine produced the predictions
- `business/board.py` — `_postmatch_prediction`, `_postmatch_card`, `_live_card` and `analyze_map_with_verdict` now route through the active `IPredictionEngine`
- `tests/test_ml_features.py`, `tests/test_ml_engine.py` — unit tests
- `requirements.txt` + `pyproject.toml` — `numpy`, `scikit-learn`, `joblib` promoted to runtime deps
- `ml_data/models/winner_v1/` — first trained artifact (1111 matches, 80/20 split)

### 0.2.1 — regression on kills / towers / duration (planned)

The shortlist (Tweedie / Gamma / квантильная) covers all three numeric
targets; each is a small `IPredictionEngine` impl that overrides one
block at a time. The `MLEngine` learns to become a full multi-target
predictor. Evaluation harness is added in the same release so we have
a quality baseline for the heuristic too.

| Target           | Estimator                                    | Notes |
|------------------|----------------------------------------------|-------|
| Kills total      | `HistGradientBoostingRegressor(loss="poisson")` first; CatBoost Tweedie 1.4 as upgrade | bounded 30..78, count data |
| Towers destroyed | `HistGradientBoostingRegressor(loss="poisson")` first; ZINB via statsmodels if zero-inflation dominates | heavy zero-inflation, bounded 0..11 |
| Duration (mean)  | `HistGradientBoostingRegressor` with log-target | positive, long tail |
| Duration (P10/P90) | XGBoost `reg:quantileerror` (alpha=0.1, 0.9) | for over/under bet thresholds |

> ✅ **Done — 2026-07-24.**  4 regressors trained (kills + duration_mean
> + duration_p10 + duration_p90), winsorize via MAD, A/B harness
> committed.  Towers regressor remains unimplemented because the
> training corpus has no per-side tower bitmask; deferred to 0.2.2
> with a re-pulled DLTV corpus.
> See `CHANGELOG.md` (0.2.1 section) for full results — ML beats
> the heuristic on **5 of 6** metrics (winner accuracy, kills MAE/RMSE,
> duration MAE/RMSE); only `winner log_loss` is slightly worse.

### 0.2.2 — towers + ZINB + probability calibration (planned)

The 0.2.1 ML is a clear win on kills / duration, but two things
block a 1.0.0 promotion:

- **Towers regressor still missing.**  Re-pull matches from DLTV
  (which carries `tower_radiant` / `tower_dire` bitmasks) into
  `ml_data/full_matches/`, then add a HistGBR(Poisson) tower
  regressor.  If zero-inflation dominates after a first fit, swap
  in a ZINB GLM via `statsmodels`.
- **Probability calibration.**  ML `winner.log_loss` is 0.064 worse
  than the heuristic's "always 0.5" baseline.  Add Platt or
  isotonic regression on a held-out fold to recalibrate the
  classifier's `predict_proba` output without retraining.

| Item | Owner | Target |
|------|-------|--------|
| Re-pull DLTV matches with tower bitmasks → `ml_data/full_matches/` | ML | 0.2.2 |
| `TowersRegressor` (HistGBR(Poisson) → ZINB if needed) | ML | 0.2.2 |
| `CalibratedClassifierCV` wrapper around `LogisticRegression` | ML | 0.2.2 |
| Re-run `scripts/eval_engines.py` and report delta | ML | 0.2.2 |

---

## 📊 Backlog — ML features (carried from the original TODO)

> The original 10-feature roadmap. Re-prioritised: **only kick off when
> `0.2.0` ML foundation is in place.** Otherwise these are aspirational.

| Priority | Feature | Target version | Status |
|----------|---------|----------------|--------|
| 🔴 High | Team Statistics Aggregator | 0.2.x | [ ] |
| 🔴 High | Hero Pair Lane Dominance | 0.3.x | [ ] |
| 🟡 Medium | Mid Lane Matchup | 0.3.x | [ ] |
| 🟡 Medium | Kill Distribution by Time | 0.2.x | [ ] |
| 🟡 Medium | First Tower Fall Timing | 0.3.x | [ ] |
| 🟢 Low | Item Timing Analysis | 0.4.x | [ ] |
| 🟢 Low | Gold/XP Graph Shape | 0.4.x | [ ] |
| 🟢 Low | Patch Meta Analysis | 0.4.x | [ ] |
| 🟢 Low | Tournament Tier Weighting | 0.2.x | [ ] |
| 🟢 Low | Player Form Tracking | 0.4.x | [ ] |

### Notes on carryover

- All of these require `ml_data/full_matches/` enriched with detailed
  per-game timeline (gold/XP, items, towers). That's a separate
  `scripts/collect_full_matches.py` effort, not a model-training one.
- `KILLS / TOWERS / DURATION / FB` over/under bets from `analysis.py`
  can already be evaluated against historical results — we should
  build the evaluation harness first, even before ML, so we have
  a quality baseline.
- Hero Pair Lane Dominance is the highest-leverage feature for
  prediction accuracy but also the most data-hungry; defer until
  `ml_data/` is mature.

---

## 📋 Resolved decisions (2026-07-24)

- [x] **Live protocol** — SSE for server→client, REST for client→server commands. Decided after discussion: no bidirectional push is needed.
- [x] **Auth (local)** — single static dev token in `.env` (header `X-API-Key`). Code is auth-aware from day one; the prod token source will swap in without code changes.
- [x] **Auth (prod)** — API keys per user + HMAC between services. Deferred to prod-prep.
- [x] **Primary storage (0.1.0)** — `JsonFileRepository` first, `PostgresRepository` later. Repository pattern lets us swap backing store without touching services.
- [x] **Object storage** — local folder `ml_data/` mounted as Docker volume; S3 in prod behind the same `IObjectStore` interface.
- [x] **ML training cadence** — APScheduler nightly + manual CLI trigger for 0.1.0. Push-based (`>1000 new samples`) is the 0.2.x trigger.
- [x] **Observability** — structured JSON logs to stdout. Prometheus + Grafana added when there's prod traffic.

## 📋 Still open

- [?] **Database hosting in prod** — managed (RDS / Cloud SQL) vs self-hosted. Defer to prod-prep.
- [?] **CDN for static frontend** — Cloudflare / Fastly in front of nginx. Defer to prod-prep.

---

## 🔒 Local-only / pre-release assumptions (revisit pre-1.0)

This project is currently pre-release and runs on a trusted LAN.
Several "good enough for now" choices are baked into the code,
each of which **must be revisited before a public deployment**.
This list is the audit checklist for the 0.4.x → 1.0.0 hardening
sprint.

### Auth & network

- **SSE auth bypass** — `gateway/_middleware.py` ships with
  `UNAUTHED_PREFIXES = ("/api/stream/",)` so the browser can
  `EventSource` without sending the `X-API-Key` header
  (the EventSource API doesn't support custom headers).  **A
  public deployment with this wide open is a DDoS vector** —
  anyone can open a long-lived connection and exhaust workers.
  Plan: cookie-based auth, login endpoint, remove from
  `UNAUTHED_PREFIXES` (planned for 0.4.0).
- **Single static dev token** — `DEV_API_KEY` in `.env`,
  `X-API-Key` header.  No per-user keys, no rotation, no HMAC
  between services.  Plan: API keys per user + HMAC for
  service-to-service calls (prod-prep).
- **CORS not pinned in CI** — `CORS_ORIGINS` is an env var;
  nothing checks that it isn't `localhost` at CI time.
  Audit P1-7 deferred — pre-release is local-only.  Plan: add
  the CI check (audit P1-7) before any public exposure.

### ML / data

- **Calibration off by default** — Platt / isotonic both
  overfit on the 1111-match corpus.  v1 ships un-calibrated.
  Revisit when corpus > 10k matches.
- **Multikill classifier degenerated** — 0 "Low" matches in
  the pro corpus, so the model only predicts "High" (acc 0.991
  on the trivial class).  Pipeline is in place; 0.3.1 will add
  data + features that give it signal.  Until then, the
  `multikill_v1` artifact is essentially a constant predictor.
- **Towers regressor not trained** — `ml_data/full_matches/`
  lacks the DLTV tower bitmask.  P2-11 area: re-pull from DLTV
  or drop the `towers` head until data is available.

### Storage & infra

- **`JsonFileRepository` only** — no Postgres.  The repository
  pattern lets us swap without touching services, but the
  swap itself (schema migration, connection pooling, JSONB
  queries) is unstarted work.  Postgres planned for 1.0.
- **Object storage = local folder** — `ml_data/` mounted as a
  Docker volume.  S3 swap behind `IObjectStore` interface is
  planned but the interface doesn't exist yet.
- **No rate-limiting at the edge** — `gateway/_rate_limit.py`
  is a token bucket per IP, but nginx / cloud load-balancer
  rate limits aren't configured.  Plan: edge rate limit +
  WAF rules.
- **Structured logs only, no metrics** — JSON to stdout;
  Prometheus + Grafana deferred to "when there's prod traffic"
  (audit P2-9).

### Observability & ops

- **No nightly eval cron** — `scripts/eval_engines.py` exists
  but is only run manually.  Audit P2-8: schedule a nightly
  eval that diff's ML vs heuristic on fresh matches and posts
  to a dashboard.
- **No CI check for env hygiene** — `.env` files are git-ignored
  but nothing verifies that committed `.env.example` stays in
  sync with what the code actually reads.

### What this list is NOT

- It is not a "to-do" in the usual sense — every item here
  is intentionally deferred.
- It is not a "known bugs" list — these are design choices
  that are *correct for local use* and *unsafe for public use*.
- A pre-1.0 audit must re-walk this list top-to-bottom and
  either fix the item or document why it's still acceptable.

---

## 🗓 Version timeline

| Version  | Theme                                       | Status      |
|----------|---------------------------------------------|-------------|
| 0.0.1    | First hardening / test pass                 | ✅ shipped  |
| 0.0.2    | Bug-fix / refactor build (`pyproject.toml`, `Makefile`, `.env.example`, logger) | ✅ shipped  |
| 0.1.0    | 3-node architecture + Docker + auth        | ✅ shipped  |
| 0.1.1    | Rate limiting + SSE endpoint                | ✅ shipped  |
| 0.2.0    | ML integration (Strategy pattern, winner-only MVP) | ✅ shipped  |
| 0.2.1    | ML regressors (kills / duration / P10 / P90) + winsorize + A/B harness | ✅ shipped  |
| 0.2.2    | Towers + probability calibration             | ✅ shipped ¹ |
| 0.3.0    | ML-driven features (Multikill classifier + scaffolding) | ✅ shipped ¹ |
| 0.3.1    | Richer-corpus features (First-to-15, Team stats, Hero pair, Kill distribution) | 📋 planned |
| 0.3.2    | SSE auth fix (local-only `/api/stream/*` bypass) | ✅ shipped ² |
| 0.3.3    | Audit P0 fixes — `test_board.py`, `test_app.py`, expanded `test_discovery.py` (276 tests) | ✅ shipped |
| 0.3.4    | Domain exception hierarchy — 15-class tree, 22 of 31 `except Exception` narrowed (310 tests) | ✅ shipped ³ |
| 0.3.5    | `~=` pins for prod deps (audit P1-5) — 9 packages pinned to current minor | ✅ shipped |
| 0.3.6    | `pip-audit` in CI (audit P1-6) — workflow blocks on prod CVE; 0 vulns in current tree | ✅ shipped |
| P1-7     | CI check CORS not `localhost`              | ⏭️  deferred — pre-release is local-only, will be re-audited pre-1.0 |
| 0.3.7    | `train.py` tests (audit P2-10) — 50 cases, 0% → 95% coverage, fixed `make_regressor` import bug (360 tests) | ✅ shipped |
| 0.3.8    | `ARCHITECTURE.md` refresh (audit P2-12) — header, §5.2, §8 status, new §13/§14 (no code change) | ✅ shipped |
| 0.3.9    | ML accuracy push (C/D/A/B/E audit) — XGBoost winner + corpus 1111→1275, +4% accuracy, logreg tuning/pair features/team aggregates reverted as net-negative | ✅ shipped |
| 0.3.10   | Honest baseline (encoder fit on train only) + lane-pair features + corpus → 2036 + retrain | ✅ shipped |
| 0.3.10a  | Discontinue degenerate multikill classifier (pro corpus 100% High) + nightly eval workflow (P2-8) | ✅ shipped |
| 0.3.11   | Corpus 2036 → 2389 (+17%) + winner_v12 retrain (XGBoost, hero+team) — +9.84% honest forward | ✅ shipped |
| 0.3.12   | XGBoost для kills + duration (hero only) — -0.46 MAE kills, -0.49 MAE duration forward | ✅ shipped |
| 0.3.13   | Cross-side lane matchups (bot 2v2, top 2v2, mid 1v1) + patch encoding — +1.0% honest forward on winner | ✅ shipped |
| 0.3.14   | Smoothing grid for matchup encoder (negative — default 3.0 already optimal); coverage diagnostic shows mid 1v1 = 82% OOS hit, bot/top 2v2 = 1-2% (still useful) | ✅ shipped |
| 0.3.15   | Per-player features (`PlayerWinRateEncoder`) — winner_v15 = current production | ✅ shipped |
| 0.3.16   | `/api/board` async rewrite (single-flight + 25s wait_for + stale auto-board) + accuracy tracking (`record_prediction` / `score_pending` / `accuracy_summary`, JSONL log) | ✅ shipped |
| 0.3.17   | Playwright `dltv_browser` for live `player.win_rate` (Phase 3) — fetches `dltv.org/matches/{id}/{slug}` HTML, caches to `ml_data/player_wr_cache.json` (5 min TTL) | ✅ shipped |
| 0.3.18   | nginx `map $http_x_api_key $effective_api_key` for dev X-API-Key auto-inject (static UI doesn't need to embed the secret) | ✅ shipped |
| 0.3.19   | Live enrichment TTL 120 s → 5 s so live picks/score don't lag DLTV | ✅ shipped |
| 0.3.20   | Playwright + match-state overlay for in-progress series (`_live_card` synthesizes from `/live/{id}.json`) + chromium binary copied to `/app/.cache/ms-playwright/` (1.61 ignores `PLAYWRIGHT_BROWSERS_PATH`) | ✅ shipped |
| 0.3.21   | Live TTL fix + match-state overlay + nginx X-API-Key + league-filter UI (chip row of top-5, bulk-select in picker) | ✅ shipped |
| 0.3.22   | Live extractor rewrite (image-hash heroes via `window.__heroes` fallback) + chromium subprocess-leak fix (shared browser + per-fetch context + zombie reaper, started eagerly at module import) + strict live filter + auto-board server-side filter (`/api/board` no longer triggers a rebuild for filtered requests — instant) | ✅ shipped |
| 0.3.23   | Real-time live data via `radiant_picks` / `dire_picks` page globals + `#live_scoreboard` (matches DLTV's visual display, no API delay).  Docker build switched to `npmmirror.com` Playwright mirror to dodge `storage.googleapis.com` timeouts. | ✅ shipped |
| 0.3.24   | Live filter hardening: hide 44 steam-only Chinese amateur cards (`LIVE_HIDE_STEAM_ONLY=1`) + map `watch-/steam-` steam_id → dltv series id for cache lookup + dedup Steam+Scraper double-adds in `get_live_and_prematch` + handle both tracker formats (top-level `steam_id` vs `maps[].steam_id`) + bump `MATCH_STATE_TTL_SEC` 5s→30s | ✅ shipped |
| 0.3.24e  | Live picks: dual-id namespace fix in `_picks_to_heroes` / `_bans_to_cards` (Hoodwink's dltv_id 120 was silently matching Pangolier's steam_id 120 → empty `#120` cards with no image) + drop `PLAYER_WR_POLL_INTERVAL_SEC` 30s→5s so the cache keeps up with DLTV's socket.io feed | ✅ shipped |
| 0.3.24f  | Live card lag cut from 5-15s to ~6-8s avg: `_wait_for_live_state` replaces fixed 3.5s `wait_for_timeout` with a `wait_for_function` predicate that returns as soon as `#live_scoreboard` scores or `radiant_picks`/`dire_picks` globals are populated (0.5-2s steady state) + drop `MATCH_STATE_TTL_SEC` 30s→8s so the cache refreshes between publisher ticks | ✅ shipped |
| 0.3.24g  | Live card info parity with DLTV: extract per-team networth (`.team__networth > .networth > span`) → `live_gold: {radiant, dire, lead_radiant}` block + surface `game_time` (MM:SS) + new `towers_over_under` prediction + `liveCard()` renders gold-lead line with green/red arrow + ahead-team name + a/b split, and switches kills/duration/towers predictions to `ТБ`/`ТМ` format (same shape as post-match card). **Does NOT include live destroyed-tower counts** — DLTV only renders those as icons on a small in-game map image, no text/DOM hook; revisit when DLTV redesigns or the socket.io message shape is reverse-engineered | ✅ shipped |
| 0.3.24h  | Live card stays alive after match end: `MATCH_STATE_TTL_SEC` 8s→1h (post-match picks/score/gold are still useful) + `update_match_state_cache` writes both `s{dltv_id}` and `s{steam_id}` alias keys (watchlist finds data without the tracker) + `_live_json_to_series` carries `_dltv_series_id` from the `/live/{id}.json` response (last-resort cache key for finished matches) + cache overlay applies `game_time` / `radiant_networth` / `dire_networth` even when picks are empty (DLTV's late-game state has empty picks but real time/score) + `liveCard` rewritten as a 3-column DLTV-style layout with 44×56 hero icons under each team name and a partial-gold block for the one-side-known case | ✅ shipped |
| 0.3.25   | ML v16 — overnight grid research (1 200+ configs, honest 5-fold CV with encoder refit per fold) → `kills` MAE 11.95→**11.56**, `duration` MAE 9.07→**8.67**, real `winner` honest 60.04% (v15's 67.6% was leaky) | ✅ shipped |
| 0.3.25e-i | UI honesty (hide towers / Ultra Kill), theme switcher, auto-refresh radio, empty-league filter, bans row, nginx cache-control, JS backtick-in-comment fix | ✅ shipped |
| 0.3.25k  | Workspace cleanup — 230+ scratch files + NUL + audit_report.txt + .coverage deleted, .gitignore updated | ✅ shipped |
| 0.3.25l-t | Live card socket.io hook (CSS-rotation robust) + game_time MM:SS normaliser + m[duration] fallback + catch-all publisher + daemon-thread split + TBD/0-picks filters — **all reverted** after user reported "live card disappeared" regression | ↩️ reverted |
| 0.3.25-rollforward | Rollback to `ec61adb` (v0.3.25k) + minimal patches: v0.3.25t-patch (publisher daemon thread, real bug) + v0.3.25l+m re-applied (hook + clock) + v0.3.25r re-applied (TBD filter) + cache trust + v0.3.25l-bugfix (hook filter by steam_id) | ✅ shipped |
| **0.4.0** | **Real-time live data via direct socket.io from Python** (`business/dltv_socket.py`, EIO=4 hand-rolled, no chromium) + WS-PING drop (server kills faster with them) + `_last_good_board` fallback (no more 0/0/0 dead-board) + **parallel `/live/{id}.json` enrichment (40-200× build speedup, 200-500s → 0.8-3.5s)** + version bump `0.3.19` → `0.4.0` | ✅ shipped |
| 0.4.1    | Dual-instance socket redundancy (continuous real-time when one connection drops, ~50 LoC, more complex reconciliation) | 📋 planned |
| 0.4.2    | Async playwright refactor (the proper fix for the chromium greenlet error if we keep the page-load path for player WR) + Cookie-based SSE auth (browser `EventSource` doesn't support custom headers, public-deploy blocker) | 📋 planned |
| 0.5.0    | Postgres migration + auto-retrain + observability (Prometheus + Grafana + OTel) | 📋 planned |
| 1.0.0    | First production-ready release              | 🎯 goal     |

### Open backlog (any minor)

- [x] **Direct DLTV socket.io client** — ✅ **shipped in 0.4.0**
      (`business/dltv_socket.py`, EIO=4 hand-rolled, no `python-socketio`
      dep, no chromium).  Lives 2-3 min ON / 5-30 s OFF in app context
      (server-side limit).  Real-time when alive, `/live/{id}.json`
      fallback when dead.  50-channel standalone test survived 120 s+.
- [x] **`_last_good_board` fallback in publisher thread** — ✅
      **shipped in 0.4.0** (`b45e46d`).  Empty build serves last
      non-empty board if < 5 min old.  No more 0/0/0 dead-board
      during 200-500 s build cycles (which used to happen 1× / cycle).
- [x] **Parallel `/live/{id}.json` enrichment (40-200× build speedup)**
      — ✅ **shipped in 0.4.0** (`7c0b178`).  200-500 s → 0.76-3.5 s
      per cycle.  `ThreadPoolExecutor(max_workers=6)` +
      `get_live_json` retries=1/timeout=1.5s.  Synthesised
      `_live_enrich_failed` series so the card stays in the board
      when the underlying JSON times out.
- [x] **Live data fallback chain** — ✅ **partially shipped in 0.4.0**.
      Currently: socket.io (when alive, real-time) → `/live/{id}.json`
      (cached 5 s, last-good-board fallback if empty) → Steam
      `GetLiveLeagueGames` (only team/series, no picks) → card with
      `_live_enrich_failed` flag.  No "live state unavailable" UI
      surface yet (the card is still rendered, just with empty picks).
- [ ] **Dual-instance socket redundancy** — DLTV's server-side
      connection limit is 30-150 s per session.  Run 2 sockets in
      parallel with round-robin: continuous real-time data even when
      one connection drops.  ~50 LoC.  Trade-off: more complex
      reconciliation if both deliver different `live_score` for
      the same `match_id`.  0.4.1 candidate.
- [ ] **Async playwright refactor** — the proper fix for the
      chromium greenlet error that surfaces under load.  Single-worker
      executor + `_is_browser_alive()` probe only partially mitigate.
      Real fix is `async_playwright` so greenlets yield properly.
      Defer until we measure an actual production incident.  0.4.2.
- [ ] **Cookie-based SSE auth** — `/api/stream/*` is currently
      unauthed for local use (intentional, see `UNAUTHED_PREFIXES`).
      0.4.2 = add a real login endpoint + signed cookie + remove
      from `UNAUTHED_PREFIXES`.  Public deploy is blocked on this.
- [ ] **Multikill classifier revisit** — discontinued in 0.3.10a
      because the 1111-match pro corpus had zero "Low" matches
      (the model collapsed to "always High").  Needs a bigger
      corpus with amateur/pub games to get the rare-multikill
      examples.  Defer until corpus >5k matches with `multikill`
      labels.
- [ ] **Towers regressor** — schema in place, but
      `full_matches.json` doesn't carry the tower bitmask.
      DatDota's `tower_radiant` / `tower_dire` are 11-bit
      bitfields; once we have them in the corpus the model can
      train.  Most predictive feature we haven't used.
- [ ] **Audit P1-15** — `except Exception` sites in business
      modules that swallow + log without re-raise.  Behaviour
      is intentional in places (per-card loops, ML fallback)
      but a few are masks for real bugs that never got fixed
      because the catch hid them.  Audit before 1.0.

¹ Calibration plumbing landed but the empirical run on the
1111-match corpus showed Platt and isotonic both overfit; the
v1 un-calibrated LogReg stays the default.  Revisit when the
corpus grows past 10k matches.

² SSE local-only fix: `/api/stream/*` bypasses auth for
LAN use.  **Public deployment requires cookie-based auth** —
remove from `UNAUTHED_PREFIXES` and add a login endpoint
(planned for 0.4.0).

²ᵃ Multikill classifier landed but the model collapses to
"always High" on the 1111-match pro corpus (zero "Low" matches).
The pipeline is in place; 0.3.1 will add the data + features
that give the classifier an actual signal.

³ 0.3.4 introduced `business/exceptions.py` with a 15-class
hierarchy rooted at `DotaAnalystError`.  22 of the 31
`except Exception` sites were narrowed; the remaining 9 are
deliberately broad (per-card loops + ML-engine fallbacks).
Backward-compatible: every new class still inherits from
`Exception`, so old catches keep working.

⁴ Audit P1-7 (CI check `CORS_ORIGINS` not `localhost`) is
deferred — the project is pre-release / local-only (SSE auth
itself is bypassed for `/api/stream/*` since 0.3.2, cookie-based
auth is the 0.4.0 plan).  A fresh security audit pre-1.0 will
revisit all P1 items, including this one.

---

_Last updated: 2026-07-27 — 0.3.24e: live picks dual-id fix (Hoodwink↔Pangolier collision) + 5s publisher. 0.3.24 (a-d): hide steam-only matches, map watch-/steam- → dltv id, dedup Steam+Scraper, both tracker formats, TTL 5s→30s. 0.3.23: real-time live data via `radiant_picks` / `dire_picks` page globals + `#live_scoreboard` + npmmirror Playwright host. 0.3.22: Docker deploy + DLTV live extractor rewrite + chromium subprocess-leak fix (shared browser + per-fetch context + zombie reaper) + strict live filter + auto-board server-side filter. 0.3.21: live TTL fix + match-state overlay + nginx X-API-Key. 0.3.20: Playwright + match-state overlay for in-progress series + chromium binary in `/app/.cache/ms-playwright/`. 0.3.19: live enrichment TTL 120s→5s. 0.3.18: nginx `map` for dev X-API-Key. 0.3.17: Playwright `dltv_browser` for live `player.win_rate`. 0.3.16: `/api/board` async rewrite + accuracy tracking. 0.3.15: per-player features (`PlayerWinRateEncoder`) — winner_v15. 0.3.10–0.3.14: XGBoost winner + cross-side lane matchups + smoothing grid. 0.3.9: ML accuracy push. 0.3.8: `ARCHITECTURE.md` refresh. 0.3.7: `train.py` tests. 0.3.6: `pip-audit` in CI. 0.3.5: `~=` pins. 0.3.4: domain exception hierarchy. 0.3.3: audit P0 fixes. 0.3.2: SSE auth fix. 0.3.0: multikill classifier. 0.2.2: calibration plumbing. 0.1.1: rate limit + SSE live updates._
