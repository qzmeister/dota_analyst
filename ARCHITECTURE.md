# Architecture

> Current architecture as of **0.3.7** (360 tests, 0 known CVE in
> prod deps).  Sections 1–10 are the original 3-node design that
> landed in 0.1.0; section 11–12 cover the ML layer; section 13
> is a compact "what changed since 0.2.0" changelog for the
> architecture, and section 14 enumerates the pre-release
> assumptions that must be revisited before 1.0.

### Document map

1. [Principles](#1-principles)
2. [Topology](#2-topology)
3. [Node specifications](#3-node-specifications)
4. [Design patterns — where each lives](#4-design-patterns--where-each-lives)
5. [Security](#5-security)
6. [Data flow](#6-data-flow)
7. [Local development](#7-local-development)
8. [Migration plan](#8-migration-plan-from-monolith-to-3-node) — status
9. [Resolved decisions](#9-resolved-decisions-2026-07-24)
10. [Still open](#10-still-open)
11. [ML layer (0.2.0)](#11-ml-layer-020)
12. [ML — multi-target regressors (0.2.1)](#12-ml--multi-target-regressors-021)
13. [Releases since 0.2.0 — architectural impact](#13-releases-since-020--architectural-impact)
14. [Local-only / pre-release assumptions](#14-local-only--pre-release-assumptions)

---

## 1. Principles

These constraints shape every decision below.

1. **Frontend is dumb.** No business logic, no API keys, no calculations.
   All data comes from REST or WebSocket. The browser renders — that's it.
2. **Gateway is the only public surface.** Nothing in the internal network
   is exposed. The gateway enforces auth, rate limit, schema, size limits
   and is the only place where CORS / origin policy lives.
3. **Business logic + ML live on a private network.** No public route
   hits them directly. They trust the gateway's requests and assume the
   gateway has already validated input.
4. **Data is the bottom layer.** Every other layer depends on it; nothing
   else does. The data layer never calls outward.
5. **Loud failures, silent recovery.** External API failures are logged
   with full context; circuit breakers stop cascading errors. User-facing
   responses always degrade gracefully (cached or fallback values).
6. **Patterns are explicit.** Where a pattern is used, it's named and
   justified in the code review checklist. We don't sprinkle decorators
   for fun.

---

## 2. Topology

```
                    PUBLIC INTERNET
                          │
                          ▼
       ┌──────────────────────────────────┐
       │   Web  (nginx, static bundle)    │   :80 / :443
       │   - HTML, CSS, JS                │
       │   - WebSocket client             │
       │   - Read-only fetch of REST/WS   │
       └──────────────┬───────────────────┘
                      │  HTTPS (REST + WSS)
       ┌──────────────▼───────────────────┐
       │  Gateway  (FastAPI, uvicorn)     │   :8000
       │  - Auth / API keys               │
       │  - Rate limit (token bucket)     │
       │  - CORS, body size, schema       │
       │  - Circuit breaker to Business   │
       │  - WebSocket origin check        │
       │  - Correlation IDs               │
       └──────────────┬───────────────────┘
                      │  private network (HTTP, mTLS in prod)
       ┌──────────────▼───────────────────────────────────────┐
       │  Business + ML  (FastAPI workers, uvicorn)           │   :8000
       │  - Prediction service (Strategy: Heuristic | ML)     │
       │  - Data collection service (adapters for DLTV etc.)  │
       │  - Repository layer (Postgres + files)               │
       │  - Pub/sub producer → Redis                          │
       │  - Scheduler (APScheduler / Celery beat)              │
       └──────────────┬────────────────────────┬───────────────┘
                      │                        │
       ┌──────────────▼──────────┐   ┌─────────▼────────┐
       │  Postgres 15            │   │  Redis 7          │
       │  - matches, predictions  │   │  - pub/sub         │
       │  - team_stats, hero_idx  │   │  - rate counters   │
       │  - model_metadata        │   │  - session cache   │
       └─────────────────────────┘   └────────────────────┘
                      │
       ┌──────────────▼──────────┐
       │  Object storage (S3)    │  (prod only)
       │  - ml_data/full_matches │  raw match JSONs
       │  - trained model blobs  │  (sklearn joblib)
       └─────────────────────────┘
```

### Why three nodes, not two

- **Two** (frontend + backend) leaves you with one process that does
  routing, validation, business logic and ML. That's a single point of
  failure and a single point of scaling pain.
- **Three** lets each layer scale independently and gives the gateway
  a single, narrow job: protect the inside from the outside.

### Why the gateway is *not* a pure proxy

- Proxies (nginx, Envoy) are great for TLS, rate limits and routing.
- But we also need **schema validation** (Pydantic) and **business
  auth rules** (e.g. "this user can only see their own watchlist").
  A proxy can't do that without Lua/JS plugins. Putting it in a small
  FastAPI service keeps the rules in Python where the rest of the
  codebase lives.

---

## 3. Node specifications

### 3.1 Web (frontend)

| Aspect        | Decision                                                                 |
|---------------|--------------------------------------------------------------------------|
| Stack         | Static HTML + minimal JS. (Optional: switch to React/Vue if complexity grows.) |
| Build         | Vite or just a static directory. No server-side rendering.                |
| State         | Local component state only. No Redux/Pinia unless absolutely needed.      |
| API calls     | `fetch()` for REST commands, native `EventSource` for live updates (SSE).  |
| Auth tokens   | Stored in `httpOnly` cookies (set by gateway) — never `localStorage`.    |
| Caching       | `Cache-Control` headers from gateway; service-worker only for assets.     |
| Bundle size   | Budget: < 200 KB gzipped.                                                 |
| Logging       | Browser console + forwarded to gateway via `X-Client-Log` header (opt-in). |
| Build artifact| `Dockerfile.web` → `nginx` image, no Python anywhere.                    |

**Hard rule:** if you find yourself writing `if (user.role === …)` in JS,
stop. Push the rule to the gateway and let the response shape drive the UI.

### 3.2 Gateway

| Aspect          | Decision                                                                  |
|-----------------|---------------------------------------------------------------------------|
| Framework       | FastAPI + uvicorn (1 worker, multiple async handlers)                     |
| Public port     | `:8000` (behind nginx for TLS termination in prod)                       |
| Auth            | `X-API-Key` header for service-to-service, OAuth2/JWT for end-users (later) |
| Rate limit      | Token bucket per API key + per IP, Redis-backed. 60 req/min default.      |
| CORS            | Whitelist from `CORS_ORIGINS` env var; credentials disabled.              |
| Body size       | 1 MB max for REST, 64 KB for WS messages.                                 |
| Schema          | Pydantic v2 on every request body and query string.                       |
| Idempotency     | `Idempotency-Key` header support on POST endpoints.                       |
| Logging         | Structured JSON, one line per request, includes correlation ID.            |
| Tracing         | OpenTelemetry, exports to OTLP collector.                                 |
| Metrics         | Prometheus `/metrics`, RED + USE method.                                  |
| Health          | `/healthz` (liveness) and `/readyz` (DB ping + Redis ping).               |
| WebSocket       | Same auth, origin check, per-connection rate limit.                       |
| SSE endpoint    | `/api/stream/matches` — `text/event-stream`, auth via query param or cookie, per-connection rate limit. Client uses `EventSource`. |
| Patterns: **Facade**       | The whole service is a facade for the Business API.            |
| Pattern: **Adapter**      | One adapter per Business endpoint to map schemas.              |
| Pattern: **Middleware**   | Composition of: correlation-id → logging → CORS → rate-limit → auth → body-size → handler. |

**Hard rule:** zero business logic in the gateway. It validates, routes,
and protects. It does not compute predictions or aggregate data.

### 3.3 Business + ML

| Aspect         | Decision                                                                 |
|----------------|--------------------------------------------------------------------------|
| Framework      | FastAPI + uvicorn (multiple workers via gunicorn in prod).                |
| Network        | Private subnet, no public IP.                                            |
| Auth           | mTLS + shared HMAC token from gateway (no end-user auth here).            |
| Persistence    | Postgres 15 (primary store) + S3/MinIO (raw match blobs, model artifacts).|
| Cache          | Redis for hot paths (hero index, league metadata).                       |
| Patterns: **Service Layer** | `PredictionService`, `DataCollectionService`, `MatchIngestionService`, `ModelTrainingService`. |
| Patterns: **Repository**    | `IMatchRepository`, `ITeamRepository`, `IModelRepository`. Concrete: `PostgresMatchRepository`, `JsonFileMatchRepository` (transitional). |
| Patterns: **Strategy**      | `IPredictionEngine` with `HeuristicEngine` and `MLEngine` implementations. Selected via `PREDICTION_ENGINE` env var. |
| Patterns: **Factory**       | `IClientFactory` produces API clients (DatDota, Steam, DLTV) with their adapters. |
| Patterns: **Adapter**       | `IDotaDataSource` interface; concrete `DatDotaAdapter`, `SteamAdapter`, `DLTVAdapter`. |
| Patterns: **Decorator**     | `@cached(ttl=…)`, `@retried(max=3)`, `@timed(metric=…)` applied to service methods. |
| Patterns: **Circuit Breaker** | `pybreaker` around every external API call. Closed by default; opens after N failures. |
| Patterns: **DTO / Schema**  | Pydantic models for all internal contracts. **No `dict`s cross layers.** |
| Patterns: **Pub/Sub**      | Redis pub-sub for `match.state.changed` events consumed by gateway.    |
| Patterns: **Scheduler**    | APScheduler in a separate worker process. Nightly retrain, daily data collection. |
| Logging        | Structured JSON, correlation IDs propagated from gateway.                |
| Health         | `/healthz` (liveness), `/readyz` (DB + Redis + model_loaded).            |

**Hard rule:** if the Business service has a `from urllib.request import …`,
it should be a banned import outside `adapters/`. All external I/O goes
through adapters with retry / circuit breaker.

---

## 4. Design patterns — where each lives

| Pattern               | Used in | Purpose                                                          |
|-----------------------|---------|------------------------------------------------------------------|
| **Facade**            | Gateway | Hide internal services from the outside world.                   |
| **Adapter**           | Gateway → Business; Business → External APIs | Translate one interface to another without leaking types.         |
| **Middleware**        | Gateway | Cross-cutting concerns: auth, logging, CORS, rate limit.         |
| **Repository**        | Business | Decouple data access from services; swap Postgres ↔ JSON file.    |
| **Service Layer**     | Business | Encapsulate business operations; testable in isolation.          |
| **Strategy**          | Business (Prediction) | Swap `HeuristicEngine` ↔ `MLEngine` ↔ `EnsembleEngine` at runtime.   |
| **Factory**           | Business (Adapters)   | Create API clients with correct config; easy to mock.             |
| **Decorator**         | Business (Hot paths)  | Add caching, retry, timing, logging without touching logic.        |
| **Circuit Breaker**   | Business (External)   | Stop cascading failures when DatDota / DLTV / Steam goes down.    |
| **DTO / Schema**      | Everywhere            | Single source of truth for data shape (Pydantic v2).             |
| **Pub/Sub**           | Business → Gateway → WS | Push live match updates without polling.                        |
| **Observer**          | Same as Pub/Sub       | Gateway subscribes to events and fans out to WebSocket clients.  |
| **DI**                | Business (Services)   | FastAPI `Depends()`; testable, swappable.                         |
| **Builder**           | Business (ML pipeline) | Complex `MLPipeline(stages=[…])` for training.                  |
| **Command**           | Business (Retrain)    | `RetrainCommand(...)` for background tasks.                       |
| **Singleton**         | Config, Logger, DB pool| One instance per process.                                        |

**Anti-patterns we explicitly avoid:**

- **Anemic domain model** — services must own behaviour, not just pass through.
- **God object** — `board.py` already smells of it (600+ lines); the Service Layer refactor will fix that.
- **Premature microservices** — only split when scaling/policy demands it. Internal calls stay HTTP/JSON for now (no gRPC overhead).
- **Singleton for state** — only for stateless config and pool handles.

---

## 5. Security

Threat model: the project will be exposed to the public internet once
it has prediction value. Assume hostile clients.

### 5.1 Layers

| Layer            | Threats addressed                              | Controls                                |
|------------------|------------------------------------------------|-----------------------------------------|
| Edge (nginx)     | DDoS, TLS, malformed HTTP                      | TLS 1.3, body size, rate limit, fail2ban|
| Gateway          | Auth, CORS, OWASP top-10                       | Schema validation, Pydantic, auth, RBAC |
| Business         | SQLi, business-logic abuse, ML poisoning       | Parameterised queries, input bounds, anomaly detection on features |
| Data             | Backups, encryption                            | Encrypted at rest, automated backups, PITR |

### 5.2 Gateway hardening checklist

- [ ] All requests have a `X-Request-Id` (generated by gateway if missing)
- [ ] All responses echo the request ID for client-side correlation
- [ ] CORS allowlist is per-environment, never `*` in prod
- [ ] All endpoints return JSON; never return internal stack traces
- [ ] Secrets only in env / secret manager, never in error messages
- [ ] No PII logged (matches are public, but user watchlists are not)
- [ ] `Content-Security-Policy` header set on web responses (nginx side)
- [ ] `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options` headers
- [ ] Rate limit: 60 req/min default, 10 req/sec burst, per API key + per IP
- [ ] WebSocket: origin check, message size limit, ping every 30s, idle timeout 5 min
- [ ] Failed-auth logging with throttle (don't let attackers fill our logs)
- [x] Dependency scan on every CI run: `pip-audit` — **landed 0.3.6** (trivy still pending)
- [ ] SBOM generated per release
- [ ] Penetrations tests: run `zap-baseline.py` against staging monthly
- [ ] **Remove `/api/stream/*` from `UNAUTHED_PREFIXES`** before public deploy — see §14

> **Local-only caveat.**  As of 0.3.7 the gateway ships with
> `UNAUTHED_PREFIXES = ("/api/stream/",)` — the SSE endpoint is
> auth-bypass for the browser's `EventSource` API (which can't
> send custom headers).  This is safe on a trusted LAN but is a
> DDoS vector on the public internet.  Cookie-based auth is
> planned for 0.4.0.  See §14.

### 5.3 Business service hardening

- [ ] No public network ingress. Only gateway can call.
- [ ] All SQL via SQLAlchemy / parameterised queries
- [ ] ML feature inputs bounded (e.g. hero IDs 1..200, win_rate 0..100)
- [ ] Anomaly detection on input distributions (alert on 100× rate spike)
- [ ] No `eval` / `exec` / `pickle.load` on untrusted data
- [ ] Model artifacts signed and verified on load

---

## 6. Data flow

### 6.1 Read path: load the board

```
Browser                 Web                Gateway              Business              Postgres
   │  GET /board         │                    │                     │                     │
   ├─────────────────────▶                    │                     │                     │
   │                     │  GET /api/board    │                     │                     │
   │                     ├───────────────────▶│  (auth, rate limit)  │                     │
   │                     │                    │  GET /internal/board │                     │
   │                     │                    ├────────────────────▶│  SELECT …           │
   │                     │                    │                     ├────────────────────▶│
   │                     │                    │                     │◀────────────────────┤
   │                     │                    │◀────────────────────┤                     │
   │                     │◀───────────────────┤                     │                     │
   │◀────────────────────┤                    │                     │                     │
   │  render Kanban      │                    │                     │                     │
```

### 6.2 Live update path: a goal is scored

```
DLTV.org  ──── poll ────▶  Business.DiscoveryService
                                │
                                │  match.state.changed
                                ▼
                           Redis pub-sub
                                │
                                ▼
                           Gateway.SSEHub
                                │
                                │  fan out (text/event-stream)
                                ▼
                           Browser EventSource clients
```

Why SSE (not WebSocket) for live updates: the only direction that matters
is server → client (state changes). Client → server commands go over REST
POST. This split keeps the surface area smaller — no bidirectional protocol
to harden, no separate upgrade headers in nginx, no ping/pong keepalives.

### 6.3 Write path: collect training data

```
Business.DataCollectionService
        │
        │  every 24h (APScheduler)
        ▼
   DatDotaAdapter.fetch_recent_matches()
        │
        │  raw match JSON
        ▼
   Postgres "matches" table
        │
        │  on >1000 new rows
        ▼
   ModelTrainingService.run()
        │
        │  new joblib artifact + metadata.json
        ▼
   S3 / model registry
```

---

## 7. Local development

### 7.1 Docker Compose layout

```yaml
# docker-compose.yml (planned for 0.1.0)
services:
  web:
    build: { context: ., dockerfile: docker/web.Dockerfile }
    ports: ["80:80"]
    depends_on: [gateway]
    networks: [frontend]

  gateway:
    build: { context: ., dockerfile: docker/gateway.Dockerfile }
    ports: ["8000:8000"]   # remove for prod, expose only via nginx
    environment:
      - BUSINESS_URL=http://business:8000
      - REDIS_URL=redis://redis:6379/0
      - CORS_ORIGINS=http://localhost:80
    depends_on: [business, redis]
    networks: [frontend, backend]

  business:
    build: { context: ., dockerfile: docker/business.Dockerfile }
    environment:
      - DATABASE_URL=postgresql://dota:dota@postgres:5432/dota
      - REDIS_URL=redis://redis:6379/0
      - STEAM_API_KEY=${STEAM_API_KEY}
      - STRAZT_API_KEY=${STRAZT_API_KEY}
      - PREDICTION_ENGINE=heuristic   # switch to "ml" in 0.2.x
    depends_on: [postgres, redis]
    networks: [backend]   # NOT exposed publicly

  postgres:
    image: postgres:15-alpine
    volumes: [pgdata:/var/lib/postgresql/data]
    networks: [backend]

  redis:
    image: redis:7-alpine
    networks: [backend]

volumes:
  pgdata:

networks:
  frontend:    # exposed to host
  backend:     # internal only
```

### 7.2 Dev workflow

```bash
# one-time
docker compose up -d postgres redis
python -m pip install -r requirements-dev.txt

# everyday
docker compose up gateway business web
pytest -q
```

---

## 8. Migration plan (from monolith to 3-node)

Step-by-step so the architecture shift doesn't break the running app.

### Phase 1 — foundations (target 0.0.2) — ✅ done

- ✅ `pyproject.toml` for proper packaging
- ✅ `pytest -q` in `Makefile`
- ✅ `.env.example` documenting every env var
- ✅ Logger (stdlib `logging` with JSON formatter in `business/_logging.py`)

### Phase 2 — extract the gateway (target 0.1.0-alpha) — ✅ done

1. ✅ **New service `gateway/`** — proxies `/api/*` to the business service, runs the full middleware chain (CORS, auth, body size, access log, correlation id).
2. ✅ **Refactor `backend/` → `business/`** — same code, new home, no public routes. Plus `/api/healthz` and `/api/readyz`.
3. ✅ **Move `frontend/` out of FastAPI and into nginx + `Dockerfile.web`** — web/public/ + web/nginx.conf + docker/Dockerfile.web.
4. ✅ **Wire `docker-compose.yml`** — three services, two networks, healthchecks.
5. ⏭ **Run both in parallel** — skipped for 0.1.0; clean cutover instead.
6. ⏭ **Cut over** — done at 0.1.0 release; old monolith is gone.

### Phase 3 — apply the patterns (target 0.1.0 + 0.2.0) — done

- ✅ `IPredictionEngine` Strategy with `analysis.py` as `HeuristicEngine` + new `MLEngine` — **done in 0.2.0**
- ⏭ Wrap external I/O in `IDotaDataSource` adapters — still pending; the codebase uses module-level clients (`dltv_client.client`, `discovery.tracker`) instead.  Refactor candidate for 0.4.x.
- ⏭ Replace `dict`-passing with Pydantic DTOs between gateway and business — pending; tradeoff vs simplicity.
- ⏭ Add `pybreaker` around every external call — pending; the `except DotaAnalystError` catches in 0.3.4 are a step in this direction.
- ✅ In-process SSE pub-sub for live updates — **done in 0.1.1** (`business/stream.py` + `GET /api/stream/matches`).  Redis upgrade still pending for cross-process scaling.

### Phase 4 — security hardening (target 0.1.0) — mostly done

- ✅ `X-API-Key` auth on all `/api/*` (gateway middleware)
- ✅ Security headers at nginx (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`)
- ✅ CORS allowlist from env
- ✅ Body size limit
- ✅ Rate limiting — **done in 0.1.1** (`gateway/_rate_limit.py`, token bucket, 60 rpm / 10 burst).  Redis-backed multi-instance counters still pending.
- ⏭ OWASP top-10 review of every endpoint
- ✅ `pip-audit` in CI — **done in 0.3.6** (workflow at `.github/workflows/pip-audit.yml`); 0 known CVEs.
- ⏭ `trivy` in CI
- ⏭ HSTS, full CSP
- ⏭ Cookie-based SSE auth — **deferred to 0.4.0** (currently `/api/stream/*` is in `UNAUTHED_PREFIXES`; safe on LAN only)

### Phase 5 — ML foundation (target 0.2.0 + 0.2.1) — done

- ✅ New `MLEngine(IPredictionEngine)` impl, behind the same Strategy
- ✅ `ModelStorage` with versioned artifacts + `metadata.json` sidecar
- ✅ A/B harness — `scripts/eval_engines.py` — Heuristic vs ML on 1111 matches
- ✅ Multi-target regressors (kills / duration_mean / P10 / P90) + MAD-based winsorize — **0.2.1**
- ⏭ Towers regressor (corpus has no tower bitmask) — **partial: ZINB factory landed in 0.2.2 but no tower-aware corpus yet**
- ⏭ Champion / Challenger deployment — needs Redis pub-sub (deferred to 0.4.x)
- ✅ Calibration plumbing (Platt / isotonic) — **0.2.2**, but empirical run on 1111 matches showed overfit; v1 un-calibrated stays the default
- ✅ Multikill classifier + first-to-15 scaffolding — **0.3.0** (multikill degenerated on pro corpus; needs richer data)

---

## 11. ML layer (0.2.0)

The 0.2.0 release wires a Strategy-pattern prediction engine and ships
the first trained model. Layout under `business/ml/`:

```
business/ml/
├── __init__.py
├── features.py    # HeroWinRateEncoder, FEATURE_ORDER, extract_features
├── storage.py     # ModelStorage — versioned joblib + metadata.json sidecar
├── engine.py      # IPredictionEngine, HeuristicEngine, MLEngine, make_engine
└── train.py       # CLI: python -m business.ml.train
```

### Why a Strategy, not a single "MLEngine" replacement

The 0.0.x heuristic was a well-calibrated set of rules for every
prediction target (winner, kills, towers, duration, first-to-15,
multikill). Replacing it wholesale with a black-box model would lose
that calibration. Instead:

- `HeuristicEngine` (default) is a thin wrapper around the existing
  `analysis.analyze()`. Output is byte-for-byte identical to the v0.0.x
  behaviour.
- `MLEngine` starts from the heuristic result and **overrides one block
  at a time**. In 0.2.0 only the `winner` block is overridden. Future
  minors add `kills` / `towers` / `duration` overrides incrementally.

This keeps every release reviewable: a 0.2.x diff that adds a kills
regressor only changes the `kills` block, never touches the winner
logic.

### Feature contract

`features.py` is the single source of truth for the model's input
schema. Two immutable names flow through both training and prediction:

```python
FEATURE_ORDER: Tuple[str, ...] = (
    "mean_hero_wr_radiant", "mean_hero_wr_dire",
    "hero_wr_r_0", "hero_wr_r_1", "hero_wr_r_2", "hero_wr_r_3", "hero_wr_r_4",
    "hero_wr_d_0", "hero_wr_d_1", "hero_wr_d_2", "hero_wr_d_3", "hero_wr_d_4",
    "radiant_minus_dire",
)
N_FEATURES = 13
```

`HeroWinRateEncoder.fit(matches)` walks the corpus and computes
per-(side, hero_id) win rates with smoothing (`alpha=5.0`) back to the
global rate. Unseen heroes fall back to the global rate, so the
predictor never divides by zero. `encode(side, hero_id)` is a single
dict lookup at predict time.

**No circular features** are allowed. `duration`, `kills`, `deaths`,
`assists`, `gpm`, `xpm`, `gold`/`xp` graphs are explicitly out — all of
them are only known after the match and would be target leakage. This
is documented in the `features.py` docstring and enforced by code review.

### Model storage

```
ml_data/models/
└── winner_v1/
    ├── model.joblib      # the sklearn estimator (joblib serialised)
    └── metadata.json     # sklearn_version, feature_names, metrics, encoder
```

Versioned directories (`{name}_v{version}/`) let us keep a production
model pinned while a freshly trained candidate lives alongside it.
The `metadata.json` sidecar is the source of truth for everything we
want to know about a model **without loading the joblib blob**: training
date, sklearn version, feature names, holdout metrics, the encoder
that goes with the model. Loading refuses if the saved `feature_names`
don't match the live `FEATURE_ORDER` — that footgun is now caught at
load time, not at predict time.

Writes are atomic: the model and the metadata are first dumped to a
`.tmp` file in the target dir, then `os.replace`'d into place. A
half-written artifact can no longer crash the load path.

### Wiring

The engine is built once at `business.app` startup. `get_default_engine()`
is a process-wide lazy singleton; `reset_default_engine()` lets tests
rebuild it after monkey-patching the env.

```python
# business/app.py
@app.on_event("startup")
def _warm_prediction_engine() -> None:
    engine = get_default_engine()
    log.info("prediction engine ready: %s", engine.name)
```

`/api/board` now returns an `"engine"` field in the response so the
caller can confirm which engine produced the predictions.

### 0.2.0 results (sanity)

- 1111 DatDota matches, balanced labels (549 radiant / 562 dire)
- 80/20 stratified split → 888 train / 223 test
- `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")` on 13 features
- **Test accuracy 0.686, ROC AUC 0.769, log loss 0.591** (vs 0.693
  baseline for a constant 50/50 predictor)
- Smoke test on 5 historical matches: ML 4/5, heuristic 2/5
  (heuristic was given no team aggregates in the smoke test — its
  real signal lives in DLTV live data)

---

## 12. ML — multi-target regressors (0.2.1)

0.2.1 extends the `IPredictionEngine` from winner-only to four
regression heads (kills / duration mean / P10 / P90) plus the
classifier.  New files:

```
business/ml/
├── features.py      # (unchanged from 0.2.0) — HeroWinRateEncoder, FEATURE_ORDER
├── storage.py       # (unchanged) — versioned joblib + metadata.json
├── engine.py        # MLEngine now multi-target; sub-models dict
├── train.py         # CLI: python -m business.ml.train --target {winner,kills,duration_mean,duration_p10,duration_p90,all}
├── outliers.py      # NEW — MAD-based robust 3σ winsorize
├── targets.py       # NEW — per-match label extractors (kills_total, duration_minutes, ...)
└── regressors.py    # NEW — sklearn / xgboost factory functions
```

### Per-target estimators

The 0.2.0 modelling shortlist asked for Tweedie / Poisson / Gamma
variances.  sklearn 1.9's `HistGradientBoostingRegressor` does
NOT expose a Tweedie loss (it supports `squared_error`,
`absolute_error`, `gamma`, `poisson`, `quantile`).  The 0.2.1
release uses the closest match from the available losses; the
Tweedie upgrade path is queued for 0.2.2 via XGBoost's
`reg:tweedie` (which does support it natively).

| Target            | Estimator                                         | Loss       | Why this loss |
|-------------------|---------------------------------------------------|------------|---------------|
| `kills`           | `HistGradientBoostingRegressor`                   | `poisson`  | Count data, non-negative, mean ≈ variance |
| `duration_mean`   | `HistGradientBoostingRegressor`                   | `gamma`    | Positive continuous with long right tail |
| `duration_p10`    | `XGBoost XGBRegressor(quantile_alpha=0.1)`        | `reg:quantileerror` | Calibrated P10 in one shot |
| `duration_p90`    | `XGBoost XGBRegressor(quantile_alpha=0.9)`        | `reg:quantileerror` | Calibrated P90 in one shot |
| `towers`          | (deferred — corpus has no per-side tower bitmask)  | —          | 0.2.2 with re-pulled DLTV corpus |

`MLEngine.analyze()` checks which sub-models are present and
overrides only the matching blocks.  Blocks whose sub-model is
missing or fails to predict stay heuristic — a half-trained
`MLEngine` is still useful and rollouts stay safe.

### Winsorize — MAD-based robust 3σ

`business.ml.outliers.winsorize_values(values, n_sigma=3.0)` clips
each value to `[median - n_sigma * 1.4826 * MAD,
median + n_sigma * 1.4826 * MAD]`.

Why MAD and not `mean ± 3*std`?  Naïve 3σ has a known failure mode
on heavy-tailed data: the outliers inflate `std`, which inflates
the clip bounds, which fails to clip the very outliers we wanted
to clip.  On a synthetic sample of [bulk in 20..50, tail in
200..1000] the empirical std is dominated by the tail and the
clip becomes a no-op.  MAD (Median Absolute Deviation) is the
textbook robust scale estimator: it is unaffected by up to 50%
of the data being outliers.  The `1.4826` constant maps MAD to
sigma on a Gaussian so the `n_sigma=3.0` knob still means
"three standard deviations" on a clean distribution.

Winsorize is applied to the **train** side only; the test side
stays raw so the metrics reflect what the model actually sees in
production.  A typical 883-match run clips ~8 kills values and
~14 duration values — a 1-2% trim, enough to focus the regressor
on the bulk.

### Eval harness

`scripts/eval_engines.py` walks every match in
`ml_data/full_matches/`, runs both engines on the same inputs,
and prints a side-by-side table:

```
======================================================================
  metric                      heuristic             ml   delta
======================================================================
  winner accuracy                0.4941         0.6904   ++0.1962  <-- ml better
  winner log_loss                0.6931         0.7575   ++0.0644  <-- heur better
  kills MAE                     10.9442         3.7948   -7.1494  <-- ml better
  kills RMSE                    14.7224         7.8376   -6.8848  <-- ml better
  duration MAE                   9.2521         3.3077   -5.9444  <-- ml better
  duration RMSE                 12.9630         7.4871   -5.4760  <-- ml better
```

Run it after every model upgrade; the table is the regression
test for "did the new model actually beat the heuristic?".  The
heuristic's `log_loss` of 0.693 is its "I always say 50/50"
ceiling — perfectly calibrated because it has no signal to be
miscalibrated about.

---

## 9. Resolved decisions (2026-07-24)

| Question                | Decision                                                          | Rationale                                                                              |
|-------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Live protocol           | **SSE for server→client, REST for client→server commands**        | One-way state push is enough; keeps nginx config simple, no upgrade dance.             |
| Auth strategy (local)  | **Single static dev token in `.env` (header `X-API-Key`)**         | Code is auth-aware from day one; only the token source differs in prod.                |
| Auth strategy (prod)    | API keys per user + HMAC between services                          | Standard for service-to-service; revisit for user-facing features in 0.2.x.            |
| Primary storage         | **JSON files first via `JsonFileRepository`, Postgres later**      | Repository pattern lets us swap the backing store without touching services.          |
| Object storage          | Local folder `ml_data/` mounted as Docker volume; S3 in prod       | Same `IObjectStore` interface — adapter swap only.                                     |
| Observability           | Structured JSON logs to stdout; Prometheus later                    | Right-sized for local. Add `prometheus_client` middleware when we have prod traffic.   |
| ML training cadence     | APScheduler nightly + manual CLI trigger; push-based in 0.2.x      | Cron is enough until we have real data; trigger threshold of 1000 matches.              |
| WebSocket usage         | **None** in 0.1.0; revisit if client ever needs push from server   | SSE covers the only use case.                                                          |

---

## 10. Still open

- **Database hosting in prod** — managed (RDS/Cloud SQL) vs self-hosted. Affects backup/PITR strategy. Defer to prod-prep.
- **CDN for static frontend** — Cloudflare/Fastly in front of nginx. Easy win, defer to prod-prep.

---

## 13. Releases since 0.2.0 — architectural impact

A compact changelog of what each post-0.2.0 release changed about
the *architecture* (not features).  See `CHANGELOG.md` for the
full release notes.

### 0.2.1 — multi-target regressors

- New `business/ml/regressors.py` factory per target (kills, duration_mean, p10, p90).  Towers was added in 0.2.2.
- `business/ml/outliers.py` — MAD-based robust 3σ winsorize (Gaussian constant 1.4826).  Applied to train side only; test side stays raw so metrics reflect production.
- `scripts/eval_engines.py` — A/B harness, prints the Heuristic vs ML table on 1111 matches.  This is the regression test for "did the new model actually beat the heuristic?".

### 0.2.2 — calibration + towers scaffolding

- `business/ml/regressors.py::make_towers_regressor_zinb` — statsmodels ZINB factory with HistGBR(Poisson) fallback when statsmodels is missing.
- `business/ml/train.py` — `--calibrate {none,sigmoid,isotonic}` flag wraps the winner LogReg in `CalibratedClassifierCV`.  Empirically didn't help on 1111 matches; v1 un-calibrated stays the default.
- **No production-code change** beyond the `regressors.py` factory; the CLI flag is opt-in.

### 0.3.0 — multikill classifier

- `business/ml/classifiers.py` — 3-class HistGB with `class_weight="balanced"`.  Multikill target derived from per-player max kills, bins matching `analysis.MULTIKILL_HIGH_SCORE=7` / `MULTIKILL_MEDIUM_SCORE=4`.
- `business/ml/targets.py::target_multikill` + `MULTIKILL_*_THRESHOLD` constants.  The classifier and the heuristic now share bins, so the A/B comparison is apples-to-apples.
- **Degenerated on the pro corpus** — 0 "Low" matches out of 1111, so the model only predicts "High".  Pipeline is in place; 0.3.1 will add the data + features that give it signal.  See §14.

### 0.3.2 — SSE auth bypass (local-only)

- `gateway/_middleware.py::UNAUTHED_PREFIXES = ("/api/stream/",)` — the SSE endpoint skips the `X-API-Key` check because the browser's `EventSource` API cannot send custom headers.  **Local-only**; cookie-based auth replaces this in 0.4.0.  See §14.

### 0.3.3 — audit P0 fixes (test coverage)

- New: `tests/test_board.py` (33 unit), `tests/test_app.py` (8 smoke with mocks), `tests/test_discovery.py` expanded from 22 to 53 cases.  **No production code change** — purely test coverage.

### 0.3.4 — domain exception hierarchy

- New: `business/exceptions.py` with a 15-class hierarchy rooted at `DotaAnalystError`.  22 of the 31 `except Exception` sites were narrowed to specific subclasses.  The remaining 9 are deliberately broad (per-card loops + ML-engine fallbacks) and pinned with comments.
- `tests/test_exceptions.py` (34 cases) pins the inheritance graph so accidental refactors surface in review.
- **Backward compatible**: every new class inherits from `Exception`, so old `except Exception` blocks still work.

### 0.3.5 — compatible-release pins

- `requirements.txt` switched from `>=` to `~=` for all 9 prod deps.  Fresh `pip install` lands in the same minor (e.g. 0.139.x), bug-fix releases flow in automatically.  See `CHANGELOG.md` for the full pin matrix.
- `requirements-dev.txt` keeps `>=` for pytest etc. — dev deps don't ship.

### 0.3.6 — `pip-audit` in CI

- New: `.github/workflows/pip-audit.yml`.  Runs on every push and PR.  Production tree (`requirements.txt`) is audited with `--strict`; failures block the build.  Dev tree is audited with `continue-on-error: true` and the report uploaded as an artifact.
- Current state: **0 known CVEs** in the prod tree.  See the workflow file for details.

### 0.3.7 — `train.py` test coverage + bug fix

- New: `tests/test_ml_train.py` (50 cases).  `business/ml/train.py` coverage went from 0% to **95.2%**.
- Bug found and fixed by the new tests: `_train_regressor` was calling `make_regressor(...)` without importing it, and `_train_multiclass_classifier` was doing a local import of `make_classifier` inside the function.  Both moved to the module top.  See `CHANGELOG.md` for the full analysis.

---

## 14. Local-only / pre-release assumptions

The project is currently pre-release and runs on a trusted LAN.
The following "good enough for now" choices are baked into the
code; each must be revisited before public deployment.

| # | Assumption | Where | Pre-1.0 action |
|---|---|---|---|
| 1 | **SSE auth bypass** — `/api/stream/*` is in `UNAUTHED_PREFIXES` | `gateway/_middleware.py` | Cookie-based auth, login endpoint, remove from allowlist (planned 0.4.0) |
| 2 | **Single static dev token** — `DEV_API_KEY` in `.env`, `X-API-Key` header | `gateway/_middleware.py` | API keys per user + HMAC between services |
| 3 | **CORS not pinned in CI** | (deferred) | Add `pip-audit`-style workflow that fails the build on `CORS_ORIGINS=localhost` (audit P1-7) |
| 4 | **Calibration off by default** — Platt/isotonic overfit on 1111 matches | `business/ml/train.py` | Revisit when corpus > 10k matches |
| 5 | **Multikill degenerated** — only "High" class learned (0 "Low" in pro corpus) | `ml_data/models/multikill_v1/` | Retrain on binary or richer data (0.3.1) |
| 6 | **Towers regressor not trained** — corpus lacks DLTV tower bitmask | `ml_data/models/` (no `towers_v1/`) | Re-pull tower-aware corpus, or drop the `towers` head until data is available |
| 7 | **`JsonFileRepository` only** — no Postgres | `business/storage.py` (when added) | Postgres behind the same `IRepository` interface (1.0) |
| 8 | **Object storage = local folder** — `ml_data/` Docker volume | `ml_data/` | S3 behind `IObjectStore` interface |
| 9 | **No edge rate limit** — gateway has the bucket, but nginx/cloud LB is not configured | nginx / infra | Edge rate limit + WAF rules |
| 10 | **Structured logs only, no metrics** | `business/_logging.py` | `prometheus_client` middleware + `/metrics` endpoint (audit P2-9) |
| 11 | **No nightly eval cron** — `scripts/eval_engines.py` is manual | `scripts/eval_engines.py` | Schedule a nightly run, post metrics to a dashboard (audit P2-8) |

This list is the **audit checklist for the 0.4.x → 1.0.0
hardening sprint**.  Items 1, 2, 7, 8 are blocking; the rest are
"should have".  See `TODO.md` §"Local-only / pre-release assumptions"
for the same list in reverse-chronological order with the
operational rationale.

---

_Last updated: 2026-07-24 — 0.3.7: `train.py` tests (50 cases, 0% → 95% coverage). 0.3.6: `pip-audit` in CI. 0.3.5: `~=` pins. 0.3.4: exception hierarchy. 0.3.3: P0 audit test coverage. 0.3.2: SSE auth bypass (local-only). 0.3.0: multikill classifier (degenerated). 0.2.2: calibration + towers. 0.2.1: multi-target regressors._
