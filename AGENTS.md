# AGENTS.md

Dota 2 professional-match draft analysis and outcome prediction system. Static
frontend + FastAPI backend + heuristic / ML prediction engine. Tournament data
only (no pub games).

## Setup commands

- Install deps:     `pip install -r requirements.txt`
- Run dev server:   `uvicorn business.app:app --reload --host 0.0.0.0 --port 8000` (business only — for full stack, see `docker-compose up`)
- Open UI:          http://localhost:8000 (business) or http://localhost (full stack)
- Collect ML data:  see `scripts/collect_full_matches.py` → `ml_data/full_matches/<match_id>.json`
- Train models:     `python -m business.ml.train` → `ml_data/models/winner_v1/`
- Use the model:    set `PREDICTION_ENGINE=ml` in `.env`, restart business service

## Required environment

`.env` at repo root, auto-loaded by `backend/__init__.py` via `python-dotenv`:
- `STEAM_API_KEY` — https://steamcommunity.com/dev/apikey (Steam Web API)
- `STRAZT_API_KEY` — Stratz GraphQL token (used for draft/lane data)

Both keys are mandatory before `uvicorn` will serve predictions.

## Project layout

- `backend/` — FastAPI app
  - `app.py` — FastAPI routes (`/`, `/api/leagues`, `/api/board`)
  - `board.py` — Kanban board assembly (prematch / live / postmatch cards)
  - `analysis.py` — Heuristic draft analysis engine (`analyze()` — do not change signature)
  - `discovery.py` — Live/prematch match discovery (DLTV scraper + Steam)
  - `dltv_client.py` — DLTV v1 API client (singleton `client`, thread-safe)
  - `datdota_client.py` — DatDota API client (ML data, 3s rate limit, 500/day)
  - `ml/` — ML engine (0.2.0+): `features.py` (target encoding), `storage.py` (versioned model store), `engine.py` (Strategy pattern: `IPredictionEngine` / `HeuristicEngine` / `MLEngine`), `train.py` (CLI)
- `gateway/` — Auth + CORS + body-size + reverse-proxy to business
- `web/` — Static UI (nginx-served): `public/index.html`, `public/app.js`, `public/style.css`
- `ml_data/` — Collected matches + trained models (gitignored, large)
- `scripts/` — One-off data collection + smoke-test utilities
- `tests/` — pytest suite (97 tests as of 0.2.0)
- `docker-compose.yml` — full stack: web + gateway + business
- `RULES.md` — Full architecture / style rules (read before refactoring business/)
- `TODO.md` — Backlog (now includes 0.2.x ML regressors: kills / towers / duration)
- `ARCHITECTURE.md` — Diagrams + design patterns + 0.2.0 ML section

## External APIs (rate limits matter)

| API    | Use                       | Limit                |
|--------|---------------------------|----------------------|
| Steam  | Live match discovery      | 100k calls/day       |
| DLTV   | Events / series / heroes  | Be polite, cache     |
| DatDota| ML data collection        | 500/day, 3s between  |
| Stratz | Draft / lane details      | Per token quota      |

All clients must implement: timeout, retry (3 attempts, exp backoff), and
fallback return on error — never raise to the route handler.

## Code style

- Python: 4-space indent, max line length 100, full type hints, docstrings on
  public functions, `snake_case` / `PascalCase` / `UPPER_CASE` per RULES.md §8.
- `analyze()` in `backend/analysis.py` is the stable contract — keep its
  signature and return shape; evolve the algorithm inside.
- No business logic in route handlers — delegate to `board.py` / `analysis.py`.
- All API clients: encapsulate HTTP, respect rate limits, log + return fallback
  on failure (no `None`, no unhandled exceptions bubbling up).

## Testing instructions

- No test suite is wired up yet. When adding tests, put them under `tests/`
  mirroring the `backend/` layout (`tests/test_analysis.py`, etc.).
- Mock the API clients in `tests/` — never hit Steam / DatDota / Stratz in CI.
- Cover the `analyze()` return shape (winner, kills, duration, towers, …) and
  board assembly edge cases (empty events, watchlist-only, missing data).
- Manual smoke: `uvicorn backend.app:app --reload`, then `curl
  http://localhost:8000/api/leagues` and `/api/board?events=1` should both
  return JSON without 500s.

## Security & secrets

- `.env` is in `.gitignore` — never commit it. Verify with `git status` before
  pushing.
- `STRAZT_API_KEY` and `STEAM_API_KEY` are user-scoped tokens. Do not log
  them, do not echo them in error messages, do not paste them into issues.
- `allow_origins=["*"]` in `app.py` is fine for local dev only — tighten
  before any public deploy.
