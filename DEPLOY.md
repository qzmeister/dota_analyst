# Public deploy guide (v0.4.0.1)

This walks you through taking the local dev stack (docker compose up)
to a public-internet deployment.  The key change from the pre-0.4.0.1
setup is the auth model: `X-API-Key` is no longer the only way in;
we now issue an HMAC-signed `dota_analyst_session` cookie on
`POST /api/auth/login` and require it on the SSE path (where the
browser can't send custom headers anyway).

## 1. Generate a strong API key

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Put it in `.env` as `DEV_API_KEY=...`.  This is the master secret:
  * The X-API-Key the static UI sends (if any).
  * The HMAC key for session cookies.
  * The credential users POST to `/api/auth/login`.

## 2. Tighten the .env

```env
# master API key (see step 1)
DEV_API_KEY=<generated>

# v0.4.0.1 prod-mode toggles
PROD_MODE=1                       # tells nginx NOT to inject the dev key
SESSION_COOKIE_SECURE=1          # add Secure attribute to the session cookie

# CORS allowlist — the only origins that can call /api/*
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# rate limits (60 rpm / 10 burst per (api_key, ip) is the dev default;
# tighten for prod)
RATE_LIMIT_RPM=30
RATE_LIMIT_BURST=5

# logging — JSON is easier to ship to a log aggregator
LOG_FORMAT=json
LOG_LEVEL=INFO
```

## 3. TLS termination

The v0.4.0.1 stack assumes HTTPS is terminated at the edge (an
nginx or a managed LB in front of `web`).  The static assets
themselves are served over plain HTTP on the internal network;
nginx on the host has a separate TLS config (or a managed LB
issues a cert).  Two recipes:

### 3a. nginx in front of the compose stack

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    location / {
        proxy_pass http://127.0.0.1:80;  # the compose-managed `web` container
        # ... standard proxy_set_header for X-Forwarded-Proto etc.
    }
}
```

### 3b. Cloudflare / managed LB

Same idea — terminate TLS at the LB, forward to the `web` container
on port 80.  Set `X-Forwarded-Proto: https` so the gateway emits
`https://` URLs in any redirect.

## 4. nginx inside the `web` container

The compose `web/nginx.conf` already supports prod mode:
  * `env PROD_MODE;` (line 1) — declared in main context.
  * `map "$PROD_MODE:$http_x_api_key" $effective_api_key` — when
    `PROD_MODE=1`, the dev key is never injected, only the
    browser's own key (or the empty string) is forwarded.
  * HSTS header is only emitted when `PROD_MODE=1`.

Recreate the web container after editing the env:

```bash
docker compose up -d web
```

## 5. gateway CORS

The gateway reads `CORS_ORIGINS` and builds a strict allowlist.
The pre-0.4.0.1 default of `http://localhost` blocks any cross-site
request; set this to the public origins (no trailing slash, no
wildcard).  Without a correct CORS origin, browsers will refuse
to attach the session cookie to the EventSource.

## 6. Session cookie

The cookie is set by the gateway on `POST /api/auth/login`:
  * `HttpOnly` — JavaScript can't read it (XSS protection).
  * `SameSite=Lax` — cross-site GETs work, but cross-site POSTs
    (CSRF) don't carry the cookie.  Since the login endpoint is
    POST and uses a JSON body, a CSRF can't trigger login.
  * `Secure` — set when `SESSION_COOKIE_SECURE=1` (the prod toggle).
    Once the edge is on HTTPS, this attribute prevents the cookie
    from being sent over an accidental HTTP downgrade.
  * `Path=/` — sent on every request to the gateway.
  * `Max-Age=604800` (7 days) — see `gateway/_session.py`.

The token is HMAC-SHA256 of `expiry_unix:nonce`, keyed by
`DEV_API_KEY`.  Stateless — no session store, no DB.  The trade-off
is that revoking a leaked cookie requires waiting up to 7 days
or rotating `DEV_API_KEY` (which invalidates every active session).

## 7. User flow

  1. User opens the app at `https://yourdomain.com/`.
  2. The static `app.js` calls `GET /api/auth/status` (cookie
     attached if present, public endpoint otherwise).
  3. If unauthenticated, the login modal appears.
  4. User pastes their API key and clicks "Войти".
  5. `POST /api/auth/login` with body `{"api_key": "..."}`.
  6. On 200, the response sets the `Set-Cookie` header.  All
     subsequent requests (including the EventSource) attach
     it automatically.
  7. If the cookie expires (7 days, or `DEV_API_KEY` rotated), the
     SSE returns 401; `app.js` re-shows the login modal.

## 8. Hardening checklist

  * [ ] `DEV_API_KEY` is a long random string (≥ 256 bits), not the
        shipped dev value.  The dev value is in CHANGELOG.md;
        rotating it is a one-shot cookie invalidation.
  * [ ] `PROD_MODE=1` in the web container's env.
  * [ ] `SESSION_COOKIE_SECURE=1` in the gateway container's env.
  * [ ] `CORS_ORIGINS` is the public origin (or list of), no `*`.
  * [ ] TLS cert auto-renews (Let's Encrypt, ACM, etc.).
  * [ ] HSTS preloaded (https://hstspreload.org/) if you're sure
        you'll never serve over plain HTTP.
  * [ ] Rate limits tightened: `RATE_LIMIT_RPM=30` (or lower),
        `RATE_LIMIT_BURST=5` for the default policy; consider
        per-user overrides if you have many users on the same
        IP (e.g. a corp NAT).
  * [ ] Logs ship to a central aggregator (CloudWatch, Loki,
        Datadog, etc.) with `LOG_FORMAT=json`.
  * [ ] Docker images pinned to a digest, not `latest`, so a
        rebuild of `latest` can't silently swap code.
  * [ ] Backup of `ml_data/` (model files + per-hero WR cache).
        Retraining is hours of Phase-3 work; lose the dir, lose
        the predictions.

## 9. Why no full user-table / OAuth / etc.?

The v0.4.0.x scope is a single-operator app.  Adding a real user
table means:
  * Password storage (Argon2 + per-user salt).
  * Session rotation on login.
  * Logout-everywhere.
  * Forgot-password flow.
  * Email verification.

Each of those is its own subsystem.  For a single admin sharing
access with a small team, a shared `DEV_API_KEY` + the login
modal is enough.  When the team grows past ~5 people, swap in
real user accounts — the cookie format (`HMAC-SHA256 over
expiry:nonce`) is already the right shape to add a `user_id`
field to, so the cookie path survives the migration.
