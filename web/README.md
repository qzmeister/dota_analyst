# Web (frontend)

Static frontend served by nginx. **Zero Python here** — this image
exists to be the dumb, terminal layer of the stack.

## Layout

```
web/
├── public/         # static assets served at /
│   ├── index.html
│   ├── app.js
│   └── style.css
├── nginx.conf      # production nginx config
└── README.md
```

## Local dev

```bash
# from project root
docker compose up web gateway business
# open http://localhost
```

## Production

Build the image once and serve behind your edge (Cloudflare / Fastly
when you have them):

```bash
docker build -f docker/Dockerfile.web -t dota-analyst-web:0.1.0 .
```

The image exposes :80 and proxies `/api/*` to the gateway service
(over the internal `backend` network). No public port other than 80
is exposed.

## Constraints

- **No business logic** in JS — every computation belongs in the
  business service. The frontend is allowed to: render, validate
  form input, time animations.
- **No API keys in JS** — auth tokens are set by the gateway as
  `httpOnly` cookies.
- **No direct calls to DLTV / Steam / DatDota** — the frontend only
  talks to the gateway, and the gateway is the only thing that
  knows about external APIs.
