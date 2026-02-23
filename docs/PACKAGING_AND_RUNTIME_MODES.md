# DILIGENT Packaging and Runtime Modes

## 1. Strategy

DILIGENT uses one active runtime file: `DILIGENT/settings/.env`.

- Local mode: run directly on host without Docker.
- Cloud mode: run with Docker (`backend` + `frontend`).
- Mode switching: replace values in `DILIGENT/settings/.env` only.
- Runtime mode changes are configuration-first; business logic does not branch by mode.

## 2. Runtime Profiles

- `DILIGENT/settings/.env.local.example`: local defaults (loopback host values, embedded DB).
- `DILIGENT/settings/.env.cloud.example`: cloud defaults (container bind host values, external DB).
- `DILIGENT/settings/.env`: active profile used by launcher, tests, and Docker runtime env loading.

## 3. Required Environment Keys

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port. |
| `UI_HOST`, `UI_PORT` | Frontend bind host/port (local preview) and host-published UI port (cloud compose). |
| `VITE_API_BASE_URL` | Frontend API base path. Use `/api` for same-origin proxying in cloud. |
| `RELOAD` | Enables backend reload in local development when `true`. |
| `DB_EMBEDDED` | `true` uses embedded SQLite; `false` uses external DB settings. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection settings used when `DB_EMBEDDED=false`. |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS settings. |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB connection and write-batching runtime settings. |
| `OPTIONAL_DEPENDENCIES` | Enables optional dependency install path in local launcher flow. |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime backend settings for plotting and ML stack. |

## 4. Local Mode (Default)

1. Copy local profile values into active env:
   - `copy /Y DILIGENT\settings\.env.local.example DILIGENT\settings\.env`
2. Start application:
   - `DILIGENT\start_on_windows.bat`
3. Run tests (optional):
   - `tests\run_tests.bat`

Local mode does not require Docker.

## 5. Cloud Mode (Docker)

1. Copy cloud profile values into active env:
   - `copy /Y DILIGENT\settings\.env.cloud.example DILIGENT\settings\.env`
2. Build images (reproducibility check):
   - `docker compose --env-file DILIGENT/settings/.env build --no-cache`
3. Start containers:
   - `docker compose --env-file DILIGENT/settings/.env up -d`
4. Stop containers:
   - `docker compose --env-file DILIGENT/settings/.env down`

Cloud topology:
- `backend`: FastAPI/Uvicorn container (`:8000` internally).
- `frontend`: Nginx container serving static frontend.
- `/api` on frontend origin is reverse-proxied to backend (`backend:8000`) with same-origin API calls.

## 6. Deterministic Build Notes

- Backend dependency graph is lockfile-backed via `uv.lock` and installed with `uv sync --frozen`.
- Frontend dependency graph is lockfile-backed via `DILIGENT/client/package-lock.json` and installed with `npm ci`.
- Docker base images are pinned to explicit tags in `docker/backend.Dockerfile` and `docker/frontend.Dockerfile`.
