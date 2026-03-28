# Packaging and Runtime Modes

Last updated: 2026-03-28

## 1. Runtime strategy

DILIGENT is configuration-first and uses one active runtime file:
- `DILIGENT/settings/.env`

Modes:
- Local mode (default developer workflow)
- Cloud mode (Docker Compose: backend + frontend)
- Desktop packaged mode (Tauri shell + local backend runtime)

Switch modes by copying a profile into `DILIGENT/settings/.env`.

## 2. Runtime profile files

- `DILIGENT/settings/.env.local.example`
- `DILIGENT/settings/.env.cloud.example`
- `DILIGENT/settings/.env.local.tauri.example`
- `DILIGENT/settings/.env` (active)
- `DILIGENT/settings/configurations.json` (non-secret defaults/tuning)

## 3. Environment contract (core keys)

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port. |
| `UI_HOST`, `UI_PORT` | Frontend host/port. |
| `VITE_API_BASE_URL` | Frontend API base path (`/api` recommended). |
| `RELOAD` | Backend auto-reload toggle for local workflow. |
| `DILIGENT_CLOUD_MODE` | Enables cloud hardening behavior. |
| `DB_EMBEDDED` | Embedded SQLite mode toggle. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection settings. |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS settings. |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB timeout and batching. |
| `OLLAMA_URL`, `OLLAMA_HOST`, `OLLAMA_PORT` | Ollama endpoint configuration. |
| `OPTIONAL_DEPENDENCIES` | Optional dependency branch for launcher flow. |
| `ACCESS_KEY_ENCRYPTION_KEY` | Fernet key for encrypted provider key storage. |

## 4. Local mode

1. Activate profile:
   - `copy /Y DILIGENT\settings\.env.local.example DILIGENT\settings\.env`
2. Start application:
   - `DILIGENT\start_on_windows.bat`
3. Optional full test run:
   - `tests\run_tests.bat`

No Docker required.

## 5. Cloud mode (Docker)

1. Activate profile:
   - `copy /Y DILIGENT\settings\.env.cloud.example DILIGENT\settings\.env`
2. Build:
   - `docker compose --env-file DILIGENT/settings/.env build --no-cache`
3. Start:
   - `docker compose --env-file DILIGENT/settings/.env up -d`
4. Stop:
   - `docker compose --env-file DILIGENT/settings/.env down`

Cloud behavior:
- `backend` serves FastAPI internally on `:8000`.
- `frontend` serves static app via Nginx on `:80`.
- Nginx proxies approved `/api/*` routes to backend.
- `/api/docs`, `/api/redoc`, `/api/openapi.json` are blocked.
- `diligent_resources` volume persists app resources.

## 6. Desktop packaged mode (Tauri)

Build entrypoint:
- `release\tauri\build_with_tauri.bat`

Typical flow:
1. Activate desktop profile:
   - `copy /Y DILIGENT\settings\.env.local.tauri.example DILIGENT\settings\.env`
2. Ensure runtimes exist:
   - `DILIGENT\start_on_windows.bat`
3. Ensure Rust toolchain is available (`rustup`).
4. Build package:
   - `release\tauri\build_with_tauri.bat`

Packaged runtime behavior:
- Tauri starts a local Python backend.
- Backend serves packaged frontend from `DILIGENT/client/dist`.
- Runtime setup prefers reusable `runtimes/.venv`, otherwise runs `uv sync --frozen`.
- On shutdown, desktop wrapper terminates backend process tree.

Windows artifacts:
- `release/windows/installers`
- `release/windows/portable`

## 7. Deterministic build notes

- Backend lockfile: `runtimes/uv.lock`.
- Frontend lockfile: `DILIGENT/client/package-lock.json`.
- Docker base images are pinned in:
  - `docker/backend.Dockerfile`
  - `docker/frontend.Dockerfile`
