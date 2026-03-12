# DILIGENT Packaging And Runtime Modes

## 1. Strategy

DILIGENT uses a single active runtime file:
- `DILIGENT/settings/.env`

Runtime switching is configuration-only:
- Local mode: host runtime (default developer flow).
- Cloud mode: Docker Compose (`backend` + `frontend`).
- Packaged desktop mode: Tauri desktop shell + bundled Python/uv runtime + backend-served SPA.
- Switch mode by copying the desired profile into `DILIGENT/settings/.env`.

No code changes are required when switching modes.

## 2. Runtime Profiles

- `DILIGENT/settings/.env.local.example`
- `DILIGENT/settings/.env.cloud.example`
- `DILIGENT/settings/.env.local.tauri.example`
- `DILIGENT/settings/.env` (active runtime configuration)
- `DILIGENT/settings/configurations.json` (non-secret defaults and service tuning)

## 3. Environment Key Contract

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host and port. |
| `UI_HOST`, `UI_PORT` | Frontend host and port for local preview and cloud publish mapping. |
| `VITE_API_BASE_URL` | Frontend API base path. Keep `/api` for proxy compatibility. |
| `RELOAD` | Enables backend reload in local workflow. |
| `DILIGENT_CLOUD_MODE` | Enables cloud hardening on backend (`true` disables direct non-`/api` routes and OpenAPI docs endpoints). |
| `OPTIONAL_DEPENDENCIES` | Enables optional dependency branch in launcher flow. |
| `DB_EMBEDDED` | `true` for SQLite, `false` for external DB mode. |
| `DB_ENGINE`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | External DB connection values. |
| `DB_SSL`, `DB_SSL_CA` | External DB TLS settings. |
| `DB_CONNECT_TIMEOUT`, `DB_INSERT_BATCH_SIZE` | DB timeout and bulk-insert settings. |
| `MPLBACKEND`, `KERAS_BACKEND` | Runtime ML/scientific backend settings. |
| `ACCESS_KEY_ENCRYPTION_KEY` | Fernet key used to encrypt/decrypt stored provider API keys. |

## 4. Local Mode (Default)

1. Activate local profile:
   - `copy /Y DILIGENT\settings\.env.local.example DILIGENT\settings\.env`
2. Launch app:
   - `DILIGENT\start_on_windows.bat`
3. Run tests (optional):
   - `tests\run_tests.bat`

Local mode does not require Docker.

## 5. Cloud Mode (Docker)

1. Activate cloud profile:
   - `copy /Y DILIGENT\settings\.env.cloud.example DILIGENT\settings\.env`
2. Build images:
   - `docker compose --env-file DILIGENT/settings/.env build --no-cache`
3. Start containers:
   - `docker compose --env-file DILIGENT/settings/.env up -d`
4. Stop containers:
   - `docker compose --env-file DILIGENT/settings/.env down`

Cloud topology:
- `backend`: FastAPI/Uvicorn on internal `:8000`, host-mapped to loopback only (`127.0.0.1:${FASTAPI_PORT}`) to avoid direct public exposure.
- `frontend`: Nginx static hosting on internal `:80`, host-mapped from `${UI_PORT}`.
- Frontend `/api` is reverse-proxied to `http://backend:8000/`.
- Nginx proxies only the application API surface used by the frontend and blocks `/api/docs`, `/api/redoc`, and `/api/openapi.json` in cloud mode.
- Backend cloud mode (`DILIGENT_CLOUD_MODE=true`) serves only `/api/*` routes and disables direct docs/openapi endpoints.
- `diligent_resources` Docker volume persists runtime data under `/app/DILIGENT/resources`.

## 6. Packaged Desktop Mode (Tauri)

Desktop packaging is launched from the repository root helper:
- `release\tauri\build_with_tauri.bat`

Preparation sequence:
1. Activate the desktop profile:
   - `copy /Y DILIGENT\settings\.env.local.tauri.example DILIGENT\settings\.env`
2. Provision portable runtimes if needed:
   - `DILIGENT\start_on_windows.bat`
3. Ensure Rust is available for desktop packaging:
   - Install `rustup` from `https://rustup.rs/`
   - Configure a default toolchain (for example `stable-x86_64-pc-windows-msvc`)
4. Build:
   - `release\tauri\build_with_tauri.bat`

Desktop packaging model:
- Tauri is only the shell.
- The packaged app starts a local Python backend automatically.
- FastAPI serves the packaged SPA from `DILIGENT/client/dist`.
- API routes are available both at their original paths and under `/api`.
- User-facing Windows artifacts are exported to:
  - `release/windows/installers`
  - `release/windows/portable`

Bundled runtime payload:
- `pyproject.toml`
- `runtimes/uv.lock` (canonical backend lockfile)
- `uv.lock` (workspace lock staged from `runtimes/uv.lock` for `uv sync`)
- `DILIGENT/server`
- `DILIGENT/scripts`
- `DILIGENT/settings`
- `DILIGENT/client/dist`
- `DILIGENT/resources/models`
- `DILIGENT/resources/sources`
- `runtimes/python`
- `runtimes/uv`
- `runtimes/nodejs`

Desktop startup behavior:
- Tauri starts at `about:blank` and renders a Rust-driven startup screen immediately.
- Rust resolves the packaged workspace, prefers a reusable `runtimes/.venv`, and otherwise runs `uv sync --frozen`.
- If the installed bundle directory is not writable, the launcher mirrors the packaged workspace into a writable per-user runtime root before starting the backend.
- Once the backend is reachable on loopback, the window redirects to `http://127.0.0.1:<FASTAPI_PORT>/`.
- On exit, the desktop app terminates the backend process tree.

Desktop icon model:
- Canonical icon source: `DILIGENT/client/public/favicon.png`
- Regeneration command:
  - `cd DILIGENT\client`
  - `npm run tauri:icon`
- Generated mobile icon folders are removed so the repository stays desktop-only.

Cleanup:
- `cd DILIGENT\client && npm run tauri:clean`
- or `DILIGENT/setup_and_maintenance.bat` -> `Clean desktop build artifacts`

## 7. Deterministic Build Notes

- Backend dependencies are lockfile-backed by `runtimes/uv.lock` (`uv sync --frozen` in Docker).
- Frontend dependencies are lockfile-backed by `DILIGENT/client/package-lock.json` (`npm ci` in Docker).
- Base images are pinned:
  - Backend: `ghcr.io/astral-sh/uv:0.8.22-python3.14-bookworm`
  - Frontend build: `node:22.12.0-alpine`
  - Frontend runtime: `nginx:1.27.4-alpine`
