# Runtime Modes

Last updated: 2026-04-24

## 1. Supported Modes

### Local development (web + API)
- Backend: FastAPI (`DILIGENT/server/app.py`)
- Frontend: Angular app served from `DILIGENT/client`
- Default ports from `.env.local.example`:
  - API: `127.0.0.1:8000`
  - UI: `127.0.0.1:7861`

### Desktop runtime (Tauri)
- Tauri wrapper (`DILIGENT/client/src-tauri`) bundles:
  - backend code
  - built frontend dist
  - portable Python/uv/Node runtimes
- Output artifacts are exported under `release/windows`.

### Containerized runtime
- Not supported in current codebase.
- No `Dockerfile` or `docker-compose` configuration exists in repository.

## 2. Startup Procedures

### Local development (recommended on Windows)

```cmd
DILIGENT\start_on_windows.bat
```

What it does:
- Ensures portable Python, uv, and Node runtimes under `runtimes/`
- Runs `uv sync`
- Installs frontend dependencies if missing
- Builds frontend dist if missing
- Starts backend via `uv run ... uvicorn`
- Starts frontend preview server

### Manual backend startup (PowerShell)

```powershell
runtimes\uv\uv.exe run --python runtimes\python\python.exe python -m uvicorn DILIGENT.server.app:app --host 127.0.0.1 --port 8000
```

### Manual frontend startup (PowerShell)

```powershell
Set-Location DILIGENT\client
npm run preview -- --host 127.0.0.1 --port 7861 --strictPort
```

### Desktop build (Tauri)

```cmd
release\tauri\build_with_tauri.bat
```

Build prerequisites:
- Portable runtimes present (`start_on_windows.bat` executed at least once)
- Rust/Cargo toolchain available and default toolchain configured
- Frontend dependencies installed

## 3. Environment Variables and Config Requirements

Primary runtime env file:
- `DILIGENT/settings/.env` (active)

Template:
- `DILIGENT/settings/.env.local.example`

Main keys:
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `UI_HOST`, `UI_PORT`
- `VITE_API_BASE_URL` (expected `/api`)
- `RELOAD`
- `OLLAMA_URL`, `OLLAMA_HOST`, `OLLAMA_PORT`
- `OPTIONAL_DEPENDENCIES`

Non-secret operational settings:
- `DILIGENT/settings/configurations.json`
  - database mode/settings
  - jobs polling interval
  - RAG and ingestion settings
  - external timeout/concurrency settings

## 4. Configuration Differences (dev vs packaged)

- Local mode:
  - Frontend served by preview/dev process.
  - Root URL redirects to docs when SPA package path is not active.
- Tauri packaged mode:
  - Backend serves packaged SPA files from bundled dist assets.
  - Runtime resources are bundled under `r/` via `tauri.conf.json`.

Feature toggles/settings:
- Cloud-vs-local model usage is runtime-configured through model configuration APIs.
- DB mode (embedded SQLite vs PostgreSQL) is controlled by `configurations.json`.

## 5. Interoperability

- Frontend and backend communicate through `/api/*`.
- Same API contract is used in browser mode and packaged desktop mode.
- Shared persistence/services across modes:
  - SQL DB (SQLite or PostgreSQL)
  - LanceDB vectors in resources path
  - resources catalogs and source documents

## 6. Limitations and Constraints

- No official Docker/container workflow in current implementation.
- Desktop build is Windows-focused in current release scripts/output layout.
- Long-running operations rely on backend job polling lifecycle; frontend must poll for completion.
- Some features require reachable external dependencies (for example Ollama, Brave Search when enabled).

## 7. Deployment and Packaging Notes

- Desktop packaging command:
  - `npm run tauri:build:release` (invoked by `release\tauri\build_with_tauri.bat`)
- Exported Windows artifacts:
  - `release/windows/installers`
  - `release/windows/portable`
- Lockfiles used for deterministic dependency state:
  - `runtimes/uv.lock`
  - `DILIGENT/client/package-lock.json`
