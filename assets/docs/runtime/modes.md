# Runtime Modes
Last updated: 2026-10-21

## Supported Modes
### Local Application
- Backend: FastAPI in `app/server/app.py`
- Production frontend: built Angular SPA served by FastAPI from `app/client/dist/browser`; `/api/*` and the SPA share one origin
- Development frontend: Angular dev server from `npm run dev`, with `/api` proxied to FastAPI
- Windows launcher and maintenance menu: `start_on_windows.ps1`
- Source production uses `FASTAPI_HOST` and `FASTAPI_PORT` (template defaults `127.0.0.1:7690`). The browser opens the backend origin.
- Development uses `UI_HOST` and `UI_PORT` (template defaults `127.0.0.1:9847`) for the Angular dev server. `UI_PORT` does not start a production process.

### Packaged Windows Desktop
- Tauri shell: `app/desktop/src-tauri`.
- Backend: embedded PyInstaller onedir runtime, launched on `127.0.0.1` with an OS-selected port.
- Frontend: Angular production files and startup splash embedded as Tauri assets. Tauri's asset resolver serves client-side routes from the bundled `index.html`.
- Backend: API-only PyInstaller sidecar on a random loopback port, owned and stopped by Tauri.
- API security: the shell passes the API origin and a one-time session token to the bundled SPA. API calls carry `X-DILIGENT-Desktop-Session`; CORS allows only `http://tauri.localhost`.
- Runtime dependencies: no Python, Node.js, npm, Rust, uv, or source checkout is required on the target machine.
- Distribution forms: single-file `*-portable.exe` for no-install use and `*.msi` for installed use, with a matching `.sha256` manifest.
- WebView2: the portable shell uses the system runtime; MSI builds can embed an offline installer with `-OfflineWebView2`.
- Persistent data: `%LOCALAPPDATA%\DILIGENT\data`.
- Extracted immutable runtime: `%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>`.
- Development ports `7690` and `9847` do not apply to packaged desktop startup.

### Containerized Runtime
- Not supported in the current codebase.
- No `Dockerfile` or `docker-compose` configuration exists in the repository.

## Interoperability
- Frontend and backend communicate through `/api/*`.
- Source production serves the frontend and API from the same origin. Development proxies `/api` from the Angular dev server to FastAPI.
- Tauri serves the frontend from its packaged origin and connects to the API on loopback using the per-launch session header.
- Shared persistence and services across modes:
  - SQL database
  - LanceDB vectors in data paths
  - data catalogs and source documents

## Limitations
- DILIGENT is designed for single-user localhost operation. Authenticated network deployment is not supported.
- No official container workflow.
- Long-running operations rely on backend job polling.
- Some features require reachable external dependencies such as Ollama.
- RAG folder selection uses the native operating-system picker in packaged
  Tauri and the localhost backend's server-side directory browser in the
  browser development surface.
- Parser-model concurrency is guarded by parser batch preflight and falls back to sequential execution when runtime or model status cannot be validated safely.
