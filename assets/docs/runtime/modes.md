# Runtime Modes
Last updated: 2026-07-29

## Supported Modes
### Local Application
- Backend: FastAPI in `app/server/app.py`
- Frontend: Angular app served from `app/client`
- Windows launcher and maintenance menu: `start_on_windows.ps1`
- Active local ports from `settings/.env`:
  - API: `127.0.0.1:7690`
  - UI: `127.0.0.1:9847`
- Template ports from `settings/.env.example` use the same defaults.

### Packaged Windows Desktop
- Tauri shell: `app/desktop/src-tauri`.
- Backend: embedded PyInstaller onedir runtime, launched on `127.0.0.1` with an OS-selected port.
- Frontend: Angular production files served by the packaged FastAPI backend.
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
- The frontend and backend use the same local API contract.
- Shared persistence and services across modes:
  - SQL database
  - LanceDB vectors in resource paths
  - resource catalogs and source documents

## Limitations
- DILIGENT is designed for single-user localhost operation. Authenticated network deployment is not supported.
- No official container workflow.
- Long-running operations rely on backend job polling.
- Some features require reachable external dependencies such as Ollama.
- Parser-model concurrency is guarded by parser batch preflight and falls back to sequential execution when runtime or model status cannot be validated safely.
