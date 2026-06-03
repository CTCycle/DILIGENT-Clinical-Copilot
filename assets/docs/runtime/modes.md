# Runtime Modes
Last updated: 2026-06-03

## Supported Modes
### Local Development
- Backend: FastAPI in `app/server/app.py`
- Frontend: Angular app served from `app/client`
- Active local ports from `settings/.env`:
  - API: `127.0.0.1:7690`
  - UI: `127.0.0.1:9847`
- Template ports from `settings/.env.local.example` use the same defaults.

### Desktop Runtime
- Tauri wrapper in `app/client/src-tauri`
- Bundles:
  - backend code
  - built frontend dist
  - portable Python and `uv` runtimes
- Output artifacts are exported under `release/windows`.

### Containerized Runtime
- Not supported in the current codebase.
- No `Dockerfile` or `docker-compose` configuration exists in the repository.

## Interoperability
- Frontend and backend communicate through `/api/*`.
- The same API contract is used in browser mode and packaged desktop mode.
- Shared persistence and services across modes:
  - SQL database
  - LanceDB vectors in resource paths
  - resource catalogs and source documents

## Limitations
- No official container workflow.
- Desktop build is Windows-focused in current release scripts and output layout.
- Long-running operations rely on backend job polling.
- Some features require reachable external dependencies such as Ollama.
- Parser-model concurrency is guarded by parser batch preflight and falls back to sequential execution when runtime or model status cannot be validated safely.
