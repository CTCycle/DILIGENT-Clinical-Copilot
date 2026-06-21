# Configuration
Last updated: 2026-06-21

## Primary Runtime Files
- Active env file: `settings/.env`
- Env template: `settings/.env.local.example`
- Structured operational settings: `settings/configurations.json`

## Default Local Ports
- Backend: `127.0.0.1:7690`
- Frontend: `127.0.0.1:9847`

## Main Keys
- `FASTAPI_HOST=127.0.0.1`
- `FASTAPI_PORT=7690`
- `UI_HOST=127.0.0.1`
- `UI_PORT=9847`
- `VITE_API_BASE_URL=/api`
- `RELOAD=false`
- `OPTIONAL_DEPENDENCIES=true`

## Operational Settings By Source
- `settings/.env`
  - database mode and connection settings
  - local host and port values
- `settings/configurations.json`
  - job polling interval
  - deployment mode, currently `local_single_user`
  - RAG and ingestion settings
  - external timeout and concurrency settings
  - excludes all database mode and connection settings
- `app/resources/catalogs/*.json`
  - canonical deterministic reference catalogs for text normalization, extraction, matching, DILI behavior, language, and security filters

## Runtime Differences
- Startup supports only `deployment.mode=local_single_user`. Any other deployment mode fails startup validation until multi-user and server deployment controls are implemented.
- Local mode serves the frontend from a preview or dev process.
- Packaged Tauri mode serves bundled SPA assets from the backend.
- Startup validates packaged SPA presence before desktop mode serves assets.
- Runtime resources are bundled under `runtime/` through `tauri.conf.json`.

## Feature Toggles
- Cloud-versus-local model usage is runtime-configured through model configuration APIs.
- Database mode, embedded SQLite versus PostgreSQL, is controlled by `settings/.env`.
- Catalog seeding is hash-checked and incremental on normal startup.
