# Configuration
Last updated: 2026-07-16

## Primary Runtime Files
- Active env file: `settings/.env`
- Env template: `settings/.env.example`
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
- `always_rebuild=true` (rebuilds the frontend at application startup when `true`; skips that build when `false`)
- `BACKEND_LOGS_VISIBLE=true` (defaults to `true` when absent)
- `OPTIONAL_DEPENDENCIES=true`

## Operational Settings By Source
- `settings/.env`
  - database mode and connection settings
  - local host and port values
- `settings/configurations.json`
  - job polling interval
  - RAG and ingestion settings
  - external timeout and concurrency settings
  - excludes all database mode and connection settings
- `app/resources/catalogs/*.json`
  - canonical deterministic reference catalogs for text normalization, extraction, matching, DILI behavior, language, and security filters

## Local Runtime
- Local runs serve the frontend from a preview or dev process.
- `BACKEND_LOGS_VISIBLE=true` opens a dedicated backend terminal; `false` keeps it hidden. The launcher defaults to visible logs when the key is absent.
- The frontend preview runs without a visible terminal window.

## Feature Toggles
- Cloud-versus-local model usage is runtime-configured through model configuration APIs.
- Database mode, embedded SQLite versus PostgreSQL, is controlled by `settings/.env`.
- Catalog seeding is hash-checked and incremental on normal startup.

## RAG Defaults
- The default Ollama embedding model is pinned in `settings/configurations.json` and should not use a mutable `:latest` tag.
- `reset_vector_collection` defaults to `false` and should only be enabled for explicit maintenance or rebuild operations.
## Database configuration

Use one canonical database contract:

```text
DATABASE_BACKEND=sqlite
DATABASE_URL=
DATABASE_SQLITE_PATH=
DATABASE_CONNECT_TIMEOUT=10
DATABASE_WRITE_BATCH_SIZE=1000
DATABASE_READ_PAGE_SIZE=1000
```

Set `DATABASE_BACKEND=postgresql` and provide `DATABASE_URL` for PostgreSQL.
The old embedded/engine/host/port/password split is no longer documented.

Provider access-key encryption material is external to the database. Set
`DILIGENT_ACCESS_KEY_MATERIAL_FILE` to a protected local file (or inject the
equivalent path through the deployment environment). The file contains
versioned Fernet material used to decrypt the `access_keys` ciphertext; it must
not be committed, copied into a database backup, or exposed through logs.
