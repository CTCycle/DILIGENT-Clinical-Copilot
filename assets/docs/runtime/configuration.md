# Configuration
Last updated: 2026-08-21

Temperature is not a deployment or operator setting; it is resolved by the
source-controlled effective inference policy immediately before each LLM call.
The model configuration API persists `reasoning_level` as `off`, `low`,
`medium`, or `high`. Existing singleton configuration payloads containing the
legacy `ollama_reasoning` boolean remain readable (`false` -> `off`, `true` ->
`medium`) and are normalized on the next write; no schema migration is needed.

See [generation policy](generation_policy.md) for the purpose responsibility
matrix, capability catalog, provider transport rules, and context-budget
provenance.

## Primary Runtime Files
- Active env file: `settings/.env`
- Env template: `settings/.env.example`
- Structured operational settings: `settings/configurations.json`

## Desktop-only environment variables

- `DILIGENT_DESKTOP=true`
- `DILIGENT_RELEASE_VERSION=major.minor.patch`
- `DILIGENT_RUNTIME_ROOT=<absolute extracted immutable runtime root>`
- `DILIGENT_DATA_ROOT=<absolute persistent user-data root>`
- `DILIGENT_SQLITE_PATH=<absolute data database path>`
- `DILIGENT_ACCESS_KEY_MATERIAL_FILE=<absolute protected key-material path>`
- `RELOAD=false`

Packaged mode requires `DILIGENT_RUNTIME_ROOT` and `DILIGENT_DATA_ROOT` together. Relative or partial desktop roots are rejected; source-mode paths are never used as a fallback.

The Tauri shell sets these desktop-only variables when it starts the frozen backend. Operators should not add them to `settings/.env` or override them manually. Packaged desktop chooses a free localhost port and records it in `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`; the packaged log is `%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log`.

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
- `BACKEND_LOGS_VISIBLE=true` (defaults to `true` when absent)

## Source-mode Resource Location

- `DILIGENT_RESOURCES_PATH=app/resources`

This optional path relocates the source-mode resource tree. Relative paths are
resolved from the repository root. The embedded SQLite database defaults to
`<resource-root>/database.db`; an explicit `DILIGENT_SQLITE_PATH` still takes
precedence. Packaged desktop mode uses `DILIGENT_RUNTIME_ROOT` and
`DILIGENT_DATA_ROOT` instead.

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
  - canonical deterministic reference catalogs for text normalization, extraction, matching, DILI behavior, language, security filters, generation policy, and model capabilities

## LLM Time Budgets

`settings/configurations.json` sets the clinical cloud-model ceiling through
`runtime.cloud_llm_timeout_cap`. It is `1800` seconds (30 minutes) so a
full clinical synthesis is not prematurely replaced with a fallback report.
This limit does not make an unreachable provider available: connectivity,
authentication, and provider-side errors still fail promptly.

## Local Runtime
- Local runs serve the frontend from a preview or dev process.
- `BACKEND_LOGS_VISIBLE=true` opens a dedicated backend terminal; `false` keeps it hidden. The launcher defaults to visible logs when the key is absent.
- The frontend preview runs without a visible terminal window.

## Packaged Desktop Runtime
- The packaged Tauri shell does not read the development UI/API port settings for its backend listener.
- The immutable runtime is addressed by version and payload digest under `%LOCALAPPDATA%\DILIGENT\runtime`.
- Mutable settings, database, logs, models, vectors, exports, state, and access-key material live under `%LOCALAPPDATA%\DILIGENT\data`.

## Feature Toggles
- Cloud-versus-local model usage is runtime-configured through model configuration APIs.
- Database mode, embedded SQLite versus PostgreSQL, is controlled by `settings/.env`.
- Alembic migrations run during installation and every application startup.
  Catalog seeding occurs during explicit initialization, or on the first startup
  that creates a database. Existing databases are migrated but not reseeded.

## RAG Defaults
- RAG uses `ibm-granite/granite-embedding-97m-multilingual-r2` at the pinned revision and AVX2 quantized ONNX artifact declared in `app/server/common/embedding/config.py`.
- `reset_vector_collection` defaults to `false` and should only be enabled for explicit maintenance or rebuild operations.
## Database configuration

Use one canonical database contract:

```text
###############################################################################
# Database Mode
###############################################################################
EMBEDDED_DATABASE=true

###############################################################################
# External Database Connection
###############################################################################
DATABASE_URL=
DATABASE_ENGINE=postgresql+psycopg
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=DILIGENT
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=

###############################################################################
# External Database Security And Performance
###############################################################################
DATABASE_SSL=false
DATABASE_SSL_CA=
DATABASE_CONNECT_TIMEOUT=30
DATABASE_INSERT_BATCH_SIZE=1000
```

Set `EMBEDDED_DATABASE=true` for SQLite or `false` for PostgreSQL.

Provider access-key encryption material is external to the database. Set
`DILIGENT_ACCESS_KEY_MATERIAL_FILE` to a protected local file (or inject the
equivalent path through the deployment environment). The file contains
versioned Fernet material used to decrypt the `access_keys` ciphertext; it must
not be committed, copied into a database backup, or exposed through logs.
## RAG embedding runtime

The multilingual Granite ONNX snapshot is downloaded lazily into `app/resources/models/embeddings/<revision>/` and loaded once per process with `onnxruntime` and `CPUExecutionProvider`. There is no backend or artifact fallback, and no PyTorch requirement. Offline mode requires a complete verified snapshot; changing the model contract requires a full vector-store rebuild. Readiness is available only after the pinned artifact digest validates.
