# Configuration
Last updated: 2026-07-23

Temperature is not a deployment or operator setting; it is resolved by the
source-controlled automatic generation policy immediately before each LLM call.

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
- `ALWAYS_REBUILD=true` (rebuilds the frontend at application startup when `true`; skips that build when `false`)
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

## Feature Toggles
- Cloud-versus-local model usage is runtime-configured through model configuration APIs.
- Database mode, embedded SQLite versus PostgreSQL, is controlled by `settings/.env`.
- Catalog seeding is hash-checked and incremental on normal startup.

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
