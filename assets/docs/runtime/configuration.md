# Configuration
Last updated: 2026-10-21

Temperature is not a deployment or operator setting; it is resolved by the
source-controlled effective inference policy immediately before each LLM call.
The model configuration API persists `reasoning_level` as `off`, `low`,
`medium`, or `high`. The obsolete boolean reasoning field is not accepted or
translated; callers must send `reasoning_level` explicitly.

See [generation policy](generation_policy.md) for the purpose responsibility
matrix, capability catalog, provider transport rules, and context-budget
provenance.

Cloud model catalogs are provider-specific. OpenCode Zen and Go resolve the
transport endpoint from the provider model metadata, with a documented fallback
only for known Go model IDs. A configured model is never silently replaced when
its provider rejects a request; the persisted fallback warning includes the
operation and sanitized upstream detail when available.

## Primary Runtime Files
- Process and deployment settings: `settings/.env` and `settings/.env.example`
- Database connection settings only: `<data-root>/database_config.json`
- Database-backed operational settings: the singleton `application_configuration`
  row, edited through the Settings UI and typed `/api/settings` endpoints

## Desktop-only environment variables

- `DILIGENT_DESKTOP=true`
- `DILIGENT_RELEASE_VERSION=major.minor.patch`
- `DILIGENT_RUNTIME_ROOT=<absolute extracted immutable runtime root>`
- `DILIGENT_DATA_ROOT=<absolute persistent user-data root>`
- `DILIGENT_ACCESS_KEY_MATERIAL_FILE=<absolute protected key-material path>`
- `RELOAD=false`

Packaged mode requires `DILIGENT_RUNTIME_ROOT` and `DILIGENT_DATA_ROOT` together. Relative or partial desktop roots are rejected; source-mode paths are never used as a fallback.

The Tauri shell sets these desktop-only variables when it starts the frozen backend. Operators should not add them to `settings/.env` or override them manually. Packaged desktop chooses a free localhost port and records it in `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`; the packaged log is `%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log`.

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

## Source-mode Data Location

- `DILIGENT_DATA_PATH=data`

This optional path relocates the source-mode data tree. Relative paths are
resolved from the repository root. The database configuration file is stored
at `<data-root>/database_config.json`; SQLite defaults to
`<data-root>/database.db` unless a different file path is selected in Settings.
Packaged desktop mode uses `DILIGENT_RUNTIME_ROOT` and `DILIGENT_DATA_ROOT`
instead.

## Operational Settings By Source
- `settings/.env`
  - local host and port values
  - data paths, Ollama endpoints, deployment values, and secrets
- `<data-root>/database_config.json`
  - selected database backend and its connection parameters only
  - PostgreSQL password protected with Windows DPAPI; no application settings
- `application_configuration` database singleton
  - job polling interval
  - model/provider selections and access-key references
  - complete RAG settings
  - ingestion, session pipeline, language detection, matcher, integration,
    timeout, and evidence settings
- `data/catalogs/*.json`
  - canonical immutable reference catalogs for text normalization, extraction,
    matching, DILI behavior, language, security filters, generation policy, and
    model capabilities

## NCBI machine access and LiverTox refresh

The LiverTox updater uses NCBI's documented machine-access services. It does
not retrieve the interactive Bookshelf HTML pages for automated discovery and
does not solve or bypass human-verification challenges.

- E-utilities identify the master-list record in `db=books` using the fixed
  tool name `DILIGENTClinicalCopilot` and the effective
  `ncbi_contact_email` value. The calls use
  `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi` and
  `esummary.fcgi`.
- Books-OAI uses the stable identifier
  `oai:books.ncbi.nlm.nih.gov:571102` and the `nbk_ftext` metadata format to
  discover a validated `masterlist*.xlsx` link when one is exposed, through
  `https://api.ncbi.nlm.nih.gov/lit/oai/books/`.
- LitArch `file_list.csv` resolves the current
  `livertox_NBK547852.tar.gz` archive path. The updater never assumes the
  randomly assigned two-level directory or an annual master-list filename;
  the index is `https://ftp.ncbi.nlm.nih.gov/pub/litarch/file_list.csv`.
- If Books-OAI does not expose a usable spreadsheet, the updater extracts one
  unambiguous safe `masterlist*.xlsx` member from that official archive.

`ncbi_contact_email` is blank by default and can be cleared in Settings.
LiverTox and combined structured-source updates require a real developer or
organization email registered with NCBI. Blank and `.local` values are rejected
before a job starts and checked again before NCBI requests. A previously saved
`.local` value remains unchanged until the user replaces or clears it. The
value is sent only as request identity, is not a credential, and is excluded
from update results, source metadata, clinical artifacts, and ordinary logs.
Requests remain sequential and no NCBI API key is required for this low-volume
workflow.

## LLM Time Budgets

The Settings UI sets the clinical cloud-model ceiling through the persisted
`runtime.cloud_llm_timeout_cap` value. Its typed default is `1800` seconds (30
minutes) so a full clinical synthesis is not prematurely replaced with a
fallback report.
This limit does not make an unreachable provider available: connectivity,
authentication, and provider-side errors still fail promptly.

## Local Runtime
- Local runs serve the frontend from a preview or dev process.
- The Windows launcher always opens a dedicated backend terminal so backend output remains visible while the application is running.
- The frontend preview runs without a visible terminal window.

## Packaged Desktop Runtime
- The packaged Tauri shell does not read the development UI/API port settings for its backend listener.
- The immutable runtime is addressed by version and payload digest under `%LOCALAPPDATA%\DILIGENT\runtime`.
- Mutable settings, database, vectors, exports, state, and access-key material live under `%LOCALAPPDATA%\DILIGENT\data`; disposable logs and embedding caches live under `%LOCALAPPDATA%\DILIGENT\data\cache`.

## Feature Toggles
- Cloud-versus-local model usage is runtime-configured through model configuration APIs.
- Database mode, SQLite versus PostgreSQL, is selected in Settings and stored in
  `database_config.json`.
- Alembic migrations run during installation and every application startup.
  Catalog seeding occurs during explicit initialization, or when startup finds
  a new application schema or an empty reference catalog.

## RAG Defaults
- RAG uses `ibm-granite/granite-embedding-97m-multilingual-r2` at the pinned revision and AVX2 quantized ONNX artifact declared in `app/server/common/embedding/config.py`.
- `reset_vector_collection` defaults to `false` and should only be enabled for explicit maintenance or rebuild operations.
- Supported inputs are fingerprinted with SHA-256 over raw bytes. Only byte-identical files deduplicate; semantic or normalized-text deduplication is not performed.
- The canonical duplicate source is selected deterministically by normalized relative-path ordering and retains the existing path-derived `document_id`. Duplicate paths remain visible in Data Inspection and are retained as canonical/alias metadata for citations and audit.
- RAG manifests report physical supported-file count, unique ingested-document count, and duplicate-file count separately. Embedding and chunking occur once per canonical source.
## Database configuration

The Database section in Settings is the only supported editor for database
connections. The versioned JSON file contains the selected backend, SQLite
path or PostgreSQL host/port/database/user/SSL settings, timeouts, and batch
sizes. It contains no unrelated application settings and is independent of the
database it configures. Writes use a validated temporary file followed by an
atomic replacement. A PostgreSQL password is protected with Windows DPAPI for
the current user; API responses never return it.

SQLite and PostgreSQL through `postgresql+psycopg` are the supported backends.
There are no active database environment variables. `.env` and process
environment variables never configure database access; when
`database_config.json` does not exist, startup creates it with the default
embedded SQLite configuration.

Database connection settings are separate from all other settings. The active
database's `application_configuration` row is the sole persistent source for
operational settings, model configuration, and other UI-managed application
settings. Switching databases does not copy or merge rows.

Database backend and connection changes can be applied live through Settings.
New API requests pause while existing API work drains; switching is rejected
while background jobs are pending or running. The target is initialized and
verified before its config is saved, and failed activation restores the prior
connection. Restart the application when changing the data root, packaged
runtime, access-key-material location, or any startup-only process/deployment
setting. If automatic service rebinding cannot recover after a failed switch,
restart to reload the saved working configuration.

Provider access-key encryption material is external to the database. Set
`DILIGENT_ACCESS_KEY_MATERIAL_FILE` to a protected local file (or inject the
equivalent path through the deployment environment). The file contains
versioned Fernet material used to decrypt the `access_keys` ciphertext; it must
not be committed, copied into a database backup, or exposed through logs.
## RAG embedding runtime

The multilingual Granite ONNX snapshot is downloaded lazily into the disposable
source cache at `runtimes/cache/embeddings/<revision>/`, or into
`%LOCALAPPDATA%\DILIGENT\data\cache\embeddings\<revision>\` for packaged desktop
execution. Hugging Face metadata and download bookkeeping are also kept under
the same cache root. It is loaded once per process with `onnxruntime` and
`CPUExecutionProvider`. There is no backend or artifact fallback, and no
PyTorch requirement. Offline mode requires a complete verified snapshot;
changing the model contract requires a full vector-store rebuild. Readiness is
available only after the pinned artifact digest validates.
