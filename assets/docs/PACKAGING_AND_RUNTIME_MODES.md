# Packaging and Runtime Modes

Last updated: 2026-04-09

## 1. Runtime strategy

DILIGENT is configuration-first and uses one active runtime file:
- `DILIGENT/settings/.env`

Modes:
- Local mode (default developer workflow)
- Desktop packaged mode (Tauri shell + local backend runtime)

Switch modes by copying a profile into `DILIGENT/settings/.env`.

## 2. Runtime profile files

- `DILIGENT/settings/.env.local.example`
- `DILIGENT/settings/.env.local.tauri.example`
- `DILIGENT/settings/.env` (active)
- `DILIGENT/settings/configurations.json` (non-secret defaults/tuning, including database settings)

## 3. Environment contract (core keys)

| Key | Purpose |
|---|---|
| `FASTAPI_HOST`, `FASTAPI_PORT` | Backend bind host/port. |
| `UI_HOST`, `UI_PORT` | Frontend host/port. |
| `KERAS_BACKEND` | Runtime Keras backend override (optional). |
| `MPLBACKEND` | Runtime Matplotlib backend override (optional). |
| `API_BASE_URL` | Frontend API base path (fixed to `/api` in Angular client constants). |
| `RELOAD` | Backend auto-reload toggle for local workflow. |
| `OLLAMA_URL`, `OLLAMA_HOST`, `OLLAMA_PORT` | Ollama endpoint configuration. |
| `OPTIONAL_DEPENDENCIES` | Optional dependency branch for launcher flow. |

Database keys are JSON-only under `configurations.json`:
- `database.embedded_database`
- `database.engine`, `database.host`, `database.port`
- `database.database_name`, `database.username`, `database.password`
- `database.ssl`, `database.ssl_ca`
- `database.connect_timeout`
- `database.insert_batch_size`, `database.insert_commit_interval`, `database.select_page_size`

Provider key storage contract:
- Provider keys are entered in-app and stored encrypted in the database.
- Encryption registry is seeded in `access_key_encryption_materials`:
  - SQLite: only on first local DB file creation.
  - PostgreSQL: only during explicit DB initialization.
- Provider keys are not provided through environment variables in the active model.

Database compatibility rule:
- Existing databases are not upgraded in place across schema-cleanup releases.
- Recreate the schema (or remove/recreate local SQLite DB files) when upgrading across this cleanup.
- Startup no longer probes legacy SQLite layouts for automatic salvage/deletion.

Ollama compatibility rule:
- Supported Ollama installations must expose the chat-capable `/api/chat` contract.
- The backend no longer retries legacy `/api/generate` compatibility paths.

## 4. Local mode

1. Activate profile:
   - `copy /Y DILIGENT\settings\.env.local.example DILIGENT\settings\.env`
2. Start application:
   - `DILIGENT\start_on_windows.bat`
3. Optional full test run:
   - `tests\run_tests.bat`

## 5. Desktop packaged mode (Tauri)

Build entrypoint:
- `release\tauri\build_with_tauri.bat`

Typical flow:
1. Activate desktop profile:
   - `copy /Y DILIGENT\settings\.env.local.tauri.example DILIGENT\settings\.env`
2. Ensure runtimes exist:
   - `DILIGENT\start_on_windows.bat`
3. Ensure Rust toolchain is available (`rustup`).
4. Build package:
   - `release\tauri\build_with_tauri.bat`

Packaged runtime behavior:
- Tauri starts a local Python backend.
- Backend serves packaged frontend from `DILIGENT/client/dist`.
- Runtime setup prefers reusable `runtimes/.venv`, otherwise runs `uv sync --frozen`.
- On shutdown, desktop wrapper terminates backend process tree.

Windows artifacts:
- `release/windows/installers`
- `release/windows/portable`

## 6. Deterministic build notes

- Backend lockfile: `runtimes/uv.lock`.
- Frontend lockfile: `DILIGENT/client/package-lock.json`.

## 7. Maintenance boundary

- `DILIGENT/setup_and_maintenance.bat` is for offline maintenance only:
  - manual database initialization
  - log cleanup
  - desktop build cleanup
  - uninstall cleanup
- Data update operations (RxNav, LiverTox, RAG embeddings) are managed from the Data Inspection UI wizards.
