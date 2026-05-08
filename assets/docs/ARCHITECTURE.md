# DILIGENT Clinical Copilot Architecture

Last updated: 2026-04-30

## 1. System Summary

DILIGENT is a local-first clinical application with:
- Backend: FastAPI (`app/server`)
- Frontend: Angular standalone + TypeScript (`app/client`)
- Optional desktop shell: Tauri (`app/client/src-tauri`)

Primary flow:
1. User submits clinical data in the Angular UI.
2. Backend validates/normalizes input and runs clinical analysis.
3. Long operations run as background jobs with poll/cancel APIs.
4. Results and catalog/session data are persisted for inspection.

## 2. Source Structure (maintained code)

```text
.
|-- start_on_windows.bat
|-- setup_and_maintenance.bat
|-- settings/
|   |-- .env
|   |-- .env.local.example
|   `-- configurations.json
|-- app/
|   |-- resources/
|   |   |-- models/
|   |   `-- sources/
|   |-- server/
|   |   |-- app.py
|   |   |-- api/
|   |   |-- configurations/
|   |   |-- domain/
|   |   |-- repositories/
|   |   |-- services/
|   |   |-- common/
|   |   `-- models/
|   |-- client/
|   |   |-- package.json
|   |   |-- src/
|   |   |   |-- main.ts
|   |   |   |-- styles.scss
|   |   |   `-- app/
|   |   |       |-- app.ts
|   |   |       |-- app.routes.ts
|   |   |       |-- core/
|   |   |       |-- components/
|   |   |       `-- pages/
|   |   `-- src-tauri/
|   |       |-- tauri.conf.json
|   |       `-- src/main.rs
|   `-- tests/
|       |-- run_tests.bat
|       |-- conftest.py
|       |-- unit/
|       `-- e2e/
`-- release/
    `-- tauri/
        `-- build_with_tauri.bat
```

Notes:
- Build/cache artifacts (`node_modules`, `.angular`, `dist`, `target`, `__pycache__`) are intentionally excluded above.

## 3. Application Entry Points

- Backend app: `app/server/app.py`
  - Builds the FastAPI app through `create_app()`, initializes settings, registers middleware/error handlers, mounts routers under `/api`, and initializes DB/runtime model config through the FastAPI lifespan startup path.
- Frontend app: `app/client/src/main.ts`
  - Bootstraps Angular `App` with `appConfig`.
- Frontend routing: `app/client/src/app/app.routes.ts`
  - Routes: `/`, `/data`, `/model-config`.
- Desktop runtime: `app/client/src-tauri/src/main.rs` + `tauri.conf.json`.
- Windows launcher: `start_on_windows.bat`.

## 4. API Endpoints

All business APIs are mounted with prefix `/api`.

Root/OpenAPI routes:
- `GET /`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`
- In packaged Tauri mode, `/` and `/{full_path:path}` serve SPA assets.

Session and clinical:
- `GET /api/health`
- `POST /api/clinical`
- `POST /api/clinical/jobs`
- `GET /api/clinical/jobs/{job_id}`
- `DELETE /api/clinical/jobs/{job_id}`

Model catalog and pull:
- `GET /api/models/list`
- `GET /api/models/pull`
- `POST /api/models/pull/jobs`
- `GET /api/models/jobs/{job_id}`
- `DELETE /api/models/jobs/{job_id}`

Model configuration:
- `GET /api/model-config`
- `PUT /api/model-config`

Access keys:
- `GET /api/access-keys`
- `POST /api/access-keys`
- `PUT /api/access-keys/{key_id}/activate`
- `DELETE /api/access-keys/{key_id}`

Research:
- `POST /api/research`

Inspection:
- `GET /api/inspection/sessions`
- `GET /api/inspection/sessions/{session_id}/report`
- `GET /api/inspection/sessions/{session_id}/timeline`
- `POST /api/inspection/sessions/{session_id}/timeline`
- `DELETE /api/inspection/sessions/{session_id}`
- `GET /api/inspection/rxnav`
- `GET /api/inspection/rxnav/{drug_id}/aliases`
- `DELETE /api/inspection/rxnav/{drug_id}`
- `GET /api/inspection/rxnav/update-config`
- `POST /api/inspection/rxnav/jobs`
- `GET /api/inspection/rxnav/jobs/{job_id}`
- `DELETE /api/inspection/rxnav/jobs/{job_id}`
- `GET /api/inspection/livertox`
- `GET /api/inspection/livertox/{drug_id}/excerpt`
- `DELETE /api/inspection/livertox/{drug_id}`
- `GET /api/inspection/livertox/update-config`
- `POST /api/inspection/livertox/jobs`
- `GET /api/inspection/livertox/jobs/{job_id}`
- `DELETE /api/inspection/livertox/jobs/{job_id}`
- `GET /api/inspection/text-normalization`
- `GET /api/inspection/text-normalization/{category}`
- `PUT /api/inspection/text-normalization/{category}`
- `DELETE /api/inspection/text-normalization/{category}/{term}`
- `GET /api/inspection/rag/update-config`
- `GET /api/inspection/rag/documents`
- `GET /api/inspection/rag/vector-store`
- `POST /api/inspection/rag/jobs`
- `GET /api/inspection/rag/jobs/{job_id}`
- `POST /api/inspection/rag/jobs/{job_id}/cancel`

## 5. Responsibilities by Layer

- Endpoint layer (`app/server/api/*`)
  - HTTP contracts, request parsing, status codes, safe exception translation.
  - Endpoint classes are wired inline during router setup and do not retain named module-level service globals.
- Service layer (`app/server/services/*`)
  - Clinical orchestration, model orchestration, inspection workflows, job control.
- Domain models (`app/server/domain/*`)
  - Pydantic/domain request-response schemas and typed contracts.
- Repository layer (`app/server/repositories/*`)
  - SQL persistence, serialization, vector store access.
- Config/common layers (`app/server/configurations/*`, `app/server/common/*`)
  - Runtime settings, constants, environment/bootstrap, logging.
  - Shared security helpers, including provider-key cryptography, live under `app/server/common/security/cryptography.py`.

Frontend boundaries:
- `app/client/src/app/pages/*`: page orchestration and user journeys.
- `app/client/src/app/components/*`: reusable visual/interaction components.
- `app/client/src/app/core/services/*`: API transport and domain-specific API clients.
- `app/client/src/app/core/state/app-state.service.ts`: shared app state and theme/page state.

## 6. Layered Request Path (example)

Endpoint -> service -> repository:

- `POST /api/clinical/jobs`
  - `app/server/api/session.py`
  - `app/server/services/session/session_service.py`
  - `app/server/repositories/serialization/data.py` + DB repositories

- `GET /api/inspection/sessions`
  - `app/server/api/data_inspection.py`
  - `app/server/services/inspection/service.py`
  - `app/server/repositories/serialization/data.py`

## 7. Persistence Mechanisms

- Relational DB (SQLAlchemy):
  - SQLite file at `app/resources/database.db` when `database.embedded_database=true`
  - PostgreSQL when external DB mode is configured
- Vector persistence:
  - LanceDB collection under `app/resources/sources/vectors`
- Filesystem resources:
  - `app/resources/sources` for source catalogs/documents/archives
  - `app/resources/models` for model-related assets
- Access key encryption:
  - Encrypted provider keys persisted in DB tables with seeded encryption material.

## 8. Async vs Sync Behavior

- Mixed FastAPI handlers:
  - `async def` for network-bound paths (model listing/pull, research, some config/session paths).
  - `def` for lightweight synchronous handlers and job-status/control paths.
- Long-running tasks are not held in request lifecycle:
  - Managed by `JobManager` (`app/server/services/runtime/jobs.py`) using daemon threads.
  - Access the shared in-process manager through `get_job_manager()`.
  - Exposed via start/poll/cancel endpoints.
- Constraint:
  - CPU-heavy or blocking operations should run via job system instead of blocking request handlers.

## 9. Architectural Constraints

- `/api` is the stable frontend-backend boundary (`API_BASE_URL="/api"` in frontend constants).
- Runtime settings come from `settings/.env` and `settings/configurations.json`.
- Runtime and security helpers have canonical service modules; transitional shims are not maintained.
- No containerized runtime is currently implemented (no Dockerfile/compose in repository).
