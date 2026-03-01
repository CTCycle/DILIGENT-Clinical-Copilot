# DILIGENT Clinical Copilot Architecture

DILIGENT Clinical Copilot is a local-first web application for Drug-Induced Liver Injury (DILI) clinical assessment. It combines a FastAPI backend with a React + TypeScript (Vite) frontend.

---

## 1. Repository Structure

- `DILIGENT/server/`: backend API, clinical pipeline, repositories, and background jobs.
- `DILIGENT/client/`: React frontend (DILI analysis page + model configuration page).
- `DILIGENT/common/`: shared constants and utility modules.
- `DILIGENT/settings/`: runtime configuration (`.env`, `.env.local.example`, `.env.cloud.example`, `configurations.json`).
- `DILIGENT/scripts/`: maintenance and data refresh scripts (`initialize_database.py`, `update_drugs_catalog.py`, `update_livertox_data.py`, `update_RAG.py`).
- `DILIGENT/resources/`: runtime data (logs, models, sources, vectors, templates).
- `docker/` + `docker-compose.yml`: containerized backend/frontend runtime.

---

## 2. Runtime Topology

- Backend: Uvicorn/FastAPI (`DILIGENT.server.app:app`) on `FASTAPI_HOST:FASTAPI_PORT` (default `127.0.0.1:8000`).
- Frontend: Vite preview server locally on `UI_HOST:UI_PORT` (default `127.0.0.1:7861`) and Nginx in Docker.
- API path contract: frontend calls `/api/*`, then proxy rewrites to backend root endpoints.
- Root behavior: backend `/` redirects to `/docs` (OpenAPI UI).

---

## 3. Backend Architecture

### 3.1 Layering
1. Routes: FastAPI routers in `server/routes/` define HTTP endpoints and response models.
2. Services: domain logic in `server/services/` (clinical parsing, disease extraction, retrieval, updater jobs, key cryptography).
3. Repositories: SQLAlchemy + database backend abstraction in `server/repositories/`.
4. Entities: request/response schemas and typed domain payloads in `server/entities/`.
5. Configuration: runtime config loading and in-memory model/runtime selection in `server/configurations/`.

### 3.2 Core API Surface
- `POST /clinical`: synchronous report generation.
- `POST /clinical/jobs`, `GET /clinical/jobs/{job_id}`, `DELETE /clinical/jobs/{job_id}`: async clinical analysis with polling/cancel.
- `GET /models/list`, `GET /models/pull`, `POST /models/pull/jobs`, `GET/DELETE /models/jobs/{job_id}`: local Ollama model management.
- `GET/PUT /model-config`: runtime model/provider settings and catalog exposure.
- `GET/POST /access-keys`, `PUT /access-keys/{id}/activate`, `DELETE /access-keys/{id}`: encrypted cloud API key lifecycle.

### 3.3 Clinical Analysis Pipeline
1. Validate and normalize patient request payload.
2. Compute hepatotoxicity pattern (ALT/ALP multiples and R score).
3. Extract therapy drugs and anamnesis drugs.
4. Extract structured disease context from anamnesis.
5. Optionally build RAG queries if `use_rag=true`.
6. Resolve LiverTox/RxNorm evidence and run LLM consultation.
7. Compose markdown report + structured JSON payload.
8. Persist session artifacts and match metadata in SQL tables.

### 3.4 Persistence and Retrieval
- DB mode is runtime-selectable: embedded SQLite (`DB_EMBEDDED=true`) or external PostgreSQL (`DB_EMBEDDED=false`).
- Clinical session outputs are persisted across normalized tables (sessions, sections, labs, drugs).
- Drug catalog and LiverTox monographs are stored in relational tables and used during matching.
- RAG vectors are stored via LanceDB under `DILIGENT/resources/sources/vectors`.

---

## 4. Frontend Architecture

- Root shell (`client/src/App.tsx`) uses `AppStateContext` for shared runtime/form/job state.
- Page routing is path-based:
  - `/`: `DiluAgentPage` (clinical input, job progress polling, markdown report rendering/export).
  - `/model-config`: `ModelConfigPage` (local model selection, cloud provider/model selection, key management access).
- Navigation is split between a compact left sidebar and settings entry.
- API client logic is centralized in `client/src/services/api.ts` and includes timeout handling and polling helpers.

---

## 5. Background Jobs

Long-running operations use a thread-based `JobManager` in `server/services/jobs.py`. Current job types are:
- `clinical`: full DILI analysis pipeline.
- `ollama_pull`: model pull/download.

Detailed behavior is documented in [BACKGROUND_JOBS.md](./BACKGROUND_JOBS.md).

---

## 6. Maintenance and Data Updates

Operational scripts under `DILIGENT/scripts/`:
- `initialize_database.py`: initializes SQL schema for active DB mode.
- `update_drugs_catalog.py`: refreshes RxNav-derived drug catalog.
- `update_livertox_data.py`: ingests/refreshes LiverTox data.
- `update_RAG.py`: re-embeds documents and refreshes vector index.

Windows operators can run these via `DILIGENT/setup_and_maintenance.bat`.
