# DILIGENT Clinical Copilot Architecture

Last updated: 2026-04-03

This document describes the current architecture at module level.
For runtime profile details, see `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`.
For job semantics, see `assets/docs/BACKGROUND_JOBS.md`.

## 1. System overview

DILIGENT is a local-first clinical application for DILI assessment with:
- Backend: FastAPI + SQLAlchemy + optional retrieval components.
- Frontend: React + TypeScript (Vite).
- Optional desktop shell: Tauri that starts a local backend.
- Optional cloud deployment: Docker (`backend` + `frontend` via Nginx).

## 2. Runtime request topology

- Backend entrypoint: `DILIGENT/server/app.py`.
- Frontend entrypoint: `DILIGENT/client/src/main.tsx`.
- Frontend calls backend through `/api/*`.
- In local dev, Vite proxies `/api/*` to FastAPI.
- In cloud mode, Nginx serves frontend assets and proxies allowed `/api/*` routes.
- In non-cloud mode, FastAPI registers both direct routes and mirrored `/api` routes.

## 3. Backend module boundaries

- `DILIGENT/server/api/*`
  - HTTP routes, request/response mapping, job start/poll/cancel endpoints.
- `DILIGENT/server/domain/*`
  - Typed payload models for clinical, jobs, models, keys, inspection, research.
- `DILIGENT/server/services/*`
  - Business logic: clinical analysis, updater flows, inspection orchestration, jobs.
  - Active clinical pipeline is request-driven from `api/session.py` and uses:
    - free-text laboratory parsing from `services/clinical/labs.py` (`laboratory_analysis` + supplemental anamnesis),
    - pattern derivation from parsed timelines in `services/clinical/hepatox.py`,
    - deterministic per-drug RUCAM estimation in `services/clinical/rucam.py`,
    - language detection and localized validation/report scaffolding.
- `DILIGENT/server/repositories/*`
  - DB and serialization boundaries (SQLite/Postgres, vector serialization, queries).
- `DILIGENT/server/models/*`
  - LLM provider clients, prompt templates, structured-output helpers.
- `DILIGENT/server/configurations/*`
  - Runtime settings and environment/config resolution.
- `DILIGENT/server/common/*`
  - Shared constants and utility functions.

## 4. Frontend boundaries

- `DILIGENT/client/src/pages/*`
  - Route-level screens (`DiliAgentPage`, `ModelConfigPage`, `DataInspectionPage`).
- `DILIGENT/client/src/components/*`
  - Reusable UI elements and modals.
- `DILIGENT/client/src/context/AppStateContext.tsx`
  - Shared application state.
- `DILIGENT/client/src/services/api.ts`
  - API client, response normalization, job polling behavior.
- `DILIGENT/client/src/types.ts`
  - Shared frontend contract types.

## 5. Current API surface (high level)

Core route groups:
- Session and clinical jobs:
  - `/clinical`
  - `/clinical/jobs`
  - `/clinical/jobs/{job_id}`
- Models and pull jobs:
  - `/models/list`
  - `/models/pull`
  - `/models/pull/jobs`
  - `/models/jobs/{job_id}`
- Model config:
  - `/model-config`
- Access keys:
  - `/access-keys`
  - `/access-keys/{id}/activate`
- Inspection:
  - `/inspection/sessions`
  - `/inspection/rxnav`
  - `/inspection/rxnav/update-config`
  - `/inspection/rxnav/jobs`
  - `/inspection/livertox`
  - `/inspection/livertox/update-config`
  - `/inspection/livertox/jobs`
  - `/inspection/rag/update-config`
  - `/inspection/rag/documents`
  - `/inspection/rag/vector-store`
  - `/inspection/rag/jobs`
- Research:
  - `/research`

All are mounted under `/api/*`; non-cloud mode also exposes direct routes.

## 6. Error and safety boundaries

- Global error handling is registered in `DILIGENT/server/api/error_handling.py`.
- Request-level correlation IDs are propagated via middleware/headers.
- Job errors are sanitized before user exposure (`services/jobs.py`).
- Sensitive internals remain server-side logs only.

## 7. Background jobs in active use

Current managed job types:
- `clinical`
- `ollama_pull`
- `rxnav_update`
- `livertox_update`
- `rag_update`

All follow start/poll/cancel patterns described in `assets/docs/BACKGROUND_JOBS.md`.

## 8. Data and resources

Project resources live under:
- `DILIGENT/resources/models`
- `DILIGENT/resources/sources`
- `DILIGENT/resources/logs`

Runtime/config files:
- `DILIGENT/settings/.env` (active profile)
- `DILIGENT/settings/.env.*.example` (profile templates)
- `DILIGENT/settings/configurations.json` (non-secret tuning defaults)

Maintenance boundary:
- `DILIGENT/setup_and_maintenance.bat` is reserved for database initialization and offline maintenance.
- RxNav/LiverTox/RAG dataset refresh operations are started from inspection UI wizards, not from `.bat` scripts.

## 9. Change impact map

- Clinical behavior change:
  - `server/api/session.py`, `server/services/clinical/*`, relevant unit/e2e tests.
- Model/provider behavior:
  - `server/api/model_config.py`, `server/api/ollama.py`, `client/src/pages/ModelConfigPage.tsx`, `client/src/services/api.ts`.
- Inspection/data updates:
  - `server/api/data_inspection.py`, `server/services/inspection.py`, updater services.
- Access key handling:
  - `server/api/access_keys.py`, `server/repositories/serialization/access_keys.py`, `client/src/components/AccessKeyModal.tsx`.
- Runtime/deployment behavior:
  - `docker/*`, `DILIGENT/start_on_windows.bat`, `release/tauri/*`, `.env profiles`.

## 10. Clinical request contract (active)

Current DILI request payload (high-level):
- `name`
- `visit_date`
- `anamnesis`
- `drugs`
- `laboratory_analysis`
- `use_rag`
- `use_web_search`
- optional runtime overrides (`use_cloud_services`, provider/model overrides)

Hard blockers in active pipeline:
- missing anamnesis
- missing visit date
- no drug with usable timing information
- insufficient laboratory data to determine hepatotoxicity pattern

Soft degradation:
- non-critical gaps are preserved as warnings in `issues` and do not block if core clinical inference is still possible.
