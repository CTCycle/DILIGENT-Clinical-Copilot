# Backend Layers
Last updated: 2026-06-03

## Responsibilities By Layer
- Endpoint layer: `app/server/api/*`
  - Owns HTTP contracts, request parsing, status codes, and safe exception translation.
  - Endpoint classes are wired inline during router setup and do not retain named module-level service globals.
- Service layer: `app/server/services/*`
  - Owns clinical orchestration, model orchestration, inspection workflows, and job control.
  - Inspection update orchestration is implemented in `app/server/services/inspection/update_jobs.py` through `DataInspectionUpdateJobRunner`, while `DataInspectionService` remains the endpoint-facing facade.
  - `app/server/services/text/vocabulary.py` provides cache-facing text normalization business access and does not manage SQLAlchemy sessions directly.
  - `app/server/services/llm/ollama_runtime.py` owns canonical Ollama runtime aliases, errors, environment helpers, message normalization, and exception mapping. Ollama service modules must import these definitions instead of duplicating or monkey-patching them.
- Domain layer: `app/server/domain/*`
  - Owns Pydantic and domain request-response schemas and typed contracts.
  - Clinical extraction schemas used by orchestration live under `app/server/domain/clinical/`.
- Runtime state: `app/server/services/runtime/state.py`
  - Internal job state only. It is not a public domain contract and must not be imported by endpoints.
- Repository layer: `app/server/repositories/*`
  - Owns SQL persistence, serialization, and vector store access.
  - Access key persistence mapping and active key retrieval stay in `app/server/repositories/serialization/access_keys.py`.
  - Reference catalog persistence and seeding are implemented through `reference_catalog_entries` and `reference_catalog_seed_runs` in `app/server/repositories/serialization/catalogs.py`.
- Config and common layers: `app/server/configurations/*`, `app/server/common/*`
  - Own runtime settings, constants, environment bootstrap, logging, and shared security helpers.
  - Provider-key cryptography lives under `app/server/common/security/cryptography.py`.

## Frontend Boundaries
- `app/client/src/app/pages/*`
  - Page orchestration and user journeys.
- `app/client/src/app/components/*`
  - Reusable visual and interaction components.
- `app/client/src/app/core/services/*`
  - API transport and domain-specific API clients.
- `app/client/src/app/core/state/*`
  - Shared app state, theme or page state, and reusable frontend state resources.

## Layered Request Paths
### `POST /api/clinical/jobs`
- `app/server/api/session.py`
- `app/server/services/session/session_service.py`
- `app/server/repositories/serialization/data.py` and DB repositories
- Performs clinical preflight before job creation, normalizes the submitted document, applies deterministic section-first extraction before clinical LLM extraction, persists evidence-locked pipeline artifacts in `session_result_payload`, and returns artifact and gate summaries through the job result.

### `POST /api/clinical/validate-input`
- `app/server/api/session.py`
- `app/server/services/session/preflight.py`
- Validates patient metadata, selected provider and model readiness, persistence reachability, extraction quality, and timed-drug feasibility without starting a background job.

### `GET /api/inspection/sessions`
- `app/server/api/data_inspection.py`
- `app/server/services/inspection/service.py`
- `app/server/repositories/serialization/data.py`

### `GET|PUT /api/inspection/sessions/{session_id}`
- `app/server/api/data_inspection.py`
- `app/server/services/inspection/service.py`
- `app/server/repositories/serialization/data.py`
- Session detail is the single session read surface for original text, parsed sections, metadata, AI preview payload, and revision audit data.

### `POST /api/inspection/sessions/{session_id}/revision/jobs`
- `app/server/api/data_inspection.py`
- `app/server/services/inspection/service.py`
- `app/server/services/session/session_service.py`
- `app/server/repositories/serialization/data.py`

## Async And Sync Behavior
- FastAPI handlers are mixed:
  - `async def` for network-bound paths such as model listing, model pull, research-related work, and some config or session paths.
  - `def` for lightweight synchronous handlers and job-status or control paths.
- Long-running tasks do not stay inside the request lifecycle.
  - They are managed by `JobManager` in `app/server/services/runtime/jobs.py` using daemon threads.
  - Shared access goes through `get_job_manager()`.
  - Work is exposed through start, poll, and cancel endpoints.
- CPU-heavy or blocking work must run through the job system instead of blocking request handlers.

## Architectural Constraints
- `/api` is the stable frontend-backend boundary and maps to frontend `API_BASE_URL="/api"`.
- Runtime settings come from `settings/.env` and `settings/configurations.json`.
- Database connection and database-mode values are sourced from `settings/.env`.
- Runtime settings are accessed through `get_server_settings()`.
- Runtime and security helpers use canonical service modules; transitional shims are not maintained.
- Supported external access-key providers are `openai`, `gemini`, and `brave`.
- Containerized runtime is not implemented in the current repository.
