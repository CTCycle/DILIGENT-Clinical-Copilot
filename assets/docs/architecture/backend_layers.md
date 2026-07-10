# Backend Layers
Last updated: 2026-07-10

## Responsibilities By Layer
- Endpoint layer: `app/server/api/*`
  - Owns HTTP contracts, request parsing, status codes, and safe exception translation.
  - Endpoint classes are wired inline during router setup and do not retain named module-level service globals.
  - `app/server/api/data_inspection.py` is the aggregate inspection router, and focused inspection endpoint modules live under `app/server/api/inspection/`.
- Service layer: `app/server/services/*`
  - Owns clinical orchestration, model orchestration, inspection workflows, and job control.
  - Inspection update orchestration is implemented in `app/server/services/inspection/update_jobs.py` through `DataInspectionUpdateJobRunner`.
  - `DataInspectionService` (in `app/server/services/inspection/service.py`) composes behavior from mixins in `update_config.py`, `revision_scaffold.py`, and `timeline.py`.
  - `ClinicalSessionService` (in `app/server/services/session/session_service.py`) composes behavior from mixins in `consultation.py` and `extraction_pipeline.py`.
  - Clinical and revision workflow helper code shared across session paths lives in `app/server/services/session/workflow_shared.py`; revision workflows must not import from first-run workflow modules.
  - Drug resolution lives in `app/server/services/clinical/drug_resolution/` and owns local-first RxNav catalog candidate generation, LiverTox candidate generation, deterministic acceptance policy, and prepared-input serialization.
  - RAG vector serialization lives in `app/server/services/rag/vector_serializer.py`.
  - `app/server/services/text/vocabulary.py` provides cache-facing text normalization business access and does not manage SQLAlchemy sessions directly.
  - `app/server/services/llm/ollama_runtime.py` owns canonical Ollama runtime aliases, errors, environment helpers, message normalization, and exception mapping. Ollama service modules must import these definitions instead of duplicating or monkey-patching them.
  - `app/server/services/llm/structured.py` owns strict JSON object extraction, schema validation, and bounded one-repair structured-output adaptation helpers for provider responses.
  - `app/server/services/llm/runtime_config.py` (formerly `configurations/llm_configs.py`) provides read access to persisted and overridden model configuration, reconciling database-backed settings with runtime defaults and per-run overrides. The configurations layer does not depend on repositories; this class lives in services to bridge both configuration and persistence.
  - `app/server/services/clinical/hepatox_constants.py` is the single source of truth for shared hepatox report regex patterns, used by both `hepatox_core.py` and `hepatox_scoring.py`.
  - Shared list deduplication `unique_preserve_order` lives in `common/utils/text_utils.py` instead of being triplicated across service modules.
- Domain layer: `app/server/domain/*`
  - Owns Pydantic and domain request-response schemas and typed contracts.
  - Clinical extraction schemas used by orchestration live under `app/server/domain/clinical/`.
  - Drug resolution decision schemas live in `app/server/domain/clinical/drug_resolution.py`.
  - Clinical claim envelopes live in `app/server/domain/clinical/claims.py` and are attached to per-drug DILI assessments for source and limitation review.
  - Deterministic DILI adjudication contracts live in `app/server/domain/clinical/dili.py`, including timeline events, Hy's Law state, componentized RUCAM, DILIN-like causality, and acceptance-question evidence payloads.
- Runtime state: `app/server/services/runtime/state.py`
  - Internal job state only. It is not a public domain contract and must not be imported by endpoints.
- Repository layer: `app/server/repositories/*`
  - Owns SQL persistence, serialization, and vector store access.
  - Access key persistence mapping and active key retrieval stay in `app/server/repositories/serialization/access_keys.py`.
  - Reference catalog persistence and seeding are implemented through `reference_catalog_entries` and `reference_catalog_seed_runs` in `app/server/repositories/serialization/catalogs.py`.
  - Database initialization (`repositories/database/initializer.py`) handles catalog seeding inline using `common/catalogs/manifest_loader.py` rather than the services layer, preserving strict layering during bootstrapping.
- Config and common layers: `app/server/configurations/*`, `app/server/common/*`
  - Own runtime settings, constants, environment bootstrap, logging, and shared security helpers.
  - Provider-key cryptography lives under `app/server/common/security/cryptography.py`.
  - Shared pure-utility modules (text normalization, chunking, seed terms, embedding model specs, list deduplication) live under `app/server/common/utils/` (`text_utils.py`, `chunking.py`, `seed_terms.py`, `embedding_model.py`) and are the canonical single source of truth — service modules import from here rather than duplicating logic.
  - Endpoint-layer request validation lives in `app/server/api/session_validation.py`.
  - Catalog snapshot provider (`common/catalogs/provider.py`) provides cross-layer access to reference catalog data through a registered provider pattern — service-layer runtime (`services/catalogs/runtime.py`) registers itself as the provider during import.
  - Catalog manifest loading (`common/catalogs/manifest_loader.py`) handles file I/O for catalog JSON manifests, decoupled from persistence logic.
  - Constants that depend on external catalog files (e.g., `CLOUD_MODEL_CHOICES`) are exposed as lazy accessor functions (`get_cloud_model_choices()`) to avoid import-time I/O side effects.
  - Logger configuration (`common/utils/logger.py`) defers file handler setup and `dictConfig` calls until `configure_logging()` is invoked during `initialize_settings()`, avoiding import-time side effects and global logging reconfiguration.

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
- Core section extraction is deterministic only and preserves verbatim source slices with heading spans, body spans, char spans, match strategy, confidence, source hash, and review flags.
- Low-confidence section extraction blocks preflight below `0.65`; review-required section extraction returns non-blocking preflight issues that the frontend must acknowledge before starting the job.
- Therapy, anamnesis, disease, and laboratory extraction use provider-agnostic structured LLM calls first for both cloud and local providers, then fall back to direct rule-based parsing after bounded validation or provider failures.
- Strict structured LLM parsing rejects prose-contaminated JSON and logs only sanitized repair diagnostics.
- Extracted drugs and diseases must carry source evidence, confidence, attribution, and current or diagnostic status where available.
- Hepatic pattern handling resolves explicit source-provided values separately from calculated R-ratio values and flags conflicts instead of silently overwriting the calculated score.
- LiverTox and RxNav matching preserve raw mentions, candidates, rejected candidates, origins, confidence, status, `DrugResolutionDecision`, accepted RxCUI, accepted LiverTox monograph identity, review flags, and warning issues for missing, ambiguous, low-confidence, or unvalidated matches.
- Resolution statuses are `accepted_exact_livertox`, `accepted_rxnav_validated`, `accepted_livertox_without_rxnav`, `ambiguous_requires_review`, `missing_rxnav`, `missing_livertox`, and `rejected_false_positive`.
- Persisted audit artifacts include section extraction audit, extraction strategy decisions, hepatic pattern resolution, and match audit details.
- Persisted clinical results include a readable `final_report` that uses the per-drug clinical consultation narrative and appends a concise deterministic DILI adjudication summary. The full rendered deterministic dossier is persisted separately as `pipeline_artifacts.structured_dili_report`, while `dili_evidence_bundle` remains the structured audit contract.
- Successful clinical jobs require database persistence. If the serializer cannot return a persisted session id, the workflow fails with a service dependency error rather than returning an unpersisted success.

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
- `app/server/api/inspection/revisions.py`
- `app/server/services/inspection/service.py`
- `app/server/services/inspection/revision_scaffold.py`
- `app/server/services/inspection/revision_agent.py`
- `app/server/repositories/serialization/data.py`
- Revision jobs currently implement the revision-agent skeleton only. They create a draft revision version shell, persist a revision run, and execute one single-model `revision_agent_issue_scan` step.
- The revision agent reviews the persisted session input, sections, generated report, result payload, optional selected text, and user instructions. It produces a structured issue inventory covering missing context, mismatched context, hallucination risk, ambiguity, unsupported claims, chronology gaps, and future tool needs.
- The skeleton does not rewrite the clinical report, rerun deterministic DILI adjudication, persist revised entities, or execute tools. Future tool routing is represented only as inert `tool_intents` in the issue-scan artifact.
- Revision result lineage is persisted through `revision_kind`, `source_session_id`, `source_version_id`, `revision_version_id`, and `pipeline_run_id`.

## Async And Sync Behavior
- FastAPI handlers are mixed:
  - `async def` for network-bound paths such as model listing, model pull, research-related work, and some config or session paths.
  - `def` for lightweight synchronous handlers and job-status or control paths.
- Long-running tasks do not stay inside the request lifecycle.
  - They are managed by `JobManager` in `app/server/services/runtime/jobs.py` using daemon threads.
  - Shared access goes through `get_job_manager()`.
  - Work is exposed through start, poll, and cancel endpoints.
- Job execution state is process-local. Revision run records remain durable, and stale `running` revision runs are failed/recoverable during service startup when their in-memory worker no longer exists.
- Job concurrency uses explicit scope keys: clinical assessment currently uses `clinical:global`, session revision uses `revision:{root_session_id}`, and catalog updates use `catalog:{job_type}`.
- CPU-heavy or blocking work must run through the job system instead of blocking request handlers.

## Architectural Constraints
- `/api` is the stable frontend-backend boundary and maps to frontend `API_BASE_URL="/api"`.
- Runtime settings come from `settings/.env` and `settings/configurations.json`.
- Database connection and database-mode values are sourced from `settings/.env`.
- Runtime settings are accessed through `get_server_settings()`.
- Runtime and security helpers use canonical service modules; transitional shims are not maintained.
- Supported external access-key providers are `openai`, `gemini`, and `brave`.
- Containerized runtime is not implemented in the current repository.

### Agentic revision boundary
`services/inspection/revision_agent.py` orchestrates a controlled revision loop. Context assembly, prompts, fixed tools, patch validation, and finalization stay in `services/inspection`; the workflow does not import or rerun first-run clinical workflow modules.
