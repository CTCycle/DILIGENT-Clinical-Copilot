# DILIGENT Clinical Copilot Architecture

DILIGENT Clinical Copilot is a local-first DILI assessment system:
- Backend: FastAPI + SQLAlchemy + optional LanceDB vector retrieval.
- Frontend: React + TypeScript (Vite), plus optional Tauri desktop shell.
- Runtime modes: local launcher, Docker cloud profile, and packaged desktop mode.

---

## 1. Runtime and Request Topology

- Backend entrypoint is `DILIGENT/server/app.py` (`uvicorn DILIGENT.server.app:app`).
- Frontend entrypoint is `DILIGENT/client/src/main.tsx`; API calls are sent through `/api/*`.
- In local Vite, `/api/*` is proxied to FastAPI with prefix rewrite.
- In Docker, Nginx serves static frontend and proxies only approved `/api` paths to backend.
- Backend exposes both direct routes (for local/dev) and mirrored `/api` routes (for proxy/production).
- In Tauri mode, Rust bootstrap code starts and supervises a local Python backend process.

---

## 2. Backend Layers and Responsibilities

1. `routes/`: HTTP contract, endpoint orchestration, job lifecycle APIs.
2. `entities/`: typed request/response and pipeline data models.
3. `services/`: clinical parsing, toxicity assessment, matching, retrieval, updater pipelines, key cryptography.
4. `models/`: LLM providers (Ollama + cloud), prompts, structured output parsing.
5. `repositories/`: SQL persistence and vector DB serialization/querying.
6. `configurations/`: `.env` + JSON configuration resolution and runtime defaults.
7. `common/`: constants, regex patterns, environment utilities, coercion helpers.

Core endpoints:
- `/clinical`
- `/clinical/jobs`, `/clinical/jobs/{job_id}`
- `/inspection/*` (sessions catalog/report, RxNav catalog/aliases/update jobs, LiverTox catalog/excerpt/update jobs)
- `/models/list`, `/models/pull`, `/models/pull/jobs`, `/models/jobs/{job_id}`
- `/model-config`
- `/access-keys`, `/access-keys/{id}/activate`
- `/research`

Boundary error policy:
- Centralized API exception handling is registered in `server/api/error_handling.py`.
- Every request receives a correlation header `X-Request-ID`.
- Unhandled exceptions are logged with request context and returned as safe, non-sensitive error payloads.

---

## 3. Frontend Structure

- Global app shell in `DILIGENT/client/src/App.tsx` with context state in `context/AppStateContext.tsx`.
- Main pages:
- `pages/DiliAgentPage.tsx`: clinical form, async job polling, progress UI, report rendering/export.
- `pages/ModelConfigPage.tsx`: model/provider selection, pull actions, cloud key management.
- `pages/DataInspectionPage.tsx`: data inspection route entry.
- API layer in `services/api.ts` centralizes fetch, timeout, error normalization, and polling loop.

---

## 4. Complete Tracked File Map (Scope + Implementation Notes)

This section is intended as the "where to edit what" map so agents do not need to open files blindly.

### 4.1 Root Files

- `.gitignore`: ignore rules for runtime artifacts, builds, caches.
- `README.md`: run modes, setup, runtime profile switching, usage overview.
- `LICENSE`: Polyform Noncommercial license text.
- `pyproject.toml`: Python package metadata, pinned backend dependencies, test extras.
- `uv.lock`: workspace lockfile copied from `runtimes/uv.lock` during launcher flows.
- `docker-compose.yml`: backend/frontend services, healthchecks, loopback backend publish, shared `diligent_resources` volume.

### 4.2 Project Docs and Figures

- `assets/docs/ARCHITECTURE.md`: architecture and file navigation reference.
- `assets/docs/BACKGROUND_JOBS.md`: job manager semantics and polling contract.
- `assets/docs/GENERAL_RULES.md`: mandatory operating rules for agents.
- `assets/docs/GUIDELINES_PYTHON.md`: Python coding conventions.
- `assets/docs/GUIDELINES_TESTS.md`: testing conventions.
- `assets/docs/GUIDELINES_TYPESCRIPT.md`: frontend TypeScript conventions.
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md`: runtime profile and packaging behavior.
- `assets/docs/README_WRITING.md`: README authoring standard.
- `assets/figures/.gitkeep`: placeholder for empty dir tracking.
- `assets/figures/manifest.json`: screenshot manifest for docs.
- `assets/figures/home.png`: landing page screenshot.
- `assets/figures/dashboard.png`: dashboard/report screenshot.
- `assets/figures/settings.png`: settings screenshot.
- `assets/figures/models-list.png`: model catalog screenshot.
- `assets/figures/model-detail.png`: model/key detail screenshot.
- `assets/figures/data-inspection.png`: data inspection screenshot.
- `assets/figures/database_browser.png`: database browser screenshot.
- `assets/figures/configurations_page.png`: configurations page screenshot.
- `assets/figures/session_page.png`: session page screenshot.

### 4.3 Launchers, Settings, Resources

- `DILIGENT/start_on_windows.bat`: portable runtime bootstrapper (Python/uv/Node), dependency sync, backend+frontend start.
- `DILIGENT/setup_and_maintenance.bat`: interactive maintenance menu (logs cleanup, uninstall, DB/data refresh, desktop cleanup).
- `DILIGENT/settings/.env`: active runtime config consumed by backend and launch scripts.
- `DILIGENT/settings/.env.local.example`: local profile template.
- `DILIGENT/settings/.env.cloud.example`: cloud/docker profile template.
- `DILIGENT/settings/.env.local.tauri.example`: desktop profile template.
- `DILIGENT/settings/configurations.json`: app tuning blocks (`jobs`, `rag`, `external_data`, `ingestion`, `drugs_matcher`).
- `DILIGENT/resources/logs/.gitkeep`: persistent logs directory placeholder.
- `DILIGENT/resources/models/.gitkeep`: local model assets directory placeholder.
- `DILIGENT/resources/sources/archives/.gitkeep`: archived source dataset placeholder.
- `DILIGENT/resources/sources/documents/.gitkeep`: RAG document corpus directory placeholder.
- `DILIGENT/resources/sources/vectors/.gitkeep`: LanceDB storage directory placeholder.

### 4.4 Frontend Web App (`DILIGENT/client`)

- `DILIGENT/client/index.html`: Vite HTML host page.
- `DILIGENT/client/package.json`: frontend deps and scripts (`dev`, `build`, `preview`, Tauri helper scripts).
- `DILIGENT/client/tsconfig.json`: browser TS compiler options.
- `DILIGENT/client/tsconfig.node.json`: Vite/node TS options.
- `DILIGENT/client/vite.config.ts`: local/preview API proxy and host/port setup.
- `DILIGENT/client/public/favicon.svg`: web favicon source.
- `DILIGENT/client/public/favicon.png`: raster favicon and Tauri icon source.
- `DILIGENT/client/src/main.tsx`: React mount point.
- `DILIGENT/client/src/App.tsx`: route-to-page shell and context provider wiring.
- `DILIGENT/client/src/styles.css`: global styles and component/page CSS.
- `DILIGENT/client/src/types.ts`: shared API/domain TypeScript types (jobs, model config, access keys, payloads).
- `DILIGENT/client/src/constants.ts`: default runtime settings, model lists, API constants, timeout defaults.
- `DILIGENT/client/src/utils.ts`: payload sanitization, visit-date normalization, report download URL helpers.
- `DILIGENT/client/src/modelConfig.ts`: cloud-provider/model choice normalization and runtime setting merge logic.
- `DILIGENT/client/src/context/AppStateContext.tsx`: global app state store, page-path mapping, state mutation actions.
- `DILIGENT/client/src/services/api.ts`: request wrappers, timeout + response parsing, clinical job poller, model/access-key APIs.
- `DILIGENT/client/src/pages/DiliAgentPage.tsx`: main workflow UI; starts clinical job, polls status, handles missing-labs confirmation, renders markdown output.
- `DILIGENT/client/src/pages/ModelConfigPage.tsx`: local model role assignment, provider/cloud model controls, pull actions, access key integrations.
- `DILIGENT/client/src/pages/DataInspectionPage.tsx`: full inspection UI (sessions catalog + report view, RxNav/LiverTox searchable previews, updater controls, modals).
- `DILIGENT/client/src/components/AccessKeyModal.tsx`: key create/list/activate/delete modal UI.
- `DILIGENT/client/src/components/BooleanToggle.tsx`: reusable toggle control.
- `DILIGENT/client/src/components/ConfigModal.tsx`: generic config modal wrapper.
- `DILIGENT/client/src/components/ConfirmModal.tsx`: reusable confirmation dialog.
- `DILIGENT/client/src/components/NavSidebar.tsx`: navigation tabs for top-level pages.
- `DILIGENT/client/src/components/ProviderAccessCard.tsx`: provider card UI for key management actions.
- `DILIGENT/client/src/components/StatusMessage.tsx`: status tone resolution and status banner component.
- `DILIGENT/client/src/hooks/useAccessKeyManager.ts`: stateful access-key CRUD/activation hook over API service.
- `DILIGENT/client/src/hooks/useModelPullActions.ts`: model pull job trigger and status management hook.
- `DILIGENT/client/src/hooks/useObjectUrlLifecycle.ts`: object URL creation/revocation lifecycle hook.

### 4.5 Desktop Wrapper (`DILIGENT/client/src-tauri`)

- `DILIGENT/client/src-tauri/Cargo.toml`: Rust crate config for Tauri shell.
- `DILIGENT/client/src-tauri/build.rs`: Tauri build-script shim.
- `DILIGENT/client/src-tauri/tauri.conf.json`: bundle metadata and desktop window/runtime config.
- `DILIGENT/client/src-tauri/src/main.rs`: desktop bootstrap; resolves workspace/runtime payload, syncs runtimes, launches backend, renders startup screens, handles shutdown.
- `DILIGENT/client/src-tauri/capabilities/default.json`: Tauri capability manifest.
- `DILIGENT/client/src-tauri/icons/128x128.png`: desktop icon asset.
- `DILIGENT/client/src-tauri/icons/128x128@2x.png`: high-density desktop icon asset.
- `DILIGENT/client/src-tauri/icons/32x32.png`: desktop icon asset.
- `DILIGENT/client/src-tauri/icons/64x64.png`: desktop icon asset.
- `DILIGENT/client/src-tauri/icons/Square107x107Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square142x142Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square150x150Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square284x284Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square30x30Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square310x310Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square44x44Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square71x71Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/Square89x89Logo.png`: Windows tile icon asset.
- `DILIGENT/client/src-tauri/icons/StoreLogo.png`: Windows Store logo asset.
- `DILIGENT/client/src-tauri/icons/icon.icns`: macOS icon bundle.
- `DILIGENT/client/src-tauri/icons/icon.ico`: Windows icon bundle.
- `DILIGENT/client/src-tauri/icons/icon.png`: generic icon asset.

### 4.6 Backend App (`DILIGENT/server`)

- `DILIGENT/server/__init__.py`: package marker.
- `DILIGENT/server/app.py`: FastAPI app construction, cloud-mode docs disabling, router registration, `/api` mirrored routes, SPA static serving for packaged mode.
- `DILIGENT/server/common/__init__.py`: package marker.
- `DILIGENT/server/common/constants.py`: path constants, table names, model choice defaults, matching stopwords, source URLs, DILI thresholds.
- `DILIGENT/server/common/utils/__init__.py`: package marker.
- `DILIGENT/server/common/utils/logger.py`: shared logger setup.
- `DILIGENT/server/common/utils/patterns.py`: compiled regex patterns reused across parsing/matching.
- `DILIGENT/server/common/utils/types.py`: robust coercion helpers (`coerce_bool/int/float/str`, positive int extraction, normalized string tuple coercion).
- `DILIGENT/server/common/utils/variables.py`: environment variable loader container.
- `DILIGENT/server/configurations/__init__.py`: configuration API re-exports.
- `DILIGENT/server/configurations/base.py`: JSON configuration file loading with mapping validation.
- `DILIGENT/server/configurations/server.py`: full runtime settings model; env override resolution; settings builders for fastapi/jobs/db/matcher/rag/external data/LLM defaults.
- `DILIGENT/server/domain/__init__.py`: package marker.
- `DILIGENT/server/domain/clinical.py`: core Pydantic clinical pipeline schemas (request payload, drug entries, disease context, hepatotoxicity scores, match metadata, final report structures).
- `DILIGENT/server/domain/jobs.py`: background job API response schemas.
- `DILIGENT/server/domain/keys.py`: access key request/response schemas.
- `DILIGENT/server/domain/model_configs.py`: model config response/update DTOs and local model card schema.
- `DILIGENT/server/domain/models.py`: model listing/pull API DTOs.
- `DILIGENT/server/domain/research.py`: research request/citation/source/response DTOs.
- `DILIGENT/server/models/__init__.py`: package marker.
- `DILIGENT/server/models/cloud.py`: cloud LLM client abstraction (OpenAI/Gemini), model listing, chat, embeddings, structured output repair loop.
- `DILIGENT/server/models/prompts.py`: prompt templates for drug extraction, disease extraction, RAG query generation, per-drug analysis, and final conclusion synthesis.
- `DILIGENT/server/models/providers.py`: Ollama client (chat/embed/stream/pull/model caching), residency policy, memory checks, structured-call fallback orchestration, provider selection logic.
- `DILIGENT/server/models/structured.py`: JSON extraction/parsing utilities for structured LLM outputs.
- `DILIGENT/server/repositories/__init__.py`: lazy export hook for repository modules.
- `DILIGENT/server/repositories/database/__init__.py`: package marker.
- `DILIGENT/server/repositories/database/backend.py`: backend protocol + factory (`SQLiteRepository`/`PostgresRepository`) and unified DB facade.
- `DILIGENT/server/repositories/database/initializer.py`: DB URL/connect-args composition and initialization helpers for SQLite/Postgres.
- `DILIGENT/server/repositories/database/postgres.py`: PostgreSQL repository implementation.
- `DILIGENT/server/repositories/database/sqlite.py`: SQLite repository implementation.
- `DILIGENT/server/repositories/database/utils.py`: engine normalization and SQL identifier validation helpers.
- `DILIGENT/server/repositories/queries/__init__.py`: package marker.
- `DILIGENT/server/repositories/queries/data.py`: query gateway wrappers for load/upsert/count/stream/paginated table access.
- `DILIGENT/server/repositories/schemas/__init__.py`: package marker.
- `DILIGENT/server/repositories/schemas/models.py`: SQLAlchemy ORM schema definitions (clinical sessions/results/sections/labs/drugs, monographs, aliases, model selections, access keys).
- `DILIGENT/server/repositories/schemas/types.py`: custom SQLAlchemy type helpers.
- `DILIGENT/server/repositories/serialization/__init__.py`: package marker.
- `DILIGENT/server/repositories/serialization/access_keys.py`: encrypted key persistence, provider activation semantics, key lookup CRUD.
- `DILIGENT/server/repositories/serialization/data.py`: high-volume persistence service for clinical sessions, LiverTox and RxNav upsert logic, alias linking, document loading/chunking/vector serialization.
- `DILIGENT/server/repositories/serialization/model_config.py`: model config persistence/snapshot serializer.
- `DILIGENT/server/repositories/vectors.py`: LanceDB wrapper (table init, dimensionality checks, index creation, record load/stream).
- `DILIGENT/server/api/__init__.py`: package marker.
- `DILIGENT/server/api/access_keys.py`: `/access-keys` CRUD + activate endpoints.
- `DILIGENT/server/api/error_handling.py`: global API error boundaries, request-ID middleware, safe exception-to-response mapping.
- `DILIGENT/server/api/model_config.py`: `/model-config` GET/PUT endpoint class; resolves provider/model defaults and local model availability cards.
- `DILIGENT/server/api/ollama.py`: `/models/list`, `/models/pull`, pull job start/status/cancel endpoints.
- `DILIGENT/server/api/research.py`: `/research` endpoint class for Tavily-backed evidence response.
- `DILIGENT/server/api/session.py`: `/clinical` and clinical job endpoints; progress callbacks; payload build/merge/validation; per-patient pipeline orchestration.
- `DILIGENT/server/api/data_inspection.py`: `/inspection` endpoints for sessions/RxNav/LiverTox catalogs, detail views, deletes, and updater job lifecycle.
- `DILIGENT/server/services/__init__.py`: package marker.
- `DILIGENT/server/services/jobs.py`: in-memory thread-backed `JobManager` and lifecycle/status tracking.
- `DILIGENT/server/services/payload.py`: incoming payload sanitization and visit-date normalization.
- `DILIGENT/server/services/clinical/__init__.py`: package marker.
- `DILIGENT/server/services/clinical/disease.py`: anamnesis disease extraction service and structured disease context generation.
- `DILIGENT/server/services/clinical/hepatox.py`: hepatotoxicity score calculation and end-to-end consultation engine (RAG retrieval, web evidence enrichment, per-drug analysis, final report generation).
- `DILIGENT/server/services/clinical/livertox.py`: LiverTox dataset service and lookup utilities.
- `DILIGENT/server/services/clinical/matches.py`: canonical/synonym/fuzzy matching pipeline with bounded caches and confidence scoring; ties therapy/anamnesis names to LiverTox records.
- `DILIGENT/server/services/clinical/parser.py`: therapy/anamnesis medication extraction (LLM + rule-based fallback), schedule/route/date parsing and normalization.
- `DILIGENT/server/services/clinical/preparation.py`: pre-analysis data preparation and match packaging.
- `DILIGENT/server/services/keys/__init__.py`: package marker.
- `DILIGENT/server/services/keys/cryptography.py`: fernet-based key encrypt/decrypt/fingerprint helpers.
- `DILIGENT/server/services/research/__init__.py`: package marker.
- `DILIGENT/server/services/research/tavily.py`: Tavily query rewrite, domain filtering, source extraction/caching/rate limiting, citation-aware answer generation.
- `DILIGENT/server/services/retrieval/__init__.py`: package marker.
- `DILIGENT/server/services/retrieval/embeddings.py`: embedding generation and similarity search/reranking pipeline.
- `DILIGENT/server/services/retrieval/query.py`: DILI-focused search query builder.
- `DILIGENT/server/services/text/__init__.py`: package marker.
- `DILIGENT/server/services/text/normalization.py`: drug query canonicalization and token-level cleanup helpers.
- `DILIGENT/server/services/text/synonyms.py`: synonym payload parsing/splitting utilities.
- `DILIGENT/server/services/updater/__init__.py`: package marker.
- `DILIGENT/server/services/updater/embeddings.py`: orchestrates corpus chunking + embedding refresh into vector DB.
- `DILIGENT/server/services/updater/livertox.py`: LiverTox archive/masterlist download, metadata validation, parsing/sanitization into unified dataset.
- `DILIGENT/server/services/updater/livertox_sanitizer.py`: LiverTox excerpt cleaning rules.
- `DILIGENT/server/services/updater/rxnav.py`: RxNav client + catalog builder with async prefetch, synonym/brand expansion, batch DB upserts.
- `DILIGENT/server/services/inspection.py`: inspection-layer orchestration for list/detail/delete operations and RxNav/LiverTox update jobs.

### 4.7 Operational Scripts and Containers

- `DILIGENT/scripts/initialize_database.py`: initializes DB schema for active backend mode.
- `DILIGENT/scripts/update_drugs_catalog.py`: runs RxNav catalog refresh pipeline.
- `DILIGENT/scripts/update_livertox_data.py`: runs LiverTox ingestion/update pipeline.
- `DILIGENT/scripts/update_rag.py`: runs RAG document vectorization refresh.
- `docker/backend.Dockerfile`: uv-based Python image; frozen sync from lockfile; FastAPI startup.
- `docker/frontend.Dockerfile`: Node build stage + Nginx runtime stage for static frontend.
- `docker/nginx/default.conf`: strict route whitelist for proxied APIs; blocks docs/openapi endpoints in cloud mode.

### 4.8 Desktop Release Packaging

- `release/tauri/build_with_tauri.bat`: release build orchestrator; validates portable runtimes, stages junction payload, builds, exports artifacts.
- `release/tauri/scripts/clean-tauri-build.ps1`: cleans `src-tauri/target/release` and `release/windows`.
- `release/tauri/scripts/export-windows-artifacts.ps1`: exports installers and portable bundle with required payload checks.

### 4.9 Runtime and Locking Support

- `runtimes/.gitkeep`: runtime directory placeholder for portable tools.
- `runtimes/uv.lock`: authoritative runtime lockfile copied by launch/build scripts.

### 4.10 Test Suite (`tests`)

- `tests/run_tests.bat`: end-to-end test runner that boots backend/frontend, runs unit then e2e, and tears down processes.
- `tests/conftest.py`: base URL fixtures and API context fixture.
- `tests/e2e/test_root_api.py`: root/docs/openapi routing checks.
- `tests/e2e/test_clinical_api.py`: request validation and clinical endpoint contract checks.
- `tests/e2e/test_models_api.py`: model list/pull endpoint behavior checks.
- `tests/e2e/test_access_keys_api.py`: access key API CRUD contract checks.
- `tests/e2e/test_research_api.py`: research endpoint behavior with/without Tavily key.
- `tests/e2e/test_app_flow.py`: high-level UI navigation smoke tests.
- `tests/e2e/test_rxnav_concurrency_diagnostic.py`: optional RxNav concurrency probe/diagnostic.
- `tests/unit/test_access_keys.py`: encryption storage and single-active-key semantics.
- `tests/unit/test_anamnesis_disease_extraction.py`: disease extraction quality and context shaping.
- `tests/unit/test_anamnesis_drug_extraction.py`: anamnesis drug extraction paths and fallback behavior.
- `tests/unit/test_database_mode_env_override.py`: DB mode/env override behavior.
- `tests/unit/test_drugs_parser.py`: therapy line parsing (schedule/route/dates/state).
- `tests/unit/test_external_data_timeouts.py`: timeout settings safety and override behavior.
- `tests/unit/test_hepatox_assessment.py`: hepatotoxicity assessment logic and retry behavior.
- `tests/unit/test_livertox_excerpt_sanitizer.py`: excerpt cleanup rules.
- `tests/unit/test_livertox_matching_pipeline.py`: matching confidence and ambiguity logic.
- `tests/unit/test_ollama_residency_policy.py`: dual-residency and model prefetch policy logic.
- `tests/unit/test_pandas_migration.py`: serialization/dataframe normalization regressions.
- `tests/unit/test_polling_interval_centralization.py`: centralized polling interval usage.
- `tests/unit/test_rag_settings_reranking.py`: reranking settings parsing and floors.
- `tests/unit/test_research_rewrite_and_selection.py`: research query rewrite and source selection behavior.
- `tests/unit/test_rxnav_catalog_concurrency.py`: RxNav batch prefetch and persistence behavior.
- `tests/unit/test_seed_scripts_idempotency.py`: idempotent seeding/upsert guarantees.
- `tests/unit/test_similarity_reranking.py`: similarity reranking behavior and fallback.
- `tests/unit/test_sqlite_repository_initialization.py`: sqlite schema initialization behavior.

---

## 5. Fast Navigation by Change Type

- Clinical output bug: start at `server/api/session.py`, then `services/clinical/hepatox.py`, `services/clinical/preparation.py`, `services/clinical/matches.py`.
- Drug parsing issue: `services/clinical/parser.py`, `services/text/normalization.py`, tests `test_drugs_parser.py` and `test_anamnesis_drug_extraction.py`.
- Model/provider config issue: backend `routes/model_config.py` + `repositories/serialization/model_config.py`; frontend `pages/ModelConfigPage.tsx`, `modelConfig.ts`, `constants.ts`.
- Access key issue: backend `routes/access_keys.py`, `repositories/serialization/access_keys.py`, `services/keys/cryptography.py`; frontend `useAccessKeyManager.ts`, `AccessKeyModal.tsx`.
- RAG/vector issue: `services/updater/embeddings.py`, `services/retrieval/embeddings.py`, `repositories/vectors.py`, `scripts/update_rag.py`.
- LiverTox/RxNav ingestion issue: `services/updater/livertox.py`, `services/updater/rxnav.py`, `scripts/update_livertox_data.py`, `scripts/update_drugs_catalog.py`.
- Docker/API gateway issue: `docker-compose.yml`, `docker/nginx/default.conf`, `docker/backend.Dockerfile`, `docker/frontend.Dockerfile`.
- Desktop packaging/runtime issue: `client/src-tauri/src/main.rs`, `release/tauri/build_with_tauri.bat`, `release/tauri/scripts/*.ps1`.

---

## 6. Background Jobs

Thread-backed jobs are centralized in `DILIGENT/server/services/jobs.py`:
- `clinical`
- `ollama_pull`

Detailed lifecycle semantics are in `assets/docs/BACKGROUND_JOBS.md`.


