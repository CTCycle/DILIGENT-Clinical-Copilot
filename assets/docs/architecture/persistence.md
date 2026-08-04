# Persistence
Last updated: 2026-08-03

## Relational Database

- SQLAlchemy-backed storage.
- Database mode and connection settings are sourced from `settings/.env`.
- SQLite uses `app/resources/database.db` when `database.embedded_database=true`; PostgreSQL is used in external database mode.

## Persisted Clinical Session Contract

- `clinical_sessions` is the source of truth for session records and metadata.
- `clinical_session_versions` owns immutable version lineage, root-session relationships, version numbers, and manual edits.
- Revision tables own the active revision-agent skeleton and canonical artifacts: `clinical_session_versions`, `clinical_session_revision_runs`, `clinical_session_revision_steps`, `clinical_session_revision_artifacts`, and `clinical_session_revision_reviews`.
- Structured revision entities are stored as `structured_case_entity` rows in `clinical_session_revision_artifacts`.
- Manual report edits create immutable `clinical_session_versions` rows with `revision_kind=manual_edit`.
- Patient timeline history is persisted only in `clinical_session_timelines`; session result payloads are not a timeline read source.
- Timeline generation metadata stays on persisted timeline records and does not rewrite original clinical-session runtime metadata.
- Evidence-locked DILI artifacts in the database-backed session result payload are `normalized_document`, `extraction_artifact`, `fact_graph`, `faithfulness_audit`, generated report metadata, discrepancy report, and `dili_evidence_bundle_index`.
- Successful clinical workflows require persistence. Repository failures, missing persisted IDs, and failed upserts are service dependency failures rather than silent in-memory success.
- SQLite enables foreign keys, a 30-second busy timeout, and WAL journaling.
- SQLAlchemy update timestamps use an application-level update hook shared by SQLite and PostgreSQL.
- Version listing and detail reads are side-effect-free; synchronization is reserved for explicit write paths.
- Durable loose JSON or Markdown assessment files are not part of the runtime contract.
- Canonical repeated observations are stored in `clinical_lab_observations` and ordered drug mentions in `clinical_drug_mentions`.
- Access-key ciphertext remains in the database; versioned Fernet key material is stored in the protected `DILIGENT_ACCESS_KEY_MATERIAL_FILE` store.
- Canonical drug identifiers use `drug_identifiers` with unique `(identifier_system, identifier_value)` ownership.
- `application_configuration` is the fixed-ID singleton for validated configuration payloads, and `reference_catalog_manifests` records installed manifest state.

## Focused Repository Ownership

- `KnowledgeRepository` owns evidence data.
- `DrugCatalogRepository` owns RxNav data.
- `ClinicalSessionRepository` owns session-result persistence.
- `SessionTimelineRepository` owns timeline rows.
- `SessionRevisionRepository` owns revision data, steps, and artifacts.
- Feature-specific file serialization remains separate from SQLAlchemy persistence. `RepositoryContext` supplies the shared engine/session factory, and application services receive only the focused repositories they need. Transactions remain explicit at the repository boundary, including atomic session persistence and batch ingestion.
- Repository serialization modules are pure row and payload conversion helpers; focused repositories own database queries and transactions and use `repositories.values` for canonical normalization.

## Reference Catalog Persistence

- Canonical manifests live in `app/resources/catalogs/*.json` and are seeded into database tables.
- RxNorm persistence uses `drug_rxnorm_codes` as the canonical RxCUI mapping table.
- Explicit database initialization seeds the canonical manifests.
- SQLite startup initializes and seeds only when the configured `.db` file is missing. An existing SQLite file is opened without schema cross-validation, reset, or reseeding.
- PostgreSQL database creation, schema initialization, and seeding are available only through the explicit launcher database action; normal application startup never creates or resets PostgreSQL state.
- Full reseed or reset remains explicit through `app/scripts/initialize_database.py --drop-existing --seed-catalogs --force-reseed-catalogs`.

## Vector Persistence

- LanceDB collections live under `app/resources/sources/vectors`.
- RAG uses the immutable Granite embedding contract in `common/embedding/config.py`; index generations carry a versioned manifest and exact embedding fingerprint.

## Filesystem Resources

- In development, `app/resources/sources` contains source catalogs, documents, archives, vectors, models, and logs.
- In packaged desktop mode, immutable catalogs and Angular assets live under the extracted runtime; databases, logs, models, source documents, vectors, exports, state, and access-key material live under `%LOCALAPPDATA%\DILIGENT\data`.
- `app/resources/catalogs` contains JSON seed manifests for database-backed reference catalogs and is copied to the immutable packaged runtime.

The extracted runtime is versioned and hash-addressed so it can be replaced during upgrades. The persistent data root is intentionally outside the runtime and is not removed by desktop artifact cleanup or MSI uninstall.

## Access Key Persistence

- Encrypted provider keys are persisted in database tables.
- Provider-scoped retrieval filters by both provider and key ID.
- The persistence constraint covers OpenAI, Gemini, DeepSeek, Anthropic, OpenCode, and Brave.

## Persistence Contract Validation

- `app/tests/persistence` runs against file-backed SQLite.
- When `TEST_DATABASE_URL` is configured, the same contract runs against PostgreSQL.
- CI runs both backends in the `persistence-contract` job.

## Model Configuration Persistence

- A newly initialized database receives canonical model defaults.
- Existing provider and model selections are read as stored; unsupported values fail validation and are not silently translated.
- Provider model catalogs are persisted in `provider_model_catalog_cache`, keyed by provider and a fingerprint of the catalog endpoint plus the active credential (or normalized Ollama endpoint). Secrets are never stored.
- `GET /api/model-config` reads cached catalog state only. The provider-specific `load` operation contacts a provider only for a cold cache, while `refresh` is the explicit replacement operation.
- Failed attempts are persisted with sanitized metadata. A failed cloud refresh preserves the last valid list; an empty Ollama installation is a valid cached result. Endpoint or active-key changes invalidate the old scope without affecting inactive credentials.
- The same SQLAlchemy schema is initialized for development and packaged desktop runtimes, so the persisted catalog survives navigation and application restarts without startup polling.

## Agentic Revision Artifacts

Revision runs persist bounded context, plan, tool trace, draft report, QA, and finalization artifacts. Successful non-dry runs create an `agentic_revision` session and attach it to the pre-created version shell; QA blockers persist as `qa_failed` drafts for human review.
