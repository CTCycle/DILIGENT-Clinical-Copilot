# Persistence
Last updated: 2026-07-13

## Relational Database
- SQLAlchemy-backed storage
- Database mode and connection settings are sourced only from `settings/.env`
- SQLite file at `app/resources/database.db` when `database.embedded_database=true`
- PostgreSQL when external DB mode is configured

## Persisted Clinical Session Contract
- `clinical_sessions` is the single source of truth for session records, versioning, revision parentage, and session metadata.
- New sessions default to `version=1`.
- Revised sessions store original_session_id and an incremented ersion.
- Revision tables store the active revision-agent skeleton and may also contain historical local data:
  - clinical_session_versions
  - clinical_session_revision_runs
  - clinical_session_revision_steps
  - clinical_session_revision_artifacts
  - clinical_session_revision_entities
  - clinical_session_revision_reviews
  - clinical_session_manual_edits
- Manual edit history remains active through clinical_session_manual_edits.
- The current runtime does not execute or read the previous deterministic session revision pipeline.
- The active revision skeleton creates draft revision version shells, revision run rows, one `revision_agent_issue_scan` step, and a `revision_agent_issue_scan` pipeline artifact. It does not create revised clinical sessions or revised entity rows yet.
- Patient timeline history is persisted only in clinical_session_timelines; session result payloads are not a timeline read source.
- Timeline generation metadata stays on persisted timeline records. Generating or regenerating a timeline must not rewrite the original clinical session runtime metadata stored in the assessment payload.
- Evidence-locked DILI artifacts are persisted inside the database-backed session result payload:
  -
ormalized_document
  - xtraction_artifact
  - act_graph
  - aithfulness_audit
  - generated report metadata
  - discrepancy report
  - un_bundle_index
- Successful clinical workflows require persistence. Serializer failures, missing persisted ids, or failed upserts are treated as service dependency failures rather than silent in-memory success.
- SQLite enables foreign keys, a 30-second busy timeout, and WAL journaling for durable cross-connection behavior.
- SQLAlchemy update timestamps are assigned by an application-level update hook so SQLite and PostgreSQL receive the same `updated_at` behavior; `server_onupdate` is not relied on as a trigger.
- Version listing and detail reads are side-effect-free. Version synchronization is reserved for explicit write paths.
- Sessions with blocking faithfulness issues may persist audit artifacts, but they must not be stored as clinically successful finalizations.
- Durable loose JSON or Markdown assessment files are not part of the runtime contract.
- Canonical repeated observations are stored in `clinical_lab_observations` and ordered drug mentions in `clinical_drug_mentions`; the older summary tables remain only for the current inspection projection during migration.
- Access-key ciphertext remains in the database, while versioned Fernet key material can be supplied through the external `DILIGENT_ACCESS_KEY_MATERIAL_FILE` store rather than the database registry.
- Canonical drug identifiers use `drug_identifiers` with unique `(identifier_system, identifier_value)` ownership.
- `application_configuration` is the fixed-id singleton for validated configuration payloads, and `reference_catalog_manifests` records the currently installed manifest state.

## Reference Catalog Persistence
- Canonical manifests live in `app/resources/catalogs/*.json`.
- Catalogs are seeded into database tables.
- RxNorm code persistence uses `drug_rxnorm_codes` as the canonical RxCUI mapping table.
- Startup performs hash-based seed checks and only reseeds manifests that are missing or changed.
- Full reseed or reset is explicit through:

```powershell
app/scripts/initialize_database.py --drop-existing --seed-catalogs --force-reseed-catalogs
```

## Vector Persistence
- LanceDB collection under `app/resources/sources/vectors`
- RAG retrieval uses vector search, LanceDB full-text search, metadata-aware fusion, and lightweight local heuristic reranking profiles.
- Default runtime configuration pins the Ollama embedding model to an immutable tag and keeps vector collection reset disabled unless explicitly requested.

## Filesystem Resources
- `app/resources/sources`
  - Source catalogs, documents, and archives
- `app/resources/catalogs`
  - JSON seed manifests for database-backed reference catalogs, including extraction validation vocabulary

## Access Key Persistence
- Encrypted provider keys are persisted in database tables.
- Encryption material is seeded and managed through shared security helpers.
- Provider-scoped key retrieval filters by both provider and key id. The persistence constraint covers OpenAI, Gemini, DeepSeek, Anthropic, OpenCode, and Brave.

## Persistence Contract Validation

- `app/tests/persistence` runs against file-backed SQLite on every invocation.
- When `TEST_DATABASE_URL` is configured, the same parametrized contract runs against PostgreSQL.
- CI runs both backends in the `persistence-contract` job with a PostgreSQL service container.

## Model Configuration Persistence
- A newly initialized database receives the canonical model defaults from the server settings.
- Existing provider and model selections are read as stored; unsupported values fail validation and are not translated, invalidated, or replaced by another selection.
- The database initializer does not run a model-selection migration. Recreate the local database when discarding obsolete selections is intended.


## Agentic revision artifacts
Revision runs persist bounded context, plan, tool trace, draft report, QA, and finalization artifacts. Successful non-dry runs create a new `agentic_revision` session and attach it to the pre-created version shell; QA blockers persist as `qa_failed` drafts for human review.
