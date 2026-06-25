# Persistence
Last updated: 2026-06-21

## Relational Database
- SQLAlchemy-backed storage
- Database mode and connection settings are sourced only from `settings/.env`
- SQLite file at `app/resources/database.db` when `database.embedded_database=true`
- PostgreSQL when external DB mode is configured

## Persisted Clinical Session Contract
- `clinical_sessions` is the single source of truth for session records, versioning, revision parentage, and session metadata.
- New sessions default to `version=1`.
- Revised sessions store original_session_id and an incremented ersion.
- Historical revision tables may remain in local databases:
  - clinical_session_versions
  - clinical_session_revision_runs
  - clinical_session_revision_steps
  - clinical_session_revision_artifacts
  - clinical_session_revision_entities
  - clinical_session_revision_reviews
  - clinical_session_manual_edits
- Manual edit history remains active through clinical_session_manual_edits.
- The current runtime does not execute or read the previous session revision pipeline.
- Patient timeline history is persisted only in clinical_session_timelines; session result payloads are not a timeline read source.
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
- Durable loose JSON or Markdown assessment files are not part of the runtime contract.

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

## Filesystem Resources
- `app/resources/sources`
  - Source catalogs, documents, and archives
- `app/resources/catalogs`
  - JSON seed manifests for database-backed reference catalogs, including extraction validation vocabulary

## Access Key Persistence
- Encrypted provider keys are persisted in database tables.
- Encryption material is seeded and managed through shared security helpers.

