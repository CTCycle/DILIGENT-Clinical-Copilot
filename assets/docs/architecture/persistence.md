# Persistence
Last updated: 2026-06-03

## Relational Database
- SQLAlchemy-backed storage
- SQLite file at `app/resources/database.db` when `database.embedded_database=true`
- PostgreSQL when external DB mode is configured

## Persisted Clinical Session Contract
- `clinical_sessions` is the single source of truth for session records, versioning, revision parentage, and session metadata.
- New sessions default to `version=1`.
- Revised sessions store `original_session_id` and an incremented `version`.
- Evidence-locked DILI artifacts are persisted inside the database-backed session result payload:
  - `normalized_document`
  - `extraction_artifact`
  - `fact_graph`
  - `faithfulness_audit`
  - generated report metadata
  - discrepancy report
  - `run_bundle_index`
- Durable loose JSON or Markdown assessment files are not part of the runtime contract.

## Reference Catalog Persistence
- Canonical manifests live in `app/resources/catalogs/*.json`.
- Catalogs are seeded into database tables.
- Startup performs hash-based seed checks and only reseeds manifests that are missing or changed.
- Full reseed or reset is explicit through:

```powershell
app/scripts/initialize_database.py --drop-existing --seed-catalogs --force-reseed-catalogs
```

## Vector Persistence
- LanceDB collection under `app/resources/sources/vectors`
- RAG retrieval uses vector search, LanceDB full-text search, metadata-aware fusion, and local cross-encoder reranking.

## Filesystem Resources
- `app/resources/sources`
  - Source catalogs, documents, and archives
- `app/resources/models`
  - Model-related assets

## Access Key Persistence
- Encrypted provider keys are persisted in database tables.
- Encryption material is seeded and managed through shared security helpers.
