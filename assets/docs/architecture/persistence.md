# Persistence
Last updated: 2026-07-10

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
- Sessions with blocking faithfulness issues may persist audit artifacts, but they must not be stored as clinically successful finalizations.
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
- Default runtime configuration pins the Ollama embedding model to an immutable tag and keeps vector collection reset disabled unless explicitly requested.

## Filesystem Resources
- `app/resources/sources`
  - Source catalogs, documents, and archives
- `app/resources/catalogs`
  - JSON seed manifests for database-backed reference catalogs, including extraction validation vocabulary

## Access Key Persistence
- Encrypted provider keys are persisted in database tables.
- Encryption material is seeded and managed through shared security helpers.


## Agentic revision artifacts
Revision runs persist bounded context, plan, tool trace, draft report, QA, and finalization artifacts. Successful non-dry runs create a new `agentic_revision` session and attach it to the pre-created version shell; QA blockers persist as `qa_failed` drafts for human review.
