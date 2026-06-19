# Persistence
Last updated: 2026-06-18

## Relational Database
- SQLAlchemy-backed storage
- Database mode and connection settings are sourced only from `settings/.env`
- SQLite file at `app/resources/database.db` when `database.embedded_database=true`
- PostgreSQL when external DB mode is configured

## Persisted Clinical Session Contract
- `clinical_sessions` is the single source of truth for session records, versioning, revision parentage, and session metadata.
- New sessions default to `version=1`.
- Revised sessions store `original_session_id` and an incremented `version`.
- Revision lineage and review state are stored additively in:
  - `clinical_session_versions`
  - `clinical_session_revision_runs`
  - `clinical_session_revision_steps`
  - `clinical_session_revision_artifacts`
  - `clinical_session_revision_entities`
  - `clinical_session_revision_reviews`
  - `clinical_session_manual_edits`
- Revision step and artifact persistence includes reviewer-instruction analysis outputs such as the normalized instruction profile and routing trace, explicit source-preprocessing mode records showing whether persisted source sections were reused or raw text was reparsed, source-version deterministic extraction reuse metadata for therapy/anamnesis/disease artifacts, source-version structured artifact reuse metadata for disease context, lab timeline, and onset context, source-version LiverTox match reuse-or-refresh provenance, source-version prior-assessment provenance for revised DILI assessments, explicit revision extraction-bundle outputs, revision entity-pipeline stage payloads, revision analysis and lookup drug-name selections, explicit revision candidate-reconciliation outputs for promoted drugs, the derived consultation-only revision entity snapshot context, revision consultation drug-name selections, a dedicated revision consultation execution payload including fallback, consultation-model, analysis-entrypoint, drug-analysis-entrypoint, report-finalization-entrypoint, conclusion-entrypoint, synthesis-mode, and revision-prompt provenance, a dedicated revision finalization execution payload, explicit final-report rebuild payloads, revision QA validation payloads, and report-comparison artifacts for each target version.
- First-class revised drugs, diseases, lab timeline entries, revision-aware LiverTox decisions, and revised DILI assessment records are also stored in `clinical_session_revision_entities` so the revision pipeline can preserve active entity outputs separately from the generic session payload.
- `clinical_session_revision_entities` rows now carry per-entity schema names such as `revised_drug_entry`, `revised_disease_entry`, `revised_lab_entry`, `revision_livertox_decision`, and `revised_dili_assessment`; repository persistence validates each payload against strict domain schemas before writing those rows.
- Patient timeline history is persisted only in `clinical_session_timelines`; session result payloads are not a timeline read source.
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
- `app/resources/tools`
  - Extraction tool manifests and deterministic tool-related assets

## Access Key Persistence
- Encrypted provider keys are persisted in database tables.
- Encryption material is seeded and managed through shared security helpers.
