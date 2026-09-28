# Runtime configuration UI validation

Date: 2026-09-28

## Scope

Validated the database-backed Settings UI after removing the standalone
`settings/configurations.json` source. The official launcher was used:

```text
.\start_on_windows.ps1 -Action Launch
```

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Official launcher | PASS | Frontend production build completed; backend health returned `ok`; frontend served on `127.0.0.1:9847`. |
| Settings API source | PASS | `GET /api/settings` returned `source=database`, categories `general,data,integrations,matching,advanced`, and a persisted `updated_at`. |
| Database payload | PASS | Singleton contains `jobs`, `rag_settings`, `runtime`, `ingestion`, `session_pipeline`, `clinical_language_detection`, and `drugs_matcher`; `rag_settings` contains 16 typed fields; root legacy `rag` is absent. |
| Legacy file/archive removal | PASS | `settings/configurations.json` is absent and `app/desktop/build/runtime_payload.json` contains no reference to it. |
| Settings categories | PASS | Rendered in-app Browser checks showed the General, Data Processing, Integrations, Drug Matching, and Advanced categories with their typed controls and `Database / Settings UI` source label. |
| RAG controls | PASS | Models page RAG modal exposed Retrieval, Chunking, Embeddings, Ranking, and Index sections, including retrieval weights, chunking, embedding batch/offline mode, reranker profile, collection/index settings, local filesystem access, and vector stream batch. |
| Save/reload/reset | PASS | Changed minimum drug-name length from `3` to `4`; save confirmation and timestamp changed; reload retained `4`; reset returned it to typed default `3`. |
| Browser console | PASS | No captured error-level console entries. |

The in-app Browser screenshots for the General, Data Processing, save/reset,
Models/RAG, and rendered category states were captured and inspected inline
during validation. The Browser surface did not expose a disk-export path; the
observed rendered state and accessibility snapshots are preserved above as
the reproducible evidence record.
