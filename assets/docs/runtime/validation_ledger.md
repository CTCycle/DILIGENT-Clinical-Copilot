# Pre-release validation ledger
Last updated: 2026-09-18

## Scope and interpretation

This is the canonical feature-state register for the current DILIGENT source/development tree. It records the validation run performed on 2026-09-18 against source revision `2e434ae1b0e276b8acbab27f65a399d306c3f691` on the `develop` branch. Packaging, publication, EXE/MSI smoke tests, and clean-machine installation were intentionally outside this audit.

The run used synthetic patient content only. No patient-identifying data was entered, no source refresh or embedding update job was started, and no live cloud-provider clinical analysis was submitted.

Status meanings:

- `PASS`: the named scope was executed and behaved as expected.
- `ATTENTION`: the surface is usable, but an external dependency, partial data state, or validation limitation remains.
- `FAIL`: the named scope has a reproducible release-blocking defect.
- `NOT TESTED`: current evidence is absent; this is not a claim that the capability is broken.
- `NOT APPLICABLE`: deliberately outside this source-only audit.

## Current release assessment

The original populated-database migration blocker has been remediated in source. SQLite migration transactions now suspend foreign-key enforcement only while Alembic performs the atomic parent-table rebuild, run `PRAGMA foreign_key_check` before commit, and restore the connection's prior enforcement state. The fix was verified with foreign keys enabled in the migration fixture and with a task-local clone of the current populated database through `start_on_windows.ps1 -Action InitializeDatabase`; all existing clinical/revision row counts and SQLite integrity checks were preserved. The shared source database was intentionally not advanced and remains at `202609100001`.

Release readiness remains blocked by the separate absence of live provider, populated-session, timeline, revision, citation, and final-report evidence. The migration remediation is not a claim that those clinical or external-provider gates passed.

Post-remediation register counts: `PASS 13`, `ATTENTION 3`, `FAIL 0`, `NOT TESTED 21`, `NOT APPLICABLE 2`.

## Feature-state register

| # | Stable capability | Area | Status | Current evidence, route, or scenario | Persistence / external dependency / gap |
|---:|---|---|---|---|---|
| 1 | Standard source launcher against the populated database | Runtime / migration | `PASS` | A task-local clone of the current populated database migrated through `202609170001` via `start_on_windows.ps1 -Action InitializeDatabase`; the same 18 sessions, 38 versions, 17 revision runs, and 54 artifacts remained available. | The shared `app/resources/database.db` was intentionally not advanced; the clone retained `PRAGMA foreign_key_check=[]` and `PRAGMA integrity_check=ok`. |
| 2 | Fresh SQLite bootstrap and Alembic head | Runtime / migration | `PASS` | Disposable source DB migrated through `202609170001`; `/api/health` returned 200. | Fresh DB evidence is complemented by the populated-clone remediation below. |
| 3 | Application shell and primary workspace navigation | UI | `PASS` | In-app Browser rendered DILI Agent, Clinical Sessions, Data Inspection, and Settings. | Browser smoke at one desktop viewport; no provider call. |
| 4 | General runtime settings persistence | Settings | `PASS` | Changed polling interval `1 -> 2`, saved, navigated, reloaded, and restored `2 -> 1`; value and persisted timestamp remained visible. | `settings/configurations.json` is back at baseline `1.0`; `.env` remained excluded. |
| 5 | Settings section surfaces | Settings | `PASS` | General, Models, Data Processing, Integrations, and Advanced routes rendered with source labels and controls. | Surface validation only for sections other than General persistence. |
| 6 | Local/cloud model configuration and access-key UX | Settings / external | `ATTENTION` | Local catalog load returned 48 models but Ollama was unavailable; cloud selection correctly reported no active OpenCode access key without accepting a secret. | Exact live provider lane was not exercised; graceful dependency failure observed. |
| 7 | Clinical input validation and preflight | Clinical workflow | `PASS` | Empty input showed four blocking reasons; synthetic valid input showed the two unavailable structured-source blockers before execution. | No analysis job was submitted because the preflight correctly blocked it. |
| 8 | Clinical analysis submission, progress, terminal state, and recovery | Clinical workflow | `NOT TESTED` | Not reached in the current browser run. | Requires populated source catalogs and a usable model lane. |
| 9 | Session creation and persisted result | Sessions / persistence | `NOT TESTED` | Not reached because clinical execution was blocked. | Existing shared rows were inspected read-only; current UI could not start on that DB. |
| 10 | Session listing, empty state, filters, and search shell | Sessions / UI | `PASS` | Fresh runtime showed the empty state and All/Successful/Failed, search, and date-filter controls. | Populated-session interaction remains untested. |
| 11 | Session selection, CRUD, deletion, and switch isolation | Sessions / persistence | `NOT TESTED` | Not reached with a current live populated runtime. | Requires a successful end-to-end session fixture. |
| 12 | Patient anamnesis, therapy, laboratories, metadata, and propagation | Clinical input | `NOT TESTED` | Synthetic text was entered and displayed, but no session was persisted. | Image and metadata subpaths are recorded separately below. |
| 13 | Patient profile image upload and image metadata | Clinical input / media | `NOT TESTED` | Upload control was present but no file was selected during this source-only run. | Needs a controlled synthetic image fixture and persistence check. |
| 14 | Multi-drug extraction, normalization, and RxNav identity linking | Clinical pipeline | `NOT TESTED` | Not reached in a live clinical run. | Unit coverage exists; current source-catalog/provider boundary was not proven. |
| 15 | Longitudinal laboratory capture and history propagation | Clinical pipeline | `NOT TESTED` | Not reached in a live clinical run. | Unit coverage exists; rendered session evidence is absent. |
| 16 | Deterministic R-score and injury-pattern classification | Clinical calculation | `PASS` | Current source calculation for ALT 360/ULN 40 and ALP 150/ULN 120 produced `R=7.2`, `hepatocellular`; boundary and missing-ULN tests passed. | Calculation is deterministic and independently recomputed; rendered final-report result not tested. |
| 17 | Per-drug assessment and cross-drug identity isolation | Clinical pipeline | `NOT TESTED` | Not reached in a live clinical run. | Requires multi-drug clinical execution and persisted evidence review. |
| 18 | LiverTox evidence retrieval and provenance | Structured evidence | `NOT TESTED` | Fresh disposable catalog was empty; preflight reported LiverTox unavailable. | Shared DB has source rows but cannot currently start through the migration. |
| 19 | RxNav evidence retrieval and provenance | Structured evidence | `NOT TESTED` | Fresh disposable catalog was empty; preflight reported RxNav unavailable. | Shared DB has source rows but cannot currently start through the migration. |
| 20 | FDA DILIrank evidence retrieval and provenance | Structured evidence | `NOT TESTED` | Fresh disposable DILIrank view was empty. | Shared DB contains DILIrank records, but populated-runtime validation is blocked. |
| 21 | RAG document listing and vector-store readiness | RAG | `PASS` | RAG view listed supported PDFs; read-only vector-store API reported collection ready, 1,317 embeddings, 99 distinct documents, dimension 384, cosine/IVF_FLAT. | Readiness/listing only; no clinical retrieval claim. |
| 22 | RAG retrieval inside a clinical analysis | RAG / clinical workflow | `NOT TESTED` | No clinical analysis was allowed past preflight. | Requires a completed analysis with persisted citations. |
| 23 | Bibliography and source provenance in the final report | Reporting | `NOT TESTED` | No final report was generated in the current live run. | Must be rechecked with structured sources and RAG enabled. |
| 24 | Final conclusion coherence and uncertainty presentation | Reporting | `NOT TESTED` | No final report was generated in the current live run. | Must be reviewed against the rendered report and persisted result. |
| 25 | Patient timeline generation, rendering, and persistence | Sessions / timeline | `NOT TESTED` | No selected populated session was available in the current runtime. | Requires a completed session and timeline job. |
| 26 | Agentic revision workflow, trace, artifacts, and persisted version | Revision | `NOT TESTED` | No revision was started in the current live run. | Shared DB contains historical revision rows, but the current app cannot start on it. |
| 27 | Manual report editing and version/history behavior | Revision / UI | `NOT TESTED` | Not reached in the current live run. | Requires a completed report and persisted version transition. |
| 28 | Human-review escalation and `requires_human_review` state | Revision / clinical safety | `NOT TESTED` | Not reached in the current live run. | Unit paths exist; rendered and persisted current-run evidence is absent. |
| 29 | Cancellation, concurrency, and stale-job guards | Background jobs | `NOT TESTED` | Not reached in the current live runtime. | Automated coverage exists; current populated migration remains a prerequisite for release evidence. |
| 30 | Structured-source inspection UI and dependency messaging | Data Inspection | `ATTENTION` | Drug Catalog, LiverTox, and DILIrank pages rendered empty on the disposable DB; RAG was populated; Update All/Update Embeddings were intentionally not invoked. | External/source data is not proven current; controls were not mutated during audit. |
| 31 | Ordered structured-source update and reconciliation jobs | Data Inspection / jobs | `NOT TESTED` | No source update job was started. | Deliberately excluded to avoid mutating source caches during validation. |
| 32 | Health and inspection API boundaries on a fresh runtime | API | `PASS` | `/api/health`, sessions, RxNav, LiverTox, DILIrank, RAG documents/vector-store, model-config, and settings requests returned expected 2xx responses. | Fresh disposable DB only. |
| 33 | Browser smoke, visible error handling, and console diagnostics | UI / QA | `PASS` | DILI Agent, Settings, Clinical Sessions, and Data Inspection rendered; browser error/warn diagnostics were empty; blocking dialogs were visible and actionable. | One viewport and smoke coverage; not a full accessibility audit. |
| 34 | Backend unit and supported model-config gates | Automated QA | `PASS` | `731 passed` unit tests; model-config slice `40 passed`; isolated app/API slice `10 passed`; migration slice now runs with SQLite foreign keys enabled. | Default test cache hit an ACL error and was rerouted to a writable QA cache. |
| 35 | Frontend test and production build gates | Automated QA | `PASS` | Angular/Vitest `23 files, 94 tests passed`; `npm run build` completed successfully. | No packaging assertion. |
| 36 | Exact OpenCode Go / DeepSeek end-to-end clinical provider lane | External provider | `NOT TESTED` | No live cloud clinical call was made; disposable DB had no active OpenCode key, and the shared runtime was blocked before provider resolution. | Must be validated without fallback before release. |
| 37 | Restart and reuse of existing persisted clinical data | Persistence / release | `ATTENTION` | The populated clone reopened through the launcher database-initialization path after migration, preserving 18 sessions, 45 drug mentions, 228 lab observations, 18 results, and SQLite integrity. | The shared source database was intentionally not advanced; rendered populated-session reuse remains untested. |
| 38 | EXE/MSI packaging, installer, checksum, and publication | Release packaging | `NOT APPLICABLE` | Explicitly outside this source/development audit. | Separate release gate. |
| 39 | Clean-machine install and Windows host smoke | Release packaging | `NOT APPLICABLE` | Explicitly outside this source/development audit. | Separate release gate. |

## Release blockers and required remediation

### B1 — populated-database migration remediation (resolved)

The current populated database has 38 `clinical_session_versions`, 17 revision runs, and 54 revision artifacts. The revision-run, review, and artifact tables contain foreign keys to `clinical_session_versions`. The migration coordinator now temporarily disables SQLite enforcement inside the existing atomic migration transaction, validates `PRAGMA foreign_key_check` before commit, and restores enforcement afterward. This allows migration `202609170001` to batch-recreate the parent table without weakening post-migration integrity.

The migration fixture now enables `PRAGMA foreign_keys=ON` and exercises revision runs, review rows, and artifacts. Ten migration tests and the full 731-test backend unit suite pass. A clone of the unchanged source database was migrated through the standard launcher initialization path with all populated counts preserved. The shared database remains unchanged by this remediation run, so direct source-runtime session UI reuse is still marked `ATTENTION`.

### B2 — current clinical/provider evidence is incomplete

Because the populated runtime cannot start, the audit could not prove a current rendered clinical report, source citations, timeline, revision, manual edit/history, or exact `opencode_go / deepseek-v4-flash` provider execution. These are `NOT TESTED`, not inferred from historical reports or catalog reachability.

### B3 — test cache permissions need an explicit environment decision

The first unit invocation failed before collection because the default cache/database location was not writable. The same suite passed after setting `DILIGENT_TEST_CACHE_ROOT` to the task-owned writable QA directory. This is classified as an environment/test-runner attention item, not a product pass.

## Validation diary

### 2026-09-18, Europe/Rome, approximately 12:26–12:47

1. Recorded the clean `develop` worktree and source revision `2e434ae1b0e276b8acbab27f65a399d306c3f691`; inspected the project index, runtime settings, and existing QA documentation.
2. Started with the prescribed launcher. It found the configured environments and waited for backend health, but the populated DB stopped at migration `202609170001`. A foreground source invocation reproduced the `sqlite3.IntegrityError` and exited. The shared DB remained at `202609100001` with `PRAGMA integrity_check=ok`.
3. Re-ran the application with a disposable SQLite DB through the source backend and client preview. Startup reached head, catalog seeding completed, health/API probes returned 2xx, and the UI rendered.
4. Exercised Settings persistence, model/access-key guardrails without entering a secret, clinical validation/preflight, empty Clinical Sessions, Data Inspection views, and RAG vector-store inspection. The synthetic case was not submitted to a model. No source update or embedding update job was invoked.
5. Ran the backend unit suite with a writable QA cache (`731 passed`), the supported model-config regression slice (`40 + 10 passed`), the frontend suite (`23 files / 94 tests passed`), and the production client build (successful).
6. Rechecked the browser after reload, verified the settings baseline remained `1`, inspected browser diagnostics (`[]`), stopped the task-owned backend/frontend listeners, and retained only small source-runtime log evidence under `assets/QA/pre-release-e2e-20260918/`. Disposable DB/cache files were removed.

### 2026-09-18, populated-database migration remediation

1. Added FK-safe SQLite migration coordination: enforcement is suspended only around the atomic Alembic transaction, `PRAGMA foreign_key_check` gates commit, and the prior connection state is restored.
2. Enabled foreign keys in the migration test engine and expanded the cancellation fixture to cover revision-run, review, and artifact child relationships.
3. Migrated a clone of the source database through `start_on_windows.ps1 -Action InitializeDatabase`; the clone reached `202609170001` with 18 sessions, 38 versions, 17 revision runs, 54 artifacts, `foreign_key_check=[]`, and `integrity_check=ok`.
4. Confirmed the source database remained at `202609100001` with its original counts and `integrity_check=ok`. The provider and live clinical workflow gates remain untested.

## Evidence register

- `app/server/migrations/versions/202609170001_add_cancelled_revision_status.py:40-45` — parent-table batch alteration now covered by the FK-safe migration transaction.
- `app/tests/unit/test_database_migrations.py:14-22,186-260` — migration fixture and its foreign-key configuration gap.
- `assets/QA/pre-release-e2e-20260918/manual-backend.stderr.log` — disposable source startup and request-runtime log.
- `assets/QA/pre-release-e2e-20260918/manual-backend.stdout.log` — disposable backend process output.
- `assets/QA/pre-release-e2e-20260918/manual-frontend-2.stdout.log` — disposable frontend preview output.
- `assets/QA/release-blocker-remediation-20260917.md` — historical evidence only; it was not used as current release proof.
- `assets/QA/release-blocker-remediation-20260918.md` — current FK-safe migration remediation evidence.

The in-app Browser captures from this run were inspected inline for DILI Agent, Settings, Clinical Sessions, and Data Inspection/RAG. The browser tool did not expose a disk-export path for those captures, so this ledger records the visual assertions and the reproducible routes rather than inventing screenshot filenames.
