# Exhaustive UI and system journey certification

Date: 2026-07-23  
Application: DILIGENT Clinical Copilot  
Runtime: backend `127.0.0.1:7690`, frontend `127.0.0.1:9847`  
LLM policy: every LLM-backed journey used OpenCode Go with `deepseek-v4-flash`.

## Executive summary

The restarted application is operational and the principal user journeys are covered through the in-app browser and canonical API contracts. Existing successful clinical sessions, editor persistence, metadata persistence, document upload/remove, timeline generation, catalog inspection, model configuration, invalid preflight, responsive layout, and background-job cancellation were exercised end to end.

Two defects were found and repaired during certification:

1. Manual report edits could reuse a version number already occupied by an orphaned revision shell, causing a SQLite uniqueness failure. The allocator now uses the maximum version number across the root session lineage, and a regression test covers the orphaned-version case.
2. Revision review actions sent UI values (`approved`/`rejected`) where the canonical API requires `approved_by_human`/`rejected_by_human`; cancelled or failed jobs also exposed review controls. The client now maps to canonical statuses and hides review controls for cancelled/failed jobs.

The main unresolved product concern is readiness of local RAG embeddings: the inventory advertises `ollama:nomic-embed-text:latest`, while current vectorization settings report no Ollama embedding model and the update job fails because the local model runtime is unavailable. This was not changed because it would alter global configuration and is outside the requested key-safe certification scope.

## Coverage

### Initial load, navigation, and runtime

- Restarted backend and frontend after system reboot.
- `/api/health` returned 200; frontend root returned 200.
- Root navigation loaded DILI Agent, Clinical Sessions, Data Inspection, and Configurations.
- Reloaded the app and returned to Clinical Sessions; session data remained available.
- No API key was added, deleted, activated, or otherwise mutated. The existing OpenCode credential remained active; only metadata was inspected.
- Configuration state remained Cloud / OpenCode Go with `deepseek-v4-flash` for clinical and extraction roles.
- Connectivity check `POST /api/model-config/connectivity-check` with OpenCode Go / `deepseek-v4-flash` returned 200 and `{ ok: true, response_preview: "OK" }`.

### DILI Agent

- Restored the experimental QA clinical case after invalid-input testing.
- Exercised Clear all and Run with empty fields. The UI displayed a blocking preflight modal with the expected missing Visit Date and empty Clinical input issues.
- Existing report controls correctly remained disabled when no report was generated.
- Earlier certification also exercised valid DILI analysis with and without RAG using the configured DeepSeek Flash model. Both long-running live runs were safely cancelled; existing successful persisted sessions were used to certify report rendering and downstream workflows.

### Clinical Sessions

- Loaded and rendered session 9, including report headings, patient/visit summary, drug evidence, laboratories, limitations, and bibliography.
- Search filter: `QA OpenCode` isolated session 9; reset restored the complete list.
- Status filters: All and Successful were exercised.
- Date filters: On date `2026-07-22` included session 9; Before `2026-07-22` included older sessions such as Bertolli Mauro and excluded the QA session; After had no matching same-day records as expected.
- Existing experimental session deletion was exercised earlier (session 5 removed); API session list now reports seven persisted sessions.
- Text Editor: source/rendered view, font decrease/increase, format selection, undo/redo, bold/italic/strike, bullet/number list, and link controls were inspected/exercised.
- Manual edit: appended a temporary marker through the UI, verified persistence and version advancement through the API, then restored the original report. Session 9 ended at version 8 with no marker.
- Metadata: saved and restored `manual_metadata`; attached `pasted-text-1.txt` through the document file chooser, verified it in the attachment list and JSON, then removed it and restored empty experimental metadata.
- Revision: started a real OpenCode Go / DeepSeek Flash revision job, observed it remain running for 60 seconds, cancelled it, and verified the job API reached `cancelled`. The cancelled draft was cleaned up through the canonical clinical-review API using `rejected_by_human`.
- Timeline: generated Timeline #11 with OpenCode Go / DeepSeek Flash, reached a completed saved state with five events and five evidence-backed events, opened the timeline, exercised evidence filtering, uncertain/empty-lane toggles, density, zoom, event inspector, and timeline settings reuse, then deleted the experimental Timeline #11. Existing Timeline #10 remains.

### Data Inspection

- RxNav: searched amoxicillin, opened and closed the aliases dialog, started an update job, requested cancellation, and verified job `72e4cb63` reached `cancelled` through the API.
- LiverTox: searched amoxicillin, opened and closed the Amoxicillin-Clavulanate excerpt, started an update job, requested cancellation, and verified job `21e40575` reached `cancelled` through the API.
- RAG: searched the document inventory, selected the existing RAG folder through the folder chooser, opened update settings, and started an embedding update. Job `bafdc5ce` reached `failed` with the surfaced error that the local model runtime is unavailable.
- Existing catalog/RAG counts and vector readiness endpoints were inspected during the prior sweep; no source data or credentials were removed.

### Configurations

- Loaded the cloud model catalog and verified the selected clinical and extraction model is `deepseek-v4-flash`.
- Searched the model catalog for DeepSeek models and refreshed the catalog.
- Opened and closed the OpenCode access-key modal. The key was displayed only as masked metadata; no mutation was performed.
- RAG settings exposed the readiness mismatch documented above. Global configuration was not changed.

### UI, accessibility, and resilience checks

- Responsive checks at a small viewport and desktop viewport reported no horizontal overflow.
- Interactive controls on the DILI Agent had zero unlabeled input/select/textarea elements in the bounded audit.
- Representative keyboard focus and accessible names were inspected.
- Refresh and SPA navigation were exercised after live data mutations.
- Dialogs, disabled/loading states, cancellation states, and error states were inspected for the major workflows.

## Findings and resolutions

### Resolved: orphaned revision version collision

- Area: Backend / integration; severity high.
- Reproduction: create failed/orphaned revision shells for a session, then save a manual report edit.
- Actual result before fix: the manual edit attempted a version number already occupied in the root lineage and returned a SQLite uniqueness error.
- Resolution: manual edit allocation now calculates the maximum `ClinicalSessionVersion.version_number` for the root session and increments it. Regression test: `test_manual_edit_skips_orphaned_revision_version_numbers`.

### Resolved: revision review contract and invalid cancelled-state controls

- Area: Frontend / integration; severity high.
- Reproduction: start a DeepSeek Flash revision, cancel it, then use the exposed Reject control.
- Actual result before fix: the UI submitted `rejected`; the API rejected it with `Input should be 'under_review', 'approved_by_human' or 'rejected_by_human'`. Cancelled drafts also retained review buttons.
- Resolution: the client maps UI actions to canonical statuses and only exposes review actions for reviewable terminal states. The cancelled draft was verified with the canonical API after cleanup.

### Open: RAG embedding runtime readiness

- Area: Backend / integration; severity medium-high.
- Reproduction: open Data Inspection → RAG → Update Embeddings with current configuration.
- Expected: the configured inventory embedding model is available or the UI clearly directs the user to a valid configured runtime.
- Actual: settings showed Ollama embedding model `Not set`; the update job failed with `Local model runtime is unavailable` even though inventory rows displayed `ollama:nomic-embed-text:latest`.
- Recommendation: align the persisted RAG vectorization configuration with the inventory/runtime readiness contract, or make the unavailable state explicit before analysis.

### Observation: long-running LLM jobs

- Revision and fresh DILI analysis jobs can remain in progress for several minutes with no terminal result. Cancellation did reach backend terminal states for revision and data-update jobs, and the DILI analysis cancellation path returned a cancellation message. Review job polling and cancellation responsiveness should be monitored in a future performance pass.

### Observation: update-job UI polling lag

- RxNav reached `cancelled` through the API while the open dialog still displayed `Processing RxNav records` several seconds after cancellation. The dialog was closed after API verification. This is an integration/UI freshness concern requiring a follow-up focused on polling termination and terminal-state refresh.

## Automated verification

- Backend: `566 passed, 2 skipped, 22 warnings` via `app/server/.venv/Scripts/python.exe -m pytest -q`.
- Frontend: `12 test files passed, 42 tests passed` via `npm test -- --watch=false --no-progress`.
- Frontend production build: succeeded. Existing SCSS budget warnings remain for `clinical-sessions-page.component.scss` (35.80 kB vs 20 kB) and `model-config-page.component.scss` (21.13 kB vs 20 kB).

## Boundary and unverified concerns

- This certification does not claim exhaustive screen-reader or WCAG conformance; it covers accessible-name presence, representative focus behavior, and responsive overflow.
- No API-key lifecycle mutation was tested by design, per explicit instruction.
- No legacy compatibility paths were added or considered; validation used current canonical routes and payload contracts only.
- External data refresh completion was intentionally not awaited after cancellation; terminal API states were verified instead.

## Live handoff

Backend and frontend remain running for continued inspection at `http://127.0.0.1:9847/` with backend health at `http://127.0.0.1:7690/api/health`.
