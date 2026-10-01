# Packaged workflow and failure-suite boundary

The exact candidate packaged shell reached startup, the authenticated-local
API boundary, native process title, restart, concurrency, forced termination,
and cleanup as recorded in `portable-smoke.md`.

The 2026-10-01 packaged-UI slice
([report](../desktop-interactive-ui-validation-20261001/report.md)) drove the
packaged Tauri/WebView2 window with the available native UI automation
against an isolated data root. The following are now certified on the
packaged portable surface:

- all four primary workspaces render (DILI Agent, Clinical Sessions, Data
  Inspection, Settings);
- Settings → General save/reload/reset and restart persistence (polling
  interval `1 → 2` saved, reload retained, reset back to `1`, and the
  persisted value survived a real process close/relaunch, verified in the
  packaged SQLite data root);
- Settings → Models renders the runtime source, local model catalog with
  install/assigned status, cloud provider keys, and current configuration;
- the DILI Agent preflight fails closed on a fresh packaged data root with
  actionable blocking items and zero sessions created;
- the packaged API boundary rechecks: `/api/health` 200, `/` 200,
  unauthenticated `/api/settings` 401;
- clean close (`Alt+F4`) terminates the packaged backend with no leftover
  process or listener, and relaunch restores health with persisted settings.

The 2026-10-01 populated-workflow slice
([report](../desktop-populated-ui-validation-20261001/report.md)) additionally
certifies on the packaged portable surface (against a populated isolated data
root, catalogs seeded via the repository's own update-persistence APIs):

- the full synthetic multi-drug analysis: 15-step pipeline to 100%, rendered
  report with R-score 4.85, per-drug commentary, LiverTox scores, RUCAM
  evidence, and persisted session/versions/drug-mentions; the safety-gate
  fail-closed `requires_human_review` storage of a flagged narrative;
- session CRUD/detail, metadata JSON save, manual report edit producing a
  superseded→current `manual_edit` version lineage with a full audit trail,
  the agentic revision workspace failing closed with a timeout diagnostic, and
  reload persistence across a real restart;
- timeline generation with the configured model, failing closed to a
  deterministic fallback chronology (3 events, all with evidence) with the
  timeout provenance rendered;
- RAG ingestion (`Update Embeddings`) — 3 physical files → 2 unique documents →
  2 vector chunks with the byte-identical duplicate flagged — plus RAG-on
  retrieval with citations (`acetaminophen.txt` in the report Bibliography);
- cooperative cancellation (Update-All and clinical job, the latter with no
  partial session persisted) and provider-timeout fail-closed paths
  (timeline/revision) with retryable state;
- the access-key lifecycle (encrypted storage, plaintext redaction,
  one-active-key rotation, deletion) in the packaged surface.

The following remain uncertified on the packaged surface:

- driving the native RAG/file folder dialogs (the packaged `IFileDialog` could
  not be reliably automated in the populated slice; the packaged default RAG
  source folder was used instead), external navigation, and broader WebView2
  behavior;
- corrupted cache, incomplete extraction, permission failure, and
  single-instance contention through the UI;
- MSI-installed lanes and genuine 3.3.0→3.4.0 upgrade (D13/D14) — blocked by
  the non-administrator token.

Existing source-mode and in-app Browser reports remain scoped to source or
browser surfaces; they are not MSI evidence. MSI-installed lanes are
**BLOCKED/UNTESTED** by the non-administrator token.

No patient information, provider credentials, tokens, or secrets were entered.