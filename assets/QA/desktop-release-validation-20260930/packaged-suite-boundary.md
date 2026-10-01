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

The following full application workflows remain uncertified on the packaged
surface:

- a populated synthetic multi-drug clinical analysis, calculations, evidence,
  and report persistence;
- session CRUD, timeline, manual edits, revisions, lineage, and reload from a
  populated packaged data root;
- the native RAG folder picker, ingestion/retrieval/deduplication, and
  populated Data Inspection;
- file dialogs, external navigation, and broader WebView2 behavior;
- offline/provider failure, cancellation, retry, interrupted jobs, and
  recovery through the packaged UI;
- corrupted cache, incomplete extraction, permission failure, and
  single-instance contention through the UI.

These remain open because a fresh packaged data root has no populated
structured sources, and cloning the credential-bearing shared database into
the disposable root was intentionally avoided. Existing source-mode and
in-app Browser reports remain scoped to source or browser surfaces; they are
not MSI evidence. MSI-installed lanes are **BLOCKED/UNTESTED** by the
non-administrator token.

No patient information, provider credentials, tokens, or secrets were entered.