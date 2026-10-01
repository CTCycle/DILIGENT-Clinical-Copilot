# D01–D14 desktop release gate matrix

Candidate: `3.4.0`, commit `ce5df0c65fb432a42b2be622789e11f5c5283a43`.

Update: the 2026-10-01 packaged-UI slice
([report](../desktop-interactive-ui-validation-20261001/report.md)) drove the
packaged portable Tauri/WebView2 window with the available native UI
automation against an isolated data root. The D06 (workspaces + Settings
persistence), D07 preflight boundary, D12 API boundary, and D03
startup/restart/cleanup rows below now carry packaged portable evidence in
addition to the source/Browser evidence recorded previously.

Update: the 2026-10-01 populated-workflow slice
([report](../desktop-populated-ui-validation-20261001/report.md)) then ran the
full synthetic multi-drug analysis, session CRUD/lineage/manual-edit/revision,
timeline generation (fail-closed fallback), RAG ingestion/retrieval with
citations, cooperative cancellation, provider-timeout fail-closed, and the
access-key lifecycle in the packaged window against a populated isolated data
root (catalogs seeded via the repository's own update-persistence APIs after
the packaged Update All confirmed the real ordered pipeline and was
cooperatively cancelled). MSI-installed lanes remain BLOCKED by the
non-administrator token; the native RAG folder dialog and broader
file-dialog/WebView2 edge cases remain interaction boundaries the automated
session could not drive.

The D01–D14 labels below are the release-validation mapping for the requested
desktop checklist. `PASS` means the stated lane was actually exercised;
`BLOCKED` and `UNTESTED` are not inferred passes from source or Browser tests.

| Gate | Scope | Portable | MSI | Evidence / unresolved blocker |
|---|---|---|---|---|
| D01 | Candidate, branch, privilege, toolchain, provenance | PASS | PASS | `baseline.md`, `build-and-artifacts.md`; current token is not admin |
| D02 | Same-source build, architecture, version manifest, hashes, runtime inventory | PASS | PARTIAL | `build-and-artifacts.md`; MSI metadata/hash pass, but signing and lifecycle remain open |
| D03 | Startup, warm restart, shutdown, backend cleanup | PASS | BLOCKED | `portable-smoke.md`; packaged-UI 2026-10-01 slice: Alt+F4 clean close, no leftover process/listener, relaunch health 200, persisted settings survived restart. MSI install required for MSI lane |
| D04 | Writable paths, spaces/non-ASCII paths, second location, `%LOCALAPPDATA%\DILIGENT\data` containment | PARTIAL | UNTESTED | `portable-smoke.md`; short locations pass, deep repository path backend connectivity fails |
| D05 | Concurrent launch, forced termination, recovery, orphan-process cleanup | PASS | BLOCKED | `portable-smoke.md`; installed MSI lane not exercised |
| D06 | Workspaces, Settings, provider/model settings, reset/reload, encrypted-key rotation/redaction | PASS | BLOCKED | Packaged-UI 2026-10-01: all four workspaces rendered; General polling interval save/reload/reset/restart persistence proven in the packaged window + DB; Models surface (catalog, roles, keys, RAG) rendered. Populated slice: qwen3.5:9b role assignment saved/DB-verified and the OpenAI access-key lifecycle (create, encrypted storage, redaction, activate, one-active-key rotation, delete) passed in the packaged surface |
| D07 | Synthetic multi-drug analysis, calculations, evidence, report persistence | PASS | BLOCKED | Packaged preflight boundary passed (fresh-data blockers, zero sessions). Populated slice: full 15-step multi-drug analysis completed at 100% (job 1b0e1e57), report rendered with R-score 4.85, 3 matched drug mentions, RUCAM evidence; session + versions persisted. Session 1 stored `failed` = hard safety-gate requires_human_review (rechallenge/causality flags), consistent with documented fail-closed design |
| D08 | Session CRUD, timeline, edits, revisions, lineage, reload | PASS | BLOCKED | Populated slice: session detail/evidence render; timeline #1 generated (LLM timeout → fail-closed fallback chronology, 3 events with evidence); metadata JSON save persisted; manual edit superseded v1 → created current v2 (`manual_edit`) with hash audit; agentic revision workspace fail-closed at step 1 with timeout diagnostic; all state survived real restart |
| D09 | Native folder selection, RAG ingestion/retrieval/deduplication, Data Inspection | PARTIAL | BLOCKED | Populated slice: Data Inspection renders populated RxNav/LiverTox/DILIrank; RAG ingestion produced 2 unique documents → 2 chunks with byte-identical duplicate flagged; RAG-on analysis (session 2, successful) cited `acetaminophen.txt` in the Bibliography. The native IFileDialog could not be reliably driven in the automated session (returned last-used folder), so the packaged default RAG source folder was used; this interaction boundary remains open |
| D10 | File dialogs, external navigation, WebView2 behavior | PARTIAL | BLOCKED | Native UI automation covers the packaged window; the RAG folder dialog was invoked but not drivable, and the broader file-dialog/WebView2 edge suites were not exercised |
| D11 | Offline/provider failure, cancellation, retry, interrupted jobs, recovery | PASS | BLOCKED | Packaged Update-All cooperative cancellation (real ordered pipeline started, cancelled cleanly); clinical-job cancellation via Stop analysis (no partial session); timeline + revision LLM timeouts fail-closed with rendered diagnostics and retryable state |
| D12 | Localhost binding, authenticated API boundary, single-instance behavior, cleanup | PASS | BLOCKED | Packaged-UI 2026-10-01: health 200, unauthenticated settings 401, root 200, single instance, cleanup pass; MSI not installed |
| D13 | Fresh MSI install, standard-user use, registry/Program Files, uninstall, reinstall, data retention | UNTESTED | BLOCKED | `msi-upgrade-boundary.md`; requires admin-capable Windows session/UAC approval |
| D14 | Genuine 3.3.0→3.4.0 upgrade, DB integrity/data preservation, signatures/distribution | PARTIAL | BLOCKED | Published old MSI/hash and workflow contract verified; genuine upgrade, DB comparison, signing, clean-machine distribution not run |

## Unresolved blockers

1. Obtain an administrator-capable Windows session or human-approved UAC
   prompt, then run the exact MSI install/upgrade/uninstall command with the
   verified `upgrade-input` files and collect verbose logs plus database
   before/after comparisons.
2. Resolve or explicitly bound the backend failure under the deep repository
   path tested in `portable-smoke.md`.
3. Drive the native folder/file dialogs (the packaged IFileDialog could not be
   reliably automated in the populated slice; the packaged default RAG source
   folder was used instead). Do not relabel source Browser evidence.
4. Sign the final distribution artifacts and perform clean-machine/signature
   verification before any publication decision.
