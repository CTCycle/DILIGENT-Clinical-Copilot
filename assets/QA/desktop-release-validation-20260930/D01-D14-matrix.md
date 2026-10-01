# D01–D14 desktop release gate matrix

Candidate: `3.4.0`, commit `ce5df0c65fb432a42b2be622789e11f5c5283a43`.

Update: the 2026-10-01 packaged-UI slice
([report](../desktop-interactive-ui-validation-20261001/report.md)) drove the
packaged portable Tauri/WebView2 window with the available native UI
automation against an isolated data root. The D06 (workspaces + Settings
persistence), D07 preflight boundary, D12 API boundary, and D03
startup/restart/cleanup rows below now carry packaged portable evidence in
addition to the source/Browser evidence recorded previously. MSI-installed
lanes remain BLOCKED by the non-administrator token, and populated
session/timeline/analysis/RAG flows remain unexercised because a fresh
packaged root has no structured sources and cloning the credential-bearing
shared database was intentionally avoided.

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
| D06 | Workspaces, Settings, provider/model settings, reset/reload, encrypted-key rotation/redaction | PARTIAL | BLOCKED | Packaged-UI 2026-10-01: all four workspaces rendered; General polling interval save/reload/reset/restart persistence proven in the packaged window + DB; Models surface (catalog, roles, keys, RAG) rendered. Key rotation/redaction and populated session/timeline flows remain unexercised in the packaged surface |
| D07 | Synthetic multi-drug analysis, calculations, evidence, report persistence | PARTIAL | BLOCKED | Packaged preflight boundary passed (blocked with actionable blockers on fresh data, zero sessions created); full packaged multi-drug analysis not run (no populated sources) |
| D08 | Session CRUD, timeline, edits, revisions, lineage, reload | UNTESTED | BLOCKED | Packaged UI flow not exercised; requires populated packaged data root |
| D09 | Native folder selection, RAG ingestion/retrieval/deduplication, Data Inspection | PARTIAL | BLOCKED | Data Inspection rendered in the packaged window; native folder dialog and ingestion/retrieval not exercised |
| D10 | File dialogs, external navigation, WebView2 behavior | BLOCKED | BLOCKED | Native UI automation covers the packaged window, but full file-dialog and WebView2 edge suites were not exercised |
| D11 | Offline/provider failure, cancellation, retry, interrupted jobs, recovery | BLOCKED | BLOCKED | Packaged failure flows not exercised |
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
3. Provide native desktop/window automation so the packaged UI workflows and
   failure conditions can be exercised. Do not relabel source Browser evidence.
4. Sign the final distribution artifacts and perform clean-machine/signature
   verification before any publication decision.
