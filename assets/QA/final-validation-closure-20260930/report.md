# DILIGENT final validation-closure campaign

Date: 2026-09-30
Branch: `develop`
Final commit: `0150ecfede7acfb580a5bc38561ff0aa5a6a4ad7`
Scope: required local source validation, deterministic RAG ingestion/retrieval,
and locally produced Windows 3.4.0 packaging

## Outcome

The required source validation scope is closed. `api.local-boundaries` is
validated across the complete current OpenAPI operation set,
`rag.ingestion-retrieval` is validated with deterministic byte-identical
deduplication, `ISSUE-005` is resolved, and the final automated regression
matrix is green. Portable 3.4.0 packaging passed locally.

The MSI artifact itself passed checksum and metadata inspection, but MSI
install/upgrade/uninstall could not be executed on this host because the
validation session is not an administrator and the package is `ALLUSERS=1`.
That is recorded as a host prerequisite, not as an MSI product PASS. An
administrator-host run remains required before publication.

## API local-boundaries

`app/tests/e2e/test_api_local_boundaries.py` now derives an explicit
method/path matrix from `/openapi.json`. The final live application exposed 68
paths and 85 operations. The test fails if the current operation set changes
without an explicit classification.

The matrix covers root/runtime, clinical, model/configuration, access keys,
session/timeline/revision, inspection/catalog, runtime observation,
structured-source jobs, and RAG. Each operation has a deterministic success,
validation, not-found, cancellation, persistence, or explicitly referenced
external integration boundary. Network-backed refresh execution is not invoked
by this local contract test; it remains owned by `data.sources.refresh`.

Final command:

```powershell
$env:DILIGENT_BROWSER_E2E_CACHE_ROOT='G:\Projects\Repositories\Active projects\DILIGENT Clinical Copilot\assets\QA\.scratch-final-browser-e2e-final'
$env:PLAYWRIGHT_BROWSERS_PATH='G:\Projects\Repositories\Active projects\DILIGENT Clinical Copilot\runtimes\cache\playwright'
.\app\tests\ci\run_browser_e2e.ps1 -Suite Full
```

The transient JUnit at
`assets/QA/.scratch-final-browser-e2e-final/pytest/browser-e2e-logs/full-junit.xml`
recorded the final result and was removed with the disposable scratch roots
after verification.

Result: **53 passed, 0 failures, 0 errors, 0 skipped**.

## ISSUE-005 and RAG deduplication

The implementation fingerprints raw bytes with SHA-256 and groups only
byte-identical supported files. Canonical selection is deterministic by
normalized relative-path ordering. The existing path-derived `document_id` is
preserved for the canonical source. All physical files remain visible in Data
Inspection, with canonical/duplicate/unsupported markers and duplicate alias
metadata. Citations use the canonical source and retain aliases for audit.

The real local Granite ONNX rebuild used a fixture with two identical files and
one different file:

| Metric | Result |
|---|---:|
| Physical supported files | 3 |
| Unique ingested documents | 2 |
| Chunks | 2 |
| Duplicate files | 1 |
| Vector documents | 2 |
| Retrieved copies of duplicated evidence | 1 |

Inspection retained all three physical paths. The canonical row carried the
duplicate alias; the duplicate row pointed to the canonical path; the distinct
file remained a second canonical document. Retrieval and bibliography
provenance used the canonical path. Fresh-generation re-indexing produced the
same canonical selection and counts. The fixture source-manifest hash was
`29d94eca3b43e35a8f5e765e30e3fa80b8186978fd8b91fbfa00994cb305a1ac`.

## Final source matrix

| Check | Result |
|---|---|
| Python compilation | Pass, isolated bytecode root |
| Alembic upgrade/head/drift | Pass; head `202609170001`, no drift |
| Ruff | Pass |
| Pyright | Pass; 0 errors, 0 warnings, 0 informations |
| Backend unit suite | **838 passed**, 7 warnings |
| SQLite persistence | **15 passed**, 14 PostgreSQL cases intentionally skipped without `TEST_DATABASE_URL` |
| Angular/Vitest | **24 files / 105 tests passed** |
| Angular production build | Pass |
| pip-audit | No known vulnerabilities |
| npm audits | 0 vulnerabilities after lockfile-only transitive remediation |
| cargo-audit | Pass; six allowed unmaintained-crate warnings, no denied unsound finding |
| Canonical Full browser suite | **53 passed, 0 failures, 0 errors, 0 skips** |

The initial empty custom SQLite unit attempt was setup-invalid and produced
missing-table cascades; the CI-style default-database rerun above is the
authoritative result. The npm lockfile remediation changed transitive versions
only. The Rust build used the repository-required pinned `1.95.0` toolchain.

## Packaged Windows validation

Build command:

```powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.4.0 -DesktopTarget All -AllowDirtyTree
```

The resulting artifacts were:

| Artifact | SHA-256 | Metadata |
|---|---|---|
| `release/DILIGENT-v3.4.0-windows-x64-portable.exe` | `29BC8E69CBBB0E8DC8446103C2D8526BF02130B724D9916C2DAE68901F2BA61A` | PE `0x8664` AMD64; 171,726,336 bytes |
| `release/DILIGENT-v3.4.0-windows-x64.msi` | `B8AA3E9734ECDE0FD2DD80573C463CEFD7674E6B9E0CEACDA395C19EAEC1A263` | Product `DILIGENT Clinical Copilot`, version `3.4.0`, UpgradeCode `{2CF8EF35-4160-59EB-89D8-01EC7D19A887}`; 170,156,032 bytes |

The checksum manifest matched both artifact hashes.

Portable smoke command:

```powershell
.\app\desktop\build\smoke_release.ps1 -Version 3.4.0 -DesktopTarget All -InstallMsi -TestUpgrade -PreviousMsiPath .\release\DILIGENT-v3.3.0-windows-x64.msi -PreviousChecksumPath .\release\DILIGENT-v3.3.0-windows-x64.sha256
```

The portable portion passed twice from one data root. It verified checksum,
AMD64/PE, first launch, runtime extraction and `extraction.complete`, a fresh
ready payload with `release_version=3.4.0`, `/api/health`, `/`, representative
packaged API boundaries, the process window title `DILIGENT Clinical Copilot`,
clean close, backend termination, port closure, and replacement of the stale
ready-file state on the immediate second launch.

The published 3.3.0 MSI checksum also matched:
`d2d169e8f25d983d9c25589e417fb88f032667f4bd6d386cf43cc50bf9a90065`.

### MSI host boundary

The 3.4.0 MSI metadata and checksum passed. Actual 3.3.0 install, 3.4.0
upgrade, launch, persistence-marker survival, and uninstall were not completed:
the current Windows token reported `IsAdministrator=False`, and the package
requires an elevated `ALLUSERS=1` installation. Two controlled `msiexec`
attempts remained idle and were stopped after confirmation. No DILIGENT process,
partial registration, installed executable, or listening packaged backend port
remained afterward. This evidence is **BLOCKED BY HOST PREREQUISITE**, not an
observed MSI defect.

## Explicitly deferred boundaries

- Narrator/Speech Recap and full component screen-reader certification are
  optional future enhancements, not required validation debt.
- `runtime.containerized` remains `NOT_IMPLEMENTED` and is outside scope unless
  container deployment becomes a requirement.
- Tagging, GitHub Release publication, code signing, offline WebView2 packaging,
  and true clean-machine certification remain separate publication/distribution
  procedures.
- A native CUA application inventory was unavailable on this host after reboot;
  the packaged smoke used the actual process window title and runtime/API
  evidence rather than claiming screenshot-level native UI certification.

## Cleanup

All task-owned DILIGENT/MSI processes were stopped after validation and no
packaged backend ports remained listening. Disposable browser, source-matrix,
RAG-fixture, and generated bytecode scratch directories were removed after
verification. Release artifacts remain because they are the validated 3.4.0
candidate outputs.
