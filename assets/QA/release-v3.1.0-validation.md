# DILIGENT Clinical Copilot v3.1.0 release validation

Date: 2026-08-13
Release candidate: synchronized `main` and `develop` after the final packaging fix
Previous release: `v3.0.0` at `d60ddf2d`

## Release delta reviewed

The review covered the full `v3.0.0..HEAD` delta, with emphasis on the latest feature commits `f7c4b12a` and `45b0ca32`:

- Clinical-session metadata, review, preflight, timeline, and report-generation paths.
- Removal of the obsolete synchronous timeline-generation API and the associated backend/frontend contract cleanup.
- Data Inspection/catalog rendering and status behavior.
- Model configuration persistence and cloud/local model availability handling.
- Portable-runtime initialization and the Windows release launcher.
- Related API contracts, unit tests, E2E flows, documentation, and the new tag-driven GitHub release workflow.

## Validation performed

| Area | Result | Evidence |
| --- | --- | --- |
| Focused backend tests | PASS | 107 passed, 1 warning in 38.19s using the release basetemp. |
| Direct model-config and app-flow E2E | PASS | 23 passed, 3 skipped, 1 warning in 20.36s. The skips require persisted inspection sessions and are not applicable to the disposable SQLite database. |
| Browser UI smoke | PASS | In-app Browser loaded the app and exercised Clinical Sessions, Data Inspection, and Configurations. No browser console errors or warnings were reported. |
| Development runtime | PASS | Backend/frontend startup succeeded; health, navigation, API loading, persistence initialization, and affected UI surfaces were exercised. |
| Release build | PASS | `.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.1.0 -DesktopTarget All -Force` produced both Windows artifacts from the release configuration after the launcher was corrected to invoke PyInstaller through the release interpreter. |
| Portable package smoke | PASS | `DILIGENT-v3.1.0-windows-x64-portable.exe` started; its bundled backend reported `release_version: 3.1.0`, `/api/health` returned HTTP 200 with `{"status":"ok"}`, `/` returned HTTP 200, WebView2 loaded the frontend/assets, and SQLite initialization completed. |
| MSI artifact | PASS | `DILIGENT-v3.1.0-windows-x64.msi` was generated and its SHA-256 matched the release manifest. |
| Artifact checksums | PASS | Portable: `df12b8739e318561af3d2a3cf37ff5e467da8d38ca7034968c47e07e5329c713`; MSI: `c1e289fdbd22b0778c124e1ba08a60b4711e00a8326976b73459eb7144f41008`. |
| Release workflow | PASS | `.github/workflows/release.yml` validates `vX.Y.Z` tags, builds both desktop targets, verifies artifacts, and creates/updates the matching GitHub Release with the EXE and MSI attached. |

The repository runner's assertions passed for the model-config slice (33 unit tests and 8 selected E2E tests), but the Windows wrapper did not terminate cleanly in the managed console after its child processes exited. This is recorded as an environment-limited runner cleanup issue; the same affected tests passed directly in fresh processes, and the exact task-owned listeners were stopped afterward.

The first GitHub Actions release attempt reached the build step but failed because the `pyinstaller.exe` entrypoint could not import `pywin32-ctypes` on the hosted Python 3.14 environment. The launcher now invokes `python -m PyInstaller`; the clean local release rebuild passed, including the frozen-backend smoke test and Tauri MSI packaging. The hosted workflow must run again from the corrected synchronized tag.

The successful release build emitted the existing Angular component-style budget warnings and PyInstaller warnings for optional test/quantization modules that are not part of the runtime path. They did not prevent the frozen backend smoke test or packaged application startup.

The packaged backend log contained only the expected warning that local Ollama was unavailable in the smoke environment. No `ERROR`, `CRITICAL`, traceback, startup failure, missing asset, or failed health request was observed.

## Release artifacts

- `release/DILIGENT-v3.1.0-windows-x64-portable.exe` — 190,400,000 bytes
- `release/DILIGENT-v3.1.0-windows-x64.msi` — 184,168,448 bytes
- `release/DILIGENT-v3.1.0-windows-x64.sha256`

## Deferred gates

Code signing, clean-machine MSI installation and uninstall, upgrade testing, offline WebView2 behavior, and enterprise deployment validation remain outside this rapid release smoke test and are not release blockers under the project's existing release guidance.
