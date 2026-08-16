# DILIGENT Clinical Copilot v3.1.0 release validation

Date: 2026-08-16
Release candidate: final mainline release preparation after the hosted packaging bootstrap fix
Previous release: `v3.0.0` at `d60ddf2d`

## Release delta reviewed

The review covered the full `v3.0.0..HEAD` delta, with emphasis on the latest feature commits `f7c4b12a` and `45b0ca32`, plus the release-hardening fixes `d5b74767`, `f003958e`, `7fc8c4c0`, `9a298e14`, `d0f9d7f6`, the hosted-DLL ordering correction, and the current hosted PATH-precedence correction:

- Clinical-session metadata, review, preflight, timeline, and report-generation paths.
- Removal of the obsolete synchronous timeline-generation API and the associated backend/frontend contract cleanup.
- Data Inspection/catalog rendering and status behavior.
- Model configuration persistence and cloud/local model availability handling.
- Portable-runtime initialization and the Windows release launcher.
- Hosted-Windows PyInstaller compatibility and stale packaged-backend readiness-marker cleanup.
- Related API contracts, unit tests, E2E flows, documentation, and the new tag-driven GitHub release workflow.

## Validation performed

| Area | Result | Evidence |
| --- | --- | --- |
| Focused backend tests | PASS | 107 passed, 1 warning in 38.19s using the release basetemp. |
| Direct model-config and app-flow E2E | PASS | 23 passed, 3 skipped, 1 warning in 20.36s. The skips require persisted inspection sessions and are not applicable to the disposable SQLite database. |
| Browser UI smoke | PASS | In-app Browser loaded the app and exercised Clinical Sessions, Data Inspection, and Configurations. No browser console errors or warnings were reported. |
| Development runtime | PASS | Backend/frontend startup succeeded; health, navigation, API loading, persistence initialization, and affected UI surfaces were exercised. |
| Release build | PASS | `.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.1.0 -DesktopTarget All -Force` produced both Windows artifacts from the final release configuration after the PyInstaller bootstrap was corrected to prepend and register the release venv DLL directory. |
| Portable package smoke | PASS | `DILIGENT-v3.1.0-windows-x64-portable.exe` passed two launches from the final local rebuild, including a fresh stale readiness marker before each launch; the bundled backends reported PIDs `12452` and `7264`, ports `54221` and `53094`, and `release_version: 3.1.0`. `/api/health`, `/`, and `/clinical-sessions` returned HTTP 200 on both launches; WebView2 loaded the frontend/assets and SQLite startup completed. |
| MSI artifact | PASS | `DILIGENT-v3.1.0-windows-x64.msi` was generated at 184,250,368 bytes and its SHA-256 matched the release manifest. |
| Artifact checksums | PASS | Final local rebuild portable: `7127a3b4660e327943540f7e4aad3e3be754c1eb361ea1176e6b6f3b9ddcef27`; MSI: `2092df6cd46b42f5b5e63e4ee0d6f45ed7b6ca08ec43f05c6d779467c29eaa51`. |
| Release workflow | CONFIGURED | `.github/workflows/release.yml` validates `vX.Y.Z` tags, builds both desktop targets, verifies artifacts, and creates/updates the matching GitHub Release with the EXE and MSI attached. The tagged hosted run remains the publication gate for the native-DLL precedence fix. |

The repository runner's assertions passed for the model-config slice (33 unit tests and 8 selected E2E tests), but the Windows wrapper did not terminate cleanly in the managed console after its child processes exited. This is recorded as an environment-limited runner cleanup issue; the same affected tests passed directly in fresh processes, and the exact task-owned listeners were stopped afterward.

The first nine GitHub Actions release attempts reached the build step but failed on hosted-Windows Python isolation: the `pyinstaller.exe` entrypoint and then `python -m PyInstaller` could not import `pywin32-ctypes`, while the corrected wrapper then exposed a `_ctypes` DLL collision during PyInstaller's advisory administrator probe, dependency scanner, and bootstrap import. The wrapper now registers and prepends the release venv's DLL directory, eagerly loads `ctypes` before the supported CFFI backend, selects that compatibility path, and retains PyInstaller's working-directory safety guard without the nonessential ctypes privilege probe. The release workflow now removes any inherited PATH entry containing a competing Python, `libffi`, or `_ctypes` native file before invoking the pinned runtime. The final local release rebuild passed, including the frozen-backend smoke test and Tauri MSI packaging; the synchronized `v3.1.0` tag run is the hosted publication gate for this final builder correction.

During the final packaged smoke test, a stale `desktop-backend-ready.json` from a prior run reproduced a real restart failure: Tauri accepted the old PID before the new backend had published its contract. Commit `f003958e` removes the stale marker before spawning the backend. The rebuilt portable package then passed both the stale-marker first launch and the immediate restart check with fresh backend PIDs and HTTP 200 health/root responses.

The successful release build emitted the existing Angular component-style budget warnings and PyInstaller warnings for optional test/quantization modules that are not part of the runtime path. They did not prevent the frozen backend smoke test or packaged application startup.

The packaged backend log contained only the expected warning that local Ollama was unavailable in the smoke environment. No `ERROR`, `CRITICAL`, traceback, startup failure, missing asset, or failed health request was observed.

## Release artifacts

- `release/DILIGENT-v3.1.0-windows-x64-portable.exe` — 190,400,000 bytes
- `release/DILIGENT-v3.1.0-windows-x64.msi` — 184,250,368 bytes
- `release/DILIGENT-v3.1.0-windows-x64.sha256`

## Deferred gates

Code signing, clean-machine MSI installation and uninstall, upgrade testing, offline WebView2 behavior, and enterprise deployment validation remain outside this rapid release smoke test and are not release blockers under the project's existing release guidance.
