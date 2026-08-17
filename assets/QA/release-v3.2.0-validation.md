# DILIGENT Clinical Copilot v3.2.0 release validation

Last updated: 2026-08-17

This record covers the validated `main` release candidate before creating the
annotated `v3.2.0` tag. Release binaries remain local under `release/` and are
not tracked in Git.

| Check | Result | Evidence |
| --- | --- | --- |
| Develop focused tests | PASS | 29 tests passed across timeline, routed-gateway, and data-inspection coverage. |
| Develop backend unit suite | PASS | 628 tests passed. |
| Backend quality | PASS | Configured Ruff, Pyright, and backend/test compilation passed locally. |
| Frontend production build | PASS | Angular production build passed; existing SCSS budget warnings remain for the model-configuration and clinical-sessions pages. |
| Remote develop CI | PASS | GitHub Actions run `32036505218` passed backend quality, persistence-contract, and Windows regression jobs for `c06edaef`. |
| Version consistency | PASS | Frontend packages, backend metadata, Cargo metadata, and Tauri configuration report `3.2.0`; Cargo metadata resolves `diligent-desktop@3.2.0`. |
| Frozen backend smoke | PASS | The release launcher produced a deterministic runtime archive with 2,344 files; `/api/health`, `/`, and `/clinical-sessions` returned HTTP 200. |
| Tauri production build | PASS | `pwsh -NoProfile -File .\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.2.0 -DesktopTarget All -Force` completed successfully. |
| Portable artifact | PASS | `release/DILIGENT-v3.2.0-windows-x64-portable.exe`, 190,399,488 bytes; SHA-256 `6206edcc87a1dfb9942234652deef16a02027e153546df61a45fab43f60876ec`. |
| MSI artifact | PASS | `release/DILIGENT-v3.2.0-windows-x64.msi`, 184,250,368 bytes; SHA-256 `e7b973095f3d0460427c47b2df2ac2fb90526822b674ee7b6117ef7665f6b429`. |
| Portable package smoke | PASS | Fresh portable launch reported `release_version: 3.2.0`; `/api/health` returned HTTP 200 on port `52830`. Smoke-test processes were stopped afterward. |
| Git hygiene | PASS | Tauri schema residue was removed after validation; no release binaries, staging output, or generated schemas are tracked. |

The tag-triggered `.github/workflows/release.yml` workflow rebuilds the tagged
revision on Windows, creates the matching GitHub Release with generated notes,
and attaches only the portable EXE and MSI. GitHub's standard source ZIP and
tarball correspond to the same tag. Code signing, clean-machine MSI
install/upgrade/uninstall, offline WebView2, and exhaustive clinical/provider
QA remain separate distribution gates.
