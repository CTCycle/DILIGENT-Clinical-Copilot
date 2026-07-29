# DILIGENT Desktop Release
Last updated: 2026-07-29

## Build inputs

Desktop packaging is isolated under `app/desktop`. Angular is built from `app/client`, the backend is frozen with PyInstaller `6.21.0` in onedir/windowed mode, and the resulting runtime is archived deterministically before it is embedded by Tauri. Tauri CLI `2.11.4`, Tauri crate `2.11.5`, and Tauri Build `2.6.3` are locked.

The runtime allowlist includes the frozen backend, Angular browser output, settings templates, and tracked reference catalogs. It excludes source code, tests, documentation, credentials, `.env`, databases, logs, models, vectors, archives, documents, caches, and development runtimes.

## Artifacts

For version `3.1.0`, successful builds publish:

```text
releases/DILIGENT-v3.1.0-windows-x64-portable.exe
releases/DILIGENT-v3.1.0-windows-x64.msi
releases/DILIGENT-v3.1.0-windows-x64.sha256
```

The portable executable is a single distribution file. It extracts immutable runtime content to `%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>` and keeps settings, database, logs, models, source documents, vectors, exports, state, and access-key material in `%LOCALAPPDATA%\DILIGENT\data`.

## Cleanup

```powershell
.\start_on_windows.ps1 -Action RemoveDesktopRelease -Version 3.1.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
```

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI and do not touch development runtimes, settings, databases, or `%LOCALAPPDATA%\DILIGENT`.

## Validation boundaries

The build script validates the PowerShell contract, frozen-backend startup, ready-file response, health endpoint, and Angular index before publishing. Full Tauri compilation, portable UI smoke testing, MSI installation/upgrade/uninstall testing, WebView2 offline installation, and code-signing remain Windows host/distribution gates and must be reported separately when the required toolchain or installer environment is unavailable.
