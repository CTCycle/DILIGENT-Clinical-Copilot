# DILIGENT Desktop Release
Last updated: 2026-08-02

## Packaging architecture

The Windows desktop release is a Tauri 2 shell under `app/desktop/src-tauri`.
The build pipeline performs these steps:

1. Build the Angular production bundle from `app/client`.
2. Freeze `app/server/desktop_entry.py` and its backend dependencies with PyInstaller `6.21.0` in onedir/windowed mode.
3. Run the frozen backend against an isolated SQLite data root and verify its ready-file contract, `/api/health`, Angular index, and SPA fallback.
4. Copy only the allowlisted backend, Angular browser output, settings templates, and reference catalogs into a deterministic runtime archive.
5. Embed that archive into the Tauri executable and optionally produce the MSI.

The locked desktop toolchain is Tauri CLI `2.11.4`, Tauri crate `2.11.5`, Tauri Build `2.6.3`, and the committed `app/desktop/src-tauri/Cargo.lock`.

The runtime allowlist excludes source code, tests, documentation, credentials, `.env`, databases, logs, models, vectors, archives, documents, caches, and development runtimes. The packaged shell uses the system WebView2 runtime; the MSI can instead carry an offline WebView2 installer when built with `-OfflineWebView2`.

## Current source and artifact status

The current source manifests report version `3.0.0`, but the latest local tag is
`v2.4.0` and the checked-in `release/` directory contains the previous `v2.4.0`
portable EXE, MSI, and checksum manifest. A `v3.0.0` build is an available build
target, not evidence of publication.

The expected output names for a verified `3.0.0` build are:

```text
release/DILIGENT-v3.0.0-windows-x64-portable.exe
release/DILIGENT-v3.0.0-windows-x64.msi
release/DILIGENT-v3.0.0-windows-x64.sha256
```

The portable executable is a single distribution file for no-install use. The MSI installs the same Tauri shell and packaged runtime. The `.sha256` file contains one SHA-256 entry per built artifact and must be checked before distribution. Publication requires separate tag, remote-release, and download/hash evidence.

## Runtime and data layout

At launch, Tauri verifies the embedded archive digest and extracts immutable content to:

```text
%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>
```

The shell starts `backend\DILIGENTBackend.exe` on a random localhost port, waits for `state\desktop-backend-ready.json` and `/api/health`, then shows the desktop window. The backend is attached to a Windows Job Object and is stopped when the shell exits.

Mutable user data is kept outside the extracted runtime:

```text
%LOCALAPPDATA%\DILIGENT\data\settings
%LOCALAPPDATA%\DILIGENT\data\resources\database.db
%LOCALAPPDATA%\DILIGENT\data\resources\logs\desktop-backend.log
%LOCALAPPDATA%\DILIGENT\data\resources\models
%LOCALAPPDATA%\DILIGENT\data\resources\sources
%LOCALAPPDATA%\DILIGENT\data\state
```

Artifact cleanup and MSI uninstall do not remove this user data. The extracted runtime is versioned and hash-addressed, so a new payload can coexist during an upgrade.

## Build

Run on a Windows x64 host with Rust/Cargo, the Windows build toolchain, the pinned portable runtimes, and network access for dependencies and the default WebView2 bootstrapper:

```powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.0.0 -DesktopTarget All
```

Use `-DesktopTarget Portable` or `-DesktopTarget Msi` for one artifact. Release builds require a clean worktree by default; use `-AllowDirtyTree` only when the dirty state is intentional and recorded. `-OfflineWebView2` is valid only with `-DesktopTarget Msi` or `All` and changes the MSI WebView2 installation mode.

Final desktop artifacts are written directly to `release/`. Intermediate desktop staging remains under `release/.staging/`; Tauri's native Cargo output remains under `app/desktop/src-tauri/target/release/`.

The build refuses to complete if the frozen backend, runtime manifest, artifact size, or MSI metadata checks fail. The portable artifact is the raw Tauri release executable copied to `release/` after those checks; remote publication is a separate maintainer action.

## Validation

The launcher validates:

- PowerShell parameter and host contracts;
- Angular production output;
- frozen backend startup, ready-file contents, `/api/health`, `/`, and `/clinical-sessions`;
- deterministic runtime archive manifest and digest;
- Tauri compilation;
- portable executable size, MSI metadata, and SHA-256 entries.

After publishing, perform a Windows host smoke test by opening the portable EXE, confirming a window titled `DILIGENT Clinical Copilot`, checking `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`, and requesting the port recorded there at `/api/health`. MSI install, upgrade, uninstall, WebView2 offline installation, code signing, and clean-machine testing remain separate distribution gates.

## Cleanup

```powershell
.\start_on_windows.ps1 -Action RemoveDesktopRelease -Version 3.0.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
```

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI, stop a running desktop application, or touch development runtimes, settings, databases, or `%LOCALAPPDATA%\DILIGENT`.
