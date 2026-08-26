# DILIGENT Desktop Release
Last updated: 2026-08-21

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

The current source manifests report version `3.3.0`. The `v3.3.0` desktop release
is created from the synchronized `main` branch. Publication is confirmed only
after the portable EXE and MSI, annotated tag, remote-release metadata, and
downloaded hash evidence have all been verified.

The expected output names for a verified `3.3.0` build are:

```text
release/DILIGENT-v3.3.0-windows-x64-portable.exe
release/DILIGENT-v3.3.0-windows-x64.msi
release/DILIGENT-v3.3.0-windows-x64.sha256
```

The portable executable is a single distribution file for no-install use. The MSI installs the same Tauri shell and packaged runtime. The `.sha256` file contains one SHA-256 entry per built artifact and must be checked before distribution. Publication requires separate tag, remote-release, and download/hash evidence.

The GitHub release attaches the portable EXE, MSI, and `.sha256` manifest. Existing remote assets are never replaced unless the local and remote bytes are identical.

## Runtime and data layout

At launch, Tauri verifies the embedded archive digest and extracts immutable content to:

```text
%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>
```

The shell shows a loading window immediately, extracts the runtime and starts `backend\DILIGENTBackend.exe` on a random localhost port off the UI-critical path, waits for `state\desktop-backend-ready.json` and `/api/health`, then navigates to the authenticated local interface. The backend is attached to a Windows Job Object and is asked to shut down cooperatively when the shell exits, with a bounded hard-kill fallback.

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

Run on a Windows x64 host with Rust 1.95.0/Cargo, the Windows build toolchain, the pinned portable runtimes, and network access for dependencies and the default WebView2 bootstrapper. The launcher pins Python 3.14.2, Node.js 22.13.0, uv 0.11.30, and PyInstaller 6.21.0; downloaded Python, Node, and uv archives are SHA-256 checked before extraction:

```powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.3.0 -DesktopTarget All -Force
```

Use `-DesktopTarget Portable` or `-DesktopTarget Msi` for one artifact. Release builds require a clean worktree by default; use `-AllowDirtyTree` only when the dirty state is intentional and recorded. `-OfflineWebView2` is valid only with `-DesktopTarget Msi` or `All` and changes the MSI WebView2 installation mode.

Final desktop artifacts are written directly to `release/`. Intermediate desktop staging remains under `assets/QA/desktop-release-staging/`, with validation output under `assets/QA/release-audit-20260826/`; the release-only native Cargo output is kept under `assets/QA/desktop-cargo-target/x86_64-pc-windows-msvc/release/`.

### Interactive artifact menu

Run `.\start_on_windows.ps1` and choose `12. Create release artifacts` to open the artifact submenu. It can build the portable executable, build the MSI installer, refresh the SHA-256 manifest from existing artifacts, or build all distribution artifacts. The portable and MSI choices run the same full desktop validation pipeline as the corresponding `-DesktopTarget Portable` or `-DesktopTarget Msi` command-line actions.

Choose `13. Remove release artifacts` to open the cleanup submenu. It can remove the portable executable, MSI installer, or checksum manifest for a selected version; remove all three artifacts for one version; or remove all versions. Removing one binary synchronizes the remaining checksum manifest. Removing all artifacts also clears generated desktop build state while preserving the tracked `app/desktop/src-tauri/generated/.gitkeep` placeholder.

The build refuses to complete if the frozen backend, runtime manifest, artifact size, or MSI metadata checks fail. The portable artifact is the raw Tauri release executable copied to `release/` after those checks; remote publication is a separate maintainer action.

## GitHub Actions publication

`.github/workflows/release.yml` runs on a `vX.Y.Z` tag. It builds both Windows desktop targets from that tagged commit, creates or updates the matching GitHub Release, and attaches the portable EXE, MSI, and `.sha256` manifest. It refuses to overwrite an existing non-identical asset. Create the tag only after `develop` and `main` have been synchronized and local release validation has passed.

The workflow uses the launcher's pinned portable Python runtime rather than installing a second host Python. The launcher removes inherited Git/MinGW, AWS CLI, hosted-Python, and other PATH entries that can provide competing `libffi` or `_ctypes` native files immediately before starting the release venv. The PyInstaller bootstrap removes hosted-toolcache Python entries from `sys.path`, registers only the pinned venv and `runtimes/python` native directories, explicitly loads the matching `libffi-8.dll` by absolute path through the CFFI bridge, preloads the supported CFFI native backend, and only then imports `ctypes`. Together these keep PyInstaller on the same embedded-Python and DLL set used by the release launcher, even when a hosted runner reconstructs PATH between workflow steps.

## Validation

The launcher validates:

- PowerShell parameter and host contracts;
- Angular production output;
- frozen backend startup, first-run Alembic head, ready-file contents,
  `/api/health`, `/`, and `/clinical-sessions`;
- deterministic runtime archive manifest and digest;
- Tauri compilation;
- portable executable size, MSI metadata, and SHA-256 entries.

After publishing, perform a Windows host smoke test by opening the portable EXE, confirming a window titled `DILIGENT Clinical Copilot`, checking `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`, and requesting the port recorded there at `/api/health`. MSI install, upgrade, uninstall, WebView2 offline installation, code signing, and clean-machine testing remain separate distribution gates.

## Cleanup

```powershell
.\start_on_windows.ps1 -Action RemoveDesktopRelease -Version 3.3.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
```

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI, stop a running desktop application, or touch development runtimes, settings, databases, or `%LOCALAPPDATA%\DILIGENT`.
