# DILIGENT Desktop Release
Last updated: 2026-09-13

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

The current source manifests report synchronized candidate version `3.4.0`. The
latest published desktop release remains `v3.3.0`; no `v3.4.0` tag or release is
created by the current local readiness work. Publication is confirmed only after
the portable EXE and MSI, annotated tag, remote-release metadata, and downloaded
hash evidence have all been verified.

The expected output names for a verified `3.4.0` build are:

```text
release/DILIGENT-v3.4.0-windows-x64-portable.exe
release/DILIGENT-v3.4.0-windows-x64.msi
release/DILIGENT-v3.4.0-windows-x64.sha256
```

The portable executable is a single distribution file for no-install use. The MSI installs the same Tauri shell and packaged runtime. The `.sha256` file contains one SHA-256 entry per built artifact and must be checked before distribution. Publication requires separate tag, remote-release, and download/hash evidence.

### v3.4.0 release-candidate gate

Status: **NOT READY**. The source and lock manifests are synchronized to strict
SemVer `3.4.0`, but the candidate must remain unpublished until all of these
independent gates have current evidence:

- hosted CI is green for the exact candidate commit;
- the complete browser E2E suite passes, including the explicit live provider
  flow using synthetic data and the approved OpenCode Go / `deepseek-v4-flash`
  configuration;
- the portable EXE launches and shuts down cleanly, and the MSI installs,
  launches, upgrades, and uninstalls on a Windows host;
- Python, npm, and Cargo dependency scans have been reviewed with no unresolved
  release-blocking findings;
- `develop` and `main` are intentionally synchronized before an annotated
  `v3.4.0` tag is created.

The local work in this session does not create a tag, publish a GitHub release,
upload assets, or push branches.

The GitHub release attaches the portable EXE, MSI, and `.sha256` manifest. Existing remote assets are never replaced unless the local and remote bytes are identical.

### Historical 2026-09-08 release audit

The latest published release is `v3.3.0`, published on 2026-09-01 from the
main-line release commit. Its GitHub release contains the portable EXE, MSI,
and SHA-256 manifest named above. The current workspace is post-release
development and has substantial functional, visual, and behavioral divergence;
it is not a byte-for-byte or behaviorally equivalent v3.3.0 build. The local
`release/` directory does not contain the published binaries, so this audit
used the live release metadata, tag ancestry, and the committed Tauri staging
manifest rather than claiming a local packaged-binary launch test.

The synthetic live-flow evidence is recorded in
[`assets/QA/e2e-validation-20260908.md`](../../QA/e2e-validation-20260908.md).
That historical evidence does not certify the current development source.
Portable-EXE and MSI host smoke tests must still be repeated against the final
tagged commit for each release.

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
%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log
%LOCALAPPDATA%\DILIGENT\data\cache\embeddings
%LOCALAPPDATA%\DILIGENT\data\resources\sources
%LOCALAPPDATA%\DILIGENT\data\state
```

Artifact cleanup and MSI uninstall do not remove this user data. The extracted runtime is versioned and hash-addressed, so a new payload can coexist during an upgrade.

## Native Windows dialogs

The Tauri desktop surface uses the native operating-system directory picker for
path-only RAG document folder selection. The dialog integration is centralized
in the Angular desktop-dialog service and is limited to the `dialog:allow-open`
capability for the authenticated localhost backend origin; it does not grant
filesystem or shell access.

Normal HTML file inputs remain in use for clinical-session image/document
metadata and patient profile images because those workflows require browser
`File` objects rather than filesystem paths. They open the native WebView2 file
picker on Windows. Non-Tauri web development uses the local backend's
server-side folder browser for RAG folder selection; it does not infer
an absolute path from browser `File` metadata. The browser folder browser is
directory-navigation only, and the backend returns the canonical path used by
vectorization. No filesystem or shell capability is granted to the Tauri
surface.

## Build

Run on a Windows x64 host with Rust 1.95.0/Cargo, the Windows build toolchain, the pinned portable runtimes, and network access for dependencies and the default WebView2 bootstrapper. The launcher pins Python 3.14.7, Node.js 22.13.0, uv 0.11.30, and PyInstaller 6.21.0; downloaded Python, Node, and uv archives are SHA-256 checked before extraction:

```powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.4.0 -DesktopTarget All
```

Use `-DesktopTarget Portable` or `-DesktopTarget Msi` for one artifact. Release builds require a clean worktree by default; use `-AllowDirtyTree` only when the dirty state is intentional and recorded. `-OfflineWebView2` is valid only with `-DesktopTarget Msi` or `All` and changes the MSI WebView2 installation mode.

Final desktop artifacts are written directly to `release/`. Intermediate desktop staging remains under `assets/QA/desktop-release-staging/`, with validation output under `assets/QA/release-audit-20260826/`; the release-only native Cargo output is kept under `runtimes/cache/cargo/target/x86_64-pc-windows-msvc/release/`.

### Interactive artifact menu

Run `.\start_on_windows.ps1` and choose `8. Create release artifacts` to open the artifact submenu. It can build the portable executable, build the MSI installer, refresh the SHA-256 manifest from existing artifacts, or build all distribution artifacts. The portable and MSI choices run the same full desktop validation pipeline as the corresponding `-DesktopTarget Portable` or `-DesktopTarget Msi` command-line actions.

Choose `9. Remove release artifacts` to open the cleanup submenu. It can remove the portable executable, MSI installer, or checksum manifest for a selected version; remove all three artifacts for one version; or remove all versions. Removing one binary synchronizes the remaining checksum manifest. Removing all artifacts also clears generated desktop build state while preserving the tracked `app/desktop/src-tauri/generated/.gitkeep` placeholder.

The build refuses to complete if the frozen backend, runtime manifest, artifact size, or MSI metadata checks fail. The portable artifact is the raw Tauri release executable copied to `release/` after those checks; remote publication is a separate maintainer action.

## Structured source updates

The Data Inspection "Update all" action starts one backend-owned job. Its
structured-source sequence is deliberately ordered:

1. RxNav refreshes and atomically reconciles the shared drug identity data.
2. LiverTox refreshes and atomically reconciles its source-owned aliases,
   monographs, and LiverTox identifiers.
3. DILIrank validates and replaces its complete FDA snapshot, then resolves
   links against the completed catalog state.

Cancellation or failure before a source's final commit preserves the previous
usable source snapshot. RAG rebuilding is a separate job and is not included in
the structured-source sequence. After a backend restart, a lost in-memory job
is surfaced as an interrupted/retryable update rather than remaining in a
permanent running state.

## GitHub Actions publication

`.github/workflows/release.yml` runs on a `vX.Y.Z` tag. Its release preflight
first requires the tagged commit to equal `origin/main`, then runs the backend
quality/migration/unit/persistence gates, frontend tests and production build,
the complete browser suite, the explicit live provider flow, dependency scans,
and Windows EXE/MSI host smoke checks. Packaging and publication depend on all
of those gates. The workflow builds both Windows desktop targets from the
verified tag, creates or updates the matching GitHub Release, and attaches the
portable EXE, MSI, and `.sha256` manifest. It refuses to overwrite an existing
non-identical asset. Create a new SemVer tag only after `develop` and `main`
have been intentionally synchronized and the local release validation has
passed.

The workflow uses the launcher's pinned portable Python runtime rather than installing a second host Python. The launcher clears inherited `PYTHONHOME`, `PYTHONPATH`, and user-site settings, then points `PYTHONHOME` at `runtimes/python` so the project venv uses the same embeddable interpreter family as the release runtime. The PyInstaller bootstrap removes hosted-toolcache Python entries from `sys.path`, registers only the pinned venv and `runtimes/python` native directories, explicitly loads the matching `libffi-8.dll` by absolute path through the CFFI bridge, preloads the supported CFFI native backend, and only then imports `ctypes`. Together these keep PyInstaller on the same embedded-Python and DLL set used by the release launcher without rewriting the host PATH.

## Validation

The launcher validates:

- PowerShell parameter and host contracts;
- Angular production output;
- frozen backend startup, first-run Alembic head, ready-file contents,
  `/api/health`, `/`, and `/clinical-sessions`;
- deterministic runtime archive manifest and digest;
- Tauri compilation;
- portable executable size, MSI metadata, and SHA-256 entries;
- `app/desktop/build/smoke_release.ps1` portable launch/close and MSI install,
  upgrade, launch, and uninstall checks on a Windows host.

After publishing, perform a Windows host smoke test by opening the portable EXE, confirming a window titled `DILIGENT Clinical Copilot`, checking `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`, and requesting the port recorded there at `/api/health`. MSI install, upgrade, uninstall, WebView2 offline installation, code signing, and clean-machine testing remain separate distribution gates.

## Cleanup

```powershell
.\start_on_windows.ps1 -Action RemoveDesktopRelease -Version 3.4.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
```

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI, stop a running desktop application, or touch development runtimes, settings, databases, or `%LOCALAPPDATA%\DILIGENT`.
