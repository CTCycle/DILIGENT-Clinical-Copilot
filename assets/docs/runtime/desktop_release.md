# DILIGENT Desktop Release
Last updated: 2026-10-01

## Exact candidate preparation — 2026-10-01

The prepared product candidate is
2f804d251769f4e01375d02459e41dd9970ec8eb on develop, with origin/develop at
the same SHA. Fresh local artifacts are present under release/:

| Artifact | SHA-256 |
|---|---|
| DILIGENT-v3.4.0-windows-x64-portable.exe | 0209cf25a8acf3a39762ffeec48488c28103e1b96c7c5a57ed8beb813574e167 |
| DILIGENT-v3.4.0-windows-x64.msi | 1df49329361b600edb2e0c07949fe00bc113b25122d55af9ba93cb52f67fd508 |

The portable smoke and exact 230-character executable / 107-character
LOCALAPPDATA deep-path replay passed twice. Exact-SHA CI run 36786345174 is
green for backend quality, PostgreSQL persistence, Windows regression, and
security. The live-provider job was skipped on push, then passed in exact-SHA
workflow-dispatch run [36828262326](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36828262326)
using the synthetic `opencode_go` / `deepseek-v4-flash` route.

Current preparation status is PORTABLE VALIDATED; MSI/UPGRADE PENDING ON
SUITABLE TEST HOST; DISTRIBUTION PENDING. The current workstation is not
administrator-capable, native Computer Use exposes no desktop app/window, and
the local EXE/MSI are NotSigned. See the current candidate evidence at
assets/QA/desktop-release-validation-20261001/report.md.
This preparation record does not create a tag, GitHub release, upload, or
distribution publication.

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

### Local 2026-09-30 validation-closure result

The local pinned-toolchain build produced the 3.4.0 candidate. The portable
EXE passed checksum, PE/AMD64 validation, first launch, runtime extraction,
fresh `desktop-backend-ready.json`, `release_version=3.4.0`, `/api/health`,
root and representative packaged API probes, the actual process window title,
clean close, backend termination, port closure, and a second launch from the
same data root that replaced the stale-ready-file state.

| Artifact | SHA-256 |
|---|---|
| `DILIGENT-v3.4.0-windows-x64-portable.exe` | `29BC8E69CBBB0E8DC8446103C2D8526BF02130B724D9916C2DAE68901F2BA61A` |
| `DILIGENT-v3.4.0-windows-x64.msi` | `B8AA3E9734ECDE0FD2DD80573C463CEFD7674E6B9E0CEACDA395C19EAEC1A263` |

The MSI checksum, ProductName, ProductVersion, Manufacturer, and UpgradeCode
metadata passed. Actual MSI install, 3.3.0-to-3.4.0 upgrade, launch, and
uninstall are **blocked on this host** because the package is `ALLUSERS=1` and
the validation session is not an administrator. The controlled Windows
Installer attempts were stopped after remaining idle; no partial registration,
packaged process, or packaged port remained. Run the MSI procedure on an
administrator host before publication. This host limitation is not evidence of
an MSI product failure.

### Current desktop revalidation follow-up — 2026-09-30

The pre-existing portable artifact was re-run from the current checkout with
`smoke_release.ps1 -Version 3.4.0 -DesktopTarget Portable`. It reached the
packaged ready-file, health, representative API, and native-process checks but
failed the required title assertion: the observed title was
`io.github.ctcycle.diligent-siw`, not `DILIGENT Clinical Copilot`. The observed
portable SHA-256 was
`29bc8e69cbbb0e8dc8446103c2d8526bf02130b724d9916c2dae68901f2ba61a`.

The exact-toolchain rebuild was blocked by the host's protected canonical uv
cache and protected project-environment files; no replacement artifact was
produced. Treat the earlier portable PASS as superseded. The fresh-candidate
continuation below is the current desktop status.

### Fresh candidate rebuild and portable continuation — 2026-09-30

The stale artifacts were retained under
[`assets/QA/desktop-release-validation-20260930/prebuild-stale-artifacts/`](../../QA/desktop-release-validation-20260930/prebuild-stale-artifacts/)
and a clean pinned-toolchain build was completed at candidate commit
`ce5df0c65fb432a42b2be622789e11f5c5283a43`. The fresh portable SHA-256 is
`27edd72ea371a4f9794e649108d379e4d4049d4e0aa4b838f374b66eea32924c`; the fresh
MSI SHA-256 is
`ac2c4cbe924d402900b37c39c12fc1b5f72615dd648e9e7a57bdcc859306466f`.

The required portable smoke passed twice with the branded native title
`DILIGENT Clinical Copilot`, HTTP health, authenticated-local API boundary,
clean backend/port shutdown, and stale-ready-file replacement. Additional
packaged-process checks passed warm restart, forced termination recovery,
concurrent launch handling, space paths, non-ASCII paths, and a short second
location. The original deep-path candidate produced a ready payload and window
but its backend did not answer loopback HTTP; the traceback identified a
Windows filename-length failure while importing the ONNX Runtime extension.
The [deep-path remediation evidence](../../QA/desktop-release-validation-20260930/deep-path-remediation.md)
records the compact-layout correction, a clean rebuild at repair commit
`2fadf0f22410665a814efc24b316cf68753e1cd6`, and two successful exact-boundary
launches at the formerly failing path lengths. Computer Use reported no native
app/window surface, so the full packaged UI workflow and failure suite remains
BLOCKED.

The published remote `v3.3.0` MSI and checksum were downloaded and verified,
and the workflow/static upgrade contracts passed. MSI installation, upgrade,
uninstall, database-preservation comparisons, verbose logs, and reinstall
remain **BLOCKED** by the non-administrator token. Fresh local EXE and MSI
signatures are `NotSigned`, so Distribution is **BLOCKED** and the candidate
is not distribution-ready.

Evidence: [desktop-release-validation-20260930](../../QA/desktop-release-validation-20260930/report.md),
including the [D01–D14 matrix](../../QA/desktop-release-validation-20260930/D01-D14-matrix.md).

### v3.4.0 release-candidate gate

Status: **PORTABLE VALIDATED; MSI/UPGRADE PENDING ON SUITABLE TEST HOST; DISTRIBUTION PENDING**. The source and lock manifests are synchronized to strict
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

This preparation record does not create a tag, publish a GitHub release, or
upload assets. The candidate remains on origin/develop without a v3.4.0 tag or
release; branch synchronization to main is intentionally deferred to the later
release session.

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
%LOCALAPPDATA%\DILIGENT\rt\<version>\<payload-sha256>
```

The shell shows a loading window immediately, extracts the runtime and starts `b\DILIGENTBackend.exe` on a random localhost port off the UI-critical path, waits for `state\desktop-backend-ready.json` and `/api/health`, then navigates to the authenticated local interface. The compact runtime directory names keep native-extension paths below Windows' DLL-loading boundary on deep user paths. The backend is attached to a Windows Job Object and is asked to shut down cooperatively when the shell exits, with a bounded hard-kill fallback.

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
