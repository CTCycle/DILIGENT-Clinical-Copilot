# DILIGENT Desktop Release
Last updated: 2026-10-18

## Current release — v3.4.0 published 2026-10-18

Status: **PUBLISHED; exact candidate and downloaded release bytes verified.**
The public [v3.4.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0)
is GitHub's latest release. Tag `v3.4.0` resolves to
`d4efa5e0bccfa552c80a5edeff2168bfaa718639`, the source used by successful
prepare run [39109627873](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39109627873)
and successful publish run
[39125173765](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39125173765).
The workflow promoted the immutable candidate without rebuilding it.

| Published artifact | SHA-256 |
|---|---|
| DILIGENT-v3.4.0-windows-x64-portable.exe | 702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26 |
| DILIGENT-v3.4.0-windows-x64.msi | 3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d |

The downloaded EXE and MSI bytes match these published digests. The downloaded
portable EXE launched, rendered DILI Agent, returned `/api/health` status
`ok`, and closed cleanly. Hosted prepare evidence covers the MSI install,
upgrade, reinstall, and uninstall lifecycle. The user authorized the unsigned
distribution policy on 2026-10-18; the release notes disclose that neither
binary has a publisher signature. The clean-host WebView2 check was waived on
2026-10-18 because only one Windows machine was available. Native QA also
recorded a known follow-up: a manual report edit clears previously saved
metadata; it was documented and not changed in this candidate. See the
[release-gate report](../../QA/release-gates-20261018/report.md) and
[operator acceptance checklist](../../QA/release-gates-20261018/operator-checklist.md).

## Historical candidate preparation — 2026-10-01

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

Current preparation status is PORTABLE VALIDATED, INCLUDING THE POPULATED
PACKAGED WORKFLOW SLICE; MSI/UPGRADE PENDING ON SUITABLE TEST HOST;
DISTRIBUTION PENDING. The current workstation is not administrator-capable,
and the local EXE/MSI are NotSigned. The packaged portable interactive-UI and
populated-workflow slices are validated via native automation (workspaces,
Settings persistence, full multi-drug analysis, sessions/timeline/revisions,
RAG ingestion/retrieval, cancellation/fail-closed, access keys); a native MSI
lifecycle and clean-machine distribution remain pending on a suitable host.
See the current candidate evidence at
assets/QA/desktop-release-validation-20261001/report.md, the packaged-UI
slice at assets/QA/desktop-interactive-ui-validation-20261001/report.md, and
the populated-workflow slice at
assets/QA/desktop-populated-ui-validation-20261001/report.md.
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

## Historical candidate status — before release-gate implementation

As of 2026-10-17, before release-gate implementation, the source manifests
reported synchronized candidate version `3.4.0`.
Fresh portable, MSI, and checksum artifacts were built on 2026-10-17 from clean
`develop` commit `37624def8a0f3eba0dfb9990f536aea4b3c0e8a9`; the portable launch
and exact deep-path smokes passed, and MSI metadata and artifact hashes were
verified. A native portable Tauri shell/dialog smoke passed on 2026-10-18, but
the broader populated native workflows were not repeated, so full package UI
parity remains `PARTIAL`. MSI lifecycle, live-provider E2E, signing, branch
synchronization, and clean-machine distribution remain separate pending gates.
The pre-publication GitHub readback earlier on 2026-10-18 showed the
[latest release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/latest)
as `v3.3.0`. Live GitHub branch refs at that point were
`develop` = `37624def8a0f3eba0dfb9990f536aea4b3c0e8a9` and `main` =
`3e341c5a84def4805c6d4ddbf36b67a9071dad3a`, so the required synchronization
was still pending. The [`v3.4.0` tag lookup](https://api.github.com/repos/CTCycle/DILIGENT-Clinical-Copilot/git/ref/tags/v3.4.0)
returned 404 in that pre-publication check. This historical state predates the
published release recorded at the top of this document. See the
[current dependency and packaged candidate evidence](../../QA/security-dependency-refresh-20261017/report.md).
At the time of this historical snapshot, publication was unconfirmed. Those
readback criteria are now satisfied for v3.4.0, as recorded at the top of this
document.

### Portable browser-workflow comparison — 2026-10-17

The source backend at `37624def` was compared with the fresh portable frozen
backend in the in-app browser using the exact candidate Angular bundle (16/16
browser files matched). Against separate disposable data roots, shared
navigation/catalog states, Settings save/reload/reset, the empty-data
fail-closed preflight, and one identical synthetic amoxicillin assessment on
local `qwen3.5:9b` produced matching user-visible results; both session lists
showed `Successful`, and persisted report text had the same SHA-256. The output
remained indeterminate and required clinician review; its persisted LLM-QA and
clinical-review states were `not_run` and `not_reviewed`.

This establishes the common browser surface and packaged backend workflow, not
the complete Tauri shell. The available UI automation exposed browser tabs but
no native-window inventory, so native shell interactions and file dialogs stay
unvalidated; overall package UI parity is `PARTIAL`. Provider-dependent cloud
checks remain pending because no provider credentials were available.

### Native portable Tauri smoke — 2026-10-18

The current v3.4.0 portable candidate was launched in its native Tauri window
with an isolated temporary data root. The four primary workspace areas
rendered. A Settings value survived navigation and clean restart, and resetting
defaults persisted across restart. The native RAG folder dialog selected an
empty disposable QA directory; the patient-image dialog opened and was
cancelled without selecting a file. An intentionally short synthetic DILI
Agent draft was blocked by native preflight (5 blockers, 1 warning); the
Clinical Sessions view remained empty, confirming no session was created. The
missing visit date appeared twice with the same title. The app closed cleanly
with no remaining package process or listener on the observed runtime ports.

This validates current-candidate shell startup, basic navigation, a Settings
persistence path, native dialog open/select/cancel behavior, and shutdown. It
does not repeat populated analysis, session/revision, or RAG ingestion flows;
overall package UI parity remains `PARTIAL`. The duplicate date blocker is a
small preflight UX cleanup item. Screenshots and detailed observations are in
the [current candidate report](../../QA/security-dependency-refresh-20261017/report.md).

The MSI was inspected without installation or elevation. Product metadata,
signature state, embedded CAB, and file-table version/size were read; the
extracted MSI payload carries the same runtime archive digest as the portable
candidate. The extracted shell PE has a different SHA-256 from the portable EXE,
so no shell-binary identity is claimed. MSI install/upgrade/uninstall/reinstall
remain admin-dependent; signing and distribution remain separate pending gates.
The detailed evidence and limitations are in the [current candidate report](../../QA/security-dependency-refresh-20261017/report.md).

The expected output names for a verified `3.4.0` build are:

```text
release/DILIGENT-v3.4.0-windows-x64-portable.exe
release/DILIGENT-v3.4.0-windows-x64.msi
release/DILIGENT-v3.4.0-windows-x64.sha256
```

The portable executable is a single distribution file for no-install use. The MSI installs the same Tauri shell and packaged runtime. The `.sha256` file contains one SHA-256 entry per built artifact and must be checked before distribution. Publication requires separate tag, remote-release, and download/hash evidence.

### Desktop validation history

The local 2026-09-30 validation-closure build produced the first 3.4.0
candidate; that portable PASS was later superseded when a re-run of the
pre-existing artifact failed the required native window-title assertion
(observed title `io.github.ctcycle.diligent-siw`). The stale artifacts were
retained under `assets/QA/desktop-release-validation-20260930/prebuild-stale-artifacts/`
(local-only, not tracked) and a clean pinned-toolchain rebuild completed at
candidate commit `ce5df0c65fb432a42b2be622789e11f5c5283a43` with the branded
title `DILIGENT Clinical Copilot`. The deep-path candidate produced a ready
payload and window but its backend did not answer loopback HTTP; the
traceback identified a Windows filename-length failure while importing the
ONNX Runtime extension. The [deep-path remediation evidence](../../QA/desktop-release-validation-20260930/deep-path-remediation.md)
records the compact-layout correction, a clean rebuild at repair commit
`2fadf0f22410665a814efc24b316cf68753e1cd6`, and two successful exact-boundary
launches at the formerly failing path lengths.

The published remote `v3.3.0` MSI and checksum were downloaded and verified,
and the workflow/static upgrade contracts passed. MSI installation, upgrade,
uninstall, database-preservation comparisons, verbose logs, and reinstall
remain **BLOCKED** by the non-administrator token. Fresh local EXE and MSI
signatures are `NotSigned`, so Distribution is **BLOCKED** and the candidate
is not distribution-ready.

Evidence: the `desktop-release-validation-20260930/report.md` is local-only
(not tracked); the tracked [D01–D14 matrix](../../QA/desktop-release-validation-20260930/D01-D14-matrix.md)
and [packaged-suite-boundary](../../QA/desktop-release-validation-20260930/packaged-suite-boundary.md)
are in git.

### Packaged-UI interactive validation — 2026-10-01

The previously BLOCKED packaged interactive-UI lane was exercised by driving
the packaged Tauri/WebView2 window with the available native Windows UI
automation against an isolated data root. All four workspaces rendered;
Settings → General polling interval save/reload/reset and restart
persistence were proven (DB-verified); Settings → Models rendered the catalog,
roles, provider keys, and current configuration; the DILI Agent preflight
failed closed on the fresh packaged data; and clean close/relaunch left no
processes or listeners.

The populated-workflow continuation
([2026-10-01 populated-workflow report](../../QA/desktop-populated-ui-validation-20261001/report.md))
then certified the full synthetic multi-drug analysis (15 steps to 100%,
rendered report with R-score and RUCAM evidence, persisted session/versions),
session CRUD/lineage with manual-edit audit and revision fail-closed, timeline
generation with fail-closed fallback, RAG ingestion/retrieval with citations
and byte-identical dedup, cooperative cancellation, provider-timeout
fail-closed paths, and the access-key lifecycle — all in the packaged window
against a populated isolated data root (catalogs seeded via the repository's
own update-persistence APIs after the packaged Update All confirmed the real
ordered pipeline and was cooperatively cancelled). The native RAG folder
dialog could not be reliably automated and remains an interaction boundary.
MSI lifecycle, signing, and clean-machine distribution remain host/procedure
boundaries. See the [packaged-UI validation report](../../QA/desktop-interactive-ui-validation-20261001/report.md)
and the updated [D01–D14 matrix](../../QA/desktop-release-validation-20260930/D01-D14-matrix.md).

### Historical v3.4.0 release-candidate gate — 2026-10-01

Status: **CURRENT-SOURCE CANDIDATE BUILT; PORTABLE, DEEP-PATH, AND NATIVE SHELL SMOKE VALIDATED; FULL NATIVE UI PARITY PARTIAL; MSI LIFECYCLE, LIVE-PROVIDER E2E, SIGNING, BRANCH SYNCHRONIZATION, AND DISTRIBUTION PENDING**. The source and lock manifests are synchronized to strict SemVer `3.4.0`; the fresh local candidate remains unpublished until all remaining independent gates have current evidence:

- hosted CI passed dependency, backend, persistence, Windows regression, and
  documentation-push checks; the push-triggered live-provider job was skipped;
- the complete browser E2E lane still needs explicit live-provider evidence
  using synthetic data and the approved OpenCode Go / `deepseek-v4-flash`
  configuration;
- the fresh portable EXE passed two launches and two exact deep-path launches;
  a current native shell/dialog smoke also passed; populated native workflows
  remain unverified on this candidate. MSI metadata and checksums passed, while
  MSI install, launch, upgrade, uninstall, and reinstall remain pending on an
  administrator-capable host; no registered DILIGENT entry was found in the
  standard current-user or machine Add/Remove Programs paths, so this host has
  no prior MSI install available as an upgrade baseline;
- both candidate artifacts are unsigned; no signing certificate or
  `signtool.exe` is available on this host. Signing must be completed and
  verified for the release artifacts;
- Python, npm, and Cargo dependency scans passed with no unresolved
  release-blocking findings;
- GitHub's `develop` and `main` endpoints returned
  `37624def8a0f3eba0dfb9990f536aea4b3c0e8a9` and
  `3e341c5a84def4805c6d4ddbf36b67a9071dad3a`, respectively; branch
  synchronization is required before an annotated `v3.4.0` tag is created.

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

The synthetic live-flow evidence and the complete end-to-end record for that
historical release audit are consolidated in the [validation
ledger](validation_ledger.md) and the [project status
ledger](../project_status_ledger.md). That historical evidence does not certify
the current development source. Portable-EXE and MSI host smoke tests must
still be repeated against the final tagged commit for each release.

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

Final desktop artifacts are written directly to `release/`. Intermediate desktop staging remains under `assets/QA/desktop-release-staging/` and historical validation output under `assets/QA/release-audit-20260826/` (both local-only; `assets/QA/**` is git-ignored except for the tracked evidence reports); the release-only native Cargo output is kept under `runtimes/cache/cargo/target/x86_64-pc-windows-msvc/release/`.

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

`.github/workflows/release.yml` is manually dispatched from synchronized
`main`/`develop` with `mode=prepare` or `mode=publish` and a strict source-matching
version. Pushing a tag does not publish a release.

Preparation requires backend quality/migration/unit gates, SQLite/PostgreSQL
persistence, frontend tests/build, the canonical `Full` browser harness, the
explicit `LiveProvider` lane, and dependency security audits. It builds once using the approved unsigned distribution policy and generates
final checksums. Receipts and artifact checks must explicitly agree on
`signing=unsigned` / `NotSigned`; no certificate or signing secret is required.
Windows smoke
verifies portable/deep-path launches and real install/upgrade/uninstall/reinstall,
including installed-shell unsigned status and populated-data preservation. The
checksum-verified published v3.3.0 MSI supplies the baseline; fixture data is
written using v3.3.0 repositories and read again by the baseline application.
The baseline `runtime` directory and current `rt` directory are explicitly
accounted for. Receipts state which checks actually ran.

The existing `OPENCODE_GO_API_KEY` repository secret is required for provider
E2E. Keep it in GitHub's secret store; never copy it into QA evidence.

The candidate artifact `desktop-candidate-<run-id>-<attempt>` retains the unsigned
EXE/MSI, checksum manifest, `candidate.json`, and Windows lifecycle receipts.
Its 30-day retention is an operational limit: an expired candidate must be
prepared and accepted again. QA diagnostics remain under `assets/QA/` and are
uploaded as workflow artifacts; private keys, `.env`, and user databases are
not candidate assets.

Complete the linked operator checklist against those hashes before publication.
The `desktop-release` environment must require a reviewer, disable administrator
bypass, and allow only `main`. Its approval attests to populated native Tauri
workflows and clean Windows 11 x64/online-WebView2/standard-user acceptance.
This session configured CTCycle as the reviewer. Automated smoke never supplies
that human attestation.

Dispatch `publish` with `candidate_run_id` and a completed `operator_evidence`
reference including candidate hashes. Publication checks the exact source SHA,
successful prepare run/attempt, every required job, immutable candidate receipt,
artifact digests, explicit unsigned status, branch equality, and absence
of both tag and release. It downloads the existing candidate without rebuilding.
After reviewer approval, it creates the tag and a draft release, uploads the
unchanged assets, downloads them again for hash/unsigned-status and portable-launch
verification, then publishes and verifies external release/tag/latest metadata.
API errors never count as tag/release absence. A failed draft remains unpublished
for investigation; existing tags/releases are never silently overwritten.

Only after publication should final detailed QA documentation be committed and
branches synchronized again. Keep the release tag fixed at the tested source SHA.

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
  upgrade, launch, uninstall, and reinstall checks on a Windows host.

After publishing, perform a Windows host smoke test by opening the portable EXE, confirming a window titled `DILIGENT Clinical Copilot`, checking `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`, and requesting the port recorded there at `/api/health`. MSI install, upgrade, uninstall, WebView2 offline installation, code signing, and clean-machine testing remain separate distribution gates.

## Cleanup

```powershell
.\start_on_windows.ps1 -Action RemoveDesktopRelease -Version 3.4.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
```

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI, stop a running desktop application, or touch development runtimes, settings, databases, or `%LOCALAPPDATA%\DILIGENT`.
