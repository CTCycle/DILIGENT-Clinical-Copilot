# Windows Desktop Release
Last updated: 2026-10-21

## Published release: v3.5.0

The public [v3.5.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.5.0) is the latest release. Tag v3.5.0 resolves to source commit `7a1b23e6f27b3d1987f1106eb2fb5f9aa40b2820`. Prepare run [39382912130](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39382912130) and protected, approved publish run [39419567722](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39419567722) succeeded.

| Published artifact | SHA-256 |
|---|---|
| DILIGENT-v3.5.0-windows-x64-portable.exe | 981fe5e3bf1db072f0e210de85c63b6baaa17be5f6a1027915124ac846435204 |
| DILIGENT-v3.5.0-windows-x64.msi | 4c6ff76d69693564b7889c26bec4784602324b404d539886ded5ab7716df8916 |
| DILIGENT-v3.5.0-windows-x64.sha256 | 8ee62e76c9f1e175a975f01dc46e4616899e4259d8cc314dde4554031286487e |

The publication workflow promoted the immutable prepare candidate without rebuilding. Its distribution verification succeeded, and the published asset digests match the candidate receipt. The candidate EXE opened DILI Agent and the other inspected pages on the local Windows 11 Pro host, returned `{"status":"ok"}` from `/api/health`, and closed cleanly. The hosted prepare run passed the required MSI lifecycle, portable/deep-path smoke, backend and browser gates, PostgreSQL persistence, dependency audits, and LiveProvider E2E.

The binaries are unsigned. The public release notes disclose the missing publisher signature and warn of a possible unknown-publisher or SmartScreen prompt.

The release owner explicitly waived a candidate-specific populated native analysis and clean-host Windows 11/WebView2 run, relying on the existing [project validation ledger](../project_status_ledger.md) for tested feature coverage. The v3.5.0 local smoke verifies package launch and health only; it is not a populated analysis or clean-host compatibility result. The approval comment records these waivers. Detailed release evidence is in the [v3.5.0 report](../../QA/release-gates-20261020/report.md). The previous [v3.4.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0) and its evidence remain documented in the [v3.4.0 report](../../QA/release-gates-20261018/report.md).

Release acceptance recorded a manual-edit metadata issue that remains in the current source. Its current behavior and user workaround are in the [status ledger](../project_status_ledger.md) and [session guide](../user/sessions_timeline_and_data.md); it was not changed in v3.4.0.

## Packaging architecture

The Windows desktop application uses a Tauri 2 shell under app/desktop/src-tauri. The build creates the Angular production bundle, freezes the FastAPI backend with PyInstaller, verifies the frozen service against an isolated SQLite database, places an allowlisted runtime into a deterministic archive, and embeds that archive in the portable executable and optional MSI.

The packaged runtime excludes source, tests, documentation, credentials, .env files, databases, logs, models, vectors, documents, caches, and development runtimes. The shell uses the system WebView2 runtime. The MSI can include an offline WebView2 installer when built with the offline option.

## Runtime and user data

At launch, the shell verifies and extracts the embedded runtime beneath:

~~~text
%LOCALAPPDATA%\DILIGENT\rt\<version>\<payload-sha256>
~~~

It starts the backend on a random localhost port, waits for the ready file and health response, and then opens the authenticated local interface. The backend is attached to a Windows Job Object and is asked to shut down cooperatively when the shell exits, with a bounded hard-kill fallback.

Mutable settings, the SQLite database, logs, embedding cache, source catalogs, and application state live beneath:

~~~text
%LOCALAPPDATA%\DILIGENT\data
~~~

Removing a package or MSI does not remove this user data. Versioned, hash-addressed runtimes can coexist during upgrades.

## Desktop dialogs

The Tauri surface uses the native operating-system directory picker for path-only RAG folder selection. Its dialog capability is limited to opening a directory for the authenticated localhost backend; it does not grant general filesystem or shell access.

Clinical-session attachments and patient profile images use normal HTML file inputs because those workflows require browser File objects. In non-Tauri source mode, the local backend provides directory navigation for RAG selection and returns the canonical path used by vectorization.

## Build

Build on Windows x64 with Rust/Cargo, the Windows C++ build tools, network access for dependencies and the WebView2 bootstrapper, and the repository launcher for pinned runtimes. Set the version to the synchronized source manifest version:

~~~powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.5.1 -DesktopTarget All
~~~

Use `-DesktopTarget Portable` or `-DesktopTarget Msi` to build one artifact. `-OfflineWebView2` applies only to MSI builds. Release builds require a clean worktree unless `-AllowDirtyTree` is supplied intentionally and recorded.

The portable EXE, MSI, and SHA-256 manifest are written under `release/`. Launcher option 8 builds artifacts; option 9 removes selected release artifacts. Intermediate build output and local validation captures are disposable and remain ignored. Only the current release evidence report and its representative screenshots are tracked under `assets/QA/`.

### Local packaged smoke

For a local portable startup check, verify the EXE against its SHA-256 manifest, open it, and confirm the window title is DILIGENT Clinical Copilot. Check `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json`, request `/api/health` on its recorded port, and confirm the backend log shows successful startup and static-asset requests. This smoke does not replace the hosted MSI lifecycle, provider E2E, native operator acceptance, or publication readback.

## Prepare and publish

The release workflow is .github/workflows/release.yml. It is manually dispatched with mode=prepare or mode=publish; pushing a tag alone does not publish a release.

Preparation runs the required backend quality and migration checks, SQLite/PostgreSQL persistence, frontend tests/build, the Windows Full browser harness, the LiveProvider lane, dependency audits, artifact checks, portable/deep-path smoke, the MSI install/launch/uninstall/reinstall lifecycle, and a populated MSI upgrade from the published v3.5.0 package. The upgrade fixture verifies session, configuration, timeline, and source-catalog data remain intact at the current migration head. It produces checksums and an immutable candidate receipt. Candidate workflow artifacts expire after 30 days; an expired candidate must be prepared and accepted again. Populated databases without a revision and databases with unknown or divergent revisions remain unsupported and are rejected without stamping.

Keep OPENCODE_GO_API_KEY in the GitHub secret store for LiveProvider E2E. Never put it in local evidence or logs.

Before publication, complete the protected desktop-release review with the exact candidate hashes and native operator evidence. Configure the environment to require a reviewer, restrict publishing to main, and disallow administrator bypass. Record unavailable checks as explicit waivers, not passes. Publication verifies the successful prepare run, exact source SHA, required gates, immutable receipt and hashes, signing state, synchronized main/develop refs, and absence of an existing tag or release. After approval it promotes the same candidate without rebuilding, creates the tag and draft release, uploads the unchanged assets, downloads them for hash and launch readback, then publishes and verifies the external release metadata. API errors do not prove that a tag or release is absent.

Signing state must be explicit for every release. The v3.4.0 release was intentionally unsigned and says so in the public notes. Do not infer the signing state of a later candidate from this release.

## Release evidence

The retained report records exact source and artifact hashes, hosted run IDs, operator acceptance, native workflows, download readback, the single-machine WebView2 waiver, and the known metadata issue. Keep detailed run output in hosted workflow artifacts or ignored local scratch space; add tracked QA material only when a future external acceptance claim requires it.

## Cleanup

~~~powershell
.\start_on_windows.ps1 -Action RemoveDesktopRelease -Version 3.4.0
.\start_on_windows.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
~~~

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI, stop a running application, or touch user data beneath %LOCALAPPDATA%\DILIGENT\data.
