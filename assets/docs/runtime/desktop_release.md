# Windows Desktop Release
Last updated: 2026-11-03

The latest published release is [v3.6.0](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.6.0), published on 2026-11-03 from tag commit `b487b98465a799f9ae1a7772fc28cb08355451f9`. This document is the durable record of release identity, artifact hashes, hosted gates, local acceptance, unsigned status, and waivers.

## Published release: v3.5.0 (historical)

The public [v3.5.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.5.0) was published on 2026-10-20. Tag v3.5.0 resolves to source commit `7a1b23e6f27b3d1987f1106eb2fb5f9aa40b2820`. Prepare run [39382912130](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39382912130) and protected, approved publish run [39419567722](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39419567722) succeeded.

| Published artifact | SHA-256 |
|---|---|
| DILIGENT-v3.5.0-windows-x64-portable.exe | 981fe5e3bf1db072f0e210de85c63b6baaa17be5f6a1027915124ac846435204 |
| DILIGENT-v3.5.0-windows-x64.msi | 4c6ff76d69693564b7889c26bec4784602324b404d539886ded5ab7716df8916 |
| DILIGENT-v3.5.0-windows-x64.sha256 | 8ee62e76c9f1e175a975f01dc46e4616899e4259d8cc314dde4554031286487e |

The publication workflow promoted the immutable prepare candidate without rebuilding. Its distribution verification succeeded, and the published asset digests match the candidate receipt. The candidate EXE opened DILI Agent and the other inspected pages on the local Windows 11 Pro host, returned `{"status":"ok"}` from `/api/health`, and closed cleanly. The hosted prepare run passed the required MSI lifecycle, portable/deep-path smoke, backend and browser gates, PostgreSQL persistence, dependency audits, and LiveProvider E2E.

The binaries are unsigned. The public release notes disclose the missing publisher signature and warn of a possible unknown-publisher or SmartScreen prompt.

The release owner explicitly waived a candidate-specific populated native analysis and clean-host Windows 11/WebView2 run, relying on the tested feature coverage in the [project status ledger](../project_status_ledger.md). The v3.5.0 local smoke verifies package launch and health only; it is not a populated analysis or clean-host compatibility result. The approval comment recorded these waivers. The earlier v3.4.0 release is summarized below.

Release acceptance recorded a manual-edit metadata issue that remains in the current source. Its current behavior and user workaround are in the [status ledger](../project_status_ledger.md) and [session guide](../user/sessions_timeline_and_data.md); it was not changed in v3.4.0.

## Published release: v3.5.1 (patch, historical)

The public [v3.5.1 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.5.1) was published on 2026-10-22. Tag v3.5.1 resolves to source commit `49f3a7f4a14b808586288ffd59befc680f8d88bd` (`main` and `develop`). Prepare run [39827941660](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39827941660) and the approved publish run [39832496262](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39832496262) succeeded; published asset digests match the candidate receipt.

| Published artifact | SHA-256 |
|---|---|
| DILIGENT-v3.5.1-windows-x64-portable.exe | 13daabd0ad2ca7f14998c66ef9980ef12bf85d810ee6d1e6d60c4075555b1074 |
| DILIGENT-v3.5.1-windows-x64.msi | 11a5a90297a365bf24ca6dc155ae490671c354e7b74cc3cf1487b7efbd0a2810 |
| DILIGENT-v3.5.1-windows-x64.sha256 | 01f508ca8f475bdcfb46eadd68e4b1068fc49e3f29f6ed1bbedae6c2c53ddfcd |

Binaries are unsigned. Before preparation, local checks passed (892 backend unit tests, 112 frontend tests, 57-test Windows Full browser suite). Local inspection of the first candidate build found the packaged window showing `asset not found`: the staged Tauri config set `frontendDist` to an absolute drive-letter path, which Tauri does not embed. The build now writes a path relative to `src-tauri`, and the smoke script no longer applies the current API-boundary assertions to the v3.5.0 upgrade baseline. The exact published portable EXE was launched natively (WebView2 inspection showed the bundled DILI Agent UI, `/api/health` ok, clean close, no orphan processes). The existing smoke checks only window title and health, so UI rendering is not automated; the owner-approved waivers were populated native analysis, clean-host WebView2, and local MSI install (needs elevation; hosted MSI lifecycle and v3.5.0 upgrade gates passed). The [v3.5.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.5.0) remains documented below.

## Published release: v3.6.0 (minor, latest)

The public [v3.6.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.6.0) is the latest release. Tag v3.6.0 resolves to source commit `b487b98465a799f9ae1a7772fc28cb08355451f9` (`main` and `develop`). Prepare run [41637519430](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/41637519430) and the approved publish run [41684111486](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/41684111486) succeeded. An earlier prepare run failed only at the LiveProvider gate because the test used the retired `deepseek-v4-flash` id (HTTP 400); the test now uses `deepseek-v4.1-flash`. Published assets were downloaded and their digests match the candidate receipt.

| Published artifact | SHA-256 |
|---|---|
| DILIGENT-v3.6.0-windows-x64-portable.exe | c8732936bb5bf2fd8fd7b17ed56ea91f6d92656d4f17154c2d8437cf0bdaa9a1 |
| DILIGENT-v3.6.0-windows-x64.msi | 1583e9b081592b8f4d720cfc96aa1f16d0d3b6d7bbab0035b2ec4f8e5e169c7e |
| DILIGENT-v3.6.0-windows-x64.sha256 | b566bb2f3bf1efa95ae60d1a625d1d477806d834eb57245280d63e2bfac24ffc |

Binaries are unsigned. Local validation of the exact candidate on Windows 11 included portable smoke with a deep path, a 13-check UI run against an isolated `LOCALAPPDATA` (styled UI, route reload, draft persistence across reload and restart, clean close), and an elevated MSI lifecycle with in-place upgrade from the published v3.5.1 MSI (sessions, versions and settings preserved, migration head verified), uninstall, and reinstall. The local release validation also recorded 1034 backend unit and persistence tests passed with 26 PostgreSQL parameterizations skipped, 194 frontend tests passed, the Angular production build passed, and 63 Windows Full browser tests passed. Seven README screenshots were captured from a running build with fictional data; the Knowledge screenshot used a small fixture rather than a full terminology import. Markdown links were checked in 42 tracked files with no broken links. The owner waived a clean-host WebView2 test and a populated native analysis.

## Published release: v3.4.0 (historical)

The public [v3.4.0 release](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/releases/tag/v3.4.0) was published on 2026-10-18 from tag commit `d4efa5e0bccfa552c80a5edeff2168bfaa718639`. Hosted prepare run [39109627873](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39109627873) and approved publish run [39125173765](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/39125173765) passed. The published portable EXE digest was `702f8df1ff1520fe3a43aafc8cfbc589c10a9502cdf2eb2d0264d78a049acd26`; the MSI digest was `3587f93c019e9983e11a5b54955d88e747d0dfbc08b3d369e0447d12e502243d`. Downloaded bytes matched the published assets.

The hosted prepare gates passed 864 backend tests, 53 browser tests, Angular/Vitest and production build, PostgreSQL persistence, LiveProvider E2E, dependency/security checks, portable/deep-path smoke, and MSI install, launch, upgrade, uninstall, and reinstall. On one Windows 11 host, an isolated synthetic DILI assessment persisted and exported a review-required report; a missing visit date blocked session/job creation; settings, sessions, versions, and timelines survived restart; and RAG cited two indexed documents while skipping a byte-identical duplicate. The local qwen3.5:9b timeline request timed out twice and saved a deterministic, source-backed fallback. The binaries were unsigned, and clean-host WebView2 compatibility was waived because only one Windows machine was available. This release evidence does not establish clinical validity or clean-host compatibility.

## Packaging architecture

The Windows desktop application uses a Tauri 2 shell under `app/desktop/src-tauri`. The build creates the Angular production bundle, stages the SPA and startup splash as Tauri frontend assets, freezes the API backend with PyInstaller, verifies the frozen service against an isolated SQLite database, places the backend and required catalogs into an allowlisted deterministic runtime archive, and embeds that archive in the portable executable and optional MSI. The backend archive excludes `app/client/dist/browser`; Tauri owns the frontend files and resolves client-side routes to its bundled `index.html`.

The packaged runtime excludes source, tests, documentation, credentials, .env files, databases, logs, models, vectors, documents, caches, and development runtimes. The shell uses the system WebView2 runtime. The MSI can include an offline WebView2 installer when built with the offline option.

## Runtime and user data

At launch, the shell shows the bundled startup splash, verifies and extracts the embedded backend runtime beneath:

~~~text
%LOCALAPPDATA%\DILIGENT\rt\<version>\<payload-sha256>
~~~

It starts the backend on a random localhost port, waits for the ready file and health response, then loads the bundled Angular interface from Tauri's local origin. The shell passes the API origin and one-time session token to the frontend; API requests include `X-DILIGENT-Desktop-Session`. The backend accepts CORS only from `http://tauri.localhost` and keeps the API bound to loopback. The backend is attached to a Windows Job Object and is asked to shut down cooperatively when the shell exits, with a bounded process termination fallback recorded in the backend log.

The shell hands the one-time bootstrap URL to the SPA once. The SPA then keeps the loopback API origin and session token in the window's `sessionStorage`, so a full page reload (for example F5 on `/clinical-sessions`) continues to call the loopback API instead of the Tauri origin. The packaged CSP disables Tauri's nonce rewriting for `style-src` only (`dangerousDisableAssetCspModification`), because Angular injects component styles at runtime, and the production Angular build does not inline critical CSS because its `onload` handler is blocked by the CSP; without both, the packaged UI renders unstyled. The backend's packaged CORS allow-list covers every request header the client sends, including `Cache-Control` and `Pragma` on no-cache reads.

Mutable settings, the SQLite database, logs, embedding cache, source catalogs, and application state live beneath:

~~~text
%LOCALAPPDATA%\DILIGENT\data
~~~

Removing a package or MSI does not remove this user data. Versioned, hash-addressed runtimes can coexist during upgrades.

## Desktop dialogs

The bundled Tauri surface uses the native operating-system directory picker for path-only RAG folder selection. Its local-window capability grants the dialog plugin's directory picker and does not grant general filesystem or shell access to remote content.

Clinical-session attachments and patient profile images use normal HTML file inputs because those workflows require browser File objects. In non-Tauri source mode, the local backend provides directory navigation for RAG selection and returns the canonical path used by vectorization.

## Build

Build on Windows x64 with Rust/Cargo, the Windows C++ build tools, network access for dependencies and the WebView2 bootstrapper, and the repository launcher for pinned runtimes. Set the version to the synchronized source manifest version:

~~~powershell
.\setup_and_maintenance.ps1 -Action BuildDesktopRelease -Version 3.6.0 -DesktopTarget All
~~~

Use `-DesktopTarget Portable` or `-DesktopTarget Msi` to build one artifact. `-OfflineWebView2` applies only to MSI builds. Release builds require a clean worktree unless `-AllowDirtyTree` is supplied intentionally and recorded.

The portable EXE, MSI, and SHA-256 manifest are written under `release/`. Launcher option 8 builds artifacts; option 9 removes selected release artifacts. Intermediate build output and local validation captures are temporary and are removed after their durable conclusions are recorded here.

### Local packaged smoke

For a local portable startup check, verify the EXE against its SHA-256 manifest, open it, and confirm the DILIGENT interface appears. Reload a client-side route such as `/clinical-sessions`, confirm the bundled CSS/JavaScript load, and check `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json` plus `/api/health` on its recorded port. Confirm the frontend can access a protected API after bootstrap, then close the window and verify the backend PID exits and its port closes. The backend log is `%LOCALAPPDATA%\DILIGENT\data\cache\logs\desktop-backend.log`. `app/desktop/build/smoke_release.ps1` automates launch, extraction, backend health, the unauthenticated 401 boundary, exact-origin CORS, and close/port release against an isolated `LOCALAPPDATA`, but it does not render the UI. For UI-level evidence, launch the executable with an isolated `LOCALAPPDATA` and `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=<port>`, attach Playwright over CDP, and check computed styles, an in-app draft save, a settings change that survives a full reload, and `Network` preflight statuses. Release builds require `rustc 1.95.0`; install it side by side with `rustup toolchain install 1.95.0 --profile minimal` and set `RUSTUP_TOOLCHAIN=1.95.0` for the build process. Each desktop build runs `uv sync --locked` without test extras, so run `uv sync --locked --all-extras` again before Playwright or pytest. This smoke does not replace the hosted MSI lifecycle, provider E2E, native operator acceptance, or publication readback.

## Prepare and publish

The release workflow is .github/workflows/release.yml. It is manually dispatched with mode=prepare or mode=publish; pushing a tag alone does not publish a release.

Preparation runs the required backend quality and migration checks, SQLite/PostgreSQL persistence, frontend tests/build, the Windows Full browser harness, the LiveProvider lane, dependency audits, artifact checks, portable/deep-path smoke, the MSI install/launch/uninstall/reinstall lifecycle, and a populated MSI upgrade from the previously published package (v3.5.1 for v3.6.0). The upgrade fixture verifies session, configuration, timeline, and source-catalog data remain intact at the current migration head. It produces checksums and an immutable candidate receipt. Candidate workflow artifacts expire after 30 days; an expired candidate must be prepared and accepted again. Populated databases without a revision and databases with unknown or divergent revisions remain unsupported and are rejected without stamping.

Keep OPENCODE_GO_API_KEY in the GitHub secret store for LiveProvider E2E. Never put it in local evidence or logs.

Before publication, complete the protected desktop-release review with the exact candidate hashes and native operator evidence. Configure the environment to require a reviewer, restrict publishing to main, and disallow administrator bypass. Record unavailable checks as explicit waivers, not passes. Publication verifies the successful prepare run, exact source SHA, required gates, immutable receipt and hashes, signing state, synchronized main/develop refs, and absence of an existing tag or release. After approval it promotes the same candidate without rebuilding, creates the tag and draft release, uploads the unchanged assets, downloads them for hash and launch readback, then publishes and verifies the external release metadata. API errors do not prove that a tag or release is absent.

Signing state must be explicit for every release. The v3.4.0 release was intentionally unsigned and says so in the public notes. Do not infer the signing state of a later candidate from this release.

## Release evidence

The release record above retains exact source and artifact hashes, hosted run IDs, native acceptance, download readback, waivers, and known release issues. Hosted workflow artifacts provide run-level logs and expire after 30 days. Local logs, screenshots, databases, and build output are temporary and are removed after any durable release conclusion is added here.

## Cleanup

~~~powershell
.\setup_and_maintenance.ps1 -Action RemoveDesktopRelease -Version 3.4.0
.\setup_and_maintenance.ps1 -Action RemoveDesktopRelease -AllDesktopReleases
~~~

These commands remove repository release artifacts and generated desktop build state only. They do not uninstall an MSI, stop a running application, or touch user data beneath %LOCALAPPDATA%\DILIGENT\data.
