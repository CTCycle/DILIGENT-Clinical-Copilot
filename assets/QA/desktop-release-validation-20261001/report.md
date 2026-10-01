# DILIGENT 3.4.0 candidate-preparation validation

Date: 2026-10-01 (Europe/Rome)

Product candidate source SHA: 2f804d251769f4e01375d02459e41dd9970ec8eb

Branch: develop

Remote candidate: origin/develop points to the same SHA

## Disposition

The candidate is prepared locally for a later release session. This record does
not approve or perform a v3.4.0 release.

No v3.4.0 tag exists, no GitHub release exists, and no release assets were
uploaded. The portable EXE, MSI, and checksum manifest under release/ are local
candidate outputs only.

## Source validation

The dependency-only candidate change was validated before it was committed:

| Check | Result |
| --- | --- |
| Python compilation | PASS |
| Alembic upgrade, head, and drift | PASS; head 202609170001; no drift |
| Ruff | PASS |
| Pyright | PASS; 0 errors, 0 warnings, 0 informations |
| Backend unit suite | PASS; 838 passed |
| SQLite persistence | PASS; 15 passed; 14 PostgreSQL cases intentionally skipped locally |
| Angular/Vitest | PASS; 24 files and 105 tests |
| Angular production build | PASS |
| pip-audit | PASS; no known vulnerabilities |
| npm audit | PASS; 0 vulnerabilities at high severity or above |
| cargo-audit | PASS; six allowed unmaintained-crate warnings and no denied unsound finding |
| Full browser E2E | PASS; 53 passed, 0 failures, 0 errors, 0 skips; shell exit 0 |

The source change is limited to the locked dependency remediation: urllib3
2.8.0 and the Angular 21.2.25 runtime/compiler/router line. The final clean
frontend install and production build passed with the synchronized lockfile.

## Hosted CI

Exact-SHA CI run 36786345174 completed successfully:

- backend-quality: PASS
- persistence-contract, including PostgreSQL: PASS
- windows-regression, including frontend build and Full browser E2E: PASS
- security-scan, including Python, Angular, and Rust audits: PASS

The live-provider-e2e job was skipped by the push event. The exact candidate
was then exercised by workflow-dispatch run
[36828262326](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36828262326),
which completed successfully on 2026-10-01. Its live-provider browser E2E
passed using the synthetic clinical flow and the exact `opencode_go` /
`deepseek-v4-flash` route, including the non-dry revision path. The hosted
dispatch did not create a tag, release, or distribution upload.

## Fresh desktop artifacts

The pinned Windows build used Rust 1.95.0, Python 3.14.7, uv 0.11.30,
PyInstaller 6.21.0, and the repository desktop toolchain. The runtime archive
contained 1,108 files, measured 400,239,424 bytes, and validated the required
frontend index, b/DILIGENTBackend.exe, and model-capability catalog entries.
The embedded runtime archive SHA-256 was
105727fc84c900b82f9d98f3f3d6b2200e984e88e98a60ebcbf5afd6af333536.

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| DILIGENT-v3.4.0-windows-x64-portable.exe | 171,720,704 bytes | 0209cf25a8acf3a39762ffeec48488c28103e1b96c7c5a57ed8beb813574e167 |
| DILIGENT-v3.4.0-windows-x64.msi | 170,164,224 bytes | 1df49329361b600edb2e0c07949fe00bc113b25122d55af9ba93cb52f67fd508 |

The release checksum manifest matches both files.

## Portable and deep-path smoke

The standard portable smoke passed with two launches, runtime extraction,
release version 3.4.0, health 200, authenticated-boundary response 401,
branded window title, stale-ready-file replacement, clean backend shutdown,
and port closure.

The current artifact was also copied to an executable path of exactly 230
characters with an isolated LOCALAPPDATA path of exactly 107 characters.
Both launches returned health 200 and settings 401, showed the branded title,
cleanly stopped the backend, and closed the assigned port. The restricted
text/log scan found zero onnxruntime, filename-length, or ImportError matches.

## MSI and distribution boundaries

Read-only MSI metadata passed:

- ProductName: DILIGENT Clinical Copilot
- ProductVersion: 3.4.0
- Manufacturer: CTCycle
- UpgradeCode: {2CF8EF35-4160-59EB-89D8-01EC7D19A887}
- ProductCode: {86889139-A144-4928-B099-4F70B21DE310}

The current token is not administrator-capable. MSI install, launch,
3.3.0-to-3.4.0 upgrade, persistence-marker verification, uninstall, and
reinstall remain PENDING ON SUITABLE ADMINISTRATOR HOST. They were not claimed
as passes.

Both local artifacts report NotSigned. Signing, signature verification,
offline WebView2 packaging, and clean-machine distribution smoke remain
PENDING DISTRIBUTION PROCEDURES. Offline WebView2 was not requested as a
separate distribution mode in this preparation session.

The native Computer Use inventory exposed no native app/window surface. The
portable process/API smoke passed, but packaged screen-level navigation,
native folder selection, and visual UI interaction remain PENDING ON SUITABLE
INTERACTIVE HOST. The candidate-specific manual procedure is recorded in
host-checklist.md.

## Cleanup and release boundary

The manually launched portable process and backend were stopped, no task-owned
packaged listeners remained, and the disposable deep-path roots were removed.
No tag, GitHub release, release upload, branch synchronization to main, or
distribution publication was performed.
