# v3.6.0 release preparation, 2026-11-02

Scope: `develop`, version 3.6.0 (artifacts built from commit `61d21293`; the only later change is the smoke-script fix below), Windows x64, local host (Windows 11, non-administrator). Nothing was merged, tagged, published, or deployed.

## Checks run
| Check | Result |
|---|---|
| compileall, Ruff, Pyright | pass (0 errors) |
| Backend unit + persistence | 1034 passed, 26 skipped (PostgreSQL parameterizations; no `TEST_DATABASE_URL`) |
| Vitest | 24 files / 194 tests passed |
| Angular production build | pass |
| Windows Full browser E2E harness | 63 passed (`e2e/browser-20261102-full.log`) |
| npm audit (client, desktop; `--audit-level=high`) | exit 0 (client: 18 moderate) |
| Desktop build (`BuildDesktopRelease -Version 3.6.0 -DesktopTarget All`, rustc 1.95.0) | pass; log `desktop/build-3.6.0.log` |
| Portable smoke (`smoke_release.ps1 -DesktopTarget Portable -RequireUnsigned -TestDeepPath`) | pass: checksum, PE check, two launches, `/api/health` 200, release version 3.6.0, clean close, backend and port released, 230-character path |
| Portable runtime inspection (isolated `LOCALAPPDATA`) | no databases, logs, tests, docs, `.env`, or local paths in the extracted runtime; `settings/.env.example` template only; data created under the isolated data root |
| MSI metadata | ProductVersion 3.6.0, Manufacturer CTCycle, same UpgradeCode as the v3.5.1 MSI (major upgrade), unsigned by policy |

## Defect fixed
`smoke_release.ps1` pinned the expected upgraded Alembic head to `202609170001`; the hosted MSI upgrade gate would fail against head `202611010003`. It now reads `HEAD_REVISION` from the migration module.

## Local build digests (SHA-256, local validation build only)
Recorded in `release/DILIGENT-v3.6.0-windows-x64.sha256`. Builds are not byte-reproducible; the hosted prepare run produces the candidate that is published.

## Not run
- MSI install/uninstall/reinstall lifecycle and populated v3.5.1 → v3.6.0 MSI upgrade: need an administrator-capable disposable host (the smoke refuses otherwise). The local `release/DILIGENT-v3.5.1-*` files are rebuilds of the current tree, not the published bytes, so they cannot serve as an upgrade baseline.
- Packaged existing-user upgrade from the published v3.5.1 binaries (database upgrade from the previous release head is covered by `tests/persistence/test_previous_release_upgrade.py`, which passed).
- Hosted PostgreSQL job, LiveProvider, LocalOllama, pip-audit, cargo-audit (not re-run), clean-host WebView2.
- Knowledge/terminology screens were not captured: the validation host has no imported knowledge base.
