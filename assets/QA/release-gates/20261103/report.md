# v3.6.0 release validation, 2026-11-03

Scope: `develop` at `7287bbb9` (screenshots and README retaken on top of the 2026-11-02 candidate; no application code changed since the [2026-11-02 report](../20261102/report.md)). Windows 11 host, non-administrator.

| Check | Result |
|---|---|
| Ruff, Pyright | pass (0 errors) |
| Backend unit + persistence | 1034 passed, 26 skipped (PostgreSQL parameterizations; no `TEST_DATABASE_URL`) |
| Vitest | 24 files / 194 tests passed |
| Windows Full browser harness (`run_browser_e2e.ps1 -Suite Full`) | 63 passed |
| README screenshots | seven 1440x900 captures from the running build with fictional data (`ui-validation/release-screenshots-20261102/capture.py`); the Knowledge capture uses a small test fixture, not a full terminology import |
| Markdown links | 42 tracked files, 0 broken; no unreferenced figures |

## Findings
- The Full browser harness leaves its Angular preview server (`scripts/preview-server.mjs`, port 9847) listening after a passing run. It was stopped manually; not release-blocking.

## Publication
- Prepare run 41637519430 (source `b487b984`) passed all gates after the LiveProvider test model id was corrected to `deepseek-v4.1-flash` (first run 41635859941 failed with provider HTTP 400 for `deepseek-v4-flash`). Publish run 41684111486 succeeded; tag `v3.6.0` resolves to `b487b984`; the downloaded assets match the candidate hashes.
- Local candidate checks: portable smoke (`desktop/smoke/3.6.0-candidate-41637519430`), a 13-check packaged UI run (`desktop/portable-3.6.0-ui`), and an elevated MSI lifecycle with upgrade from the published v3.5.1 MSI (`desktop/smoke/3.6.0-candidate-msi`) all passed.
- Waived: clean-host WebView2 test, populated native analysis. Binaries are unsigned.
