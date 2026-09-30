# QA Regression
Last updated: 2026-10-01

The exact v3.4.0 candidate preparation evidence is recorded in
assets/QA/desktop-release-validation-20261001/report.md. It preserves the
distinction between green source/hosted gates and the administrator,
native-interactive, signing, and external-auth boundaries that cannot be
proven on this workstation.

## Scope

The mandatory regression path is the deterministic product matrix described in
[`coding/testing_and_quality.md`](../coding/testing_and_quality.md). It covers
backend/static/frontend/build checks, SQLite and hosted PostgreSQL persistence,
the synthetic access-key lifecycle, security audits, and the Windows browser
slice. External provider and heavyweight embedding checks are named integration
lanes, not unexplained skips in the core result.

## Canonical Windows browser harness

From the repository root, run:

```powershell
.\app\tests\ci\run_browser_e2e.ps1 -Suite Full
```

`Full` creates an isolated SQLite path and access-key encryption-material path,
seeds reference catalogs plus the smallest repository-backed clinical session
and persisted timeline fixture, starts the deterministic fake Ollama service,
starts backend and frontend, runs the E2E directory, writes a JUnit result, and
cleans up all owned process trees. The Full collection explicitly excludes:

- `test_live_provider_flow.py`, which is run only by the provider lane;
- `test_multilingual_embedding_runtime.py`, which remains an explicit optional
  embedding integration path.

The harness fails if the mandatory JUnit result contains any skipped test. It
does not infer skip counts by grepping console output. Read the generated
backend/frontend/Ollama logs when readiness or collection fails.

The external provider lane is:

```powershell
.\app\tests\ci\run_browser_e2e.ps1 -Suite LiveProvider
```

It runs only `test_live_provider_flow.py` and requires
`DILIGENT_LIVE_PROVIDER_E2E=1` plus the hosted `OPENCODE_GO_API_KEY`. A real
provider key is not required for access-key lifecycle validation; provider-side
credential acceptance belongs to the provider component.

## Persistence boundary

The local persistence command is useful for SQLite-only work:

```powershell
Set-Location .\app\server
.\.venv\Scripts\python.exe -m pytest ..\tests\persistence -q
```

Without `TEST_DATABASE_URL`, PostgreSQL parameterizations are intentionally
skipped. The mandatory hosted `persistence-contract` job supplies PostgreSQL
and executes the same contract against both engines; a local zero-skip number
is not the acceptance criterion.

## Packaged desktop smoke test

After a Windows desktop release build, validate the built portable artifact
through the separate release-readiness procedure:

1. Verify `release\DILIGENT-v<version>-windows-x64-portable.exe` and its
   matching `.sha256` entry.
2. Open the portable EXE and confirm a responding window titled `DILIGENT
   Clinical Copilot`.
3. Confirm `%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>\extraction.complete` exists.
4. Read `%LOCALAPPDATA%\DILIGENT\data\state\desktop-backend-ready.json` and
   request `/api/health` on its recorded port.
5. Confirm the desktop backend log contains successful startup and
   static-asset requests.

This is a release-readiness procedure, not a validation-component gate. It
does not replace MSI install/upgrade/uninstall, offline WebView2, code-signing,
or clean-machine distribution checks. Packaged desktop uses a random backend
port and is not tested through source-mode `7690`/`9847` URLs.

## Quality and cleanup

The backend quality job runs compilation, Alembic upgrade/head/drift checks,
Ruff, Pyright, and the complete unit suite. The Windows job runs Angular/Vitest,
the production build, Playwright installation, and the canonical Full harness.
The security job runs pip-audit, Angular and desktop npm audits, and
cargo-audit. Stop task-owned backend/frontend/fake-Ollama processes after local
runs and verify ports `7690`, `9847`, and `11435` are free.

## Final source closure — 2026-09-30

The final local source matrix passed after the API matrix and RAG duplicate
policy changes:

- backend compilation, Alembic head/drift, Ruff, and Pyright passed;
- the complete backend unit suite passed **838 tests**;
- SQLite persistence passed **15 tests**, with 14 PostgreSQL parameterizations
  intentionally skipped because no local `TEST_DATABASE_URL` was configured;
- Angular/Vitest passed **24 files / 105 tests**, and the production build passed;
- pip-audit reported no known vulnerabilities, both npm audits reported zero
  vulnerabilities, and cargo-audit passed with six allowed unmaintained-crate
  warnings;
- the canonical command `.\app\tests\ci\run_browser_e2e.ps1 -Suite Full`
  recorded **53 passed, 0 failed, 0 errors, 0 skipped** in the final JUnit.

The local client lockfile was refreshed with `npm audit fix --package-lock-only`
only; the audit remediation changed transitive versions and did not change the
declared package manifest. The API suite now fails on an unclassified OpenAPI
operation, so adding a public route requires an explicit validation disposition.

The real local RAG fixture verified raw-byte duplicate handling: 3 physical
supported files became 2 unique ingested documents, 2 chunks, 1 duplicate, and
2 vector documents. Both physical paths remained in Data Inspection, and
retrieval returned one canonical citation. This is the required byte-identical
policy; semantic deduplication is not implied.

Earlier portable 3.4.0 smoke evidence passed, but the current re-run failed
the native window-title assertion: `io.github.ctcycle.diligent-siw` was
reported instead of `DILIGENT Clinical Copilot`. MSI checksum and metadata
passed, but install/upgrade/uninstall remain host-blocked on a
non-administrator Windows session because the package is `ALLUSERS=1`. The
current result is not converted into a release PASS; see the consolidated
[desktop revalidation follow-up](../../QA/final-validation-closure-20260930/report.md).
