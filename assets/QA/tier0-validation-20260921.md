# Tier 0 validation evidence
Last updated: 2026-09-21

## Run identity

- Repository: `CTCycle/DILIGENT-Clinical-Copilot`
- Branch: `develop`
- HEAD: `d0e8ba1809d7a79615f300403891ae2066a574a7`
- Worktree at start: clean
- Environment: Windows source mode, Python `3.14.7`, SQLite, configured source launcher
- Evidence scope: synthetic/disposable validation only; no cloud clinical call, source refresh, access-key mutation, or developer-database mutation

The V00/V02 baseline run started before the ledger updates in this commit. The
exact working-tree revision and pre-edit clean state are therefore part of the
evidence boundary; the occupied-port subcases were executed afterward against
the same HEAD and disposable runtime policy.

## V00 — revision and evidence baseline

Status: **PASS**

Checks completed:

1. `git rev-parse HEAD` → `d0e8ba1809d7a79615f300403891ae2066a574a7`.
2. `git rev-parse --abbrev-ref HEAD` → `develop`.
3. `git status --porcelain=v1` → clean.
4. `git ls-tree -r --name-only HEAD` → 713 tree entries.
5. Historical snapshot `ba47761aa456e78923a0df824f76ee4016ee0af6` resolved as a commit.
6. Snapshot drift was limited to `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `start_on_windows.ps1`, Python/runtime documentation, `project_status_ledger.md`, and `runtime/validation_ledger.md`.
7. The 109 local Markdown links parsed from `assets/docs/project_status_ledger.md` all resolved. No missing evidence path was silently accepted.

The snapshot drift does not invalidate unrelated clinical claims wholesale. It
does require launcher and automated-gate claims to be anchored to the current
V01/V02 evidence below.

## V01 — source startup, migration, health, populated reuse, shutdown, and port ownership

Status: **FAIL**

The source launcher was exercised with `EMBEDDED_DATABASE=true`,
`DATABASE_SQLITE_PATH`, and `DILIGENT_SQLITE_PATH` pointing to disposable files
under `runtimes/cache/validation/tier0-v01-20260921`. `settings/.env` and the
developer database were not edited.

### Fresh disposable SQLite run

- Two `start_on_windows.ps1 -Action InitializeDatabase` runs completed.
- Alembic reached `202609170001`.
- `PRAGMA foreign_key_check` returned `[]`.
- `PRAGMA integrity_check` returned `ok`.
- The fresh database contained 24 tables, one application-configuration row,
  821 reference-catalog entries, and zero clinical sessions.
- `start_on_windows.ps1 -Action Launch` completed successfully.
- `GET /api/health` returned `200` with `{"status":"ok"}`.
- `GET /` on the frontend returned `200`.
- `GET /api/inspection/sessions` returned `200` with an empty result.
- The launcher-owned backend and frontend processes were identified by exact
  command line and stopped by explicit process IDs. Ports `7690` and `9847`
  were then free.

### Populated disposable clone

The current `app/resources/database.db` was read-only inspected before the
clone: Alembic `202609170001`, 18 sessions, 50 versions, 28 revision runs, 90
artifacts, `foreign_key_check=[]`, and `integrity_check=ok`. A byte copy was
made below the disposable validation root and passed through:

1. `start_on_windows.ps1 -Action InitializeDatabase`.
2. `start_on_windows.ps1 -Action Launch`.
3. `GET /api/health` → `200`.
4. `GET /api/inspection/sessions` → `200`, total `18`.

The clone retained 18 sessions, 50 versions, 28 revision runs, 90 artifacts,
Alembic `202609170001`, `foreign_key_check=[]`, and `integrity_check=ok` after
launcher initialization and application startup. Its owned process tree was
stopped explicitly and the ports were rechecked as free.

### Occupied-port ownership subcase

A controlled `python -m http.server 7690 --bind 127.0.0.1` listener was started
from the disposable validation root with PID `2576`. The launcher was then
run with the same disposable SQLite overrides used above.

- Expected contract: leave a foreign listener in place and exit with an
  actionable port-conflict failure; do not terminate a process it does not
  own.
- Observed launcher output: `Releasing port 7690 from PID 2576`.
- The controlled listener was terminated by the launcher before startup.
- The launcher then started the backend and frontend, returned exit code `0`,
  and `/api/health` returned `200` from the newly started backend.
- The relevant implementation unconditionally calls `taskkill.exe` for every
  listener returned by `Stop-PortListeners` before launching the application
  (`start_on_windows.ps1`, lines 635-640 and 783-784).

This fails the roadmap's occupied-port ownership gate and is tracked as
`VAL-V01-01`, mapped to project issue `ISSUE-006`. A repeat with a controlled
foreign listener on the frontend port `9847` (PID `27248`) produced the same
result: the listener was released, the launcher returned `0`, and the frontend
then returned `200`.

Primary failure class: configuration / UX-workflow (unsafe process ownership).
No source fix was applied during this validation run; revalidation is required
after ownership-scoped port handling is implemented.

## V02 — automated baseline gates

Status: **PASS**

Commands and results:

```text
app\tests\run_tests.bat unit
742 passed, 7 warnings in 48.89s
```

```text
cd app\client
npm run test -- --no-watch
23 test files, 96 tests passed
npm run build -- --progress=false
Application bundle generation complete
```

The backend runner used the canonical `runtimes/cache/pytest` hierarchy. The
frontend commands used `runtimes/cache/npm`. The warnings were dependency/API
deprecation warnings; no test failure occurred.

## Cleanup and boundaries

- Disposable fresh and populated SQLite files were removed from
  `runtimes/cache/validation/tier0-v01-20260921`.
- The disposable occupied-port run roots were removed after the controlled
  listener and launcher-owned process trees were stopped.
- Ports `7690` and `9847` were free after cleanup.
- No matching launcher-owned backend/frontend process remained.
- No settings, access-key material, source catalog, vector store, or developer
  database was mutated.
- Tier 0 does not establish live provider correctness, source-refresh safety,
  clinical-report correctness, accepted revision finalization, packaged
  desktop behavior, or release readiness.

## Next evaluation boundary

Resume with Tier 1 only after `ISSUE-006` is closed or explicitly accepted as
an environment/launcher limitation. Then proceed to the roadmap's highest-risk
clinical and stateful slices: exact provider execution (`V21`),
multi-drug/longitudinal correctness (`V22`), source refresh (`V36`–`V39`),
timeline failure/cancellation (`V42`), and QA-clean revision finalization
(`V44`).
