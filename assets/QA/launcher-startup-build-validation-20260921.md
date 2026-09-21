# Launcher, startup, and build-state validation

Validation date: 2026-09-22
Repository: `CTCycle/DILIGENT-Clinical-Copilot`
Branch: `develop`
Validation starting revision: `e85faba5faf914fe43109e4251ac9f2d3a10407b`
Host: Windows, PowerShell 7, portable Python 3.14.7, Node.js 22.13.0, npm 10.9.2

This record covers the current source-mode launcher contract. The historical
`assets/QA/tier0-validation-20260921.md` record was not rewritten.

## Repository and baseline checks

- `git ls-tree -r --name-only develop`: 715 entries at validation time.
- `git grep -n "ALWAYS_REBUILD"`: no active references.
- Configured source-mode ports: backend `7690`, frontend `9847`.
- `settings/.env` remained runtime configuration; it was not included in the
  frontend production fingerprint.

Pre-change warm, no-conflict launcher samples from the same checkout were
`11198 ms` and `6006 ms`. These are raw single-run samples, not a statistical
benchmark. After the implementation, a warm no-conflict launch completed in
`7404 ms` with both dependency synchronization and Angular build skipped. The
prompt-inclusive conflict sample was excluded from timing comparison because it
included manual confirmation wait time. No percentage improvement is claimed.

## Port-conflict matrix

| Case | Result | Observed contract |
|---|---|---|
| Both configured ports free | PASS | Launch proceeded without a confirmation prompt. |
| One configured port occupied; answer `No` | PASS | Launcher aborted; the holder remained listening and no termination was attempted. |
| Occupied port in noninteractive launch | PASS | Launcher aborted safely with no termination. |
| Backend and frontend held by two PIDs; answer `Yes` | PASS | One combined prompt listed both unique PIDs; each was handled once; ports were rechecked before spawn. |
| One PID held both configured ports; answer `Yes` | PASS | One process entry listed ports `7690, 9847`; one direct PID termination attempt was made. |
| Holder exited while confirmation prompt was open | PASS | Fresh discovery found no records, so no process was terminated and launch continued. |
| Replacement acquired a port after confirmation began | PASS | Final authoritative scan reported the replacement PID/port mapping and launch aborted; the replacement was then cleaned explicitly. |
| Termination/access-denied failure | NOT INDEPENDENTLY INDUCED | The implementation reports failed PIDs and relies on the final port scan; no elevated/system-owned failure was forced during this run. |

The launch path now scans both ports with one `netstat.exe -ano -p tcp` pass,
aggregates by unique PID, resolves only a safe process name, confirms once,
rescans before termination, uses direct PID termination for the confirmed
holders, and performs an authoritative final rescan. The explicit
`KillApplicationProcesses` action retained its repository-qualified process-tree
cleanup behavior.

## Build freshness matrix

- Missing marker: PASS. A normal launch rebuilt the production output, ran
  `npm ci` from the lockfile, and wrote the marker after the stable build.
- Valid marker and output: PASS. The final warm launch reported current output,
  skipped npm dependency synchronization, and skipped Angular build.
- Production source input change: PASS. A temporary production input caused
  `BuildInputsChanged` and a rebuild; frontend dependencies were reused when
  their dependency fingerprint was unchanged.
- Metadata-only source touch: PASS. Changing only the `styles.scss` write time
  did not trigger a rebuild.
- Test-only `*.spec.ts` change: PASS. A temporary spec file did not change the
  production build decision.
- Explicit `-Action RebuildFrontend`: PASS. It rebuilt regardless of the
  recorded fingerprint; the marker was restored by a subsequent source-mode
  build.
- Direct frontend gate: PASS. `npm run build -- --progress=false` succeeded.
  Angular cleanup removed the ignored marker as part of replacing `dist`, and
  the next normal launch correctly treated the missing marker as stale and
  rebuilt it.

The generated source-mode marker was inspected with these values:

```json
{
  "schema_version": 1,
  "node_version": "22.13.0",
  "build_command": "npm run build"
}
```

The fingerprint implementation uses sorted normalized repository-relative paths
and SHA-256 content digests. It excludes test specs, test configuration,
preview/dev-server scripts, `.env`, backend Python, documentation, and QA
files. The marker is written through an atomic replacement after matching
pre-build and post-build fingerprints. Package manifest, lockfile, or pinned
Node changes are classified as dependency-input changes and request lockfile
synchronization before a rebuild. Desktop-release output does not require the
source-mode marker.

## Runtime and application checks

- Warm runtime readiness passed without `uv sync` or frontend dependency work.
- Initial missing-marker source launch completed after backend/runtime repair,
  frontend dependency synchronization, and Angular build.
- Final ordinary source launch returned `/api/health` HTTP `200` with
  `{"status":"ok"}`.
- Final frontend root returned HTTP `200` with `text/html; charset=utf-8`.
- Existing `app/resources/database.db` remained intact: 74,063,872 bytes,
  SQLite `pragma integrity_check=ok`, and 18 persisted `clinical_sessions`.
- Fresh/disposable SQLite migration and seeding paths passed in the required
  unit suite, including `test_sqlite_startup_initializes_and_seeds_when_database_file_is_missing`
  and `test_fresh_sqlite_database_reaches_head_and_is_idempotent`.

## Automated gates

- `app\\tests\\run_tests.bat unit`: PASS, 748 passed, 7 existing deprecation
  warnings, 42.72 seconds.
- `npm run test -- --no-watch`: PASS, 23 files and 96 tests.
- `npm run build -- --progress=false`: PASS.
- `app/server/.venv/Scripts/python.exe -m pytest -q app/tests/unit/test_launcher_contract.py`:
  PASS, 6 tests.
- PowerShell parser validation of `start_on_windows.ps1`: PASS.
- `git diff --check`: PASS; only normal LF-to-CRLF working-copy warnings were
  emitted by Git.

## Remaining boundaries

This evidence does not claim a browser-rendered UI audit, packaged desktop
startup, clean-machine installation, or an independently forced access-denied
termination. Those remain separate validation gates. No provider credentials,
PHI, or secret-bearing command lines were recorded.
