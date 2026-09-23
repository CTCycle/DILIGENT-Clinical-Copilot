# Automated regression gate revalidation
Last updated: 2026-09-23

## Scope

Rechecked `test.automated-regression` from clean `develop` HEAD `f61a9ff6`,
which matched `origin/develop` at task start. The prior exact-SHA hosted run
[35895672146](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35895672146)
reported Alembic drift, Python dependency audit findings, and a browser E2E
failure.

All runtime and test data used disposable SQLite databases under this QA
folder. No shared database, saved model settings, credentials, or live
provider state was changed.

## Findings and changes

- Hosted `pip-audit` had reported `CVE-2026-63374` and `CVE-2026-64847` in
  `anyio==4.9.0`, with `4.14.2` as the fixed version. The direct pin and
  `app/server/uv.lock` were updated to `4.14.2`.
- The hosted browser failure reproduced locally in
  `test_dili_running_job_state_is_not_restored_after_refresh`. The test
  expected a running clinical job to disappear after reload, but the current
  `DiliJobTrackerService` deliberately persists the job ID in
  `dili-agent-active-job-v1` and reconnects to it. The existing service unit
  spec already covered that behavior. The browser test now checks that the
  same job resumes, completes without a second submission, renders its
  report, and clears the saved marker.
- Hosted Alembic drift did not reproduce locally against a fresh SQLite
  database: upgrade, head verification, and metadata comparison all passed
  at `202609170001`.
- The first hosted rerun then exposed a CI setup defect: the configured
  SQLite database was under `runtimes/cache/pytest/ci`, but the workflow did
  not create that parent directory. The Alembic process therefore failed
  before it could open the database. CI now creates the directory first; a
  local simulation starting from a missing parent passed `upgrade head`,
  `current --check-heads`, and `check`.
- That rerun also exposed two Pyright errors where the timeline and session
  cancellation helpers passed a generic `Awaitable` directly to
  `asyncio.create_task`. Both now schedule a typed coroutine wrapper. Local
  Pyright passed with zero errors, and both focused cancellation regressions
  passed: `test_stage_stop_check_cancels_inflight_extraction_task` and
  `test_timeline_job_cancellation_stops_extraction_before_persistence`.

## Validation results

| Gate | Result |
|---|---|
| Locked environment | `uv sync --locked --project app/server --all-extras` installed `anyio 4.14.2` into the existing server environment. |
| Python dependency audit | The CI-equivalent locked export and strict `pip-audit` run completed with `No known vulnerabilities found`. |
| Fresh SQLite / Alembic | `upgrade head`, `current --check-heads`, and `check` passed; Alembic reported no new upgrade operations. |
| Full backend unit suite | `789 passed`; 7 existing deprecation warnings. |
| Model-config unit slice | `40 passed`; one Google GenAI deprecation warning. |
| Focused browser recovery | The updated refresh-recovery test passed. |
| Full browser E2E suite | `41 passed, 5 skipped`. The initial run before the stale assertion was corrected had `40 passed, 5 skipped, 1 failed`. |
| In-app Browser shell check | DILI Agent title, primary navigation, clinical input, and empty report state rendered. No analysis was submitted. |
| Exact-SHA hosted CI | Run [35908957190](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/35908957190) completed successfully for `3e73082f619024140cfdb92340831f450c6a3e13`. Backend quality passed, including Alembic drift, Ruff, Pyright, and `789 passed`; security scans passed for Python, Angular, desktop, and Rust; SQLite/PostgreSQL persistence contract passed. |
| Hosted Windows regression | Frontend tests: `24 files, 105 passed`; Angular production build passed; model-config suite `40 passed, 1 warning`; browser E2E `39 passed, 7 skipped`. |

The official launcher verified its stored frontend fingerprint and reused
the current output. Its attempt to open the default browser returned
Windows `Access denied`; the in-app Browser then opened the local page and
rendered the shell. This run therefore does not claim a fresh local
production build. The previous local `0xC0000005` build failure remains
unresolved, although the hosted Windows build passed. The seven hosted E2E
skips were conditional: the live provider and pinned multilingual embedding
tests are opt-in; two model API tests require an available Ollama model; and
three session/timeline UI tests require persisted session data absent from
the isolated CI database. The separate live-provider workflow lane is also
skipped on push and requires an explicit dispatch plus repository secret.

## Final status

`runtime.database.sqlite-migrations` remains `VALIDATED` for the exercised
SQLite path, now confirmed by exact-SHA hosted CI. `test.automated-regression`
remains `PARTIAL`: all configured hosted jobs pass, but the fresh local
production-build crash is unresolved and the conditional provider,
embedding, Ollama, and persisted-session browser cases remain skipped. No
packaging, clean-machine, live-provider, access-key, or desktop release gate
is promoted by this report.

The disposable databases and pytest cache created for this run were removed.
Ports `7690` and `9847` were free after cleanup, and task-started backend and
frontend processes were stopped.
