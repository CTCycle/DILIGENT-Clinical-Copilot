# Revision local-Ollama validation

Last updated: 2026-09-28

Date: 2026-09-28
Baseline: `84ac223e` (`develop`, equal to `origin/develop` before this
working-tree change)

## Scope

This follow-up covered the next locally actionable revision slice:

- worker-start ordering and persisted recovery metadata;
- one fresh non-dry revision attempt with the currently discovered local
  Ollama model;
- persisted fail-closed provider failure and retry behavior; and
- source-report preservation across both failed attempts.

OpenCode Go, approved access-key mutation, NCBI LiverTox refresh, desktop
publication, and spoken screen-reader output remained independent gates.

All live revision work used disposable SQLite clones. The shared database,
settings, provider credentials, and canonical provider/model caches were not
mutated.

## Defect found and fixed

Before this change, `start_revision_job` started the background worker before
the worker's job identifier and final recovery configuration were persisted.
The worker could therefore race the caller's second SQLite write. The
reproduction hit `sqlite3.OperationalError: database is locked` while updating
the revision run configuration.

The fix:

1. allocates the job identifier before persistence;
2. stores that identifier and the revision version identifier in the version
   and pipeline-run configuration before worker launch;
3. lets `JobManager.start_job` accept the preallocated identifier and rejects
   duplicate identifiers; and
4. starts the worker only after the persisted recovery metadata is complete.

The source revision remains unchanged until a revision is finalized through
the existing accepted-session path.

## Current Ollama discovery

The direct local server catalog was rechecked after the model-update concern.
At approximately 10:54 Europe/Rome on 2026-09-28, `/api/tags` returned seven
models, including the exact `qwen3.5:9b` model used below:

- model: `qwen3.5:9b`
- parameter size: `9.7B`
- digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
- Ollama server: `0.34.0`

The model was selected from that live catalog rather than assumed from an old
ledger entry. If the local catalog changes later, this evidence does not
transfer to a different model name or digest.

## Automated evidence

| Check | Result |
|---|---|
| Revision-agent/persistence suite (`test_revision_agent_skeleton.py`) | **33 passed**, 1 dependency deprecation warning |
| Combined revision/runtime/API regression (`test_revision_agent_skeleton.py`, `test_runtime_jobs.py`, `test_dili_robust_pipeline.py`) | **53 passed**, 2 dependency/deprecation warnings |
| Ruff on changed Python source and test files | Passed |
| Pyright on changed Python source files | **0 errors, 0 warnings, 0 informations** |
| `git diff --check` | Passed; only normal LF/CRLF conversion warnings were reported |

`test_revision_job_persists_job_identity_before_worker_start` now reads both
persisted revision records before allowing the worker thread to start. It
asserts that the job identifier and revision version identifier are already
available for recovery.

## Isolated live-provider evidence

The disposable clone was configured with cloud services disabled and the
resolved Revision role `ollama / qwen3.5:9b`. The source report was:

> Possible drug-induced liver injury from amoxicillin. Clinical review is required.

The first fresh final-code attempt ran while the local service was unavailable:

| Attempt | Job | Pipeline run | Result |
|---|---|---|---|
| Fresh | `95e38fb4` | `178faacb7fbc4454a6f06907575c5a8c` | Fail-closed at planning: Ollama connection unavailable |
| Retry | `c80ac3d6` | `4a42a7d71e18471895ba53bf75b45900` | Same fail-closed unavailable-runtime boundary |

Both attempts persisted exact `ollama / qwen3.5:9b` metadata, retained the
`job_id` and `revision_version_id` recovery fields, persisted only the
`revision_agent_context` artifact before failure, and left the source report
unchanged.

The installed Ollama server was then started directly and the retry route was
executed again against the persisted failed run:

| Attempt | Job | Pipeline run | Version | Result |
|---|---|---|---:|---|
| Available-service retry | `cd7d4faa` | `ce5b911c275c438c9faade1fe31a4652` | 66 | Fail-closed at `revision_agent_planner` after the configured 45-second local inference cap |

The persisted failure was sanitized to the user-safe job message while the
revision step retained bounded diagnostics (`provider=ollama`,
`model=qwen3.5:9b`, operation `revision_agent`, no secret-bearing detail).
The run and version remained failed/draft, and the source report was unchanged.
The Ollama server and all task-owned disposable runtimes were stopped or
removed after capture.

## Final gate disposition

| Gate | Final status | Evidence boundary and remaining limitation |
|---|---|---|
| `revision.agentic-lifecycle` | `PARTIAL` | Worker-start ordering, persisted recovery metadata, fresh local-provider fail-closed behavior, sanitized timeout/unavailable errors, and retry creation are validated. No accepted revision was produced in this local model lane; timeout/tool-failure injection beyond provider timeout, true process restart, broader lifecycle branches, and current cloud-provider execution remain incomplete. |
| `revision.accepted-session-finalization` | `VALIDATED` for the previously accepted synthetic path only | This slice did not create a new accepted child session. The prior exact accepted evidence remains bounded to its recorded provider/model and authorized review state. |
| `api.local-boundaries` | `WORKING` for the revision start/status/retry persistence boundary | The selected start, status, retry, run, step, and version records were exercised; the complete 68-path response/error and mutating-route matrix remains outside scope. |
| `test.automated-regression` | `PARTIAL` | The affected backend/API regression is green. Frontend, hosted live-provider, and conditional browser lanes were not rerun because no frontend code changed and no OpenCode Go secret was injected. |
| `model.provider.local-ollama` | `VALIDATED` for the previously stated timeline scope; revision scope remains partial | The current catalog still contains `qwen3.5:9b`, but its revision planning call exceeded the 45-second local cap in this environment. This does not change the supported timeline status or establish local revision acceptance. |
| `model.provider.opencode-go` | `PARTIAL` | No OpenCode Go request was made; the local process had no approved `OPENCODE_GO_API_KEY`. |
| `auth.access-key-management` | `BLOCKED` | No approved disposable credential was available or mutated. |
| `data.sources.refresh` | `PARTIAL` | NCBI LiverTox remains behind the upstream human-verification challenge. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged/hosted/clean-machine publication evidence remains separate. |

## Cleanup

The temporary SQLite clones, task-owned Python cache, direct Ollama server,
and validation helpers were cleaned up. No application, test, browser, or
helper process from this slice was intentionally left running.
