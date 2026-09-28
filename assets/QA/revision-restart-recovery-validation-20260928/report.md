# Revision restart and tool-failure validation

Date: 2026-09-28 (Europe/Rome)

## Selected scope

The current validation ledger identified `revision.agentic-lifecycle` as the
next actionable local slice. Provider acceptance remains dependent on an
approved OpenCode Go secret or a completing local model lane, so this follow-up
covered the adjacent failure boundaries that do not require credentials:

- startup reconciliation after a backend process restart;
- terminalization of persisted active revision steps when their in-memory
  worker is gone;
- retry from the recovered persisted run without replacing the source report;
- an injected revision-tool failure with sanitized persisted diagnostics; and
- the application startup ordering that runs reconciliation after the database
  and job manager are ready, before provider initialization.

The RAG duplicate-file policy, live OpenCode Go acceptance, access-key
material, NCBI source refresh, desktop publication, and spoken screen-reader
output were reviewed as independent gates and were not promoted by this slice.

## Implementation change

The ledger and architecture documentation described startup reconciliation, but
the application lifespan did not invoke it. A missing revision worker was only
marked recoverable when a later status poll happened, and active persisted
steps could remain `running`.

The current implementation now:

1. runs `reconcile_interrupted_revision_jobs` during application startup after
   database initialization and job-manager startup;
2. finds persisted `running` revision runs whose configured job id has no live
   pending/running worker;
3. marks those runs and any still-active revision steps `failed` with the
   existing sanitized recoverable message; and
4. leaves the source session and draft shell intact so the persisted run can be
   retried under a new job/run identity.

## Evidence

### Process-restart recovery

A disposable SQLite database contained a running revision run and a running
planner step. A fresh `JobManager` represented the post-restart process. The
startup reconciliation path returned the orphaned run, persisted the run and
step as terminal `failed`, preserved the source report, and allowed the retry
route to complete a deterministic accepted synthetic revision as
`llm_qa_passed` under a new identity. No provider request was made.

### Tool failure

The allow-listed `read_session_context` tool was replaced in the test with a
controlled exception containing a secret-shaped token. The revision job failed
closed, persisted the generic retry message at the job/run boundary, recorded a
failed task step without the token, and left the source report unchanged.

### Test and quality results

- Focused revision/startup suite: **37 passed**, 1 existing dependency
  deprecation warning.
- Affected migration/job/API/cloud-error checks: **41 passed**, 1 existing
  dependency deprecation warning.
- Full backend unit suite: **811 passed**, 7 existing dependency/deprecation
  warnings.
- Ruff: passed for all changed backend and test files.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- Python compilation: passed for changed backend files.
- `git diff --check`: passed.

All databases, pytest basetemp/cache paths, and bytecode used for this run
were isolated under this QA directory. No shared database, credential, or
provider cache was mutated.

## Gate status after this slice

| Gate | Final status | Boundary after this validation |
|---|---|---|
| `revision.agentic-lifecycle` | `PARTIAL` | Startup worker-loss recovery, active-step terminalization, sanitized tool failure, source preservation, and deterministic retry are validated. Fresh accepted current-provider execution, broader model variance, timeout/provider matrix beyond existing coverage, and every lifecycle branch remain open. |
| `api.local-boundaries` | `WORKING` | Revision startup/status/retry persistence and error contracts are covered; the complete 68-path response/error and mutating-route matrix remains outside scope. |
| `test.automated-regression` | `PARTIAL` | The full local backend unit suite and affected static checks are green. Frontend, hosted live-provider, conditional browser, and hosted exact-commit lanes remain separate. |
| `model.provider.opencode-go` | `PARTIAL` | No approved current key was available, so no cloud request was made. Historical exact-provider evidence remains bounded. |
| `auth.access-key-management` | `BLOCKED` | No approved disposable credential was available or mutated. |
| `data.inspection.catalogs` / `data.sources.refresh` | `PARTIAL` | NCBI Bookshelf still requires human verification for the LiverTox master list. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged, hosted packaging, clean-machine, and publication prerequisites remain separate. |
| spoken Narrator/Speech Recap output | `UNVALIDATED` | Rendered, keyboard, and accessibility-tree evidence still does not establish audible output. |

The existing validated boundaries for source startup, SQLite migrations, the
application shell, runtime/model settings, clinical workflow, local Ollama
timeline, RAG edge behavior, session persistence, and accepted synthetic
revision finalization remain unchanged. `ISSUE-005` duplicate-file policy
remains open pending a product decision.
