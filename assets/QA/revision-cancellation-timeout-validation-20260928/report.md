# Revision cancellation and timeout validation

Last updated: 2026-09-28

Date: 2026-09-28 (Europe/Rome)
Baseline: `8bda9cf1` (`develop`, equal to `origin/develop` before this
evidence/test change)

## Scope and slice selection

The canonical ledger identified `revision.agentic-lifecycle` as the next
locally actionable gate after startup reconciliation, orphaned-worker recovery,
and the unavailable/slow local-Ollama attempts. This follow-up covered the
adjacent lifecycle boundaries that do not require an approved provider secret:

- public `DataInspectionService.cancel_revision_job` behavior while a revision
  worker is still executing;
- same-root admission during the cancellation unwind and retry after the
  worker exits;
- terminalization of the active planner step and draft version; and
- persisted, retryable, sanitized metadata for a provider timeout.

OpenCode Go acceptance, approved access-key mutation, NCBI LiverTox refresh,
duplicate-file policy, spoken screen-reader output, desktop publication, and
container deployment remain independent gates. No provider request or
credential mutation was made by this slice.

## Current implementation validation

`test_service_cancellation_preserves_source_and_allows_retry_after_worker_exit`
held the deterministic planner call at its provider boundary, cancelled the
revision through the service, and verified that:

1. the in-memory job remained `running` with `stop_requested=true` until the
   worker could observe the cooperative stop;
2. the persisted revision run was immediately marked `cancelled`, while a
   retry remained blocked by the same-root running scope;
3. releasing the planner terminalized the job and planner step as `cancelled`
   and left the draft version detached from a clinical session;
4. a retry created a new pipeline/job identity and completed its deterministic
   no-op path as `requires_human_review`; and
5. the source report was unchanged before and after the retry.

`test_revision_timeout_persists_retryable_provider_metadata` raised the real
`LLMTimeout` contract for `ollama / qwen3.5:9b`. The persisted run and planner
step retained the exact provider, model, operation, `error_code=timeout`, and
`retryable=true`, while the source report remained unchanged. No raw provider
detail or secret-shaped value was persisted.

No application defect was found in this boundary, so no product source change
was required. The two tests are retained as regression coverage for the public
service cancellation path and timeout persistence contract.

## Test and quality evidence

| Check | Result |
|---|---|
| Revision, runtime-job, startup, cloud-error, API-contract, and clinical-route regression | **79 passed**, 2 existing dependency/deprecation warnings |
| Revision-agent file after adding the two regression checks | **50 passed** within the selected revision suite |
| Targeted Ruff on revision source, repository, recovery/scaffold, and revision tests | Passed |
| Pyright from `app/server` using the repository configuration | **0 errors, 0 warnings, 0 informations** |
| `git diff --check` | Passed |

A broad repository Ruff scan was not used as a clean gate: it reported 150
pre-existing findings in unrelated tests, along with protected-cache access
warnings. No unrelated lint cleanup was folded into this slice.

## Gate disposition after this slice

| Gate | Final status | Evidence boundary and remaining limitation |
|---|---|---|
| `revision.agentic-lifecycle` | `PARTIAL` | Service cancellation, cooperative worker unwind, same-root admission, retry after cancellation, timeout diagnostics, source preservation, startup recovery, tool failure, unavailable-provider failure, and deterministic retry boundaries are validated. Fresh accepted current-provider execution, broader provider/model variance, and every lifecycle branch remain incomplete. |
| `revision.accepted-session-finalization` | `VALIDATED` for the exact previous synthetic accepted path | This slice deliberately cancelled before finalization and did not create a new accepted child session. The prior accepted record remains bounded to its recorded provider/model and review state. |
| `api.local-boundaries` | `WORKING` for the exercised revision cancellation/status/retry boundary | The service-level cancellation and retry contracts passed; the complete 68-path response/error and mutating-route matrix remains outside scope. |
| `test.automated-regression` | `PARTIAL` | The selected local backend/API/static checks are green. Hosted live-provider, conditional browser, and exact-commit hosted lanes remain incomplete. |
| `model.provider.opencode-go` | `PARTIAL` | No approved current key was available and no cloud request was made. Historical exact-provider evidence remains bounded. |
| `model.provider.local-ollama` | `VALIDATED` for the existing timeline scope; revision scope remains partial | The timeout contract is validated without changing the supported timeline boundary. No accepted local revision was produced. |
| `auth.access-key-management` | `BLOCKED` | No approved disposable credential was available or mutated. |
| `data.inspection.catalogs` / `data.sources.refresh` | `PARTIAL` | The NCBI Bookshelf human-verification blocker remains. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged, hosted packaging, clean-machine, and publication prerequisites remain separate. |
| Spoken Narrator/Speech Recap output | `UNVALIDATED` | Rendered, keyboard, and accessibility-tree evidence does not establish audible output. |

## Cleanup and remaining action

The tests used disposable SQLite databases and the repository’s isolated pytest
cache root. No shared database, provider cache, credential, application
server, browser, or long-running helper process was left running. Generated
Python bytecode and the task-local scratch cache were removed after validation.

The next revision action remains a fresh non-dry accepted or fail-closed run
through an approved hosted/injected provider lane or a completing explicitly
configured local model, followed by broader provider/retry variance. The
unchanged external blockers above must not be inferred as passed from this
credential-free lifecycle slice.
