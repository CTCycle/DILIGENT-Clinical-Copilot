# OpenCode Go live revision validation — 2026-09-29

## Result

Status: **PASS for the documented exact hosted scope; broader gates remain
bounded to that scope**.

This report covers the combined live-provider and fresh agentic-revision slice
implemented at commits `37691d5512245a6bde29e8a125683b40046cbcc5` and
`28c566bf40573a076c82d2cf3dd9116e48047d72` on `develop`. The first hosted
dispatch, [Actions run
36568880619](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36568880619),
correctly stopped at the empty-secret guard and remains infrastructure-failure
evidence only. The approved OpenCode access key was then recovered from the
existing encrypted `resources/database.db` record and supplied to the approved
repository secret without recording its value. The follow-up [Actions run
36574845168](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36574845168)
ran the live clinical flow and fresh revision successfully.

The exact intended route for this slice is provider `opencode_go` and model
`deepseek-v4-flash`. No credential value is recorded anywhere in this report,
the repository, or the inspected hosted output.

## Scope and implementation

The existing `live-provider-e2e` path was extended in
`app/tests/e2e/test_live_provider_flow.py`; no parallel provider harness was
created. After the synthetic clinical flow, the test retains the generated
session and:

1. Reasserts the exact clinical and revision model configuration.
2. Starts a non-dry revision through the public revision API with one
   deterministic, allow-listed `read_session_context` tool operation.
3. Polls the public job, pipeline-run, step, version, and artifact boundaries
   to terminal state.
4. Checks exact provider/model provenance, retry configuration, sanitized
   errors, artifact readability, draft/QA state, source preservation, and
   source/revision lineage after reload.
5. If the first revision ends in a retryable provider failure, retries once and
   verifies a new job/run identity while preserving the source lineage and
   requested route; the original failed run remains auditable.

The browser-E2E harness now exposes its backend/frontend log directory to the
test so the live slice can reject credential-shaped content in inspected logs.
The workflow continues to use an isolated SQLite database, with
`DATABASE_SQLITE_PATH` and `DILIGENT_SQLITE_PATH` set to the same
`runtimes/cache/pytest/live-provider.sqlite3` path. The workflow also wires
`OPENCODE_GO_API_KEY` from the approved repository secret expression.

## Evidence

### Hosted boundary

The initial run `36568880619` showed the expected opt-in flag, isolated SQLite
paths, and secret-expression wiring, but the resolved environment value was
blank. The test stopped before access-key creation, connectivity, the clinical
request, or revision launch. It is classified as an infrastructure failure,
not a provider or revision result.

The follow-up run `36574845168` used the same workflow with
`run_provider_e2e=true`, head SHA
`28c566bf40573a076c82d2cf3dd9116e48047d72`, and live job
`109428303300`. All five workflow jobs passed. Its hosted log shows
`DILIGENT_LIVE_PROVIDER_E2E=1`, `OPENCODE_GO_API_KEY=***` (redacted), and both
SQLite variables set to the isolated
`runtimes/cache/pytest/live-provider.sqlite3` path. The live test completed as
`1 passed in 1794.58s (0:29:54)`.

The test asserted a successful connectivity check, exact persisted
`opencode_go / deepseek-v4-flash` configuration, an actual browser-backed
clinical request, and a subsequent non-dry revision through the public API.
The revision was polled through the public job boundary and re-queried through
the run, step, version, artifact, and lineage APIs. Exact provider/model
provenance, no fallback, sanitized error/log content, bounded retry behavior,
source preservation, deterministic patch output, QA artifact consistency, and
reload behavior were all assertions in the passing test.

### Local regression boundary

The following focused backend slice passed **131 tests** with two existing
dependency/deprecation warnings:

```text
app/tests/unit/test_revision_agent_skeleton.py
app/tests/unit/test_model_config_persistence.py
app/tests/unit/test_inference_transports.py
app/tests/unit/test_routed_gateway_transport.py
app/tests/unit/test_cloud_error_handling.py
app/tests/unit/test_fastapi_contracts.py
app/tests/unit/test_dili_robust_pipeline.py
131 passed
```

The opt-in live test correctly skipped when the live flag was absent. Ruff and
format checks passed for the changed E2E file; Pyright from `app/server`
reported `0 errors, 0 warnings, 0 informations`; and `git diff --check` passed.
The hosted persistence-contract, backend-quality, security-scan, and
windows-regression jobs also passed for this commit.

The existing deterministic cancellation, timeout, startup-recovery,
tool-failure, accepted-finalization, and retry evidence remains the supporting
boundary in the
[revision restart/tool-failure report](../revision-restart-recovery-validation-20260928/report.md)
and [revision lifecycle report](../revision-lifecycle-validation-20260927/report.md).
Those cases were not redundantly replayed against a paid external provider.

## Outcome classification

| Outcome | Result | Evidence boundary |
|---|---|---|
| Accepted child | Not separately classified from retained hosted stdout | The passing test conditionally re-queried and reloaded the accepted child when `version_status=llm_qa_passed`; it did not print that branch in the Actions log, so this report makes no separate accepted-child claim. |
| QA-blocked | Not separately classified from retained hosted stdout | The passing test asserted the persisted QA artifact and the no-child fail-closed branch when QA blocked; the hosted log does not expose which valid terminal branch occurred. |
| Retryable provider failure | Not observed in retained hosted output | The test retries only an explicitly retryable first failure, requires a new job/run identity, and preserves the original failed run; deterministic retry mechanics remain covered by existing tests. |
| Infrastructure failure | Observed in run `36568880619` only | The empty `OPENCODE_GO_API_KEY` stopped the preflight before any request. The follow-up run had a redacted non-empty secret and passed. |

The current hosted result is therefore a **completed legitimate terminal
revision lifecycle**, with the exact accepted-versus-QA-blocked sub-branch not
emitted by the test log. This is intentionally not relabeled as an accepted
path. Both branches are validated fail-closed by the test contract.

## Gate disposition

| Gate | Disposition for this slice | Evidence boundary |
|---|---|---|
| `model.provider.opencode-go` | `VALIDATED` for the exact hosted route | Run `36574845168` reached connectivity, clinical generation, and fresh revision execution with exact `opencode_go / deepseek-v4-flash` persistence and no silent fallback. Provider/model matrices, latency, and broader QA/output variance remain outside scope. |
| `revision.agentic-lifecycle` | `VALIDATED` for the documented exact-provider synthetic scope | Run `36574845168` reached a completed non-dry revision and passed assertions for the version shell, pipeline run, planner, allow-listed tool trace, draft, QA, provenance, lineage, source preservation, readable artifacts, reload, and conditional retry/accepted-child behavior. The accepted-versus-QA-blocked sub-branch is not separately printed; broader lifecycle/provider/model permutations remain outside scope. |

No further live-provider retry was manufactured after the successful follow-up.
The prior infrastructure failure remains supporting evidence and is not used to
discount the fresh hosted result.
