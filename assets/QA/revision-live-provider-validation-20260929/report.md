# OpenCode Go live revision validation — 2026-09-29

## Result

Status: **INFRASTRUCTURE-BLOCKED; gates remain PARTIAL**.

This report covers the combined live-provider and fresh agentic-revision slice
implemented at commit `37691d5512245a6bde29e8a125683b40046cbcc5` on `develop`.
The hosted workflow was dispatched with `run_provider_e2e=true` at [Actions run
36568880619](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36568880619).
The core hosted jobs passed, but the [live-provider job
failed](https://github.com/CTCycle/DILIGENT-Clinical-Copilot/actions/runs/36568880619/job/109407982389)
at the existing credential preflight because `OPENCODE_GO_API_KEY` was empty in
the runner environment. No cloud request was made, so this run cannot promote
either live-provider or fresh-live-revision evidence.

The exact intended route for this slice is provider `opencode_go` and model
`deepseek-v4-flash`. No credential value is recorded anywhere in this report or
the inspected hosted output.

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

The hosted job showed the expected opt-in flag, isolated SQLite paths, and
secret-expression wiring, but the resolved `OPENCODE_GO_API_KEY` environment
value was blank. The test stopped at its explicit guard before access-key
creation, connectivity, the clinical request, or revision launch. Therefore
there is no hosted evidence in this run for provider connectivity, actual
OpenCode Go model execution, revision steps, artifacts, QA, retry, or lineage.

The authenticated local `gh secret list` query returned no repository secret
names. No secret value was read, inferred, copied, or persisted. The missing
secret is an environment/configuration boundary, not evidence of a provider
transport or revision-product defect.

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
tool-failure, and retry evidence remains the supporting boundary in the
[revision restart/tool-failure report](../revision-restart-recovery-validation-20260928/report.md),
[revision cancellation/timeout report](../revision-cancellation-timeout-validation-20260928/report.md),
and [revision lifecycle report](../revision-lifecycle-validation-20260927/report.md).
Those cases were not redundantly replayed against a paid external provider.

## Gate disposition

| Gate | Disposition for this slice | Evidence boundary |
|---|---|---|
| `model.provider.opencode-go` | `PARTIAL` — infrastructure-blocked | The current test asserts exact `opencode_go / deepseek-v4-flash` configuration and fail-closed secret handling, but the hosted run made no provider request. Historical exact-route records remain supporting evidence only. |
| `revision.agentic-lifecycle` | `PARTIAL` — fresh live slice not executed | Deterministic persistence, cancellation, timeout, restart recovery, tool failure, retry, accepted finalization, and UI evidence remain valid for their documented scopes. This run adds no fresh live revision outcome because the provider preflight stopped first. |

Outcome categories for this run:

- Accepted: **not observed**.
- QA-blocked: **not observed**.
- Retryable provider failure: **not observed**; the retry branch was correctly
  not manufactured after the credential preflight failure.
- Infrastructure failure: **observed** — the approved repository secret was
  unavailable/empty on the hosted runner.

## Required follow-up

Configure the approved repository secret `OPENCODE_GO_API_KEY` without placing
its value in source, logs, artifacts, or this ledger, then rerun the existing
workflow with `run_provider_e2e=true` on this commit or a later commit. Promote
the gates only if that fresh hosted run reaches the revision request and
demonstrates the complete persisted lifecycle; otherwise retain the precise
terminal outcome as accepted, QA-blocked, retryable failure, or infrastructure
failure.
