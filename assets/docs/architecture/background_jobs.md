# Background Jobs
Last updated: 2026-06-25

## Scope
DILIGENT uses a centralized thread-based job manager for long-running operations.

## Core Components
- Manager: `app/server/services/runtime/jobs.py`
- Shared in-process access point: `get_job_manager()`
- API models: `app/server/domain/jobs.py`
- Runtime state: `app/server/services/runtime/state.py`

## Job State Contract
Each job tracks:
- `job_id`
- `job_type`
- `status`: `pending`, `running`, `completed`, `failed`, `cancelled`
- `progress`
- `result`
- `error`
- `created_at`
- `completed_at`
- `version`
- `stop_requested`

## Execution Behavior
- Jobs run in daemon threads.
- Job execution state is process-local. Clinical and update job ids are not durable across backend restarts.
- Revision runs are persisted separately from the in-memory job state; startup reconciliation marks stale `running` revision runs as failed/recoverable when no in-process worker can own them.
- `start_job` can auto-inject `job_id` into runners.
- `update_result` merges interim patches with final payload.
- Unhandled runner exceptions mark the job `failed` unless cancellation was requested.
- Detailed exception traces remain in server logs only.

## Active Job Types
- `clinical`
  - Preflight: `POST /api/clinical/validate-input`
  - Start: `POST /api/clinical/jobs`
  - Poll or cancel: `GET|DELETE /api/clinical/jobs/{job_id}`
- `ollama_pull`
  - Start: `POST /api/models/pull/jobs`
  - Poll or cancel: `GET|DELETE /api/models/jobs/{job_id}`
- `rxnav_update`
  - Start: `POST /api/inspection/rxnav/jobs`
  - Poll or cancel: `GET|DELETE /api/inspection/rxnav/jobs/{job_id}`
- `livertox_update`
  - Start: `POST /api/inspection/livertox/jobs`
  - Poll or cancel: `GET|DELETE /api/inspection/livertox/jobs/{job_id}`
- `rag_update`
  - Start: `POST /api/inspection/rag/jobs`
  - Poll or cancel: `GET|DELETE /api/inspection/rag/jobs/{job_id}`

## Polling Contract
1. Start endpoints return `JobStartResponse` with `job_id`, `status`, and `poll_interval`.
2. Status endpoints return `JobStatusResponse`.
3. Cancel endpoints return `JobCancelResponse`.

Additional rules:
- Clinical status responses are explicitly non-cacheable through `Cache-Control: no-store`.
- Clients should treat `version` as monotonic and ignore out-of-order older snapshots.
- Clinical progress snapshots expose canonical granular stage keys such as `drugs.extracting`, `retrieval.evidence`, `report.generating`, `session.saving`, and terminal `completed`; generic internal wrapper stages are not persisted as the user-facing stage.
- Inspection update jobs may include `phase`, `step_index`, `step_count`, `progress_message`, and `summary`.
- Inspection update runners use cooperative cancellation and progress callbacks consistently across `rxnav`, `livertox`, and `rag`.
- Session revision jobs reprocess the persisted session text, create a new session version, and persist a `revision_audit` payload with parser cross-validation, selected-focus context, user instructions, detected-drug diffs, model overrides, and conclusion action metadata.
- Missing in-memory revision job status returns a recoverable failed status instead of a bare not-found response so the frontend can reload the persisted revision run and offer retry when the draft shell is still valid.
- Clinical and session revision jobs must persist their successful result payloads before returning completion. Persistence failures move the job to failed state with sanitized error metadata.

Frontend polling is implemented through app-lifetime tracker services such as
`app/client/src/app/core/services/dili-job-tracker.service.ts` and stops on
terminal states. Active DILI job linkage is persisted in session storage so the
UI can reattach after route navigation or a same-session page refresh while the
backend worker is still running.

## Cancellation Rules
- Pending jobs can be marked `cancelled` immediately.
- Running jobs receive `stop_requested=True` and remain `running` until the worker reaches a terminal transition.
- Runner code must check `get_job_manager().should_stop(job_id)` or an injected `JobManager` at safe checkpoints.

If a runner does not check stop requests, cancellation is delayed.

## Clinical Job Notes
- Clinical jobs run input preflight before job creation.
- Clinical assessment concurrency is explicitly scoped as `clinical:global`.
- Revision concurrency is scoped per root clinical session; a second revision for the same root session conflicts, while unrelated root sessions can follow their configured policy independently.
- Catalog update concurrency is scoped per catalog target.
- Completed results include database-backed evidence-lock artifacts and gate fields such as `manual_review_required`, `blocking_issues`, `pipeline_artifacts`, and `run_bundle_index`.
- These artifacts are persisted through the clinical session result payload rather than loose files.
- Failed clinical results omit raw anamnesis, drug text, laboratory text, and patient image base64 content. Failure payloads expose generic error text plus `failure_metadata` for diagnostics.
- DILI sessions perform deterministic section validation before clinical extraction and may parallelize parser-model extraction only when parser batch preflight marks concurrency as safe; otherwise the workflow falls back to sequential execution.

## New Job Checklist
1. Add a runner function returning `dict[str, Any]`.
2. Check `should_stop(job_id)` during long steps.
3. Publish interim progress or result updates.
4. Expose start, poll, and cancel routes.
5. Prevent conflicting duplicates where needed with `is_job_running(job_type)`.

## Disabled Scaffolds
- Session revision routes remain registered only as rewrite scaffolding and return 501 Not Implemented until a replacement workflow is implemented.

