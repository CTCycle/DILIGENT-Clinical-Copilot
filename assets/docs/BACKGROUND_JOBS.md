# Background Job Management

Last updated: 2026-05-21

DILIGENT uses a centralized thread-based job manager for long-running operations.

## 1. Core components

- Manager: `app/server/services/runtime/jobs.py`
- Shared in-process access point: `get_job_manager()`
- API models: `app/server/domain/jobs.py`

## 2. Job state contract

Each job tracks:
- `job_id` (short UUID prefix)
- `job_type`
- `status`: `pending` | `running` | `completed` | `failed` | `cancelled`
- `progress` (0-100)
- `result` (optional JSON payload)
- `error` (sanitized user-safe message)
- `created_at`, `completed_at`
- `version` (monotonic state version incremented on every state/result/progress mutation)
- `stop_requested` (cooperative cancellation flag)

## 3. Execution behavior

- Jobs run in daemon threads.
- `start_job` can auto-inject `job_id` into runners.
- `update_result` merges interim patches with final payload.
- Unhandled runner exceptions mark job `failed` unless cancellation was requested.
- Detailed exception traces remain server logs only.

## 4. Active job types and endpoints

- `clinical`
  - Preflight: `POST /api/clinical/validate-input`
  - Start: `POST /api/clinical/jobs`
  - Poll/cancel: `GET|DELETE /api/clinical/jobs/{job_id}`
- `ollama_pull`
  - Start: `POST /api/models/pull/jobs`
  - Poll/cancel: `GET|DELETE /api/models/jobs/{job_id}`
- `rxnav_update`
  - Start: `POST /api/inspection/rxnav/jobs`
  - Poll/cancel: `GET|DELETE /api/inspection/rxnav/jobs/{job_id}`
- `livertox_update`
  - Start: `POST /api/inspection/livertox/jobs`
  - Poll/cancel: `GET|DELETE /api/inspection/livertox/jobs/{job_id}`
- `rag_update`
  - Start: `POST /api/inspection/rag/jobs`
  - Poll: `GET /api/inspection/rag/jobs/{job_id}`
  - Cancel: `POST /api/inspection/rag/jobs/{job_id}/cancel`
- `session_revision`
  - Start: `POST /api/inspection/sessions/{session_id}/revision/jobs`
  - Poll/cancel: `GET|DELETE /api/inspection/sessions/revision/jobs/{job_id}`

## 5. Polling pattern

Standard contract:
1. Start endpoint returns `JobStartResponse` (`job_id`, `status`, `poll_interval`).
2. Status endpoint returns `JobStatusResponse`.
   - Clinical status responses are explicitly non-cacheable (`Cache-Control: no-store` headers).
   - Clients should treat `version` as monotonic and ignore out-of-order older snapshots.
3. Cancel endpoint returns `JobCancelResponse`.
4. Inspection update jobs may include phase-aware result fields:
   - `phase`, `step_index`, `step_count`, `progress_message`, `summary`.
5. Inspection updater runners use cooperative cancellation (`should_stop`) and progress callbacks consistently across `rxnav`, `livertox`, and `rag`.
6. Session revision jobs report clinical pipeline progress, reprocess the full persisted session text, create a new session version, and persist a `revision_audit` payload with parser cross-validation, selected-focus context, user revision instruction, detected-drug diffs, model overrides, and conclusion action metadata.

Frontend polling is implemented in `app/client/src/app/core/services/api.ts` and stops on terminal states.

## 6. Cancellation rules

Cancellation is cooperative:
- Pending jobs can be set to `cancelled` immediately.
- Running jobs receive `stop_requested=True`.
- Runner code must check `get_job_manager().should_stop(job_id)` or its injected `JobManager` at safe checkpoints.

If a runner does not check stop requests, cancellation is delayed.

Current inspection cancellation/progress checkpoints are implemented by the RxNav/RxNorm, LiverTox, and RAG update runners.

Clinical jobs run input preflight before job creation. The completed clinical job result includes database-backed evidence-lock artifacts and gate fields such as `manual_review_required`, `blocking_issues`, `pipeline_artifacts`, and `run_bundle_index`. These artifacts are persisted through the clinical session result payload rather than written as loose files.

## 7. New job implementation checklist

1. Add runner function returning `dict[str, Any]`.
2. Check `get_job_manager().should_stop(job_id)` or an injected `JobManager` during long steps.
3. Publish interim progress/result updates.
4. Expose start/poll/cancel routes.
5. Prevent conflicting duplicates where needed (`is_job_running(job_type)`).
