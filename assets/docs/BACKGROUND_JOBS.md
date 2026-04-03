# Background Job Management

Last updated: 2026-04-03

DILIGENT uses a centralized thread-based job manager for long-running operations.

## 1. Core components

- Manager: `DILIGENT/server/services/jobs.py`
- Shared singleton: `job_manager`
- API models: `DILIGENT/server/domain/jobs.py`

## 2. Job state contract

Each job tracks:
- `job_id` (short UUID prefix)
- `job_type`
- `status`: `pending` | `running` | `completed` | `failed` | `cancelled`
- `progress` (0-100)
- `result` (optional JSON payload)
- `error` (sanitized user-safe message)
- `created_at`, `completed_at`
- `stop_requested` (cooperative cancellation flag)

## 3. Execution behavior

- Jobs run in daemon threads.
- `start_job` can auto-inject `job_id` into runners.
- `update_result` merges interim patches with final payload.
- Unhandled runner exceptions mark job `failed` unless cancellation was requested.
- Detailed exception traces remain server logs only.

## 4. Active job types and endpoints

- `clinical`
  - Start: `POST /clinical/jobs`
  - Poll/cancel: `GET|DELETE /clinical/jobs/{job_id}`
- `ollama_pull`
  - Start: `POST /models/pull/jobs`
  - Poll/cancel: `GET|DELETE /models/jobs/{job_id}`
- `rxnav_update`
  - Start: `POST /inspection/rxnav/jobs`
  - Poll/cancel: `GET|DELETE /inspection/rxnav/jobs/{job_id}`
- `livertox_update`
  - Start: `POST /inspection/livertox/jobs`
  - Poll/cancel: `GET|DELETE /inspection/livertox/jobs/{job_id}`
- `rag_update`
  - Start: `POST /inspection/rag/jobs`
  - Poll: `GET /inspection/rag/jobs/{job_id}`
  - Cancel: `POST /inspection/rag/jobs/{job_id}/cancel`

Equivalent `/api/*` prefixed routes are always available.

## 5. Polling pattern

Standard contract:
1. Start endpoint returns `JobStartResponse` (`job_id`, `status`, `poll_interval`).
2. Status endpoint returns `JobStatusResponse`.
3. Cancel endpoint returns `JobCancelResponse`.
4. Inspection update jobs may include phase-aware result fields:
   - `phase`, `step_index`, `step_count`, `progress_message`, `summary`.

Frontend polling is implemented in `DILIGENT/client/src/services/api.ts` and stops on terminal states.

## 6. Cancellation rules

Cancellation is cooperative:
- Pending jobs can be set to `cancelled` immediately.
- Running jobs receive `stop_requested=True`.
- Runner code must check `job_manager.should_stop(job_id)` at safe checkpoints.

If a runner does not check stop requests, cancellation is delayed.

## 7. New job implementation checklist

1. Add runner function returning `dict[str, Any]`.
2. Check `job_manager.should_stop(job_id)` during long steps.
3. Publish interim progress/result updates.
4. Expose start/poll/cancel routes.
5. Prevent conflicting duplicates where needed (`is_job_running(job_type)`).
