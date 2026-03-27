# Background Job Management

DILIGENT uses a centralized thread-based job system for long-running API tasks (clinical analysis and model pulls) so request handlers stay responsive.

## Core Components

- Manager location: `DILIGENT/server/services/jobs.py`
- Shared singleton: `job_manager`
- Response models: `DILIGENT/server/domain/jobs.py`

## Job Model

Each job is represented by a thread-safe `JobState`:
- `job_id`: 8-char UUID prefix.
- `job_type`: logical group (`clinical`, `ollama_pull`, etc.).
- `status`: `pending`, `running`, `completed`, `failed`, `cancelled`.
- `progress`: 0.0 to 100.0.
- `result`: optional JSON-serializable payload.
- `error`: compact, user-safe failure message (sensitive internals are not exposed).
- `created_at` / `completed_at`: monotonic timestamps.
- `stop_requested`: cooperative cancellation flag.

## Threading Behavior

- Jobs are executed in daemon threads (`threading.Thread(..., daemon=True)`).
- `start_job` can inject `job_id` into runners automatically when supported.
- Completion merges partial `result` patches with final runner output.
- Exceptions mark the job as `failed` unless cancellation was requested.
- Exception details are still fully logged server-side (`exc_info=True`) for diagnostics.

## Current Job Types in DILIGENT

- `clinical`:
  - Started by `POST /clinical/jobs`.
  - Runner performs async clinical pipeline via `asyncio.run(...)`.
  - Progress stage metadata is updated incrementally (e.g., extraction, RAG lookup, report composition).
- `ollama_pull`:
  - Started by `POST /models/pull/jobs`.
  - Runner pulls models from Ollama and reports completion state.
- `rxnav_update`:
  - Started by `POST /inspection/rxnav/jobs`.
  - Runner executes in-app RxNav catalog refresh with cooperative cancellation checks.
  - Progress/result metadata include update-stage messages (for UI progress bars).
- `livertox_update`:
  - Started by `POST /inspection/livertox/jobs`.
  - Runner executes in-app LiverTox refresh with cooperative cancellation checks.
  - Progress/result metadata include update-stage messages (for UI progress bars).

## API Contract Pattern

1. Start: endpoint returns `JobStartResponse` (`job_id`, `status`, `poll_interval`).
2. Poll: `GET .../jobs/{job_id}` returns `JobStatusResponse`.
3. Cancel: `DELETE .../jobs/{job_id}` sets stop request and returns `JobCancelResponse`.

The frontend (`client/src/services/api.ts`) polls using the server-provided interval and stops when status is terminal (`completed`, `failed`, `cancelled`).

## Cancellation Rules

Cancellation is cooperative:
- Pending jobs can be marked `cancelled` immediately.
- Running jobs receive `stop_requested=True`.
- Runner code must check `job_manager.should_stop(job_id)` at safe checkpoints.

If your runner ignores stop checks, cancellation will be delayed until completion/failure.

## Implementation Pattern for New Jobs

1. Add a synchronous runner that returns `dict[str, Any]`.
2. Make runner periodically check `job_manager.should_stop(job_id)`.
3. Use `job_manager.update_progress(...)` and `job_manager.update_result(...)` for interim updates.
4. Expose start/poll/cancel routes.
5. Guard duplicates with `job_manager.is_job_running(job_type)` when needed.

Minimal import example:

```python
from DILIGENT.server.services.jobs import job_manager
```
