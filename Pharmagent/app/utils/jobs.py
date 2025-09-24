from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


type JobPayload = dict[str, Any]
type JobCoroutine = Callable[..., Awaitable[JobPayload]]

_RESULT_SENTINEL = object()


###############################################################################
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


###############################################################################
@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    detail: str | None = None
    result: JobPayload | None = None
    status_code: int | None = None


###############################################################################
class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.events: dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    async def submit(self, func: JobCoroutine, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        record = JobRecord(job_id, JobStatus.QUEUED, now, now, detail="Job queued")
        event = asyncio.Event()
        async with self.lock:
            self.jobs[job_id] = record
            self.events[job_id] = event
        loop = asyncio.get_running_loop()
        loop.create_task(self._run_job(job_id, func, args, kwargs))
        return job_id

    # -------------------------------------------------------------------------
    async def set_progress(self, job_id: str, progress: float, detail: str | None = None) -> None:
        await self._update_job(job_id, progress=progress, detail=detail)

    # -------------------------------------------------------------------------
    async def get_job(self, job_id: str) -> JobPayload | None:
        async with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return None
            return self._serialize(record)

    # -------------------------------------------------------------------------
    async def wait_for_completion(self, job_id: str) -> JobPayload | None:
        async with self.lock:
            event = self.events.get(job_id)
        if event is None:
            return None
        await event.wait()
        return await self.get_job(job_id)

    # -------------------------------------------------------------------------
    async def _run_job(
        self, job_id: str, func: JobCoroutine, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        await self._update_job(job_id, status=JobStatus.RUNNING, detail="Job started")
        try:
            result = await func(job_id, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            await self._handle_failure(job_id, exc)
        else:
            await self._finalize_success(job_id, result)

    # -------------------------------------------------------------------------
    async def _handle_failure(self, job_id: str, exc: Exception) -> None:
        detail = getattr(exc, "detail", str(exc))
        status_code = getattr(exc, "status_code", 500)
        try:
            status_value = int(status_code)
        except Exception:  # noqa: BLE001
            status_value = 500
        await self._update_job(
            job_id,
            status=JobStatus.FAILED,
            detail=str(detail),
            progress=1.0,
            result=None,
            status_code=status_value,
        )
        await self._signal_completion(job_id)

    # -------------------------------------------------------------------------
    async def _finalize_success(self, job_id: str, result: JobPayload | None) -> None:
        await self._update_job(
            job_id,
            status=JobStatus.COMPLETED,
            detail="Job completed",
            progress=1.0,
            result=result or {},
            status_code=200,
        )
        await self._signal_completion(job_id)

    # -------------------------------------------------------------------------
    async def _signal_completion(self, job_id: str) -> None:
        async with self.lock:
            event = self.events.pop(job_id, None)
        if event is not None:
            event.set()

    # -------------------------------------------------------------------------
    async def _update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        detail: str | None = None,
        result: JobPayload | None | object = _RESULT_SENTINEL,
        status_code: int | None = None,
    ) -> None:
        async with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = max(0.0, min(1.0, progress))
            if detail is not None:
                record.detail = detail
            if result is not _RESULT_SENTINEL:
                record.result = result  # type: ignore[assignment]
            if status_code is not None:
                record.status_code = status_code
            record.updated_at = datetime.now(UTC)

    # -------------------------------------------------------------------------
    def _serialize(self, record: JobRecord) -> JobPayload:
        return {
            "job_id": record.job_id,
            "status": record.status.value,
            "progress": record.progress,
            "detail": record.detail,
            "result": record.result,
            "status_code": record.status_code,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }
