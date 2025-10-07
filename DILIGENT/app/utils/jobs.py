from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx

from DILIGENT.app.constants import (
    API_BASE_URL,
    PHARMACOLOGY_LIVERTOX_STATUS_ENDPOINT,
)

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
    # -----------------------------------------------------------------------------
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.events: dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()

    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    async def set_progress(
        self, job_id: str, progress: float, detail: str | None = None
    ) -> None:
        await self._update_job(job_id, progress=progress, detail=detail)

    # -----------------------------------------------------------------------------
    async def get_job(self, job_id: str) -> JobPayload | None:
        async with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return None
            return self._serialize(record)

    # -----------------------------------------------------------------------------
    async def wait_for_completion(self, job_id: str) -> JobPayload | None:
        async with self.lock:
            event = self.events.get(job_id)
        if event is None:
            return None
        await event.wait()
        return await self.get_job(job_id)

    # -----------------------------------------------------------------------------
    async def _run_job(
        self,
        job_id: str,
        func: JobCoroutine,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        await self._update_job(job_id, status=JobStatus.RUNNING, detail="Job started")
        try:
            result = await func(job_id, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            await self._handle_failure(job_id, exc)
        else:
            await self._finalize_success(job_id, result)

    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    async def _signal_completion(self, job_id: str) -> None:
        async with self.lock:
            event = self.events.pop(job_id, None)
        if event is not None:
            event.set()

    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def _append_progress(progress_log: list[str], status: Any, detail: Any) -> None:
    if not isinstance(detail, str) or not detail:
        return
    label = status if isinstance(status, str) and status else "status"
    entry = f"{label}: {detail}"
    if entry not in progress_log:
        progress_log.append(entry)


# -----------------------------------------------------------------------------
def _format_progress_log(progress_log: list[str]) -> str:
    if not progress_log:
        return ""
    return "\nProgress log:\n" + "\n".join(progress_log)


# -----------------------------------------------------------------------------
async def _await_livertox_job(
    client: httpx.AsyncClient,
    initial_status: dict[str, Any],
    *,
    poll_interval: float = 60.0,
    timeout: float | None = None,
) -> tuple[dict[str, Any], list[str]] | str:
    progress_log: list[str] = []
    status = initial_status.get("status")
    detail = initial_status.get("detail")
    _append_progress(progress_log, status, detail)

    normalized_status = status.lower() if isinstance(status, str) else ""
    result = initial_status.get("result")
    job_id = initial_status.get("job_id")
    if normalized_status == "failed":
        failure = (
            detail
            if isinstance(detail, str) and detail
            else "Backend reported job failure."
        )
        status_code = initial_status.get("status_code")
        if isinstance(status_code, int):
            failure = f"{failure} (status {status_code})"
        return (
            "[ERROR] LiverTox import failed: "
            + failure
            + _format_progress_log(progress_log)
        )
    if normalized_status == "completed":
        if isinstance(result, dict):
            return result, progress_log
        return (
            "[ERROR] Backend did not provide job result on completion."
            + _format_progress_log(progress_log)
        )

    if not isinstance(job_id, str) or not job_id:
        if isinstance(result, dict):
            return result, progress_log
        expected_keys = ("file_path", "records", "processed_entries")
        if any(key in initial_status for key in expected_keys):
            return initial_status, progress_log
        return (
            "[ERROR] Backend response did not include a job ID."
            + _format_progress_log(progress_log)
        )

    status_url = f"{API_BASE_URL}{PHARMACOLOGY_LIVERTOX_STATUS_ENDPOINT}/{job_id}"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout if timeout is not None else None

    while True:
        if deadline is not None and loop.time() > deadline:
            progress_suffix = _format_progress_log(progress_log)
            waited = max(1, int(round(timeout))) if timeout is not None else 0
            message = (
                "[INFO] LiverTox import is still running after waiting "
                f"{waited} seconds."
                f"\nJob ID: {job_id}"
                f"\nStatus URL: {status_url}"
                "\nThe ingestion continues in the background; you can keep this tab "
                "open or retry later to refresh the status."
            )
            if progress_suffix:
                message += progress_suffix
            return message

        try:
            status_response = await client.get(status_url)
            status_response.raise_for_status()
        except httpx.TimeoutException:
            return "[ERROR] Polling job status timed out." + _format_progress_log(
                progress_log
            )
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            code = exc.response.status_code if exc.response else "unknown"
            return (
                "[ERROR] Backend returned an error while checking job status."
                f"\nURL: {status_url}"
                f"\nStatus: {code}"
                f"\nResponse body:\n{body}" + _format_progress_log(progress_log)
            )
        except Exception as exc:  # noqa: BLE001
            return (
                f"[ERROR] Unexpected error while polling job status: {exc}"
                + _format_progress_log(progress_log)
            )

        try:
            status_payload = status_response.json()
        except ValueError:
            return (
                "[ERROR] Backend status response was not valid JSON."
                + _format_progress_log(progress_log)
            )

        if not isinstance(status_payload, dict):
            return (
                "[ERROR] Unexpected status response format from backend."
                + _format_progress_log(progress_log)
            )

        status = status_payload.get("status")
        detail = status_payload.get("detail")
        _append_progress(progress_log, status, detail)
        normalized_status = status.lower() if isinstance(status, str) else ""

        if normalized_status == "failed":
            failure = (
                detail
                if isinstance(detail, str) and detail
                else "Backend reported job failure."
            )
            status_code = status_payload.get("status_code")
            if isinstance(status_code, int):
                failure = f"{failure} (status {status_code})"
            return (
                "[ERROR] LiverTox import failed: "
                + failure
                + _format_progress_log(progress_log)
            )

        if normalized_status == "completed":
            result = status_payload.get("result")
            if isinstance(result, dict):
                return result, progress_log
            return (
                "[ERROR] Backend did not provide job result on completion."
                + _format_progress_log(progress_log)
            )

        await asyncio.sleep(poll_interval)
