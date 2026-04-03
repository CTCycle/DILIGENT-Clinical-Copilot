from __future__ import annotations

import inspect
import threading
import uuid
import asyncio
from time import monotonic
from typing import Any

from collections.abc import Callable

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.domain.jobs import JobState


SENSITIVE_ERROR_TOKENS: tuple[str, ...] = (
    "traceback",
    "stack",
    "token",
    "secret",
    "password",
    "authorization",
    "api key",
    "access key",
)


###############################################################################
class JobErrorSanitizer:
    # -------------------------------------------------------------------------
    @staticmethod
    def can_show_exception_message(message: str) -> bool:
        candidate = message.strip()
        if not candidate:
            return False
        if len(candidate) > 180:
            return False
        lowered = candidate.casefold()
        return not any(token in lowered for token in SENSITIVE_ERROR_TOKENS)

    # -------------------------------------------------------------------------
    @classmethod
    def build_safe_job_error_message(cls, exc: Exception) -> str:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return "Operation timed out. Please retry."
        if isinstance(exc, FileNotFoundError):
            return "A required file was not found. Check configuration and retry."
        if isinstance(exc, ConnectionError):
            return "A dependency could not be reached. Please retry shortly."
        if isinstance(exc, ValueError):
            candidate = str(exc).split("\n")[0]
            if cls.can_show_exception_message(candidate):
                return candidate
            return "Input validation failed. Review the request and retry."

        candidate = str(exc).split("\n")[0]
        if cls.can_show_exception_message(candidate):
            return candidate
        return "Operation failed unexpectedly. Please retry."

###############################################################################
class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())[:8]
        state = JobState(job_id=job_id, job_type=job_type, status="pending")
        runner_kwargs = kwargs.copy() if kwargs else {}

        if self.runner_accepts_job_id(runner):
            runner_kwargs["job_id"] = job_id

        with self.lock:
            self.jobs[job_id] = state

        thread = threading.Thread(
            target=self.run_job,
            args=(job_id, runner, args, runner_kwargs),
            daemon=True,
        )

        with self.lock:
            self.threads[job_id] = thread

        state.update(status="running")
        thread.start()

        logger.info("Started job %s (type=%s)", job_id, job_type)
        return job_id

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None
        return state.snapshot()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return False
        if state.status not in ("pending", "running"):
            return False
        if state.status == "pending":
            state.update(stop_requested=True, status="cancelled", completed_at=monotonic())
            logger.info("Cancelled pending job %s", job_id)
            return True
        state.update(stop_requested=True)
        logger.info("Cancellation requested for job %s", job_id)
        return True

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        with self.lock:
            for state in self.jobs.values():
                if state.status in ("pending", "running"):
                    if job_type is None or state.job_type == job_type:
                        return True
        return False

    # -------------------------------------------------------------------------
    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            states = list(self.jobs.values())
        results: list[dict[str, Any]] = []
        for state in states:
            if job_type is None or state.job_type == job_type:
                results.append(state.snapshot())
        return results

    # -------------------------------------------------------------------------
    def should_stop(self, job_id: str) -> bool:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return True
        return state.stop_requested

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, progress: float) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state:
            state.update(progress=min(100.0, max(0.0, progress)))

    # -------------------------------------------------------------------------
    def update_result(self, job_id: str, patch: dict[str, Any]) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return
        with state.lock:
            existing = state.result or {}
            merged = {**existing, **patch}
            state.result = merged

    # -------------------------------------------------------------------------
    def run_job(
        self,
        job_id: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return
        if state.stop_requested:
            state.update(status="cancelled", completed_at=monotonic())
            logger.info("Job %s cancelled before execution", job_id)
            return

        try:
            result = runner(*args, **kwargs)
            if state.stop_requested:
                state.update(status="cancelled", completed_at=monotonic())
            else:
                result_payload = result or {}
                with state.lock:
                    merged = {**(state.result or {}), **result_payload}
                state.update(
                    status="completed",
                    result=merged if merged else None,
                    progress=100.0,
                    completed_at=monotonic(),
                )
                logger.info("Job %s completed successfully", job_id)
        except Exception as exc:  # noqa: BLE001
            if state.stop_requested:
                state.update(status="cancelled", completed_at=monotonic())
                logger.info("Job %s cancelled during execution", job_id)
                return
            error_msg = JobErrorSanitizer.build_safe_job_error_message(exc)
            state.update(status="failed", error=error_msg, completed_at=monotonic())
            logger.error(
                "Job %s failed type=%s message=%s",
                job_id,
                type(exc).__name__,
                error_msg,
            )
            logger.debug("Job %s error details", job_id, exc_info=True)

    # -------------------------------------------------------------------------
    def runner_accepts_job_id(self, runner: Callable[..., dict[str, Any]]) -> bool:
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return False
        parameters = list(signature.parameters.values())
        for param in parameters:
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return any(param.name == "job_id" for param in parameters)


###############################################################################
job_manager = JobManager()

