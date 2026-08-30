from __future__ import annotations

import asyncio
import inspect
import threading
import uuid
from collections.abc import Callable
from functools import lru_cache
from time import monotonic
from typing import Any

from common.utils.error_filters import get_sensitive_error_tokens
from common.utils.logger import logger
from services.runtime.state import JobState


###############################################################################
class JobErrorSanitizer:
    LOCAL_MODEL_MEMORY_MESSAGE = (
        "Local model could not be loaded due to insufficient memory. "
        "Choose a smaller local model or free memory and retry."
    )
    LOCAL_MODEL_UNAVAILABLE_MESSAGE = (
        "Local model runtime is unavailable. Start Ollama, verify the selected "
        "model is installed, and retry."
    )

    # -------------------------------------------------------------------------
    @staticmethod
    def classify_llm_runtime_failure(message: str) -> str | None:
        lowered = message.casefold()
        if any(
            token in lowered
            for token in (
                "out-of-memory",
                "out of memory",
                "cudamalloc",
                "failed to allocate",
                "cuda out of memory",
            )
        ):
            return JobErrorSanitizer.LOCAL_MODEL_MEMORY_MESSAGE
        if any(
            token in lowered
            for token in (
                "ollama request failed",
                "failed to request ollama",
                "all connection attempts failed",
                "connection refused",
                "ollama server exited",
                "ollama executable not found",
                "model ",
            )
        ) and any(
            token in lowered
            for token in (
                "ollama",
                "not found",
                "connection",
                "unavailable",
                "executable",
            )
        ):
            return JobErrorSanitizer.LOCAL_MODEL_UNAVAILABLE_MESSAGE
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def can_show_exception_message(message: str) -> bool:
        candidate = message.strip()
        if not candidate:
            return False
        if len(candidate) > 180:
            return False
        lowered = candidate.casefold()
        return not any(token in lowered for token in get_sensitive_error_tokens())

    # -------------------------------------------------------------------------
    @classmethod
    def build_safe_job_error_message(cls, exc: Exception) -> str:
        classified = cls.classify_llm_runtime_failure(str(exc))
        if classified:
            return classified
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
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.accepting_jobs = True
        self.max_terminal_jobs = 256

    # -------------------------------------------------------------------------
    def begin_startup(self) -> None:
        with self.lock:
            self.accepting_jobs = True

    # -------------------------------------------------------------------------
    def start_job(
        self,
        job_type: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        scope_key: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())[:8]
        state = JobState(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            scope_key=scope_key,
        )
        runner_kwargs = kwargs.copy() if kwargs else {}

        if self.runner_accepts_job_id(runner):
            runner_kwargs["job_id"] = job_id

        thread = threading.Thread(
            target=self.run_job,
            args=(job_id, runner, args, runner_kwargs),
            daemon=True,
        )

        with self.lock:
            if not self.accepting_jobs:
                raise RuntimeError("Job manager is shutting down")
            self.jobs[job_id] = state
            self.threads[job_id] = thread
            state.update(status="running")
            thread.start()

        logger.info("Started job %s (type=%s scope=%s)", job_id, job_type, scope_key)
        return job_id

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None
        return state.snapshot()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None
        already_requested = False
        was_pending = False
        with state.lock:
            if state.status not in ("pending", "running"):
                return None
            if state.stop_requested:
                already_requested = True
            else:
                state.stop_requested = True
                state.version += 1
                if state.status == "pending":
                    was_pending = True
                    state.status = "cancelled"
                    state.completed_at = monotonic()
        if already_requested:
            return state.snapshot()
        if was_pending:
            with self.lock:
                self._prune_terminal_jobs_locked()
            logger.info("Cancelled pending job %s", job_id)
        else:
            logger.info("Cancellation requested for job %s", job_id)
        return state.snapshot()

    # -------------------------------------------------------------------------
    def is_job_running(
        self, job_type: str | None = None, *, scope_key: str | None = None
    ) -> bool:
        with self.lock:
            for state in self.jobs.values():
                if state.status in ("pending", "running"):
                    if job_type is not None and state.job_type != job_type:
                        continue
                    if scope_key is not None and state.scope_key != scope_key:
                        continue
                    return True
        return False

    # -------------------------------------------------------------------------
    def get_running_job(
        self, job_type: str, *, scope_key: str | None = None
    ) -> dict[str, Any] | None:
        with self.lock:
            states = list(self.jobs.values())
        for state in states:
            if state.job_type != job_type:
                continue
            if scope_key is not None and state.scope_key != scope_key:
                continue
            if state.status in ("pending", "running"):
                return state.snapshot()
        return None

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
    def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop new work and wait briefly for cooperative workers to finish."""
        with self.lock:
            self.accepting_jobs = False
            states = list(self.jobs.values())
            threads = list(self.threads.values())

        for state in states:
            with state.lock:
                if state.status not in ("pending", "running"):
                    continue
                state.stop_requested = True
                state.version += 1
                if state.status == "pending":
                    state.status = "cancelled"
                    state.completed_at = monotonic()

        with self.lock:
            self._prune_terminal_jobs_locked()
        deadline = monotonic() + max(0.0, timeout)
        for thread in threads:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        with self.lock:
            finished = not any(thread.is_alive() for thread in self.threads.values())
            self._prune_terminal_jobs_locked()
        return finished

    # -------------------------------------------------------------------------
    def update_progress(self, job_id: str, progress: float) -> None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state:
            clamped = min(100.0, max(0.0, progress))
            non_decreasing = max(state.progress, clamped)
            state.update(progress=non_decreasing, last_activity_at=monotonic())

    # -------------------------------------------------------------------------
    def update_result(
        self, job_id: str, patch: dict[str, Any]
    ) -> dict[str, Any] | None:
        with self.lock:
            state = self.jobs.get(job_id)
        if state is None:
            return None
        state.update(last_activity_at=monotonic())
        return state.merge_result(patch).model_dump()

    # -------------------------------------------------------------------------
    def run_job(
        self,
        job_id: str,
        runner: Callable[..., dict[str, Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        try:
            with self.lock:
                state = self.jobs.get(job_id)
            if state is None:
                return
            if state.stop_requested:
                self._finish_terminal_state(
                    state,
                    status="cancelled",
                    completed_at=monotonic(),
                )
                return
            try:
                result = runner(*args, **kwargs)
                if state.stop_requested:
                    self._finish_terminal_state(
                        state,
                        status="cancelled",
                        completed_at=monotonic(),
                    )
                else:
                    result_payload = result or {}
                    with state.lock:
                        merged = {**(state.result or {}), **result_payload}
                    self._finish_terminal_state(
                        state,
                        status="completed",
                        result=merged if merged else None,
                        progress=100.0,
                        completed_at=monotonic(),
                    )
                    logger.info("Job %s completed successfully", job_id)
            except Exception as exc:  # noqa: BLE001
                if state.stop_requested:
                    self._finish_terminal_state(
                        state,
                        status="cancelled",
                        completed_at=monotonic(),
                    )
                    logger.info("Job %s cancelled during execution", job_id)
                    return
                error_msg = JobErrorSanitizer.build_safe_job_error_message(exc)
                self._finish_terminal_state(
                    state,
                    status="failed",
                    error=error_msg,
                    completed_at=monotonic(),
                )
                logger.error(
                    "Job %s failed type=%s message=%s",
                    job_id,
                    type(exc).__name__,
                    error_msg,
                )
                logger.debug("Job %s error details", job_id, exc_info=True)
        finally:
            with self.lock:
                self.threads.pop(job_id, None)
                self._prune_terminal_jobs_locked()

    # -------------------------------------------------------------------------
    def _finish_terminal_state(self, state: JobState, **updates: Any) -> None:
        with self.lock:
            if self.jobs.get(state.job_id) is not state:
                return
            state.update(**updates)
            self._prune_terminal_jobs_locked()

    # -------------------------------------------------------------------------
    def _prune_terminal_jobs_locked(self) -> None:
        terminal = [
            state
            for state in self.jobs.values()
            if state.status in {"completed", "failed", "cancelled"}
        ]
        overflow = len(terminal) - self.max_terminal_jobs
        if overflow <= 0:
            return
        terminal.sort(
            key=lambda state: (
                state.completed_at
                if state.completed_at is not None
                else state.created_at
            )
        )
        for state in terminal[:overflow]:
            if state.job_id not in self.threads:
                self.jobs.pop(state.job_id, None)

    # -------------------------------------------------------------------------
    def runner_accepts_job_id(self, runner: Callable[..., dict[str, Any]]) -> bool:
        try:
            signature = inspect.signature(runner)
        except TypeError, ValueError:
            return False
        parameters = list(signature.parameters.values())
        for param in parameters:
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return any(param.name == "job_id" for param in parameters)


###############################################################################
@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager()
