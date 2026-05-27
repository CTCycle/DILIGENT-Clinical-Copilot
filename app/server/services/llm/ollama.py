from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any, NoReturn

from common.exceptions import (
    ServiceConflictError,
    ServiceDependencyError,
    ServiceError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from common.utils.logger import logger
from configurations.startup import server_settings
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from domain.models import ModelListResponse
from services.llm.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaTimeout,
)
from services.runtime.jobs import (
    JobManager,
)

SAFE_OLLAMA_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}$")


###############################################################################
def sanitize_model_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ServiceValidationError("Model name must not be empty.")
    if not SAFE_OLLAMA_MODEL_RE.fullmatch(normalized):
        raise ServiceValidationError("Invalid model name.")
    return normalized


###############################################################################
def clamp_progress(value: float) -> float:
    return max(0.0, min(100.0, value))


###############################################################################
def coerce_positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if number <= 0:
        return None
    return number


###############################################################################
def resolve_pull_progress(current_progress: float, event: dict[str, Any]) -> float:
    status_text = str(event.get("status", "")).strip().lower()
    if status_text == "success":
        return 100.0

    total = coerce_positive_float(event.get("total"))
    completed = coerce_positive_float(event.get("completed"))
    if total is not None and completed is not None:
        computed = (completed / total) * 100.0
        bounded = clamp_progress(computed)
        return max(current_progress, min(97.0, bounded))

    if "pulling manifest" in status_text:
        return max(current_progress, 12.0)
    if "verifying" in status_text:
        return max(current_progress, 98.0)
    if "writing manifest" in status_text:
        return max(current_progress, 99.0)
    if "removing any unused layers" in status_text:
        return max(current_progress, 99.5)
    if "pulling" in status_text:
        return max(current_progress, 8.0)
    return current_progress


###############################################################################
def resolve_pull_progress_message(name: str, event: dict[str, Any]) -> str:
    total = coerce_positive_float(event.get("total"))
    completed = coerce_positive_float(event.get("completed"))
    status_text = str(event.get("status", "")).strip()
    if total is not None and completed is not None:
        percentage = clamp_progress((completed / total) * 100.0)
        return f"Pulling '{name}'... {percentage:.1f}%"
    if status_text:
        return status_text
    return f"Pulling '{name}' from Ollama."


###############################################################################
class PullProgressUpdater:
    def __init__(
        self,
        *,
        name: str,
        job_id: str,
        initial_progress: float,
        jobs: JobManager,
    ) -> None:
        self.name = name
        self.job_id = job_id
        self.current_progress = initial_progress
        self.jobs = jobs

    # -------------------------------------------------------------------------
    async def __call__(self, event: dict[str, Any]) -> None:
        if self.jobs.should_stop(self.job_id):
            raise RuntimeError("Model pull stop requested.")

        self.current_progress = resolve_pull_progress(self.current_progress, event)
        self.jobs.update_progress(self.job_id, self.current_progress)

        total = coerce_positive_float(event.get("total"))
        completed = coerce_positive_float(event.get("completed"))
        progress_patch: dict[str, Any] = {
            "progress_status": str(event.get("status", "")).strip().lower()
            or "running",
            "progress_message": resolve_pull_progress_message(self.name, event),
        }
        if total is not None:
            progress_patch["total_bytes"] = int(total)
        if completed is not None:
            progress_patch["completed_bytes"] = int(completed)
        self.jobs.update_result(self.job_id, progress_patch)


###############################################################################
async def pull_model_async(
    *,
    name: str,
    stream: bool,
    jobs: JobManager,
    client_factory: Callable[[], Any],
    job_id: str | None = None,
) -> dict[str, Any]:
    async with client_factory() as client:
        await client.start_server()
        local = set(await client.list_models())
        already = name in local
        if job_id is not None:
            jobs.update_result(
                job_id,
                {
                    "model": name,
                    "progress_status": "ready" if already else "running",
                    "progress_message": (
                        f"Model '{name}' is already installed."
                        if already
                        else f"Pulling '{name}' from Ollama."
                    ),
                },
            )
        if not already:
            logger.info("Downloading model %s from Ollama library", name)
            if job_id is not None:
                initial_progress = 6.0
                jobs.update_progress(job_id, initial_progress)
                progress_updater = PullProgressUpdater(
                    name=name,
                    job_id=job_id,
                    initial_progress=initial_progress,
                    jobs=jobs,
                )
                await client.pull(
                    name, stream=stream, progress_callback=progress_updater
                )
            else:
                await client.pull(name, stream=stream)

        if job_id is not None:
            jobs.update_progress(job_id, 100.0)
            jobs.update_result(
                job_id,
                {
                    "model": name,
                    "pulled": not already,
                    "progress_status": "success",
                    "progress_message": f"Model '{name}' is available locally.",
                },
            )
        return {
            "model": name,
            "pulled": not already,
            "progress_status": "success",
            "progress_message": f"Model '{name}' is available locally.",
        }


###############################################################################
def run_model_pull_job(
    *,
    name: str,
    stream: bool,
    job_id: str,
    jobs: JobManager,
    client_factory: Callable[[], Any],
) -> dict[str, Any]:
    if jobs.should_stop(job_id):
        jobs.update_result(
            job_id,
            {
                "model": name,
                "progress_status": "cancelled",
                "progress_message": "Pull cancelled before execution.",
            },
        )
        return {}
    jobs.update_progress(job_id, 2.0)
    return asyncio.run(
        pull_model_async(
            name=name,
            stream=stream,
            jobs=jobs,
            client_factory=client_factory,
            job_id=job_id,
        )
    )


###############################################################################
class OllamaService:
    JOB_TYPE = "ollama_pull"

    def __init__(
        self,
        *,
        job_manager: JobManager,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.job_manager = job_manager
        self.client_factory = client_factory or OllamaClient

    # -------------------------------------------------------------------------
    @staticmethod
    def raise_ollama_service_error(exc: Exception, *, action: str) -> NoReturn:
        if isinstance(exc, OllamaTimeout):
            raise ServiceDependencyError(
                "Ollama request timed out. Please retry.",
                status_code=504,
            ) from exc
        if isinstance(exc, OllamaError):
            raise ServiceDependencyError(
                "Ollama service is unavailable. Verify Ollama is running and retry.",
                status_code=502,
            ) from exc
        raise ServiceError(
            f"Unexpected error while {action}.",
        ) from exc

    # -------------------------------------------------------------------------
    def start_pull_job(self, *, name: str, stream: bool) -> JobStartResponse:
        model_name = sanitize_model_name(name)
        if self.job_manager.is_job_running(self.JOB_TYPE):
            raise ServiceConflictError("A model pull job is already in progress")

        job_id = self.job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=run_model_pull_job,
            kwargs={
                "name": model_name,
                "stream": stream,
                "jobs": self.job_manager,
                "client_factory": self.client_factory,
            },
        )
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise ServiceError("Failed to initialize model pull job")
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message="Model pull job started",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_pull_job_status(self, *, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise ServiceNotFoundError("Job not found.")
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    def cancel_pull_job(self, *, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise ServiceNotFoundError("Job not found.")
        cancelled_snapshot = self.job_manager.cancel_job(job_id)
        success = cancelled_snapshot is not None
        if success:
            logger.info("Model pull stop requested for job %s", job_id)
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -------------------------------------------------------------------------
    async def list_available_models(self) -> ModelListResponse:
        try:
            async with self.client_factory() as client:
                if hasattr(client, "start_server"):
                    await client.start_server()
                models = await client.list_models()
            return ModelListResponse(models=models, count=len(models))
        except (OllamaTimeout, OllamaError, Exception) as exc:
            self.raise_ollama_service_error(exc, action="listing models")

