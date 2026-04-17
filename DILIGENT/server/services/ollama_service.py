from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import HTTPException, status

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from DILIGENT.server.domain.models import ModelListResponse, ModelPullResponse
from DILIGENT.server.services.jobs import job_manager
from DILIGENT.server.services.llm.providers import OllamaClient, OllamaError, OllamaTimeout

SAFE_OLLAMA_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}$")


###############################################################################
def sanitize_model_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model name must not be empty.",
        )
    if not SAFE_OLLAMA_MODEL_RE.fullmatch(normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid model name.",
        )
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
    def __init__(self, *, name: str, job_id: str, initial_progress: float) -> None:
        self.name = name
        self.job_id = job_id
        self.current_progress = initial_progress

    # -------------------------------------------------------------------------
    async def __call__(self, event: dict[str, Any]) -> None:
        if job_manager.should_stop(self.job_id):
            raise RuntimeError("Model pull stop requested.")

        self.current_progress = resolve_pull_progress(self.current_progress, event)
        job_manager.update_progress(self.job_id, self.current_progress)

        total = coerce_positive_float(event.get("total"))
        completed = coerce_positive_float(event.get("completed"))
        progress_patch: dict[str, Any] = {
            "progress_status": str(event.get("status", "")).strip().lower() or "running",
            "progress_message": resolve_pull_progress_message(self.name, event),
        }
        if total is not None:
            progress_patch["total_bytes"] = int(total)
        if completed is not None:
            progress_patch["completed_bytes"] = int(completed)
        job_manager.update_result(self.job_id, progress_patch)


###############################################################################
async def pull_model_async(name: str, stream: bool, job_id: str | None = None) -> dict[str, Any]:
    async with OllamaClient() as client:
        local = set(await client.list_models())
        already = name in local
        if job_id is not None:
            job_manager.update_result(
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
                job_manager.update_progress(job_id, initial_progress)
                progress_updater = PullProgressUpdater(
                    name=name,
                    job_id=job_id,
                    initial_progress=initial_progress,
                )
                await client.pull(name, stream=stream, progress_callback=progress_updater)
            else:
                await client.pull(name, stream=stream)

        if job_id is not None:
            job_manager.update_progress(job_id, 100.0)
            job_manager.update_result(
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
def run_model_pull_job(name: str, stream: bool, job_id: str) -> dict[str, Any]:
    if job_manager.should_stop(job_id):
        job_manager.update_result(
            job_id,
            {
                "model": name,
                "progress_status": "cancelled",
                "progress_message": "Pull cancelled before execution.",
            },
        )
        return {}
    job_manager.update_progress(job_id, 2.0)
    return asyncio.run(pull_model_async(name=name, stream=stream, job_id=job_id))


###############################################################################
class OllamaService:
    JOB_TYPE = "ollama_pull"

    # -------------------------------------------------------------------------
    @staticmethod
    def raise_ollama_http_exception(exc: Exception, *, action: str) -> None:
        if isinstance(exc, OllamaTimeout):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Ollama request timed out. Please retry.",
            ) from exc
        if isinstance(exc, OllamaError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama service is unavailable. Verify Ollama is running and retry.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while {action}.",
        ) from exc

    # -------------------------------------------------------------------------
    async def pull_model(self, *, name: str, stream: bool) -> ModelPullResponse:
        model_name = sanitize_model_name(name)
        try:
            async with OllamaClient() as client:
                local = set(await client.list_models())
                already = model_name in local
                if not already:
                    logger.info("Downloading model %s from Ollama library", model_name)
                    await client.pull(model_name, stream=stream)
                return ModelPullResponse(status="success", pulled=(not already), model=model_name)
        except (OllamaTimeout, OllamaError, Exception) as exc:
            self.raise_ollama_http_exception(exc, action="pulling model")

    # -------------------------------------------------------------------------
    def start_pull_job(self, *, name: str, stream: bool) -> JobStartResponse:
        model_name = sanitize_model_name(name)
        if job_manager.is_job_running(self.JOB_TYPE):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A model pull job is already in progress",
            )

        job_id = job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=run_model_pull_job,
            kwargs={"name": model_name, "stream": stream},
        )
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize model pull job",
            )
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message="Model pull job started",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def get_pull_job_status(*, job_id: str) -> JobStatusResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    @staticmethod
    def cancel_pull_job(*, job_id: str) -> JobCancelResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        success = job_manager.cancel_job(job_id)
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
            async with OllamaClient() as client:
                models = await client.list_models()
            return ModelListResponse(models=models, count=len(models))
        except (OllamaTimeout, OllamaError, Exception) as exc:
            self.raise_ollama_http_exception(exc, action="listing models")
