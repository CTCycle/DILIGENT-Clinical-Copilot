from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from DILIGENT.server.models.providers import OllamaClient, OllamaError, OllamaTimeout
from DILIGENT.server.entities.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from DILIGENT.server.entities.models import ModelListResponse, ModelPullResponse
from DILIGENT.server.configurations import server_settings
from DILIGENT.server.services.jobs import job_manager
from DILIGENT.server.common.utils.logger import logger

router = APIRouter(prefix="/models", tags=["models"])
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
async def pull_model_async(name: str, stream: bool) -> dict[str, Any]:
    async with OllamaClient() as client:
        local = set(await client.list_models())
        already = name in local
        if not already:
            logger.info("Downloading model %s from Ollama library", name)
            await client.pull(name, stream=stream)
        return {"model": name, "pulled": not already}


###############################################################################
def run_model_pull_job(name: str, stream: bool, job_id: str) -> dict[str, Any]:
    if job_manager.should_stop(job_id):
        return {}
    job_manager.update_progress(job_id, 5.0)
    return asyncio.run(pull_model_async(name=name, stream=stream))


###############################################################################
class OllamaEndpoint:
    JOB_TYPE = "ollama_pull"

    def __init__(self, *, router: APIRouter) -> None:
        self.router = router        

    # -------------------------------------------------------------------------
    async def pull_model(
        self,
        name: str = Query(
            ...,
            min_length=1,
            max_length=200,
            description="Exact Ollama model name, e.g. 'llama3.1:8b'",
        ),
        stream: bool = Query(
            False,
            description="If True, stream pull from Ollama. Endpoint returns only final status (no SSE).",
        ),
    ) -> ModelPullResponse:
        model_name = sanitize_model_name(name)
        try:
            async with OllamaClient() as client:
                local = set(await client.list_models())
                already = model_name in local
                if not already:
                    logger.info(f"Downloading model {model_name} from Ollama library")
                    await client.pull(model_name, stream=stream)
                return ModelPullResponse(
                    status="success", pulled=(not already), model=model_name
                )
        except Exception as exc:
            if isinstance(exc, OllamaTimeout):
                raise HTTPException(status_code=504, detail=str(exc))
            if isinstance(exc, OllamaError):
                raise HTTPException(status_code=502, detail=str(exc))
            raise HTTPException(
                status_code=500, detail="Unexpected error while pulling model"
            )

    # -------------------------------------------------------------------------
    def start_pull_job(
        self,
        name: str = Query(
            ...,
            min_length=1,
            max_length=200,
            description="Exact Ollama model name, e.g. 'llama3.1:8b'",
        ),
        stream: bool = Query(
            False,
            description="If True, stream pull from Ollama. Job returns only final status.",
        ),
    ) -> JobStartResponse:
        model_name = sanitize_model_name(name)
        if job_manager.is_job_running(self.JOB_TYPE):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A model pull job is already in progress",
            )

        job_id = job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=run_model_pull_job,
            kwargs={
                "name": model_name,
                "stream": stream,
            },
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
    def get_pull_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    def cancel_pull_job(self, job_id: str) -> JobCancelResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
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
        except Exception as exc:
            if isinstance(exc, OllamaTimeout):
                raise HTTPException(status_code=504, detail=str(exc))
            if isinstance(exc, OllamaError):
                raise HTTPException(status_code=502, detail=str(exc))
            raise HTTPException(
                status_code=500, detail="Unexpected error while listing models"
            )
        
    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/pull",
            self.pull_model,
            methods=["GET"],
            response_model=ModelPullResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/pull/jobs",
            self.start_pull_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_pull_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_pull_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/list",
            self.list_available_models,
            methods=["GET"],
            response_model=ModelListResponse,
            status_code=status.HTTP_200_OK,
        )


endpoint = OllamaEndpoint(router=router)
endpoint.add_routes()

