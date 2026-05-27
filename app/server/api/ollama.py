from __future__ import annotations

from fastapi import APIRouter, Query, status

from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from domain.models import ModelListResponse
from services.llm.ollama import OllamaService
from services.llm.ollama_client import OllamaClient
from services.runtime.jobs import get_job_manager

router = APIRouter(prefix="/models", tags=["models"])


###############################################################################
class OllamaEndpoint:
    def __init__(self, *, router: APIRouter, service: OllamaService) -> None:
        self.router = router
        self.service = service

    def start_pull_job(
        self,
        name: str = Query(
            ...,
            min_length=1,
            max_length=200,
            description="Exact Ollama model name, e.g. 'llama3.1:8b'",
        ),
        stream: bool = Query(
            True,
            description="If True, stream pull from Ollama to expose incremental progress.",
        ),
    ) -> JobStartResponse:
        return self.service.start_pull_job(name=name, stream=stream)

    # -------------------------------------------------------------------------
    def get_pull_job_status(self, job_id: str) -> JobStatusResponse:
        return self.service.get_pull_job_status(job_id=job_id)

    # -------------------------------------------------------------------------
    def cancel_pull_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_pull_job(job_id=job_id)

    # -------------------------------------------------------------------------
    async def list_available_models(self) -> ModelListResponse:
        return await self.service.list_available_models()

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
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


OllamaEndpoint(
    router=router,
    service=OllamaService(
        job_manager=get_job_manager(),
        client_factory=lambda: OllamaClient(),
    ),
).add_routes()

