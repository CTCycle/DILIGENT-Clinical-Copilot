from __future__ import annotations

from fastapi import APIRouter, Query, status

from DILIGENT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from DILIGENT.server.domain.models import ModelListResponse, ModelPullResponse
from DILIGENT.server.services.llm.providers import OllamaClient
from DILIGENT.server.services import ollama_service as ollama_service_module
from DILIGENT.server.services.ollama_service import OllamaService

router = APIRouter(prefix="/models", tags=["models"])
service = OllamaService()
job_manager = ollama_service_module.job_manager


###############################################################################
class OllamaEndpoint:
    def __init__(self, *, router: APIRouter, service: OllamaService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def sync_service_seams() -> None:
        ollama_service_module.job_manager = job_manager

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
        self.sync_service_seams()
        ollama_service_module.OllamaClient = OllamaClient
        return await self.service.pull_model(name=name, stream=stream)

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
            True,
            description="If True, stream pull from Ollama to expose incremental progress.",
        ),
    ) -> JobStartResponse:
        self.sync_service_seams()
        return self.service.start_pull_job(name=name, stream=stream)

    # -------------------------------------------------------------------------
    def get_pull_job_status(self, job_id: str) -> JobStatusResponse:
        self.sync_service_seams()
        return self.service.get_pull_job_status(job_id=job_id)

    # -------------------------------------------------------------------------
    def cancel_pull_job(self, job_id: str) -> JobCancelResponse:
        self.sync_service_seams()
        return self.service.cancel_pull_job(job_id=job_id)

    # -------------------------------------------------------------------------
    async def list_available_models(self) -> ModelListResponse:
        self.sync_service_seams()
        ollama_service_module.OllamaClient = OllamaClient
        return await self.service.list_available_models()

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


endpoint = OllamaEndpoint(router=router, service=service)
endpoint.add_routes()
