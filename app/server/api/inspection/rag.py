from __future__ import annotations

from fastapi import APIRouter, Body, Query, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.inspection import (
    CatalogListFilters,
    InspectionRagUpdateRequest,
    InspectionUpdateConfigResponse,
    LanceVectorStoreSummaryResponse,
    RagDocumentListResponse,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.service import DataInspectionService


###############################################################################
class InspectionRagEndpoint(InspectionJobEndpointMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

    # -------------------------------------------------------------------------
    def get_rag_update_config(self) -> InspectionUpdateConfigResponse:
        payload = self.service.build_update_config_response("rag")
        return InspectionUpdateConfigResponse(**payload)

    # -------------------------------------------------------------------------
    def list_rag_documents(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> RagDocumentListResponse:
        filters = CatalogListFilters(search=search, offset=offset, limit=limit)
        return RagDocumentListResponse(
            **self.service.list_rag_documents(
                search=filters.search,
                offset=filters.offset,
                limit=filters.limit,
            )
        )

    # -------------------------------------------------------------------------
    def get_rag_vector_store(self) -> LanceVectorStoreSummaryResponse:
        return LanceVectorStoreSummaryResponse(
            **self.service.get_rag_vector_store_summary()
        )

    # -------------------------------------------------------------------------
    def start_rag_update_job(
        self,
        request: InspectionRagUpdateRequest | None = Body(default=None),
    ) -> JobStartResponse:
        request = request or InspectionRagUpdateRequest()
        return self.start_update_job(
            job_type=self.service.RAG_JOB_TYPE,
            message="RAG embeddings update job started",
            overrides=request.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    def get_rag_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.RAG_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_rag_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.RAG_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/rag/update-config",
            self.get_rag_update_config,
            methods=["GET"],
            response_model=InspectionUpdateConfigResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/documents",
            self.list_rag_documents,
            methods=["GET"],
            response_model=RagDocumentListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/vector-store",
            self.get_rag_vector_store,
            methods=["GET"],
            response_model=LanceVectorStoreSummaryResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/jobs",
            self.start_rag_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/rag/jobs/{job_id}",
            self.get_rag_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/jobs/{job_id}",
            self.cancel_rag_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
