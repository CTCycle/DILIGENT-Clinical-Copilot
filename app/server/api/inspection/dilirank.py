from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.dilirank import (
    DiliRankCatalogResponse,
    DiliRankDrugResponse,
    DiliRankUpdateConfigResponse,
    DiliRankUpdateRequest,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.service import DataInspectionService

###############################################################################
class InspectionDiliRankEndpoint(InspectionJobEndpointMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

    # -------------------------------------------------------------------------
    def list_catalog(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> DiliRankCatalogResponse:
        return DiliRankCatalogResponse(
            **self.service.list_dilirank_catalog(
                search=search,
                offset=offset,
                limit=limit,
            )
        )

    # -------------------------------------------------------------------------
    def get_drug_records(self, drug_id: int) -> DiliRankDrugResponse:
        payload = self.service.get_dilirank_records(drug_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DILIrank record not found.",
            )
        return DiliRankDrugResponse(**payload)

    # -------------------------------------------------------------------------
    def get_update_config(self) -> DiliRankUpdateConfigResponse:
        return DiliRankUpdateConfigResponse(
            **self.service.build_update_config_response("dilirank")
        )

    # -------------------------------------------------------------------------
    def start_dilirank_update_job(
        self,
        request: DiliRankUpdateRequest | None = Body(default=None),
    ) -> JobStartResponse:
        request = request or DiliRankUpdateRequest()
        return self.start_update_job(
            job_type=self.service.DILIRANK_JOB_TYPE,
            message="DILIrank 2.0 update job started",
            overrides=request.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    def get_dilirank_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.DILIRANK_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_dilirank_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.DILIRANK_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/dilirank",
            self.list_catalog,
            methods=["GET"],
            response_model=DiliRankCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dilirank/update-config",
            self.get_update_config,
            methods=["GET"],
            response_model=DiliRankUpdateConfigResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dilirank/jobs",
            self.start_dilirank_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/dilirank/{drug_id}",
            self.get_drug_records,
            methods=["GET"],
            response_model=DiliRankDrugResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dilirank/jobs/{job_id}",
            self.get_dilirank_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dilirank/jobs/{job_id}",
            self.cancel_dilirank_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )