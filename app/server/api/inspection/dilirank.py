from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, status

from common.utils.logger import logger
from domain.dilirank import (
    DiliRankCatalogResponse,
    DiliRankDrugResponse,
    DiliRankUpdateConfigResponse,
    DiliRankUpdateRequest,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.dilirank import DiliRankInspectionService

###############################################################################
class InspectionDiliRankEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DiliRankInspectionService,
    ) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def list_catalog(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> DiliRankCatalogResponse:
        return DiliRankCatalogResponse(
            **self.service.list_catalog(search=search, offset=offset, limit=limit)
        )

    # -------------------------------------------------------------------------
    def get_drug_records(self, drug_id: int) -> DiliRankDrugResponse:
        payload = self.service.get_drug_records(drug_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DILIrank record not found.",
            )
        return DiliRankDrugResponse(**payload)

    # -------------------------------------------------------------------------
    def get_update_config(self) -> DiliRankUpdateConfigResponse:
        return DiliRankUpdateConfigResponse(
            **self.service.build_update_config_response()
        )

    # -------------------------------------------------------------------------
    def start_update_job(
        self,
        request: DiliRankUpdateRequest | None = Body(default=None),
    ) -> JobStartResponse:
        request = request or DiliRankUpdateRequest()
        try:
            payload = self.service.start_update_job(
                self.service.JOB_TYPE,
                overrides=request.model_dump(),
            )
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            logger.warning("DILIrank update job rejected: %s", detail)
            raise HTTPException(
                status_code=error_status,
                detail=(
                    "A DILIrank update job is already running."
                    if error_status == status.HTTP_409_CONFLICT
                    else "Invalid DILIrank update request."
                ),
            ) from exc
        poll_interval = payload.get("poll_interval")
        return JobStartResponse(
            job_id=str(payload["job_id"]),
            job_type=str(payload["job_type"]),
            status=str(payload["status"]),
            message="DILIrank 2.0 update job started",
            poll_interval=(
                float(poll_interval)
                if isinstance(poll_interval, int | float)
                else 1.0
            ),
        )

    # -------------------------------------------------------------------------
    def get_update_job_status(self, job_id: str) -> JobStatusResponse:
        payload = self.service.get_job_status(
            job_id,
            expected_type=self.service.JOB_TYPE,
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobStatusResponse(**payload)

    # -------------------------------------------------------------------------
    def cancel_update_job(self, job_id: str) -> JobCancelResponse:
        if not self.service.cancel_job(job_id, expected_type=self.service.JOB_TYPE):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobCancelResponse(
            job_id=job_id,
            success=True,
            message="Cancellation requested",
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
            self.start_update_job,
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
            self.get_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dilirank/jobs/{job_id}",
            self.cancel_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
