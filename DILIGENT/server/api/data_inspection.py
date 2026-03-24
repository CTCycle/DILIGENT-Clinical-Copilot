from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query, status

from DILIGENT.server.configurations import server_settings
from DILIGENT.server.domain.inspection import (
    CatalogListFilters,
    DeleteEntityResponse,
    DateFilterMode,
    DrugAliasesResponse,
    LiverToxCatalogResponse,
    LiverToxExcerptResponse,
    RxNavCatalogResponse,
    SessionCatalogResponse,
    SessionReportResponse,
    SessionListFilters,
    SessionStatus,
)
from DILIGENT.server.domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from DILIGENT.server.services.inspection import DataInspectionService


router = APIRouter(prefix="/inspection", tags=["inspection"])
service = DataInspectionService()


###############################################################################
class DataInspectionEndpoint:
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def build_job_start_response(
        self,
        *,
        payload: dict[str, object],
        message: str,
    ) -> JobStartResponse:
        return JobStartResponse(
            job_id=str(payload["job_id"]),
            job_type=str(payload["job_type"]),
            status=str(payload["status"]),
            message=message,
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def start_update_job(self, *, job_type: str, message: str) -> JobStartResponse:
        try:
            payload = self.service.start_update_job(job_type)
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=error_status, detail=detail) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        return self.build_job_start_response(payload=payload, message=message)

    # -------------------------------------------------------------------------
    def get_update_job_status(self, *, job_id: str, job_type: str) -> JobStatusResponse:
        payload = self.service.get_job_status(job_id, expected_type=job_type)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**payload)

    # -------------------------------------------------------------------------
    def cancel_update_job(self, *, job_id: str, job_type: str) -> JobCancelResponse:
        success = self.service.cancel_job(job_id, expected_type=job_type)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobCancelResponse(
            job_id=job_id,
            success=True,
            message="Cancellation requested",
        )

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        search: str | None = Query(default=None),
        status_filter: SessionStatus | None = Query(default=None, alias="status"),
        date_mode: DateFilterMode | None = Query(default=None),
        filter_date: date | None = Query(default=None, alias="date"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> SessionCatalogResponse:
        filters = SessionListFilters(
            search=search,
            status=status_filter,
            date_mode=date_mode,
            date=filter_date,
            offset=offset,
            limit=limit,
        )
        payload = self.service.list_sessions(
            search=filters.search,
            status_filter=filters.status,
            date_mode=filters.date_mode,
            filter_date=filters.date,
            offset=filters.offset,
            limit=filters.limit,
        )
        return SessionCatalogResponse(**payload)

    # -------------------------------------------------------------------------
    def get_session_report(self, session_id: int) -> SessionReportResponse:
        report = self.service.get_session_report(session_id)
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session report not found: {session_id}",
            )
        return SessionReportResponse(session_id=session_id, report=report)

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        return DeleteEntityResponse(deleted=True)

    # -------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> RxNavCatalogResponse:
        filters = CatalogListFilters(search=search, offset=offset, limit=limit)
        payload = self.service.list_rxnav_catalog(
            search=filters.search,
            offset=filters.offset,
            limit=filters.limit,
        )
        return RxNavCatalogResponse(**payload)

    # -------------------------------------------------------------------------
    def get_rxnav_aliases(self, drug_id: int) -> DrugAliasesResponse:
        payload = self.service.get_rxnav_alias_groups(drug_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Drug not found: {drug_id}",
            )
        return DrugAliasesResponse(**payload)

    # -------------------------------------------------------------------------
    def delete_rxnav_drug(self, drug_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_drug(drug_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Drug not found: {drug_id}",
            )
        return DeleteEntityResponse(deleted=True)

    # -------------------------------------------------------------------------
    def start_rxnav_update_job(self) -> JobStartResponse:
        return self.start_update_job(
            job_type=self.service.RXNAV_JOB_TYPE,
            message="RxNav update job started",
        )

    # -------------------------------------------------------------------------
    def get_rxnav_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.RXNAV_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_rxnav_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.RXNAV_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> LiverToxCatalogResponse:
        filters = CatalogListFilters(search=search, offset=offset, limit=limit)
        payload = self.service.list_livertox_catalog(
            search=filters.search,
            offset=filters.offset,
            limit=filters.limit,
        )
        return LiverToxCatalogResponse(**payload)

    # -------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> LiverToxExcerptResponse:
        payload = self.service.get_livertox_excerpt(drug_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LiverTox excerpt not found for drug: {drug_id}",
            )
        return LiverToxExcerptResponse(**payload)

    # -------------------------------------------------------------------------
    def delete_livertox_drug(self, drug_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_drug(drug_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Drug not found: {drug_id}",
            )
        return DeleteEntityResponse(deleted=True)

    # -------------------------------------------------------------------------
    def start_livertox_update_job(self) -> JobStartResponse:
        return self.start_update_job(
            job_type=self.service.LIVERTOX_JOB_TYPE,
            message="LiverTox update job started",
        )

    # -------------------------------------------------------------------------
    def get_livertox_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.LIVERTOX_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_livertox_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.LIVERTOX_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/sessions",
            self.list_sessions,
            methods=["GET"],
            response_model=SessionCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/report",
            self.get_session_report,
            methods=["GET"],
            response_model=SessionReportResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.delete_session,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )

        self.router.add_api_route(
            "/rxnav",
            self.list_rxnav_catalog,
            methods=["GET"],
            response_model=RxNavCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/{drug_id}/aliases",
            self.get_rxnav_aliases,
            methods=["GET"],
            response_model=DrugAliasesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/{drug_id}",
            self.delete_rxnav_drug,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/jobs",
            self.start_rxnav_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/rxnav/jobs/{job_id}",
            self.get_rxnav_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/jobs/{job_id}",
            self.cancel_rxnav_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )

        self.router.add_api_route(
            "/livertox",
            self.list_livertox_catalog,
            methods=["GET"],
            response_model=LiverToxCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/{drug_id}/excerpt",
            self.get_livertox_excerpt,
            methods=["GET"],
            response_model=LiverToxExcerptResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/{drug_id}",
            self.delete_livertox_drug,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/jobs",
            self.start_livertox_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/livertox/jobs/{job_id}",
            self.get_livertox_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/jobs/{job_id}",
            self.cancel_livertox_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


endpoint = DataInspectionEndpoint(router=router, service=service)
endpoint.add_routes()
