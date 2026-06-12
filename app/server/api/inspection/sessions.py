from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Body, HTTPException, Query, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.inspection import (
    DateFilterMode,
    DeleteEntityResponse,
    ManualReportEditAudit,
    ManualReportEditRequest,
    ManualReportEditResponse,
    SessionCatalogResponse,
    SessionDetailResponse,
    SessionListFilters,
    SessionStatus,
    SessionUpdateRequest,
    SessionVersionComparisonResponse,
    SessionVersionDetailResponse,
    SessionVersionListResponse,
)
from services.inspection.service import DataInspectionService


###############################################################################
class InspectionSessionEndpoint(InspectionJobEndpointMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

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
    def get_session_detail(self, session_id: int) -> SessionDetailResponse:
        detail = self.service.get_session_detail(session_id)
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return SessionDetailResponse(**detail)

    # -------------------------------------------------------------------------
    def update_session(
        self,
        session_id: int,
        request: SessionUpdateRequest | None = Body(default=None),
    ) -> SessionDetailResponse:
        request = request or SessionUpdateRequest()
        detail = self.service.update_session(
            session_id,
            report_text=request.report_text,
            edited_fields=request.edited_fields,
            reviewer_note=request.reviewer_note,
            edited_by=request.edited_by,
            metadata=request.metadata,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return SessionDetailResponse(**detail)

    # -------------------------------------------------------------------------
    def list_session_versions(self, session_id: int) -> SessionVersionListResponse:
        items = self.service.list_session_versions(session_id)
        return SessionVersionListResponse(items=items)

    # -------------------------------------------------------------------------
    def get_session_version(
        self,
        session_id: int,
        version_id: int,
    ) -> SessionVersionDetailResponse:
        payload = self.service.get_session_version_detail(
            session_id,
            version_id=version_id,
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session version not found.",
            )
        return SessionVersionDetailResponse(**payload)

    # -------------------------------------------------------------------------
    def compare_session_versions(
        self,
        session_id: int,
        left_version_id: int,
        right_version_id: int,
    ) -> SessionVersionComparisonResponse:
        try:
            payload = self.service.compare_session_versions(
                session_id,
                left_version_id=left_version_id,
                right_version_id=right_version_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session version comparison target not found.",
            )
        return SessionVersionComparisonResponse(**payload)

    # -------------------------------------------------------------------------
    def list_manual_edits(self, session_id: int) -> list[ManualReportEditAudit]:
        payload = self.service.list_manual_report_edits(session_id)
        return [ManualReportEditAudit(**row) for row in payload]

    # -------------------------------------------------------------------------
    def manual_edit_session_report(
        self,
        session_id: int,
        request: ManualReportEditRequest,
    ) -> ManualReportEditResponse:
        response = self.service.manual_edit_report(
            session_id,
            report_text=request.report_text,
            edited_fields=request.edited_fields,
            reviewer_note=request.reviewer_note,
            edited_by=request.edited_by,
            metadata=request.metadata,
        )
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return ManualReportEditResponse(**response)

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return DeleteEntityResponse(deleted=True)

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
            "/sessions/{session_id}",
            self.get_session_detail,
            methods=["GET"],
            response_model=SessionDetailResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions",
            self.list_session_versions,
            methods=["GET"],
            response_model=SessionVersionListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions/{version_id}",
            self.get_session_version,
            methods=["GET"],
            response_model=SessionVersionDetailResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions/{left_version_id}/compare/{right_version_id}",
            self.compare_session_versions,
            methods=["GET"],
            response_model=SessionVersionComparisonResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.update_session,
            methods=["PUT"],
            response_model=SessionDetailResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/report",
            self.manual_edit_session_report,
            methods=["PUT"],
            response_model=ManualReportEditResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/manual-edits",
            self.list_manual_edits,
            methods=["GET"],
            response_model=list[ManualReportEditAudit],
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.delete_session,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
