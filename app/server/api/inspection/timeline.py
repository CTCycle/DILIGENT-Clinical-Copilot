from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, status

from api.inspection.common import InspectionEndpointBase
from common.utils.logger import logger
from domain.patient_timeline import (
    PatientTimeline,
    SessionTimelineListResponse,
    SessionTimelineRegenerateRequest,
)
from services.inspection.service import DataInspectionService


###############################################################################
class InspectionTimelineEndpoint(InspectionEndpointBase):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

    # -------------------------------------------------------------------------
    def get_session_timeline(self, session_id: int) -> PatientTimeline:
        timeline = self.service.get_session_timeline(session_id)
        if timeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session timeline not found.",
            )
        return timeline

    # -------------------------------------------------------------------------
    def get_session_timeline_by_id(
        self,
        session_id: int,
        timeline_id: int,
    ) -> PatientTimeline:
        timeline = self.service.get_session_timeline_by_id(session_id, timeline_id)
        if timeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session timeline not found.",
            )
        return timeline

    # -------------------------------------------------------------------------
    def list_session_timelines(self, session_id: int) -> SessionTimelineListResponse:
        items = self.service.list_session_timelines(session_id)
        return SessionTimelineListResponse(items=items)

    # -------------------------------------------------------------------------
    def generate_session_timeline(
        self,
        session_id: int,
        request: SessionTimelineRegenerateRequest | None = Body(default=None),
        force_regenerate_query: bool = Query(default=False, alias="force_regenerate"),
    ) -> PatientTimeline:
        request = request or SessionTimelineRegenerateRequest()
        force_regenerate = bool(force_regenerate_query or request.force_regenerate)
        try:
            timeline = self.service.generate_session_timeline(
                session_id,
                force_regenerate=force_regenerate,
            )
        except RuntimeError as exc:
            detail_message = str(exc)
            lowered_detail = detail_message.casefold()
            if (
                "cooling down" in lowered_detail
                or "already in progress" in lowered_detail
            ):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=detail_message,
                ) from exc
            if "failed to list ollama models" in lowered_detail:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Timeline generation requires a reachable local model runtime. "
                        "Start Ollama and retry."
                    ),
                ) from exc
            logger.warning(
                "Session timeline generation failed session_id=%s error=%s",
                session_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Timeline generation is currently unavailable. Please retry.",
            ) from exc
        if timeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return timeline

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/sessions/{session_id}/timelines",
            self.list_session_timelines,
            methods=["GET"],
            response_model=SessionTimelineListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timelines",
            self.generate_session_timeline,
            methods=["POST"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timelines/{timeline_id}",
            self.get_session_timeline_by_id,
            methods=["GET"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline",
            self.get_session_timeline,
            methods=["GET"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline",
            self.generate_session_timeline,
            methods=["POST"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
