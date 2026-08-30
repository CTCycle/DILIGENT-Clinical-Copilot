from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from domain.patient_timeline import (
    PatientTimeline,
    SessionTimelineListResponse,
    SessionTimelinePreview,
    SessionTimelineRegenerateRequest,
)
from domain.inspection import DeleteEntityResponse
from services.inspection.service import DataInspectionService


###############################################################################
class InspectionTimelineEndpoint(InspectionJobEndpointMixin):
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

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
        return SessionTimelineListResponse(
            items=[SessionTimelinePreview(**item) for item in items]
        )

    # -------------------------------------------------------------------------
    def start_session_timeline_job(
        self,
        session_id: int,
        request: SessionTimelineRegenerateRequest | None = Body(default=None),
    ) -> JobStartResponse:
        request = request or SessionTimelineRegenerateRequest()
        try:
            payload = self.service.start_session_timeline_job(
                session_id,
                force_regenerate=bool(request.force_regenerate),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        return self.build_job_start_response(
            payload=payload,
            message="Timeline generation started.",
        )

    # -------------------------------------------------------------------------
    def get_session_timeline_job_status(
        self, session_id: int, job_id: str
    ) -> JobStatusResponse:
        payload = self.service.get_session_timeline_job_status(session_id, job_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found."
            )
        return JobStatusResponse(**payload)

    # -------------------------------------------------------------------------
    def cancel_session_timeline_job(
        self, session_id: int, job_id: str
    ) -> JobCancelResponse:
        payload = self.service.get_session_timeline_job_status(session_id, job_id)
        if payload is None or not self.service.cancel_job(
            job_id, expected_type=self.service.SESSION_TIMELINE_JOB_TYPE
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found."
            )
        return JobCancelResponse(
            job_id=job_id, success=True, message="Cancellation requested"
        )

    # -------------------------------------------------------------------------
    def delete_session_timeline(
        self, session_id: int, timeline_id: int
    ) -> DeleteEntityResponse:
        if not self.service.delete_session_timeline(session_id, timeline_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session timeline not found.",
            )
        return DeleteEntityResponse(deleted=True)

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
            "/sessions/{session_id}/timelines/{timeline_id}",
            self.get_session_timeline_by_id,
            methods=["GET"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timelines/{timeline_id}",
            self.delete_session_timeline,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline-jobs",
            self.start_session_timeline_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline-jobs/{job_id}",
            self.get_session_timeline_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline-jobs/{job_id}",
            self.cancel_session_timeline_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
