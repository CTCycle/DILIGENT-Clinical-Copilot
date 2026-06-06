from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.inspection import (
    RevisionArtifactListResponse,
    RevisionClinicalReviewActionListResponse,
    RevisionClinicalReviewUpdateRequest,
    RevisionClinicalReviewUpdateResponse,
    RevisionEntityListResponse,
    RevisionPipelineRunResponse,
    RevisionPipelineStepListResponse,
    SessionRevisionRequest,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.service import DataInspectionService


class InspectionRevisionEndpoint(InspectionJobEndpointMixin):
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

    def start_session_revision(
        self,
        session_id: int,
        request: SessionRevisionRequest | None = Body(default=None),
    ) -> JobStartResponse:
        request = request or SessionRevisionRequest()
        try:
            payload = self.service.start_revision_job(
                session_id,
                selected_text=request.selected_text,
                revision_instruction=request.revision_instruction,
                model_overrides=request.model_overrides,
                metadata=request.metadata,
            )
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_404_NOT_FOUND
                if "not found" in detail.casefold()
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=error_status, detail=detail) from exc
        return self.build_job_start_response(
            payload=payload,
            message="Session revision job started",
        )

    def get_session_revision_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.REVISION_JOB_TYPE,
        )

    def cancel_session_revision(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.REVISION_JOB_TYPE,
        )

    def get_revision_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> RevisionPipelineRunResponse:
        payload = self.service.get_revision_run(pipeline_run_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision pipeline run not found.",
            )
        return RevisionPipelineRunResponse(**payload)

    def retry_revision_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> JobStartResponse:
        try:
            payload = self.service.retry_revision_job(pipeline_run_id)
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_404_NOT_FOUND
                if "not found" in detail.casefold()
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=error_status, detail=detail) from exc
        return self.build_job_start_response(
            payload=payload,
            message="Session revision retry job started",
        )

    def list_revision_pipeline_steps(
        self,
        pipeline_run_id: str,
    ) -> RevisionPipelineStepListResponse:
        payload = self.service.list_revision_steps(pipeline_run_id)
        return RevisionPipelineStepListResponse(items=payload)

    def list_session_revision_artifacts(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionArtifactListResponse:
        detail = self.service.get_session_version_detail(
            session_id,
            version_id=version_id,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session version not found.",
            )
        payload = self.service.list_revision_artifacts(
            revision_version_id=version_id,
        )
        return RevisionArtifactListResponse(items=payload)

    def list_session_revision_entities(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionEntityListResponse:
        detail = self.service.get_session_version_detail(
            session_id,
            version_id=version_id,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session version not found.",
            )
        payload = self.service.list_revision_entities(
            revision_version_id=version_id,
        )
        return RevisionEntityListResponse(items=payload)

    def list_session_revision_reviews(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionClinicalReviewActionListResponse:
        detail = self.service.get_session_version_detail(
            session_id,
            version_id=version_id,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session version not found.",
            )
        payload = self.service.list_revision_reviews(
            revision_version_id=version_id,
        )
        return RevisionClinicalReviewActionListResponse(items=payload)

    def update_session_revision_clinical_review(
        self,
        session_id: int,
        version_id: int,
        request: RevisionClinicalReviewUpdateRequest,
    ) -> RevisionClinicalReviewUpdateResponse:
        try:
            payload = self.service.update_revision_clinical_review(
                session_id,
                version_id=version_id,
                clinical_review_status=request.clinical_review_status,
                reviewer_note=request.reviewer_note,
                reviewed_by=request.reviewed_by,
                metadata=request.metadata,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session version not found.",
            )
        return RevisionClinicalReviewUpdateResponse(**payload)

    def add_routes(self) -> None:
        self.router.add_api_route(
            "/sessions/{session_id}/revision/jobs",
            self.start_session_revision,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sessions/revision/jobs/{job_id}",
            self.get_session_revision_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/revision/jobs/{job_id}",
            self.cancel_session_revision,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/revision/pipeline-runs/{pipeline_run_id}",
            self.get_revision_pipeline_run,
            methods=["GET"],
            response_model=RevisionPipelineRunResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/revision/pipeline-runs/{pipeline_run_id}/retry",
            self.retry_revision_pipeline_run,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sessions/revision/pipeline-runs/{pipeline_run_id}/steps",
            self.list_revision_pipeline_steps,
            methods=["GET"],
            response_model=RevisionPipelineStepListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions/{version_id}/artifacts",
            self.list_session_revision_artifacts,
            methods=["GET"],
            response_model=RevisionArtifactListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions/{version_id}/entities",
            self.list_session_revision_entities,
            methods=["GET"],
            response_model=RevisionEntityListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions/{version_id}/reviews",
            self.list_session_revision_reviews,
            methods=["GET"],
            response_model=RevisionClinicalReviewActionListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/versions/{version_id}/clinical-review",
            self.update_session_revision_clinical_review,
            methods=["PUT"],
            response_model=RevisionClinicalReviewUpdateResponse,
            status_code=status.HTTP_200_OK,
        )
