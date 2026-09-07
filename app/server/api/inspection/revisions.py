from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.inspection import (
    RevisionArtifactListResponse,
    RevisionArtifactResponse,
    RevisionClinicalReviewActionListResponse,
    RevisionClinicalReviewActionResponse,
    RevisionClinicalReviewUpdateRequest,
    RevisionClinicalReviewUpdateResponse,
    RevisionEntityListResponse,
    RevisionEntityResponse,
    RevisionPipelineRunResponse,
    RevisionPipelineStepListResponse,
    RevisionPipelineStepResponse,
    SessionRevisionRequest,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.revision_scaffold import (
    SessionRevisionConflictError,
    SessionRevisionNotFoundError,
    SessionRevisionValidationError,
)
from services.inspection.service import DataInspectionService

###############################################################################
class InspectionRevisionEndpoint(InspectionJobEndpointMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

    # -------------------------------------------------------------------------
    def start_session_revision(
        self,
        session_id: int,
        request: SessionRevisionRequest | None = Body(default=None),
    ) -> JobStartResponse:
        try:
            payload = self.service.start_revision_job(session_id, request)
        except SessionRevisionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except SessionRevisionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except SessionRevisionValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return self.build_job_start_response(
            payload=payload,
            message="Revision agent started.",
        )

    # -------------------------------------------------------------------------
    def get_session_revision_status(self, job_id: str) -> JobStatusResponse:
        payload = self.service.get_revision_job_status(job_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision job not found.",
            )
        return JobStatusResponse(**payload)

    # -------------------------------------------------------------------------
    def cancel_session_revision(self, job_id: str) -> JobCancelResponse:
        success = self.service.cancel_revision_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision job not found.",
            )
        return JobCancelResponse(
            job_id=job_id,
            success=True,
            message="Cancellation requested",
        )

    # -------------------------------------------------------------------------
    def get_revision_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> RevisionPipelineRunResponse:
        run = self.service.get_revision_run(pipeline_run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision run not found.",
            )
        return RevisionPipelineRunResponse(**run)

    # -------------------------------------------------------------------------
    def retry_revision_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> JobStartResponse:
        try:
            payload = self.service.retry_revision_job(pipeline_run_id)
        except SessionRevisionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except SessionRevisionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except SessionRevisionValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return self.build_job_start_response(
            payload=payload,
            message="Revision agent retry started.",
        )

    # -------------------------------------------------------------------------
    def list_revision_pipeline_steps(
        self,
        pipeline_run_id: str,
    ) -> RevisionPipelineStepListResponse:
        if self.service.get_revision_run(pipeline_run_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision run not found.",
            )
        items = [
            RevisionPipelineStepResponse(**item)
            for item in self.service.list_revision_steps(pipeline_run_id)
        ]
        return RevisionPipelineStepListResponse(items=items)

    # -------------------------------------------------------------------------
    def list_session_revision_artifacts(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionArtifactListResponse:
        try:
            items = self.service.list_revision_artifacts(
                session_id,
                version_id=version_id,
            )
        except SessionRevisionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return RevisionArtifactListResponse(
            items=[RevisionArtifactResponse(**item) for item in items]
        )

    # -------------------------------------------------------------------------
    def list_session_revision_entities(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionEntityListResponse:
        try:
            items = self.service.list_revision_entities(
                session_id,
                version_id=version_id,
            )
        except SessionRevisionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return RevisionEntityListResponse(
            items=[RevisionEntityResponse(**item) for item in items]
        )

    # -------------------------------------------------------------------------
    def list_session_revision_reviews(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionClinicalReviewActionListResponse:
        try:
            items = self.service.list_revision_reviews(
                session_id,
                version_id=version_id,
            )
        except SessionRevisionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return RevisionClinicalReviewActionListResponse(
            items=[RevisionClinicalReviewActionResponse(**item) for item in items]
        )

    # -------------------------------------------------------------------------
    def update_session_revision_clinical_review(
        self,
        session_id: int,
        version_id: int,
        request: RevisionClinicalReviewUpdateRequest,
    ) -> RevisionClinicalReviewUpdateResponse:
        try:
            action = self.service.update_revision_clinical_review(
                session_id,
                version_id=version_id,
                clinical_review_status=request.clinical_review_status,
                reviewer_note=request.reviewer_note,
                reviewed_by=request.reviewed_by,
                metadata=request.metadata,
            )
        except SessionRevisionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        if action is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision review target not found.",
            )
        detail = self.service.get_session_version_detail(
            session_id,
            version_id=version_id,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revision version not found.",
            )
        return RevisionClinicalReviewUpdateResponse(
            version=detail["version"],
            review_action=RevisionClinicalReviewActionResponse(**action),
        )

    # -------------------------------------------------------------------------
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
