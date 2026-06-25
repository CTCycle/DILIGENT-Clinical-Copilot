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
from services.inspection.revision_scaffold import (
    SESSION_REVISION_DISABLED_MESSAGE,
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
    @staticmethod
    def _raise_revision_not_implemented() -> None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=SESSION_REVISION_DISABLED_MESSAGE,
        )

    # -------------------------------------------------------------------------
    def start_session_revision(
        self,
        session_id: int,
        request: SessionRevisionRequest | None = Body(default=None),
    ) -> JobStartResponse:
        del session_id, request
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def get_session_revision_status(self, job_id: str) -> JobStatusResponse:
        del job_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def cancel_session_revision(self, job_id: str) -> JobCancelResponse:
        del job_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def get_revision_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> RevisionPipelineRunResponse:
        del pipeline_run_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def retry_revision_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> JobStartResponse:
        del pipeline_run_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_revision_pipeline_steps(
        self,
        pipeline_run_id: str,
    ) -> RevisionPipelineStepListResponse:
        del pipeline_run_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_session_revision_artifacts(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionArtifactListResponse:
        del session_id, version_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_session_revision_entities(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionEntityListResponse:
        del session_id, version_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_session_revision_reviews(
        self,
        session_id: int,
        version_id: int,
    ) -> RevisionClinicalReviewActionListResponse:
        del session_id, version_id
        self._raise_revision_not_implemented()

    # -------------------------------------------------------------------------
    def update_session_revision_clinical_review(
        self,
        session_id: int,
        version_id: int,
        request: RevisionClinicalReviewUpdateRequest,
    ) -> RevisionClinicalReviewUpdateResponse:
        del session_id, version_id, request
        self._raise_revision_not_implemented()

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
