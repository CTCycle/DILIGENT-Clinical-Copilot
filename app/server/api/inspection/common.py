from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from common.utils.logger import logger
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.service import DataInspectionService


###############################################################################
class InspectionEndpointBase:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        self.router = router
        self.service = service


###############################################################################
class InspectionJobEndpointMixin(InspectionEndpointBase):

    # -------------------------------------------------------------------------
    def build_job_start_response(
        self,
        *,
        payload: dict[str, object],
        message: str,
    ) -> JobStartResponse:
        poll_interval_value = payload.get("poll_interval")
        poll_interval = (
            float(poll_interval_value)
            if isinstance(poll_interval_value, int | float)
            else 1.0
        )
        return JobStartResponse(
            job_id=str(payload["job_id"]),
            job_type=str(payload["job_type"]),
            status=str(payload["status"]),
            message=message,
            poll_interval=poll_interval,
        )

    # -------------------------------------------------------------------------
    def start_update_job(
        self,
        *,
        job_type: str,
        message: str,
        overrides: dict[str, object] | None = None,
    ) -> JobStartResponse:
        try:
            payload = self.service.start_update_job(job_type, overrides=overrides)
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            logger.warning(
                "Inspection update job rejected type=%s detail=%s", job_type, detail
            )
            safe_detail = (
                "An update job is already running."
                if error_status == status.HTTP_409_CONFLICT
                else "Invalid update request."
            )
            raise HTTPException(status_code=error_status, detail=safe_detail) from exc
        except RuntimeError as exc:
            logger.warning(
                "Inspection update job failed to start type=%s error=%s", job_type, exc
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Update job could not start. Please retry.",
            ) from exc
        return self.build_job_start_response(payload=payload, message=message)

    # -------------------------------------------------------------------------
    def get_update_job_status(self, *, job_id: str, job_type: str) -> JobStatusResponse:
        payload = self.service.get_job_status(job_id, expected_type=job_type)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobStatusResponse(**payload)

    # -------------------------------------------------------------------------
    def cancel_update_job(self, *, job_id: str, job_type: str) -> JobCancelResponse:
        success = self.service.cancel_job(job_id, expected_type=job_type)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobCancelResponse(
            job_id=job_id,
            success=True,
            message="Cancellation requested",
        )
